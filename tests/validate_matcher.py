#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The ort-rocket authors
#
# Off-device validation gate for the structural subgraph matcher (src/rocket_match.cc). It
# reimplements the matcher's op-topology walks in Python and, for each local RF-DETR export, asserts
# that the structural result agrees with the exporter-name-based resolution the EP previously used:
#
#   * the patch-embed Conv is found and its kernel size selects the right variant,
#   * the expected number of encoder layers resolve fully,
#   * for every layer, the q/k/v/o/fc1/fc2 weight and bias EDGES are the exact tensors the
#     `/backbone/.../{query,key,...}/{MatMul,Add}` node names point at (weight-tensor identity), and
#   * the structural projector-output seed's ancestor set is byte-identical to the previous hardcoded
#     seed's ancestor set (the claimed node set is unchanged).
#
# This is a dev-box gate: it needs exported RF-DETR ONNX artifacts and the `onnx` package. It
# does NOT need onnxruntime, the NPU, or a build. Pass the .onnx paths on the command line, or
# set ORT_ROCKET_MODEL_DIR to a directory of rfdetr-*.onnx exports (default: the current
# directory). Exits non-zero if any model fails, so it doubles as a matcher regression guard.

import glob, os, sys
import onnx

HARDCODED_SEED = "/backbone/backbone.0/projector/stages.0/stages.0.1/Transpose_1_output_0"
EXPECT_LAYERS = {16: 12, 14: 11}   # patch size -> encoder layer count (nano / base)
LAYOUT = {"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Cast", "Identity",
          "DequantizeLinear", "QuantizeLinear"}


class G:
    def __init__(self, path):
        g = onnx.load(path).graph
        self.nodes = list(g.node)
        self.init = {i.name for i in g.initializer}
        self.ishape = {i.name: tuple(i.dims) for i in g.initializer}
        self.prod, self.cons = {}, {}
        for n in self.nodes:
            for o in n.output:
                if o:
                    self.prod[o] = n
            for x in n.input:
                if x:
                    self.cons.setdefault(x, []).append(n)
        self.order = {id(n): i for i, n in enumerate(self.nodes)}

    def op(self, n):
        return n.op_type if n is not None else None

    def attr(self, n, name):
        for a in n.attribute:
            if a.name == name:
                return list(a.ints)
        return []

    def resolves_to_init(self, t):
        if t in self.init:
            return True
        p = self.prod.get(t)
        return p is not None and p.op_type == "DequantizeLinear" and bool(p.input) and p.input[0] in self.init

    def act_input(self, n):
        for t in n.input:
            if t and not self.resolves_to_init(t):
                return t
        return None

    def has_init(self, n):
        return any(t and self.resolves_to_init(t) for t in n.input)

    def weight_edge(self, n):
        for t in n.input:
            if t and self.resolves_to_init(t):
                return t
        return None


def proj_matmul_back(g, t, depth=14):
    p = g.prod.get(t)
    if p is None or depth < 0:
        return None
    if p.op_type == "MatMul":
        return p
    if p.op_type not in (LAYOUT | {"Add", "Mul"}):
        return None
    for x in p.input:
        if not x or g.resolves_to_init(x):
            continue
        r = proj_matmul_back(g, x, depth - 1)
        if r is not None:
            return r
    return None


def back_layout(g, t, target):
    seen = set()
    while t and t not in seen:
        seen.add(t)
        p = g.prod.get(t)
        if p is None:
            return None
        if p.op_type == target:
            return p
        if p.op_type not in LAYOUT:
            return None
        t = g.act_input(p)
    return None


def fwd_first(g, t, pred, cross):
    fr, seen = [t], set()
    while fr:
        x = fr.pop(0)
        if x in seen:
            continue
        seen.add(x)
        for c in g.cons.get(x, []):
            if pred(c):
                return c
            if g.op(c) in cross:
                fr += list(c.output)
    return None


def fwd_matmul(g, t):
    return fwd_first(g, t, lambda c: c.op_type == "MatMul", LAYOUT)


def bias_add(g, mm):
    return fwd_first(g, mm.output[0], lambda c: c.op_type == "Add" and g.has_init(c), LAYOUT) if mm else None


def after_bias(g, mm):
    a = bias_add(g, mm)
    return a.output[0] if a else (mm.output[0] if mm else None)


def resolve_layer(g, sm):
    scores = back_layout(g, sm.input[0], "MatMul")
    if not scores or len(scores.input) < 2:
        return None
    q, k = proj_matmul_back(g, scores.input[0]), proj_matmul_back(g, scores.input[1])
    ctx = fwd_matmul(g, sm.output[0])
    if not ctx or len(ctx.input) < 2:
        return None

    def reaches_sm(t, d=8):
        seen = set()
        while t and t not in seen and d > 0:
            seen.add(t); d -= 1
            p = g.prod.get(t)
            if p is None:
                return False
            if p is sm:
                return True
            if g.op(p) not in LAYOUT:
                return False
            t = g.act_input(p)
        return False

    v = proj_matmul_back(g, ctx.input[1] if reaches_sm(ctx.input[0]) else ctx.input[0])
    o = fwd_matmul(g, ctx.output[0])
    if not all((q, k, v, o)):
        return None
    ls1 = fwd_first(g, after_bias(g, o), lambda c: c.op_type == "Mul" and g.has_init(c), LAYOUT)
    res1 = fwd_first(g, ls1.output[0], lambda c: c.op_type == "Add" and not g.has_init(c), LAYOUT) if ls1 else None
    if not res1:
        return None
    norm2 = fwd_first(g, res1.output[0], lambda c: c.op_type == "LayerNormalization", LAYOUT)
    if not norm2:
        return None
    fc1 = fwd_matmul(g, norm2.output[0])
    if not fc1:
        return None
    fc2, fr, seen = None, [after_bias(g, fc1)], set()
    while fr and fc2 is None:
        x = fr.pop(0)
        if not x or x in seen:
            continue
        seen.add(x)
        for c in g.cons.get(x, []):
            if c.op_type == "MatMul":
                fc2 = c; break
            fr += list(c.output)
    if not fc2:
        return None
    return dict(q=q, k=k, v=v, o=o, fc1=fc1, fc2=fc2,
                bq=bias_add(g, q), bk=bias_add(g, k), bv=bias_add(g, v),
                bo=bias_add(g, o), bf1=bias_add(g, fc1), bf2=bias_add(g, fc2))


def find_patch_conv(g):
    for n in g.nodes:
        if g.op(n) != "Conv" or len(n.input) < 2:
            continue
        ks, st = g.attr(n, "kernel_shape"), g.attr(n, "strides")
        if len(ks) != 2 or ks != st or not g.resolves_to_init(n.input[1]):
            continue
        wt = n.input[1]
        if wt not in g.init:
            dq = g.prod.get(wt)
            wt = dq.input[0] if dq and dq.input else wt
        shp = g.ishape.get(wt)
        if shp and len(shp) == 4 and shp[1] == 3 and shp[2] == ks[0]:
            return n, ks[0]
    return None, None


def descendants(g, start):
    out, fr = set(), [start]
    while fr:
        t = fr.pop()
        for c in g.cons.get(t, []):
            if id(c) in out:
                continue
            out.add(id(c)); fr += list(c.output)
    return out


def projector_exit(g, patch):
    desc = descendants(g, patch.output[0])
    convs = [n for n in g.nodes if id(n) in desc and g.op(n) == "Conv" and n is not patch]
    if not convs:
        return None
    last = max(convs, key=lambda n: g.order[id(n)])
    finish = {"Transpose", "LayerNormalization", "Sigmoid", "Mul", "Cast", "DequantizeLinear", "QuantizeLinear"}
    best, best_ord, fr, seen = last.output[0], g.order[id(last)], [last.output[0]], set()
    while fr:
        t = fr.pop()
        if t in seen:
            continue
        seen.add(t)
        p = g.prod.get(t)
        if p is not None and g.op(p) not in ("QuantizeLinear", "DequantizeLinear") and g.order[id(p)] > best_ord:
            best, best_ord = t, g.order[id(p)]
        for c in g.cons.get(t, []):
            if g.op(c) in finish:
                fr += list(c.output)
    return best


def ancestors(g, tensor):
    claimed, stack, seen = set(), [tensor], set()
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        p = g.prod.get(t)
        if p is None:
            continue
        claimed.add(id(p))
        stack += [x for x in p.input if x]
    return claimed


def check(path):
    g = G(path)
    fails = []
    patch, psz = find_patch_conv(g)
    if not patch:
        return [f"{path}: patch Conv not found"]
    desc = descendants(g, patch.output[0])
    sms = sorted((n for n in g.nodes if g.op(n) == "Softmax" and id(n) in desc), key=lambda n: g.order[id(n)])
    layers = [L for L in (resolve_layer(g, sm) for sm in sms) if L]
    exp = EXPECT_LAYERS.get(psz)
    if exp is None or len(layers) != exp:
        fails.append(f"layer count {len(layers)} != expected {exp} for patch {psz}")

    # weight-edge identity vs the exporter-name-based resolution, per role
    for i, L in enumerate(layers):
        pref = f"/layer.{i}/"

        def named(suf):
            for n in g.nodes:
                if pref in n.name and n.name.endswith(suf):
                    return n
            return None
        for role, suf in [("q", "/query/MatMul"), ("k", "/key/MatMul"), ("v", "/value/MatMul"),
                          ("o", "/output/dense/MatMul"), ("fc1", "/mlp/fc1/MatMul"), ("fc2", "/mlp/fc2/MatMul"),
                          ("bq", "/query/Add"), ("bk", "/key/Add"), ("bv", "/value/Add"),
                          ("bo", "/output/dense/Add"), ("bf1", "/mlp/fc1/Add"), ("bf2", "/mlp/fc2/Add")]:
            nb = named(suf)
            if nb is None:
                continue   # a renamed export legitimately lacks the old name; structural still resolved it
            se, ne = g.weight_edge(L[role]), g.weight_edge(nb)
            if se != ne or se is None:
                fails.append(f"L{i} {role}: structural edge {se} != name-based {ne}")

    # claimed-node set identity: structural seed vs the previous hardcoded seed
    exit_t = projector_exit(g, patch)
    if exit_t is None:
        fails.append("projector exit not found")
    elif HARDCODED_SEED in g.prod:
        s, h = ancestors(g, exit_t), ancestors(g, HARDCODED_SEED)
        if s != h:
            fails.append(f"claimed set differs: +{len(s - h)}/-{len(h - s)} vs hardcoded seed")
    return fails


def main():
    # Model artifacts: paths given on the command line, else rfdetr-*.onnx under
    # ORT_ROCKET_MODEL_DIR (default: the current directory).
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        model_dir = os.environ.get("ORT_ROCKET_MODEL_DIR", ".")
        models = sorted(glob.glob(os.path.join(model_dir, "rfdetr-*.onnx")))
    if not models:
        print("no RF-DETR .onnx artifacts found (pass paths on the command line or set "
              "ORT_ROCKET_MODEL_DIR); nothing to check")
        return 0
    bad = 0
    for m in models:
        fails = check(m)
        tag = "PASS" if not fails else "FAIL"
        print(f"[{tag}] {os.path.basename(m)}")
        for f in fails:
            print(f"        {f}"); bad += 1
    print(f"\n{len(models)} model(s), {bad} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
