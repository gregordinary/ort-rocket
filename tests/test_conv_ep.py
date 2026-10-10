#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""On-device faithfulness gate for the ort-rocket Conv handler.

The handler claims Conv nodes one at a time, each with the Relu or LeakyRelu that is its only
consumer, and runs each on librocketnpu's resident im2col convolution in fp16. Every case below runs
twice in one process, once on the ORT CPU EP and once with ort-rocket, and is scored element by
element:

  * per node: worst |ep - cpu| / mag, where mag is the element's magnitude sum (the sum over taps of
    |x * w|, plus |bias|) computed in fp64. The gate is 2^-8. A wrong layout, a dropped tap or a
    shifted pad puts an element's error at the order of its magnitude; fp16 rounding of the input,
    the weight and the entry's partial sums stays near 2^-10. Relu and LeakyRelu (alpha <= 1) move
    two values no further apart, so the same gate holds after a folded activation. These cases run
    under ROCKET_ORT_CONV=all, so the cost rule does not decide which shapes are tested:
      - synthetic shapes along stride, pads (including asymmetric), auto_pad SAME_UPPER / SAME_LOWER
        / VALID, kernel size, dilation, a 1x1 read in place, a patch embedding, batch, a bias, an
        output channel count that is not a multiple of 16, and the activation fold (Relu, LeakyRelu
        at alpha 0.2, and a Conv whose output is also a graph output or has two consumers, which
        must not fold);
      - with --pix2pix, every Conv of that model with its own weight and bias and the input it
        receives in the model, captured from a CPU run.
  * refusals: nodes the handler must leave on the CPU (grouped, depthwise, a weight that is a graph
    input, a symbolic spatial size) run on the CPU EP and match it bit for bit.
  * the cost rule, at its default: a shape it admits (pix2pix's c2) runs on the EP, and one it
    refuses (a 3x3 64->64 conv at 19x19) runs on the CPU and matches it bit for bit.
  * placement: every claimed case executed on the EP, read from ONNX Runtime's profile, and is not
    bit-identical to the CPU EP's output.
  * end to end, with --pix2pix: the generator's output image (max abs, PSNR at peak-to-peak 2) at
    ORT_DISABLE_ALL and ORT_ENABLE_ALL with the ConvTranspose and Conv claims both on, and
    ROCKET_ORT_CONV=0 leaving every Conv on the CPU.

  .../python test_conv_ep.py <lib.so> [--pix2pix pix2pix.onnx]
"""
import gc
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from ep_common import CPU_EP, EpSession

REL_GATE = 2.0 ** -8      # per-node worst |err| / mag
PIX_MAXABS = 0.05         # pix2pix image, output in [-1, 1]
PIX_PSNR = 40.0           # dB, peak-to-peak 2

FAILS = []


def record(ok, what):
    if not ok:
        FAILS.append(what)
    return ok


# ---------------------------------------------------------------------------------------------
# Reference arithmetic
# ---------------------------------------------------------------------------------------------
def same_pads(n, k, s, d, upper):
    out = (n + s - 1) // s
    total = max((out - 1) * s + (k - 1) * d + 1 - n, 0)
    b = total // 2 if upper else total - total // 2
    return b, total - b


def resolve_pads(x_shape, k, s, d, p, auto_pad):
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        up = auto_pad == "SAME_UPPER"
        pt, pb = same_pads(x_shape[2], k[0], s[0], d[0], up)
        pl, pr = same_pads(x_shape[3], k[1], s[1], d[1], up)
        return pt, pl, pb, pr
    if auto_pad == "VALID":
        return 0, 0, 0, 0
    return tuple(p)


def conv_mag(x, w, b, s, pads, d):
    """fp64 magnitude sum of a 2-D Conv: sum over taps of |x w| plus |b|. x [N,C,H,W],
    w [OC,C,KH,KW], pads [top, left, bottom, right]."""
    x = np.abs(x.astype(np.float64)); w = np.abs(w.astype(np.float64))
    N, C, H, W = x.shape
    OC, _, KH, KW = w.shape
    sy, sx = s; dy, dx = d
    pt, pl, pb, pr = pads
    OH = (H + pt + pb - ((KH - 1) * dy + 1)) // sy + 1
    OW = (W + pl + pr - ((KW - 1) * dx + 1)) // sx + 1
    xp = np.zeros((N, C, H + pt + pb, W + pl + pr))
    xp[:, :, pt:pt + H, pl:pl + W] = x
    out = np.zeros((N, OC, OH, OW))
    if b is not None:
        out += np.abs(b.astype(np.float64))[None, :, None, None]
    for ky in range(KH):
        for kx in range(KW):
            y0, x0 = ky * dy, kx * dx
            xs = xp[:, :, y0:y0 + (OH - 1) * sy + 1:sy, x0:x0 + (OW - 1) * sx + 1:sx]
            out += np.einsum("nchw,oc->nohw", xs, w[:, :, ky, kx], optimize=True)
    return out


def rel_err(ep, cpu, mag):
    err = np.abs(ep.astype(np.float64) - cpu.astype(np.float64))
    ratio = np.where(mag > 0, err / np.maximum(mag, 1e-300), np.where(err > 0, np.inf, 0.0))
    i = int(np.argmax(ratio))
    return float(ratio.flat[i]), float(err.flat[i]), float(mag.flat[i]), float(err.max())


def log2s(v):
    return f"2^{math.log2(v):.1f}" if v > 0 else ("0" if v == 0 else "inf")


# ---------------------------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------------------------
def cpu_run(path, feeds, outs=None):
    s = ort.InferenceSession(path, providers=[CPU_EP])
    r = s.run(outs, feeds)
    del s
    return r


def ep_run(lib, path, feeds, outs=None, opt=ort.GraphOptimizationLevel.ORT_DISABLE_ALL):
    with EpSession(lib, path, "rocket", log_level=3, opt_level=opt) as sep:
        r = sep.run(outs, feeds)
    return r, sep.placement


def on_ep(placement):
    return sum(len(n) for p, n in placement.items() if p != CPU_EP)


class Env:
    """Set environment variables for the duration of a block."""

    def __init__(self, **kv):
        self.kv = kv
        self.old = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


# ---------------------------------------------------------------------------------------------
# Single-node models
# ---------------------------------------------------------------------------------------------
def conv_model(path, x_shape, w, b, attrs, w_as_input=False, sym_hw=False, act=None, alpha=0.2,
               conv_out_too=False, two_consumers=False):
    """A Conv, optionally followed by an activation. conv_out_too also exposes the Conv's output as
    a graph output; two_consumers adds a second consumer (an Identity) of it."""
    inits = []
    ins = ["X", "W"]
    graph_in = [helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, ["n", x_shape[1], "h", "w"] if sym_hw else list(x_shape))]
    if w_as_input:
        graph_in.append(helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w.shape)))
    else:
        inits.append(numpy_helper.from_array(w.astype(np.float32), "W"))
    if b is not None:
        inits.append(numpy_helper.from_array(b.astype(np.float32), "B"))
        ins.append("B")
    nodes = [helper.make_node("Conv", ins, ["C"], **attrs)]
    outs = ["Y"]
    if act == "Relu":
        nodes.append(helper.make_node("Relu", ["C"], ["Y"]))
    elif act == "LeakyRelu":
        nodes.append(helper.make_node("LeakyRelu", ["C"], ["Y"], alpha=alpha))
    else:
        nodes[0].output[0] = "Y"
    if conv_out_too:
        outs.append("C")
    if two_consumers:
        nodes.append(helper.make_node("Identity", ["C"], ["C2"]))
        outs.append("C2")
    g = helper.make_graph(nodes, "conv", graph_in,
                          [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outs], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)
    return outs


def attrs_of(k, s, p=None, d=(1, 1), group=1, auto_pad=None):
    a = dict(kernel_shape=list(k), strides=list(s), dilations=list(d), group=group)
    if auto_pad:
        a["auto_pad"] = auto_pad
    else:
        a["pads"] = list(p)
    return a


def act_np(y, act, alpha):
    if act == "Relu":
        return np.maximum(y, 0)
    if act == "LeakyRelu":
        return np.where(y >= 0, y, y * alpha)
    return y


def node_case(lib, tmp, name, x, w, b, k, s, p=(0, 0, 0, 0), d=(1, 1), auto_pad=None, act=None,
              alpha=0.2, conv_out_too=False, two_consumers=False, expect_nodes=1, mode="all"):
    """One claimed case. expect_nodes: how many fused nodes the EP runs (1; the activation, when it
    folds, is inside the one fused node, and when it does not it stays a CPU node)."""
    path = os.path.join(tmp, name + ".onnx")
    outs = conv_model(path, x.shape, w, b, attrs_of(k, s, p, d, auto_pad=auto_pad), act=act,
                      alpha=alpha, conv_out_too=conv_out_too, two_consumers=two_consumers)
    cpu = cpu_run(path, {"X": x})
    with Env(ROCKET_ORT_CONV=mode):
        ep, placement = ep_run(lib, path, {"X": x})
    pads = resolve_pads(x.shape, k, s, d, p, auto_pad)
    mag = conv_mag(x, w, b, s, pads, d)
    ok_all = True
    for oname, e, c in zip(outs, ep, cpu):
        shape_ok = e.shape == c.shape == mag.shape
        r, err, m, emax = rel_err(e, c, mag) if shape_ok else (np.inf, 0, 0, 0)
        ident = shape_ok and np.array_equal(e, c)
        ok = shape_ok and r <= REL_GATE and not ident
        ok_all &= ok
        print(f"[test] node {name:26s} {oname:2s} x{list(x.shape)} -> {list(e.shape)}: worst |err|/mag "
              f"{log2s(r)} (err {err:.2e} at mag {m:.2e}), max|err| {emax:.2e}"
              f"{'  IDENTICAL' if ident else ''}  {'ok' if ok else 'FAIL'}")
    n_ep = on_ep(placement)
    placed = n_ep == expect_nodes
    print(f"[test]   placement: {dict((p_, len(n_)) for p_, n_ in placement.items())}"
          f"  {'ok' if placed else 'FAIL (expected %d on the EP)' % expect_nodes}")
    return record(ok_all and placed, "node " + name)


def refusal_case(lib, tmp, name, x_shape, w, b, attrs, w_as_input=False, sym_hw=False, mode="all"):
    """A node the handler must leave on the CPU: the EP run equals the CPU run bit for bit and the
    profile shows no EP node."""
    path = os.path.join(tmp, name + ".onnx")
    conv_model(path, x_shape, w, b, attrs, w_as_input=w_as_input, sym_hw=sym_hw)
    rs = np.random.RandomState(7)
    feeds = {"X": rs.randn(*x_shape).astype(np.float32)}
    if w_as_input:
        feeds["W"] = w.astype(np.float32)
    cpu = cpu_run(path, feeds)[0]
    with Env(ROCKET_ORT_CONV=mode):
        ep, placement = ep_run(lib, path, feeds)
    ok = on_ep(placement) == 0 and np.array_equal(ep[0], cpu)
    print(f"[test] refusal {name:25s} on EP {on_ep(placement)}, identical {np.array_equal(ep[0], cpu)}"
          f"  {'ok' if ok else 'FAIL'}")
    return record(ok, "refusal " + name)


def synthetic(lib, tmp):
    rs = np.random.RandomState(0)

    def W(oc, ic, kh, kw, scale=None):
        sc = scale if scale is not None else 1.0 / math.sqrt(ic * kh * kw)
        return (rs.randn(oc, ic, kh, kw) * sc).astype(np.float32)

    def X(n, c, h, w):
        return rs.randn(n, c, h, w).astype(np.float32)

    def B(n):
        return (rs.randn(n) * 0.5).astype(np.float32)

    node_case(lib, tmp, "k3s1p1", X(1, 32, 20, 20), W(32, 32, 3, 3), None, (3, 3), (1, 1), (1, 1, 1, 1))
    node_case(lib, tmp, "k4s2p1_bias", X(1, 64, 32, 32), W(64, 64, 4, 4), B(64), (4, 4), (2, 2), (1, 1, 1, 1))
    node_case(lib, tmp, "k3s2p1_odd_n3", X(3, 32, 17, 23), W(48, 32, 3, 3), B(48), (3, 3), (2, 2), (1, 1, 1, 1))
    node_case(lib, tmp, "k3s3_gather", X(1, 32, 21, 30), W(32, 32, 3, 3), None, (3, 3), (3, 3), (1, 1, 1, 1))
    node_case(lib, tmp, "asym_p0011", X(1, 32, 16, 16), W(32, 32, 3, 3), None, (3, 3), (2, 2), (0, 0, 1, 1))
    node_case(lib, tmp, "asym_p2103", X(1, 32, 15, 18), W(32, 32, 3, 3), None, (3, 3), (1, 1), (2, 1, 0, 3))
    node_case(lib, tmp, "same_upper_s2", X(1, 32, 16, 15), W(32, 32, 3, 3), None, (3, 3), (2, 2),
              auto_pad="SAME_UPPER")
    node_case(lib, tmp, "same_lower_s2", X(1, 32, 16, 15), W(32, 32, 4, 4), None, (4, 4), (2, 2),
              auto_pad="SAME_LOWER")
    node_case(lib, tmp, "valid_k5", X(1, 16, 20, 20), W(32, 16, 5, 5), None, (5, 5), (1, 1), auto_pad="VALID")
    node_case(lib, tmp, "dil2", X(1, 32, 20, 20), W(32, 32, 3, 3), None, (3, 3), (1, 1), (2, 2, 2, 2), d=(2, 2))
    node_case(lib, tmp, "1x1_in_place_bias", X(1, 96, 24, 24), W(64, 96, 1, 1), B(64), (1, 1), (1, 1))
    node_case(lib, tmp, "1x1_oc40_odd", X(1, 64, 9, 11), W(40, 64, 1, 1), B(40), (1, 1), (1, 1))
    node_case(lib, tmp, "patch_k16s16_ic3", X(1, 3, 64, 64), W(96, 3, 16, 16), B(96), (16, 16), (16, 16))
    node_case(lib, tmp, "k7s2p3_ic3", X(1, 3, 64, 64), W(32, 3, 7, 7), B(32), (7, 7), (2, 2), (3, 3, 3, 3))
    node_case(lib, tmp, "k1x5", X(1, 32, 12, 20), W(32, 32, 1, 5), None, (1, 5), (1, 1), (0, 2, 0, 2))
    # the activation fold
    node_case(lib, tmp, "fold_relu", X(1, 64, 32, 32), W(64, 64, 3, 3), B(64), (3, 3), (1, 1),
              (1, 1, 1, 1), act="Relu")
    node_case(lib, tmp, "fold_leaky_0.2", X(1, 64, 32, 32), W(128, 64, 4, 4), B(128), (4, 4), (2, 2),
              (1, 1, 1, 1), act="LeakyRelu", alpha=0.2)
    # not folded: the Conv's output is also a graph output, or has a second consumer. The Relu then
    # runs on the CPU, so its output Y is scored against the Conv's mag like the others.
    node_case(lib, tmp, "nofold_graph_output", X(1, 32, 16, 16), W(32, 32, 3, 3), None, (3, 3), (1, 1),
              (1, 1, 1, 1), act="Relu", conv_out_too=True)
    node_case(lib, tmp, "nofold_two_consumers", X(1, 32, 16, 16), W(32, 32, 3, 3), None, (3, 3), (1, 1),
              (1, 1, 1, 1), act="Relu", two_consumers=True)

    # refusals
    refusal_case(lib, tmp, "grouped_g2", (1, 32, 8, 8), W(16, 16, 3, 3), None,
                 attrs_of((3, 3), (1, 1), (1, 1, 1, 1), group=2))
    refusal_case(lib, tmp, "depthwise", (1, 32, 8, 8), W(32, 1, 3, 3), None,
                 attrs_of((3, 3), (1, 1), (1, 1, 1, 1), group=32))
    refusal_case(lib, tmp, "weight_is_input", (1, 16, 8, 8), W(16, 16, 3, 3), None,
                 attrs_of((3, 3), (1, 1), (1, 1, 1, 1)), w_as_input=True)
    refusal_case(lib, tmp, "symbolic_hw", (1, 16, 8, 8), W(16, 16, 3, 3), None,
                 attrs_of((3, 3), (1, 1), (1, 1, 1, 1)), sym_hw=True)

    # the cost rule at its default
    node_case(lib, tmp, "cost_admits_pix2pix_c2", X(1, 64, 128, 128), W(128, 64, 4, 4), B(128), (4, 4),
              (2, 2), (1, 1, 1, 1), mode="1")
    refusal_case(lib, tmp, "cost_refuses_19x19", (1, 64, 19, 19), W(64, 64, 3, 3), None,
                 attrs_of((3, 3), (1, 1), (1, 1, 1, 1)), mode="1")


# ---------------------------------------------------------------------------------------------
# Model layers and end to end
# ---------------------------------------------------------------------------------------------
def capture_conv_inputs(model_path, feeds, tmp):
    """Run the model on the CPU EP with every Conv's input promoted to a graph output.
    Returns [(node, input array)] and the model."""
    m = onnx.load(model_path)
    nodes = [n for n in m.graph.node if n.op_type == "Conv"]
    have = {o.name for o in m.graph.output}
    for n in nodes:
        if n.input[0] not in have:
            m.graph.output.append(helper.make_tensor_value_info(n.input[0], TensorProto.FLOAT, None))
    path = os.path.join(tmp, "capture.onnx")
    onnx.save(m, path)
    outs = cpu_run(path, feeds, [n.input[0] for n in nodes])
    return list(zip(nodes, outs)), m


def model_layers(lib, tmp, model_path, feeds, tag):
    caps, m = capture_conv_inputs(model_path, feeds, tmp)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    for i, (n, x) in enumerate(caps):
        a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
        w = inits[n.input[1]]
        b = inits[n.input[2]] if len(n.input) > 2 and n.input[2] else None
        k = tuple(a.get("kernel_shape", w.shape[2:]))
        s = tuple(a.get("strides", (1, 1)))
        p = tuple(a.get("pads", (0, 0, 0, 0)))
        d = tuple(a.get("dilations", (1, 1)))
        node_case(lib, tmp, f"{tag}_c{i + 1}_{w.shape[1]}to{w.shape[0]}", x.astype(np.float32),
                  w, b, k, s, p, d)


def facade_label_map(seed=0):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_convtranspose_ep as T  # noqa: E402
    return T.facade_label_map(seed)


def pix2pix(lib, tmp, path):
    x_lab = facade_label_map()
    x_rnd = np.random.RandomState(1).uniform(-1, 1, (1, 3, 256, 256)).astype(np.float32)
    in_name = ort.InferenceSession(path, providers=[CPU_EP]).get_inputs()[0].name
    model_layers(lib, tmp, path, {in_name: x_lab}, "pix2pix")
    for tag, x in (("label map", x_lab), ("uniform", x_rnd)):
        cpu = cpu_run(path, {in_name: x})[0]
        for opt in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            ep, placement = ep_run(lib, path, {in_name: x}, opt=opt)
            ep = ep[0]
            d = np.abs(ep.astype(np.float64) - cpu)
            mse = float(np.mean(d ** 2))
            psnr = 10 * math.log10(4.0 / mse) if mse > 0 else float("inf")
            n_ep = on_ep(placement)
            # eight ConvTranspose nodes and seven Conv nodes (c2-c8; the cost rule leaves c1)
            ok = n_ep == 15 and float(d.max()) <= PIX_MAXABS and psnr >= PIX_PSNR
            print(f"[test] pix2pix e2e ({tag}, {opt.name}): {n_ep} node(s) on the EP, max abs "
                  f"{d.max():.3e}, mean abs {d.mean():.3e}, PSNR {psnr:.1f} dB  {'ok' if ok else 'FAIL'}")
            record(ok, f"pix2pix e2e {tag} {opt.name}")
    with Env(ROCKET_ORT_CONV="0"):
        ep, placement = ep_run(lib, path, {in_name: x_lab})
    ok = on_ep(placement) == 8
    print(f"[test] pix2pix with ROCKET_ORT_CONV=0: {on_ep(placement)} node(s) on the EP (the "
          f"ConvTranspose claim alone)  {'ok' if ok else 'FAIL'}")
    record(ok, "pix2pix knob off")


def main():
    args = sys.argv[1:]
    lib = os.path.abspath(args[0])

    def opt(flag):
        return os.path.abspath(args[args.index(flag) + 1]) if flag in args else None

    tmp = tempfile.mkdtemp(prefix="ort_rocket_conv_")
    try:
        synthetic(lib, tmp)
        if opt("--pix2pix"):
            pix2pix(lib, tmp, opt("--pix2pix"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    gc.collect()
    print("RESULT:", "PASS" if not FAILS else "FAIL " + ", ".join(FAILS),
          f"(node gate |err|/mag <= {log2s(REL_GATE)})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
