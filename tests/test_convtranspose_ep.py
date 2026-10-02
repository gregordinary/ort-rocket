#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""On-device faithfulness gate for the ort-rocket ConvTranspose handler.

The handler claims ConvTranspose nodes one at a time and runs each on librocketnpu's resident
transposed convolution in fp16. Every case below runs twice in one process, once on the ORT CPU EP
and once with ort-rocket, and is scored element by element:

  * per node: worst |ep - cpu| / mag, where mag is the element's magnitude sum (the sum over taps
    of |x * w|, plus |bias|) computed in fp64. The gate is 2^-8. A wrong layout, a dropped tap or a
    shifted pad puts an element's error at the order of its magnitude, 2^0; fp16 rounding of the
    input, the weight and the entry's partial sums stays near 2^-10.
      - synthetic shapes along stride, padding (including asymmetric), output_padding, kernel size,
        dilation, batch, and a node whose input size is symbolic, run at a size the entry takes and
        at one its plan refuses (that size runs on the handler's fp32 host path);
      - with --pix2pix / --sam-decoder, every ConvTranspose of those models with the model's own
        weight and bias and the input it receives in the model, captured from a CPU run.
  * refusals: nodes the handler must leave on the CPU (grouped, depthwise, auto_pad SAME,
    output_shape, a weight that is a graph input) run on the CPU EP and match it bit for bit.
  * placement: every claimed case executed on the EP, read from ONNX Runtime's profile, and is not
    bit-identical to the CPU EP's output.
  * end to end, with --pix2pix: the generator's output image (max abs, PSNR at peak-to-peak 2) and
    ROCKET_ORT_CONVTRANSPOSE=0 claiming nothing; with --sam-decoder and --sam-embedding: the mask
    logits (max abs), the per-mask IoU of the logit > 0 masks, and the IoU predictions, at P=1 and
    P=64 points.

  sudo -E .../python test_convtranspose_ep.py <lib.so> [--pix2pix pix2pix.onnx]
        [--sam-decoder prompt_encoder_mask_decoder.onnx --sam-embedding image_embedding.npy]
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
SAM_IOU = 0.99            # per-mask IoU of the logit > 0 masks
SAM_SCORE = 0.02          # iou_scores max abs

FAILS = []


def record(ok, what):
    if not ok:
        FAILS.append(what)
    return ok


# ---------------------------------------------------------------------------------------------
# Reference arithmetic
# ---------------------------------------------------------------------------------------------
def convt_mag(x, w, b, strides, pads, dil, opad):
    """fp64 magnitude sum of a 2-D ConvTranspose: sum over taps of |x w| plus |b|. x [N,C,H,W],
    w [C,OC,KH,KW], pads [top, left, bottom, right]. Computed as col2im: one GEMM over input
    channels, then each tap's plane added at its offset."""
    x = np.abs(x.astype(np.float64)); w = np.abs(w.astype(np.float64))
    N, C, H, W = x.shape
    _, OC, KH, KW = w.shape
    sy, sx = strides; dy, dx = dil
    pt, pl, pb, pr = pads
    OH = (H - 1) * sy - pt - pb + dy * (KH - 1) + opad[0] + 1
    OW = (W - 1) * sx - pl - pr + dx * (KW - 1) + opad[1] + 1
    out = np.zeros((N, OC, OH, OW))
    if b is not None:
        out += np.abs(b.astype(np.float64))[None, :, None, None]
    for n in range(N):
        P = np.einsum("chw,cokl->oklhw", x[n], w, optimize=True)
        for kh in range(KH):
            for kw in range(KW):
                ph = np.arange(H) * sy - pt + kh * dy
                pw = np.arange(W) * sx - pl + kw * dx
                my = (ph >= 0) & (ph < OH); mx = (pw >= 0) & (pw < OW)
                if not my.any() or not mx.any():
                    continue
                out[n][:, ph[my][:, None], pw[mx][None, :]] += P[:, kh, kw][:, my][:, :, mx]
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


def ep_run(lib, path, feeds, outs=None, opt=ort.GraphOptimizationLevel.ORT_DISABLE_ALL, reps=1):
    with EpSession(lib, path, "rocket", log_level=3, opt_level=opt) as sep:
        for _ in range(reps):
            r = sep.run(outs, feeds)
    return r, sep.placement


def on_ep(placement):
    return sum(len(n) for p, n in placement.items() if p != CPU_EP)


# ---------------------------------------------------------------------------------------------
# Single-node models
# ---------------------------------------------------------------------------------------------
def ct_model(path, x_shape, w, b, attrs, w_as_input=False, sym_hw=False):
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
    node = helper.make_node("ConvTranspose", ins, ["Y"], **attrs)
    g = helper.make_graph([node], "ct", graph_in,
                          [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def attrs_of(k, s, p, d=(1, 1), op=(0, 0), group=1, **extra):
    a = dict(kernel_shape=list(k), strides=list(s), pads=list(p), dilations=list(d), group=group)
    if any(op):
        a["output_padding"] = list(op)
    a.update(extra)
    return a


def node_case(lib, tmp, name, x, w, b, k, s, p, d=(1, 1), op=(0, 0), sym_hw=False, sizes=None):
    """One claimed single-node case. sizes: extra (N,H,W) inputs to run through the same session
    (a symbolic-size node)."""
    path = os.path.join(tmp, name + ".onnx")
    ct_model(path, x.shape, w, b, attrs_of(k, s, p, d, op), sym_hw=sym_hw)
    xs = [x]
    rs = np.random.RandomState(sum(name.encode()))
    for (n, h, ww) in sizes or []:
        xs.append(rs.randn(n, x.shape[1], h, ww).astype(np.float32))
    cpu = [cpu_run(path, {"X": xi})[0] for xi in xs]
    with EpSession(lib, path, "rocket", log_level=3) as sep:
        ep = [sep.run(None, {"X": xi})[0] for xi in xs]
    ok_all = True
    for xi, e, c in zip(xs, ep, cpu):
        mag = convt_mag(xi, w, b, s, p, d, op)
        shape_ok = e.shape == c.shape == mag.shape
        r, err, m, emax = rel_err(e, c, mag) if shape_ok else (np.inf, 0, 0, 0)
        ident = shape_ok and np.array_equal(e, c)
        ok = shape_ok and r <= REL_GATE and not ident
        ok_all &= ok
        print(f"[test] node {name:28s} x{list(xi.shape)} -> {list(e.shape)}: worst |err|/mag "
              f"{log2s(r)} (err {err:.2e} at mag {m:.2e}), max|err| {emax:.2e}"
              f"{'  IDENTICAL' if ident else ''}  {'ok' if ok else 'FAIL'}")
    placed = on_ep(sep.placement) > 0
    print(f"[test]   placement: {dict((p_, len(n_)) for p_, n_ in sep.placement.items())}"
          f"  {'ok' if placed else 'FAIL (not claimed)'}")
    return record(ok_all and placed, "node " + name)


def refusal_case(lib, tmp, name, x_shape, w, b, attrs, w_as_input=False):
    """A node the handler must leave on the CPU: the EP run equals the CPU run bit for bit and the
    profile shows no EP node."""
    path = os.path.join(tmp, name + ".onnx")
    ct_model(path, x_shape, w, b, attrs, w_as_input=w_as_input)
    rs = np.random.RandomState(7)
    feeds = {"X": rs.randn(*x_shape).astype(np.float32)}
    if w_as_input:
        feeds["W"] = w.astype(np.float32)
    cpu = cpu_run(path, feeds)[0]
    ep, placement = ep_run(lib, path, feeds)
    ok = on_ep(placement) == 0 and np.array_equal(ep[0], cpu)
    print(f"[test] refusal {name:25s} on EP {on_ep(placement)}, identical {np.array_equal(ep[0], cpu)}"
          f"  {'ok' if ok else 'FAIL'}")
    return record(ok, "refusal " + name)


def synthetic(lib, tmp):
    rs = np.random.RandomState(0)

    def W(ic, oc, kh, kw, scale=0.1):
        return (rs.randn(ic, oc, kh, kw) * scale).astype(np.float32)

    def X(n, c, h, w):
        return rs.randn(n, c, h, w).astype(np.float32)

    # stride / pad / kernel / output_padding / dilation / asymmetric pads / batch
    node_case(lib, tmp, "k4s2p1", X(1, 64, 16, 16), W(64, 32, 4, 4), None, (4, 4), (2, 2), (1, 1, 1, 1))
    node_case(lib, tmp, "k4s2p1_bias_n3", X(3, 48, 12, 12), W(48, 16, 4, 4),
              (rs.randn(16) * 0.5).astype(np.float32), (4, 4), (2, 2), (1, 1, 1, 1))
    node_case(lib, tmp, "k2s2_bias", X(1, 64, 16, 16), W(64, 32, 2, 2),
              (rs.randn(32) * 0.5).astype(np.float32), (2, 2), (2, 2), (0, 0, 0, 0))
    node_case(lib, tmp, "k3s1p1", X(1, 32, 20, 20), W(32, 48, 3, 3), None, (3, 3), (1, 1), (1, 1, 1, 1))
    node_case(lib, tmp, "k3s2p1_opad1", X(1, 32, 15, 15), W(32, 16, 3, 3), None, (3, 3), (2, 2),
              (1, 1, 1, 1), op=(1, 1))
    node_case(lib, tmp, "k4s4_bias", X(1, 48, 10, 10), W(48, 48, 4, 4),
              (rs.randn(48) * 0.5).astype(np.float32), (4, 4), (4, 4), (0, 0, 0, 0))
    node_case(lib, tmp, "k5s3p2_rect", X(1, 16, 9, 13), W(16, 24, 5, 5), None, (5, 5), (3, 3),
              (2, 2, 2, 2))
    node_case(lib, tmp, "k3x1s2x1", X(1, 32, 8, 12), W(32, 16, 3, 1), None, (3, 1), (2, 1), (1, 0, 1, 0))
    node_case(lib, tmp, "k3s1d2p2", X(1, 32, 12, 12), W(32, 32, 3, 3), None, (3, 3), (1, 1),
              (2, 2, 2, 2), d=(2, 2))
    node_case(lib, tmp, "k4s2_asym_p1p0", X(1, 32, 10, 10), W(32, 16, 4, 4), None, (4, 4), (2, 2),
              (1, 1, 0, 0))
    node_case(lib, tmp, "k3s2_asym_opad", X(1, 32, 9, 11), W(32, 16, 3, 3), None, (3, 3), (2, 2),
              (1, 2, 1, 1), op=(1, 1))
    node_case(lib, tmp, "ic3_k4s2", X(1, 3, 32, 32), W(3, 8, 4, 4), None, (4, 4), (2, 2), (1, 1, 1, 1))
    # A symbolic input size: 64x64 packs; 96x96 puts P past the entry's 64 MiB bound (4096 taps x
    # 9216 pixels x 2 bytes) and runs on the host path; then 64x64 again, and a batch of two.
    node_case(lib, tmp, "sym_hw_host_split", X(1, 32, 64, 64), W(32, 256, 4, 4, 0.05), None, (4, 4),
              (2, 2), (1, 1, 1, 1), sym_hw=True, sizes=[(1, 96, 96), (1, 64, 64), (2, 40, 48)])

    # refusals
    refusal_case(lib, tmp, "grouped_g2", (1, 32, 8, 8), W(32, 8, 4, 4), None,
                 attrs_of((4, 4), (2, 2), (1, 1, 1, 1), group=2))
    refusal_case(lib, tmp, "depthwise", (1, 32, 8, 8), W(32, 1, 4, 4), None,
                 attrs_of((4, 4), (2, 2), (1, 1, 1, 1), group=32))
    refusal_case(lib, tmp, "auto_pad_same", (1, 16, 8, 8), W(16, 8, 3, 3), None,
                 dict(kernel_shape=[3, 3], strides=[2, 2], auto_pad="SAME_UPPER"))
    refusal_case(lib, tmp, "output_shape", (1, 16, 8, 8), W(16, 8, 3, 3), None,
                 dict(kernel_shape=[3, 3], strides=[2, 2], output_shape=[16, 16]))
    refusal_case(lib, tmp, "weight_is_input", (1, 16, 8, 8), W(16, 8, 4, 4), None,
                 attrs_of((4, 4), (2, 2), (1, 1, 1, 1)), w_as_input=True)
    refusal_case(lib, tmp, "trailing_pad_crop", (1, 16, 8, 8), W(16, 8, 4, 4), None,
                 attrs_of((4, 4), (2, 2), (0, 0, 2, 2)))


# ---------------------------------------------------------------------------------------------
# Model layers and end to end
# ---------------------------------------------------------------------------------------------
def ct_nodes(model):
    return [n for n in model.graph.node if n.op_type == "ConvTranspose"]


def capture_ct_inputs(model_path, feeds, tmp):
    """Run the model on the CPU EP with every ConvTranspose's input promoted to a graph output.
    Returns [(node, input array)]."""
    m = onnx.load(model_path)
    nodes = ct_nodes(m)
    have = {o.name for o in m.graph.output}
    for n in nodes:
        if n.input[0] not in have:
            m.graph.output.append(helper.make_tensor_value_info(n.input[0], TensorProto.FLOAT, None))
    path = os.path.join(tmp, "capture.onnx")
    onnx.save(m, path)
    outs = cpu_run(path, feeds, [n.input[0] for n in nodes])
    return list(zip(nodes, outs)), m


def model_layers(lib, tmp, model_path, feeds, tag):
    caps, m = capture_ct_inputs(model_path, feeds, tmp)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    for i, (n, x) in enumerate(caps):
        a = {at.name: helper.get_attribute_value(at) for at in n.attribute}
        w = inits[n.input[1]]
        b = inits[n.input[2]] if len(n.input) > 2 and n.input[2] else None
        k = tuple(a.get("kernel_shape", w.shape[2:]))
        s = tuple(a.get("strides", (1, 1)))
        p = tuple(a.get("pads", (0, 0, 0, 0)))
        d = tuple(a.get("dilations", (1, 1)))
        op = tuple(a.get("output_padding", (0, 0)))
        node_case(lib, tmp, f"{tag}_ct{i + 1}_{w.shape[0]}to{w.shape[1]}", x.astype(np.float32),
                  w, b, k, s, p, d, op)


def facade_label_map(seed=0):
    """A structured stand-in for a CMP facade label map (the pix2pix facades generator's input):
    flat class colors, a wall with rows of windows, a door, a cornice. Normalized to [-1, 1]."""
    rs = np.random.RandomState(seed)
    pal = np.array([[0, 0, 170], [0, 85, 255], [255, 85, 0], [0, 255, 255], [170, 255, 85],
                    [255, 255, 0], [85, 255, 170], [255, 0, 0], [170, 0, 0]], np.float32)
    img = np.tile(pal[1], (256, 256, 1))
    img[:24] = pal[0]
    img[24:32] = pal[4]
    for r in range(4):
        for c in range(5):
            y0 = 44 + r * 48 + rs.randint(-2, 3)
            x0 = 12 + c * 48 + rs.randint(-2, 3)
            img[y0:y0 + 30, x0:x0 + 26] = pal[2]
            img[y0 + 30:y0 + 34, x0 - 2:x0 + 28] = pal[3]
    img[200:256, 108:148] = pal[5]
    x = img.transpose(2, 0, 1)[None] / 127.5 - 1.0
    return x.astype(np.float32)


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
            ok = n_ep == 8 and float(d.max()) <= PIX_MAXABS and psnr >= PIX_PSNR
            print(f"[test] pix2pix e2e ({tag}, {opt.name}): {n_ep} node(s) on the EP, max abs "
                  f"{d.max():.3e}, mean abs {d.mean():.3e}, PSNR {psnr:.1f} dB  {'ok' if ok else 'FAIL'}")
            record(ok, f"pix2pix e2e {tag} {opt.name}")
    os.environ["ROCKET_ORT_CONVTRANSPOSE"] = "0"
    try:
        cpu = cpu_run(path, {in_name: x_lab})[0]
        ep, placement = ep_run(lib, path, {in_name: x_lab})
    finally:
        del os.environ["ROCKET_ORT_CONVTRANSPOSE"]
    ok = on_ep(placement) == 0 and np.array_equal(ep[0], cpu)
    print(f"[test] pix2pix with ROCKET_ORT_CONVTRANSPOSE=0: {on_ep(placement)} node(s) on the EP, "
          f"identical to the CPU EP {np.array_equal(ep[0], cpu)}  {'ok' if ok else 'FAIL'}")
    record(ok, "pix2pix knob off")


def sam_image_pe(model_path):
    """SAM's image-wide positional embedding [1,256,64,64], from the decoder's own
    shared_image_embedding.positional_embedding [2,128] (the transformers SamModel formula)."""
    m = onnx.load(model_path)
    g = next(numpy_helper.to_array(i) for i in m.graph.initializer
             if i.name == "shared_image_embedding.positional_embedding").astype(np.float64)
    s = 64
    grid = np.ones((s, s))
    y = (np.cumsum(grid, 0) - 0.5) / s
    x = (np.cumsum(grid, 1) - 0.5) / s
    c = np.stack([x, y], -1) * 2 - 1
    c = 2 * np.pi * (c @ g)
    pe = np.concatenate([np.sin(c), np.cos(c)], -1)
    return pe.transpose(2, 0, 1)[None].astype(np.float32)


def sam_feeds(emb, pe, npts):
    if npts == 1:
        pts = np.array([[[[520.0, 400.0]]]], np.float32)
    else:
        g = int(round(math.sqrt(npts)))
        v = (np.arange(g) + 0.5) * 1024.0 / g
        pts = np.array([[x, y] for y in v for x in v], np.float32).reshape(1, npts, 1, 2)
    labels = np.ones((1, npts, 1), np.int64)
    return {"input_points": pts, "input_labels": labels, "image_embeddings": emb,
            "image_positional_embeddings": pe}


def sam_decoder(lib, tmp, path, emb_path):
    emb = np.load(emb_path).astype(np.float32)
    pe = sam_image_pe(path)
    model_layers(lib, tmp, path, sam_feeds(emb, pe, 1), "sam")
    for npts in (1, 64):
        feeds = sam_feeds(emb, pe, npts)
        cs, cm = cpu_run(path, feeds, ["iou_scores", "pred_masks"])
        (es, em), placement = ep_run(lib, path, feeds, ["iou_scores", "pred_masks"],
                                     opt=ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        dm = float(np.abs(em.astype(np.float64) - cm).max())
        ds = float(np.abs(es.astype(np.float64) - cs).max())
        a = (em > 0).reshape(-1, 256 * 256); b = (cm > 0).reshape(-1, 256 * 256)
        inter = (a & b).sum(1); union = (a | b).sum(1)
        iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        n_ep = on_ep(placement)
        ok = n_ep == 2 and float(iou.min()) >= SAM_IOU and ds <= SAM_SCORE
        print(f"[test] SAM decoder P={npts}: {n_ep} node(s) on the EP, mask logit max abs {dm:.3e} "
              f"(logit range {cm.min():.1f}..{cm.max():.1f}), mask IoU min {iou.min():.5f} mean "
              f"{iou.mean():.5f} over {iou.size} masks, iou_scores max abs {ds:.2e}  "
              f"{'ok' if ok else 'FAIL'}")
        record(ok, f"sam decoder P={npts}")


def main():
    args = sys.argv[1:]
    lib = os.path.abspath(args[0])

    def opt(flag):
        return os.path.abspath(args[args.index(flag) + 1]) if flag in args else None

    tmp = tempfile.mkdtemp(prefix="ort_rocket_ct_")   # the per-case models; the layer cases carry real weights
    try:
        synthetic(lib, tmp)
        if opt("--pix2pix"):
            pix2pix(lib, tmp, opt("--pix2pix"))
        if opt("--sam-decoder") and opt("--sam-embedding"):
            sam_decoder(lib, tmp, opt("--sam-decoder"), opt("--sam-embedding"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    gc.collect()
    print("RESULT:", "PASS" if not FAILS else "FAIL " + ", ".join(FAILS),
          f"(node gate |err|/mag <= {log2s(REL_GATE)})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
