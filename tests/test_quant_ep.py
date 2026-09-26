#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Quantized-ONNX consumption test for the ort-rocket EP.

Proves, on device, that the EP consumes an int8/int4 QDQ RF-DETR model: it recognizes the
DequantizeLinear-wrapped weights, dequantizes them to fp16 at Compile (int8/int4/int32 biases,
per-tensor and per-channel), and runs the backbone+projector on the NPU with detections faithful
to a plain CPU-EP run of the SAME quantized model (the class set must match; scores within a
quantization band). With ROCKET_ORT_INT8=1 it additionally exercises the native-int8 (W8A8)
datapath, which must stay faithful (slower is expected -- int8 matmul reads int32 back to the
host; see rocket_backbone.h).

ROCKET_ORT_STRICT=1 is set unless the caller chose, so an int8 marshaling miss fails Compile
rather than falling back to fp16 and passing as the int8 datapath. A matcher miss runs the model
on the CPU EP and matches the reference exactly, so the test also asserts the EP executed a node
and that neither output is bit-identical to the CPU EP's. A CPU run with no detection fails.

Run with sudo -E (NPU privilege + ROCKET_* env):
  sudo -E .../python test_quant_ep.py <lib.so> <quant.onnx> <input_nchw.npy> [--int8]
"""
import argparse
import os

import numpy as np
import onnxruntime as ort

from ep_common import EpSession, check_not_identical, check_placement

REG_NAME = "rocket"
SCORE_TOL = 0.06   # quantization band (int8/int4 shift scores a few %; classes must still match)
BOX_TOL = 0.05     # the same band on a matched box's normalized cxcywh, largest coordinate
DET_THRESH = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def decode(dets, labels, thr=DET_THRESH):
    probs = sigmoid(labels)
    cls, score = probs.argmax(1), probs.max(1)
    out = [{"query": int(q), "class_id": int(cls[q]), "score": float(score[q]),
            "box": [float(v) for v in dets[q]]} for q in range(dets.shape[0]) if score[q] >= thr]
    out.sort(key=lambda d: -d["score"])
    return out


def run_ep(onnx_path, x):
    """The EP's outputs, and the placement ONNX Runtime's profile recorded."""
    with EpSession(EP_LIB, onnx_path, REG_NAME, log_level=1) as sep:
        out = dict(zip(sep.output_names(), sep.run(None, {sep.input_name(): x})))
    return out, sep.placement


def main():
    global EP_LIB
    ap = argparse.ArgumentParser()
    ap.add_argument("lib"); ap.add_argument("onnx"); ap.add_argument("input")
    ap.add_argument("--int8", action="store_true", help="also exercise the native-int8 datapath")
    args = ap.parse_args()
    EP_LIB = args.lib
    x = np.load(args.input)
    mode = "native-int8 (W8A8)" if (args.int8 or os.environ.get("ROCKET_ORT_INT8")) else "dequant->fp16"
    print(f"[test] quantized model {os.path.basename(args.onnx)}  input {x.shape}  mode={mode}")
    if args.int8:
        os.environ["ROCKET_ORT_INT8"] = "1"
    # Read at Compile. Without it a native-int8 marshaling miss falls back to fp16 and the run
    # passes as the int8 datapath it never took.
    os.environ.setdefault("ROCKET_ORT_STRICT", "1")

    # CPU-EP reference on the SAME quantized model.
    scpu = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ocpu = dict(zip([o.name for o in scpu.get_outputs()],
                    scpu.run(None, {scpu.get_inputs()[0].name: x})))
    cpu_det = decode(ocpu["dets"][0], ocpu["labels"][0])

    oep, placement = run_ep(args.onnx, x)
    ep_det = decode(oep["dets"][0], oep["labels"][0])

    # Did the EP take the model, and did the NPU compute? A matcher miss passes the match below.
    ok_ran = check_placement(placement)
    ok_ran = check_not_identical([(n, oep[n], ocpu[n]) for n in ("dets", "labels")]) and ok_ran

    print(f"[test] detections: CPU={len(cpu_det)} EP={len(ep_det)} (thr={DET_THRESH})")
    if not cpu_det:
        print("[test] the CPU run has no detection at this threshold, so nothing is compared: "
              "use an input with objects in it")
    # Match by class + best box IoU rather than query index: heavy quantization (esp. int4) can
    # reshuffle which query fires for an object, so the faithful check is that every CPU
    # detection has a same-class EP detection over the same box.
    def iou(a, b):  # cxcywh normalized
        ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
        bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
        ix, iy = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua > 0 else 0.0
    ok = True
    for g in cpu_det:
        cands = [(iou(g["box"], e["box"]), e) for e in ep_det if e["class_id"] == g["class_id"]]
        best = max(cands, key=lambda t: t[0], default=(0.0, None))
        if best[1] is None or best[0] < 0.5:
            print(f"  cls{g['class_id']} score {g['score']:.3f}: NO MATCH (best IoU {best[0]:.2f})")
            ok = False
            continue
        ds = abs(best[1]["score"] - g["score"])
        db = max(abs(a - b) for a, b in zip(best[1]["box"], g["box"]))
        good = ds <= SCORE_TOL and db <= BOX_TOL
        ok = ok and good
        print(f"  cls{g['class_id']} score {g['score']:.3f}/{best[1]['score']:.3f} "
              f"IoU {best[0]:.3f} dscore {ds:.2e} dbox {db:.2e} {'OK' if good else 'FAIL'}")

    # EP detections no CPU detection accounts for. Reported, not graded: the CPU EP runs the
    # QDQ activations in int8 where the EP runs them in fp16, so a query near the threshold can
    # cross it on one side only.
    def matched(e):
        return any(g["class_id"] == e["class_id"] and iou(g["box"], e["box"]) >= 0.5 for g in cpu_det)
    extra = [e for e in ep_det if not matched(e)]
    if extra:
        print(f"[warn] {len(extra)} EP detection(s) with no CPU counterpart: "
              + ", ".join(f"q{e['query']}/c{e['class_id']}/{e['score']:.3f}" for e in extra[:6]))

    passed = ok and ok_ran and bool(cpu_det)
    print("\nRESULT:", "PASS" if passed else "FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
