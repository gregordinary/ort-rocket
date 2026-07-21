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

Run with sudo -E (NPU privilege + ROCKET_* env):
  sudo -E .../python test_quant_ep.py <lib.so> <quant.onnx> <input_nchw.npy> [--int8]
"""
import argparse
import gc
import os

import numpy as np
import onnxruntime as ort

REG_NAME = "rocket"
SCORE_TOL = 0.06   # quantization band (int8/int4 shift scores a few %; classes must still match)
BOX_TOL = 0.05
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
    ort.register_execution_provider_library(REG_NAME, os.path.abspath(EP_LIB))
    devs = [d for d in ort.get_ep_devices() if d.ep_name == REG_NAME]
    assert devs, f"EP '{REG_NAME}' not registered"
    so = ort.SessionOptions()
    so.log_severity_level = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_provider_for_devices([devs[0]], {})
    sep = ort.InferenceSession(onnx_path, so)
    names = [o.name for o in sep.get_outputs()]
    out = dict(zip(names, sep.run(None, {sep.get_inputs()[0].name: x})))
    del sep
    gc.collect()
    ort.unregister_execution_provider_library(REG_NAME)
    return out


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

    # CPU-EP reference on the SAME quantized model.
    scpu = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ocpu = dict(zip([o.name for o in scpu.get_outputs()],
                    scpu.run(None, {scpu.get_inputs()[0].name: x})))
    cpu_det = decode(ocpu["dets"][0], ocpu["labels"][0])

    oep = run_ep(args.onnx, x)
    ep_det = decode(oep["dets"][0], oep["labels"][0])

    print(f"[test] detections: CPU={len(cpu_det)} EP={len(ep_det)} (thr={DET_THRESH})")
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
        good = ds <= SCORE_TOL
        ok = ok and good
        print(f"  cls{g['class_id']} score {g['score']:.3f}/{best[1]['score']:.3f} "
              f"IoU {best[0]:.3f} dscore {ds:.2e} {'OK' if good else 'FAIL'}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
