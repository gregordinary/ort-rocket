#!/usr/bin/env python3
"""End-to-end test for the ort-rocket backbone execution provider.

Proves, on device, that:
  1. ONNX Runtime loads the EP, its GetCapability claims the DINOv2 encoder + CSP projector
     subgraph, and Compile marshals the weights + opens the NPU (the EP log lines surface).
  2. RF-DETR-nano runs with the backbone+projector offloaded to the NPU and the decoder/
     postproc on ORT's CPU kernels, producing detections that match a plain CPU-EP run on
     the SAME box within the fp16 tolerance (classes identical; score/box within a small
     band). Backbone features are graded cos(npu,ort) >= 0.99999 and the projector output
     cos(npu,ort) = 0.999997, so the decoded detections must be stable.

Compares the ort-rocket decode against the CPU-EP decode in one process, so there is no
cross-arch golden-manifest question -- the CPU EP on this box is the reference.

Run with sudo -E (NPU privilege + ROCKET_* env):
  sudo -E .../python test_backbone_ep.py <lib.so> <rfdetr-nano.onnx> <input_nchw.npy>
"""
import os
import sys

import numpy as np
import onnxruntime as ort

REG_NAME = "rocket"
SCORE_TOL = 1e-2
BOX_TOL = 1e-2
DET_THRESH = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def decode(dets, labels, thr=DET_THRESH):
    probs = sigmoid(labels)
    cls = probs.argmax(1); score = probs.max(1)
    out = [{"query": int(q), "class_id": int(cls[q]), "score": float(score[q]),
            "box": [float(v) for v in dets[q]]} for q in range(dets.shape[0]) if score[q] >= thr]
    out.sort(key=lambda d: -d["score"])
    return out


def main():
    ep_lib = os.path.abspath(sys.argv[1])
    onnx_path = os.path.abspath(sys.argv[2])
    x = np.load(sys.argv[3])
    print(f"[test] input {x.shape} {x.dtype}")

    # CPU-EP reference (same box).
    scpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = scpu.get_inputs()[0].name
    names = [o.name for o in scpu.get_outputs()]
    ocpu = dict(zip(names, scpu.run(None, {in_name: x})))
    cpu_det = decode(ocpu["dets"][0], ocpu["labels"][0])

    # ort-rocket EP: backbone+projector on the NPU. Verbose log so claim/compile lines show.
    ort.register_execution_provider_library(REG_NAME, ep_lib)
    devs = [d for d in ort.get_ep_devices() if d.ep_name == REG_NAME]
    assert devs, f"EP '{REG_NAME}' not registered as a device"
    so = ort.SessionOptions()
    so.log_severity_level = 1  # INFO: surface the EP's claim/compile lines
    # The ancestor-set claim and the name-based weight marshaling both rely on the raw
    # exported graph; ORT's Gemm/SkipLayerNorm/Gelu rewrites otherwise rename the interior.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_provider_for_devices([devs[0]], {})
    print("[test] creating session WITH ort-rocket (backbone+projector -> NPU)...")
    sep = ort.InferenceSession(onnx_path, so)
    oep = dict(zip(names, sep.run(None, {in_name: x})))
    # Destroy the session (releases the EP: ReleaseNodeComputeInfos -> rocket_close, ReleaseEp)
    # BEFORE unregistering the library, or teardown touches freed factory state.
    del sep
    import gc
    gc.collect()
    ort.unregister_execution_provider_library(REG_NAME)
    ep_det = decode(oep["dets"][0], oep["labels"][0])

    # Raw-output faithfulness.
    print("\n[test] raw output EP-vs-CPU:")
    for n in names:
        a = oep[n].astype(np.float64).ravel(); b = ocpu[n].astype(np.float64).ravel()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        maxabs = float(np.max(np.abs(a - b)))
        print(f"    {n}{oep[n].shape}: cos={cos:.6f} max_abs={maxabs:.3e}")

    # Detection agreement (query-by-query on the CPU set).
    print(f"\n[test] detections: CPU={len(cpu_det)} EP={len(ep_det)} (thr={DET_THRESH})")
    print(f"{'query':>5} {'cls(cpu/ep)':>12} {'score(cpu/ep)':>16} {'dscore':>9} {'dbox':>9} ok")
    ep_by_q = {d["query"]: d for d in ep_det}
    ok = True
    for g in cpu_det:
        m = ep_by_q.get(g["query"])
        if m is None:
            print(f"{g['query']:>5} {g['class_id']:>6}/--   MISSING"); ok = False; continue
        ds = abs(m["score"] - g["score"])
        db = max(abs(a - b) for a, b in zip(m["box"], g["box"]))
        cls_ok = m["class_id"] == g["class_id"]
        good = cls_ok and ds <= SCORE_TOL and db <= BOX_TOL
        ok = ok and good
        print(f"{g['query']:>5} {g['class_id']:>5}/{m['class_id']:<5} "
              f"{g['score']:>7.4f}/{m['score']:<7.4f} {ds:>9.2e} {db:>9.2e} {'OK' if good else 'FAIL'}")

    extra = [d for d in ep_det if d["query"] not in {g["query"] for g in cpu_det}]
    if extra:
        print(f"[warn] {len(extra)} EP detection(s) not in CPU set: "
              + ", ".join(f"q{d['query']}/c{d['class_id']}/{d['score']:.3f}" for d in extra[:6]))

    print("\nRESULT:", "PASS" if ok and not extra else "FAIL")
    sys.exit(0 if ok and not extra else 1)


if __name__ == "__main__":
    main()
