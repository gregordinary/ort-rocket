#!/usr/bin/env python3
"""On-device faithfulness gate for the ort-rocket SAM ViT-Det image-encoder offload.

Runs the SAM vision encoder twice in one process -- once on the ORT CPU EP, once with ort-rocket
offloading the windowed + rel-pos encoder (and conv neck) to the NPU -- and grades the neck output
(last_hidden_state [1,256,64,64]) agreement. Same box, same session graph, so the CPU EP is the
reference (no cross-arch golden question). A matcher miss runs the encoder on the CPU EP at cosine
1.0, so the gate also asserts the EP executed a node (from ONNX Runtime's profile) and that the
output is not bit-identical to the CPU EP's.

  sudo -E .../python test_sam_ep.py <lib.so> <sam_vision.onnx> <input_nchw.npy>
"""
import os, sys, time
import numpy as np
import onnxruntime as ort

from ep_common import EpSession, check_not_identical, check_placement

FEAT_COS = 0.999   # exit gate: encoder-feature cosine vs the CPU EP


def main():
    ep_lib = os.path.abspath(sys.argv[1])
    onnx_path = os.path.abspath(sys.argv[2])
    x = np.load(sys.argv[3])
    print(f"[test] input {x.shape} {x.dtype}")

    scpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = scpu.get_inputs()[0].name
    t0 = time.time()
    cpu = scpu.run(["last_hidden_state"], {in_name: x})[0]
    t_cpu = time.time() - t0

    # INFO log surfaces the EP claim line. The profiler is on, so the timing below carries its
    # small per-node cost on the EP side only.
    with EpSession(ep_lib, onnx_path, "rocket", log_level=1) as sep:
        ep = sep.run(["last_hidden_state"], {in_name: x})[0]     # warm (Compile ran packing already)
        t0 = time.time()
        ep = sep.run(["last_hidden_state"], {in_name: x})[0]
        t_ep = time.time() - t0

    a = ep.astype(np.float64).ravel(); b = cpu.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(f"[test] last_hidden_state {ep.shape}: cos(ep,cpu)={cos:.7f} rel_l2={rel:.3e} "
          f"max_abs={np.max(np.abs(a-b)):.3e}")
    print(f"[test] warm encode: cpu-EP {t_cpu*1e3:.0f} ms  rocket-EP {t_ep*1e3:.0f} ms  "
          f"({t_cpu/t_ep:.2f}x)")
    ok = check_placement(sep.placement)
    ok = check_not_identical([("last_hidden_state", ep, cpu)]) and ok
    ok = ok and cos >= FEAT_COS
    print("RESULT:", "PASS" if ok else "FAIL", f"(gate cos>={FEAT_COS})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
