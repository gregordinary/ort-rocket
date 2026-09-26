#!/usr/bin/env python3
"""On-device faithfulness gate for the ort-rocket SigLIP (plain-ViT) offload.

Runs the SigLIP vision encoder twice in one process -- once on the ORT CPU EP, once with ort-rocket
offloading the encoder to the NPU -- and grades the last_hidden_state agreement. Same-box, same
session graph, so there is no cross-arch golden question: the CPU EP is the reference. A matcher
miss runs the encoder on the CPU EP at cosine 1.0, so the gate also asserts the EP executed a node
(from ONNX Runtime's profile) and that the output is not bit-identical to the CPU EP's.

  sudo -E .../python test_siglip_ep.py <lib.so> <siglip_vision.onnx> <input_nchw.npy>
"""
import os, sys
import numpy as np
import onnxruntime as ort

from ep_common import EpSession, check_not_identical, check_placement

EMB_COS = 0.9999   # exit gate: embedding cosine vs the CPU EP

def main():
    ep_lib = os.path.abspath(sys.argv[1])
    onnx_path = os.path.abspath(sys.argv[2])
    x = np.load(sys.argv[3])
    print(f"[test] input {x.shape} {x.dtype}")

    scpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = scpu.get_inputs()[0].name
    cpu = scpu.run(["last_hidden_state"], {in_name: x})[0]

    with EpSession(ep_lib, onnx_path, "rocket", log_level=2) as sep:
        ep = sep.run(["last_hidden_state"], {in_name: x})[0]

    a = ep.astype(np.float64).ravel(); b = cpu.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(f"[test] last_hidden_state {ep.shape}: cos(ep,cpu)={cos:.7f} rel_l2={rel:.3e} "
          f"max_abs={np.max(np.abs(a-b)):.3e}")
    ok = check_placement(sep.placement)
    ok = check_not_identical([("last_hidden_state", ep, cpu)]) and ok
    ok = ok and cos >= EMB_COS
    print("RESULT:", "PASS" if ok else "FAIL", f"(gate cos>={EMB_COS})")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
