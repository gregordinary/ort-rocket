#!/usr/bin/env python3
"""On-device faithfulness gate for the ort-rocket SigLIP (plain-ViT) offload.

Runs the SigLIP vision encoder twice in one process -- once on the ORT CPU EP, once with ort-rocket
offloading the encoder to the NPU -- and grades the last_hidden_state agreement. Same-box, same
session graph, so there is no cross-arch golden question: the CPU EP is the reference.

  sudo -E .../python test_siglip_ep.py <lib.so> <siglip_vision.onnx> <input_nchw.npy>
"""
import os, sys, gc
import numpy as np
import onnxruntime as ort

EMB_COS = 0.9999   # exit gate: embedding cosine vs the CPU EP

def main():
    ep_lib = os.path.abspath(sys.argv[1])
    onnx_path = os.path.abspath(sys.argv[2])
    x = np.load(sys.argv[3])
    print(f"[test] input {x.shape} {x.dtype}")

    scpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = scpu.get_inputs()[0].name
    cpu = scpu.run(["last_hidden_state"], {in_name: x})[0]

    ort.register_execution_provider_library("rocket", ep_lib)
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "rocket"]
    assert devs, "rocket EP not registered"
    so = ort.SessionOptions()
    so.log_severity_level = 2
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_provider_for_devices([devs[0]], {})
    sep = ort.InferenceSession(onnx_path, so)
    ep = sep.run(["last_hidden_state"], {in_name: x})[0]
    del sep; gc.collect()
    ort.unregister_execution_provider_library("rocket")

    a = ep.astype(np.float64).ravel(); b = cpu.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(f"[test] last_hidden_state {ep.shape}: cos(ep,cpu)={cos:.7f} rel_l2={rel:.3e} "
          f"max_abs={np.max(np.abs(a-b)):.3e}")
    ok = cos >= EMB_COS
    print("RESULT:", "PASS" if ok else "FAIL", f"(gate cos>={EMB_COS})")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
