#!/usr/bin/env python3
"""On-device faithfulness gate for the ort-rocket CLIP (plain-ViT) offload.

Runs the CLIP ViT-B/16 vision encoder twice in one process -- once on the ORT CPU EP, once with
ort-rocket offloading the encoder to the NPU -- and grades both graph outputs against the CPU EP:

  * last_hidden_state -- the offloaded encoder exit ([1, L+1, d], un-normed; CLIP keeps the cls
    token in the sequence and does not post-LN it).
  * pooler_output     -- exercises the host tail (Gather the cls row -> post_layernorm) composed
    with the NPU encoder, i.e. the full CLIP image-embedding path.

Same-box, same session graph, so the CPU EP is the reference (no cross-arch golden question). A
matcher miss runs the encoder on the CPU EP at cosine 1.0, so the gate also asserts the EP executed
a node (from ONNX Runtime's profile) and that neither output is bit-identical to the CPU EP's.

  sudo -E .../python test_clip_ep.py <lib.so> <clip_vision.onnx> <input_nchw.npy>
"""
import os, sys
import numpy as np
import onnxruntime as ort

from ep_common import EpSession, check_not_identical, check_placement

EMB_COS = 0.9999   # exit gate: output cosine vs the CPU EP

def cos_rel(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return cos, rel, float(np.max(np.abs(a - b)))

def main():
    ep_lib = os.path.abspath(sys.argv[1])
    onnx_path = os.path.abspath(sys.argv[2])
    x = np.load(sys.argv[3])
    print(f"[test] input {x.shape} {x.dtype}")
    outs = ["last_hidden_state", "pooler_output"]

    scpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = scpu.get_inputs()[0].name
    cpu = scpu.run(outs, {in_name: x})

    with EpSession(ep_lib, onnx_path, "rocket", log_level=2) as sep:
        ep = sep.run(outs, {in_name: x})

    ok = check_placement(sep.placement)
    ok = check_not_identical(list(zip(outs, ep, cpu))) and ok
    for name, e, c in zip(outs, ep, cpu):
        cos, rel, mx = cos_rel(e, c)
        good = cos >= EMB_COS
        ok = ok and good
        print(f"[test] {name:18} {e.shape}: cos(ep,cpu)={cos:.7f} rel_l2={rel:.3e} max_abs={mx:.3e}"
              f"  {'ok' if good else 'FAIL'}")
    print("RESULT:", "PASS" if ok else "FAIL", f"(gate cos>={EMB_COS})")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
