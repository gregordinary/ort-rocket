#!/usr/bin/env python3
"""Capture the raw ort-rocket EP output tensors to .npy for a byte-identity diff.

Used to prove a value-preserving refactor (e.g. a geometry generalization) leaves the
on-NPU output bit-identical: run once per build, then np.array_equal the two captures.

  sudo -E .../python capture_ep_raw.py <lib.so> <model.onnx> <input_nchw.npy> <out_prefix>
"""
import os
import sys
import numpy as np
import onnxruntime as ort

REG_NAME = "rocket"


def main():
    ep_lib = os.path.abspath(sys.argv[1])
    onnx_path = os.path.abspath(sys.argv[2])
    x = np.load(sys.argv[3])
    out_prefix = sys.argv[4]

    ort.register_execution_provider_library(REG_NAME, ep_lib)
    devs = [d for d in ort.get_ep_devices() if d.ep_name == REG_NAME]
    assert devs, f"EP '{REG_NAME}' not registered as a device"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_provider_for_devices([devs[0]], {})
    sess = ort.InferenceSession(onnx_path, so)
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    out = dict(zip(names, sess.run(None, {in_name: x})))
    del sess
    import gc
    gc.collect()
    ort.unregister_execution_provider_library(REG_NAME)

    for n in names:
        p = f"{out_prefix}_{n}.npy"
        np.save(p, out[n])
        print(f"[capture] {p} {out[n].shape} {out[n].dtype}")


if __name__ == "__main__":
    main()
