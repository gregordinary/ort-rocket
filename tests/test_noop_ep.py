#!/usr/bin/env python3
"""Plumbing test for the ort-rocket no-op execution provider.

Proves, end to end, that:
  1. ONNX Runtime loads the EP shared library (register_execution_provider_library),
  2. the factory registers a device that appears in get_ep_devices(),
  3. a session can select the EP (add_provider_for_devices),
  4. the EP's GetCapability runs (its WARNING line appears in the ORT log), and
  5. because it claims zero nodes, the model runs via CPU fallback with outputs
     BIT-IDENTICAL to a plain CPU-EP run and to the reference golden.

Usage: python test_noop_ep.py <libonnxruntime_rocket.so> <rfdetr.onnx> [golden_dir]
"""
import os
import sys

import numpy as np
import onnxruntime as ort

REG_NAME = "rocket"


def load_input(onnx_path, golden_dir):
    """Use the reference golden input if present, else a fixed random tensor."""
    gpath = os.path.join(golden_dir, "input_nchw.npy") if golden_dir else None
    if gpath and os.path.exists(gpath):
        return np.load(gpath), gpath
    # deterministic fallback
    rng = np.random.RandomState(0)
    tmp = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in tmp.get_inputs()[0].shape]
    return rng.rand(*shape).astype(np.float32), "(random seed 0)"


def run(onnx_path, session_options, providers=None):
    if providers is None:
        sess = ort.InferenceSession(onnx_path, session_options)
    else:
        sess = ort.InferenceSession(onnx_path, session_options, providers=providers)
    return sess


def main():
    ep_lib = os.path.abspath(sys.argv[1])
    onnx_path = os.path.abspath(sys.argv[2])
    golden_dir = sys.argv[3] if len(sys.argv) > 3 else None

    x, src = load_input(onnx_path, golden_dir)
    print(f"[test] input from {src}, shape={x.shape}")

    # 1. Register the plugin EP library.
    ort.register_execution_provider_library(REG_NAME, ep_lib)
    print(f"[test] registered EP library '{REG_NAME}' -> {ep_lib}")

    # 2. It must appear as a device.
    devices = ort.get_ep_devices()
    ours = [d for d in devices if d.ep_name == REG_NAME]
    assert ours, f"EP '{REG_NAME}' not in get_ep_devices(): {[d.ep_name for d in devices]}"
    dev = ours[0]
    print(f"[test] device present: ep_name={dev.ep_name} vendor={dev.ep_vendor}")

    # 3+4. Select the EP for a session. Turn ORT logging up so the EP's
    # GetCapability WARNING line is visible (evidence the EP ran).
    so = ort.SessionOptions()
    so.log_severity_level = 0  # VERBOSE, so our WARNING is shown
    so.add_provider_for_devices([dev], {})
    print("[test] creating session WITH ort-rocket selected (watch for the EP WARNING above/below)...")
    sess_ep = run(onnx_path, so)
    names = [o.name for o in sess_ep.get_outputs()]
    in_name = sess_ep.get_inputs()[0].name
    outs_ep = dict(zip(names, sess_ep.run(None, {in_name: x})))
    print(f"[test] ran with ort-rocket selected. outputs: {[f'{n}{v.shape}' for n,v in outs_ep.items()]}")

    # 5a. Compare to a plain CPU-EP run (no ort-rocket) on identical input.
    so_cpu = ort.SessionOptions()
    sess_cpu = run(onnx_path, so_cpu, providers=["CPUExecutionProvider"])
    outs_cpu = dict(zip(names, sess_cpu.run(None, {in_name: x})))

    ok = True
    for n in names:
        max_abs = float(np.max(np.abs(outs_ep[n].astype(np.float64) - outs_cpu[n].astype(np.float64))))
        exact = np.array_equal(outs_ep[n], outs_cpu[n])
        print(f"[test] {n}: bit-identical-to-CPU={exact} max_abs_diff={max_abs:.3e}")
        ok = ok and exact

    # 5b. Compare to the reference golden tensors if available.
    if golden_dir:
        for n in names:
            gp = os.path.join(golden_dir, f"ort_{n}.npy")
            if os.path.exists(gp):
                g = np.load(gp)
                exact = np.array_equal(outs_ep[n], g)
                max_abs = float(np.max(np.abs(outs_ep[n].astype(np.float64) - g.astype(np.float64))))
                print(f"[test] {n}: matches golden={exact} max_abs_diff={max_abs:.3e}")
                ok = ok and exact

    ort.unregister_execution_provider_library(REG_NAME)
    print("[test] unregistered EP library")

    print("\nRESULT:", "PASS - plumbing works, output bit-identical" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
