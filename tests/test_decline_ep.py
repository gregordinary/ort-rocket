#!/usr/bin/env python3
"""Gate: a model the EP recognizes and cannot run still creates its session, on the CPU EP.

The EP builds a family claim's whole state in GetCapability, before it claims: the weights, the NPU
device and the resident context. A refusal there leaves the encoder on the CPU EP with a warning,
where a refusal in Compile would fail session creation. This runs one model under the environment
named on the command line, a case the EP must refuse, and checks that:

  * the session is created and runs,
  * no node runs on the EP (the per-node claims are off unless the command line turns them on), and
  * every output is bit-identical to a CPU EP session at the same optimization level.

  sudo -E .../python test_decline_ep.py <lib.so> <model.onnx> <input.npy> KEY=VALUE [KEY=VALUE ...]

Cases this covers:

  ROCKET_DEV=/dev/accel/none        any family: the NPU does not open
  ROCKET_CHIP=rk3576                any family: the part is not an RK3588
  ROCKET_ORT_RESIDENT=0             CLIP or Depth Anything: the one-shot encoder does not run them
  (no variable)                     a QDQ model of a family other than RF-DETR

ROCKET_CHIP is read once per process, so run each case in its own process.
"""
import os, sys
import numpy as np
import onnxruntime as ort

from ep_common import CPU_EP, EpSession


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    ep_lib = os.path.abspath(sys.argv[1])
    onnx_path = os.path.abspath(sys.argv[2])
    x = np.load(sys.argv[3])
    env = dict(a.split("=", 1) for a in sys.argv[4:])
    env.setdefault("ROCKET_ORT_CONV", "0")
    env.setdefault("ROCKET_ORT_CONVTRANSPOSE", "0")
    os.environ.update(env)
    print(f"[test] {os.path.basename(onnx_path)}  input {x.shape} {x.dtype}  env "
          + " ".join(f"{k}={v}" for k, v in sorted(env.items())))

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    scpu = ort.InferenceSession(onnx_path, so, providers=[CPU_EP])
    feeds = {scpu.get_inputs()[0].name: x}
    names = [o.name for o in scpu.get_outputs()]
    cpu = scpu.run(names, feeds)

    try:
        with EpSession(ep_lib, onnx_path, "rocket", log_level=2) as sep:
            ep = sep.run(names, feeds)
    except Exception as e:   # session creation or the run failed: the refusal did not decline
        print(f"[test] the EP session failed: {e}")
        print("RESULT: FAIL")
        sys.exit(1)

    on_ep = sum(len(n) for p, n in sep.placement.items() if p != CPU_EP)
    ok = on_ep == 0
    print(f"[test] placement: {on_ep} node(s) on the EP  {'ok' if ok else 'FAIL'}")
    for name, e, c in zip(names, ep, cpu):
        same = np.asarray(e).shape == np.asarray(c).shape and np.array_equal(e, c)
        ok = ok and same
        print(f"[test] {name}: bit-identical to the CPU EP  {'ok' if same else 'FAIL'}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
