#!/usr/bin/env python3
"""The EP's spinning warning: one line per session that claims work while intra-op threads spin.

ONNX Runtime's spinning intra-op threads take the cores the EP's workers and host passes need, and
a plugin EP cannot turn spinning off, so it warns. Each case creates one session in a child process
(the warning is logged in GetCapability, before any run) and counts the line, with spinning on and
with session.intra_op.allow_spinning=0:

  * family    the family model at ORT_DISABLE_ALL: the encoder claim alone. One warning, then none.
  * nothing   the family model at ORT_ENABLE_ALL with the per-node claims off: the matcher misses
              and nothing is claimed. No warning either way.
  * per-node  (optional) a model with ConvTranspose or Conv nodes at ORT_ENABLE_ALL. One, then none.

A family model with per-node claims of its own (Depth Anything) still warns once.

Usage: python test_spin_warning_ep.py <libonnxruntime_rocket.so> <family.onnx> [per-node.onnx]
"""
import os
import subprocess
import sys

NEEDLE = "allow_spinning is on"

CHILD = r"""
import os, sys, onnxruntime as ort
lib, model, opt, spin = sys.argv[1:5]
ort.register_execution_provider_library("rocket", os.path.abspath(lib))
dev = [d for d in ort.get_ep_devices() if d.ep_name == "rocket"][0]
so = ort.SessionOptions()
so.log_severity_level = 2
so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL if opt == "disable"
                               else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
if spin == "off":
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
so.add_provider_for_devices([dev], {})
s = ort.InferenceSession(model, so)
del s
print("CHILD-OK")
"""


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    lib, family = sys.argv[1], sys.argv[2]
    cases = [("family", family, "disable", {}, 1),
             ("nothing", family, "all", {"ROCKET_ORT_CONV": "0", "ROCKET_ORT_CONVTRANSPOSE": "0"}, 0)]
    if len(sys.argv) > 3:
        cases.append(("per-node", sys.argv[3], "all", {}, 1))
    fails = 0
    for name, model, opt, env, want_on in cases:
        for spin in ("on", "off"):
            want = want_on if spin == "on" else 0
            p = subprocess.run([sys.executable, "-c", CHILD, lib, model, opt, spin],
                               capture_output=True, text=True, env={**os.environ, **env})
            out = p.stdout + p.stderr
            n = out.count(NEEDLE)
            ok = p.returncode == 0 and "CHILD-OK" in out and n == want
            fails += not ok
            print(f"[test] {name:8s} spinning {spin:3s}: {n} warning(s), want {want}  {'ok' if ok else 'FAIL'}")
            if not ok:
                for line in out.splitlines()[-6:]:
                    print("[test]   " + line)
    print("\nRESULT:", "PASS - one warning per spinning session that claims work" if fails == 0
          else f"FAIL - {fails} case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
