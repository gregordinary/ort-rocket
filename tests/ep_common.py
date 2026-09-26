# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared checks for the ort-rocket EP tests: did the EP take the model, and did it compute?

A matcher miss is not an error. The EP claims nothing, ONNX Runtime runs the whole model on its
CPU kernels, and the outputs equal the CPU-EP reference at cosine 1.0. So a test that grades only
the numbers passes on a model the NPU never touched. Two checks close that:

  * placement: the session runs with ONNX Runtime's profiler on, and the profile names the
    provider of every node that executed. At least one must be the EP's.
  * identity: fp16 on the NPU never reproduces fp32 on the CPU bit for bit, so an encoder-fed
    output that equals the reference exactly was computed by the CPU kernels.
"""
import gc
import glob
import json
import os
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import onnxruntime as ort

CPU_EP = "CPUExecutionProvider"


class EpSession:
    """An ort-rocket session with the profiler on. Use as a context manager: the session is
    released BEFORE the library is unregistered, or teardown touches freed factory state."""

    def __init__(self, ep_lib, onnx_path, reg_name="rocket", log_level=2):
        self.reg_name = reg_name
        self.profile_dir = tempfile.mkdtemp(prefix="ort_rocket_prof_")
        ort.register_execution_provider_library(reg_name, os.path.abspath(ep_lib))
        devs = [d for d in ort.get_ep_devices() if d.ep_name == reg_name]
        assert devs, f"EP '{reg_name}' not registered as a device"
        so = ort.SessionOptions()
        so.log_severity_level = log_level
        # The matcher keys on the raw exported topology, which ONNX Runtime's fusions rewrite.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(self.profile_dir, "ep")
        so.add_provider_for_devices([devs[0]], {})
        self.sess = ort.InferenceSession(onnx_path, so)
        self.placement = None

    def __enter__(self):
        return self

    def run(self, output_names, feeds):
        return self.sess.run(output_names, feeds)

    def input_name(self):
        return self.sess.get_inputs()[0].name

    def output_names(self):
        return [o.name for o in self.sess.get_outputs()]

    def close(self):
        """Read the profile, release the session, unregister the library. Returns the placement:
        provider name -> the set of node names that provider executed."""
        if self.sess is None:
            return self.placement
        path = self.sess.end_profiling()
        self.placement = read_placement(path)
        self.sess = None
        gc.collect()
        ort.unregister_execution_provider_library(self.reg_name)
        shutil.rmtree(self.profile_dir, ignore_errors=True)
        return self.placement

    def __exit__(self, *exc):
        self.close()
        return False


def read_placement(profile_path):
    """provider -> set of node names, from an ONNX Runtime profile's Node events."""
    if not profile_path or not os.path.exists(profile_path):
        cands = glob.glob(os.path.join(os.path.dirname(profile_path or "."), "*.json"))
        profile_path = cands[0] if cands else None
    placement = defaultdict(set)
    if not profile_path:
        return placement
    with open(profile_path) as f:
        events = json.load(f)
    for e in events:
        if e.get("cat") != "Node":
            continue
        prov = e.get("args", {}).get("provider")
        name = e.get("args", {}).get("node_name") or e.get("name", "")
        if prov:
            placement[prov].add(name)
    return placement


def check_placement(placement):
    """At least one executed node belongs to a provider other than ONNX Runtime's CPU one. The
    session holds only the EP and the CPU fallback, so any other provider is the EP."""
    counts = {p: len(n) for p, n in sorted(placement.items())}
    on_ep = sum(n for p, n in counts.items() if p != CPU_EP)
    ok = on_ep > 0
    print(f"[test] placement: {counts} -> {on_ep} node(s) on the EP  {'ok' if ok else 'FAIL'}")
    if not ok:
        print("[test]   the EP claimed nothing, so the CPU kernels computed every output")
    return ok


def check_not_identical(pairs):
    """pairs: [(name, ep_output, cpu_output)], each an output the offloaded encoder feeds. Fails
    when any is bit-identical to the CPU EP's, which fp16 on the NPU cannot produce against fp32
    on the CPU: max_abs == 0 means the CPU kernels computed it."""
    identical = []
    for name, e, c in pairs:
        e = np.asarray(e); c = np.asarray(c)
        if e.shape == c.shape and np.array_equal(e, c):
            identical.append(name)
    ok = not identical
    tag = "ok" if ok else "FAIL"
    print(f"[test] bit-identical to the CPU EP: {len(identical)} of {len(pairs)} output(s)"
          f"{' (' + ', '.join(identical) + ')' if identical else ''}  {tag}")
    return ok
