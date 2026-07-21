#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The ort-rocket authors
"""Process-per-stream throughput: the ort-rocket EP pool vs the CPU-EP pool.

A single encode is partly host-bound (score readback, softmax, layout, the CPU tail), so the NPU
sits idle during that host work and aggregate throughput scales by running several independent
streams -- one process per stream. This reproduces the pool column of the README Performance table.

The honest headline is best-over-best: the EP pool's best aggregate rate against the CPU EP's *own*
best pooled rate. An 8-thread ORT_ENABLE_ALL CPU already saturates all cores at single stream, but a
CPU pool with fewer threads per process can still edge it up -- that is the denominator, not the
single-thread CPU rate. The shared 3-core NPU caps the EP pool at ~2x its own single stream, so P=4
is typically the peak (P>=5 oversubscribes the four A76 cores and drops).

The best EP host-core placement is model-dependent (--pin-mode):
  * unpinned  (ROCKET_ORT_PIN=0)        -- the OS spreads host threads over all 8 cores.  RF-DETR.
  * affinity  (per-process ROCKET_CPU_AFFINITY=<distinct A76>) -- SAM's pthread attention workers.
  * default   -- the default A76 pin (single-stream-latency optimized; a pool anti-pattern).

Run with sudo -E so the /dev/accel privilege and the ROCKET_* env both survive:
  sudo -E .../python tools/bench_pool.py <lib.so> <model.onnx> <input.npy> \
      [--procs 1 2 3 4] [--iters N] [--pin-mode unpinned|affinity|default] \
      [--cpu-configs 1x8 2x4 4x2] [--no-cpu]
"""
import argparse
import glob
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import onnxruntime as ort

REG = "rocket"


def big_cores():
    """A76 (big) CPU ids by cpuinfo_max_freq. The sysfs path holds '/cpu' twice (.../cpu/cpuN/...),
    so the id is parsed with a regex -- a plain split('/cpu') yields '' and silently drops every
    core, falling back to 'all cores are big' (which would pin pool workers onto the A55 littles)."""
    freq = {}
    for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
        m = re.search(r"/cpu(\d+)/cpufreq/cpuinfo_max_freq$", p)
        if not m:
            continue
        try:
            with open(p) as fh:
                freq[int(m.group(1))] = int(fh.read())
        except (OSError, ValueError):
            pass
    if not freq:
        return list(range(os.cpu_count() or 8))
    mx = max(freq.values())
    return sorted(c for c, f in freq.items() if f == mx)


def ep_worker(lib, onnx, npy, iters, barrier, q, idx, pin_mode):
    # Place the stream before the session is created: Compile builds the resident ctx and its first
    # pin latches the affinity, so the env must be set here (this is a fresh spawned process).
    if pin_mode == "unpinned":
        os.environ["ROCKET_ORT_PIN"] = "0"
    elif pin_mode == "affinity":
        big = big_cores()
        if big:
            os.environ["ROCKET_CPU_AFFINITY"] = str(big[idx % len(big)])
    x = np.load(npy)
    ort.register_execution_provider_library(REG, os.path.abspath(lib))
    dev = [d for d in ort.get_ep_devices() if d.ep_name == REG][0]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_provider_for_devices([dev], {})
    sess = ort.InferenceSession(onnx, so)
    feeds = {sess.get_inputs()[0].name: x}
    sess.run(None, feeds)  # cold, discarded
    barrier.wait()
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feeds)
    q.put((time.perf_counter() - t0, iters))
    q.close()
    q.join_thread()  # ensure the result is flushed to the pipe before we hard-exit
    os._exit(0)      # skip the EP/driver teardown race at interpreter shutdown (benign)


def cpu_worker(onnx, npy, threads, iters, barrier, q):
    x = np.load(npy)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
    feeds = {sess.get_inputs()[0].name: x}
    sess.run(None, feeds)  # cold, discarded
    barrier.wait()
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feeds)
    q.put((time.perf_counter() - t0, iters))
    q.close()
    q.join_thread()  # ensure the result is flushed to the pipe before we hard-exit
    os._exit(0)      # skip the EP/driver teardown race at interpreter shutdown (benign)


def run_ep_pool(lib, onnx, npy, procs, iters, pin_mode):
    """Aggregate throughput = total inferences / max worker wall (all start together at a barrier)."""
    barrier = mp.Barrier(procs)
    q = mp.Queue()
    ps = [mp.Process(target=ep_worker, args=(lib, onnx, npy, iters, barrier, q, i, pin_mode))
          for i in range(procs)]
    for p in ps:
        p.start()
    dur = [q.get() for _ in range(procs)]
    for p in ps:
        p.join()
    return sum(n for _, n in dur) / max(d for d, _ in dur)


def run_cpu_pool(onnx, npy, procs, threads, iters):
    barrier = mp.Barrier(procs)
    q = mp.Queue()
    ps = [mp.Process(target=cpu_worker, args=(onnx, npy, threads, iters, barrier, q))
          for _ in range(procs)]
    for p in ps:
        p.start()
    dur = [q.get() for _ in range(procs)]
    for p in ps:
        p.join()
    return sum(n for _, n in dur) / max(d for d, _ in dur)


def main():
    ap = argparse.ArgumentParser(description="ort-rocket EP vs CPU-EP process-pool throughput")
    ap.add_argument("lib", help="path to libonnxruntime_rocket.so")
    ap.add_argument("onnx", help="path to the .onnx model")
    ap.add_argument("npy", help="path to an NCHW input tensor (.npy)")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="EP pool sizes to sweep (default 1 2 3 4)")
    ap.add_argument("--iters", type=int, default=10, help="warm iterations per process (default 10)")
    ap.add_argument("--pin-mode", choices=["default", "unpinned", "affinity"], default="unpinned",
                    help="EP host-core placement (default unpinned = RF-DETR recipe; SAM wants affinity)")
    ap.add_argument("--cpu-configs", nargs="+", default=["1x8", "2x4", "4x2"],
                    help="CPU-EP baseline PxT configs, procs x intra_op_threads (default 1x8 2x4 4x2)")
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU-EP baseline (EP scaling only)")
    args = ap.parse_args()
    mp.set_start_method("spawn")

    print(f"iters/proc={args.iters}  EP pin-mode={args.pin_mode}")
    ep_best = 0.0
    for p in args.procs:
        thr = run_ep_pool(args.lib, args.onnx, args.npy, p, args.iters, args.pin_mode)
        ep_best = max(ep_best, thr)
        print(f"  EP   procs={p}  aggregate={thr:7.4f} img/s")

    cpu_best = 0.0
    if not args.no_cpu:
        for cfg in args.cpu_configs:
            p, t = (int(v) for v in cfg.lower().split("x"))
            thr = run_cpu_pool(args.onnx, args.npy, p, t, args.iters)
            cpu_best = max(cpu_best, thr)
            print(f"  CPU  procs={p} threads={t}  aggregate={thr:7.4f} img/s")

    print(f"\n  EP best-pooled  = {ep_best:.4f} img/s")
    if cpu_best:
        print(f"  CPU best-pooled = {cpu_best:.4f} img/s")
        print(f"  best-over-best throughput = {ep_best / cpu_best:.2f}x")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
