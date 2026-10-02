#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The ort-rocket authors
"""Warm single-stream latency: the ort-rocket EP vs the ONNX Runtime CPU EP.

Reproduces the single-stream column of the README Performance table for any recognized family
(RF-DETR, CLIP / SigLIP, SAM, Depth Anything v2). The CPU baseline runs ORT_ENABLE_ALL on all
threads -- the config a real CPU-only user runs, which is the honest comparison: the EP needs
ORT_DISABLE_ALL for its topology matcher, and a CPU baseline also on DISABLE_ALL would be denied
its own LayerNorm/Gelu/MatMul fusions and flatter the NPU. A cold run is discarded; the median of
N warm runs is reported, with the speedup cpu/ep.

The input .npy is an NCHW tensor of the shape the model expects (e.g. [1,3,224,224] for SigLIP-B/16,
[1,3,1024,1024] for SAM-ViT-B). The per-family tests under tests/ save such reference inputs. A
multi-input model (the Laya decision model: input_ids, attention_mask, marker_pos, marker_mask,
qtype) takes an .npz instead, whose arrays are named for the inputs, bare or with an `in_` prefix.

Run with sudo -E so the /dev/accel privilege and the ROCKET_* env both survive:
  sudo -E .../python tools/bench_ep.py <lib.so> <model.onnx> <input.npy|.npz> [--iters N] [--threads T]
"""
import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort

REG = "rocket"


def warm_median(sess, feeds, iters):
    """Median / min / p90 wall time (ms) over `iters` warm runs; one cold run discarded first."""
    sess.run(None, feeds)  # cold, discarded
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[int(len(ts) * 0.9)]


def main():
    ap = argparse.ArgumentParser(description="ort-rocket EP vs CPU-EP single-stream latency")
    ap.add_argument("lib", help="path to libonnxruntime_rocket.so")
    ap.add_argument("onnx", help="path to the .onnx model")
    ap.add_argument("npy", help="an NCHW input tensor (.npy), or named inputs (.npz)")
    ap.add_argument("--iters", type=int, default=20, help="warm iterations (default 20)")
    ap.add_argument("--threads", type=int, default=8, help="intra-op threads (default 8)")
    args = ap.parse_args()

    # CPU EP, graph opt ON, all threads -- the honest baseline a CPU-only user runs.
    so_cpu = ort.SessionOptions()
    so_cpu.intra_op_num_threads = args.threads
    so_cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu = ort.InferenceSession(args.onnx, so_cpu, providers=["CPUExecutionProvider"])
    if args.npy.endswith(".npz"):
        z = np.load(args.npy)
        feeds = {}
        for i in cpu.get_inputs():
            key = i.name if i.name in z.files else "in_" + i.name
            feeds[i.name] = z[key]
    else:
        feeds = {cpu.get_inputs()[0].name: np.load(args.npy)}
    cpu_med, cpu_min, cpu_p90 = warm_median(cpu, feeds, args.iters)

    # ort-rocket EP, graph opt OFF (the matcher keys on the raw exported op topology).
    ort.register_execution_provider_library(REG, os.path.abspath(args.lib))
    dev = [d for d in ort.get_ep_devices() if d.ep_name == REG][0]
    so_ep = ort.SessionOptions()
    so_ep.intra_op_num_threads = args.threads
    so_ep.log_severity_level = 3
    so_ep.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so_ep.add_provider_for_devices([dev], {})
    ep = ort.InferenceSession(args.onnx, so_ep)
    ep_med, ep_min, ep_p90 = warm_median(ep, feeds, args.iters)

    print(f"iters={args.iters} threads={args.threads}  (warm, cold discarded)")
    print(f"  cpu-EP ENABLE_ALL    median={cpu_med:8.2f} ms  min={cpu_min:7.2f}  p90={cpu_p90:7.2f}")
    print(f"  rocket-EP (NPU enc)  median={ep_med:8.2f} ms  min={ep_min:7.2f}  p90={ep_p90:7.2f}")
    print(f"\n  single-stream speedup (cpu/ep) = {cpu_med / ep_med:.2f}x")
    # The EP/driver teardown races at interpreter shutdown (a segfault after all compute has
    # completed -- benign). Flush the results and hard-exit so the numbers always print and the
    # process exit code stays clean.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
