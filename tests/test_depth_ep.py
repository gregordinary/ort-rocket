#!/usr/bin/env python3
"""On-device faithfulness gate for the ort-rocket Depth Anything v2 backbone offload.

Runs the full model twice in one process -- once on the ORT CPU EP, once with ort-rocket offloading
the DINOv2 encoder and its four DPT taps to the NPU -- and grades the depth map. The DPT head runs
on ORT's CPU kernels in both, so this measures exactly the offloaded part end to end. Same box, same
session graph, so the CPU EP is the reference (no cross-arch golden question).

Two gates:
  * predicted_depth cosine >= 0.9999 vs the CPU EP, plus AbsRel / delta-1 agreement as the
    task-level read (a cosine can look fine while a scale error ruins a depth map),
  * with --taps, per-tap cosine >= 0.999 on each of the four encoder taps the head consumes. That
    needs the tap tensors promoted to graph outputs, so the run builds a tap-augmented copy of the
    model once and caches it beside the original.

  sudo -E .../python test_depth_ep.py <lib.so> <depth_anything_v2_small_static.onnx> <input_nchw.npy> [--taps]
"""
import os, sys, gc, time
import numpy as np
import onnxruntime as ort

DEPTH_COS = 0.9999   # exit gate: depth-map cosine vs the CPU EP
TAP_COS = 0.999      # exit gate: per-tap encoder cosine vs the CPU EP
NRUNS = 5            # timed warm runs per session (median reported)


def governor():
    """The big cores' cpufreq governor. It belongs in any envelope this prints: under `ondemand` the
    ramp penalizes the CPU-bound host tail and understates the NPU path (measured on SAM)."""
    try:
        with open("/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def find_tap_tensors(onnx_path):
    """The DPT taps: a LayerNormalization whose only consumer is the cls-stripping Slice. The 24
    encoder norms all feed MatMuls, so this picks out exactly the head's taps."""
    import onnx
    g = onnx.load(onnx_path, load_external_data=False).graph
    cons = {}
    for n in g.node:
        for x in n.input:
            if x:
                cons.setdefault(x, []).append(n)
    taps = []
    for n in g.node:
        if n.op_type != "LayerNormalization":
            continue
        c = cons.get(n.output[0], [])
        if len(c) == 1 and c[0].op_type == "Slice":
            taps.append(n.output[0])
    return taps


def build_tap_model(onnx_path, taps):
    """A copy of the model with the tap tensors promoted to graph outputs, so both EPs can be graded
    on them. Cached: the export is ~100 MB and unchanged between runs."""
    import onnx
    from onnx import helper, TensorProto
    out_path = onnx_path.replace(".onnx", "_taps.onnx")
    if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(onnx_path):
        return out_path
    model = onnx.load(onnx_path)
    have = {o.name for o in model.graph.output}
    for t in taps:
        if t not in have:
            model.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    onnx.save(model, out_path)
    print(f"[test] wrote tap-augmented model {out_path}")
    return out_path


def rocket_session(onnx_path, ep_lib):
    so = ort.SessionOptions()
    so.log_severity_level = 1   # INFO -> surface the EP claim line
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "rocket"]
    assert devs, "rocket EP not registered"
    so.add_provider_for_devices([devs[0]], {})
    return ort.InferenceSession(onnx_path, so)


def cos_of(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def depth_metrics(ep, cpu):
    """Task-level agreement between the two depth maps. Depth Anything v2 predicts INVERSE relative
    depth, so compare on the raw prediction and ignore non-positive reference pixels."""
    a = np.asarray(ep, dtype=np.float64).ravel()
    b = np.asarray(cpu, dtype=np.float64).ravel()
    m = b > 1e-6
    if not m.any():
        return float("nan"), float("nan")
    absrel = float(np.mean(np.abs(a[m] - b[m]) / b[m]))
    ratio = np.maximum(a[m] / b[m], b[m] / np.maximum(a[m], 1e-12))
    d1 = float(np.mean(ratio < 1.25))
    return absrel, d1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_taps = "--taps" in sys.argv
    ep_lib = os.path.abspath(args[0])
    onnx_path = os.path.abspath(args[1])
    x = np.load(args[2])
    print(f"[test] input {x.shape} {x.dtype}")

    taps = find_tap_tensors(onnx_path) if want_taps else []
    if want_taps:
        print(f"[test] DPT taps found: {len(taps)}")
        assert len(taps) == 4, f"expected 4 DPT taps, found {len(taps)}"
        onnx_path = build_tap_model(onnx_path, taps)
    want = ["predicted_depth"] + taps

    print(f"[test] cpu governor={governor()}  warm runs={NRUNS} (first discarded: the NPU clock "
          f"parks at idle, so a cold run reads low)")

    def timed(sess):
        sess.run(want, {in_name: x})                   # cold: discarded
        ts, out = [], None
        for _ in range(NRUNS):
            t0 = time.time()
            out = sess.run(want, {in_name: x})
            ts.append(time.time() - t0)
        return out, float(np.median(ts)), ts

    scpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = scpu.get_inputs()[0].name
    cpu, t_cpu, ts_cpu = timed(scpu)
    del scpu; gc.collect()

    ort.register_execution_provider_library("rocket", ep_lib)
    sep = rocket_session(onnx_path, ep_lib)
    got, t_ep, ts_ep = timed(sep)                      # Compile already packed the weights
    del sep; gc.collect()                              # session BEFORE unregister, or teardown segfaults
    ort.unregister_execution_provider_library("rocket")

    ok = True
    dcos = cos_of(got[0], cpu[0])
    absrel, d1 = depth_metrics(got[0], cpu[0])
    maxabs = float(np.max(np.abs(got[0].astype(np.float64) - cpu[0].astype(np.float64))))
    rng = float(np.ptp(cpu[0].astype(np.float64)))
    print(f"[test] predicted_depth {got[0].shape}: cos(ep,cpu)={dcos:.7f} "
          f"max_abs={maxabs:.3e} (ref range {rng:.3f})")
    print(f"[test] task agreement: AbsRel={absrel:.5f} delta-1={d1:.5f}")
    if dcos < DEPTH_COS:
        ok = False

    for i, t in enumerate(taps):
        c = cos_of(got[1 + i], cpu[1 + i])
        print(f"[test] tap {i} {t}: cos(ep,cpu)={c:.7f}")
        if c < TAP_COS:
            ok = False

    print(f"[test] warm e2e (median of {NRUNS}): cpu-EP {t_cpu*1e3:.0f} ms  rocket-EP {t_ep*1e3:.0f} ms "
          f"({t_cpu/t_ep:.2f}x)")
    print(f"[test]   cpu-EP runs    {[round(t*1e3) for t in ts_cpu]} ms")
    print(f"[test]   rocket-EP runs {[round(t*1e3) for t in ts_ep]} ms")
    print("RESULT:", "PASS" if ok else "FAIL",
          f"(gates depth cos>={DEPTH_COS}" + (f", tap cos>={TAP_COS})" if taps else ")"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
