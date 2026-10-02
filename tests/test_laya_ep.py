#!/usr/bin/env python3
"""Faithfulness gate for the ort-rocket ModernBERT offload, on the Laya decision model.

Replays recorded Laya requests: each case .npz holds the exact session inputs Laya's own
ONNXAgent built (input_ids, attention_mask, marker_pos, marker_mask, qtype) and, as the
reference, the CPU EP's logits and act_logits for them (at ORT_DISABLE_ALL). The EP arm runs the
same graph with ort-rocket offloading the encoder and the decision head's transformer layers.

Graded per case: the selected answer (argmax over each question's live options) must agree, and
the option probabilities must stay within PROB_TOL of the reference. A matcher miss would run the
whole model on the CPU kernels and pass the numbers, so the gate also asserts the EP executed a
node and that the logits are not bit-identical to the reference.

  python3 test_laya_ep.py <lib.so> <laya.onnx> <case.npz> [case.npz ...] [--cpu] [--bench N]

--cpu recomputes the reference on this box's CPU EP (at ORT_ENABLE_ALL) instead of reading the
recorded one (ORT_DISABLE_ALL).
--bench N also reports the warm median of N EP runs (and of N CPU runs with --cpu).
"""
import os, sys, time
import numpy as np
import onnxruntime as ort

from ep_common import EpSession, check_not_identical, check_placement

PROB_TOL = 5e-3   # max |p_ep - p_ref| over every live option
FEEDS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def softmax_live(logits, mask):
    z = np.where(mask, logits.astype(np.float64), -np.inf)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def median_ms(fn, n):
    fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_cpu = "--cpu" in sys.argv
    bench = int(sys.argv[sys.argv.index("--bench") + 1]) if "--bench" in sys.argv else 0
    if bench:
        args = [a for a in args if a != str(bench)]
    ep_lib, onnx_path, cases = os.path.abspath(args[0]), os.path.abspath(args[1]), args[2:]
    zs = [(os.path.basename(c)[:-4], np.load(c)) for c in cases]
    feeds = [{k: z["in_" + k] for k in FEEDS} for _, z in zs]

    refs, cpu_ms = [], []
    if use_cpu:
        so = ort.SessionOptions()   # the CPU-only user's configuration, and the timing baseline
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        scpu = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        for f in feeds:
            refs.append(scpu.run(["logits", "act_logits"], f))
            cpu_ms.append(median_ms(lambda: scpu.run(["logits", "act_logits"], f), bench) if bench else None)
        del scpu
    else:
        refs = [[z["out_logits"], z["out_act_logits"]] for _, z in zs]

    outs, ep_ms = [], []
    with EpSession(ep_lib, onnx_path, "rocket", log_level=2) as sep:
        for f in feeds:
            outs.append(sep.run(["logits", "act_logits"], f))
            ep_ms.append(median_ms(lambda: sep.run(["logits", "act_logits"], f), bench) if bench else None)

    ok = check_placement(sep.placement)
    ok = check_not_identical([(n + ":logits", o[0], r[0]) for (n, _), o, r in zip(zs, outs, refs)]) and ok
    for i, ((name, z), o, r) in enumerate(zip(zs, outs, refs)):
        mm = z["in_marker_mask"]
        B, T = z["in_input_ids"].shape
        pe, pr = softmax_live(o[0], mm), softmax_live(r[0], mm)
        dp = float(np.max(np.abs(np.where(mm, pe - pr, 0))))
        dl = float(np.max(np.abs(np.where(mm, o[0] - r[0], 0))))
        da = float(np.max(np.abs(o[1] - r[1])))
        agree = bool(np.all(np.argmax(np.where(mm, o[0], -np.inf), -1) ==
                            np.argmax(np.where(mm, r[0], -np.inf), -1)))
        good = agree and dp <= PROB_TOL
        ok = ok and good
        t = ""
        if bench:
            t = f"  ep {ep_ms[i]:.1f} ms" + (f", cpu {cpu_ms[i]:.1f} ms ({cpu_ms[i] / ep_ms[i]:.2f}x)" if use_cpu else "")
        print(f"[test] {name:24s} B={B} T={T:4d}  answers {'agree' if agree else 'DIFFER'}  "
              f"max|dp| {dp:.2e}  max|dlogit| {dl:.2e}  max|dact| {da:.2e}  {'ok' if good else 'FAIL'}{t}")
    print("RESULT:", "PASS" if ok else "FAIL", f"(answers agree, max|dp| <= {PROB_TOL})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
