# ort-rocket

## AI Disclosure

With the exception of prior work this may build on, ort-rocket was developed by AI, primarily
Claude Code (Opus 4.8). Human involvement was mostly limited to setting project goals and
providing hardware access. This is a side project for curiosity's sake and comes with no
guarantee of quality, accuracy, or update frequency.

## About ort-rocket

An ONNX Runtime execution provider for Rockchip NPUs (validated on the RK3588) via the mainline
`rocket` DRM-accel driver. It offloads the encoder of transformer vision models -- ViT backbones
and their conv necks -- to the NPU through the standalone `rocket-userspace` driver library, and
falls back to ONNX Runtime's own CPU kernels for everything it does not claim (deformable-attention
sampling, detection heads, pooling heads, and other host ops).

It builds as a runtime-loadable plugin execution provider `.so` that stock ONNX Runtime loads via
`register_execution_provider_library`; the NPU then appears as a selectable device, exactly like
ONNX Runtime's other execution providers. Any ONNX Runtime application can load it and run an
unmodified `.onnx` model.

Five model families are recognized and offloaded today, identified by graph topology rather than
exporter-generated tensor names:

- **RF-DETR** (nano + base) -- the DINOv2 ViT backbone and the CSP feature projector; the
  deformable-attention decoder and detection head stay on the CPU.
- **CLIP and SigLIP** image encoders -- the plain global-attention ViT; the pooling head stays on
  the CPU.
- **SAM** (Segment Anything) ViT-Det image encoder -- windowed and global attention with on-chip
  decomposed relative-position bias, plus the conv neck; the prompt encoder and mask decoder stay
  on the CPU.
- **Depth Anything v2** -- the DINOv2 backbone; the DPT depth head stays on the CPU.

A graph that is not one of these, or whose recognizable topology has been rewritten by graph
optimization (see below), runs entirely on ONNX Runtime's CPU kernels, unchanged.

```
.
  rocket-userspace/   # the driver library (dependency): git clone https://github.com/gregordinary/rocket-userspace
  ort-rocket/         # this project: the ONNX Runtime execution provider
```

This is a separate project from the driver: `rocket-userspace` (the `librocketnpu` dependency)
builds and runs on its own; this provider links it. The full per-family faithfulness and
performance detail, the quantized-model behavior, the complete `ROCKET_ORT_*` knob table, and the
matcher / implementation notes are in [API.md](API.md).

## Performance

The value here is **throughput and CPU offload, not single-stream latency.** On the RK3588 the
matmul is DMA/dispatch-bound (not MAC-bound) and the Cortex-A76 is a strong baseline, so offloading
even an encoder that is 90-100% of a model's compute yields at most a modest single-stream speedup
-- and, for one model, a loss. The NPU earns its place by running several streams concurrently while
the A76 cores handle the host tail.

All figures warm, RK3588 @ 600 MHz, EP versus an 8-thread ONNX Runtime CPU EP on `ORT_ENABLE_ALL`;
every offloaded encoder is numerically faithful to the CPU. [HW sweep] Reproduce the single-stream
column with [`tools/bench_ep.py`](tools/bench_ep.py) and the pool column with
[`tools/bench_pool.py`](tools/bench_pool.py).

| Model | Encoder offloaded | Single-stream (xCPU) | Pool (xsingle-stream) | Faithfulness vs CPU EP |
|---|---|---:|---:|---|
| SigLIP-B/16 | plain ViT, d=768, 196 tok | 1.27x | -- | last_hidden cos 0.9999864 |
| CLIP ViT-B/16 | plain ViT, d=768, 197 tok | 1.22x | -- | last_hidden cos 0.9999955 |
| RF-DETR base | DINOv2 ViT-S, d=384, windowed | 1.05x | ~1.6x | COCO mAP 0.564 = 0.564 (200-img) |
| RF-DETR nano | DINOv2 ViT-S, d=384, windowed | 1.03x | ~1.6x | COCO mAP 0.5049 vs 0.5051 (500-img) |
| SAM ViT-B | ViT-Det, d=768, 4096 tok | 1.03-1.11x | ~1.71x | enc cos 0.9999995, mask-IoU 0.9998 |
| Depth Anything v2 Small | DINOv2 ViT-S, d=384, global 1370 tok | 0.78x | -- | depth cos 0.9999999 |

**Which models win is predictable.** The NPU runs the projection GEMMs (large-K matmuls) about 2.7x
more efficiently per-MAC than attention, whose score matmul contracts over the head dimension (64)
and materializes an `ntok x ntok` score matrix per head. So a model's single-stream envelope tracks
its *effective* attention-to-projection ratio -- `ntok / 4d`, counting windowed attention at its
window length, not the width alone:

| Model | ntok | d | effective ntok/4d | attention | single-stream |
|---|---:|---:|---:|---|---:|
| SigLIP / CLIP B/16 | ~196 | 768 | 0.06 | global, short | 1.22-1.27x |
| RF-DETR base | 1601 | 384 | ~0.07 | windowed 101 | 1.05x |
| SAM ViT-B | 4096 | 768 | ~0.10 | windowed 196 + 4 global | 1.05x |
| Depth Anything v2 Small | 1370 | 384 | 0.89 | global 1370 | 0.78x |

Short-sequence wide encoders (CLIP/SigLIP) win; windowed encoders sit at parity; a long
global-attention encoder (Depth Anything Small) is the first to lose single-stream. Widening the
encoder helps (d=768 beats d=384) by raising the projection share, but it cannot shrink an
attention cost that only sequence length controls.

**Throughput is the headline.** A single encode is partly host-bound (score readback, softmax,
layout, the CPU tail), so the NPU's value is running one process per stream (P=4 peaks the 3-core
NPU). The best host-core placement is model-dependent -- RF-DETR scales best with the host threads
left unpinned (`ROCKET_ORT_PIN=0`), SAM with a per-process `ROCKET_CPU_AFFINITY` spread across
distinct A76 cores. The NPU pool scales ~1.6x (RF-DETR) / ~1.71x (SAM) over a single NPU stream, and
frees the A76 cores a CPU-only run would saturate. The CPU pools too, though, so the NPU-vs-CPU
number measured against the CPU's *own* best pool is narrower: SAM is 1.44x there, and RF-DETR's
margin is thinner (the CPU pools well on that workload). Reproduce the vs-CPU-pool ratio with
[`tools/bench_pool.py`](tools/bench_pool.py). The encoder-only families (CLIP/SigLIP/SAM) also have a
~100% offload share, so no host tail dilutes the encoder's edge.

The remaining lever is a tiled, fused global-attention pass (an FA-2-style running softmax that
keeps the `ntok x ntok` scores off the host DMA path). It would lift both the single-stream number
and the pool ceiling for the attention-bound models (SAM's four global layers, Depth Anything's
every layer); it is the obvious next contribution.

## What runs on the NPU

The EP claims one encoder subgraph per recognized family and hands everything else back to ONNX
Runtime:

- **Offloaded:** the ViT encoder (patch-embed conv, LayerNorm, multi-head self-attention, FFN,
  GELU), reused across families; RF-DETR's CSP projector; SAM's windowed/global attention with
  on-chip decomposed relative-position bias and its conv neck; Depth Anything's four intermediate
  encoder taps.
- **Host (ONNX Runtime CPU):** deformable-attention `GridSample` sampling and the DETR detection
  head (RF-DETR); the attention-pool / cls-token pooling heads (SigLIP/CLIP); the prompt encoder and
  mask decoder (SAM); the DPT reassemble/fusion head and its `align_corners` resizes (Depth
  Anything). None of these has an NPU route -- `GridSample` and `align_corners` resize both need a
  gather the RK3588 NPU lacks -- and each is a small share of its model.

**Graph optimization must be off.** The matcher keys on the exported op topology, so the session
must run with `ORT_DISABLE_ALL`: ONNX Runtime's Gemm/Gelu/LayerNorm fusions (engaged from
`ORT_ENABLE_BASIC` up) rewrite the encoder block into a topology the matcher does not model, and the
model then runs entirely on the CPU. A graph that is not one of the five families, or whose
optimizations were left on, is left on the CPU -- never mis-claimed.

**Quantized models.** The EP consumes int8 and int4 QDQ ONNX (ONNX Runtime's `quantize_static`
format) and by default dequantizes the weights to fp16, running the same fp16 datapath -- faithful,
and the fast path on this stack. A native W8A8 int8 datapath (`ROCKET_ORT_INT8=1`) runs the encoder
GEMMs as int8 x int8 but is ~1.8x slower (mainline `rocket` has no on-chip int32 accumulation, so
every int8 matmul reads its int32 result back to the host); it is a faithfulness / experimentation
mode, not a speed lever. Detail and the accuracy numbers are in [API.md](API.md#quantized-models).

## Architecture

A plugin execution provider (an out-of-tree EP shared library).

- ONNX Runtime `dlopen`s the library and calls the exported `CreateEpFactories` / `ReleaseEpFactory`
  entry points. The factory reports the devices it supports; ONNX Runtime then asks the EP, via
  `GetCapability`, which subgraphs it can run, and hands each claimed subgraph back through
  `Compile`. Whatever the EP does not claim runs on ONNX Runtime's own kernels.
- The provider calls ONNX Runtime only through the `OrtApi` / `OrtEpApi` function tables passed to
  `CreateEpFactories`, so the `.so` does not link `libonnxruntime`; it needs only the ONNX Runtime
  headers to build.
- It links `librocketnpu` (the `rocket-userspace` project) for the on-NPU operators.
- A matcher registry tries each family signature in turn during `GetCapability` and claims the first
  that matches, so the offloaded subgraph is identified by graph structure -- the attention/MLP
  pattern, LayerScale, the patch-embed and neck convs, and family-distinctive nodes like SAM's
  relative-position Einsums or Depth Anything's four-tap DPT exit -- not by tensor names. A re-export
  that renames the interior still matches.

## Requirements

- An RK3588 board on a mainline kernel carrying the `rocket` DRM-accel driver, with
  `/dev/accel/accel0` present (`lsmod | grep rocket`).
- The sibling `rocket-userspace` driver library, cloned next to this repo or installed as a
  `rocketnpu` package.
- An ONNX Runtime build exposing the plugin-EP API (`register_execution_provider_library`).
- Privilege to open the accel node -- run with `sudo -E` (the `-E` preserves the `ROCKET_*` env
  knobs plain `sudo` strips).

**Clock.** The NPU boots at 200 MHz and the EP is correct there, but every figure above is at
600 MHz: apply the `patches/rocket` clock patch and load the module with
`rocket_npu_clk_hz=600000000`.

## Build

Requires a C++17 compiler and CMake. The ONNX Runtime C/C++ API headers for the target ONNX Runtime
version are vendored under `third_party/onnxruntime/include`; override with `-DORT_HEADER_DIR=<path>`
to build against a different release.

```sh
cmake -S . -B build
cmake --build build -j
# -> build/libonnxruntime_rocket.so
```

## Use from an application

```python
import onnxruntime as ort

ort.register_execution_provider_library("rocket", "/path/to/libonnxruntime_rocket.so")
dev = next(d for d in ort.get_ep_devices() if d.ep_name == "rocket")

so = ort.SessionOptions()
# The topology matcher keys on the raw exported ops, so disable ONNX Runtime's graph
# optimizations: its Gemm/SkipLayerNorm/Gelu fusions rewrite the encoder into a topology
# the matcher does not recognize (even ORT_ENABLE_BASIC fuses every MatMul+Add into Gemm).
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so.add_provider_for_devices([dev], {})
sess = ort.InferenceSession("model.onnx", so)
outputs = sess.run(None, {"input": x})
```

The provider loads for any ONNX Runtime application and runs an unmodified `.onnx` model; the NPU
offload activates for a recognized encoder and any other graph runs on the CPU. For a CLIP / SigLIP
/ Depth-Anything variant whose attention head dimension is not 64, set `ROCKET_ORT_SIGLIP_HEADS` to
the head count (the EP reports the required value if it cannot infer it). For multi-stream
throughput, run one process per stream; the best host-core placement is model-dependent (RF-DETR
unpinned via `ROCKET_ORT_PIN=0`, SAM a per-process `ROCKET_CPU_AFFINITY` spread) -- see
[API.md](API.md#runtime-knobs) for the per-model recipe and the full knob table.

## The rocket NPU stack

This provider is one frontend of an open source stack for Rockchip NPUs -- three userspace projects
plus a set of optional kernel patches:

- **[`rocket-userspace`](https://github.com/gregordinary/rocket-userspace)** (`librocketnpu`) -- the
  userspace driver, matmul, and on-NPU op library. The dependency; usable on its own.
- **ort-rocket** (this project) -- an ONNX Runtime execution provider for transformer vision models.
- **[`ggml-rocket`](https://github.com/gregordinary/ggml-rocket)** -- a ggml backend `.so` for
  `llama.cpp` / `whisper.cpp`, linking the same driver.
- **[`tflite-rocket`](https://github.com/gregordinary/tflite-rocket)** -- a TFLite external delegate
  for detection models, linking the same driver.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope) -- optional out-of-tree
  kernel-module patches (clock / voltage / IOMMU). They raise the NPU clock from its 200 MHz boot
  default to 600 MHz; the figures here assume them.

## License & credits

`ort-rocket` is GPL-3.0-or-later (it links the GPL-3 `rocket-userspace` driver library, whose NPU
register headers are the GPL-3 reverse-engineering by Jasbir Matharu). It targets
[ONNX Runtime](https://onnxruntime.ai) (MIT) as an out-of-tree plugin execution provider; the ONNX
Runtime C/C++ API headers vendored under `third_party/onnxruntime/` are MIT-licensed (Copyright
Microsoft Corporation). It builds on the `rocket-userspace` driver for the NPU path.
