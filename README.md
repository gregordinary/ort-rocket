# ort-rocket

## AI disclosure

Except for the prior work it builds on, ort-rocket was developed by AI, primarily Claude. Human
involvement was mostly limited to setting project goals and providing hardware access. This is a
side project for curiosity's sake and comes with no guarantee of quality, accuracy, or update
frequency.

## About ort-rocket

An ONNX Runtime execution provider for Rockchip NPUs (validated on the RK3588) via the mainline
`rocket` DRM-accel driver. It offloads transformer encoders to the NPU through the standalone
`rocket-userspace` driver library: ViT backbones and their conv necks, and ModernBERT text
encoders. In any model it also offloads the `ConvTranspose` layers, one node at a time. Everything
it does not claim falls back to ONNX Runtime's own CPU kernels: deformable-attention sampling,
detection heads, pooling heads and other host ops.

It builds as a runtime-loadable plugin execution provider `.so`, which stock ONNX Runtime loads
via `register_execution_provider_library`. The NPU then appears as a selectable device, exactly
like ONNX Runtime's other execution providers. Any ONNX Runtime application can load it and run an
unmodified `.onnx` model.

Six model families are recognized and offloaded today, identified by graph topology rather than
exporter-generated tensor names:

- **RF-DETR** (nano + base): the DINOv2 ViT backbone and the CSP feature projector. The
  deformable-attention decoder and detection head stay on the CPU.
- **CLIP and SigLIP** image encoders: the plain global-attention ViT. The pooling head stays on
  the CPU.
- **SAM** (Segment Anything) ViT-Det image encoder: windowed and global attention with on-chip
  decomposed relative-position bias, plus the conv neck. The prompt encoder and mask decoder
  stay on the CPU.
- **Depth Anything v2**: the DINOv2 backbone. The DPT depth head stays on the CPU.
- **ModernBERT** text encoders, with an optional stack of torch pre-norm transformer layers
  after them: the encoder and those layers, over a batch of variable-length sequences. Validated
  on the Laya decision model, whose marker scoring and action heads stay on the CPU.

Separately from the families, the EP claims each `ConvTranspose` node it can run, in any graph and
at any graph-optimization level. It takes a 2-D, `group` 1, fp32 node with a constant weight. Generators and
decoders whose upsampling is a learned transposed convolution gain the most, such as pix2pix, where
it is two thirds of the CPU run. Depthwise and grouped transposes stay on the CPU.

A graph outside that set runs on ONNX Runtime's CPU kernels, except for its `ConvTranspose` nodes.
So does one whose recognizable encoder topology has been rewritten by graph optimization, described
below.

```
.
  rocket-userspace/   # the driver library (dependency): git clone https://github.com/gregordinary/rocket-userspace
  ort-rocket/         # this project: the ONNX Runtime execution provider
```

This is a separate project from the driver. `rocket-userspace`, the `librocketnpu` dependency,
builds and runs on its own, and this provider links it. The full per-family faithfulness and
performance detail, the quantized-model behavior, the complete `ROCKET_ORT_*` knob table, and the
matcher / implementation notes are in [API.md](API.md).

## Performance

The value here is **throughput and CPU offload rather than single-stream latency.** On the
RK3588 the matmul is DMA/dispatch-bound rather than MAC-bound, and the Cortex-A76 is a strong
baseline. So offloading even an encoder that is 90-100% of a model's compute yields at most a
modest single-stream speedup, and for one model a loss. The NPU earns its place by running
several streams concurrently while the A76 cores handle the host tail.

All figures warm, RK3588 @ 600 MHz, EP versus an 8-thread ONNX Runtime CPU EP on
`ORT_ENABLE_ALL`. Every offloaded encoder is numerically faithful to the CPU. [HW sweep]
Reproduce the single-stream
column with [`tools/bench_ep.py`](tools/bench_ep.py) and the pool column with
[`tools/bench_pool.py`](tools/bench_pool.py).

| Model | Encoder offloaded | Single-stream (xCPU) | Pool (xsingle-stream) | Faithfulness vs CPU EP |
|---|---|---:|---:|---|
| SigLIP-B/16 | plain ViT, d=768, 196 tok | 1.27x | n/a | last_hidden cos 0.9999864 |
| CLIP ViT-B/16 | plain ViT, d=768, 197 tok | 1.22x | n/a | last_hidden cos 0.9999955 |
| RF-DETR base | DINOv2 ViT-S, d=384, windowed | 1.05x | ~1.6x | COCO mAP 0.564 = 0.564 (200-img) |
| RF-DETR nano | DINOv2 ViT-S, d=384, windowed | 1.03x | ~1.6x | COCO mAP 0.5049 vs 0.5051 (500-img) |
| SAM ViT-B | ViT-Det, d=768, 4096 tok | 1.03-1.11x | ~1.71x | enc cos 0.9999995, mask-IoU 0.9998 |
| Depth Anything v2 Small | DINOv2 ViT-S, d=384, global 1370 tok | 0.78x | n/a | depth cos 0.9999999 |
| Laya English | ModernBERT-large, d=1024, 48-512 tok | 1.66-3.11x | n/a | answers agree 4/4, max prob diff 5.1e-4 |
| Laya typed decisions | ModernBERT-large, d=1024, 58-639 tok | 1.87-3.03x | n/a | answers agree 3/3, max prob diff 3.3e-4 |
| Laya multilingual | mmBERT-base, d=768, 56-726 tok | 1.12-2.46x | n/a | answers agree 6/6, max prob diff 3.3e-4 |

**Which models win is predictable.** The NPU runs the projection GEMMs, the large-K matmuls,
about 2.7x more efficiently per-MAC than attention. Attention's score matmul contracts over the
head dimension (64) and materializes an `ntok x ntok` score matrix per head. So a model's
single-stream envelope tracks
its *effective* attention-to-projection ratio, `ntok / 4d`, counting windowed attention at its
window length, not the width alone:

| Model | ntok | d | effective ntok/4d | attention | single-stream |
|---|---:|---:|---:|---|---:|
| SigLIP / CLIP B/16 | ~196 | 768 | 0.06 | global, short | 1.22-1.27x |
| RF-DETR base | 1601 | 384 | ~0.07 | windowed 101 | 1.05x |
| SAM ViT-B | 4096 | 768 | ~0.10 | windowed 196 + 4 global | 1.05x |
| Depth Anything v2 Small | 1370 | 384 | 0.89 | global 1370 | 0.78x |

Short-sequence wide encoders (CLIP/SigLIP) win, windowed encoders sit at parity, and a long
global-attention encoder (Depth Anything Small) is the first to lose single-stream. Widening the
encoder helps (d=768 beats d=384) by raising the projection share, but it cannot shrink an
attention cost that only sequence length controls.

The screen prices the NPU's side and holds the CPU's per-MAC rate fixed. That holds across the
vision families and not for the text one. ONNX Runtime's CPU kernels run the Laya graph, a d=1024
encoder with masked fp32 attention, at ~77 GOP/s. So Laya reads 3.03x at an `ntok/4d` of 0.16,
where the vision families predict parity. The low end of each Laya range is its
shortest request, where the NPU's per-call dispatch dominates.

**Throughput is the headline.** A single encode is partly host-bound, in score readback,
softmax, layout and the CPU tail. So the NPU's value is running one process per stream, and P=4
peaks the 3-core NPU.

The best host-core placement is model-dependent. RF-DETR scales best with the host threads left
unpinned (`ROCKET_ORT_PIN=0`), and SAM with a per-process `ROCKET_CPU_AFFINITY` spread across
distinct A76 cores. The NPU pool scales ~1.6x on RF-DETR and ~1.71x on SAM over a single NPU
stream. It also frees the A76 cores a CPU-only run would saturate.

The CPU pools too, so the NPU-vs-CPU number measured against the CPU's *own* best pool is
narrower. SAM is 1.44x there, and RF-DETR's margin is thinner, because the CPU pools well on
that workload. Reproduce the vs-CPU-pool ratio with
[`tools/bench_pool.py`](tools/bench_pool.py). The encoder-only families (CLIP/SigLIP/SAM) also
have a ~100% offload share, so no host tail dilutes the encoder's edge.

**ConvTranspose** layers gain in proportion to their share of the model. Inside the model the EP
runs each in 0.2-0.9x the CPU kernel's 4-thread time. A layer with almost no input is the
exception, such as pix2pix's first, at 1.6x. Each output is within 2^-10.4 of its magnitude sum.
The figures are against the CPU EP at `ORT_ENABLE_ALL` with the same intra-op thread count, warm
medians over four to six rotated passes. [HW sweep]

| Model | ConvTranspose share | 4 threads | 2 threads | Faithfulness vs CPU EP |
|---|---:|---:|---:|---|
| pix2pix facades generator | 65% | 1.76x | 2.15x | image max abs 3.7e-3, PSNR 80.8 dB |
| SAM ViT-B mask decoder, 1 point | 18% | 1.05x | 1.10x (spinning on) | mask IoU >= 0.9975 |
| SAM ViT-B mask decoder, 64 points | 22% | 1.10x | 1.15x | worst of 192 masks IoU 0.9931 |

Those figures run the EP session with `session.intra_op.allow_spinning` set to `0`. With ONNX
Runtime's default, its idle threads spin through each offloaded node, and pix2pix at 4 threads
drops to 1.46x.

The remaining lever is a tiled, fused global-attention pass, an FA-2-style running softmax that
keeps the `ntok x ntok` scores off the host DMA path. It would lift both the single-stream number
and the pool ceiling for the attention-bound models (SAM's four global layers, Depth Anything's
every layer). It is the obvious next contribution.

## What runs on the NPU

The EP claims one encoder subgraph per recognized family, and each `ConvTranspose` it can run, and
hands everything else back to ONNX Runtime:

**Offloaded**, in each case to the NPU:

- The ViT encoder, meaning the patch-embed conv, LayerNorm, multi-head self-attention, FFN and
  GELU. It is reused across families.
- RF-DETR's CSP projector.
- SAM's windowed and global attention, with on-chip decomposed relative-position bias, and its
  conv neck.
- Depth Anything's four intermediate encoder taps.
- ModernBERT's encoder and its post-encoder transformer layers, including RoPE, global and
  banded sliding attention, and GeGLU.
- `ConvTranspose` layers in any model, such as a U-Net generator's upsampling path, SAM's two mask
  upscalers and the DPT head's reassemble upsampling.

**Host**, on ONNX Runtime's CPU kernels:

- Deformable-attention `GridSample` sampling and the DETR detection head (RF-DETR).
- The attention-pool and cls-token pooling heads (SigLIP/CLIP).
- The prompt encoder and mask decoder (SAM), except the decoder's two `ConvTranspose` upscalers.
- The DPT reassemble and fusion head, and its `align_corners` resizes (Depth Anything).
- The token-marker gather, scorer and action heads (Laya), a few small MLPs over a handful of rows.

None of these has an NPU route. `GridSample` and `align_corners` resize both need a gather the
RK3588 NPU lacks, and each is a small share of its model.

**Graph optimization must be off for a family claim.** The matcher keys on the exported op
topology, so the session must run with `ORT_DISABLE_ALL`. The `ConvTranspose` claim does not
depend on it. ONNX Runtime's Gemm, Gelu and LayerNorm fusions engage
from `ORT_ENABLE_BASIC` up. They rewrite the encoder block into a topology the matcher does not
model, and the
model then runs on the CPU, apart from any `ConvTranspose`. A graph that is not one of the six
families, or whose optimizations were left on, is left on the CPU, never mis-claimed.

**Quantized models.** The EP consumes int8 and int4 QDQ ONNX, ONNX Runtime's `quantize_static`
format. By default it dequantizes the weights to fp16 and runs the same fp16 datapath, which is
faithful and the fast path on this stack.

A native W8A8 int8 datapath (`ROCKET_ORT_INT8=1`) runs the encoder GEMMs as int8 x int8, and is
~1.8x slower. The stack implements no on-chip int32 accumulation, so every int8 matmul reads its
int32 result back to the host. It is a faithfulness and experimentation mode rather than a speed
lever. Detail and the accuracy numbers are in [API.md](API.md#quantized-models).

## Architecture

A plugin execution provider, meaning an out-of-tree EP shared library:

- ONNX Runtime `dlopen`s the library and calls the exported `CreateEpFactories` and
  `ReleaseEpFactory` entry points. The factory reports the devices it supports. ONNX Runtime then
  asks the EP, via `GetCapability`, which subgraphs it can run, and hands each claimed subgraph
  back through `Compile`. Whatever the EP does not claim runs on ONNX Runtime's own kernels.
- The provider calls ONNX Runtime only through the `OrtApi` and `OrtEpApi` function tables passed
  to `CreateEpFactories`. So the `.so` does not link `libonnxruntime`, and needs only the ONNX
  Runtime headers to build.
- It links `librocketnpu`, the `rocket-userspace` project, for the on-NPU operators.
- A matcher registry tries each family signature in turn during `GetCapability` and claims the
  first that matches. The offloaded subgraph is therefore identified by graph structure rather
  than by tensor names. The signature is the attention and MLP pattern, LayerScale, and the
  patch-embed and neck convs. Family-distinctive nodes complete it, such as SAM's
  relative-position Einsums or Depth Anything's four-tap DPT exit. A re-export that renames the
  interior still matches.
- The same `GetCapability` pass then claims each eligible `ConvTranspose` outside the family claim
  as a fused node of its own. `Compile` packs its weight once into the NPU buffers of one context
  shared by all of them.

## Requirements

- An RK3588 board on a mainline kernel carrying the `rocket` DRM-accel driver, with
  `/dev/accel/accel0` present (`lsmod | grep rocket`).
- The sibling `rocket-userspace` driver library, cloned next to this repo or installed as a
  `rocketnpu` package.
- An ONNX Runtime build exposing the plugin-EP API (`register_execution_provider_library`).
- Privilege to open the accel node. Run with `sudo -E` (the `-E` preserves the `ROCKET_*` env
  knobs plain `sudo` strips).

**Clock.** The NPU boots at 200 MHz and the EP is correct there, and every figure above is at
600 MHz. Apply the `patches/rocket` clock patch and load the module with
`rocket_npu_clk_hz=600000000`.

## Build

Requires a C++17 compiler and CMake. The ONNX Runtime C/C++ API headers for the target ONNX
Runtime version are vendored under `third_party/onnxruntime/include`. Override with
`-DORT_HEADER_DIR=<path>` to build against a different release.

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

The provider loads for any ONNX Runtime application and runs an unmodified `.onnx` model. The
NPU offload activates for a recognized encoder and for `ConvTranspose` nodes, and everything else
runs on the CPU.

For a model whose `ConvTranspose` layers are offloaded, turn off intra-op spinning, or ONNX
Runtime's idle threads take the cores the offloaded node needs:

```python
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
```

For a CLIP, SigLIP or Depth-Anything variant whose attention head dimension is not 64, set
`ROCKET_ORT_SIGLIP_HEADS` to the head count. The EP reports the required value if it cannot infer
it.

For multi-stream throughput, run one process per stream. The best host-core placement is
model-dependent: RF-DETR unpinned via `ROCKET_ORT_PIN=0`, and SAM a per-process
`ROCKET_CPU_AFFINITY` spread. See
[API.md](API.md#runtime-knobs) for the per-model recipe and the full knob table.

## The rocket NPU stack

This provider is one frontend of an open source stack for Rockchip NPUs, three userspace
projects plus a set of optional kernel patches:

- **[`rocket-userspace`](https://github.com/gregordinary/rocket-userspace)** (`librocketnpu`):
  the userspace driver, matmul, and on-NPU op library. It is the dependency, and usable on its
  own.
- **ort-rocket** (this project): an ONNX Runtime execution provider for transformer vision models.
- **[`ggml-rocket`](https://github.com/gregordinary/ggml-rocket)**: a ggml backend `.so` for
  `llama.cpp` / `whisper.cpp`, linking the same driver.
- **[`tflite-rocket`](https://github.com/gregordinary/tflite-rocket)**: a TFLite external delegate
  for detection models, linking the same driver.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope): optional out-of-tree
  kernel-module patches for clock, voltage and IOMMU. They raise the NPU clock from its 200 MHz
  boot default to 600 MHz, and the figures here assume them.

## License and credits

`ort-rocket` is GPL-3.0-or-later. It links the GPL-3 `rocket-userspace` driver library, whose
NPU register headers are the GPL-3 reverse-engineering by Jasbir Matharu. It targets
[ONNX Runtime](https://onnxruntime.ai) (MIT) as an out-of-tree plugin execution provider. The
ONNX Runtime C/C++ API headers vendored under `third_party/onnxruntime/` are MIT-licensed
(Copyright Microsoft Corporation). It builds on the `rocket-userspace` driver for the NPU path.
