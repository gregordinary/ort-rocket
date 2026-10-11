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
encoders. In any model it also offloads the `ConvTranspose` layers one node at a time. It offloads
the `Conv` layers that a cost rule says the NPU runs faster. Everything
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

It claims `Conv` nodes the same way, with a static input size, plus a cost rule. The rule estimates
the node on the NPU and on ONNX Runtime's 4-thread CPU kernel, and claims it where the NPU is clearly
faster. A Relu or LeakyRelu that is the conv's only consumer is folded into the claim.

A graph outside that set runs on ONNX Runtime's CPU kernels, except for these per-node claims.
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

A single stream runs 1.03-2.5x faster than ONNX Runtime's own CPU kernels on a vision model. A
ModernBERT encoder runs 1.3-3.9x faster, depending on the request. On the RK3588 the matmul is
DMA/dispatch-bound rather than MAC-bound, and the Cortex-A76 is a strong baseline. So the gain is
set by how much of a model is projection rather than attention. Running one process per stream adds
throughput, and the NPU frees the A76 cores a CPU-only run would occupy.

Single-stream figures are warm medians, RK3588 @ 600 MHz, with the CPU governor at
`performance`, over four passes with the arm order rotated. Both arms run on the four Cortex-A76
cores (`taskset -c 4-7`) at 4 intra-op threads. The CPU EP runs at `ORT_ENABLE_ALL`, and the EP
session runs with intra-op spinning off. That is the CPU EP's fastest setting on this SoC, and the
only one whose time repeats from process to process. Every offloaded encoder is numerically
faithful to the CPU. [HW sweep, 2026-10-10] Reproduce the single-stream column with
[`tools/bench_ep.py`](tools/bench_ep.py) and the pool column with
[`tools/bench_pool.py`](tools/bench_pool.py):

```sh
sudo -E taskset -c 4-7 python tools/bench_ep.py build/libonnxruntime_rocket.so model.onnx input.npy --threads 4 --no-spin
```

| Model | Encoder offloaded | Single-stream (xCPU) | Pool (xsingle-stream) | Faithfulness vs CPU EP |
|---|---|---:|---:|---|
| SigLIP-B/16 | plain ViT, d=768, 196 tok | 2.44-2.52x | n/a | last_hidden cos 0.9999864 |
| CLIP ViT-B/16 | plain ViT, d=768, 197 tok | 2.39-2.48x | n/a | last_hidden cos 0.9999955 |
| RF-DETR base | DINOv2 ViT-S, d=384, windowed | 1.61-1.65x | ~1.6x | COCO mAP 0.564 = 0.564 (200-img) |
| RF-DETR nano | DINOv2 ViT-S, d=384, windowed | 1.57-1.59x | ~1.6x | COCO mAP 0.5049 vs 0.5051 (500-img) |
| SAM ViT-B | ViT-Det, d=768, 4096 tok | 1.03-1.05x | ~1.71x | enc cos 0.9999995, mask-IoU 0.9998 |
| Depth Anything v2 Small | DINOv2 ViT-S, d=384, global 1370 tok | 1.33-1.35x | n/a | depth cos 0.9999999 |
| Laya English | ModernBERT-large, d=1024, 48-512 tok | 1.99-3.95x | n/a | answers agree 4/4, max prob diff 5.1e-4 |
| Laya typed decisions | ModernBERT-large, d=1024, 58-639 tok | 2.24-3.77x | n/a | answers agree 3/3, max prob diff 3.3e-4 |
| Laya multilingual | mmBERT-base, d=768, 56-726 tok | 1.32-2.80x | n/a | answers agree 6/6, max prob diff 3.3e-4 |

The Laya rows are against the CPU EP at ONNX Runtime's default thread count, over two rotated
passes [HW sweep, 2026-10-10]. Their EP session runs at its default spinning.

With spinning left on, the vision families run up to 1.19x slower on the EP. SigLIP then reads
2.12-2.14x, Depth Anything 1.19-1.24x and RF-DETR nano 1.43-1.46x. CLIP, RF-DETR base and SAM move
by 2% or less. Against an explicit 8-thread CPU EP the ratios read higher and change from process
to process. Eight threads span both CPU clusters, and the CPU EP's time then falls into
per-process modes up to 1.75x apart.

**Which models win follows their attention shape.** The NPU runs the projection GEMMs, the
large-K matmuls, more efficiently per MAC than attention. Attention's score matmul contracts over
only the head dimension (64), and each head's `ntok x ntok` score matrix crosses to the host for
its softmax.
So a model's single-stream ratio falls with its *effective* attention-to-projection ratio,
`ntok / 4d`. It is averaged over the layers, counting a windowed layer at its window length and a
global layer at the full sequence:

| Model | ntok | d | effective ntok/4d | attention | single-stream |
|---|---:|---:|---:|---|---:|
| SigLIP / CLIP B/16 | ~196 | 768 | 0.06 | global, short | 2.39-2.52x |
| RF-DETR nano | 580 | 384 | 0.17 | windowed 145, 3 of 12 layers global | 1.57-1.59x |
| RF-DETR base | 1616 | 384 | 0.33 | windowed 101, 3 of 11 layers global | 1.61-1.65x |
| SAM ViT-B | 4096 | 768 | 0.49 | windowed 196, 4 of 12 layers global | 1.03-1.05x |
| Depth Anything v2 Small | 1370 | 384 | 0.89 | global 1370 | 1.33-1.35x |

Short-sequence wide encoders (CLIP, SigLIP) gain most, and the attention-heavy models least. The
screen is coarse, and two pairs break its order. RF-DETR base reads slightly above nano, so it
does not separate models whose ratios sit that close.

SAM reads below Depth Anything. Its four global layers each attend over 4096 tokens, at 16.8
million scores per head, the largest single attention in the set. That is the likely cause,
inferred and not isolated. Widening the encoder helps (d=768 beats d=384) by raising the
projection share, but it cannot shrink an attention cost that only sequence length controls.

The screen prices the NPU's side and holds the CPU's per-MAC rate fixed. That holds across the
vision families and not for the text one. ONNX Runtime's CPU kernels run the Laya graph, a d=1024
encoder with masked fp32 attention, at ~77 GOP/s. So Laya reads 3.56-3.77x at an `ntok/4d` of 0.16,
where RF-DETR nano reads 1.57-1.59x. The low end of each Laya range is its shortest request, where
the NPU's per-call dispatch dominates.

**Throughput.** A single encode is partly host-bound, in score readback, softmax, layout and the
CPU tail. So running one process per stream adds throughput, and P=4 peaks the 3-core NPU.

The best host-core placement is model-dependent. RF-DETR scales best with the host threads left
unpinned (`ROCKET_ORT_PIN=0`), and SAM with a per-process `ROCKET_CPU_AFFINITY` spread across
distinct A76 cores. The NPU pool scales ~1.6x on RF-DETR and ~1.71x on SAM over a single NPU
stream. It also frees the A76 cores a CPU-only run would saturate. The pool figures date from
July 2026, on a build whose single stream was slower, and are not re-measured.

The CPU pools too, so the NPU-vs-CPU number measured against the CPU's *own* best pool is
narrower. SAM is 1.44x there, and RF-DETR's margin is thinner, because the CPU pools well on
that workload. Reproduce the vs-CPU-pool ratio with
[`tools/bench_pool.py`](tools/bench_pool.py). The encoder-only families (CLIP/SigLIP/SAM) also
have a ~100% offload share, so no host tail dilutes the encoder's edge.

**ConvTranspose** layers gain in proportion to their share of the model. Inside the model the EP
ran each in 0.2-0.9x the CPU kernel's 4-thread time on a 2026-09-27 build. A layer with almost no
input is the exception, such as pix2pix's first, at 1.6x. Each output is within 2^-10.4 of its
magnitude sum. The table is against the CPU EP at `ORT_ENABLE_ALL` with the same intra-op thread
count. Each figure is a warm median over four to six rotated passes, with the CPU governor at
`performance` [HW sweep, 2026-10-10].

| Model | ConvTranspose share | 4 threads | 2 threads | Faithfulness vs CPU EP |
|---|---:|---:|---:|---|
| pix2pix facades generator | 65% | 1.83x | 2.23x | image max abs 3.7e-3, PSNR 80.8 dB |
| SAM ViT-B mask decoder, 1 point | 18% | 1.07x | 1.13x (spinning on) | mask IoU >= 0.9975 |
| SAM ViT-B mask decoder, 64 points | 22% | 1.12x | 1.15x | worst of 192 masks IoU 0.9931 |

Those figures run the EP session with `session.intra_op.allow_spinning` set to `0`. With ONNX
Runtime's default, its idle threads spin through each offloaded node, and pix2pix at 4 threads
dropped to 1.46x on the 2026-09-27 build.

**Conv** layers are claimed where the cost rule admits them. With both per-node claims, pix2pix at
4 threads runs in 50.5 ms against the CPU EP's 146.8, 2.91x. The ConvTranspose claim alone measures
1.86x in the same run, with the CPU governor at `performance`, and the pair is 2.70x on `ondemand`.
The image stays within 3.9e-3 of the CPU EP's, at a PSNR of 77.3 dB. On Depth Anything v2-small the
rule claims 10 of 30 head convs, and the wall is 0.94-0.97x the claim off [HW sweep, 2026-10-10].

## What runs on the NPU

The EP claims one encoder subgraph per recognized family. It also claims each `ConvTranspose` it can
run and each `Conv` its cost rule admits. Everything else goes back to ONNX Runtime:

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
- `Conv` layers in any model where the cost rule admits them, such as a U-Net generator's encoder
  and the DPT head's strided reassemble conv.

**Host**, on ONNX Runtime's CPU kernels:

- Deformable-attention `GridSample` sampling and the DETR detection head (RF-DETR).
- The attention-pool and cls-token pooling heads (SigLIP/CLIP).
- The prompt encoder and mask decoder (SAM), except the decoder's two `ConvTranspose` upscalers.
- The DPT reassemble and fusion head, and its `align_corners` resizes (Depth Anything), apart from
  the per-node claims.
- The token-marker gather, scorer and action heads (Laya), a few small MLPs over a handful of rows.

None of these has an NPU route. `GridSample` and `align_corners` resize both need a gather the
RK3588 NPU lacks, and each is a small share of its model.

**Graph optimization must be off for a family claim.** The matcher keys on the exported op
topology, so the session must run with `ORT_DISABLE_ALL`. The `ConvTranspose` and `Conv` claims
do not depend on it. ONNX Runtime's Gemm, Gelu and LayerNorm fusions engage
from `ORT_ENABLE_BASIC` up. They rewrite the encoder block into a topology the matcher does not
model, so the model runs on the CPU apart from the per-node claims. A graph that is not one of the six
families, or whose optimizations were left on, is left on the CPU, never mis-claimed.

**Quantized models.** The EP consumes int8 and int4 QDQ ONNX, ONNX Runtime's `quantize_static`
format. By default it dequantizes the weights to fp16 and runs the same fp16 datapath, which is
faithful and the fast path on this stack.

A native W8A8 int8 datapath (`ROCKET_ORT_INT8=1`) runs the encoder GEMMs as int8 x int8, and is
~1.7x slower. Each of its matmuls returns int32 results to the host, which requantizes them. It is
an experimentation mode rather than a speed lever. Detail and the accuracy numbers are in
[API.md](API.md#quantized-models).

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
- The same `GetCapability` pass then claims each eligible `ConvTranspose` and `Conv` outside the
  family claim, each as a fused node of its own. `Compile` packs its weight once into the NPU buffers
  of one context shared by all of them.

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
NPU offload activates for a recognized encoder and for the per-node claims, and everything else
runs on the CPU.

Turn off intra-op spinning in a session that uses the EP. Otherwise ONNX Runtime's idle threads
take the cores the offloaded work needs, which slows four of the six families and the per-node
claims. The EP logs one warning when it claims work in a session that spins:

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

This provider is one frontend of an open source stack for Rockchip NPUs. The stack is a driver
library, three frontends, a submit provider for the vendor kernel, and a set of optional kernel
patches:

- **[`rocket-userspace`](https://github.com/gregordinary/rocket-userspace)** (`librocketnpu`):
  the userspace driver, matmul, and on-NPU op library. It is the dependency, and usable on its
  own.
- **ort-rocket** (this project): an ONNX Runtime execution provider for transformer encoders and
  per-layer convolutions.
- **[`ggml-rocket`](https://github.com/gregordinary/ggml-rocket)**: a ggml backend `.so` for
  `llama.cpp` / `whisper.cpp`, linking the same driver.
- **[`tflite-rocket`](https://github.com/gregordinary/tflite-rocket)**: a TFLite external delegate
  for detection models, linking the same driver.
- **[`rknpu-submit`](https://github.com/gregordinary/rknpu-submit)**: a submit provider that runs
  `librocketnpu` on the vendor `rknpu` BSP kernel driver instead of mainline `rocket`.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope): optional out-of-tree
  kernel-module patches for clock, voltage and IOMMU. They raise the NPU clock from its 200 MHz
  boot default to 600 MHz, and the figures here assume them.

## License and credits

`ort-rocket` is GPL-3.0-or-later. It links the GPL-3 `rocket-userspace` driver library, whose
NPU register headers are the GPL-3 reverse-engineering by Jasbir Matharu. It targets
[ONNX Runtime](https://onnxruntime.ai) (MIT) as an out-of-tree plugin execution provider. The
ONNX Runtime C/C++ API headers vendored under `third_party/onnxruntime/` are MIT-licensed
(Copyright Microsoft Corporation). It builds on the `rocket-userspace` driver for the NPU path.
