# ort-rocket

## AI disclosure

Except for the prior work it builds on, ort-rocket was developed by AI, primarily Claude. Human
involvement was mostly limited to setting project goals and providing hardware access. This is a
side project for curiosity's sake and comes with no guarantee of quality, accuracy, or update
frequency.

## About ort-rocket

ort-rocket is an ONNX Runtime execution provider (EP) for Rockchip NPUs, validated on the RK3588.
It drives the NPU through the mainline `rocket` DRM-accel driver and `librocketnpu`, the driver
library from the separate `rocket-userspace` project.

The EP is a plugin execution provider: a shared library that stock ONNX Runtime 1.27 loads at run
time with `register_execution_provider_library`. The EP then appears in ONNX Runtime's list of EP
devices, and a session selects it like any other EP. The application runs an unmodified `.onnx`
model.

The EP offloads two kinds of work:

- **Family claims.** The EP recognizes the encoder of five model families by graph topology, and
  claims it as one fused node. The families are ViT image encoders with their convolution necks,
  and ModernBERT text encoders.
- **Per-node claims.** In any model, the EP claims each `ConvTranspose` node it can run. It also
  claims each `Conv` node that its cost rule estimates is faster on the NPU.

Inside a fused node, the matmuls run on the NPU. The normalization, activation, bias, residual and
softmax passes run on the host CPU. Everything the EP does not claim runs on ONNX Runtime's CPU
EP, such as deformable-attention sampling, detection heads and pooling heads.

[API.md](API.md) is the reference for each family, the per-node claims, quantized models and the
environment variables.

## What runs on the NPU

| Model | Claimed by the EP | Left on the CPU EP | Status |
|---|---|---|---|
| RF-DETR nano and base | The DINOv2 ViT-S backbone and the CSP feature projector, as one fused node | The deformable-attention decoder and the detection head | Supported |
| CLIP and SigLIP image encoders, one family | The plain-ViT encoder | The pooling head | Supported |
| SAM ViT-B (Segment Anything) | The ViT-Det image encoder, with its windowed and global attention and the conv neck | The prompt encoder and the mask decoder, apart from the decoder's two `ConvTranspose` upscalers | Supported |
| Depth Anything v2 Small | The DINOv2 backbone, up to its four intermediate taps | The DPT head, apart from its per-node claims | Supported |
| Depth Anything v2 Base | The same as Small | The same as Small | Supported |
| ModernBERT text encoders, with an optional stack of pre-norm transformer layers after them | The encoder and those layers, over a batch of variable-length sequences | The model's task heads | Supported. Validated on the Laya decision model |
| `ConvTranspose` nodes, in any model | A 2-D, `group` 1, fp32 node with a constant weight | Depthwise and grouped transposes | Supported (default) |
| `Conv` nodes, in any model | A 2-D, `group` 1, fp32 node with a constant weight and a static input size, where the cost rule admits it | Depthwise and grouped convs, and a node the cost rule rejects | Supported (default) |
| RF-DETR int8 and int4 QDQ models | The same encoder, with its weights dequantized to fp16 | | Supported (default) |
| RF-DETR native int8 (W8A8) | The encoder's projection GEMMs, run as int8, ~1.8x slower than the fp16 path | | Supported (opt-in) |
| QDQ models of the other families | | The whole model, apart from its per-node claims | Not supported. The EP declines the encoder with a warning |

> [!CAUTION]
> A family claim needs graph optimization off. Run the session at `ORT_DISABLE_ALL`. From
> `ORT_ENABLE_BASIC` up, ONNX Runtime's Gemm, Gelu and LayerNorm fusions rewrite the encoder into
> a topology the matcher does not recognize. The encoder then runs on the CPU EP, with no error.
> The per-node claims work at every optimization level.

When the matcher recognizes a family, the EP marshals the encoder's weights, opens the NPU and
packs the resident weights before it claims the encoder. If any of these steps fails, the EP logs
a warning and leaves the encoder on the CPU EP, and session creation goes on. The common causes are
a process that cannot open the NPU and a part other than the RK3588. The EP also declines a QDQ
model of any family but RF-DETR.

> [!NOTE]
> Set `session.intra_op.allow_spinning` to `0` in a session that uses the EP. With intra-op
> spinning on, ONNX Runtime's idle threads occupy the cores that the EP's host work needs. A vision
> model then runs up to 1.23x slower. The EP logs one warning when it claims work in a session that
> spins.

The matcher recognizes a family by the structure of its encoder, not by tensor names, so a
re-export that renames tensors still matches. RF-DETR is the exception. The EP reads its stem's
cls token and position table, its feature norm and its projector by their exported names. It
therefore declines a renamed RF-DETR export. A model outside the families runs on the CPU EP,
apart from its per-node claims.

`GridSample` and a resize with `align_corners` both need a gather the RK3588 NPU does not have.
RF-DETR's deformable-attention sampling and Depth Anything's DPT resizes therefore run on the CPU
EP, and each is a small share of its model.

## Terms

| Term | Definition | Source | Nearest standard term |
|---|---|---|---|
| Claim | A node the EP takes in its answer to ONNX Runtime's `GetCapability` query, and then runs. A node the EP declines runs on the CPU EP. | ONNX Runtime EP API | Node assignment |
| Cost rule | The test that decides whether the EP claims a `Conv` node, by comparing an estimate of the NPU call with an estimate of the CPU EP's kernel. | Coined | |
| Family | A model architecture whose encoder the matcher recognizes, and which the EP claims as one fused node. | Coined | |
| Family claim | The fused node that holds a recognized family's encoder. | Coined | |
| Fused node | A single node that the EP compiles and runs in place of a set of claimed nodes. | ONNX Runtime EP API | |
| Intra-op spinning | ONNX Runtime's thread-pool behavior in which an idle intra-op thread busy-waits for work instead of sleeping (`session.intra_op.allow_spinning`). | ONNX Runtime session options | |
| Matcher | The part of the EP that recognizes a family by its graph topology rather than by its tensor names. | Coined | Pattern matcher |
| One-shot path | The computation that converts and packs every weight on each run (`ROCKET_ORT_RESIDENT` set to `0`). | Coined | |
| Per-node claim | The claim of one `ConvTranspose` or `Conv` node as a fused node of its own, in any model. | Coined | |
| Resident weights | Weights that the EP converts to fp16 and packs into NPU buffers once, when the session is created, and reuses on every run. | Coined | Prepacked weights |

## Performance

On a single stream, the EP runs a vision model 1.08-3.09x as fast as the CPU EP, and a ModernBERT
request 1.50-5.36x as fast. The gain depends on how much of a model's work is in its projections
rather than in attention, which the [`ntok/4d` ratio](#attention-shape-and-speedup) estimates. The
lowest speedup is SAM's, at near-parity.

**Conditions:** RK3588 (a Turing RK1), NPU 928 MHz (the 900 MHz request), CPU governor
`performance`, warm medians. Both runs use the four Cortex-A76 cores (`taskset -c 4-7`) and 4
intra-op threads. The CPU EP runs at `ORT_ENABLE_ALL`, and the EP session at `ORT_DISABLE_ALL`
with intra-op spinning off [HW sweep 2026-10-27].

**Not characterized:** RF-DETR's COCO mAP and SAM's mask IoU on this build. They were measured on
July 2026 builds. Depth Anything's speedups under the `Conv` cost rule's current estimate, which
admits more of its head convs than the one they were measured under. On Small it runs the model in
0.975-1.008x that one's time [HW sweep 2026-10-29].

| Model | Encoder offloaded | Speedup over the CPU EP | Agreement with the CPU EP |
|---|---|---:|---|
| SigLIP-B/16 | Plain ViT, d=768, 196 tokens | 3.05-3.09x | `last_hidden_state` cosine 0.9999868 |
| CLIP ViT-B/16 | Plain ViT, d=768, 197 tokens | 3.01-3.04x | `last_hidden_state` cosine 0.9999890 |
| Depth Anything v2 Base | DINOv2 ViT-B, d=768, global, 1370 tokens | 2.25-2.28x | Depth cosine 0.9999999 |
| RF-DETR base | DINOv2 ViT-S, d=384, windowed | 1.83-1.88x | COCO mAP 0.564 against 0.564 (200 images) |
| RF-DETR nano | DINOv2 ViT-S, d=384, windowed | 1.70-1.73x | COCO mAP 0.5049 against 0.5051 (500 images) |
| Depth Anything v2 Small | DINOv2 ViT-S, d=384, global, 1370 tokens | 1.63-1.65x | Depth cosine 0.9999998 |
| SAM ViT-B | ViT-Det, d=768, 4096 tokens | 1.08-1.10x | Encoder cosine 0.9999996, mask IoU 0.9998 |
| Laya English | ModernBERT-large, d=1024, 48-512 tokens | 2.34-5.05x | 4 of 4 answers agree, option probabilities within 5.1e-4 |
| Laya typed decisions | ModernBERT-large, d=1024, 58-639 tokens | 2.68-5.36x | 3 of 3 answers agree, within 3.3e-4 |
| Laya multilingual | mmBERT-base, d=768, 56-726 tokens | 1.50-3.57x | 6 of 6 answers agree, within 3.3e-4 |

The Laya rows have their own conditions. The CPU EP runs at ONNX Runtime's default thread count,
which ran this graph faster than 8 threads. The EP session runs with spinning on.
[API.md](API.md#modernbert-text-encoders) has their conditions and the per-request figures.

Reproduce a vision row with [`tools/bench_ep.py`](tools/bench_ep.py):

```sh
# sudo resets PATH, so name the virtual environment's interpreter
sudo -E taskset -c 4-7 "$(command -v python)" tools/bench_ep.py build/libonnxruntime_rocket.so model.onnx input.npy --threads 4 --no-spin
```

### Attention shape and speedup

The NPU runs the projection GEMMs, the matmuls with a large K, more efficiently per MAC than it
runs attention. Attention's score matmul contracts over the head dimension (64) only, and each
head's `ntok × ntok` score matrix goes to the host for its softmax. A model's single-stream
speedup therefore falls as its effective attention-to-projection ratio, `ntok/4d`, rises. The
ratio is averaged over the layers, counting a windowed layer at its window length and a global
layer at the full sequence:

| Model | `ntok` | `d` | Effective `ntok/4d` | Attention | Single-stream speedup |
|---|---:|---:|---:|---|---:|
| SigLIP-B/16, CLIP ViT-B/16 | ~196 | 768 | 0.06 | Global, short | 3.01-3.09x |
| RF-DETR nano | 580 | 384 | 0.17 | Windowed 145, 3 of 12 layers global | 1.70-1.73x |
| RF-DETR base | 1616 | 384 | 0.33 | Windowed 101, 3 of 11 layers global | 1.83-1.88x |
| Depth Anything v2 Base | 1370 | 768 | 0.45 | Global 1370 | 2.25-2.28x |
| SAM ViT-B | 4096 | 768 | 0.49 | Windowed 196, 4 of 12 layers global | 1.08-1.10x |
| Depth Anything v2 Small | 1370 | 384 | 0.89 | Global 1370 | 1.63-1.65x |

The ratio orders the global-attention encoders. Short-sequence wide encoders (CLIP, SigLIP) gain
most, then Depth Anything Base, then Small. The two Depth Anything variants run the same 1370
tokens, and Base's d=768 halves the ratio and raises the speedup from 1.63-1.65x to 2.25-2.28x.

The windowed encoders measure below a global encoder at a similar ratio: RF-DETR, at 0.17 and
0.33, measures below Depth Anything Base at 0.45. RF-DETR base measures above nano, so the ratio
does not separate models whose speedups are that close. SAM measures lowest. Its four global layers
each attend over 4096 tokens, at 16.8 million scores per head, the largest single attention in the
set. These layers are the likely cause [hypothesis].

The ratio holds the CPU EP's rate per MAC fixed, which is true across the vision models and not
for the text model. The CPU EP runs the Laya graph, a d=1024 encoder with masked fp32 attention,
at ~77 GOP/s. Laya therefore measures 5.35-5.36x at an `ntok/4d` of 0.16, where RF-DETR nano
measures 1.70-1.73x. The low end of each Laya range is its shortest request, where the NPU's
per-call dispatch dominates.

### Per-node claims

A `ConvTranspose` or `Conv` claim gains in proportion to that op's share of the model.

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), CPU governor `performance`, warm
medians. Both runs use 4 intra-op threads on the four A76 cores and run at `ORT_ENABLE_ALL`. The
EP session runs with intra-op spinning off [HW sweep 2026-10-27].

| Model | Claims | CPU EP | ort-rocket | Speedup | Agreement with the CPU EP |
|---|---|---:|---:|---:|---|
| pix2pix facades generator | `ConvTranspose` | 146.6 ms | 76.4 ms | 1.92x | Image within 3.7e-3, PSNR 80.8 dB |
| pix2pix facades generator | `ConvTranspose` and `Conv` | 146.6 ms | 47.4 ms | 3.09x | Image within 3.9e-3, PSNR 77.3 dB |
| SAM ViT-B mask decoder, 1 point | `ConvTranspose` | 83.2 ms | 78.5 ms | 1.06x | Mask IoU 0.9975 or better |
| SAM ViT-B mask decoder, 64 points | `ConvTranspose` | 4625.5 ms | 4155.5 ms | 1.11x | Worst of 192 masks at IoU 0.9931 |

On Depth Anything v2 Small, the cost rule claims 16 of the DPT head's 30 convs. The model then
runs in 0.93-0.97x the time it takes with the `Conv` claim off [HW sweep 2026-10-29].

### Multi-stream throughput

A single stream leaves the NPU partly idle while the host runs the attention softmax and the nodes
left on the CPU EP. Running one process per stream adds throughput: the NPU pool's best is ~1.5x the
single-stream rate on RF-DETR nano and ~1.6x on SAM. The CPU EP gains little from the same layout.
Against the CPU EP's own best pool, the NPU pool is 2.36-2.67x on RF-DETR nano and 1.72-1.75x on
SAM. [API.md](API.md#multi-stream-throughput) has the per-model process layout.

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), CPU governor `performance`, warm. Each EP
process runs 4 intra-op threads with spinning off. The CPU EP's best pool is the best of 2x4, 4x2
and 8x1 (processes x threads) [HW sweep 2026-10-27 and 2026-10-29].

**Not characterized:** RF-DETR base, and SAM's CPU EP at 8 processes of 1 thread, which does not
fit in 32 GB.

## Hardware support

| Part | Status |
|---|---|
| RK3588 | Supported. Every figure in this README was measured on one, a Turing RK1. |
| RK3576 | Not supported. `librocketnpu`'s fp16 programs run on the RK3588 only. The EP reads the part from the device tree and claims nothing on another part, so the whole model runs on the CPU EP. |

## Requirements

- An RK3588 board on a mainline kernel with the `rocket` DRM-accel driver loaded, so that
  `/dev/accel/accel0` exists (`lsmod | grep rocket`).
- ONNX Runtime 1.27, the release whose headers this repository vendors (API version 27). The EP
  is validated on 1.27.0.
- `rocket-userspace`, either installed as a `rocketnpu` CMake package or cloned next to this
  repository. Its own build needs libdrm, found through `pkg-config`, and the `rocket` uAPI header
  `drm/rocket_accel.h`.
- CMake 3.20 or later, and a compiler for C11 and C++17 that accepts `_Float16` in C++, which on
  aarch64 is GCC 13 or later.
- Privilege to open the accel node. Run with `sudo -E`, because plain `sudo` drops the `ROCKET_*`
  environment variables.

### Clock

The NPU boots at 200 MHz, and the EP computes correctly there. Above 200 MHz the RK3588's NPU
clock follows the rail voltage, so a requested rate is not the rate the NPU runs at.

Apply the `patches/rocket` clock and voltage patches, and load the module with
`rocket_npu_clk_hz=900000000`. On the RK1's 0.80 V rail that request runs the NPU at 928 MHz.

The timings in this README were measured with the 900 MHz request, at 928 MHz. The 600 MHz
request runs at ~816 MHz on the same rail, and the 900 MHz request is 1.044x faster on matmul
device time.

> [!CAUTION]
> With the NPU cores pinned on (`power/control=on`), an `rmmod` at the 900 MHz request hard-locks
> the board, and only a power cycle recovers it. Unload the module only while all three cores are
> runtime-suspended.

## Build and run

### Build

Clone `rocket-userspace` next to this repository:

```
.
  rocket-userspace/   # git clone https://github.com/gregordinary/rocket-userspace
  ort-rocket/         # this repository
```

Then build the EP:

```sh
cmake -S . -B build
cmake --build build -j
# -> build/libonnxruntime_rocket.so
```

> [!NOTE]
> CMake looks for `librocketnpu` in this order: the directory `-DROCKETNPU_DIR=<path>` names, an
> installed `rocketnpu` package, then a `rocket-userspace` checkout next to this repository. An
> installed package therefore takes precedence over the checkout. After changing the checkout,
> reinstall it, or pass `-DROCKETNPU_DIR`.

The EP calls ONNX Runtime only through the `OrtApi` and `OrtEpApi` function tables that ONNX
Runtime passes to it. It therefore does not link `libonnxruntime`, and needs only the ONNX Runtime
headers to build. The headers are vendored under `third_party/onnxruntime/include`. To build
against another release's headers, pass `-DORT_HEADER_DIR=<path>`. The plugin-EP interface
changes between ONNX Runtime releases, so build against the headers of the release you run.

### Run from Python

```python
import onnxruntime as ort

ort.register_execution_provider_library("rocket", "/path/to/libonnxruntime_rocket.so")
dev = next(d for d in ort.get_ep_devices() if d.ep_name == "rocket")

so = ort.SessionOptions()
# A family claim needs the exported topology, which ONNX Runtime's graph optimizations rewrite.
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
so.add_provider_for_devices([dev], {})
sess = ort.InferenceSession("model.onnx", so)
outputs = sess.run(None, {"input": x})
```

The EP's name is the name the library is registered under, here `rocket`.

For multi-stream throughput, run one process per stream.
[API.md](API.md#multi-stream-throughput) gives each model's host-core placement.

> [!CAUTION]
> Release every session, `SessionOptions` and EP device handle that references the EP before
> calling `ort.unregister_execution_provider_library`. A reference still alive at that call is
> destroyed at interpreter exit, after the library is unmapped, and the process crashes.

## Troubleshooting

### Encoder left on the CPU EP

Most often the session runs with graph optimization on, or the model is not one of the families.
Set the session's log level to INFO and read the EP's claim line:

```python
so.log_severity_level = 1
```

`ort-rocket: claiming <n> nodes (...) as one fused node` means the EP made a family claim.
`ort-rocket: no encoder family matched (<reason>)` means it did not, and the reason names what the
matcher missed. If the session runs above `ORT_DISABLE_ALL`, set it to `ORT_DISABLE_ALL` and run
again. The per-node claims also log, at INFO, the nodes they left on the CPU EP, counted by reason.

`ort-rocket: not claiming the <family> encoder, which runs on ONNX Runtime's CPU kernels:
<reason>` is logged at WARNING. The EP recognized the family and could not run it. The reason
names the step that failed. Common ones are a process that cannot open `/dev/accel/accel0`
(below), a QDQ model of a family other than RF-DETR, and a renamed RF-DETR export.

### Session creation failure

A family claim declines instead of failing, as above. A per-node claim reads and packs its weight
in `Compile`, so a failure there fails session creation, with an error that names the node. To
leave those nodes on the CPU EP, set `ROCKET_ORT_CONV=0` or `ROCKET_ORT_CONVTRANSPOSE=0`.

### `session.intra_op.allow_spinning` warning

ONNX Runtime's idle intra-op threads spin while each claimed node runs, and take the cores the EP
needs. Set it to `0`:

```python
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
```

### Slower than the figures above

Check the NPU clock. At its 200 MHz boot rate the EP computes correctly, and runs slower than at
the rates in [Clock](#clock). The figures above are warm medians, and a process's first run is
slower than its later runs.

### `Permission denied` opening `/dev/accel/accel0`

The node exists and this process cannot open it. Run with `sudo -E`, which keeps the `ROCKET_*`
environment variables that plain `sudo` drops.

## The rocket NPU stack

This EP is one frontend of an open source stack for Rockchip NPUs. The stack is a driver library,
three frontends, a submit provider for the vendor kernel, and a set of optional kernel patches:

- **[`rocket-userspace`](https://github.com/gregordinary/rocket-userspace)** (`librocketnpu`): the
  userspace driver, matmul, and on-NPU op library. It is the dependency, and usable on its own.
- **`ort-rocket`**: this EP, an ONNX Runtime execution provider for transformer encoders and
  per-node convolutions.
- **[`ggml-rocket`](https://github.com/gregordinary/ggml-rocket)**: a ggml backend `.so` for
  `llama.cpp` and `whisper.cpp`, linking the same library.
- **[`tflite-rocket`](https://github.com/gregordinary/tflite-rocket)**: a TFLite external delegate
  for detection models, linking the same library.
- **[`rknpu-submit`](https://github.com/gregordinary/rknpu-submit)**: a submit provider that runs
  `librocketnpu` on the vendor `rknpu` BSP kernel driver instead of mainline `rocket`.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope): optional out-of-tree
  kernel-module patches for clock, voltage and IOMMU. They raise the NPU clock above its 200 MHz
  boot default. The timings here were measured with them, at 928 MHz (the 900 MHz request).

## License and credits

`ort-rocket` is GPL-3.0-or-later. It links the GPL-3 `rocket-userspace` driver library, whose NPU
register headers are the GPL-3 reverse-engineering by Jasbir Matharu. It targets
[ONNX Runtime](https://onnxruntime.ai) (MIT) as an out-of-tree plugin execution provider. The ONNX
Runtime C and C++ API headers vendored under `third_party/onnxruntime/` are MIT-licensed
(Copyright Microsoft Corporation).
