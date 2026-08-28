# ort-rocket: API and tuning reference

The per-family faithfulness and performance detail, the quantized-model behavior, the complete
`ROCKET_ORT_*` knob table, and the matcher / implementation notes. The [README](README.md) is the
guide; this is the reference.

## Model families

The EP recognizes five transformer vision families by graph topology and offloads each one's
encoder, reusing a single on-NPU ViT primitive (patch-embed conv, LayerNorm, multi-head
self-attention, FFN, GELU) across all of them. The compute is the `rocket-userspace` DINOv2 / plain
-ViT encoder path, validated at both d=384 and d=768. All faithfulness figures are in-process
against the ONNX Runtime CPU EP on the same input; all latency figures are warm, RK3588 @ 600 MHz,
resident, one A76-pinned stream. [HW sweep]

### RF-DETR (nano + base)

Offloads the DINOv2 ViT-S backbone (windowed + global attention, LayerScale) and the CSP feature
projector as one fused node. The deformable-attention decoder (bilinear `GridSample`, no NPU gather
route) and the detection head stay on the CPU, only 5.8% of base's runtime, so the offloaded share
is ~90%. COCO mAP@[.5:.95] is at parity with the CPU EP: nano 0.5049 vs 0.5051 (500-image slice),
base 0.564 vs 0.564 (200-image slice).
Single-stream nano 1.03x / base 1.05x; process pool ~1.6x over a single stream (its edge over the
CPU's own best pool is thinner; see the envelope below). Detections match a CPU run
within fp16 tolerance (classes identical, score <= 1.2e-3, box <= 1.7e-4).

### CLIP and SigLIP image encoders

One plain-ViT matcher covers both (`google/siglip-base-patch16-224`,
`openai/clip-vit-base-patch16`). The pooling head (SigLIP's MAP attention-pool, CLIP's cls-token
Gather and post-LN) stays on the CPU, a ~0% tail. CLIP's cls token, pre-encoder LayerNorm, and
quick-GELU are all resolved from the graph and handled encoder-side. `last_hidden_state` cosine:
SigLIP 0.9999864, CLIP 0.9999955 (CLIP `pooler_output` 0.9999712). Single-stream 1.27x (SigLIP) /
1.22x (CLIP), the best in the set, because the ~100% offload share and short 196/197-token
sequence put the work in the projections, not attention. An odd token count (CLIP's 197) is padded
to a multiple of 4 with zero rows.

### SAM ViT-Det image encoder

`facebook/sam-vit-base`: a high-resolution (1024 px, 4096-token) ViT-Det encoder with 8 windowed and
4 global attention layers and decomposed relative-position bias, plus a two-conv neck. The prompt
encoder and mask decoder stay on the CPU. The relative-position bias, a q-dependent additive term
resolved from the raw ONNX parameter by a Toeplitz gather, is computed on-chip in fp16 (validated
cos >= 0.9999991 per layer vs fp32). Encoder cos 0.9999995; mask-IoU 0.9998 over six point prompts
vs the CPU EP. Single-stream 1.03-1.11x; process pool 1.44x best-over-best (1.71x over its own single
stream). The four global layers materialize the full 4096 x 4096 score matrix per head, which is the
bottleneck; see the tiled-attention lever below.

### Depth Anything v2

`depth-anything/Depth-Anything-V2-Small-hf`: the DINOv2 ViT-S backbone (LayerScale, a fused qkv that
splits key/value/query) feeding four intermediate taps into the DPT head. The DPT reassemble /
fusion head (20% of runtime: conv + `align_corners` resize + concat) stays on the CPU,
`align_corners` resize needs a lattice the NPU's fixed-stride resize cannot express, and the head's
convs are low-intensity at high resolution. Depth-map cos 0.9999999, AbsRel 0.00057, delta-1
1.00000; per-tap cos >= 0.9999969. **Single-stream 0.78x, a loss.** Depth Anything is the first
family whose attention is both global and long (1370 tokens every layer), and attention is the part
the NPU runs worst, so its `ntok/4d` of 0.89 lands it below parity. It ships for coverage and pool
throughput, not single-stream. The Base variant (d=768) is matcher-supported and projects better,
but is not expected to turn single-stream positive (attention cost scales with sequence length,
which width does not shrink); Small is the device-validated one.

## The performance envelope

At the RK3588's current operating point the resident matmul runs ~460 GOP/s across precisions,
about 15% of MAC peak, so it is DMA/dispatch-bound, not MAC-bound. Two consequences shape every
number above.

1. **A ViT-S (d=384) encoder runs only ~1.05x faster on the NPU than on the 8-core A76.** Backing
   the ratio out of RF-DETR base's shipped envelope (EP 993 ms vs CPU 1039 ms, backbone 89.7% of
   runtime) gives r ~= 1.047. This is a ceiling on single-stream: even at 100% offload share, e2e
   cannot exceed r. Width raises it (d=768 SigLIP gets 1.27x); it does not otherwise move.

2. **The edge lives in the projections and is eroded by global attention length.** The projection
   GEMMs (deep-K matmuls) run ~2.7x better time-per-MAC than the attention score matmul, which
   contracts over the head dim (64, far below the K the CBUF wants to amortize) and materializes
   `ntok x ntok` scores per head. A model's envelope therefore follows its effective `ntok/4d` (the
   attention-to-projection FLOP ratio, counting windowed attention at its window length), not its
   width; the table in the README orders every shipped measurement by it.

**Throughput** is where the stack wins, though less than a single number suggests. The NPU pool
scales ~1.6x (RF-DETR) / ~1.71x (SAM) over a single NPU stream, capped near ~2x by the 3-core NPU
(P=4 peak; P>=5 oversubscribes the four A76 cores and drops). But the CPU pools too, so the
like-for-like NPU-vs-CPU comparison is against the CPU's *own* best pool: SAM measures 1.44x there,
and RF-DETR's margin is thinner (the CPU pools well on that workload). `tools/bench_pool.py` reports
this vs-CPU-pool ratio. The best host-core placement is model-dependent: RF-DETR scales best with
the host threads left unpinned (`ROCKET_ORT_PIN=0`, so the OS spreads them across all eight cores),
while SAM's pthread attention workers need a per-process `ROCKET_CPU_AFFINITY` spread across distinct
A76 cores (its `ROCKET_ORT_AFFINITY_BASE` is inert).

**The open lever** is a tiled, fused global-attention pass, an FA-2-style running softmax that
injects the relative-position bias per K/V tile and never materializes the full `ntok x ntok` score
matrix on the host DMA path. It attacks the dominant term for SAM's four global layers and Depth
Anything's every layer, and would raise both single-stream and the pool ceiling. It is the
highest-value next contribution.

## Quantized models

The EP consumes int8 and int4 quantized ONNX in ONNX Runtime's QDQ format
(`DequantizeLinear`-wrapped weights from `quantize_static`). At compile it recognizes int8/int4
codes, per-tensor or per-channel scales, symmetric or asymmetric zero-points, and int32 biases.

**Default: dequantize to fp16.** The weights are dequantized once and run through the existing fp16
datapath (weights quantized, activations fp16). Same `.onnx` in, no extra flags, `ORT_DISABLE_ALL`
as always. This is faithful and the fast path. On RF-DETR-nano at 600 MHz the int8 model this way
scores COCO mAP 0.458 vs 0.516 for fp32; the ~0.06 gap is the model's own quantization loss (the
same int8 model on the ORT CPU EP scores 0.428, so NPU consumption is *more* accurate, keeping
activations in fp16). int4 weight-only runs the same way; int4 is lossy for a ViT (mAP ~0.19), a
property of int4-on-ViT, not of the path.

**Opt-in native int8 (`ROCKET_ORT_INT8=1`).** Executes the encoder projection GEMMs as W8A8 (int8
weights, host-quantized int8 activations from the model's static scales; attention, projector, and
norms stay fp16). Faithful (mAP 0.456, matching the fp16 path's 0.458) but ~1.8x the fp16 latency,
because mainline `rocket` has no on-chip int32 accumulation and every int8 matmul reads its int32
result back to the host. A faithfulness / experimentation mode, not a speed lever; leave it off
for deployment. `ROCKET_ORT_STRICT=1` turns a native-int8 marshaling miss into a hard failure
(instead of a silent fp16 fallback) and prints a self-check summary.

Produce a quantized model with the ONNX Runtime static-quantization API
(`onnxruntime.quantization.quantize_static`, int8 QDQ or `--weight-bits 4` for int4 weight-only);
`tests/test_quant_ep.py` runs one on device against a CPU-EP reference.

## Recommended configurations

Set the EP up as in [Use from an application](README.md#use-from-an-application): register the
library, `ORT_DISABLE_ALL`, `sudo -E`, 600 MHz clock. Then:

| Workload | Add | Effect |
|---|---|---|
| Single image, lowest latency | nothing (resident on by default) | Parity to 1.27x by family; the resident ctx packs weights once |
| Batch / offline (embed a dataset, depth over video) | one process per stream (RF-DETR `ROCKET_ORT_PIN=0`; SAM per-process `ROCKET_CPU_AFFINITY=<distinct A76>`) | process-pool throughput, the axis where the NPU wins (SAM 1.44x vs the CPU's own best pool) |
| CLIP/SigLIP/Depth, head dim != 64 | `ROCKET_ORT_SIGLIP_HEADS=<n_head>` | Required; the EP errors with the needed value if it cannot infer the head count |
| Model must fit less RAM | a Q8 / Q4 QDQ ONNX | Footprint, not speed; dequant-to-fp16 is faithful and as fast as fp16 |
| Numeric experiment | `ROCKET_ORT_INT8=1` (+ `ROCKET_ORT_STRICT=1`) | Native W8A8; faithful but ~1.8x slower, so never for deployment |

## Runtime knobs

The EP reads a set of `ROCKET_ORT_*` env vars, plus the driver's `ROCKET_CPU_AFFINITY`. `sudo`
strips the environment, so always use `sudo -E`.

| var | default | meaning |
|---|---|---|
| `ROCKET_ORT_RESIDENT` | on | Resident prepacked-multicore encoder ctx: pack static weights once, re-pack only activations per image. `=0` uses the one-shot per-call chain (simpler, slower) |
| `ROCKET_ORT_THREADS` | 3 | NPU worker count (clamped 1-16), fanned across the 3 NPU cores |
| `ROCKET_ORT_SIGLIP_HEADS` | d/64 | Attention head count for the plain-ViT / SigLIP / CLIP / Depth families; required when d is not a multiple of 64 (the EP errors with the value to set) |
| `ROCKET_ORT_HT` | 4 (1 if single-threaded) | Host LayerNorm / GELU fan-out width across A76 cores (clamped 1-8) |
| `ROCKET_ORT_PIN` | on | Pin host workers to A76 big cores; `=0` (unpinned) lets the OS spread streams across all cores, the RF-DETR process-pool recipe |
| `ROCKET_ORT_AFFINITY_BASE` | 0 | Per-process big-core rotation base, so an in-process stream pool partitions cores instead of stacking them |
| `ROCKET_CPU_AFFINITY` | unset | (driver knob) restrict the process's NPU workers to a core set; set per process to a distinct A76 core to spread a SAM multi-process pool (RF-DETR pools better unpinned; see `ROCKET_ORT_PIN`) |
| `ROCKET_ORT_INT8` | off | Native W8A8 int8 encoder GEMMs (faithful, ~1.8x slower) |
| `ROCKET_ORT_STRICT` | off | With `ROCKET_ORT_INT8`, escalate a marshaling miss from silent fp16 fallback to a hard Compile failure + self-check summary |
| `ROCKET_ORT_I8_MASK` | 0x1F | Bitmask selecting which of the 5 encoder GEMMs run int8 (A/B diagnostic) |
| `ROCKET_ORT_PROF` | off | Print a per-phase timing breakdown (attention / projections / LayerNorm / scatter) at exit |

## Matcher and implementation notes

- **Graph optimization must be disabled** (`ORT_DISABLE_ALL`). The matcher recognizes the raw
  exported op topology; ONNX Runtime's Gemm / Gelu / SkipLayerNorm fusions, engaged from
  `ORT_ENABLE_BASIC` up, rewrite the encoder block into a topology the matcher does not model, so
  the model falls back entirely to the CPU. Recognizing the fused forms is a known follow-on.
- **Structural, name-robust matching.** Each family is a topology signature (attention / MLP walk,
  LayerScale, patch-embed / neck convs, plus a family discriminator: SAM's relative-position
  Einsum pair, Depth Anything's four-tap DPT exit). The registry tries them in a fixed order and
  claims the first match; a re-export that renames interior tensors still matches. Over-claiming
  would silently corrupt output (there is no fallback once claimed), so every signature ships with an
  off-device oracle test before it claims on device.
- **The resident context** packs the static weights into NPU buffers once at `Compile` and fans
  attention across the NPU cores; each image re-packs only its activations. `ROCKET_ORT_RESIDENT=0`
  falls back to a per-call chain.
- **Odd token counts pad to a multiple of 4.** The prepacked matmul mis-computes for `M % 4 != 0`
  (rows are the conv's spatial height), so a token count like CLIP's 197 or Depth Anything's 1370 is
  padded with zero rows (a zero matmul input yields zero output, never poisoning a real token).
- **Null optional inputs are a device-only crash.** An omitted optional node input (`Pad`
  constant_value, `Slice` steps, `Resize` roi/scales) returns a null value-info handle whose name
  lookup segfaults, and the Python `onnx` oracle never sees it (it returns "" for omitted inputs),
  so it surfaces only on device. Families with `Pad` / `Slice` / `Resize` / `Split` (SAM, Depth
  Anything) guard every edge-name lookup.
- **EP ABI.** The provider calls ONNX Runtime only through the `OrtApi` / `OrtEpApi` tables, so it
  does not link `libonnxruntime`; it needs only the vendored headers
  (`third_party/onnxruntime/include`, overridable with `-DORT_HEADER_DIR`). Pin the ONNX Runtime
  version, since the plugin-EP ABI moves between releases.
