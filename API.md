# ort-rocket: API and tuning reference

This reference covers each model family the EP claims, the `ConvTranspose` and `Conv` claims,
quantized models, multi-stream throughput and the environment variables. The
[README](README.md) covers the build, a first run and the headline figures, and its
[Terms](README.md#terms) define the EP's own terms.

## Recommended configurations

Every configuration starts from the session in the README's
[Run from Python](README.md#run-from-python). That session registers the library, runs at
`ORT_DISABLE_ALL` with intra-op spinning off, and runs under `sudo -E`. Then:

| Workload | Set | Effect |
|---|---|---|
| One image at a time, lowest latency | Nothing more. Resident weights are the default | 1.08-3.09x the CPU EP, by vision family |
| A batch or an offline job, such as embedding a dataset or depth over a video | One process per stream, with the model's [host-core placement](#host-core-placement) | More throughput than one stream. The NPU pool is 2.36-2.67x the CPU EP's best pool on RF-DETR nano and 1.72-1.75x on SAM |
| CLIP, SigLIP or Depth Anything whose attention `Reshape` does not name the head dimension, when that dimension is not 64 | `ROCKET_ORT_SIGLIP_HEADS=<heads>` | Required. The EP otherwise takes `d/64` heads and logs a warning. It cannot detect a wrong count that divides `d` |
| An RF-DETR QDQ model | Nothing more | The EP dequantizes the weights to fp16 and runs the fp16 path, at its speed |
| A numeric experiment with int8 activations, on RF-DETR | `ROCKET_ORT_INT8=1`, and `ROCKET_ORT_STRICT=1` to leave the encoder on the CPU EP when int8 cannot run | Native W8A8, ~1.8x slower than the fp16 path |

## Model families

The EP recognizes five families by graph topology, and claims each one's encoder as one fused
node. Four are vision families: RF-DETR, CLIP and SigLIP, SAM, and Depth Anything v2. The fifth is
ModernBERT.

Each family runs its matmuls, including attention's two, on the NPU. Its normalization,
activation, bias, residual and softmax passes run on the host. RF-DETR's backbone and projector are
the EP's own code, built on `librocketnpu`'s matmul and attention. CLIP, SigLIP and Depth Anything
run `librocketnpu`'s resident plain-ViT encoder, SAM its ViT-Det encoder, and ModernBERT its
ModernBERT encoder.

Every agreement figure below compares the EP with the CPU EP on the same input, in one process.

**Conditions:** for the vision single-stream figures, RK3588, NPU 928 MHz (the 900 MHz request),
resident weights, CPU governor `performance`, warm medians. Both runs use 4 intra-op threads on the
four A76 cores (`taskset -c 4-7`). The CPU EP runs at `ORT_ENABLE_ALL`, and the EP session with
intra-op spinning off [HW sweep 2026-10-27]. The encoder and depth-map agreement figures are from
the same build.

**Not characterized:** RF-DETR's COCO mAP and detection agreement, and SAM's mask IoU, on this
build. They were measured on July 2026 builds. Depth Anything's single-stream speedups under the
`Conv` cost rule's current NPU estimate, and Base's agreement under it. They were measured under the
estimate before it, which admits 10 of Small's 30 head convs. The current one admits 16. On Small
the current one runs the model in 0.975-1.008x the earlier one's time, at cosine 0.9999998 and
AbsRel 0.00055 [HW sweep 2026-10-29].

### RF-DETR (nano and base)

The EP claims the DINOv2 ViT-S backbone, with its windowed and global attention and LayerScale,
and the CSP feature projector, as one fused node. The deformable-attention decoder and the
detection head stay on the CPU EP, because the decoder's bilinear `GridSample` has no NPU gather.
They are 5.8% of base's run time, and the offloaded share is ~90%.

COCO mAP@[.5:.95] matches the CPU EP's: nano scores 0.5049 against 0.5051 on a 500-image slice,
and base 0.564 against 0.564 on a 200-image slice. Detections match a CPU EP run with the same
classes, scores within 1.2e-3 and boxes within 1.7e-4.

The single-stream speedup is 1.70-1.73x for nano and 1.83-1.88x for base.

The EP finds the encoder by structure. It reads the stem's cls token and position table, the
feature norm and the projector by name. It therefore declines a renamed export, which then runs on
the CPU EP.

### CLIP and SigLIP image encoders

One plain-ViT matcher covers both, as one family, validated on `google/siglip-base-patch16-224`
and `openai/clip-vit-base-patch16`. The pooling head stays on the CPU EP and takes ~0% of the run
time: SigLIP's MAP attention pool, and CLIP's cls-token `Gather` and post-LayerNorm. The matcher
reads CLIP's cls token, pre-encoder LayerNorm and quick-GELU from the graph, and the encoder
applies them. The attention head count is read from the `Reshape` that splits the query into heads.

The `last_hidden_state` cosine against the CPU EP is 0.9999868 for SigLIP and 0.9999890 for
CLIP, and CLIP's `pooler_output` cosine is 0.9999914.

The single-stream speedup is 3.05-3.09x for SigLIP and 3.01-3.04x for CLIP, the highest of the
vision models. Nearly the whole model is offloaded, and at 196 and 197 tokens most of its work is
in the projections rather than in attention.

### SAM ViT-Det image encoder

`facebook/sam-vit-base` is a ViT-Det encoder at 1024 px, or 4096 tokens. It has 8 windowed and 4
global attention layers with decomposed relative-position bias, and a two-conv neck. The prompt
encoder and the mask decoder stay on the CPU EP, apart from the decoder's two `ConvTranspose`
upscalers, which the [ConvTranspose claim](#convtranspose) takes.

The relative-position bias is an additive term that depends on q. The EP builds it from the raw
ONNX parameter with a Toeplitz gather. The host computes the bias in fp32, rounds it to fp16, and
adds it inside the host softmax. Of attention, the two matmuls run on the NPU. A host simulation of
an all-fp16 encoder, bias included, stays at a cosine of 0.9999991 or better per layer against
fp32 [host-computed].

The encoder's cosine against the CPU EP is 0.9999996, and the mask IoU over six point prompts is
0.9998. The single-stream speedup is 1.08-1.10x. Each of the four global layers computes the full
4096 × 4096 score matrix per head, which is the likely cause of the near-parity [hypothesis].

### Depth Anything v2

`depth-anything/Depth-Anything-V2-Small-hf` is a DINOv2 ViT-S backbone with LayerScale and a fused
qkv that is split into query, key and value. It feeds four intermediate taps to the DPT head. The
matcher accepts two or more taps that share one LayerNorm affine.

The DPT reassemble and fusion head stays on the CPU EP, apart from its per-node claims. Those
are its two `ConvTranspose` layers, and the 16 of its 30 convs that the cost rule admits. A resize
with `align_corners` needs a sampling lattice that the NPU's fixed-stride resize cannot express.
The other convs have a low arithmetic intensity at high resolution.

The depth map's cosine against the CPU EP is 0.9999998, its AbsRel 0.00056 and its delta-1
1.00000, and each tap's cosine is 0.9999968 or better. The single-stream speedup is 1.63-1.65x.

Attention in this model is both global and long, at 1370 tokens in every layer, the largest
effective `ntok/4d` in the set at 0.89. The NPU writes each head's scores to the host as plain
rows, and reads the softmax's probabilities back the same way. Neither matrix is converted to or
from the NPU's tile layout. At this layer (1370 tokens, 6 heads of 64), the two attention matmuls
take about a third of each head's attention time. The host softmax over the score matrix takes the
rest, so the model gains less than the projection-heavy families.

The Base variant (d=768, 12 heads) matches too. Its depth map's cosine against the CPU EP is
0.9999999, its AbsRel 0.00049 and its delta-1 1.00000. Each tap's cosine is 0.999945 or better
[HW sweep 2026-10-27]. Its single-stream speedup is 2.25-2.28x, above Small's at the same
sequence length. Its wider projections raise the projection share [hypothesis].

### ModernBERT text encoders

The EP claims a ModernBERT encoder, and an optional stack of torch pre-norm transformer layers
after it (`nn.TransformerEncoderLayer` with `norm_first`), as one fused node. The node takes the
token ids, and the attention mask and a per-sequence type id where the model has them, as int64,
int32 or bool. It returns the last layer's hidden states. The matcher reads RoPE, the global and
sliding layers, GeGLU and the bias-less LayerNorms from the graph. That includes each layer's RoPE
frequencies, and the sliding window, which must be the same in every layer.

Attention runs on the NPU. RoPE, GeGLU, the LayerNorms and the residual stream run on the host.
The residual stays fp32, because ModernBERT-large's residual reaches 29000, where fp16 values are
16 apart.

The family is validated on `convaiinnovations/laya` at revision `55cf4c4e`. Laya's English and
typed-decisions checkpoints are ModernBERT-large (d=1024, 28 layers), and its multilingual
checkpoint is mmBERT-base (d=768, 22 layers). Laya's marker gather, scorer and action MLPs stay
on the CPU EP.

A batch is B right-padded sequences, one per question. Each sequence runs at its own length, so
padding costs nothing, and each projection runs once over the live rows of the whole batch. Each
row of the attention mask must be a prefix of ones with at least one 1. Any other row fails the
run with `attention_mask is not right-padded` or `a row has no live tokens`.

On 13 recorded requests (48-726 tokens, batches of 1 and 3), every answer agrees with the CPU EP's,
and the largest option-probability difference is 5.1e-4.

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), CPU governor `performance`, warm. The
CPU EP runs at `ORT_ENABLE_ALL` with ONNX Runtime's default thread count, which ran this graph
faster than 8 threads did (1343 against 1580 ms on a 3-question request). The EP session runs with
intra-op spinning on [HW sweep 2026-10-27].

| Checkpoint | Request | CPU EP | ort-rocket | Speedup |
|---|---|---:|---:|---:|
| English | 48 tokens | 378-382 ms | 161 ms | 2.34-2.36x |
| English | 3 questions, 53 tokens | 1230-1241 ms | 282-284 ms | 4.33-4.40x |
| English | 512 tokens | 4867-5040 ms | 983-997 ms | 4.95-5.05x |
| typed decisions | 58 tokens | 459-462 ms | 171-172 ms | 2.68x |
| typed decisions | 639 tokens | 6478-6485 ms | 1210-1211 ms | 5.35-5.36x |
| multilingual | 58 tokens | 155-159 ms | 103 ms | 1.50-1.54x |
| multilingual | 3 questions, 58 tokens | 406-408 ms | 188-227 ms | 1.79-2.17x |
| multilingual | 726 tokens | 2566-2596 ms | 719-911 ms | 2.85-3.57x |

#### Exporting Laya

Laya's own `scripts/export_onnx.py` traces a batch-1 dummy. Under torch 2.9 and later,
`torch.onnx.export` defaults to the dynamo exporter, which bakes that batch into the head's
reshape. Every request with more than one question then fails inside ONNX Runtime. Export with a
batch-2 dummy and `dynamic_shapes`, or pass `dynamo=False`. The EP is validated on the dynamo
graph.

#### Laya's `ONNXAgent`

The agent builds its session with the CUDA or CPU provider at `ORT_ENABLE_ALL`. Build an
ort-rocket session at `ORT_DISABLE_ALL`, and assign it to the agent's `session` attribute after
construction. The agent's tokenization and decoding then run unchanged.

### Evidence

- **Vision single-stream figures:** four passes with the configuration order rotated, and the page
  cache dropped and memory compacted before every process. The NPU's PVTPLL counter read 927.8 MHz
  before the run. On the 2026-10-11 build at ~816 MHz (the 600 MHz request), the same protocol
  measured SigLIP 2.64-2.67x, CLIP 2.58-2.64x and RF-DETR nano 1.65-1.70x. It measured RF-DETR base
  1.77-1.79x, SAM 1.03-1.04x and Depth Anything Small 1.58-1.61x.
- **The CPU EP's times:** the same on both builds. They are 343-345 ms on SigLIP, 324-326 ms on
  CLIP, 1344-1348 ms on Depth Anything Small and 3951-3970 ms on Base. On RF-DETR they are 344-345
  ms on nano and 948-950 ms on base, and on SAM 10.97-11.01 s.
- **The CPU EP baseline:** 4 intra-op threads on the four A76 cores is the CPU EP's fastest
  setting on four of the five vision models. Its time repeats within 0.2% across processes. An
  explicit 8 threads span both CPU clusters, and the CPU EP's time then falls into per-process
  modes up to 1.75x apart. Ratios against 8 threads measure higher and vary from process to
  process [HW sweep 2026-10-09].
- **Intra-op spinning:** with spinning on, the EP's time is 1.19-1.23x its time with spinning off on
  SigLIP and 1.10-1.19x on Depth Anything Small. It is 1.06-1.10x on Base and 1.10-1.11x on RF-DETR
  nano, each pair within one pass. CLIP, RF-DETR base and SAM move by 4% or less. SigLIP then
  measures 2.51-2.55x, Depth Anything Small 1.38-1.48x and RF-DETR nano 1.55-1.56x [HW sweep
  2026-10-27].
- **Laya:** two passes with the configuration order rotated, a process per configuration, and a
  memory reset before every configuration. The multilingual checkpoint's EP time differed between
  the two passes by up to 1.27x (719 against 911 ms at 726 tokens, 188 against 227 ms on three
  questions), so its ranges are wide. On the 2026-10-11 build at ~816 MHz (the 600 MHz request), the
  speedups were 2.03-4.16x on English, 2.29-4.18x on typed decisions and 1.32-2.59x on multilingual.
  The CPU EP's times were within 1.1% of the table's.
- **Attention's host passes:** the 2026-10-11 build writes the scores and reads the probabilities
  as plain rows, which removed two host layout passes. Against the build before, the EP's time
  is 0.84-0.86x on Depth Anything and 0.93-0.95x on SigLIP and CLIP. It is 0.91-0.95x on RF-DETR
  [HW sweep 2026-10-11, governor `ondemand`, confined to the A76 cores, spinning off].

## ConvTranspose

The EP claims ONNX `ConvTranspose` nodes one at a time, in any graph, each as a fused node of its
own. Each runs on `librocketnpu`'s resident transposed convolution: one resident fp16 matmul over
the input's channel planes, then a host scatter-add of the kernel taps.

The claim coexists with a family claim. A `ConvTranspose` inside a claimed encoder is part of the
family claim. One outside it is claimed on its own, such as each of the two in Depth Anything's
DPT head.
`ROCKET_ORT_CONVTRANSPOSE=0` turns the claim off.

### Claim conditions

The EP claims a 2-D `ConvTranspose` in the ONNX domain, with `group` 1 and fp32 input and output,
under these conditions:

- Its weight, and its bias if it has one, is a constant fp32 initializer.
- Its `auto_pad` is `NOTSET` or `VALID`, and it has no `output_shape` attribute.
- Its trailing pads are at most its leading pad plus `output_padding`.
- Its strides, dilations and `output_padding` are at most 64, and its pads at most 4096.
- `librocketnpu`'s planner accepts its shape.

Every other `ConvTranspose` stays on the CPU EP. That includes depthwise and grouped transposes,
`SAME` padding, an `output_shape` attribute, a weight computed in the graph, and a QDQ model's
dequantized weight. At INFO, the session log counts the nodes left on the CPU EP, by reason.

### Run-time behavior

The EP packs each node's weight once, when the session is created. Every claimed `ConvTranspose`
and `Conv` in a session shares one NPU context. A batch runs as one call per image. The claim does
not depend on graph optimization, and works at `ORT_ENABLE_ALL`.

For a symbolic input height or width, as in SAM's mask decoder, the EP packs the weight when a
run first sees each size. A size the planner rejects then runs on the EP's
single-threaded fp32 host path.

With intra-op spinning on, ONNX Runtime's idle threads spin while each claimed node runs, and take
the cores its NPU workers and scatter-add need. On pix2pix at 4 threads, on a 2026-09-27 build,
spinning cost 17 ms of an 83 ms run, and the speedup fell to 1.46x. The CPU EP's time did not
change (146 ms either way).

### Agreement

The EP's output is scored against the CPU EP's, element by element. A test runs every
`ConvTranspose` of pix2pix and of SAM's mask decoder, each with the model's own weights and its
input in the model. The worst error is 2^-11.6 to 2^-10.4 of the element's magnitude sum, the sum
over taps of `abs(x w)` plus the bias.

End to end, the pix2pix facades generator's output image differs by at most 3.7e-3. Its PSNR is
80.8 dB on a structured label map and 77.1 dB on uniform noise. SAM's masks agree at a per-mask
IoU of 0.9975 or better at one point. At 64 points the worst of 192 masks is 0.9931, and the mean
is 0.9992. SAM's IoU predictions do not depend on the upscalers and are unchanged [HW sweep].

### Performance

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), CPU governor `performance`, warm
medians, memory reset before every process. Single-stream against the CPU EP at `ORT_ENABLE_ALL`
with the sequential executor. Both runs use the same intra-op thread count, and the EP session
runs with spinning off. At 4 threads both runs use the four A76 cores. At 2 threads the CPU EP
runs on two A76 cores. The EP runs on four, or on the same two with `ROCKET_CPU_AFFINITY`
[HW sweep 2026-10-27].

| Model | `ConvTranspose` share of the CPU EP run | Threads | CPU EP | ort-rocket | Speedup |
|---|---:|---:|---:|---:|---:|
| pix2pix facades U-Net-256 generator | 65% | 4 | 145.6 ms | 78.0 ms | 1.87x |
| pix2pix facades U-Net-256 generator | 66% | 2 | 258.7 ms | 114.2 ms (116.2 ms on the same two cores) | 2.27x (2.23x) |
| SAM ViT-B mask decoder, 1 point | 18% | 4 | 83.2 ms | 78.5 ms | 1.06x |
| SAM ViT-B mask decoder, 1 point | 18% | 2, spinning on | 135.7 ms | 125.2 ms | 1.08x |
| SAM ViT-B mask decoder, 64 points | 22% | 4 | 4625.5 ms | 4155.5 ms | 1.11x |
| SAM ViT-B mask decoder, 64 points | 20% | 2 | 7777.3 ms | 6703.3 ms | 1.16x |

The speedup follows the share. Inside the model at 4 threads, on a 2026-09-27 build, the EP ran
each node in 0.20-0.86x the CPU kernel's time, conversions included. The exception is pix2pix's
first layer, a `1×1` input against an 8 MB weight, at 1.55x (1.63 against 1.05 ms). A model
therefore gains little unless `ConvTranspose` dominates it.

#### Evidence

Each figure is a median over six passes with the configuration order rotated (four at 64 points),
with each ratio paired within its pass. The paired ratios span 1.85-1.90x and 2.24-2.28x on pix2pix,
and 1.01-1.08x, 1.05-1.15x, 1.11-1.12x and 1.14-1.17x on SAM's decoder, in the table's order. The
same-two-cores figure is a separate run on 2026-10-29, also six passes after a warm-up pass. Its
order alternates, the NPU cores are held on (`power/control=on`), and its paired ratios span
2.21-2.24x. On the 2026-10-10 build at ~816 MHz (the 600 MHz request), pix2pix measured 146.1
against 79.6 ms (1.83x) and 258.8 against 116.4 ms (2.23x). It measured 2.19x on the same two cores,
and SAM's decoder 1.07x, 1.13x, 1.12x and 1.15x.

## Conv

The EP claims ONNX `Conv` nodes one at a time, outside any family claim, each as a fused node of
its own. Each runs on `librocketnpu`'s resident im2col convolution. The host unfolds the input
into channel planes, on several threads for a large input, and one resident fp16 matmul does the
rest. `ROCKET_ORT_CONV=0` turns the claim off.

A Relu or LeakyRelu can join the `Conv`'s fused node. It must be the `Conv`'s only consumer, and
the `Conv`'s output must not be a graph output. The EP applies it, with the bias, while it widens
the fp16 output to fp32.

### Claim conditions

The EP claims a 2-D `Conv` with `group` 1, fp32 input and output, and a static input height and
width, under these conditions:

- Its weight, and its bias if it has one, is a constant fp32 initializer.
- Its `auto_pad` is `NOTSET`, `VALID`, `SAME_UPPER` or `SAME_LOWER`.
- `librocketnpu`'s planner accepts its shape.
- The [cost rule](#cost-rule) admits it.

Every other `Conv` stays on the CPU EP:

- Depthwise and grouped convs
- A symbolic input size
- A weight computed in the graph, or a QDQ model's dequantized weight
- A shape the cost rule rejects

At INFO, the session log counts the nodes left on the CPU EP by reason. It also prints each
claimed node's two estimates.

The EP packs each node's weight once, when the session is created, on the NPU context the
`ConvTranspose` claim uses. A batch runs as one call per image.

The claim does not depend on graph optimization. At `ORT_ENABLE_ALL` the family matcher misses,
so the EP claims a family model's convs one by one instead. On RF-DETR nano those are the patch
embedding and the projector's eight convs. The four detections above 0.5 then agree with the CPU
EP's, with scores within 0.001 and boxes at IoU 0.9996 or better [HW sweep].

### Cost rule

The EP claims a `Conv` only where it estimates the NPU is faster. The NPU estimate is
`librocketnpu`'s `rocket_conv2d_prepacked_estimate_ms`, a linear model in the convolution's matmul
dimensions. The CPU estimate is a linear model in the node's MACs, its weight size and its tensor
sizes. It was fitted at 4 intra-op threads on the A76 cores. If the NPU call plus the fp32
conversions is at most 0.9x the CPU estimate, the EP claims the node.

The 0.9 margin covers what a claimed node costs inside a session above its back-to-back time. That
cost is 1.01-1.04x on Depth Anything's DPT neck and 0.95-1.20x on pix2pix's encoder, with the CPU
governor at `performance`. The CPU estimate was fitted on the `ondemand` governor, where the CPU
EP's kernels run 3-8% slower. On a board at `performance`, the rule therefore claims slightly more
readily. With fewer intra-op threads, the CPU EP is slower than the rule assumes. The rule can
then decline a node the NPU would run faster, and never claims a slower one for that reason.

The NPU estimate is fitted to back-to-back calls with three NPU workers, at 928 MHz (the 900 MHz
request) on the `ondemand` governor. Over its 22 fit shapes, pix2pix's encoder, Depth Anything's
neck and two patch embeddings, it is within 0.85-1.16x of the measured call [HW sweep 2026-10-27].

`ROCKET_ORT_CONV_MARGIN` sets another margin. `ROCKET_ORT_CONV=all` claims every `Conv` the
planner accepts, without the cost rule, for measurement.

### Agreement

The EP's output is scored against the CPU EP's, element by element. Every `Conv` of pix2pix runs
with the model's own weights and its input in the model. The worst error is 2^-12.9 to 2^-10.9
of the element's magnitude sum. With both per-node claims, the pix2pix output image differs by at
most 3.9e-3. Its PSNR is 77.3 dB on a structured label map and 75.9 dB on uniform noise
[HW sweep]. Depth Anything's depth map, with 16 of its convs claimed, is at cosine 0.9999998 and
AbsRel 0.00055 [HW sweep 2026-10-29].

### Performance

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), warm medians, memory reset before every
process, single-stream at 4 intra-op threads on the A76 cores. The CPU EP and the EP session both
run at `ORT_ENABLE_ALL`, with spinning off [HW sweep 2026-10-27].

| Model | CPU governor | CPU EP | `ConvTranspose` claim | `ConvTranspose` and `Conv` claims | Speedup over the CPU EP |
|---|---|---:|---:|---:|---:|
| pix2pix facades U-Net-256 generator | `performance` | 146.6 ms | 76.4 ms | 47.4 ms | 1.92x -> 3.09x |
| pix2pix facades U-Net-256 generator | `ondemand` | 158.3 ms | 84.5 ms | 54.3 ms | 1.87x -> 2.91x |

The CPU EP's own time changes with the governor. With spinning off, its threads block between
operators, and `ondemand` lowers their clock as well.

The `Conv` claim takes seven of pix2pix's eight encoder convs, and leaves the first on the CPU EP.
Its three input channels make it a poor matmul, at 1.16x the CPU kernel's time on the 2026-10-10
build. Against the `ConvTranspose` claim alone, the two claims together take 0.60-0.65x the time.

On Depth Anything v2 Small, at the `ORT_DISABLE_ALL` the family claim needs, the cost rule claims
16 of the DPT head's 30 convs. The model then runs in 0.931-0.948x the time it takes with the
`Conv` claim off on `performance`, and 0.949-0.969x on `ondemand` [HW sweep 2026-10-29].

#### Evidence

Each figure is a median over five passes with the configuration order and the governor order
rotated, with each ratio paired within its pass. The page cache was dropped and memory compacted
before every configuration. With both claims, the paired speedups span 2.95-3.10x on `performance`
and 2.89-2.94x on `ondemand`. On the 2026-10-10 build at ~816 MHz (the 600 MHz request), the table
measured 146.8, 79.1 and 50.5 ms on `performance` (1.86x -> 2.91x). On `ondemand` it measured 158.0,
87.1 and 58.6 ms (1.81x -> 2.70x), and the two claims took 0.62-0.70x the `ConvTranspose` claim
alone.

The NPU estimate's fit is two passes of `tests/conv_model_bench.c`, 15 calls each after a warm-up.
The estimate before it was fitted at ~816 MHz, and is within 0.91-1.48x of the same calls at 928
MHz.

The Depth Anything ratios are one five-pass A/B of three configurations: the claim off, the current
estimate and the estimate before it. The earlier estimate claimed 10 convs, at 0.937-0.951x on
`performance` and 0.942-0.979x on `ondemand`. Against it, the current estimate measured 0.983-1.001x
and 0.975-1.008x paired, and the two claim the same convs on pix2pix and on RF-DETR nano. On the
2026-10-10 build the earlier estimate measured 0.940-0.965x and 0.952-0.967x.

## Quantized models

The EP consumes RF-DETR models in ONNX Runtime's QDQ format, the `DequantizeLinear`-wrapped
weights that `quantize_static` produces. When the session is created it reads int8, uint8, int4
and uint4 codes, per-tensor or per-channel scales, symmetric or asymmetric zero points, and int32
biases.

The EP declines a QDQ model of any other family. It logs a warning, and the model runs on the CPU
EP.

### Dequantized to fp16

By default the EP dequantizes the weights once, when the session is created, and runs the fp16
path. The weights take the quantized values, and the activations stay fp16. The same `.onnx` runs
with no extra setting, at `ORT_DISABLE_ALL` as for any family claim, and at the fp16 path's speed.

On RF-DETR nano, the int8 model scores COCO mAP 0.443 through the EP on a 500-image slice of
val2017. The fp32 model scores 0.505 on the CPU EP [HW sweep 2026-10-09]. The ~0.06 gap is the
model's own quantization loss. The same int8 model scores 0.408 on the CPU EP. The EP is the more
accurate of the two because it keeps the activations in fp16.

An int4 weight-only model runs the same way. It scores mAP ~0.19, because int4 weights are lossy
for a ViT.

### Native int8

`ROCKET_ORT_INT8=1` runs RF-DETR's encoder projection GEMMs as W8A8: int8 weights, and int8
activations quantized on the host with the model's static scales. These are the patch embedding
and the Q, K, V, attention-output, fc1 and fc2 projections. Attention, the projector and the norms
stay fp16. It needs resident weights, symmetric per-channel int8 weights and int8 activations, and
declines uint8 activations. The other families ignore it.

Its mAP matches the fp16 path's on a 200-image slice, 0.456 against 0.458. On RF-DETR nano it runs
at 1.75-1.80x the fp16 path's latency [HW sweep 2026-10-27]. The run used 4 threads on the A76
cores, spinning off, governor `performance` and NPU 928 MHz (the 900 MHz request). On one image, its
worst score is farther from the fp32 model's than the CPU EP's int8 run, 0.961 against 0.402. The
mean differences are near-equal, 0.00161 and 0.00144. The fp16 path on the same int8 model is nearer
than either, at 0.317 worst and 0.00137 mean [HW sweep 2026-10-27].

Each of its matmuls returns int32 results to the host, which requantizes them. Leave it off for
deployment.

When the int8 projections cannot run, the EP logs a warning and runs the fp16 path. They cannot
run when the int8 weights cannot be marshaled, or when the weights are not resident. The weights
are not resident under `ROCKET_ORT_RESIDENT=0`, or when the resident contexts do not build. With
`ROCKET_ORT_STRICT=1` the EP logs an error and declines the encoder instead. The model then runs on
the CPU EP, and never passes on the fp16 path.

### Producing a quantized model

Produce one with `onnxruntime.quantization.quantize_static`, with `quant_format=QuantFormat.QDQ`
and `per_channel=True`. For native int8, also pass `activation_type=QuantType.QInt8`. An int4
weight-only model takes `weight_type=QuantType.QInt4` and `op_types_to_quantize=["MatMul"]`. The
MatMul-only restriction matters because int4 QDQ ties each int32 bias scale to the coarse int4
weight scale, which rounds small biases to zero.

`tests/test_quant_ep.py` runs an RF-DETR model on the NPU against a CPU EP reference. The CPU EP
quantizes an int8 QDQ model's activations and the EP does not, so for such a model pass
`--fp32 <the unquantized model>`. The EP's scores must then be no farther from the fp32 model's
than the CPU EP's int8 run is.

### Evidence

The native int8 latency is a median over four passes with the configuration order rotated.

## Multi-stream throughput

A single stream is partly host-bound: the host runs the attention softmax and the nodes left on
the CPU EP while the NPU waits. One process per stream therefore adds throughput, and the NPU
pool's best is ~1.5x the single-stream rate on RF-DETR nano and ~1.6x on SAM. The CPU EP gains
little from the same layout, at 0.98-1.10x its own single stream. The comparison below is against
the CPU EP's own best pool.

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), resident weights, CPU governor
`performance`, warm, memory reset before every process. Each EP process runs 4 intra-op threads at
`ORT_DISABLE_ALL` with spinning off. RF-DETR nano runs unpinned (`ROCKET_ORT_PIN=0`), and SAM with
each process on its own A76 core. The CPU EP runs at `ORT_ENABLE_ALL` as 2 processes of 4 threads, 4
of 2, or 8 of 1 [HW sweep 2026-10-27 and 2026-10-29].

**Not characterized:** RF-DETR base, RF-DETR nano above five processes, and SAM's CPU EP at 8
processes of 1 thread, which does not fit in 32 GB.

| Model | NPU pool, best | NPU processes | CPU EP pool, best | CPU EP processes x threads | Against the CPU EP's best pool |
|---|---:|---:|---:|---|---:|
| RF-DETR nano | 7.57-7.61 img/s | 5 | 2.85-3.20 img/s | 8 x 1 | 2.36-2.67x |
| SAM ViT-B image encoder | 0.157-0.164 img/s | 4 | 0.090-0.095 img/s | 4 x 2 | 1.72-1.75x |

SAM's pool peaks at four processes, one per A76 core, and falls to 0.126-0.129 img/s at five.
RF-DETR nano's still rises 1.04-1.06x from four processes to five, the largest pool measured.

[`tools/bench_pool.py`](tools/bench_pool.py) runs both pools and prints the best-over-best ratio.
Its EP worker runs `--ep-threads` intra-op threads, 4 by default, and `--no-spin` turns spinning
off, as in the figures above. Every process loads its own copy of the model, and the tool ends
with an error when the OOM killer takes a worker. It feeds one `.npy` input, so it does not run a
multi-input model such as Laya.

### Host-core placement

The best host-core placement depends on the model:

- **RF-DETR:** leave the EP's host threads unpinned (`ROCKET_ORT_PIN=0`), so that the OS spreads
  them across all eight cores.
- **SAM:** give each process a different A76 core with `ROCKET_CPU_AFFINITY`, such as `4` in the
  first process and `5` in the second. `bench_pool.py --pin-mode affinity` does this.

### Evidence

The pool figures are two passes, each pool size and CPU EP layout in its own process. The order of
the NPU pool and the CPU EP pool is reversed between the passes. Each ratio pairs the two pools of
one pass. Pass 0 ran on 2026-10-27 and pass 1 on 2026-10-29, with no change to the board's governor
or idle settings between them. RF-DETR nano ran 15 iterations per process and SAM 5.

The single-stream rates are from the runs behind the [model families](#model-families)' speedups.
There the EP took 199.2-202.8 ms on RF-DETR nano and 9981-10162 ms on SAM. The CPU EP took 344-345
ms and 10974-11008 ms. The pool's own one-process rows are slower: 3.11-3.33 img/s for RF-DETR nano
unpinned, and 0.0511-0.0512 img/s for SAM on one A76 core. The four-process pools are 2.16-2.33x and
3.06-3.21x those rows.

On July 2026 builds, SAM's NPU pool was 1.44x the CPU EP's best pool. The NPU pool scaled ~1.6x on
RF-DETR and ~1.71x on SAM over one NPU stream.

## Environment variables

The EP reads the `ROCKET_ORT_*` variables below, and `librocketnpu` reads `ROCKET_CPU_AFFINITY`.
Plain `sudo` drops the environment, so run with `sudo -E`. A variable whose default is `0` turns on
for any nonzero integer, and `on` or `true` does not turn it on.

### Encoder families

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_ORT_RESIDENT` | on | Resident weights for the vision families. `0` selects the one-shot path for RF-DETR, SigLIP and SAM. Only the resident encoder runs CLIP and Depth Anything, so with `0` the EP declines those, with a warning. ModernBERT ignores it. |
| `ROCKET_ORT_THREADS` | 3 | NPU worker threads per context, spread across the three NPU cores. Accepts 1-8. A larger value runs as 8, and a value below 1 runs as 3, each with a warning. The per-node claims' context reads it too, and the vision families read it only with resident weights. |
| `ROCKET_ORT_SIGLIP_HEADS` | read from the graph | The attention head count for CLIP, SigLIP and Depth Anything, which overrides the count the EP reads from the attention's `Reshape`. Where that `Reshape` does not name the head dimension, the EP takes `d/64` heads and logs a warning, and the variable is required for a head dimension other than 64. A value of 0 or below means the graph's count. |
| `ROCKET_ORT_MB_HOST` | `0` | ModernBERT only. A nonzero value runs the encoder in host mode, with every GEMM and attention on the CPU, as an off-device check of the EP. The per-node claims still need the NPU. |

### Host threads and placement

`ROCKET_ORT_HT`, `ROCKET_ORT_PIN` and `ROCKET_ORT_AFFINITY_BASE` apply to RF-DETR only. The other
families' host passes run on `librocketnpu`'s threads, which `ROCKET_CPU_AFFINITY` places.

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_ORT_HT` | 4, or 1 when `ROCKET_ORT_THREADS` is 1 | Host threads for RF-DETR's backbone LayerNorm, GELU, bias and residual passes. Accepts 1-8, and ignores any other value. |
| `ROCKET_ORT_PIN` | on | Pins the EP's RF-DETR host threads to the A76 cores. `0` leaves them unpinned, which scales an RF-DETR process pool best. `librocketnpu`'s threads keep their own pinning. |
| `ROCKET_ORT_AFFINITY_BASE` | 0 | A per-process offset into the A76 core list: the EP's host thread `i` pins to core `(base + i)` modulo the list's length. Processes partition the cores only when each runs one host thread (`ROCKET_ORT_HT=1`). It needs `ROCKET_ORT_PIN` on. |
| `ROCKET_CPU_AFFINITY` | auto: the highest-frequency cores, the A76 cores on the RK3588 | `librocketnpu`'s CPU list for its pinned threads, such as `4-7` or `4,5`: the NPU workers, the attention and host workers, and the EP's RF-DETR host threads. Each pinned thread takes one CPU from the list. `off`, `none` or `0` turns pinning off. |

### Per-node claims

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_ORT_CONVTRANSPOSE` | on | `0` leaves every `ConvTranspose` on the CPU EP. |
| `ROCKET_ORT_CONV` | the cost rule | `0` or `off` leaves every `Conv` on the CPU EP, and `all` claims every `Conv` the planner accepts, without the cost rule. Any other value applies the cost rule. |
| `ROCKET_ORT_CONV_MARGIN` | 0.9 | The cost rule's margin. The EP claims a `Conv` when its estimated NPU call plus conversions is at most this fraction of the CPU estimate. Accepts any value above 0. |

### Quantized models

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_ORT_INT8` | `0` | Runs RF-DETR's encoder projections as native W8A8. It needs resident weights. See [Native int8](#native-int8). The other families ignore it. |
| `ROCKET_ORT_STRICT` | `0` | RF-DETR only. With `ROCKET_ORT_INT8`, declines the encoder with an error when the int8 projections cannot run, instead of running the fp16 path with a warning. The model then runs on the CPU EP. It also logs a self-check line at INFO when the session is created. |
| `ROCKET_ORT_I8_MASK` | `0x1F` | Selects which of RF-DETR's GEMMs `ROCKET_ORT_INT8` runs as int8: `0x1` the patch embedding, `0x2` Q, K and V, `0x4` the attention output, `0x8` fc1, `0x10` fc2. A diagnostic. |

### Diagnostics

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_ORT_PROF` | `0` | A nonzero value prints timing to stderr. RF-DETR's backbone prints a per-phase breakdown after every image. Each claimed `ConvTranspose` or `Conv` prints its call count and per-call conversion and NPU times when the session releases it. |

The other families' encoders print their timing through `librocketnpu`'s variables instead:
`ROCKET_SIGLIP_PROF=1` for CLIP, SigLIP and Depth Anything, and `ROCKET_MB_PROF=1` for ModernBERT.

## Implementation notes

- **Structural matching.** Each family is a topology signature. It has an attention and MLP
  walk, the patch-embed and neck convs for a vision family, and LayerScale for RF-DETR and Depth
  Anything.
  A family discriminator completes it, such as SAM's relative-position `Einsum` pair or Depth
  Anything's four-tap DPT exit. The matcher tries ModernBERT, Depth Anything, RF-DETR, SAM, then
  CLIP and SigLIP, and claims the first that matches.
- **Token counts.** `librocketnpu`'s matmul entries refuse a row count that is not a multiple
  of 4. The resident CLIP, SigLIP and Depth Anything encoder pads its token count with zero rows,
  and its attention masks the pad keys. CLIP's 197 tokens become 200, and Depth Anything's 1370
  become 1372.
  RF-DETR's token and patch counts must be multiples of 4, and the EP declines any other geometry.
- **Claim order.** For a family, the EP marshals the weights, opens the NPU and packs the
  resident weights in `GetCapability`, before it claims the encoder. A failed step there leaves the
  encoder on the CPU EP, with a warning. ONNX Runtime cannot return a claimed node to the CPU EP
  once `Compile` runs, so a failure there would fail session creation instead. `Compile` only maps
  the fused node's inputs and outputs onto the prepared state.
- **Concurrency.** Concurrent `Run` calls on one session are serialized. A family claim holds a
  lock for its whole computation, and a session's per-node claims share another. For parallel
  streams, run one process per stream, or one session per thread.
- **Tests.** Each script in `tests/` is standalone and takes the built library, a model and an
  input on its command line. No model is included.
