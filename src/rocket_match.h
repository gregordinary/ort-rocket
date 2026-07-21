// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_match -- structural (op-topology) matcher for the RF-DETR DINOv2 encoder + CSP projector
// subgraph. It identifies the offloadable subgraph and every weight-bearing node by GRAPH TOPOLOGY
// and edge-walking, not by exporter-generated tensor/node names, so the EP survives an RF-DETR
// re-export (torch/opset/version bump renames the interior tensors and the anonymous projection
// initializers). The session must still run with graph optimization DISABLED: ONNX Runtime's
// Gemm/FusedMatMul/Gelu/SkipLayerNorm fusions rewrite this block into a different op topology (even
// ORT_ENABLE_BASIC fuses every MatMul+Add into Gemm), which this matcher -- keyed on the raw
// exported ops -- does not model.
//
// The match is validated equivalent to the previous name-based marshaling on all four shipping
// exports (nano/base fp32, nano int8/int4 QDQ): the resolved weight tensor for every role is the
// same edge, and the claimed node set is byte-identical to the old hardcoded-seed ancestor set.
//
// One encoder layer's raw topology (the pattern this matches), repeated per layer:
//   LayerNormalization -> q,k,v (MatMul+Add) -> [reshape/transpose + a computed d_head^-0.25 scale
//   Mul on Q and K] -> scores MatMul -> Softmax -> context MatMul(.,V) -> o (MatMul+Add) ->
//   LayerScale Mul -> residual Add -> LayerNormalization -> fc1 (MatMul+Add) -> erf-GELU -> fc2
//   (MatMul+Add) -> LayerScale Mul -> residual Add. QDQ models interleave Quantize/Dequantize pairs,
//   which the walks treat as transparent layout ops.

#ifndef ORT_ROCKET_MATCH_H
#define ORT_ROCKET_MATCH_H

#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

namespace rocket_match {

// One matched encoder layer: the weight-bearing nodes, in rocket-datapath roles. Weights are read by
// the EP from these nodes' initializer edges (following a DequantizeLinear for a QDQ model).
struct MatchedLayer {
  const OrtNode* q = nullptr;     // query   projection MatMul  (weight [D,D])
  const OrtNode* k = nullptr;     // key     projection MatMul  (weight [D,D])
  const OrtNode* v = nullptr;     // value   projection MatMul  (weight [D,D])
  const OrtNode* o = nullptr;     // output-dense  MatMul       (weight [D,D])
  const OrtNode* fc1 = nullptr;   // mlp fc1  MatMul            (weight [D,DFF])
  const OrtNode* fc2 = nullptr;   // mlp fc2  MatMul            (weight [DFF,D])
  const OrtNode* bq = nullptr;    // bias Add feeding q  (bias [D])
  const OrtNode* bk = nullptr;
  const OrtNode* bv = nullptr;
  const OrtNode* bo = nullptr;
  const OrtNode* bf1 = nullptr;   // mlp fc1 bias Add (bias [DFF])
  const OrtNode* bf2 = nullptr;   // mlp fc2 bias Add (bias [D])
  const OrtNode* norm1 = nullptr; // LayerNormalization feeding q/k/v (gamma/beta = input 1/2)
  const OrtNode* norm2 = nullptr; // LayerNormalization feeding fc1
  const OrtNode* ls1 = nullptr;   // LayerScale Mul after o   (lambda1 = its initializer operand)
  const OrtNode* ls2 = nullptr;   // LayerScale Mul after fc2 (lambda2 = its initializer operand)
  // SAM ViT-Det (Family::SamVitDet) extras -- null/false for the DINOv2 and plain-ViT families.
  const OrtNode* rel_h = nullptr; // decomposed rel-pos height Einsum ("bhwc,hkc->bhwk"); its constant
                                  // operand is the resolved rel_pos_h table [S,S,dhead]
  const OrtNode* rel_w = nullptr; // decomposed rel-pos width Einsum ("bhwc,wkc->bhwk")
  bool fused_qkv = false;         // q==k==v is ONE fused qkv MatMul (weight [3D,D]); SAM splits it into
                                  // q/k/v downstream, so all three roles point at the same node
  bool windowed = false;          // windowed attention (window_partition) this layer; false == global
  // Fused-qkv column blocks, for a family whose fused weight is NOT in q,k,v order. The exported
  // MatMul weight is [D, 3D]; block b spans columns [b*D, (b+1)*D). Depth Anything v2's onnxsim
  // fold emits HF DINOv2's source order (key, value, query), so these are DERIVED by walking each
  // role back to the Split output it came from -- never assumed. -1 means "not a split-block
  // family" (SAM feeds its whole [3D,D] weight through and splits it inside the encoder).
  int qkv_blk_q = -1, qkv_blk_k = -1, qkv_blk_v = -1;
};

// The backbone family a matcher recognizes. MatchAny tries the registered matchers in order and
// tags the winning match with its family, so Compile can select the right marshaling path. Extend
// this as new families land (SAM, ...).
enum class Family { None, DinoV2RfDetr, SiglipVit, SamVitDet, DepthAnythingDpt };

// The whole structural match. `ok` is false with `reason` set when the graph does not present the
// expected topology -- GetCapability then claims nothing (full CPU fallback) rather than claiming a
// subgraph Compile would fail to marshal.
struct BackboneMatch {
  bool ok = false;
  Family family = Family::None;       // which registered family matched (set by MatchAny)
  std::string reason;                 // human-readable miss reason (logged when !ok)
  const OrtNode* patch_conv = nullptr;  // stem patch-embed Conv (weight [D,IC,P,P], bias [D])
  int patch_size = 0;                 // Conv kernel/stride (16 nano, 14 base) -> selects the geometry
  std::vector<MatchedLayer> layers;   // ordered by graph position == encoder layer index
  std::string exit_tensor;            // claim seed: the ancestor set of this tensor is the offloaded
                                      // subgraph. DINOv2: the CSP-projector output. SiglipVit: the
                                      // encoder exit -- the post-LayerNorm output for SigLIP, or the
                                      // last residual (== last_hidden_state) for CLIP (no seq post-LN).
                                      // Empty in a fused subgraph where that tensor is the graph output.
  // Plain-ViT (SiglipVit) extras -- the DINOv2 path leaves these null/empty. SigLIP and CLIP share
  // this family; the CLIP-only fields (pre_ln, cls_embed, quick_gelu) stay null/empty/false for SigLIP.
  const OrtNode* post_ln = nullptr;   // encoder final LayerNormalization over the sequence (SigLIP);
                                      // null for CLIP, whose post_layernorm applies only to the pooled
                                      // cls row on the host tail, so last_hidden_state is un-normed.
  std::string pos_embed;              // position-embedding initializer ([ntok,d]); ntok = L(+cls) rows
  const OrtNode* pre_ln = nullptr;    // pre-encoder LayerNormalization (CLIP's pre_layrnorm); null for
                                      // SigLIP. Sits between the stem embeddings Add and layer 0.
  std::string cls_embed;              // class/cls embedding initializer ([n_prefix*d]); empty for SigLIP.
                                      // Present => a prefix (cls) token is prepended; pos has +n_prefix rows.
  bool quick_gelu = false;            // MLP nonlinearity is quick-GELU x*sigmoid(1.702x) (CLIP) vs the
                                      // gelu-tanh SigLIP trains with. Detected from the fc1->fc2 chain.
  // DPT multi-tap exit (DepthAnythingDpt). A dense-prediction head consumes SEVERAL intermediate
  // encoder outputs rather than the final one, so the claim has several seeds and the fused node
  // several outputs. Empty for every single-exit family, where `exit_tensor` is the lone seed.
  std::vector<std::string> tap_tensors;   // per tap, the normalizing LayerNorm's output tensor
  std::vector<const OrtNode*> tap_norms;  // per tap, that LayerNormalization node (gamma/beta edges)
  std::vector<int> tap_layers;            // per tap, the encoder layer index it reads (ascending)
};

// The DINOv2 + CSP-projector (RF-DETR) family matcher: run the structural match over `graph` (the
// full model in GetCapability, or the fused subgraph in Compile). Never allocates weights or touches
// the NPU; pure graph analysis. Callers should prefer MatchAny -- it also tags the family.
BackboneMatch MatchBackbone(const OrtApi& api, const OrtGraph* graph);

// The plain-ViT (CLIP/SigLIP) family matcher: a pre-norm ViT encoder with absolute position
// embeddings and NO LayerScale, whose tail is a pooled embedding rather than a CSP projector. It is
// the DINOv2 walk relaxed -- no LayerScale Mul before each residual, and the SigLIP MAP
// attention-pooling head excluded (its query is a learned probe, so its attention has no
// pre-attention LayerNorm). It matches both SigLIP and CLIP, resolving the CLIP variant structurally:
// a prepended cls token (.cls_embed set, pos has +1 row), a pre-encoder LayerNorm (.pre_ln set),
// quick-GELU (.quick_gelu set), and an un-normed last_hidden_state exit (.post_ln null, since CLIP's
// post_layernorm applies only to the pooled cls row -- left on the host tail). SigLIP leaves those at
// null/empty/false and exits at the sequence post-LayerNorm. Pure graph analysis; prefer MatchAny.
BackboneMatch MatchPlainViT(const OrtApi& api, const OrtGraph* graph);

// The SAM ViT-Det (Family::SamVitDet) image-encoder matcher: a pre-norm ViT whose attention adds a
// decomposed relative-position bias to the QK^T scores (two Einsum nodes -- the SAM signature no
// other family emits), uses a FUSED qkv Linear (one MatMul split into q/k/v), and runs windowed
// attention on most layers with periodic global layers. There is no cls token and no LayerScale; the
// exit is the conv neck (conv1x1 -> LN -> conv3x3 -> LN) output == last_hidden_state. Per matched
// layer it records rel_h/rel_w (the rel-pos Einsums), fused_qkv (q==k==v), and windowed. The stem
// pos_embed (a 2D [1,H,W,d] initializer added after the patch conv) lands in .pos_embed. Pure graph
// analysis; prefer MatchAny.
BackboneMatch MatchSamVitDet(const OrtApi& api, const OrtGraph* graph);

// The Depth Anything v2 (Family::DepthAnythingDpt) matcher: a LayerScale DINOv2 ViT whose exit is a
// DPT dense-prediction head. Structurally it is the RF-DETR backbone's own family, so this matcher
// is as much DISCRIMINATION as resolution, and it must be tried BEFORE MatchBackbone -- which misses
// this graph only by accident (its layout-only scores back-walk does not cross DA v2's
// Mul(QK^T, 1/sqrt(dh))), and would otherwise claim all 12 layers plus the entire DPT head as a
// "projector exit". The discriminator is the multi-tap exit: the head consumes FOUR intermediate
// encoder outputs, each through its own LayerNormalization and a cls-stripping Slice, where
// RF-DETR's CSP projector consumes one. Per layer it records the FUSED qkv MatMul (an onnxsim fold
// of HF DINOv2's three Linears) with the per-role column block DERIVED from the Split, since the
// exported order is (key, value, query), not (q,k,v). The tap tensors/norms/layers land in
// .tap_tensors/.tap_norms/.tap_layers and are the claim seeds. Pure graph analysis; prefer MatchAny.
BackboneMatch MatchDepthAnythingDpt(const OrtApi& api, const OrtGraph* graph);

// The matcher registry entry point: try every registered family matcher in order, returning the
// first that matches (with .family set) or a combined-reason miss (.ok = false) if none do. This is
// the seam a new target hooks -- register its signature; the GetCapability/Compile call sites are
// unchanged.
BackboneMatch MatchAny(const OrtApi& api, const OrtGraph* graph);

}  // namespace rocket_match

#endif  // ORT_ROCKET_MATCH_H
