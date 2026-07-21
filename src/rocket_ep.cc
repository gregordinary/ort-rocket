// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// ort-rocket: an ONNX Runtime plugin execution provider that offloads transformer
// vision-model encoders to Rockchip NPUs via the mainline rocket driver.
//
// A matcher registry (rocket_match) recognizes five model families by graph topology --
// RF-DETR (DINOv2 backbone + CSP projector), CLIP and SigLIP (plain ViT), SAM (ViT-Det),
// and Depth Anything v2 (DINOv2 backbone + DPT taps). GetCapability claims the first family
// whose signature matches; Compile marshals that family's initializer weights (projections
// transposed to rocket [out,in]; fp32 -> fp16; LayerScale folded into Wo/bo,Wf2/bf2 where
// present); Compute runs the encoder on the NPU. Everything a family does not claim -- the
// deformable decoder and DETR postproc, the pooling heads, the SAM prompt/mask decoder, the
// DPT depth head -- is not claimed and falls back to ORT's CPU kernels.
//
// Each family's compute is graded bit-faithful to the ORT CPU EP off-device before it claims
// on device. This file is the ORT plumbing: partition, weight marshaling, and kernel-context
// I/O; the on-NPU datapaths live in rocket_backbone / rocket_projector (RF-DETR) and the
// librocketnpu SigLIP and SAM encoders (the other families).
//
// Subgraphs and weight-bearing nodes are located by GRAPH TOPOLOGY (rocket_match), not by
// exporter-generated tensor/node names, so a re-export (torch/opset/version bump) that renames
// the interior still matches. Per-family geometry (patch size, token/layer counts, width) is
// read from the graph. The session must run with graph optimization DISABLED -- ORT's
// Gemm/FusedMatMul/Gelu/SkipLayerNorm fusions rewrite the encoder into a contrib-op topology the
// matcher (keyed on the raw exported ops) does not model; supporting graph-opt-on is a separate,
// larger effort.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_c_api.h"

#include "rocket_npu.h"
#include "rocket_backbone.h"
#include "rocket_projector.h"
#include "rocket_sam.h"
#include "rocket_siglip.h"
#include "rocket_match.h"

namespace {

// The claimed subgraph is the ANCESTOR set of the projector output tensor: convex by construction
// (if U is on a path between two ancestors of a tensor X, U is itself an ancestor of X), with exactly
// that tensor out and the image in. It subsumes the whole DINOv2 encoder, the backbone->projector
// de-window glue (Slice/Reshape/Transpose), and the CSP projector, so one NPU fused node spans
// backbone + projector with no mid-graph CPU hop. The seed tensor and every weight-bearing node are
// located by GRAPH TOPOLOGY (rocket_match), not by name, so the claim survives an RF-DETR re-export
// that renames the interior tensors and the anonymous projection initializers. The session must still
// run with graph optimization DISABLED: ORT's Gemm/FusedMatMul/Gelu/SkipLayerNorm rewrites replace
// this block with a different op topology the matcher (keyed on the raw exported ops) does not model.

inline void DiscardStatus(const OrtApi& api, OrtStatus* status) {
  if (status != nullptr) api.ReleaseStatus(status);
}

// ---------------------------------------------------------------------------
// Compiled backbone+projector state: owns the fp16 weights and the rocket fd. Built once
// in Compile, pointed at by the OrtNodeComputeInfo, freed in ReleaseState.
// ---------------------------------------------------------------------------
struct CompiledBackbone {
  const OrtApi& api;
  const OrtLogger* logger = nullptr;
  int fd = -1;
  std::vector<std::vector<_Float16>> owned;   // backing storage for every weight
  rocket_backbone_weights w{};
  rocket_projector_weights pw{};              // the CSP projector's fp16 weights
  // Optional native-int8 encoder projections (ROCKET_ORT_INT8): raw int8 codes + per-out-channel
  // scales, in addition to the always-marshaled fp16 weights. cb->w.i8 points at i8w when live.
  rocket_backbone_int8 i8w{};
  std::vector<std::vector<int8_t>>  i8_codes;
  std::vector<std::vector<float>>   i8_scales;
  std::vector<std::vector<int32_t>> i8_rowsums;   // per-out-channel sum_in(w) for the zp fold
  int8_t*  alloc_i8(size_t n) { i8_codes.emplace_back(n);   return i8_codes.back().data(); }
  float*   alloc_sc(size_t n) { i8_scales.emplace_back(n);  return i8_scales.back().data(); }
  int32_t* alloc_rs(size_t n) { i8_rowsums.emplace_back(n); return i8_rowsums.back().data(); }
  // Resident (prepacked, multicore) contexts: weights packed into NPU BOs once at Compile
  // and reused across every image. NULL => the one-shot fallback (ROCKET_ORT_RESIDENT=0, or
  // a ctx-create failure). The ctxs read w/pw's bias/gamma pointers per call, so `owned`
  // must outlive them -- it does (same object).
  rocket_backbone_ctx* bctx = nullptr;
  rocket_projector_ctx* pctx = nullptr;
  // Plain-ViT (CLIP/SigLIP) family (Family::SiglipVit). When is_siglip, the DINOv2 backbone+projector
  // fields above are unused; the encoder runs through rocket_siglip instead. `sm` holds the geometry +
  // fp16 weight pointers (into `owned`); `sctx` is the resident ctx (NULL => one-shot rocket_siglip_encode).
  bool is_siglip = false;
  rocket_siglip_model sm{};
  rocket_siglip_ctx* sctx = nullptr;
  // SAM ViT-Det family (Family::SamVitDet). When is_sam, the encoder runs through rocket_sam; the
  // DINOv2/projector and siglip fields above are unused. `sm2` holds geometry + fp16 weight pointers
  // (into `owned`); `sctx2` is the resident ctx (NULL => the one-shot rocket_sam_encode fallback).
  bool is_sam = false;
  rocket_sam_model sm2{};
  rocket_sam_ctx* sctx2 = nullptr;
  // Depth Anything v2 (Family::DepthAnythingDpt). The encoder is the plain-ViT one, so this reuses
  // `sm`/`sctx`; what differs is the exit -- the fused node emits one tensor per DPT tap instead of
  // one. tap_out_slot[t] is the ORT output index carrying tap t, resolved at Compile by matching the
  // fused node's output names against the matcher's tap tensors (never positional guesswork).
  bool is_depth = false;
  std::vector<size_t> tap_out_slot;
  // The fused node lists every initializer as an input; the image is the one
  // non-initializer input. Its index, found at Compile, is what Compute reads.
  size_t image_in_idx = 0;

  // Every OrtNodeComputeInfo::CreateState hands out this same shared state, so all concurrent
  // Run()s on one session drive the one bctx/pctx, the one rocket fd, and one set of ctx-owned
  // per-image scratch buffers. Those contexts are not thread-safe (each rewrites shared per-shape
  // scratch/BOs per call -- see the librocketnpu ctx contracts and rocket_backbone.c). ONNX
  // Runtime permits concurrent Run() on a session, so a caller that wraps one session in a
  // thread pool would race. Hold this across the whole ComputeImpl body to serialize the NPU
  // work per session; the lock is negligible next to the dispatch. The benchmarked deployment is
  // process-per-stream (one session, serial Runs), where this never contends. For real
  // concurrency use process-per-stream or one session per thread.
  std::mutex run_mtx;

  // Per-Compute host scratch, reused across images. Single-Run-per-session (run_mtx) makes
  // this safe. Sized lazily on the first Compute from the geometry; the resize is a no-op on every
  // subsequent call, so steady-state inference does no per-image heap traffic. Previously each
  // Compute malloc/freed several MB (image copy + 4 feature taps + 4 de-windowed taps + output).
  std::vector<_Float16> s_img;                 // [IC*img*img] fp16 image copy
  std::vector<_Float16> s_feat[RBB_NTAP];      // [ntok*D] per feature tap
  std::vector<_Float16> s_tap[RPJ_NTAP];       // [np*RPJ_DIN] per de-windowed tap
  std::vector<_Float16> s_y;                   // [np*RPJ_CO] projector output (token-major)
  std::vector<_Float16> s_out;                 // [L*d] SigLIP encoder output (for fp32-output widening)

  explicit CompiledBackbone(const OrtApi& a) : api(a) {}
  ~CompiledBackbone() {
    if (bctx) rocket_backbone_ctx_free(bctx);
    if (pctx) rocket_projector_ctx_free(pctx);
    if (sctx) rocket_siglip_ctx_free(sctx);
    if (sctx2) rocket_sam_ctx_free(sctx2);
    if (fd >= 0) rocket_close(fd);
  }

  _Float16* alloc(size_t n) {
    owned.emplace_back(n);
    return owned.back().data();
  }
};

// ---------------------------------------------------------------------------
// OrtEp instance.
// ---------------------------------------------------------------------------
struct RocketEp : OrtEp {
  RocketEp(const OrtApi& api, std::string ep_name, const OrtLogger& lg)
      : OrtEp{}, ort_api(api), name(std::move(ep_name)), logger(lg) {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  }

  const OrtApi& ort_api;
  std::string name;
  const OrtLogger& logger;

  void Log(OrtLoggingLevel lvl, const std::string& msg) const {
    DiscardStatus(ort_api, ort_api.Logger_LogMessage(&logger, lvl, msg.c_str(),
                                                     ORT_FILE, __LINE__, __FUNCTION__));
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* p) noexcept {
    return static_cast<const RocketEp*>(p)->name.c_str();
  }

  // Claim the DINOv2 encoder subgraph (the ancestor set of the 4 feature tensors) as one
  // fused node. Convex by construction; the boundary is the image in and the 4 features out.
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* support_info) noexcept {
    auto* ep = static_cast<RocketEp*>(this_ptr);
    const OrtApi& api = ep->ort_api;
    const OrtEpApi& ep_api = *api.GetEpApi();

    // Structural match: locate the encoder+projector by op topology (no name dependency), trying
    // each registered backbone family in turn (MatchAny). On any miss -- not a recognized graph, or
    // a graph whose optimizations were left on so the block was fused away -- claim nothing and let
    // ORT place everything on its CPU kernels.
    rocket_match::BackboneMatch match = rocket_match::MatchAny(api, graph);
    if (!match.ok) {
      ep->Log(ORT_LOGGING_LEVEL_INFO,
              std::string("ort-rocket: not claiming (") + match.reason +
                  "); full CPU fallback. This EP offloads the encoder of RF-DETR, CLIP/SigLIP, SAM, and Depth Anything v2; run "
                  "the session with graph_optimization_level=ORT_DISABLE_ALL so the exported op "
                  "topology survives.");
      return nullptr;
    }
    // Durable: family-specific pre-claim validation, so a partial/unexpected match degrades to
    // CPU here rather than hard-failing session creation in Compile (the EP API has no way to
    // un-claim once Compile runs). DINOv2 selects an RF-DETR variant by patch size and requires the
    // exact encoder-layer count; plain-ViT (SigLIP) claims its encoder directly (the ancestor BFS
    // below is family-agnostic -- it seeds from match.exit_tensor). Patch 16 collides between the two
    // families, so this MUST branch on family, not patch size.
    if (match.family == rocket_match::Family::DinoV2RfDetr) {
      const rocket_backbone_geom* geom = rocket_backbone_geom_select(match.patch_size);
      if (!geom || match.layers.size() != static_cast<size_t>(geom->nlayer)) {
        ep->Log(ORT_LOGGING_LEVEL_WARNING,
                "ort-rocket: matched a DINOv2-shaped subgraph (patch " + std::to_string(match.patch_size) +
                    ", " + std::to_string(match.layers.size()) + " encoder layers) that does not fit a "
                    "known RF-DETR variant; not claiming (full CPU fallback).");
        return nullptr;
      }
    }
    // SAM ViT-Det claims like plain-ViT: the ancestor BFS below is family-agnostic (it seeds from
    // match.exit_tensor -- the conv-neck output). Compile marshals it into the rocket_sam encoder.
    const bool is_siglip = match.family == rocket_match::Family::SiglipVit;
    const bool is_sam = match.family == rocket_match::Family::SamVitDet;

    size_t num_nodes = 0;
    if (OrtStatus* st = api.Graph_GetNumNodes(graph, &num_nodes)) return st;
    std::vector<const OrtNode*> nodes(num_nodes);
    if (OrtStatus* st = api.Graph_GetNodes(graph, nodes.data(), num_nodes)) return st;

    // producer[tensor name] -> node index; node_inputs[i] -> its input tensor names.
    std::unordered_map<std::string, size_t> producer;
    std::vector<std::vector<std::string>> node_inputs(num_nodes);
    auto value_names = [&](const OrtValueInfo* const* vis, size_t n, std::vector<std::string>* out) -> OrtStatus* {
      for (size_t k = 0; k < n; k++) {
        if (!vis[k]) continue;   // omitted optional edge -> null OrtValueInfo* (GetValueInfoName segfaults)
        const char* nm = nullptr;
        if (OrtStatus* st = api.GetValueInfoName(vis[k], &nm)) return st;
        if (nm) out->emplace_back(nm);
      }
      return nullptr;
    };
    for (size_t i = 0; i < num_nodes; i++) {
      size_t no = 0, ni = 0;
      if (OrtStatus* st = api.Node_GetNumOutputs(nodes[i], &no)) return st;
      std::vector<const OrtValueInfo*> outs(no);
      if (OrtStatus* st = api.Node_GetOutputs(nodes[i], outs.data(), no)) return st;
      std::vector<std::string> onames;
      if (OrtStatus* st = value_names(outs.data(), no, &onames)) return st;
      for (auto& s : onames) producer.emplace(s, i);
      if (OrtStatus* st = api.Node_GetNumInputs(nodes[i], &ni)) return st;
      std::vector<const OrtValueInfo*> ins(ni);
      if (OrtStatus* st = api.Node_GetInputs(nodes[i], ins.data(), ni)) return st;
      if (OrtStatus* st = value_names(ins.data(), ni, &node_inputs[i])) return st;
    }

    // BFS backward from the structurally-found exit tensor(s), collecting all ancestor nodes. A
    // single-exit family seeds one tensor; a DPT head (DepthAnythingDpt) seeds all of its taps. The
    // union of ancestor sets is still ancestor-closed, hence still convex, so the multi-seed claim
    // is as safe a partition as the single-seed one -- it just leaves a fused node with one output
    // per tap instead of one.
    std::vector<char> claimed_flag(num_nodes, 0);
    std::vector<std::string> stack;
    std::unordered_map<std::string, char> visited;
    if (!match.tap_tensors.empty())
      for (const std::string& t : match.tap_tensors) stack.emplace_back(t);
    else
      stack.emplace_back(match.exit_tensor);
    while (!stack.empty()) {
      std::string t = std::move(stack.back());
      stack.pop_back();
      if (visited.count(t)) continue;
      visited.emplace(t, 1);
      auto it = producer.find(t);
      if (it == producer.end()) continue;    // initializer or graph input: stop
      size_t ni = it->second;
      claimed_flag[ni] = 1;
      for (const std::string& in : node_inputs[ni])
        if (!visited.count(in)) stack.push_back(in);
    }

    std::vector<const OrtNode*> claimed;
    for (size_t i = 0; i < num_nodes; i++)
      if (claimed_flag[i]) claimed.push_back(nodes[i]);

    if (claimed.empty()) {   // exit tensor had no producer in this graph -- should not happen post-match
      ep->Log(ORT_LOGGING_LEVEL_WARNING, "ort-rocket: matched but claimed no nodes; full CPU fallback.");
      return nullptr;
    }
    const char* fam_desc =
        match.family == rocket_match::Family::DepthAnythingDpt
            ? "Depth Anything v2 DINOv2 encoder, exit the DPT taps (head stays on host)"
        : is_sam     ? "SAM ViT-Det encoder (windowed + rel-pos), exit neck output"
        : is_siglip  ? "plain-ViT encoder, exit last_hidden_state"
                     : "DINOv2 backbone+projector";
    ep->Log(ORT_LOGGING_LEVEL_INFO,
            "ort-rocket: claiming " + std::to_string(claimed.size()) + " nodes (" + fam_desc +
                ", patch " + std::to_string(match.patch_size) + ", " +
                std::to_string(match.layers.size()) + " layers) as one fused node");
    return ep_api.EpGraphSupportInfo_AddNodesToFuse(support_info, claimed.data(),
                                                    claimed.size(), nullptr);
  }

  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                             const OrtNode** fused_nodes, size_t count,
                                             OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** /*ep_context_nodes*/) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                       OrtNodeComputeInfo** infos,
                                                       size_t count) noexcept;
};

// ---------------------------------------------------------------------------
// Weight marshaling helpers (Compile time).
// ---------------------------------------------------------------------------
struct GraphMaps {
  std::unordered_map<std::string, const OrtValue*> init;    // initializer name -> value
  std::unordered_map<std::string, const OrtNode*> producer; // tensor name -> producing node
  // A quantized model (QDQ) replaces each fp32 weight X with an int8 initializer X_quantized
  // fed through a DequantizeLinear into the original consumer. This maps the DQ's quantized
  // input (X_quantized) -> the DQ node, so a weight can be resolved by name (X_quantized) or
  // by walking the DQ that feeds a MatMul/Add. Empty for an fp32 model.
  std::unordered_map<std::string, const OrtNode*> dq_by_q;  // DQ input[0] name -> DQ node
};

// A node's operator type ("MatMul", "DequantizeLinear", ...). Returns "" on error.
static std::string NodeOp(const OrtApi& api, const OrtNode* n) {
  const char* op = nullptr;
  if (api.Node_GetOperatorType(n, &op) != nullptr || !op) return std::string();
  return std::string(op);
}

// A node's input tensor names.
static OrtStatus* NodeInputNames(const OrtApi& api, const OrtNode* n, std::vector<std::string>* out) {
  size_t ni = 0;
  if (OrtStatus* st = api.Node_GetNumInputs(n, &ni)) return st;
  std::vector<const OrtValueInfo*> ins(ni);
  if (OrtStatus* st = api.Node_GetInputs(n, ins.data(), ni)) return st;
  for (const OrtValueInfo* vi : ins) {
    if (!vi) { out->emplace_back(""); continue; }   // omitted optional input -> null (would segfault); keep the slot
    const char* nm = nullptr;
    if (OrtStatus* st = api.GetValueInfoName(vi, &nm)) return st;
    out->emplace_back(nm ? nm : "");
  }
  return nullptr;
}

// Build name->initializer, name->node, tensor->producer, and DQ-quantized-input->DQ maps for
// the fused subgraph.
static OrtStatus* BuildMaps(const OrtApi& api, const OrtGraph* g, GraphMaps& m) {
  size_t ni = 0;
  if (OrtStatus* st = api.Graph_GetNumInitializers(g, &ni)) return st;
  std::vector<const OrtValueInfo*> inits(ni);
  if (OrtStatus* st = api.Graph_GetInitializers(g, inits.data(), ni)) return st;
  for (const OrtValueInfo* vi : inits) {
    if (!vi) continue;
    const char* nm = nullptr;
    if (OrtStatus* st = api.GetValueInfoName(vi, &nm)) return st;
    const OrtValue* val = nullptr;
    if (OrtStatus* st = api.ValueInfo_GetInitializerValue(vi, &val)) return st;
    if (nm && val) m.init.emplace(nm, val);
  }
  size_t nn = 0;
  if (OrtStatus* st = api.Graph_GetNumNodes(g, &nn)) return st;
  std::vector<const OrtNode*> nodes(nn);
  if (OrtStatus* st = api.Graph_GetNodes(g, nodes.data(), nn)) return st;
  for (const OrtNode* n : nodes) {
    // producer map: every output tensor -> this node
    size_t no = 0;
    if (OrtStatus* st = api.Node_GetNumOutputs(n, &no)) return st;
    std::vector<const OrtValueInfo*> outs(no);
    if (OrtStatus* st = api.Node_GetOutputs(n, outs.data(), no)) return st;
    for (const OrtValueInfo* vi : outs) {
      if (!vi) continue;
      const char* on = nullptr;
      if (OrtStatus* st = api.GetValueInfoName(vi, &on)) return st;
      if (on) m.producer.emplace(on, n);
    }
    // DQ index: quantized-input name -> the DequantizeLinear node
    if (NodeOp(api, n) == "DequantizeLinear") {
      std::vector<std::string> in;
      if (OrtStatus* st = NodeInputNames(api, n, &in)) return st;
      if (!in.empty()) m.dq_by_q.emplace(in[0], n);
    }
  }
  return nullptr;
}

// Raw initializer accessor: element type, count, dims, and a data pointer (valid while the
// graph lives). Used both for fp32 weights and for the int8 quantized/scale/zero_point trio.
struct RawInit {
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  const void* data = nullptr;
  size_t count = 0;
  std::vector<int64_t> dims;
};
static OrtStatus* ReadInit(const OrtApi& api, const GraphMaps& m, const std::string& name, RawInit* r) {
  auto it = m.init.find(name);
  if (it == m.init.end())
    return api.CreateStatus(ORT_NOT_FOUND, ("ort-rocket: initializer not found: " + name).c_str());
  const OrtValue* v = it->second;
  OrtTensorTypeAndShapeInfo* info = nullptr;
  if (OrtStatus* st = api.GetTensorTypeAndShape(v, &info)) return st;
  OrtStatus* st = api.GetTensorElementType(info, &r->et);
  if (!st) st = api.GetTensorShapeElementCount(info, &r->count);
  size_t nd = 0;
  if (!st) st = api.GetDimensionsCount(info, &nd);
  if (!st) { r->dims.resize(nd); st = api.GetDimensions(info, r->dims.data(), nd); }
  api.ReleaseTensorTypeAndShapeInfo(info);
  if (st) return st;
  return api.GetTensorData(v, &r->data);
}

// Read a scalar int64 node attribute (e.g. DequantizeLinear 'axis'); returns `def` if absent.
static int64_t ReadIntAttr(const OrtApi& api, const OrtNode* n, const char* name, int64_t def) {
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return def; }
  if (!attr) return def;
  int64_t v = def;
  size_t out = 0;
  st = api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_INT, &v, sizeof(v), &out);
  if (st) { api.ReleaseStatus(st); return def; }
  return v;
}

// Read a scalar float node attribute (e.g. LayerNormalization 'epsilon'); returns `def` if absent.
static float ReadFloatAttr(const OrtApi& api, const OrtNode* n, const char* name, float def) {
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return def; }
  if (!attr) return def;
  float v = def;
  size_t out = 0;
  st = api.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_FLOAT, &v, sizeof(v), &out);
  if (st) { api.ReleaseStatus(st); return def; }
  return v;
}

// Read a quantized value as an int, at flat (logical) index i. int32 is the QDQ bias
// representation (scale = in_scale*weight_scale); int4/uint4 (weight-only) are packed
// two-per-byte, element 2k in the low nibble and 2k+1 in the high nibble (the ONNX convention),
// int4 sign-extended from 4 bits.
static inline int QAt(const RawInit& r, size_t i) {
  switch (r.et) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:  return static_cast<const int8_t*>(r.data)[i];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return static_cast<const int32_t*>(r.data)[i];
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4: {
      uint8_t b = static_cast<const uint8_t*>(r.data)[i >> 1];
      int v = (i & 1) ? (b >> 4) : (b & 0x0F);
      return (v & 0x8) ? v - 16 : v;            // sign-extend 4-bit
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4: {
      uint8_t b = static_cast<const uint8_t*>(r.data)[i >> 1];
      return (i & 1) ? (b >> 4) : (b & 0x0F);
    }
    default: return static_cast<const uint8_t*>(r.data)[i];   // UINT8
  }
}

// Dequantize a DequantizeLinear node (q_init, scale, optional zero_point) into fp32 `out`.
// Handles per-tensor (scalar scale) and per-channel (scale vector along the DQ 'axis') int8/
// uint8 weights: out[i] = (q[i] - zp[c]) * scale[c]. This reconstructs the fp16-datapath weight
// from a QDQ model faithfully (weights are symmetric per-channel for QInt8, but zp is applied
// generally so an asymmetric weight quant would work too).
static OrtStatus* DequantDQ(const OrtApi& api, const GraphMaps& m, const OrtNode* dq,
                            std::vector<float>* out) {
  std::vector<std::string> in;
  if (OrtStatus* st = NodeInputNames(api, dq, &in)) return st;
  if (in.size() < 2)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: DequantizeLinear missing scale");
  RawInit q, s, z;
  if (OrtStatus* st = ReadInit(api, m, in[0], &q)) return st;
  if (OrtStatus* st = ReadInit(api, m, in[1], &s)) return st;
  bool has_z = in.size() >= 3 && !in[2].empty() && m.init.count(in[2]);
  if (has_z) { if (OrtStatus* st = ReadInit(api, m, in[2], &z)) return st; }
  if (q.et != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 && q.et != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 &&
      q.et != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && q.et != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 &&
      q.et != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4)
    return api.CreateStatus(ORT_INVALID_GRAPH,
        "ort-rocket: unsupported quantized weight dtype (expected int8/uint8/int4/uint4/int32)");
  const float* sc = static_cast<const float*>(s.data);
  out->resize(q.count);
  if (s.count <= 1) {                                   // per-tensor
    float sf = sc[0];
    int zp = has_z ? QAt(z, 0) : 0;
    for (size_t i = 0; i < q.count; i++) (*out)[i] = (QAt(q, i) - zp) * sf;
    return nullptr;
  }
  // per-channel: channel c = (i / stride) % dim[axis], stride = prod(dims[axis+1:]). Prefer the
  // DequantizeLinear 'axis' attribute; if it doesn't fit the scale length (e.g. a rank-1 int32
  // bias whose DQ omits axis, so it defaults to 1 out of range), fall back to the unique tensor
  // axis whose dim == scale length (ambiguous only for a square weight, where the attr is set).
  int64_t axis = ReadIntAttr(api, dq, "axis", 1);
  if (axis < 0) axis += (int64_t)q.dims.size();
  if (axis < 0 || axis >= (int64_t)q.dims.size() || (size_t)q.dims[axis] != s.count) {
    int found = -1;
    for (size_t d = 0; d < q.dims.size(); d++)
      if ((size_t)q.dims[d] == s.count) { found = (found == -1) ? (int)d : -2; }
    if (found < 0)
      return api.CreateStatus(ORT_INVALID_GRAPH,
          "ort-rocket: cannot resolve the per-channel DequantizeLinear axis for a weight");
    axis = found;
  }
  size_t stride = 1;
  for (size_t d = axis + 1; d < q.dims.size(); d++) stride *= (size_t)q.dims[d];
  size_t chan = (size_t)q.dims[axis];
  for (size_t i = 0; i < q.count; i++) {
    size_t c = (i / stride) % chan;
    int zp = has_z ? QAt(z, c) : 0;
    (*out)[i] = (QAt(q, i) - zp) * sc[c];
  }
  return nullptr;
}

// Fetch a weight by its (fp32) initializer name as fp32, dequantizing on the fly if the model
// is quantized: an fp32 model has the initializer `name`; a QDQ model replaced it with
// `name`_quantized fed through a DequantizeLinear (found via dq_by_q). Fills `out`.
static OrtStatus* WeightByName(const OrtApi& api, const GraphMaps& m, const std::string& name,
                               std::vector<float>* out) {
  auto it = m.init.find(name);
  if (it != m.init.end()) {
    RawInit r;
    if (OrtStatus* st = ReadInit(api, m, name, &r)) return st;
    if (r.et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      return api.CreateStatus(ORT_INVALID_GRAPH, ("ort-rocket: initializer not fp32: " + name).c_str());
    out->assign(static_cast<const float*>(r.data), static_cast<const float*>(r.data) + r.count);
    return nullptr;
  }
  auto dqit = m.dq_by_q.find(name + "_quantized");
  if (dqit != m.dq_by_q.end()) return DequantDQ(api, m, dqit->second, out);
  return api.CreateStatus(ORT_NOT_FOUND,
      ("ort-rocket: weight not found (fp32 or quantized): " + name).c_str());
}

static inline void ToF16(const float* src, _Float16* dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = (_Float16)src[i];
}
// dst[out][in] = src[in][out] (transpose an ONNX [in,out] projection to rocket [out,in]),
// optionally scaling output row o by lam[o] (the LayerScale fold on Wo/Wf2).
static inline void TransposeF16(const float* src, int in, int out, const float* lam, _Float16* dst) {
  for (int o = 0; o < out; o++) {
    float s = lam ? lam[o] : 1.0f;
    for (int i = 0; i < in; i++) dst[(size_t)o * in + i] = (_Float16)(src[(size_t)i * out + o] * s);
  }
}
// One column block of a FUSED projection: src is ONNX [in, nblk*out], and block `blk` spans columns
// [blk*out, (blk+1)*out). dst[out][in] is that block transposed to rocket [out,in]. Used for a fused
// qkv MatMul whose per-role block index the matcher derived (Depth Anything's order is key,value,query).
static inline void TransposeBlockF16(const float* src, int in, int out, int nblk, int blk,
                                     _Float16* dst) {
  const int row = nblk * out, col0 = blk * out;
  for (int o = 0; o < out; o++)
    for (int i = 0; i < in; i++)
      dst[(size_t)o * in + i] = (_Float16)src[(size_t)i * row + col0 + o];
}

// Read a weight TENSOR (a node input edge, resolved by the structural matcher) as fp32: a plain
// initializer -- fp32, or fp16 -- or a DequantizeLinear output (QDQ int8/int4/int32),
// dequantized on the fly. This is the edge-keyed analog of WeightByName: it reads whatever tensor the
// topology match points at, with no name assumption. Fills `out`.
static OrtStatus* ReadTensorF32(const OrtApi& api, const GraphMaps& m, const std::string& tensor,
                                std::vector<float>* out) {
  if (m.init.count(tensor)) {
    RawInit r;
    if (OrtStatus* st = ReadInit(api, m, tensor, &r)) return st;
    if (r.et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      out->assign(static_cast<const float*>(r.data), static_cast<const float*>(r.data) + r.count);
      return nullptr;
    }
    if (r.et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {   // fp16 ONNX export
      out->resize(r.count);
      const _Float16* h = static_cast<const _Float16*>(r.data);
      for (size_t i = 0; i < r.count; i++) (*out)[i] = static_cast<float>(h[i]);
      return nullptr;
    }
    return api.CreateStatus(ORT_INVALID_GRAPH,
        ("ort-rocket: weight initializer is not fp32 or fp16: " + tensor).c_str());
  }
  auto p = m.producer.find(tensor);
  if (p != m.producer.end() && NodeOp(api, p->second) == "DequantizeLinear")
    return DequantDQ(api, m, p->second, out);
  return api.CreateStatus(ORT_NOT_FOUND, ("ort-rocket: weight tensor not resolvable: " + tensor).c_str());
}

// The weight/bias input EDGE of a node: the single input tensor that resolves to an initializer,
// directly (fp32/fp16 model) or through a DequantizeLinear whose quantized source is an initializer
// (QDQ model -- skips a DQ on an activation input, whose source is not an initializer). "" if none.
static std::string InitInputTensor(const OrtApi& api, const GraphMaps& m, const OrtNode* n) {
  std::vector<std::string> in;
  if (NodeInputNames(api, n, &in)) return std::string();
  for (const std::string& t : in) {
    if (t.empty()) continue;
    if (m.init.count(t)) return t;                                 // fp32/fp16 weight directly
    auto p = m.producer.find(t);
    if (p != m.producer.end() && NodeOp(api, p->second) == "DequantizeLinear") {
      std::vector<std::string> din;
      if (!NodeInputNames(api, p->second, &din) && !din.empty() && m.init.count(din[0])) return t;
    }
  }
  return std::string();
}

// Read a node's weight/bias edge as fp32 (structural analog of WeightOfNode, keyed on the node ref
// the matcher produced rather than a node name). `role` names the weight for a clear error.
static OrtStatus* WeightOfNodeRef(const OrtApi& api, const GraphMaps& m, const OrtNode* n,
                                  const char* role, std::vector<float>* out) {
  std::string t = InitInputTensor(api, m, n);
  if (t.empty())
    return api.CreateStatus(ORT_NOT_FOUND,
        (std::string("ort-rocket: no weight edge on matched node for ") + role).c_str());
  return ReadTensorF32(api, m, t, out);
}

// Plain-ViT (SigLIP) Compute: read the preprocessed NCHW image, run the rocket_siglip encoder, and
// write the [1,L,d] token-feature output. No dewindow/projector (the encoder exit is the post-LN);
// no transpose (the output is already [token,channel]). The caller holds run_mtx.
static OrtStatus* ComputeSiglipImpl(CompiledBackbone* st, OrtKernelContext* kctx) {
  const OrtApi& api = st->api;
  const rocket_siglip_model& sm = st->sm;

  const OrtValue* in = nullptr;
  if (OrtStatus* s = api.KernelContext_GetInput(kctx, st->image_in_idx, &in)) return s;

  const size_t img_n = static_cast<size_t>(sm.ic) * sm.image_size * sm.image_size;
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* s = api.GetTensorTypeAndShape(in, &info)) return s;
    OrtStatus* s = api.GetTensorElementType(info, &et);
    size_t ecount = 0;
    if (!s) s = api.GetTensorShapeElementCount(info, &ecount);
    api.ReleaseTensorTypeAndShapeInfo(info);
    if (s) return s;
    if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "ort-rocket: image input must be fp32 or fp16");
    if (ecount != img_n)
      return api.CreateStatus(ORT_INVALID_ARGUMENT,
          ("ort-rocket: image input element count " + std::to_string(ecount) + " != expected " +
           std::to_string(img_n) + " ([1," + std::to_string(sm.ic) + "," +
           std::to_string(sm.image_size) + "," + std::to_string(sm.image_size) + "])").c_str());
  }
  const void* raw = nullptr;
  if (OrtStatus* s = api.GetTensorData(in, &raw)) return s;

  st->s_img.resize(img_n);
  _Float16* img = st->s_img.data();   // this fp16 NCHW buffer IS pixels_chw (the encoder im2cols it)
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    std::memcpy(img, raw, img_n * sizeof(_Float16));
  } else {
    const float* f = static_cast<const float*>(raw);
    for (size_t i = 0; i < img_n; i++) img[i] = (_Float16)f[i];
  }

  const int ntok = sm.L + sm.n_prefix;                  // tokens = patches (+ CLIP cls); == pos rows
  const int64_t shape[3] = {1, ntok, sm.d};
  OrtValue* out = nullptr;
  if (OrtStatus* s = api.KernelContext_GetOutput(kctx, 0, shape, 3, &out)) return s;
  void* od = nullptr;
  if (OrtStatus* s = api.GetTensorMutableData(out, &od)) return s;
  ONNXTensorElementDataType out_et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  {
    OrtTensorTypeAndShapeInfo* oi = nullptr;
    if (OrtStatus* s = api.GetTensorTypeAndShape(out, &oi)) return s;
    OrtStatus* s = api.GetTensorElementType(oi, &out_et);
    api.ReleaseTensorTypeAndShapeInfo(oi);
    if (s) return s;
  }

  const size_t n_out = static_cast<size_t>(ntok) * sm.d;
  // fp16 output: encode straight into the ORT buffer; fp32 output: encode into scratch then widen.
  _Float16* y = (out_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                    ? static_cast<_Float16*>(od)
                    : (st->s_out.resize(n_out), st->s_out.data());
  int rc = st->sctx ? rocket_siglip_encode_ctx(st->sctx, img, y, nullptr)
                    : rocket_siglip_encode(st->fd, &sm, img, y, nullptr);
  if (rc != 0)
    return api.CreateStatus(ORT_FAIL, ("ort-rocket: siglip encode failed rc=" + std::to_string(rc)).c_str());
  if (out_et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    float* of = static_cast<float*>(od);
    for (size_t i = 0; i < n_out; i++) of[i] = (float)y[i];
  }
  return nullptr;
}

// Depth Anything v2 Compute: read the preprocessed NCHW image, run the plain-ViT encoder in its
// multi-tap mode, and write one [1,ntok,d] output per DPT tap. The DPT head itself (reassemble,
// fusion, the align_corners resizes) stays on ORT's CPU kernels. The caller holds run_mtx.
static OrtStatus* ComputeDepthImpl(CompiledBackbone* st, OrtKernelContext* kctx) {
  const OrtApi& api = st->api;
  const rocket_siglip_model& sm = st->sm;

  const OrtValue* in = nullptr;
  if (OrtStatus* s = api.KernelContext_GetInput(kctx, st->image_in_idx, &in)) return s;
  const size_t img_n = static_cast<size_t>(sm.ic) * sm.image_size * sm.image_size;
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* s = api.GetTensorTypeAndShape(in, &info)) return s;
    OrtStatus* s = api.GetTensorElementType(info, &et);
    size_t ecount = 0;
    if (!s) s = api.GetTensorShapeElementCount(info, &ecount);
    api.ReleaseTensorTypeAndShapeInfo(info);
    if (s) return s;
    if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "ort-rocket: depth image input must be fp32 or fp16");
    if (ecount != img_n)
      return api.CreateStatus(ORT_INVALID_ARGUMENT,
          ("ort-rocket: depth image element count " + std::to_string(ecount) + " != expected " +
           std::to_string(img_n)).c_str());
  }
  const void* raw = nullptr;
  if (OrtStatus* s = api.GetTensorData(in, &raw)) return s;
  st->s_img.resize(img_n);
  _Float16* img = st->s_img.data();
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) std::memcpy(img, raw, img_n * sizeof(_Float16));
  else { const float* f = static_cast<const float*>(raw); for (size_t i = 0; i < img_n; i++) img[i] = (_Float16)f[i]; }

  const int ntok = sm.L + sm.n_prefix;
  const size_t Td = static_cast<size_t>(ntok) * sm.d;
  // The encoder writes every tap block into one contiguous buffer; each is then widened (or copied)
  // into its own ORT output. Reused across images like the other families' scratch.
  st->s_out.resize(Td * sm.n_taps);
  _Float16* taps = st->s_out.data();
  int rc = st->sctx ? rocket_siglip_encode_ctx(st->sctx, img, taps, nullptr)
                    : rocket_siglip_encode(st->fd, &sm, img, taps, nullptr);
  if (rc != 0)
    return api.CreateStatus(ORT_FAIL, ("ort-rocket: depth encode failed rc=" + std::to_string(rc)).c_str());

  const int64_t shape[3] = {1, ntok, sm.d};
  for (int t = 0; t < sm.n_taps; t++) {
    OrtValue* out = nullptr;
    if (OrtStatus* s = api.KernelContext_GetOutput(kctx, st->tap_out_slot[t], shape, 3, &out)) return s;
    void* od = nullptr;
    if (OrtStatus* s = api.GetTensorMutableData(out, &od)) return s;
    ONNXTensorElementDataType out_et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    {
      OrtTensorTypeAndShapeInfo* oi = nullptr;
      if (OrtStatus* s = api.GetTensorTypeAndShape(out, &oi)) return s;
      OrtStatus* s = api.GetTensorElementType(oi, &out_et);
      api.ReleaseTensorTypeAndShapeInfo(oi);
      if (s) return s;
    }
    const _Float16* src = taps + (size_t)t * Td;
    if (out_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) std::memcpy(od, src, Td * sizeof(_Float16));
    else { float* of = static_cast<float*>(od); for (size_t i = 0; i < Td; i++) of[i] = (float)src[i]; }
  }
  return nullptr;
}

// SAM ViT-Det Compute: read the preprocessed NCHW image, run the rocket_sam encoder, and write the
// neck output [1,neck_out,grid,grid] (== the ONNX last_hidden_state). The caller holds run_mtx.
static OrtStatus* ComputeSamImpl(CompiledBackbone* st, OrtKernelContext* kctx) {
  const OrtApi& api = st->api;
  const rocket_sam_model& sm = st->sm2;

  const OrtValue* in = nullptr;
  if (OrtStatus* s = api.KernelContext_GetInput(kctx, st->image_in_idx, &in)) return s;
  const size_t img_n = static_cast<size_t>(sm.ic) * sm.image_size * sm.image_size;
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (OrtStatus* s = api.GetTensorTypeAndShape(in, &info)) return s;
    OrtStatus* s = api.GetTensorElementType(info, &et);
    size_t ecount = 0;
    if (!s) s = api.GetTensorShapeElementCount(info, &ecount);
    api.ReleaseTensorTypeAndShapeInfo(info);
    if (s) return s;
    if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "ort-rocket: SAM image input must be fp32 or fp16");
    if (ecount != img_n)
      return api.CreateStatus(ORT_INVALID_ARGUMENT,
          ("ort-rocket: SAM image element count " + std::to_string(ecount) + " != expected " +
           std::to_string(img_n)).c_str());
  }
  const void* raw = nullptr;
  if (OrtStatus* s = api.GetTensorData(in, &raw)) return s;
  st->s_img.resize(img_n);
  _Float16* img = st->s_img.data();
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) std::memcpy(img, raw, img_n * sizeof(_Float16));
  else { const float* f = static_cast<const float*>(raw); for (size_t i = 0; i < img_n; i++) img[i] = (_Float16)f[i]; }

  const int64_t shape[4] = {1, sm.neck_out, sm.grid, sm.grid};
  OrtValue* out = nullptr;
  if (OrtStatus* s = api.KernelContext_GetOutput(kctx, 0, shape, 4, &out)) return s;
  void* od = nullptr;
  if (OrtStatus* s = api.GetTensorMutableData(out, &od)) return s;
  ONNXTensorElementDataType out_et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  {
    OrtTensorTypeAndShapeInfo* oi = nullptr;
    if (OrtStatus* s = api.GetTensorTypeAndShape(out, &oi)) return s;
    OrtStatus* s = api.GetTensorElementType(oi, &out_et);
    api.ReleaseTensorTypeAndShapeInfo(oi);
    if (s) return s;
  }
  const size_t n_out = static_cast<size_t>(sm.neck_out) * sm.grid * sm.grid;
  _Float16* y = (out_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                    ? static_cast<_Float16*>(od)
                    : (st->s_out.resize(n_out), st->s_out.data());
  int rc = st->sctx2 ? rocket_sam_encode_ctx(st->sctx2, img, y, nullptr)
                     : rocket_sam_encode(st->fd, &sm, img, y, nullptr);
  if (rc != 0)
    return api.CreateStatus(ORT_FAIL, ("ort-rocket: SAM encode failed rc=" + std::to_string(rc)).c_str());
  if (out_et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    float* of = static_cast<float*>(od);
    for (size_t i = 0; i < n_out; i++) of[i] = (float)y[i];
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// OrtNodeComputeInfo implementation.
// ---------------------------------------------------------------------------
struct BackboneComputeInfo : OrtNodeComputeInfo {
  explicit BackboneComputeInfo(CompiledBackbone* st) : OrtNodeComputeInfo{}, state(st) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }
  CompiledBackbone* state;

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* /*ctx*/,
                                                 void** compute_state) noexcept {
    *compute_state = static_cast<BackboneComputeInfo*>(this_ptr)->state;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kctx) noexcept {
    (void)this_ptr;
    auto* st = static_cast<CompiledBackbone*>(compute_state);
    // Serialize concurrent Run()s on this session: the state (ctxs, fd, scratch) is shared and
    // not thread-safe. See CompiledBackbone::run_mtx.
    std::lock_guard<std::mutex> run_guard(st->run_mtx);
    if (st->is_depth) return ComputeDepthImpl(st, kctx);
    if (st->is_siglip) return ComputeSiglipImpl(st, kctx);
    if (st->is_sam) return ComputeSamImpl(st, kctx);
    const OrtApi& api = st->api;
    const rocket_backbone_geom* g = st->w.geom;

    // The preprocessed image [1,3,img,img] fp32 (at the index found in Compile; the fused
    // node also lists every initializer as an input).
    const OrtValue* in = nullptr;
    if (OrtStatus* s = api.KernelContext_GetInput(kctx, st->image_in_idx, &in)) return s;

    const size_t img_n = static_cast<size_t>(RBB_IC) * g->img * g->img;
    // Runtime image-input check: the image bytes are about to be read as exactly img_n elements. Reject an
    // unsupported dtype or an element count that does not match the selected geometry (e.g. a
    // same-patch checkpoint exported at a different resolution) rather than reading with the wrong
    // stride and returning silent garbage. Cheap -- two scalar checks per inference. Accept an
    // fp16 image input (an fp16 ONNX export) in addition to fp32.
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    {
      OrtTensorTypeAndShapeInfo* info = nullptr;
      if (OrtStatus* s = api.GetTensorTypeAndShape(in, &info)) return s;
      OrtStatus* s = api.GetTensorElementType(info, &et);
      size_t ecount = 0;
      if (!s) s = api.GetTensorShapeElementCount(info, &ecount);
      api.ReleaseTensorTypeAndShapeInfo(info);
      if (s) return s;
      if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        return api.CreateStatus(ORT_INVALID_ARGUMENT,
            "ort-rocket: image input must be fp32 or fp16");
      if (ecount != img_n)
        return api.CreateStatus(ORT_INVALID_ARGUMENT,
            ("ort-rocket: image input element count " + std::to_string(ecount) + " != expected " +
             std::to_string(img_n) + " ([1," + std::to_string(RBB_IC) + "," +
             std::to_string(g->img) + "," + std::to_string(g->img) +
             "] for the selected geometry)").c_str());
    }
    const void* raw = nullptr;
    if (OrtStatus* s = api.GetTensorData(in, &raw)) return s;

    // Reused host scratch: resize is a no-op after the first Compute, so no per-image heap
    // traffic in steady state. run_mtx makes reuse across concurrent Runs safe. The backbone
    // consumes fp16; convert an fp32 input, or copy an already-fp16 input straight through.
    st->s_img.resize(img_n);
    _Float16* img = st->s_img.data();
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      std::memcpy(img, raw, img_n * sizeof(_Float16));
    } else {
      const float* img_f32 = static_cast<const float*>(raw);
      for (size_t i = 0; i < img_n; i++) img[i] = (_Float16)img_f32[i];
    }

    _Float16* feats[RBB_NTAP];
    for (int t = 0; t < RBB_NTAP; t++) {
      st->s_feat[t].resize(static_cast<size_t>(g->ntok) * g->d);
      feats[t] = st->s_feat[t].data();
    }
    // Resident (prepacked, multicore) path when the ctxs are live; else the one-shot chain.
    int rc = st->bctx ? rocket_backbone_run_ctx(st->bctx, img, feats)
                      : rocket_backbone_run(st->fd, &st->w, img, feats);
    if (rc != 0)
      return api.CreateStatus(ORT_FAIL, ("ort-rocket: backbone run failed rc=" + std::to_string(rc)).c_str());

    // De-window each feature (drop cls, un-window-partition to grid x grid spatial), then run
    // the CSP projector on the NPU: 4 taps [np,384] -> one [np,256] fused feature map.
    const _Float16* taps[RPJ_NTAP];
    for (int t = 0; t < RPJ_NTAP; t++) {
      st->s_tap[t].resize(static_cast<size_t>(g->np) * RPJ_DIN);
      rocket_projector_dewindow(g, feats[t], st->s_tap[t].data());
      taps[t] = st->s_tap[t].data();
    }
    st->s_y.resize(static_cast<size_t>(g->np) * RPJ_CO);
    _Float16* y = st->s_y.data();
    rc = st->pctx ? rocket_projector_run_ctx(st->pctx, taps, y)
                  : rocket_projector_run(st->fd, &st->pw, taps, y);
    if (rc != 0)
      return api.CreateStatus(ORT_FAIL, ("ort-rocket: projector run failed rc=" + std::to_string(rc)).c_str());

    // The single fused-node output [1,256,grid,grid] (NCHW): the projector is token-major [np,256]
    // with position s = h*grid+w, so NCHW[c][h][w] = y[(h*grid+w)*256 + c]. The output dtype follows
    // the model -- fp32 for the standard RF-DETR export, fp16 for an all-fp16 export -- so query
    // it and write the matching type. Cache-block the transpose so a tile of y (strided reads) and of
    // the output (strided writes) both stay resident, instead of sweeping all np positions per channel.
    const int64_t shape[4] = {1, RPJ_CO, g->grid, g->grid};
    OrtValue* out = nullptr;
    if (OrtStatus* s = api.KernelContext_GetOutput(kctx, 0, shape, 4, &out)) return s;
    void* od = nullptr;
    if (OrtStatus* s = api.GetTensorMutableData(out, &od)) return s;
    ONNXTensorElementDataType out_et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    {
      OrtTensorTypeAndShapeInfo* oi = nullptr;
      if (OrtStatus* s = api.GetTensorTypeAndShape(out, &oi)) return s;
      OrtStatus* s = api.GetTensorElementType(oi, &out_et);
      api.ReleaseTensorTypeAndShapeInfo(oi);
      if (s) return s;
    }
    const int np = g->np;
    constexpr int BS = 16;
    auto transpose_out = [&](auto* of) {   // T deduced float or _Float16; _Float16->T is exact/widening
      for (int c0 = 0; c0 < RPJ_CO; c0 += BS)
        for (int s0 = 0; s0 < np; s0 += BS) {
          const int ce = c0 + BS < RPJ_CO ? c0 + BS : RPJ_CO;
          const int se = s0 + BS < np ? s0 + BS : np;
          for (int c = c0; c < ce; c++)
            for (int s = s0; s < se; s++)
              of[(size_t)c * np + s] = y[(size_t)s * RPJ_CO + c];
        }
    };
    if (out_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) transpose_out(static_cast<_Float16*>(od));
    else                                                 transpose_out(static_cast<float*>(od));
    return nullptr;
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                            void* /*compute_state*/) noexcept {
    // State is owned by the OrtNodeComputeInfo (freed in ReleaseNodeComputeInfos).
  }
};

// Find the image input among the fused node's inputs: the one input that is NOT an
// initializer (ORT lists every weight/constant as an input; the image is the lone activation).
static OrtStatus* MapImageInput(const OrtApi& api, const OrtNode* fused, const GraphMaps& m,
                                CompiledBackbone* cb) {
  size_t nin = 0;
  if (OrtStatus* st = api.Node_GetNumInputs(fused, &nin)) return st;
  std::vector<const OrtValueInfo*> ins(nin);
  if (OrtStatus* st = api.Node_GetInputs(fused, ins.data(), nin)) return st;
  int found = -1;
  for (size_t i = 0; i < nin; i++) {
    if (!ins[i]) continue;
    const char* nm = nullptr;
    if (OrtStatus* st = api.GetValueInfoName(ins[i], &nm)) return st;
    if (nm && m.init.find(nm) == m.init.end()) {
      if (found >= 0)
        return api.CreateStatus(ORT_INVALID_GRAPH,
            "ort-rocket: >1 non-initializer backbone input; expected only the image");
      found = static_cast<int>(i);
    }
  }
  if (found < 0)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: no image input on the fused node");
  cb->image_in_idx = static_cast<size_t>(found);
  return nullptr;
}

// Compile-time image-input check: validate the fused node's image input against the selected geometry.
// The pinned datapath assumes a [1, RBB_IC, img, img] fp32 image; the variant is keyed on the
// patch-conv size (which disambiguates nano/base) but NOT on resolution, so a same-patch export at
// a different resolution would select the wrong geometry and silently mis-read. Here we reject a
// concrete dtype/shape mismatch loudly. Symbolic/unknown dims can't be checked at Compile (Compute
// enforces the element count on every call regardless); *concrete reports whether the [1,IC,img,img]
// shape was fully pinned, so STRICT can note when it wasn't. Requires image_in_idx (set by
// MapImageInput).
static OrtStatus* ValidateImageShape(const OrtApi& api, const OrtNode* fused,
                                     const CompiledBackbone* cb, bool* concrete) {
  *concrete = false;
  const rocket_backbone_geom* g = cb->w.geom;
  size_t nin = 0;
  if (OrtStatus* st = api.Node_GetNumInputs(fused, &nin)) return st;
  if (cb->image_in_idx >= nin)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: image input index out of range");
  std::vector<const OrtValueInfo*> ins(nin);
  if (OrtStatus* st = api.Node_GetInputs(fused, ins.data(), nin)) return st;
  const OrtTypeInfo* ti = nullptr;                       // graph-owned, not released
  if (OrtStatus* st = api.GetValueInfoTypeInfo(ins[cb->image_in_idx], &ti)) return st;
  const OrtTensorTypeAndShapeInfo* tsi = nullptr;        // borrowed from ti, not released
  if (OrtStatus* st = api.CastTypeInfoToTensorInfo(ti, &tsi)) return st;
  if (!tsi) return nullptr;                              // not a tensor type; Compute still checks
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (OrtStatus* st = api.GetTensorElementType(tsi, &et)) return st;
  if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED && et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)   // fp16 image input is accepted
    return api.CreateStatus(ORT_INVALID_GRAPH,
        "ort-rocket: image input is not fp32 or fp16 (unsupported input dtype)");
  size_t nd = 0;
  if (OrtStatus* st = api.GetDimensionsCount(tsi, &nd)) return st;
  if (nd != 4) return nullptr;                           // rank unknown/other; Compute enforces count
  std::vector<int64_t> dims(nd);
  if (OrtStatus* st = api.GetDimensions(tsi, dims.data(), nd)) return st;
  // NCHW [1, IC, img, img]; only concrete (>0) dims are checked -- <=0 is a symbolic dim.
  const int64_t want[4] = {1, RBB_IC, g->img, g->img};
  const char* what[4] = {"batch", "channels", "height", "width"};
  bool all_concrete = true;
  for (int i = 0; i < 4; i++) {
    if (dims[i] <= 0) { all_concrete = false; continue; }
    if (dims[i] != want[i])
      return api.CreateStatus(ORT_INVALID_GRAPH,
          (std::string("ort-rocket: image input ") + what[i] + " " + std::to_string(dims[i]) +
           " != expected " + std::to_string(want[i]) + " for the selected geometry (patch " +
           std::to_string(g->patch) + ", img " + std::to_string(g->img) + ")").c_str());
  }
  *concrete = all_concrete;
  return nullptr;
}

// Marshal every fp16 weight the backbone needs, driven by the structural match. The per-layer
// projections/biases/norms/LayerScales are read from the matched nodes' initializer edges (the
// anonymous `onnx::MatMul_*` weights that name-based marshaling could only reach through
// exporter-generated node names). The stem cls/position table and the shared feature norm keep their
// PyTorch state_dict names: those survive an opset/exporter re-export (only an RF-DETR module refactor
// renames them, which also breaks the topology match). The datapath layout is unchanged: patch weight
// laid out [D,pk_pad] with a zero K-tail, projections transposed [in,out]->[out,in], LayerScale
// (lambda1/lambda2) folded into Wo/bo and Wf2/bf2.
static OrtStatus* MarshalWeights(const OrtApi& api, const GraphMaps& m,
                                 const rocket_match::BackboneMatch& match, CompiledBackbone* cb) {
  const std::string EMB = "backbone.0.encoder.encoder.embeddings.";
  std::vector<float> vbuf;
  const float* d = nullptr; size_t c = 0;
  auto read_tensor = [&](const std::string& t) -> OrtStatus* {   // by graph edge (structural)
    if (OrtStatus* st = ReadTensorF32(api, m, t, &vbuf)) return st;
    d = vbuf.data(); c = vbuf.size(); return nullptr;
  };
  auto read_node = [&](const OrtNode* n, const char* role) -> OrtStatus* {   // node's weight/bias edge
    if (OrtStatus* st = WeightOfNodeRef(api, m, n, role, &vbuf)) return st;
    d = vbuf.data(); c = vbuf.size(); return nullptr;
  };
  auto read_name = [&](const std::string& nm) -> OrtStatus* {   // stable state_dict name
    if (OrtStatus* st = WeightByName(api, m, nm, &vbuf)) return st;
    d = vbuf.data(); c = vbuf.size(); return nullptr;
  };

  // --- geometry from the patch Conv kernel size (structural, no sqrt-from-count arithmetic) ---
  const rocket_backbone_geom* geom = rocket_backbone_geom_select(match.patch_size);
  if (!geom)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: unrecognized patch size from the matched conv");
  cb->w.geom = geom;
  cb->pw.geom = geom;

  // --- stem: patch-embed conv weight [D,IC,P,P]=[D,pk] -> [D,pk_pad] (zero K-tail), bias [D] ---
  std::vector<std::string> pin;
  if (OrtStatus* st = NodeInputNames(api, match.patch_conv, &pin)) return st;
  if (pin.size() < 3 || pin[1].empty() || pin[2].empty())
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: patch conv missing weight/bias");
  if (OrtStatus* st = read_tensor(pin[1])) return st;
  if (c != (size_t)geom->d * geom->pk)
    return api.CreateStatus(ORT_INVALID_GRAPH,
        "ort-rocket: patch conv weight size does not match the selected geometry");
  {
    _Float16* pw = cb->alloc((size_t)geom->d * geom->pk_pad);
    for (int o = 0; o < geom->d; o++)
      for (int k = 0; k < geom->pk; k++)
        pw[(size_t)o * geom->pk_pad + k] = (_Float16)d[(size_t)o * geom->pk + k];
    cb->w.patch_w = pw;
  }
  if (OrtStatus* st = read_tensor(pin[2])) return st;
  cb->w.patch_b = cb->alloc(c); ToF16(d, const_cast<_Float16*>(cb->w.patch_b), c);

  // cls token / position table: stable state_dict names (a fixed embeddings singleton, no anonymous
  // initializer to disambiguate).
  if (OrtStatus* st = read_name(EMB + "cls_token")) return st;
  cb->w.cls = cb->alloc(c); ToF16(d, const_cast<_Float16*>(cb->w.cls), c);
  if (OrtStatus* st = read_name(EMB + "position_embeddings")) return st;
  cb->w.pos = cb->alloc(c); ToF16(d, const_cast<_Float16*>(cb->w.pos), c);

  // --- feature norm (shared gamma/beta), stable state_dict name ---
  if (OrtStatus* st = read_name("backbone.0.encoder.encoder.layernorm.weight")) return st;
  cb->w.fn_g = cb->alloc(c); ToF16(d, const_cast<_Float16*>(cb->w.fn_g), c);
  if (OrtStatus* st = read_name("backbone.0.encoder.encoder.layernorm.bias")) return st;
  cb->w.fn_b = cb->alloc(c); ToF16(d, const_cast<_Float16*>(cb->w.fn_b), c);

  // --- encoder layers (structural: every weight from a matched node's edge) ---
  std::vector<float> lam1v, lam2v;
  // gamma/beta of a matched LayerNormalization (inputs 1 and 2).
  auto ln_gb = [&](const OrtNode* ln, const _Float16** g, const _Float16** b) -> OrtStatus* {
    std::vector<std::string> in;
    if (OrtStatus* st = NodeInputNames(api, ln, &in)) return st;
    if (in.size() < 3) return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: matched LayerNorm missing gamma/beta");
    if (OrtStatus* st = read_tensor(in[1])) return st;
    *g = cb->alloc(c); ToF16(d, const_cast<_Float16*>(*g), c);
    if (OrtStatus* st = read_tensor(in[2])) return st;
    *b = cb->alloc(c); ToF16(d, const_cast<_Float16*>(*b), c);
    return nullptr;
  };
  for (int L = 0; L < geom->nlayer; L++) {
    const rocket_match::MatchedLayer& ml = match.layers[L];
    rocket_backbone_layer& lw = cb->w.layer[L];
    if (OrtStatus* st = ln_gb(ml.norm1, &lw.ln1_g, &lw.ln1_b)) return st;
    if (OrtStatus* st = ln_gb(ml.norm2, &lw.ln2_g, &lw.ln2_b)) return st;
    // LayerScale lambdas (own storage; the Mul's initializer operand), used by the folds below.
    if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, ml.ls1), &lam1v)) return st;
    if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, ml.ls2), &lam2v)) return st;
    const float* lam1 = lam1v.data();
    const float* lam2 = lam2v.data();
    // projections: read the matched MatMul's weight edge (ONNX [in,out]), transpose -> [out,in],
    // folding LayerScale into Wo/Wf2.
    auto projw = [&](const OrtNode* mm, const float* lam, int in, int out, const _Float16** dst,
                     const char* role) -> OrtStatus* {
      if (OrtStatus* st = read_node(mm, role)) return st;
      if (c != (size_t)in * out)
        return api.CreateStatus(ORT_INVALID_GRAPH,
            (std::string("ort-rocket: projection weight shape mismatch for ") + role).c_str());
      *dst = cb->alloc((size_t)in * out); TransposeF16(d, in, out, lam, const_cast<_Float16*>(*dst)); return nullptr;
    };
    if (OrtStatus* st = projw(ml.q,   nullptr, geom->d,   geom->d,   &lw.wq,  "wq"))  return st;
    if (OrtStatus* st = projw(ml.k,   nullptr, geom->d,   geom->d,   &lw.wk,  "wk"))  return st;
    if (OrtStatus* st = projw(ml.v,   nullptr, geom->d,   geom->d,   &lw.wv,  "wv"))  return st;
    if (OrtStatus* st = projw(ml.o,   lam1,    geom->d,   geom->d,   &lw.wo,  "wo"))  return st;
    if (OrtStatus* st = projw(ml.fc1, nullptr, geom->d,   geom->dff, &lw.wf1, "wf1")) return st;
    if (OrtStatus* st = projw(ml.fc2, lam2,    geom->dff, geom->d,   &lw.wf2, "wf2")) return st;
    // biases: the matched bias-Add's edge; bo/bf2 scaled by LayerScale.
    auto biasw = [&](const OrtNode* add, const float* lam, const _Float16** dst, const char* role) -> OrtStatus* {
      if (OrtStatus* st = read_node(add, role)) return st;
      _Float16* b = cb->alloc(c); *dst = b;
      for (size_t i = 0; i < c; i++) b[i] = (_Float16)(d[i] * (lam ? lam[i] : 1.0f));
      return nullptr;
    };
    if (OrtStatus* st = biasw(ml.bq,  nullptr, &lw.bq,  "bq"))  return st;
    if (OrtStatus* st = biasw(ml.bk,  nullptr, &lw.bk,  "bk"))  return st;
    if (OrtStatus* st = biasw(ml.bv,  nullptr, &lw.bv,  "bv"))  return st;
    if (OrtStatus* st = biasw(ml.bo,  lam1,    &lw.bo,  "bo"))  return st;
    if (OrtStatus* st = biasw(ml.bf1, nullptr, &lw.bf1, "bf1")) return st;
    if (OrtStatus* st = biasw(ml.bf2, lam2,    &lw.bf2, "bf2")) return st;
  }
  cb->w.eps = 1e-6f;
  return nullptr;
}

// Marshal a plain-ViT (CLIP/SigLIP) encoder into cb->sm, a rocket_siglip_model backed by one
// contiguous fp16 buffer in `owned`. Geometry is derived from the graph (patch-conv weight shape,
// fc1 shape, pos-embed rows, matched layer count); n_head defaults to head-dim 64 (ViT-B/L),
// overridable via ROCKET_ORT_SIGLIP_HEADS. Per-layer weights are laid out in rocket_siglip's exact
// order/stride; the ONNX MatMul projections ([in,out]) are transposed to rocket [out,in]; there is
// no LayerScale. The encoder does its own im2col, so patch_W stays [d][ic*kh*kw] (no K-pad).
// CLIP is resolved from the same match: a prepended cls token (n_prefix rows, prefix from
// match.cls_embed), a pre-encoder LayerNorm (match.pre_ln), quick-GELU (match.quick_gelu), a
// bias-less patch Conv, and no sequence post-LN (match.post_ln null). eps is read from the norm1
// epsilon attribute (1e-6 SigLIP, 1e-5 CLIP) rather than hardcoded.
static OrtStatus* MarshalSiglip(const OrtApi& api, const GraphMaps& m,
                                const rocket_match::BackboneMatch& match, CompiledBackbone* cb) {
  std::vector<float> vbuf;
  // --- geometry from the graph ---
  std::vector<std::string> pin;
  if (OrtStatus* st = NodeInputNames(api, match.patch_conv, &pin)) return st;
  if (pin.size() < 2 || pin[1].empty())
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip patch conv missing weight");
  const bool has_patch_b = (pin.size() >= 3 && !pin[2].empty());   // CLIP's patch Conv is bias-less
  RawInit pcw;
  if (OrtStatus* st = ReadInit(api, m, pin[1], &pcw)) return st;   // patch conv weight [d,ic,kh,kw]
  if (pcw.dims.size() != 4)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip patch conv weight is not rank-4");
  const int d = (int)pcw.dims[0], ic = (int)pcw.dims[1], kh = (int)pcw.dims[2], kw = (int)pcw.dims[3];
  const int nlayer = (int)match.layers.size();
  const int patch_dim = ic * kh * kw;
  // d_ff from the fc1 weight edge ([in=d, out=d_ff]); token count from the pos-embedding rows.
  RawInit fc1r, posr;
  if (OrtStatus* st = ReadInit(api, m, InitInputTensor(api, m, match.layers[0].fc1), &fc1r)) return st;
  if (fc1r.dims.size() != 2 || (int)fc1r.dims[0] != d)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip fc1 weight shape unexpected");
  const int d_ff = (int)fc1r.dims[1];
  if (OrtStatus* st = ReadInit(api, m, match.pos_embed, &posr)) return st;
  if (posr.dims.size() != 2 || (int)posr.dims[1] != d)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip position embedding shape unexpected");
  const int ntok = (int)posr.dims[0];                    // one position row per token (incl any cls)
  int n_head = 0;
  if (const char* hz = std::getenv("ROCKET_ORT_SIGLIP_HEADS")) n_head = std::atoi(hz);
  if (n_head <= 0) n_head = d / 64;                       // ViT-B/L head dim is 64
  if (n_head <= 0 || d % n_head != 0)
    return api.CreateStatus(ORT_INVALID_GRAPH,
        "ort-rocket: cannot infer siglip n_head (set ROCKET_ORT_SIGLIP_HEADS)");
  // ViT patch grids are square, so the token count splits into a side*side grid plus prefix (cls)
  // tokens: n_prefix = ntok - floor(sqrt(ntok))^2 (0 for SigLIP, 1 for CLIP's cls). The cls-token
  // presence must agree with the matcher's structural class-embedding find.
  int side = 1; while ((side + 1) * (side + 1) <= ntok) side++;
  const int L = side * side;                             // patch count
  const int n_prefix = ntok - L;
  const bool has_pre = (match.pre_ln != nullptr);        // CLIP pre_layrnorm
  const bool has_post = (match.post_ln != nullptr);      // SigLIP sequence post-LN; CLIP has none
  if (n_prefix < 0 || n_prefix > 8 || kh != kw || kh != match.patch_size)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip token count / patch geometry inconsistent");
  if ((n_prefix > 0) != (!match.cls_embed.empty()))
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip cls-token geometry / class-embedding mismatch");
  const float eps = ReadFloatAttr(api, match.layers[0].norm1, "epsilon", 1e-5f);  // CLIP 1e-5, SigLIP 1e-6

  // --- one contiguous fp16 buffer, laid out for rocket_siglip. Optional regions (patch bias, cls
  //     prefix, pre-LN, post-LN) are present per the matched variant; pos has one row per token. ---
  const size_t ls = (size_t)2 * d + 4 * ((size_t)d * d + d) + 2 * d +
                    ((size_t)d_ff * d + d_ff) + ((size_t)d * d_ff + d);   // layer_stride_elems(d,d_ff)
  const size_t total = (size_t)d * patch_dim
                     + (has_patch_b ? (size_t)d : 0)
                     + (size_t)n_prefix * d
                     + (has_pre ? (size_t)2 * d : 0)
                     + (size_t)ntok * d
                     + (size_t)nlayer * ls
                     + (has_post ? (size_t)2 * d : 0);
  _Float16* buf = cb->alloc(total);
  size_t off = 0;
  auto take = [&](size_t n) -> _Float16* { _Float16* p = buf + off; off += n; return p; };
  // read a graph edge as fp32 (into vbuf), verify count, and copy (optionally transposing [in,out]->[out,in])
  auto put = [&](const std::string& edge, size_t expect, _Float16* dst) -> OrtStatus* {
    if (OrtStatus* st = ReadTensorF32(api, m, edge, &vbuf)) return st;
    if (expect && vbuf.size() != expect)
      return api.CreateStatus(ORT_INVALID_GRAPH, ("ort-rocket: siglip weight size mismatch: " + edge).c_str());
    ToF16(vbuf.data(), dst, vbuf.size());
    return nullptr;
  };
  auto putT = [&](const OrtNode* mm, int in, int out, _Float16* dst) -> OrtStatus* {
    const std::string edge = InitInputTensor(api, m, mm);
    if (OrtStatus* st = ReadTensorF32(api, m, edge, &vbuf)) return st;
    if (vbuf.size() != (size_t)in * out)
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip projection weight shape mismatch");
    TransposeF16(vbuf.data(), in, out, nullptr, dst);
    return nullptr;
  };
  // gamma/beta of a matched LayerNormalization (inputs 1,2) into two [d] slots.
  auto ln_gb = [&](const OrtNode* ln, _Float16* g, _Float16* b) -> OrtStatus* {
    std::vector<std::string> in;
    if (OrtStatus* st = NodeInputNames(api, ln, &in)) return st;
    if (in.size() < 3) return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip LayerNorm missing gamma/beta");
    if (OrtStatus* st = put(in[1], d, g)) return st;
    return put(in[2], d, b);
  };

  rocket_siglip_model& sm = cb->sm;
  // stem: patch_W [d][patch_dim] (conv weight is already [oc][ic*kh*kw], no transpose)
  sm.patch_W = take((size_t)d * patch_dim);
  if (OrtStatus* st = put(pin[1], (size_t)d * patch_dim, const_cast<_Float16*>(sm.patch_W))) return st;
  if (has_patch_b) {                                     // patch bias [d] (absent for CLIP)
    sm.patch_b = take(d);
    if (OrtStatus* st = put(pin[2], d, const_cast<_Float16*>(sm.patch_b))) return st;
  } else sm.patch_b = nullptr;
  if (n_prefix > 0) {                                    // prefix (cls) embedding [n_prefix][d] (CLIP)
    sm.prefix = take((size_t)n_prefix * d);
    if (OrtStatus* st = put(match.cls_embed, (size_t)n_prefix * d, const_cast<_Float16*>(sm.prefix))) return st;
  } else sm.prefix = nullptr;
  if (has_pre) {                                         // pre-encoder LayerNorm (CLIP pre_layrnorm)
    sm.pre_g = take(d); sm.pre_b = take(d);
    if (OrtStatus* st = ln_gb(match.pre_ln, const_cast<_Float16*>(sm.pre_g),
                              const_cast<_Float16*>(sm.pre_b))) return st;
  } else { sm.pre_g = nullptr; sm.pre_b = nullptr; }
  sm.pos = take((size_t)ntok * d);                       // position table [ntok][d]
  if (OrtStatus* st = put(match.pos_embed, (size_t)ntok * d, const_cast<_Float16*>(sm.pos))) return st;

  // encoder layers, each block in rocket_siglip's exact order (== layer_stride_elems)
  sm.layers = take((size_t)nlayer * ls);
  for (int Lx = 0; Lx < nlayer; Lx++) {
    const rocket_match::MatchedLayer& ml = match.layers[Lx];
    _Float16* lb = const_cast<_Float16*>(sm.layers) + (size_t)Lx * ls;
    size_t o = 0;
    auto slot = [&](size_t n) -> _Float16* { _Float16* p = lb + o; o += n; return p; };
    _Float16* ln1_g = slot(d); _Float16* ln1_b = slot(d);
    if (OrtStatus* st = ln_gb(ml.norm1, ln1_g, ln1_b)) return st;
    if (OrtStatus* st = putT(ml.q, d, d, slot((size_t)d * d))) return st;
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bq), d, slot(d))) return st;
    if (OrtStatus* st = putT(ml.k, d, d, slot((size_t)d * d))) return st;
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bk), d, slot(d))) return st;
    if (OrtStatus* st = putT(ml.v, d, d, slot((size_t)d * d))) return st;
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bv), d, slot(d))) return st;
    if (OrtStatus* st = putT(ml.o, d, d, slot((size_t)d * d))) return st;
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bo), d, slot(d))) return st;
    _Float16* ln2_g = slot(d); _Float16* ln2_b = slot(d);
    if (OrtStatus* st = ln_gb(ml.norm2, ln2_g, ln2_b)) return st;
    if (OrtStatus* st = putT(ml.fc1, d, d_ff, slot((size_t)d_ff * d))) return st;   // [d_ff,d]
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bf1), d_ff, slot(d_ff))) return st;
    if (OrtStatus* st = putT(ml.fc2, d_ff, d, slot((size_t)d * d_ff))) return st;   // [d,d_ff]
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bf2), d, slot(d))) return st;
    if (o != ls) return api.CreateStatus(ORT_FAIL, "ort-rocket: siglip layer stride mismatch (internal)");
  }
  // optional post-LayerNorm over the sequence (SigLIP's encoder exit affine). CLIP has none here --
  // its post_layernorm applies only to the pooled cls row, left on the ORT-CPU host tail.
  if (has_post) {
    sm.post_g = take(d);
    sm.post_b = take(d);
    std::vector<std::string> in;
    if (OrtStatus* st = NodeInputNames(api, match.post_ln, &in)) return st;
    if (in.size() < 3) return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: siglip post-LayerNorm missing gamma/beta");
    if (OrtStatus* st = put(in[1], d, const_cast<_Float16*>(sm.post_g))) return st;
    if (OrtStatus* st = put(in[2], d, const_cast<_Float16*>(sm.post_b))) return st;
  } else { sm.post_g = nullptr; sm.post_b = nullptr; }
  if (off != total) return api.CreateStatus(ORT_FAIL, "ort-rocket: siglip buffer layout mismatch (internal)");

  sm.d = d; sm.n_layers = nlayer; sm.n_head = n_head; sm.d_ff = d_ff;
  sm.L = L; sm.patch_dim = patch_dim; sm.ic = ic; sm.kh = kh; sm.kw = kw;
  sm.stride = match.patch_size; sm.image_size = side * match.patch_size; sm.eps = eps;
  sm.n_prefix = n_prefix; sm.act_kind = match.quick_gelu ? 1 : 0;   // CLIP: cls token + quick-GELU
  sm.layer_stride = ls;
  sm.map = (void*)buf; sm.map_size = total * sizeof(_Float16);   // non-NULL sentinel; NEVER rocket_siglip_free
  return nullptr;
}

// Marshal a Depth Anything v2 DINOv2 encoder into cb->sm, a rocket_siglip_model. The compute is the
// plain-ViT encoder -- this family's three structural differences all resolve at marshal time rather
// than needing their own datapath:
//
//   1. FUSED qkv. One ONNX MatMul [d,3d] whose column blocks are (key, value, query), not (q,k,v).
//      The matcher DERIVED each role's block by walking back to the Split, so the three [d,d]
//      projections are sliced out by that index -- never by an assumed ordering.
//   2. LayerScale. lambda1/lambda2 fold into (Wo,bo) and (Wf2,bf2) by scaling output channel o,
//      exactly as the RF-DETR marshaling does. Algebraically identical, and it costs the datapath
//      nothing.
//   3. A DPT multi-tap exit. The head reads several intermediate layer outputs through ONE shared
//      LayerNorm, so that norm becomes post_g/post_b and the tapped layers become tap_layer[] --
//      the encoder emits one normalized block per tap.
//
// The activation is DINOv2's exact erf GELU (act_kind 2), and the patch contraction ic*p*p is padded
// up to a multiple of 32 because the NPU matmul requires K%32 and patch 14 gives 588, not 768.
static OrtStatus* MarshalDepth(const OrtApi& api, const GraphMaps& m,
                               const rocket_match::BackboneMatch& match, CompiledBackbone* cb) {
  std::vector<float> vbuf;
  const int nlayer = (int)match.layers.size();
  const int ntaps = (int)match.tap_layers.size();
  if (ntaps <= 0 || ntaps > ROCKET_SIGLIP_MAX_TAPS)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth tap count out of range");

  // --- geometry from the graph ---
  std::vector<std::string> pin;
  if (OrtStatus* st = NodeInputNames(api, match.patch_conv, &pin)) return st;
  if (pin.size() < 3 || pin[1].empty() || pin[2].empty())
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth patch conv missing weight/bias");
  RawInit pcw;
  if (OrtStatus* st = ReadInit(api, m, pin[1], &pcw)) return st;      // [d,ic,kh,kw]
  if (pcw.dims.size() != 4)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth patch conv weight is not rank-4");
  const int d = (int)pcw.dims[0], ic = (int)pcw.dims[1], kh = (int)pcw.dims[2], kw = (int)pcw.dims[3];
  const int patch_dim = ic * kh * kw;
  const int patch_dim_pad = (patch_dim + 31) & ~31;                  // NPU matmul needs K%32
  RawInit fc1r, posr;
  if (OrtStatus* st = ReadInit(api, m, InitInputTensor(api, m, match.layers[0].fc1), &fc1r)) return st;
  if (fc1r.dims.size() != 2 || (int)fc1r.dims[0] != d)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth fc1 weight shape unexpected");
  const int d_ff = (int)fc1r.dims[1];
  // The interpolated position embedding folds to a constant. It is rank-3 [1,ntok,d] here (DINOv2
  // keeps the batch axis) where the plain-ViT exports are rank-2, so accept either.
  if (OrtStatus* st = ReadInit(api, m, match.pos_embed, &posr)) return st;
  int ntok = 0;
  if (posr.dims.size() == 2 && (int)posr.dims[1] == d) ntok = (int)posr.dims[0];
  else if (posr.dims.size() == 3 && (int)posr.dims[0] == 1 && (int)posr.dims[2] == d) ntok = (int)posr.dims[1];
  else return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth position embedding shape unexpected");
  int n_head = 0;
  if (const char* hz = std::getenv("ROCKET_ORT_SIGLIP_HEADS")) n_head = std::atoi(hz);
  if (n_head <= 0) n_head = d / 64;                                  // DA v2 S/B/L are all head-dim 64
  if (n_head <= 0 || d % n_head != 0)
    return api.CreateStatus(ORT_INVALID_GRAPH,
        "ort-rocket: cannot infer depth n_head (set ROCKET_ORT_SIGLIP_HEADS)");
  int side = 1; while ((side + 1) * (side + 1) <= ntok) side++;
  const int L = side * side;
  const int n_prefix = ntok - L;                                     // the cls token
  if (n_prefix != 1 || kh != kw || kh != match.patch_size)
    return api.CreateStatus(ORT_INVALID_GRAPH,
        "ort-rocket: depth token count / patch geometry inconsistent (expected one cls token)");
  const float eps = ReadFloatAttr(api, match.layers[0].norm1, "epsilon", 1e-6f);

  // --- one contiguous fp16 buffer, laid out for rocket_siglip ---
  const size_t ls = (size_t)2 * d + 4 * ((size_t)d * d + d) + 2 * d +
                    ((size_t)d_ff * d + d_ff) + ((size_t)d * d_ff + d);   // layer_stride_elems
  const size_t total = (size_t)d * patch_dim_pad + d                      // patch_W (padded), patch_b
                     + (size_t)n_prefix * d                               // cls
                     + (size_t)ntok * d                                   // pos
                     + (size_t)nlayer * ls
                     + (size_t)2 * d;                                     // the shared tap LayerNorm
  _Float16* buf = cb->alloc(total);
  size_t off = 0;
  auto take = [&](size_t n) -> _Float16* { _Float16* p = buf + off; off += n; return p; };
  auto put = [&](const std::string& edge, size_t expect, _Float16* dst) -> OrtStatus* {
    if (OrtStatus* st = ReadTensorF32(api, m, edge, &vbuf)) return st;
    if (expect && vbuf.size() != expect)
      return api.CreateStatus(ORT_INVALID_GRAPH, ("ort-rocket: depth weight size mismatch: " + edge).c_str());
    ToF16(vbuf.data(), dst, vbuf.size());
    return nullptr;
  };
  auto ln_gb = [&](const OrtNode* ln, _Float16* g, _Float16* b) -> OrtStatus* {
    std::vector<std::string> in;
    if (OrtStatus* st = NodeInputNames(api, ln, &in)) return st;
    if (in.size() < 3) return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth LayerNorm missing gamma/beta");
    if (OrtStatus* st = put(in[1], d, g)) return st;
    return put(in[2], d, b);
  };

  rocket_siglip_model& sm = cb->sm;
  // stem: patch_W laid out [d][patch_dim_pad] with a zeroed K-tail (the encoder zero-fills the
  // matching im2col columns, so the padding cannot change the product).
  sm.patch_W = take((size_t)d * patch_dim_pad);
  {
    if (OrtStatus* st = ReadTensorF32(api, m, pin[1], &vbuf)) return st;
    if (vbuf.size() != (size_t)d * patch_dim)
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth patch conv weight size mismatch");
    _Float16* pw = const_cast<_Float16*>(sm.patch_W);
    std::memset(pw, 0, (size_t)d * patch_dim_pad * sizeof(_Float16));
    for (int o = 0; o < d; o++)
      for (int k = 0; k < patch_dim; k++)
        pw[(size_t)o * patch_dim_pad + k] = (_Float16)vbuf[(size_t)o * patch_dim + k];
  }
  sm.patch_b = take(d);
  if (OrtStatus* st = put(pin[2], d, const_cast<_Float16*>(sm.patch_b))) return st;
  sm.prefix = take((size_t)n_prefix * d);                            // cls token [1,1,d]
  if (OrtStatus* st = put(match.cls_embed, (size_t)n_prefix * d, const_cast<_Float16*>(sm.prefix))) return st;
  sm.pre_g = nullptr; sm.pre_b = nullptr;                            // DINOv2 has no pre-encoder LN
  sm.pos = take((size_t)ntok * d);
  if (OrtStatus* st = put(match.pos_embed, (size_t)ntok * d, const_cast<_Float16*>(sm.pos))) return st;

  sm.layers = take((size_t)nlayer * ls);
  std::vector<float> lam1v, lam2v;
  for (int Lx = 0; Lx < nlayer; Lx++) {
    const rocket_match::MatchedLayer& ml = match.layers[Lx];
    _Float16* lb = const_cast<_Float16*>(sm.layers) + (size_t)Lx * ls;
    size_t o = 0;
    auto slot = [&](size_t n) -> _Float16* { _Float16* p = lb + o; o += n; return p; };
    // LayerScale lambdas, folded into (Wo,bo) and (Wf2,bf2) below.
    if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, ml.ls1), &lam1v)) return st;
    if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, ml.ls2), &lam2v)) return st;
    if (lam1v.size() != (size_t)d || lam2v.size() != (size_t)d)
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth LayerScale lambda width mismatch");
    const float* lam1 = lam1v.data();
    const float* lam2 = lam2v.data();
    // The fused qkv weight [d,3d], sliced by the DERIVED per-role column block.
    std::vector<float> qkvw;
    if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, ml.q), &qkvw)) return st;
    if (qkvw.size() != (size_t)d * 3 * d)
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth fused qkv weight shape mismatch");
    if (ml.qkv_blk_q < 0 || ml.qkv_blk_k < 0 || ml.qkv_blk_v < 0)
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth qkv column blocks unresolved");
    // a bias/weight edge read straight into a slot, with an optional LayerScale fold
    auto putT = [&](const OrtNode* mm, int in, int out, const float* lam, _Float16* dst) -> OrtStatus* {
      if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, mm), &vbuf)) return st;
      if (vbuf.size() != (size_t)in * out)
        return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth projection weight shape mismatch");
      TransposeF16(vbuf.data(), in, out, lam, dst);
      return nullptr;
    };
    auto putBias = [&](const OrtNode* add, const float* lam, _Float16* dst, int n) -> OrtStatus* {
      if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, add), &vbuf)) return st;
      if (vbuf.size() != (size_t)n)
        return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: depth bias width mismatch");
      for (int i = 0; i < n; i++) dst[i] = (_Float16)(vbuf[i] * (lam ? lam[i] : 1.0f));
      return nullptr;
    };
    _Float16* ln1_g = slot(d); _Float16* ln1_b = slot(d);
    if (OrtStatus* st = ln_gb(ml.norm1, ln1_g, ln1_b)) return st;
    TransposeBlockF16(qkvw.data(), d, d, 3, ml.qkv_blk_q, slot((size_t)d * d));
    if (OrtStatus* st = putBias(ml.bq, nullptr, slot(d), d)) return st;
    TransposeBlockF16(qkvw.data(), d, d, 3, ml.qkv_blk_k, slot((size_t)d * d));
    if (OrtStatus* st = putBias(ml.bk, nullptr, slot(d), d)) return st;
    TransposeBlockF16(qkvw.data(), d, d, 3, ml.qkv_blk_v, slot((size_t)d * d));
    if (OrtStatus* st = putBias(ml.bv, nullptr, slot(d), d)) return st;
    if (OrtStatus* st = putT(ml.o, d, d, lam1, slot((size_t)d * d))) return st;      // LayerScale 1
    if (OrtStatus* st = putBias(ml.bo, lam1, slot(d), d)) return st;
    _Float16* ln2_g = slot(d); _Float16* ln2_b = slot(d);
    if (OrtStatus* st = ln_gb(ml.norm2, ln2_g, ln2_b)) return st;
    if (OrtStatus* st = putT(ml.fc1, d, d_ff, nullptr, slot((size_t)d_ff * d))) return st;
    if (OrtStatus* st = putBias(ml.bf1, nullptr, slot(d_ff), d_ff)) return st;
    if (OrtStatus* st = putT(ml.fc2, d_ff, d, lam2, slot((size_t)d * d_ff))) return st;   // LayerScale 2
    if (OrtStatus* st = putBias(ml.bf2, lam2, slot(d), d)) return st;
    if (o != ls) return api.CreateStatus(ORT_FAIL, "ort-rocket: depth layer stride mismatch (internal)");
  }

  // The tap LayerNorm. All taps share one affine (they are one nn.LayerNorm applied several times),
  // which the marshaling requires rather than assumes: a per-tap affine would need a per-tap slot.
  sm.post_g = take(d);
  sm.post_b = take(d);
  if (OrtStatus* st = ln_gb(match.tap_norms[0], const_cast<_Float16*>(sm.post_g),
                            const_cast<_Float16*>(sm.post_b))) return st;
  for (int t = 1; t < ntaps; t++) {
    std::vector<std::string> a, b;
    if (OrtStatus* st = NodeInputNames(api, match.tap_norms[0], &a)) return st;
    if (OrtStatus* st = NodeInputNames(api, match.tap_norms[t], &b)) return st;
    if (a.size() < 3 || b.size() < 3 || a[1] != b[1] || a[2] != b[2])
      return api.CreateStatus(ORT_INVALID_GRAPH,
          "ort-rocket: depth taps do not share one LayerNorm affine (unsupported variant)");
  }
  if (off != total) return api.CreateStatus(ORT_FAIL, "ort-rocket: depth buffer layout mismatch (internal)");

  sm.d = d; sm.n_layers = nlayer; sm.n_head = n_head; sm.d_ff = d_ff;
  sm.L = L; sm.patch_dim = patch_dim; sm.patch_dim_pad = patch_dim_pad;
  sm.ic = ic; sm.kh = kh; sm.kw = kw;
  sm.stride = match.patch_size; sm.image_size = side * match.patch_size; sm.eps = eps;
  sm.n_prefix = n_prefix; sm.act_kind = 2;            // DINOv2's exact erf GELU
  sm.n_taps = ntaps;
  for (int t = 0; t < ntaps; t++) sm.tap_layer[t] = match.tap_layers[t];
  sm.layer_stride = ls;
  sm.map = (void*)buf; sm.map_size = total * sizeof(_Float16);   // sentinel; NEVER rocket_siglip_free
  return nullptr;
}

// ---------------------------------------------------------------------------
// SAM ViT-Det marshaling (Family::SamVitDet).
// ---------------------------------------------------------------------------

// Resolve a decomposed rel-pos table from one of the attention's two Einsum nodes. The table
// operand is not a folded initializer: the traced get_rel_pos is Einsum <- Gather <- Transpose <-
// Reshape <- Resize <- INIT, where INIT is the raw rel_pos parameter [1, dhead, 2S-1] (the reshaped
// nn.Parameter). Since q_size==k_size the Resize is identity, so get_rel_pos reduces to the Toeplitz
// gather Rh[i,j,k] = param[(i-j)+(S-1), k], with param[r,k] = raw[0,k,r]. Fills `out` [S*S*dhead]
// (fp32) and reports S and dhead.
static OrtStatus* ResolveRelPos(const OrtApi& api, const GraphMaps& m, const OrtNode* einsum,
                                int* S_out, int* dhead_out, std::vector<float>* out) {
  std::vector<std::string> ein;
  if (OrtStatus* st = NodeInputNames(api, einsum, &ein)) return st;
  // walk each operand's data (input[0]) chain to a rank-3 initializer [1,dhead,2S-1].
  std::string raw_name;
  for (const std::string& start : ein) {
    std::string t = start;
    for (int hop = 0; hop < 16 && !t.empty(); hop++) {
      auto ii = m.init.find(t);
      if (ii != m.init.end()) {
        RawInit r;
        if (!ReadInit(api, m, t, &r) && r.dims.size() == 3) raw_name = t;
        break;   // hit an initializer on this operand: stop (rank-3 => the table)
      }
      auto p = m.producer.find(t);
      if (p == m.producer.end()) break;
      std::vector<std::string> pin;
      if (NodeInputNames(api, p->second, &pin) || pin.empty() || pin[0].empty()) break;
      t = pin[0];
    }
    if (!raw_name.empty()) break;
  }
  if (raw_name.empty())
    return api.CreateStatus(ORT_NOT_FOUND, "ort-rocket: SAM rel-pos raw parameter not found");
  RawInit r;
  if (OrtStatus* st = ReadInit(api, m, raw_name, &r)) return st;
  const int dhead = (int)r.dims[1], twoSm1 = (int)r.dims[2], S = (twoSm1 + 1) / 2;
  if (dhead <= 0 || twoSm1 <= 0 || (twoSm1 & 1) == 0)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM rel-pos param shape unexpected");
  std::vector<float> raw;
  if (OrtStatus* st = ReadTensorF32(api, m, raw_name, &raw)) return st;   // [dhead*(2S-1)]
  out->resize((size_t)S * S * dhead);
  for (int i = 0; i < S; i++)
    for (int j = 0; j < S; j++) {
      const int rr = (i - j) + (S - 1);
      for (int k = 0; k < dhead; k++)
        (*out)[((size_t)i * S + j) * dhead + k] = raw[(size_t)k * twoSm1 + rr];   // raw[0,k,rr]
    }
  *S_out = S; *dhead_out = dhead;
  return nullptr;
}

// Find the conv neck (conv1x1 -> LN -> conv3x3 -> LN) by walking the linear data spine back from the
// encoder exit (the neck output tensor). Crosses Transpose/Reshape/Cast; the two Conv and two
// LayerNormalization nodes are collected in reverse (exit->stem) order.
static OrtStatus* FindNeck(const OrtApi& api, const GraphMaps& m, const std::string& exit_tensor,
                           const OrtNode** conv1, const OrtNode** conv2,
                           const OrtNode** ln1, const OrtNode** ln2) {
  std::vector<const OrtNode*> convs, lns;
  std::string t = exit_tensor;
  for (int hop = 0; hop < 40 && !t.empty(); hop++) {
    auto p = m.producer.find(t);
    if (p == m.producer.end()) break;
    const OrtNode* n = p->second;
    const std::string op = NodeOp(api, n);
    if (op == "Conv") convs.push_back(n);
    else if (op == "LayerNormalization") lns.push_back(n);
    else if (op != "Transpose" && op != "Reshape" && op != "Cast") break;   // left the neck chain
    std::vector<std::string> in;
    if (NodeInputNames(api, n, &in) || in.empty()) break;
    t = in[0];
  }
  if (convs.size() != 2 || lns.size() != 2)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM neck (2 conv + 2 LN) not found");
  // reverse order: convs[0]=conv2 (nearest exit, 3x3), convs[1]=conv1 (1x1); lns[0]=ln2, lns[1]=ln1.
  *conv2 = convs[0]; *conv1 = convs[1];
  *ln2 = lns[0]; *ln1 = lns[1];
  return nullptr;
}

// Marshal the matched SAM ViT-Det graph into a rocket_sam_model: one contiguous fp16 buffer with the
// stem, per-layer weights (fused qkv transposed [d,3d]->[3d,d]; resolved rel-pos tables), and the
// conv neck. Geometry is read from the graph (patch conv, pos-embed, rel-pos param, fc1, conv1).
static OrtStatus* MarshalSam(const OrtApi& api, const GraphMaps& m,
                             const rocket_match::BackboneMatch& match, CompiledBackbone* cb) {
  std::vector<float> vbuf;
  const int nlayer = (int)match.layers.size();
  if (nlayer <= 0 || nlayer > ROCKET_SAM_MAX_LAYERS)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM layer count out of range");

  // --- geometry ---
  std::vector<std::string> pin;
  if (OrtStatus* st = NodeInputNames(api, match.patch_conv, &pin)) return st;
  if (pin.size() < 3 || pin[1].empty() || pin[2].empty())
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM patch conv missing weight/bias");
  RawInit pcw;
  if (OrtStatus* st = ReadInit(api, m, pin[1], &pcw)) return st;   // [d,ic,p,p]
  if (pcw.dims.size() != 4)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM patch conv weight not rank-4");
  const int d = (int)pcw.dims[0], ic = (int)pcw.dims[1], patch = (int)pcw.dims[2];
  const int patch_dim = ic * patch * patch;
  RawInit posr;
  if (OrtStatus* st = ReadInit(api, m, match.pos_embed, &posr)) return st;   // [1,grid,grid,d]
  if (posr.dims.size() != 4 || (int)posr.dims[3] != d || posr.dims[1] != posr.dims[2])
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM pos_embed shape unexpected");
  const int grid = (int)posr.dims[1];
  RawInit fc1r;
  if (OrtStatus* st = ReadInit(api, m, InitInputTensor(api, m, match.layers[0].fc1), &fc1r)) return st;
  if (fc1r.dims.size() != 2 || (int)fc1r.dims[0] != d)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM fc1 weight shape unexpected");
  const int d_ff = (int)fc1r.dims[1];

  // resolve every layer's rel-pos tables first (they set per-layer side S and the head dim).
  std::vector<std::vector<float>> relh(nlayer), relw(nlayer);
  std::vector<int> Sl(nlayer);
  int dhead = 0, win = 0;
  for (int L = 0; L < nlayer; L++) {
    int sh = 0, sw = 0, dh1 = 0, dh2 = 0;
    if (OrtStatus* st = ResolveRelPos(api, m, match.layers[L].rel_h, &sh, &dh1, &relh[L])) return st;
    if (OrtStatus* st = ResolveRelPos(api, m, match.layers[L].rel_w, &sw, &dh2, &relw[L])) return st;
    if (sh != sw || dh1 != dh2)
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM rel_h/rel_w geometry disagree");
    Sl[L] = sh;
    if (!dhead) dhead = dh1; else if (dhead != dh1)
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM head dim varies across layers");
    if (match.layers[L].windowed && !win) win = sh;   // a windowed layer's side is the window size
  }
  if (dhead <= 0 || d % dhead)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM head dim inconsistent");
  const int n_head = d / dhead;
  if (!win) win = grid;   // all-global (no windowed layer): win unused, default to grid
  // the global-layer side must equal the grid; validate.
  for (int L = 0; L < nlayer; L++)
    if (Sl[L] != (match.layers[L].windowed ? win : grid))
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM per-layer side / windowed mismatch");

  // --- neck weights ---
  const OrtNode *nc1 = nullptr, *nc2 = nullptr, *nln1 = nullptr, *nln2 = nullptr;
  if (OrtStatus* st = FindNeck(api, m, match.exit_tensor, &nc1, &nc2, &nln1, &nln2)) return st;
  RawInit nc1w;
  if (OrtStatus* st = ReadInit(api, m, InitInputTensor(api, m, nc1), &nc1w)) return st;   // [no,d,1,1]
  if (nc1w.dims.size() != 4 || (int)nc1w.dims[1] != d)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM neck conv1 shape unexpected");
  const int neck_out = (int)nc1w.dims[0];
  // disambiguate conv1 (1x1) vs conv2 (3x3) by kernel, in case the walk order ever differs.
  {
    RawInit a, b;
    if (OrtStatus* st = ReadInit(api, m, InitInputTensor(api, m, nc1), &a)) return st;
    if (OrtStatus* st = ReadInit(api, m, InitInputTensor(api, m, nc2), &b)) return st;
    if (a.dims.size() == 4 && b.dims.size() == 4 && a.dims[2] == 3 && b.dims[2] == 1) {
      const OrtNode* t = nc1; nc1 = nc2; nc2 = t;
      const OrtNode* u = nln1; nln1 = nln2; nln2 = u;
    }
  }
  const float eps = ReadFloatAttr(api, match.layers[0].norm1, "epsilon", 1e-6f);

  // --- size the contiguous fp16 buffer ---
  size_t total = (size_t)d * patch_dim + d + (size_t)grid * grid * d;   // patch_W, patch_b, pos
  for (int L = 0; L < nlayer; L++) {
    const int S = Sl[L];
    total += (size_t)2 * d                       // ln1 g,b
           + (size_t)3 * d * d + 3 * d           // Wqkv, bqkv
           + (size_t)d * d + d                   // Wo, bo
           + (size_t)2 * d                       // ln2 g,b
           + (size_t)d_ff * d + d_ff             // Wf1, bf1
           + (size_t)d * d_ff + d                // Wf2, bf2
           + (size_t)2 * S * S * dhead;          // Rh, Rw
  }
  total += (size_t)neck_out * d + 2 * neck_out   // neck_c1, ln1 g,b
         + (size_t)neck_out * neck_out * 9 + 2 * neck_out;  // neck_c2, ln2 g,b
  _Float16* buf = cb->alloc(total);
  size_t off = 0;
  auto take = [&](size_t n) -> _Float16* { _Float16* p = buf + off; off += n; return p; };
  auto put = [&](const std::string& edge, size_t expect, _Float16* dst) -> OrtStatus* {
    if (OrtStatus* st = ReadTensorF32(api, m, edge, &vbuf)) return st;
    if (expect && vbuf.size() != expect)
      return api.CreateStatus(ORT_INVALID_GRAPH, ("ort-rocket: SAM weight size mismatch: " + edge).c_str());
    ToF16(vbuf.data(), dst, vbuf.size());
    return nullptr;
  };
  auto putT = [&](const OrtNode* mm, int in, int out, _Float16* dst) -> OrtStatus* {
    const std::string edge = InitInputTensor(api, m, mm);
    if (OrtStatus* st = ReadTensorF32(api, m, edge, &vbuf)) return st;
    if (vbuf.size() != (size_t)in * out)
      return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM projection weight shape mismatch");
    TransposeF16(vbuf.data(), in, out, nullptr, dst);
    return nullptr;
  };
  auto ln_gb = [&](const OrtNode* ln, _Float16* g, _Float16* b) -> OrtStatus* {
    std::vector<std::string> in;
    if (OrtStatus* st = NodeInputNames(api, ln, &in)) return st;
    if (in.size() < 3) return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: SAM LayerNorm missing gamma/beta");
    if (OrtStatus* st = put(in[1], (size_t)0, g)) return st;
    return put(in[2], (size_t)0, b);
  };

  rocket_sam_model& sm = cb->sm2;
  // stem
  sm.patch_W = take((size_t)d * patch_dim);
  if (OrtStatus* st = put(pin[1], (size_t)d * patch_dim, const_cast<_Float16*>(sm.patch_W))) return st;
  sm.patch_b = take(d);
  if (OrtStatus* st = put(pin[2], d, const_cast<_Float16*>(sm.patch_b))) return st;
  sm.pos = take((size_t)grid * grid * d);
  if (OrtStatus* st = put(match.pos_embed, (size_t)grid * grid * d, const_cast<_Float16*>(sm.pos))) return st;

  for (int L = 0; L < nlayer; L++) {
    const rocket_match::MatchedLayer& ml = match.layers[L];
    const int S = Sl[L];
    _Float16* g1 = take(d); _Float16* b1 = take(d);
    if (OrtStatus* st = ln_gb(ml.norm1, g1, b1)) return st;
    sm.ln1_g[L] = g1; sm.ln1_b[L] = b1;
    sm.Wqkv[L] = take((size_t)3 * d * d);
    if (OrtStatus* st = putT(ml.q, d, 3 * d, const_cast<_Float16*>(sm.Wqkv[L]))) return st;   // [d,3d]->[3d,d]
    sm.bqkv[L] = take((size_t)3 * d);
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bq), (size_t)3 * d, const_cast<_Float16*>(sm.bqkv[L]))) return st;
    sm.Wo[L] = take((size_t)d * d);
    if (OrtStatus* st = putT(ml.o, d, d, const_cast<_Float16*>(sm.Wo[L]))) return st;
    sm.bo[L] = take(d);
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bo), d, const_cast<_Float16*>(sm.bo[L]))) return st;
    _Float16* g2 = take(d); _Float16* b2 = take(d);
    if (OrtStatus* st = ln_gb(ml.norm2, g2, b2)) return st;
    sm.ln2_g[L] = g2; sm.ln2_b[L] = b2;
    sm.Wf1[L] = take((size_t)d_ff * d);
    if (OrtStatus* st = putT(ml.fc1, d, d_ff, const_cast<_Float16*>(sm.Wf1[L]))) return st;   // [d,dff]->[dff,d]
    sm.bf1[L] = take(d_ff);
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bf1), d_ff, const_cast<_Float16*>(sm.bf1[L]))) return st;
    sm.Wf2[L] = take((size_t)d * d_ff);
    if (OrtStatus* st = putT(ml.fc2, d_ff, d, const_cast<_Float16*>(sm.Wf2[L]))) return st;   // [dff,d]->[d,dff]
    sm.bf2[L] = take(d);
    if (OrtStatus* st = put(InitInputTensor(api, m, ml.bf2), d, const_cast<_Float16*>(sm.bf2[L]))) return st;
    // rel-pos tables (already resolved to fp32; copy to fp16)
    sm.Rh[L] = take((size_t)S * S * dhead);
    ToF16(relh[L].data(), const_cast<_Float16*>(sm.Rh[L]), relh[L].size());
    sm.Rw[L] = take((size_t)S * S * dhead);
    ToF16(relw[L].data(), const_cast<_Float16*>(sm.Rw[L]), relw[L].size());
    sm.Sl[L] = S; sm.windowed[L] = ml.windowed ? 1 : 0;
  }

  // neck
  sm.neck_c1 = take((size_t)neck_out * d);
  if (OrtStatus* st = put(InitInputTensor(api, m, nc1), (size_t)neck_out * d, const_cast<_Float16*>(sm.neck_c1))) return st;
  sm.neck_ln1_g = take(neck_out); sm.neck_ln1_b = take(neck_out);
  if (OrtStatus* st = ln_gb(nln1, const_cast<_Float16*>(sm.neck_ln1_g), const_cast<_Float16*>(sm.neck_ln1_b))) return st;
  sm.neck_c2 = take((size_t)neck_out * neck_out * 9);
  if (OrtStatus* st = put(InitInputTensor(api, m, nc2), (size_t)neck_out * neck_out * 9, const_cast<_Float16*>(sm.neck_c2))) return st;
  sm.neck_ln2_g = take(neck_out); sm.neck_ln2_b = take(neck_out);
  if (OrtStatus* st = ln_gb(nln2, const_cast<_Float16*>(sm.neck_ln2_g), const_cast<_Float16*>(sm.neck_ln2_b))) return st;

  if (off != total) return api.CreateStatus(ORT_FAIL, "ort-rocket: SAM buffer layout mismatch (internal)");

  sm.d = d; sm.n_layers = nlayer; sm.n_head = n_head; sm.dhead = dhead; sm.d_ff = d_ff;
  sm.grid = grid; sm.win = win; sm.patch = match.patch_size; sm.image_size = grid * match.patch_size;
  sm.ic = ic; sm.patch_dim = patch_dim; sm.neck_out = neck_out; sm.eps = eps;
  sm.map = (void*)buf; sm.map_size = total * sizeof(_Float16);   // non-NULL sentinel; NEVER rocket_sam_free
  return nullptr;
}

// Marshal the CSP projector weights. Unlike the encoder projections (anonymous exporter names,
// resolved structurally), the projector weights carry PyTorch state_dict names, which survive an
// opset/exporter re-export -- only an RF-DETR projector-module refactor renames them, and that also
// changes the conv-stack topology the matcher keys on -- so they are read by those stable names.
// Conv weights kept in native [OC,IC,KH,KW] layout, the 1x1 weights as [OC,IC] for the matmul path --
// no transpose, no LayerScale; the channel-LN affine is its own gamma/beta.
static OrtStatus* MarshalProjector(const OrtApi& api, const GraphMaps& m, CompiledBackbone* cb) {
  const std::string PP = "backbone.0.projector.stages.0.0.";
  std::vector<float> vbuf;
  auto load = [&](const std::string& name, const _Float16** dst) -> OrtStatus* {
    if (OrtStatus* st = WeightByName(api, m, name, &vbuf)) return st;
    const float* d = vbuf.data(); size_t c = vbuf.size();
    *dst = cb->alloc(c); ToF16(d, const_cast<_Float16*>(*dst), c); return nullptr;
  };
  if (OrtStatus* st = load(PP + "cv1.conv.weight", &cb->pw.cv1_w)) return st;   // [256,1536,1,1]->[256*1536]
  if (OrtStatus* st = load(PP + "cv1.bn.weight",   &cb->pw.cv1_g)) return st;
  if (OrtStatus* st = load(PP + "cv1.bn.bias",     &cb->pw.cv1_b)) return st;
  if (OrtStatus* st = load(PP + "cv2.conv.weight", &cb->pw.cv2_w)) return st;   // [256,640,1,1]->[256*640]
  if (OrtStatus* st = load(PP + "cv2.bn.weight",   &cb->pw.cv2_g)) return st;
  if (OrtStatus* st = load(PP + "cv2.bn.bias",     &cb->pw.cv2_b)) return st;
  if (OrtStatus* st = load("backbone.0.projector.stages.0.1.weight", &cb->pw.fin_g)) return st;
  if (OrtStatus* st = load("backbone.0.projector.stages.0.1.bias",   &cb->pw.fin_b)) return st;
  for (int j = 0; j < 3; j++)
    for (int cc = 0; cc < 2; cc++) {
      const std::string mp = PP + "m." + std::to_string(j) + ".cv" + std::to_string(cc + 1) + ".";
      if (OrtStatus* st = load(mp + "conv.weight", &cb->pw.m_w[j][cc])) return st;   // [128,128,3,3]
      if (OrtStatus* st = load(mp + "bn.weight",   &cb->pw.m_g[j][cc])) return st;
      if (OrtStatus* st = load(mp + "bn.bias",     &cb->pw.m_b[j][cc])) return st;
    }
  cb->pw.eps = 1e-6f;
  return nullptr;
}

// ---------------------------------------------------------------------------
// Native-int8 projection marshaling (ROCKET_ORT_INT8). Retains the raw int8 codes + per-out-
// channel scales for the 5 encoder GEMMs; the fp16 marshaling above still runs (attention, bias,
// norms, projector stay fp16). Requires an int8-QDQ model with symmetric (zp==0) per-channel
// weights; any miss (fp32/int4/asymmetric) makes MarshalInt8 fail and the EP fall back to fp16.
// ---------------------------------------------------------------------------

// The DequantizeLinear feeding a matched node's weight (its input whose DQ quantized-source is an
// initializer). NULL if the node's weight is a plain fp32 initializer (not quantized) or absent.
static const OrtNode* WeightDQOfNodeRef(const OrtApi& api, const GraphMaps& m, const OrtNode* node) {
  std::vector<std::string> in;
  if (NodeInputNames(api, node, &in) != nullptr) return nullptr;
  for (const std::string& nm : in) {
    auto p = m.producer.find(nm);
    if (p != m.producer.end() && NodeOp(api, p->second) == "DequantizeLinear") {
      std::vector<std::string> din;
      if (NodeInputNames(api, p->second, &din) == nullptr && !din.empty() && m.init.count(din[0]))
        return p->second;
    }
  }
  return nullptr;
}

// Read a weight DequantizeLinear's int8 codes + per-channel scale + zero-point, validating int8
// symmetric quant. Fills raw accessors (data valid while the graph lives).
static OrtStatus* ReadInt8DQ(const OrtApi& api, const GraphMaps& m, const OrtNode* dq,
                             RawInit* q, RawInit* s, bool* zp_zero) {
  std::vector<std::string> in;
  if (OrtStatus* st = NodeInputNames(api, dq, &in)) return st;
  if (in.size() < 2) return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: int8 DQ missing scale");
  if (OrtStatus* st = ReadInit(api, m, in[0], q)) return st;
  if (OrtStatus* st = ReadInit(api, m, in[1], s)) return st;
  if (q->et != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
    return api.CreateStatus(ORT_INVALID_GRAPH,
        "ort-rocket: ROCKET_ORT_INT8 needs an int8-quantized weight (got a non-int8 dtype)");
  *zp_zero = true;
  if (in.size() >= 3 && !in[2].empty() && m.init.count(in[2])) {
    RawInit z;
    if (OrtStatus* st = ReadInit(api, m, in[2], &z)) return st;
    for (size_t i = 0; i < z.count; i++) if (QAt(z, i) != 0) { *zp_zero = false; break; }
  }
  return nullptr;
}

// The per-tensor activation scale + zero-point on a matched node's activation input (its in[0],
// produced by a DequantizeLinear). Sets *a_scale=0 (dynamic fallback) if the input is not quantized.
static OrtStatus* ActQuantOfNodeRef(const OrtApi& api, const GraphMaps& m, const OrtNode* node,
                                    float* a_scale, int* a_zp) {
  *a_scale = 0.f; *a_zp = 0;
  std::vector<std::string> in;
  if (OrtStatus* st = NodeInputNames(api, node, &in)) return st;
  if (in.empty()) return nullptr;
  auto p = m.producer.find(in[0]);
  if (p == m.producer.end() || NodeOp(api, p->second) != "DequantizeLinear") return nullptr;
  std::vector<std::string> din;
  if (OrtStatus* st = NodeInputNames(api, p->second, &din)) return st;
  if (din.size() < 2) return nullptr;
  RawInit s;
  if (OrtStatus* st = ReadInit(api, m, din[1], &s)) return st;
  if (s.count != 1) return nullptr;                       // expect a per-tensor activation scale
  *a_scale = static_cast<const float*>(s.data)[0];
  if (din.size() >= 3 && !din[2].empty() && m.init.count(din[2])) {
    RawInit z;
    if (OrtStatus* st = ReadInit(api, m, din[2], &z)) return st;
    // i8_gemm quantizes activations into int8_t [-128,127]. A quint8-activation model
    // (--activation-type quint8) has a uint8 zero-point ~128 and codes in [0,255]; storing those as
    // int8_t saturates the entire positive half -> silent-wrong. The zero-point dtype mirrors the
    // quantized activation dtype, so reject uint8 here. This fails MarshalInt8 and the caller falls
    // back to fp16 (or, under STRICT, aborts) rather than silently miscomputing.
    if (z.et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
      return api.CreateStatus(ORT_INVALID_GRAPH,
          "ort-rocket: ROCKET_ORT_INT8 needs int8 (qint8) activations; this model quantized "
          "activations as uint8 (quint8), which i8_gemm cannot represent -- re-export with "
          "--activation-type qint8, or use the default fp16 path");
    if (z.count == 1) *a_zp = QAt(z, 0);
  }
  return nullptr;
}

// Marshal a projection weight (MatMul, ONNX [K=in, N=out], per-channel axis=1) to rocket [N,K]
// int8 + scale[N], plus the input-activation static scale/zero-point and the per-out-channel
// weight row-sums (for the zero-point fold in i8_gemm).
static OrtStatus* MarshalInt8Proj(const OrtApi& api, const GraphMaps& m, CompiledBackbone* cb,
                                  const OrtNode* mm, const char* role, int N, int K, const float* lam,
                                  rocket_i8_proj* proj) {
  const OrtNode* dq = WeightDQOfNodeRef(api, m, mm);
  if (!dq) return api.CreateStatus(ORT_INVALID_GRAPH,
      (std::string("ort-rocket: no int8 weight DQ on matched node for ") + role).c_str());
  RawInit q, s; bool zpz = true;
  if (OrtStatus* st = ReadInt8DQ(api, m, dq, &q, &s, &zpz)) return st;
  if (!zpz) return api.CreateStatus(ORT_INVALID_GRAPH,
      "ort-rocket: ROCKET_ORT_INT8 needs symmetric (zero-point 0) weight quant");
  if (q.count != (size_t)K * N || s.count != (size_t)N)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: int8 projection shape mismatch");
  int8_t*  w  = cb->alloc_i8((size_t)N * K);
  float*   sc = cb->alloc_sc(N);
  int32_t* rs = cb->alloc_rs(N);
  const int8_t* src = static_cast<const int8_t*>(q.data);
  // Fold DINOv2 LayerScale (lam, the per-out-channel lambda on Wo/Wf2) into the per-channel
  // dequant scale: the int8 output is b_scale[n]*(A.W), so scaling b_scale[n] by lam[n] applies
  // LayerScale exactly (the zero-point correction rides along, and bo/bf2 already carry it).
  for (int n = 0; n < N; n++) {
    int32_t sum = 0;
    for (int k = 0; k < K; k++) { int8_t v = src[(size_t)k * N + n]; w[(size_t)n * K + k] = v; sum += v; }
    sc[n] = static_cast<const float*>(s.data)[n] * (lam ? lam[n] : 1.0f);   // [K,N]->[N,K] + LayerScale
    rs[n] = sum;
  }
  proj->w = w; proj->s = sc; proj->rowsum = rs;
  if (OrtStatus* st = ActQuantOfNodeRef(api, m, mm, &proj->a_scale, &proj->a_zp)) return st;
  return nullptr;
}

// Marshal the patch-embed conv weight (ONNX [D, IC, KH, KW] = [D, pk], per-channel axis=0) to
// rocket [D, pk_pad] int8 (real pk columns, zero tail) + scale[D]. The weight DQ is read from the
// matched patch Conv node's weight edge.
static OrtStatus* MarshalInt8Patch(const OrtApi& api, const GraphMaps& m, CompiledBackbone* cb,
                                   const OrtNode* conv, int D, int pk, int pk_pad,
                                   rocket_i8_proj* proj) {
  const OrtNode* dq = WeightDQOfNodeRef(api, m, conv);
  if (!dq)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: patch weight not int8-quantized");
  RawInit q, s; bool zpz = true;
  if (OrtStatus* st = ReadInt8DQ(api, m, dq, &q, &s, &zpz)) return st;
  if (!zpz) return api.CreateStatus(ORT_INVALID_GRAPH,
      "ort-rocket: ROCKET_ORT_INT8 needs symmetric patch weight quant");
  if (q.count != (size_t)D * pk || s.count != (size_t)D)
    return api.CreateStatus(ORT_INVALID_GRAPH, "ort-rocket: int8 patch shape mismatch");
  int8_t* w = cb->alloc_i8((size_t)D * pk_pad);   // zero-inited -> [pk, pk_pad) tail is 0
  float*  sc = cb->alloc_sc(D);
  const int8_t* src = static_cast<const int8_t*>(q.data);
  for (int d = 0; d < D; d++) {
    for (int k = 0; k < pk; k++) w[(size_t)d * pk_pad + k] = src[(size_t)d * pk + k];
    sc[d] = static_cast<const float*>(s.data)[d];
  }
  proj->w = w; proj->s = sc;
  return nullptr;
}

// Marshal all 5 encoder projections + patch as native int8, driven by the structural match. On any
// failure the caller falls back to the fp16 path (this is an opt-in faithfulness mode, never fatal).
static OrtStatus* MarshalInt8(const OrtApi& api, const GraphMaps& m,
                              const rocket_match::BackboneMatch& match, CompiledBackbone* cb) {
  const rocket_backbone_geom* g = cb->w.geom;
  if (OrtStatus* st = MarshalInt8Patch(api, m, cb, match.patch_conv,
                                       g->d, g->pk, g->pk_pad, &cb->i8w.patch)) return st;
  std::vector<float> lam1, lam2;
  for (int L = 0; L < g->nlayer; L++) {
    const rocket_match::MatchedLayer& ml = match.layers[L];
    // LayerScale lambdas (dequantized like any other weight); folded into Wo (lambda1) and Wf2
    // (lambda2), matching the fp16 marshaling's projw(..., lam1/lam2, ...).
    if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, ml.ls1), &lam1)) return st;
    if (OrtStatus* st = ReadTensorF32(api, m, InitInputTensor(api, m, ml.ls2), &lam2)) return st;
    if (OrtStatus* st = MarshalInt8Proj(api, m, cb, ml.q,   "q",   g->d,   g->d,   nullptr,     &cb->i8w.q[L]))  return st;
    if (OrtStatus* st = MarshalInt8Proj(api, m, cb, ml.k,   "k",   g->d,   g->d,   nullptr,     &cb->i8w.k[L]))  return st;
    if (OrtStatus* st = MarshalInt8Proj(api, m, cb, ml.v,   "v",   g->d,   g->d,   nullptr,     &cb->i8w.v[L]))  return st;
    if (OrtStatus* st = MarshalInt8Proj(api, m, cb, ml.o,   "o",   g->d,   g->d,   lam1.data(), &cb->i8w.o[L]))  return st;
    if (OrtStatus* st = MarshalInt8Proj(api, m, cb, ml.fc1, "fc1", g->dff, g->d,   nullptr,     &cb->i8w.f1[L])) return st;
    if (OrtStatus* st = MarshalInt8Proj(api, m, cb, ml.fc2, "fc2", g->d,   g->dff, lam2.data(), &cb->i8w.f2[L])) return st;
  }
  return nullptr;
}

// Validate the fused node has exactly one output (the projector feature map). Its identity is fixed
// structurally by the matcher's exit tensor -- the ancestor-set claim guarantees a single convex
// boundary -- so no output-name check is needed here.
static OrtStatus* MapOutputs(const OrtApi& api, const OrtNode* fused, CompiledBackbone* /*cb*/) {
  size_t no = 0;
  if (OrtStatus* st = api.Node_GetNumOutputs(fused, &no)) return st;
  if (no != 1)
    return api.CreateStatus(ORT_INVALID_GRAPH,
        ("ort-rocket: expected 1 fused output (the projector map), got " + std::to_string(no) +
         " (partition boundary is not the projector output)").c_str());
  return nullptr;
}

// A DPT head reads several encoder taps, so the fused node has one output per tap. Their ORDER is
// ORT's, not the matcher's, so resolve it by NAME: each fused output must be one of the tap tensors
// the claim seeded on, and every tap must appear exactly once. Compute then writes tap t into
// tap_out_slot[t]. Getting this wrong would silently feed the head's four reassemble stages the
// wrong depths, so a mismatch is fatal rather than best-effort.
static OrtStatus* MapDepthOutputs(const OrtApi& api, const OrtNode* fused,
                                  const rocket_match::BackboneMatch& match, CompiledBackbone* cb) {
  const size_t ntaps = match.tap_tensors.size();
  size_t no = 0;
  if (OrtStatus* st = api.Node_GetNumOutputs(fused, &no)) return st;
  if (no != ntaps)
    return api.CreateStatus(ORT_INVALID_GRAPH,
        ("ort-rocket: expected " + std::to_string(ntaps) + " fused outputs (one per DPT tap), got " +
         std::to_string(no)).c_str());
  std::vector<const OrtValueInfo*> outs(no);
  if (OrtStatus* st = api.Node_GetOutputs(fused, outs.data(), no)) return st;
  cb->tap_out_slot.assign(ntaps, no);   // `no` == unset sentinel (never a valid slot)
  for (size_t i = 0; i < no; i++) {
    if (!outs[i]) continue;             // omitted optional edge -> null (GetValueInfoName segfaults)
    const char* nm = nullptr;
    if (OrtStatus* st = api.GetValueInfoName(outs[i], &nm)) return st;
    if (!nm) continue;
    for (size_t t = 0; t < ntaps; t++)
      if (match.tap_tensors[t] == nm) { cb->tap_out_slot[t] = i; break; }
  }
  for (size_t t = 0; t < ntaps; t++)
    if (cb->tap_out_slot[t] == no)
      return api.CreateStatus(ORT_INVALID_GRAPH,
          ("ort-rocket: DPT tap " + std::to_string(t) + " (" + match.tap_tensors[t] +
           ") is not among the fused node's outputs").c_str());
  return nullptr;
}

OrtStatus* ORT_API_CALL RocketEp::CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                              const OrtNode** fused_nodes, size_t count,
                                              OrtNodeComputeInfo** node_compute_infos,
                                              OrtNode** /*ep_context_nodes*/) noexcept {
  auto* ep = static_cast<RocketEp*>(this_ptr);
  const OrtApi& api = ep->ort_api;
  if (count != 1)
    return api.CreateStatus(ORT_INVALID_ARGUMENT,
                            "ort-rocket: expected exactly one fused backbone subgraph");

  auto cb = std::make_unique<CompiledBackbone>(api);
  cb->logger = &ep->logger;
  cb->fd = rocket_open();
  if (cb->fd < 0)
    return api.CreateStatus(ORT_EP_FAIL,
                            ("ort-rocket: rocket_open failed (" + std::to_string(cb->fd) +
                             "); is /dev/accel/accel0 present and privileged?").c_str());
  // GetCapability already ran the structural match on the full graph and only claimed on success
  // (a non-matching graph is left entirely on CPU, never reaching here). This re-runs the match
  // on the fused subgraph -- which contains exactly the claimed nodes -- to drive edge-walk weight
  // marshaling. A failure here is an internal inconsistency (the claim and the marshal disagree), not
  // an export-name mismatch; surface it clearly. There is no CPU fallback at Compile, so it matters
  // that GetCapability shoulders the accept/reject decision up front.
  auto stage = [&](OrtStatus* st, const char* where) -> OrtStatus* {
    if (!st) return nullptr;
    const char* msg = api.GetErrorMessage(st);
    ep->Log(ORT_LOGGING_LEVEL_ERROR,
            std::string("ort-rocket: Compile failed in ") + where + ": " + (msg ? msg : "") +
                ". The structurally-claimed subgraph could not be marshaled; session creation will "
                "abort. This indicates a matcher/marshaler inconsistency, please report it.");
    return st;
  };
  const char* iz = std::getenv("ROCKET_ORT_INT8");
  const bool want_int8 = iz && std::atoi(iz) != 0;
  // ROCKET_ORT_STRICT: fail loudly on the pinned-export assumptions instead of silently. It does
  // not change the always-on guards below (geometry alignment and input shape/dtype are always
  // enforced); it adds a Compile-time self-check summary and escalates the otherwise-soft int8
  // marshaling miss from a silent fp16 fallback to a hard Compile failure.
  const char* sz = std::getenv("ROCKET_ORT_STRICT");
  const bool strict = sz && std::atoi(sz) != 0;

  // Structural match on the fused subgraph (drives the edge-walk marshaling below).
  rocket_match::BackboneMatch match = rocket_match::MatchAny(api, graphs[0]);
  if (!match.ok)
    return api.CreateStatus(ORT_EP_FAIL,
        (std::string("ort-rocket: the claimed subgraph did not re-match structurally at Compile (") +
         match.reason + "); this is an internal inconsistency").c_str());
  GraphMaps maps;
  if (OrtStatus* st = stage(BuildMaps(api, graphs[0], maps), "BuildMaps")) return st;

  // Depth Anything v2: the DINOv2 encoder marshals into the plain-ViT compute (LayerScale folded
  // into the projections, the fused qkv sliced by its derived column blocks), configured to emit one
  // normalized block per DPT tap. The head stays on the host, so the fused node has one output per
  // tap rather than one. Self-contained path, like SigLIP/SAM.
  if (match.family == rocket_match::Family::DepthAnythingDpt) {
    if (OrtStatus* st = stage(MarshalDepth(api, maps, match, cb.get()), "MarshalDepth")) return st;
    if (OrtStatus* st = stage(MapImageInput(api, fused_nodes[0], maps, cb.get()), "MapImageInput")) return st;
    if (OrtStatus* st = stage(MapDepthOutputs(api, fused_nodes[0], match, cb.get()), "MapDepthOutputs")) return st;
    cb->is_depth = true;
    const char* rz = std::getenv("ROCKET_ORT_RESIDENT");
    if (!(rz && std::atoi(rz) == 0)) {   // resident by default; one-shot when ROCKET_ORT_RESIDENT=0
      int nthreads = 3;
      if (const char* tz = std::getenv("ROCKET_ORT_THREADS")) { int v = std::atoi(tz); if (v >= 1 && v <= 16) nthreads = v; }
      cb->sctx = rocket_siglip_ctx_create(&cb->sm, nthreads);
      if (!cb->sctx)
        ep->Log(ORT_LOGGING_LEVEL_WARNING,
                "ort-rocket: depth resident ctx create failed; the one-shot encoder does not serve a "
                "multi-tap model, so Compute will fail -- check the weight pack (IOVA limit?)");
    }
    std::string taps;
    for (size_t t = 0; t < match.tap_layers.size(); t++)
      taps += (t ? "," : "") + std::to_string(match.tap_layers[t]);
    ep->Log(ORT_LOGGING_LEVEL_INFO,
            std::string("ort-rocket: Depth Anything v2 encoder compiled; image input at index ") +
                std::to_string(cb->image_in_idx) + ", d=" + std::to_string(cb->sm.d) + " heads=" +
                std::to_string(cb->sm.n_head) + " layers=" + std::to_string(cb->sm.n_layers) +
                " ntok=" + std::to_string(cb->sm.L + cb->sm.n_prefix) + " taps=[" + taps + "] img=" +
                std::to_string(cb->sm.image_size) + ", mode=" + (cb->sctx ? "resident" : "one-shot"));
    auto info = std::make_unique<BackboneComputeInfo>(cb.release());
    node_compute_infos[0] = info.release();
    return nullptr;
  }

  // Plain-ViT (SigLIP) family: marshal into the rocket_siglip encoder and hand back its compute.
  // Its geometry/IO is not DINOv2's (no projector, output [1,L,d]), so it takes a self-contained
  // path and returns before the DINOv2 geometry/projector/int8 wiring below.
  if (match.family == rocket_match::Family::SiglipVit) {
    if (OrtStatus* st = stage(MarshalSiglip(api, maps, match, cb.get()), "MarshalSiglip")) return st;
    if (OrtStatus* st = stage(MapImageInput(api, fused_nodes[0], maps, cb.get()), "MapImageInput")) return st;
    if (OrtStatus* st = stage(MapOutputs(api, fused_nodes[0], cb.get()), "MapOutputs")) return st;
    cb->is_siglip = true;
    const char* rz = std::getenv("ROCKET_ORT_RESIDENT");
    if (!(rz && std::atoi(rz) == 0)) {   // resident by default; one-shot when ROCKET_ORT_RESIDENT=0
      int nthreads = 3;
      if (const char* tz = std::getenv("ROCKET_ORT_THREADS")) { int v = std::atoi(tz); if (v >= 1 && v <= 16) nthreads = v; }
      cb->sctx = rocket_siglip_ctx_create(&cb->sm, nthreads);
      if (!cb->sctx)
        ep->Log(ORT_LOGGING_LEVEL_WARNING,
                "ort-rocket: siglip resident ctx create failed; falling back to the one-shot encoder");
    }
    ep->Log(ORT_LOGGING_LEVEL_INFO,
            std::string("ort-rocket: plain-ViT (") + (cb->sm.n_prefix > 0 ? "CLIP" : "SigLIP") +
                ") encoder compiled; image input at index " + std::to_string(cb->image_in_idx) +
                ", d=" + std::to_string(cb->sm.d) + " heads=" + std::to_string(cb->sm.n_head) +
                " layers=" + std::to_string(cb->sm.n_layers) + " L=" + std::to_string(cb->sm.L) +
                " n_prefix=" + std::to_string(cb->sm.n_prefix) + " act=" +
                (cb->sm.act_kind ? "quick_gelu" : "gelu_tanh") + " img=" +
                std::to_string(cb->sm.image_size) + ", mode=" + (cb->sctx ? "resident" : "one-shot"));
    auto info = std::make_unique<BackboneComputeInfo>(cb.release());
    node_compute_infos[0] = info.release();
    return nullptr;
  }

  // SAM ViT-Det family: marshal into the rocket_sam encoder (its own geometry/IO -- output
  // [1,neck_out,grid,grid]), like the plain-ViT path, and return before the DINOv2 wiring below.
  if (match.family == rocket_match::Family::SamVitDet) {
    if (OrtStatus* st = stage(MarshalSam(api, maps, match, cb.get()), "MarshalSam")) return st;
    if (OrtStatus* st = stage(MapImageInput(api, fused_nodes[0], maps, cb.get()), "MapImageInput")) return st;
    if (OrtStatus* st = stage(MapOutputs(api, fused_nodes[0], cb.get()), "MapOutputs")) return st;
    cb->is_sam = true;
    const char* rz = std::getenv("ROCKET_ORT_RESIDENT");
    if (!(rz && std::atoi(rz) == 0)) {   // resident by default; one-shot when ROCKET_ORT_RESIDENT=0
      int nthreads = 3;
      if (const char* tz = std::getenv("ROCKET_ORT_THREADS")) { int v = std::atoi(tz); if (v >= 1 && v <= 16) nthreads = v; }
      cb->sctx2 = rocket_sam_ctx_create(&cb->sm2, nthreads);
      if (!cb->sctx2)
        ep->Log(ORT_LOGGING_LEVEL_WARNING,
                "ort-rocket: SAM resident ctx create failed; falling back to the one-shot encoder");
    }
    ep->Log(ORT_LOGGING_LEVEL_INFO,
            std::string("ort-rocket: SAM ViT-Det encoder compiled; image input at index ") +
                std::to_string(cb->image_in_idx) + ", d=" + std::to_string(cb->sm2.d) + " heads=" +
                std::to_string(cb->sm2.n_head) + " layers=" + std::to_string(cb->sm2.n_layers) +
                " grid=" + std::to_string(cb->sm2.grid) + " win=" + std::to_string(cb->sm2.win) +
                " neck_out=" + std::to_string(cb->sm2.neck_out) + " img=" +
                std::to_string(cb->sm2.image_size) + ", mode=" + (cb->sctx2 ? "resident" : "one-shot"));
    auto info = std::make_unique<BackboneComputeInfo>(cb.release());
    node_compute_infos[0] = info.release();
    return nullptr;
  }

  // Only the DINOv2/RF-DETR family remains wired below. Reaching here with another family is an
  // internal inconsistency (GetCapability claims only wired families) -- fail loudly, no fallback.
  if (match.family != rocket_match::Family::DinoV2RfDetr)
    return api.CreateStatus(ORT_EP_FAIL,
        "ort-rocket: matched a backbone family with no Compile marshaling path");

  if (OrtStatus* st = stage(MarshalWeights(api, maps, match, cb.get()), "MarshalWeights")) return st;
  if (OrtStatus* st = stage(MarshalProjector(api, maps, cb.get()), "MarshalProjector")) return st;
  if (OrtStatus* st = stage(MapImageInput(api, fused_nodes[0], maps, cb.get()), "MapImageInput")) return st;
  if (OrtStatus* st = stage(MapOutputs(api, fused_nodes[0], cb.get()), "MapOutputs")) return st;

  // Reject a geometry whose M (ntok/np) is not %4 -- the prepacked matmul would return garbage
  // at rc=0. Always-on; both shipping variants pass, so this only fires on a future table entry.
  {
    const char* why = nullptr;
    if (rocket_backbone_geom_check(cb->w.geom, &why) != 0)
      return api.CreateStatus(ORT_INVALID_GRAPH,
          (std::string("ort-rocket: selected geometry fails a matmul-alignment invariant: ") +
           (why ? why : "unknown") + " -- this variant cannot compute correctly on the NPU").c_str());
  }
  // Compile-time image-input check: validate the image input dtype/shape against the selected geometry.
  bool shape_concrete = false;
  if (OrtStatus* st = stage(ValidateImageShape(api, fused_nodes[0], cb.get(), &shape_concrete),
                            "ValidateImageShape"))
    return st;

  // Opt-in native-int8 encoder projections. Marshals the raw int8 weights on top of the fp16
  // ones; on any miss (fp32/int4/asymmetric/quint8-activation model) it logs and falls back to the
  // fp16 path -- never fatal, unless ROCKET_ORT_STRICT escalates it. Slower than fp16 on this stack
  // (int8 matmul reads int32 back to the host); a faithfulness/experimentation mode, not the
  // default. See rocket_backbone.h.
  if (want_int8) {
    if (OrtStatus* st = MarshalInt8(api, maps, match, cb.get())) {
      const char* msg = api.GetErrorMessage(st);
      if (strict) {
        ep->Log(ORT_LOGGING_LEVEL_ERROR, std::string("ort-rocket: ROCKET_ORT_INT8 marshaling failed "
                "under ROCKET_ORT_STRICT (") + (msg ? msg : "") + "); aborting rather than falling "
                "back to fp16");
        return st;
      }
      ep->Log(ORT_LOGGING_LEVEL_WARNING, std::string("ort-rocket: ROCKET_ORT_INT8 set but native-int8 "
              "marshaling failed (") + (msg ? msg : "") + "); using the fp16 path");
      DiscardStatus(api, st);
      cb->w.i8 = nullptr;
    } else {
      cb->w.i8 = &cb->i8w;
      ep->Log(ORT_LOGGING_LEVEL_INFO, "ort-rocket: native-int8 encoder projections enabled (W8A8 on "
              "the NPU; slower than fp16 -- faithfulness mode)");
    }
  }

  if (strict) {
    const rocket_backbone_geom* g = cb->w.geom;
    ep->Log(ORT_LOGGING_LEVEL_INFO,
            "ort-rocket: STRICT self-check OK -- variant patch=" + std::to_string(g->patch) +
                " img=" + std::to_string(g->img) + " ntok=" + std::to_string(g->ntok) + " np=" +
                std::to_string(g->np) + " (M%4 ok, N%32 ok); image input idx=" +
                std::to_string(cb->image_in_idx) + " expects fp32 [1," + std::to_string(RBB_IC) +
                "," + std::to_string(g->img) + "," + std::to_string(g->img) + "]" +
                (shape_concrete ? "" : " (input shape is not fully pinned at Compile; the element "
                                       "count is enforced on every Compute)"));
  }

  // Build the resident (prepacked, multicore) contexts once, packing every static weight into
  // NPU BOs so Compute re-packs nothing per image. Disable with ROCKET_ORT_RESIDENT=0 (the
  // one-shot chain, kept as the A/B reference). Worker count via ROCKET_ORT_THREADS (default
  // 3 -- matching the 3 NPU cores; more workers oversubscribe and slow the backbone). A
  // ctx-create failure (e.g. an IOVA-limit weight pack) is not fatal -- log and fall back to
  // the one-shot chain rather than fail the session.
  const char* rz = std::getenv("ROCKET_ORT_RESIDENT");
  bool want_resident = !(rz && std::atoi(rz) == 0);
  if (want_resident) {
    int nthreads = 3;
    if (const char* tz = std::getenv("ROCKET_ORT_THREADS")) {
      int v = std::atoi(tz);
      if (v >= 1 && v <= 16) nthreads = v;
    }
    cb->bctx = rocket_backbone_ctx_create(&cb->w, nthreads);
    cb->pctx = rocket_projector_ctx_create(&cb->pw, nthreads);
    if (!cb->bctx || !cb->pctx) {
      ep->Log(ORT_LOGGING_LEVEL_WARNING,
              "ort-rocket: resident ctx create failed; falling back to the one-shot chain");
      if (cb->bctx) { rocket_backbone_ctx_free(cb->bctx); cb->bctx = nullptr; }
      if (cb->pctx) { rocket_projector_ctx_free(cb->pctx); cb->pctx = nullptr; }
    }
  }

  ep->Log(ORT_LOGGING_LEVEL_INFO,
          std::string("ort-rocket: backbone+projector compiled; image input at index ") +
              std::to_string(cb->image_in_idx) + ", weights marshaled, NPU fd open, mode=" +
              (cb->bctx ? "resident" : "one-shot"));
  auto info = std::make_unique<BackboneComputeInfo>(cb.release());
  node_compute_infos[0] = info.release();
  return nullptr;
}

void ORT_API_CALL RocketEp::ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                        OrtNodeComputeInfo** infos,
                                                        size_t count) noexcept {
  for (size_t i = 0; i < count; i++) {
    auto* info = static_cast<BackboneComputeInfo*>(infos[i]);
    delete info->state;
    delete info;
  }
}

// ---------------------------------------------------------------------------
// OrtEpFactory: reports the device and creates the compiling EP.
// ---------------------------------------------------------------------------
struct RocketEpFactory : OrtEpFactory {
  RocketEpFactory(const char* ep_name, const OrtApi& api, const OrtEpApi& ep, const OrtLogger& lg)
      : OrtEpFactory{}, ort_api(api), ep_api(ep), name(ep_name), logger(lg) {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;
    IsStreamAware = IsStreamAwareImpl;
  }

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  std::string name;
  const OrtLogger& logger;
  const std::string vendor = "ort-rocket";
  const uint32_t vendor_id = 0x726B;  // 'rk'
  const std::string version = "0.5.0";  // 5 model families; int8/int4 QDQ-ONNX consumption (dequant->fp16)

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* f) noexcept {
    return static_cast<const RocketEpFactory*>(f)->name.c_str();
  }
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* f) noexcept {
    return static_cast<const RocketEpFactory*>(f)->vendor.c_str();
  }
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* f) noexcept {
    return static_cast<const RocketEpFactory*>(f)->vendor_id;
  }
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* f) noexcept {
    return static_cast<const RocketEpFactory*>(f)->version.c_str();
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices, OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices, size_t* p_num) noexcept {
    auto* f = static_cast<RocketEpFactory*>(this_ptr);
    size_t& n = *p_num;
    n = 0;
    for (size_t i = 0; i < num_devices && n < max_ep_devices; ++i) {
      if (f->ort_api.HardwareDevice_Type(devices[i]) == OrtHardwareDeviceType_CPU) {
        OrtEpDevice* dev = nullptr;
        if (OrtStatus* st = f->ep_api.CreateEpDevice(f, devices[i], nullptr, nullptr, &dev)) return st;
        ep_devices[n++] = dev;
      }
    }
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              size_t num_devices, const OrtSessionOptions* /*so*/,
                                              const OrtLogger* logger, OrtEp** ep) noexcept {
    auto* f = static_cast<RocketEpFactory*>(this_ptr);
    *ep = nullptr;
    if (num_devices != 1)
      return f->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "ort-rocket EP is registered for a single CPU device only");
    *ep = std::make_unique<RocketEp>(f->ort_api, f->name, *logger).release();
    return nullptr;
  }

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
    delete static_cast<RocketEp*>(ep);
  }
  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* /*t*/, const OrtMemoryInfo* /*mi*/,
                                                     const OrtKeyValuePairs* /*o*/, OrtAllocator** a) noexcept {
    *a = nullptr; return nullptr;
  }
  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*t*/, OrtAllocator* /*a*/) noexcept {}
  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* /*t*/, OrtDataTransferImpl** dt) noexcept {
    *dt = nullptr; return nullptr;
  }
  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*t*/) noexcept { return false; }
};

}  // namespace

extern "C" {

__attribute__((visibility("default")))
OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger, OrtEpFactory** factories,
                             size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  if (max_factories < 1)
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "ort-rocket needs at least one factory slot");
  factories[0] = std::make_unique<RocketEpFactory>(registration_name, *ort_api, *ep_api,
                                                   *default_logger).release();
  *num_factories = 1;
  return nullptr;
}

__attribute__((visibility("default")))
OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<RocketEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
