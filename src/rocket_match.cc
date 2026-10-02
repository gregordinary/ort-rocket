// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_match -- see rocket_match.h. Structural op-topology matcher. This is a direct transcription
// of a prototype validated against all four shipping RF-DETR exports (weight-tensor identity per role
// and byte-identical claimed-node set vs the previous name-based marshaling). The walks treat a QDQ
// model's Quantize/Dequantize pairs as transparent layout ops, so one matcher serves fp32, int8, and
// int4 exports.

#include "rocket_match.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocket_match {
namespace {

// Layout ops crossed as transparent by the walks: pure reshapes plus the Q/DQ pair a QDQ model wraps
// tensors in. Bias Add and the attention-scale Mul are crossed only where noted (they carry data).
bool IsLayout(const std::string& op) {
  return op == "Reshape" || op == "Transpose" || op == "Squeeze" || op == "Unsqueeze" ||
         op == "Cast" || op == "Identity" || op == "DequantizeLinear" || op == "QuantizeLinear";
}

// Cached, name-free view of the graph: op type, input/output tensor edges, initializer shapes, and
// producer/consumer maps. Everything the matcher needs, built once. Any OrtApi failure sets `err`,
// which MatchBackbone treats as a non-match.
class Ctx {
 public:
  Ctx(const OrtApi& api, const OrtGraph* g) : api_(api) { Build(g); }
  bool err = false;

  const std::vector<const OrtNode*>& nodes() const { return nodes_; }
  const std::string& op(const OrtNode* n) const { return info(n).op; }
  const std::vector<std::string>& ins(const OrtNode* n) const { return info(n).ins; }
  const std::vector<std::string>& outs(const OrtNode* n) const { return info(n).outs; }
  int order(const OrtNode* n) const { auto it = order_.find(n); return it == order_.end() ? -1 : it->second; }

  const OrtNode* prod(const std::string& t) const {
    auto it = producer_.find(t); return it == producer_.end() ? nullptr : it->second;
  }
  const std::vector<const OrtNode*>& cons(const std::string& t) const {
    static const std::vector<const OrtNode*> empty;
    auto it = consumers_.find(t); return it == consumers_.end() ? empty : it->second;
  }
  bool is_init(const std::string& t) const { return inits_.count(t) != 0; }
  const std::vector<int64_t>* init_shape(const std::string& t) const {
    auto it = initshape_.find(t); return it == initshape_.end() ? nullptr : &it->second;
  }

  // A tensor resolves to a weight if it is an initializer, or (QDQ) the output of a DequantizeLinear
  // whose quantized input is an initializer. Returns whether it resolves (the underlying weight is
  // read later by the EP, following the same DQ).
  bool resolves_to_init(const std::string& t) const {
    if (is_init(t)) return true;
    const OrtNode* p = prod(t);
    return p && op(p) == "DequantizeLinear" && !ins(p).empty() && is_init(ins(p)[0]);
  }
  // The node's single non-weight ("activation") input; "" if none.
  std::string act_input(const OrtNode* n) const {
    for (const std::string& t : ins(n))
      if (!t.empty() && !resolves_to_init(t)) return t;
    return std::string();
  }
  bool has_init(const OrtNode* n) const {
    for (const std::string& t : ins(n))
      if (!t.empty() && resolves_to_init(t)) return true;
    return false;
  }

  // kernel_shape / strides int-list attribute.
  std::vector<int64_t> attr_ints(const OrtNode* n, const char* name) const {
    std::vector<int64_t> out;
    const OrtOpAttr* attr = nullptr;
    OrtStatus* st = api_.Node_GetAttributeByName(n, name, &attr);
    if (st) { api_.ReleaseStatus(st); return out; }
    if (!attr) return out;
    int64_t buf[16]; size_t used = 0;
    st = api_.ReadOpAttr(attr, OrtOpAttrType::ORT_OP_ATTR_INTS, buf, sizeof(buf), &used);
    if (st) { api_.ReleaseStatus(st); return out; }
    for (size_t i = 0; i < used / sizeof(int64_t); i++) out.push_back(buf[i]);
    return out;
  }

 private:
  struct NInfo { std::string op; std::vector<std::string> ins, outs; };
  const OrtApi& api_;
  std::vector<const OrtNode*> nodes_;
  std::unordered_map<const OrtNode*, NInfo> info_;
  std::unordered_map<const OrtNode*, int> order_;
  std::unordered_map<std::string, const OrtNode*> producer_;
  std::unordered_map<std::string, std::vector<const OrtNode*>> consumers_;
  std::unordered_set<std::string> inits_;
  std::unordered_map<std::string, std::vector<int64_t>> initshape_;

  const NInfo& info(const OrtNode* n) const {
    static const NInfo empty;
    auto it = info_.find(n); return it == info_.end() ? empty : it->second;
  }
  bool names_of(const OrtValueInfo* const* vis, size_t n, std::vector<std::string>* out) {
    for (size_t k = 0; k < n; k++) {
      // An omitted OPTIONAL input (e.g. Pad's constant_value/axes, Slice's steps, Resize's roi) is
      // returned as a null OrtValueInfo* -- keep it as an empty "" edge (all the walks already skip
      // empty edges) rather than dereferencing it, which would segfault. RF-DETR/CLIP/SigLIP have no
      // such ops; SAM's window_partition Pad + unpartition Slice are the first to exercise this.
      if (!vis[k]) { out->emplace_back(); continue; }
      const char* nm = nullptr;
      if (OrtStatus* st = api_.GetValueInfoName(vis[k], &nm)) { api_.ReleaseStatus(st); err = true; return false; }
      out->emplace_back(nm ? nm : "");
    }
    return true;
  }
  void Build(const OrtGraph* g) {
    size_t nn = 0;
    if (OrtStatus* st = api_.Graph_GetNumNodes(g, &nn)) { api_.ReleaseStatus(st); err = true; return; }
    nodes_.resize(nn);
    if (OrtStatus* st = api_.Graph_GetNodes(g, nodes_.data(), nn)) { api_.ReleaseStatus(st); err = true; return; }
    for (size_t i = 0; i < nn; i++) {
      const OrtNode* n = nodes_[i];
      order_[n] = static_cast<int>(i);
      NInfo ni;
      const char* op = nullptr;
      if (OrtStatus* st = api_.Node_GetOperatorType(n, &op)) { api_.ReleaseStatus(st); err = true; return; }
      ni.op = op ? op : "";
      size_t no = 0, nin = 0;
      if (OrtStatus* st = api_.Node_GetNumOutputs(n, &no)) { api_.ReleaseStatus(st); err = true; return; }
      std::vector<const OrtValueInfo*> outs(no);
      if (OrtStatus* st = api_.Node_GetOutputs(n, outs.data(), no)) { api_.ReleaseStatus(st); err = true; return; }
      if (!names_of(outs.data(), no, &ni.outs)) return;
      if (OrtStatus* st = api_.Node_GetNumInputs(n, &nin)) { api_.ReleaseStatus(st); err = true; return; }
      std::vector<const OrtValueInfo*> insv(nin);
      if (OrtStatus* st = api_.Node_GetInputs(n, insv.data(), nin)) { api_.ReleaseStatus(st); err = true; return; }
      if (!names_of(insv.data(), nin, &ni.ins)) return;
      for (const std::string& o : ni.outs) if (!o.empty()) producer_[o] = n;
      for (const std::string& in : ni.ins) if (!in.empty()) consumers_[in].push_back(n);
      info_[n] = std::move(ni);
    }
    // initializers: names + shapes
    size_t ni = 0;
    if (OrtStatus* st = api_.Graph_GetNumInitializers(g, &ni)) { api_.ReleaseStatus(st); err = true; return; }
    std::vector<const OrtValueInfo*> inits(ni);
    if (OrtStatus* st = api_.Graph_GetInitializers(g, inits.data(), ni)) { api_.ReleaseStatus(st); err = true; return; }
    for (const OrtValueInfo* vi : inits) {
      const char* nm = nullptr;
      if (OrtStatus* st = api_.GetValueInfoName(vi, &nm)) { api_.ReleaseStatus(st); err = true; return; }
      if (!nm) continue;
      inits_.insert(nm);
      const OrtTypeInfo* ti = nullptr;
      if (OrtStatus* st = api_.GetValueInfoTypeInfo(vi, &ti)) { api_.ReleaseStatus(st); continue; }
      const OrtTensorTypeAndShapeInfo* tsi = nullptr;
      if (OrtStatus* st = api_.CastTypeInfoToTensorInfo(ti, &tsi)) { api_.ReleaseStatus(st); continue; }
      if (!tsi) continue;
      size_t nd = 0;
      if (OrtStatus* st = api_.GetDimensionsCount(tsi, &nd)) { api_.ReleaseStatus(st); continue; }
      std::vector<int64_t> dims(nd);
      if (nd) { if (OrtStatus* st = api_.GetDimensions(tsi, dims.data(), nd)) { api_.ReleaseStatus(st); continue; } }
      initshape_[nm] = std::move(dims);
    }
  }
};

// ---- topology walks (mirror the validated prototype) ----------------------------------

// Back-walk from a projection output tensor to its producing MatMul, crossing layout ops, the bias
// Add, and the attention-scale Mul (its scale operand is a computed subgraph, so we follow whichever
// operand actually reaches a MatMul). Bounded DFS.
const OrtNode* ProjMatmulBack(const Ctx& c, const std::string& t, int depth = 14) {
  const OrtNode* p = c.prod(t);
  if (!p || depth < 0) return nullptr;
  const std::string& o = c.op(p);
  if (o == "MatMul") return p;
  // Split is crossed for SAM's fused-qkv unbind (one qkv MatMul -> Split -> Squeeze -> q/k/v); it is
  // inert for the DINOv2/plain-ViT families, whose attention paths contain no Split.
  if (!(IsLayout(o) || o == "Add" || o == "Mul" || o == "Split")) return nullptr;
  for (const std::string& x : c.ins(p)) {
    if (x.empty() || c.resolves_to_init(x)) continue;   // skip weight/bias/scale-const operands
    if (const OrtNode* r = ProjMatmulBack(c, x, depth - 1)) return r;
  }
  return nullptr;
}

// Skip backward through layout ops (incl a Q/DQ wrap) to the producing MatMul.
const OrtNode* MatmulBackLayout(const Ctx& c, const std::string& t0) {
  std::string t = t0; std::unordered_set<std::string> seen;
  while (!t.empty() && !seen.count(t)) {
    seen.insert(t);
    const OrtNode* p = c.prod(t);
    if (!p) return nullptr;
    if (c.op(p) == "MatMul") return p;
    if (!IsLayout(c.op(p))) return nullptr;
    t = c.act_input(p);
  }
  return nullptr;
}

// Forward through layout ops to the next MatMul.
const OrtNode* FwdToMatmul(const Ctx& c, const std::string& t0) {
  std::vector<std::string> fr{t0}; std::unordered_set<std::string> seen;
  while (!fr.empty()) {
    std::string x = fr.front(); fr.erase(fr.begin());
    if (seen.count(x)) continue;
    seen.insert(x);
    for (const OrtNode* cc : c.cons(x)) {
      if (c.op(cc) == "MatMul") return cc;
      if (IsLayout(c.op(cc))) for (const std::string& o : c.outs(cc)) fr.push_back(o);
    }
  }
  return nullptr;
}

// Back through layout ops to a LayerNormalization.
const OrtNode* NormBack(const Ctx& c, const std::string& t0) {
  std::string t = t0; std::unordered_set<std::string> seen;
  while (!t.empty() && !seen.count(t)) {
    seen.insert(t);
    const OrtNode* p = c.prod(t);
    if (!p) return nullptr;
    if (c.op(p) == "LayerNormalization") return p;
    if (!IsLayout(c.op(p))) return nullptr;
    t = c.act_input(p);
  }
  return nullptr;
}

// BFS forward from t, crossing only layout ops, returning the first consumer satisfying pred.
const OrtNode* FwdFirst(const Ctx& c, const std::string& t0,
                        const std::function<bool(const OrtNode*)>& pred) {
  std::vector<std::string> fr{t0}; std::unordered_set<std::string> seen; int visited = 0;
  while (!fr.empty() && visited++ < 256) {
    std::string x = fr.front(); fr.erase(fr.begin());
    if (seen.count(x)) continue;
    seen.insert(x);
    for (const OrtNode* cc : c.cons(x)) {
      if (pred(cc)) return cc;
      if (IsLayout(c.op(cc))) for (const std::string& o : c.outs(cc)) fr.push_back(o);
    }
  }
  return nullptr;
}

// The bias Add for a projection MatMul (first Add-with-initializer reachable forward through layout).
const OrtNode* BiasAddOf(const Ctx& c, const OrtNode* mm) {
  if (!mm || c.outs(mm).empty()) return nullptr;
  return FwdFirst(c, c.outs(mm)[0], [&](const OrtNode* n) { return c.op(n) == "Add" && c.has_init(n); });
}
// Tensor after the bias add (or the MatMul output if none).
std::string AfterBias(const Ctx& c, const OrtNode* mm) {
  const OrtNode* add = BiasAddOf(c, mm);
  if (add && !c.outs(add).empty()) return c.outs(add)[0];
  return (mm && !c.outs(mm).empty()) ? c.outs(mm)[0] : std::string();
}

// LayerScale Mul (Mul with a [D] initializer operand) reachable forward from the o/fc2 bias-add,
// then the residual Add (an Add with NO initializer operand). Either may be null.
void LayerScaleAndResidual(const Ctx& c, const std::string& t, const OrtNode** ls, const OrtNode** res) {
  *ls = FwdFirst(c, t, [&](const OrtNode* n) { return c.op(n) == "Mul" && c.has_init(n); });
  *res = nullptr;
  if (*ls && !c.outs(*ls).empty())
    *res = FwdFirst(c, c.outs(*ls)[0], [&](const OrtNode* n) { return c.op(n) == "Add" && !c.has_init(n); });
}

// From norm2 output: fc1 (next MatMul through layout), then fc2 (next MatMul after fc1's bias-add,
// crossing the erf-GELU chain -- any op).
void FindMlp(const Ctx& c, const std::string& norm2_out, const OrtNode** fc1, const OrtNode** fc2) {
  *fc1 = FwdToMatmul(c, norm2_out);
  *fc2 = nullptr;
  if (!*fc1) return;
  std::vector<std::string> fr{AfterBias(c, *fc1)}; std::unordered_set<std::string> seen;
  while (!fr.empty()) {
    std::string x = fr.front(); fr.erase(fr.begin());
    if (x.empty() || seen.count(x)) continue;
    seen.insert(x);
    for (const OrtNode* cc : c.cons(x)) {
      if (c.op(cc) == "MatMul") { *fc2 = cc; return; }
      for (const std::string& o : c.outs(cc)) fr.push_back(o);
    }
  }
}

// From an attention Softmax, resolve q/k/v/o MatMuls and norm1.
bool ClassifyAttention(const Ctx& c, const OrtNode* sm, const OrtNode** q, const OrtNode** k,
                       const OrtNode** v, const OrtNode** o, const OrtNode** norm1) {
  if (c.ins(sm).empty()) return false;
  const OrtNode* scores = MatmulBackLayout(c, c.ins(sm)[0]);   // through any Q/DQ on the score output
  if (!scores || c.ins(scores).size() < 2) return false;
  *q = ProjMatmulBack(c, c.ins(scores)[0]);                    // scores input 0 = Q, input 1 = K
  *k = ProjMatmulBack(c, c.ins(scores)[1]);
  const OrtNode* ctx = c.outs(sm).empty() ? nullptr : FwdToMatmul(c, c.outs(sm)[0]);
  if (!ctx || c.ins(ctx).size() < 2) return false;
  // the V input is the ctx operand whose back-walk does NOT reach the softmax.
  auto reaches_sm = [&](std::string t) {
    std::unordered_set<std::string> seen; int d = 8;
    while (!t.empty() && !seen.count(t) && d-- > 0) {
      seen.insert(t);
      const OrtNode* p = c.prod(t);
      if (!p) return false;
      if (p == sm) return true;
      if (!IsLayout(c.op(p))) return false;
      t = c.act_input(p);
    }
    return false;
  };
  const std::string& v_in = reaches_sm(c.ins(ctx)[0]) ? c.ins(ctx)[1] : c.ins(ctx)[0];
  *v = ProjMatmulBack(c, v_in);
  *o = c.outs(ctx).empty() ? nullptr : FwdToMatmul(c, c.outs(ctx)[0]);
  if (!*q || !*k || !*v || !*o) return false;
  *norm1 = NormBack(c, c.act_input(*q));
  return true;
}

bool ResolveLayer(const Ctx& c, const OrtNode* sm, MatchedLayer* L) {
  const OrtNode *q, *k, *v, *o, *norm1;
  if (!ClassifyAttention(c, sm, &q, &k, &v, &o, &norm1)) return false;
  const OrtNode *ls1, *res1;
  LayerScaleAndResidual(c, AfterBias(c, o), &ls1, &res1);
  if (!res1) return false;
  const OrtNode* norm2 = c.outs(res1).empty() ? nullptr :
      FwdFirst(c, c.outs(res1)[0], [&](const OrtNode* n) { return c.op(n) == "LayerNormalization"; });
  if (!norm2) return false;
  const OrtNode *fc1, *fc2;
  FindMlp(c, c.outs(norm2).empty() ? std::string() : c.outs(norm2)[0], &fc1, &fc2);
  if (!fc1 || !fc2) return false;
  const OrtNode *ls2, *res2;
  LayerScaleAndResidual(c, AfterBias(c, fc2), &ls2, &res2);
  L->q = q; L->k = k; L->v = v; L->o = o; L->fc1 = fc1; L->fc2 = fc2;
  L->bq = BiasAddOf(c, q); L->bk = BiasAddOf(c, k); L->bv = BiasAddOf(c, v);
  L->bo = BiasAddOf(c, o); L->bf1 = BiasAddOf(c, fc1); L->bf2 = BiasAddOf(c, fc2);
  L->norm1 = norm1; L->norm2 = norm2; L->ls1 = ls1; L->ls2 = ls2;
  return L->bq && L->bk && L->bv && L->bo && L->bf1 && L->bf2 && L->norm1 && L->norm2 && L->ls1 && L->ls2;
}

// The stem patch-embed Conv: kernel_shape == strides (non-overlapping patches), weight [OC,3,K,K].
const OrtNode* FindPatchConv(const Ctx& c, int* patch) {
  for (const OrtNode* n : c.nodes()) {
    if (c.op(n) != "Conv" || c.ins(n).size() < 2) continue;
    std::vector<int64_t> ks = c.attr_ints(n, "kernel_shape"), st = c.attr_ints(n, "strides");
    if (ks.size() != 2 || ks != st) continue;
    if (!c.resolves_to_init(c.ins(n)[1])) continue;
    // weight shape via the (possibly DQ'd) initializer
    std::string wt = c.ins(n)[1];
    if (!c.is_init(wt)) { const OrtNode* dq = c.prod(wt); if (dq && !c.ins(dq).empty()) wt = c.ins(dq)[0]; }
    const std::vector<int64_t>* shp = c.init_shape(wt);
    if (!shp || shp->size() != 4) continue;
    if ((*shp)[1] != 3 || (*shp)[2] != ks[0]) continue;   // IC==3, KH==kernel
    *patch = static_cast<int>(ks[0]);
    return n;
  }
  return nullptr;
}

// Forward-reachable node set from a tensor (the patch-conv's descendants bound the encoder+projector).
void Descendants(const Ctx& c, const std::string& start, std::unordered_set<const OrtNode*>* out) {
  std::vector<std::string> fr{start};
  while (!fr.empty()) {
    std::string t = fr.back(); fr.pop_back();
    for (const OrtNode* cc : c.cons(t)) {
      if (out->count(cc)) continue;
      out->insert(cc);
      for (const std::string& o : c.outs(cc)) fr.push_back(o);
    }
  }
}

// Projector output tensor: the deepest tensor reachable from the LAST projector Conv (the only Convs
// besides the patch Conv) through the channel-LN/SiLU finishing ops. `best` advances only on a real
// projector op, never a trailing Quantize/Dequantize -- matching the pre-quant seed on a QDQ model.
std::string FindProjectorExit(const Ctx& c, const OrtNode* patch_conv) {
  std::unordered_set<const OrtNode*> desc;
  if (!c.outs(patch_conv).empty()) Descendants(c, c.outs(patch_conv)[0], &desc);
  const OrtNode* last = nullptr;
  for (const OrtNode* n : desc)
    if (c.op(n) == "Conv" && n != patch_conv && (!last || c.order(n) > c.order(last))) last = n;
  if (!last || c.outs(last).empty()) return std::string();
  auto finish = [&](const std::string& op) {
    return op == "Transpose" || op == "LayerNormalization" || op == "Sigmoid" || op == "Mul" ||
           op == "Cast" || op == "DequantizeLinear" || op == "QuantizeLinear";
  };
  std::string best = c.outs(last)[0]; int best_ord = c.order(last);
  std::vector<std::string> fr{c.outs(last)[0]}; std::unordered_set<std::string> seen;
  while (!fr.empty()) {
    std::string t = fr.back(); fr.pop_back();
    if (seen.count(t)) continue;
    seen.insert(t);
    const OrtNode* p = c.prod(t);
    if (p && c.op(p) != "QuantizeLinear" && c.op(p) != "DequantizeLinear" && c.order(p) > best_ord) {
      best = t; best_ord = c.order(p);
    }
    for (const OrtNode* cc : c.cons(t))
      if (finish(c.op(cc))) for (const std::string& o : c.outs(cc)) fr.push_back(o);
  }
  return best;
}

// ---- plain-ViT (CLIP/SigLIP) walks: the DINOv2 layer relaxed ---------------------------
// A pre-norm ViT encoder layer with NO LayerScale. Differences from ResolveLayer: the scores
// back-walk crosses the attention-scale Mul (SigLIP scales after Q.K^T, so ProjMatmulBack, not
// MatmulBackLayout); the residual Add follows the o/fc2 bias-add directly (no LayerScale Mul); and
// norm1 is REQUIRED -- the SigLIP MAP pooling head's attention has a learned-probe query with no
// pre-attention LayerNorm, so requiring norm1 drops it. ls1/ls2 stay null (plain ViT has none).
bool ResolvePlainLayer(const Ctx& c, const OrtNode* sm, MatchedLayer* L) {
  if (c.ins(sm).empty()) return false;
  const OrtNode* scores = ProjMatmulBack(c, c.ins(sm)[0]);   // crosses the scale Mul
  if (!scores || c.ins(scores).size() < 2) return false;
  const OrtNode* q = ProjMatmulBack(c, c.ins(scores)[0]);
  const OrtNode* k = ProjMatmulBack(c, c.ins(scores)[1]);
  const OrtNode* ctx = c.outs(sm).empty() ? nullptr : FwdToMatmul(c, c.outs(sm)[0]);
  if (!ctx || c.ins(ctx).size() < 2) return false;
  auto reaches_sm = [&](std::string t) {
    std::unordered_set<std::string> seen; int d = 8;
    while (!t.empty() && !seen.count(t) && d-- > 0) {
      seen.insert(t);
      const OrtNode* p = c.prod(t);
      if (!p) return false;
      if (p == sm) return true;
      if (!IsLayout(c.op(p))) return false;
      t = c.act_input(p);
    }
    return false;
  };
  const std::string& v_in = reaches_sm(c.ins(ctx)[0]) ? c.ins(ctx)[1] : c.ins(ctx)[0];
  const OrtNode* v = ProjMatmulBack(c, v_in);
  const OrtNode* o = c.outs(ctx).empty() ? nullptr : FwdToMatmul(c, c.outs(ctx)[0]);
  if (!q || !k || !v || !o) return false;
  const OrtNode* norm1 = NormBack(c, c.act_input(q));
  if (!norm1) return false;   // excludes the MAP pooling head (probe query has no pre-attention LN)
  // residual Add directly after the out_proj bias-add (no LayerScale Mul between)
  const OrtNode* res1 = FwdFirst(c, AfterBias(c, o),
      [&](const OrtNode* n) { return c.op(n) == "Add" && !c.has_init(n); });
  if (!res1) return false;
  const OrtNode* norm2 = c.outs(res1).empty() ? nullptr :
      FwdFirst(c, c.outs(res1)[0], [&](const OrtNode* n) { return c.op(n) == "LayerNormalization"; });
  if (!norm2) return false;
  const OrtNode *fc1, *fc2;
  FindMlp(c, c.outs(norm2).empty() ? std::string() : c.outs(norm2)[0], &fc1, &fc2);
  if (!fc1 || !fc2) return false;
  L->q = q; L->k = k; L->v = v; L->o = o; L->fc1 = fc1; L->fc2 = fc2;
  L->bq = BiasAddOf(c, q); L->bk = BiasAddOf(c, k); L->bv = BiasAddOf(c, v);
  L->bo = BiasAddOf(c, o); L->bf1 = BiasAddOf(c, fc1); L->bf2 = BiasAddOf(c, fc2);
  L->norm1 = norm1; L->norm2 = norm2; L->ls1 = nullptr; L->ls2 = nullptr;
  return L->bq && L->bk && L->bv && L->bo && L->bf1 && L->bf2;
}

// The residual Add after a layer's fc2 bias-add (the second residual; feeds the next layer's norm1
// or, for the last layer, the encoder post-LayerNorm).
const OrtNode* SecondResidual(const Ctx& c, const MatchedLayer& L) {
  return FwdFirst(c, AfterBias(c, L.fc2),
      [&](const OrtNode* n) { return c.op(n) == "Add" && !c.has_init(n); });
}

// Follow the primary (data) input backward through Expand + layout ops to an initializer edge; ""
// if none. Only input[0] is followed (Expand/Reshape/Transpose carry data on input 0; their shape /
// perm operands may themselves be initializers, which must NOT be mistaken for the data tensor).
std::string ResolveToInit(const Ctx& c, const std::string& t0, int depth = 8) {
  if (t0.empty() || depth < 0) return std::string();
  if (c.is_init(t0)) return t0;
  const OrtNode* p = c.prod(t0);
  if (!p || c.ins(p).empty()) return std::string();
  if (c.op(p) == "Expand" || IsLayout(c.op(p))) return ResolveToInit(c, c.ins(p)[0], depth - 1);
  return std::string();
}

// The stem: from layer-0's norm1, walk back through an optional pre-encoder LayerNorm (CLIP's
// pre_layrnorm) and layout ops to the embeddings Add (tokens + position). Extract:
//   *pos  -- the Add operand resolving to an initializer, directly or via a Gather
//            (position_embedding.weight indexed by arange). Required.
//   *cls  -- the class/cls embedding initializer, when the other (token) operand is a Concat that
//            prepends an Expand'd initializer (CLIP's cls token). Empty for SigLIP.
//   *pre_ln -- the LayerNormalization crossed on the way (CLIP); null for SigLIP.
// Returns false if the embeddings Add is not reached.
bool FindStem(const Ctx& c, const OrtNode* norm1, std::string* pos, std::string* cls,
              const OrtNode** pre_ln) {
  *pos = std::string(); *cls = std::string(); *pre_ln = nullptr;
  if (!norm1) return false;
  std::string t = c.act_input(norm1);
  std::unordered_set<std::string> seen;
  const OrtNode* add = nullptr;
  while (!t.empty() && !seen.count(t)) {
    seen.insert(t);
    const OrtNode* p = c.prod(t);
    if (!p) return false;
    if (c.op(p) == "Add") { add = p; break; }
    if (c.op(p) == "LayerNormalization") { *pre_ln = p; t = c.act_input(p); continue; }
    if (IsLayout(c.op(p))) { t = c.act_input(p); continue; }
    return false;
  }
  if (!add) return false;
  std::string tok;
  for (const std::string& in : c.ins(add)) {
    if (in.empty()) continue;
    if (c.is_init(in)) { *pos = in; continue; }         // pos added as a direct initializer
    const OrtNode* p = c.prod(in);
    if (p && c.op(p) == "Gather" && !c.ins(p).empty() && c.is_init(c.ins(p)[0])) {
      *pos = c.ins(p)[0]; continue;                     // pos = Gather(position_embedding.weight, arange)
    }
    tok = in;                                           // the non-pos (token) operand
  }
  if (pos->empty()) return false;
  const OrtNode* tp = tok.empty() ? nullptr : c.prod(tok);
  if (tp && c.op(tp) == "Concat")                       // CLIP: Concat([Expand(cls), patch tokens])
    for (const std::string& ci : c.ins(tp)) {
      std::string e = ResolveToInit(c, ci);
      if (!e.empty()) { *cls = e; break; }
    }
  return true;
}

// The MLP nonlinearity between fc1's bias-add and fc2: a Sigmoid marks quick-GELU (x*sigmoid(1.702x),
// CLIP); its absence leaves the gelu-tanh/erf SigLIP path. Bounded BFS forward crossing any op,
// stopping at the next MatMul (fc2).
bool DetectQuickGelu(const Ctx& c, const MatchedLayer& L) {
  std::vector<std::string> fr{AfterBias(c, L.fc1)}; std::unordered_set<std::string> seen; int n = 0;
  while (!fr.empty() && n++ < 64) {
    std::string x = fr.front(); fr.erase(fr.begin());
    if (x.empty() || seen.count(x)) continue;
    seen.insert(x);
    for (const OrtNode* cc : c.cons(x)) {
      if (c.op(cc) == "MatMul") continue;               // reached fc2 on this branch; stop expanding
      if (c.op(cc) == "Sigmoid") return true;
      for (const std::string& o : c.outs(cc)) fr.push_back(o);
    }
  }
  return false;
}

}  // namespace

BackboneMatch MatchBackbone(const OrtApi& api, const OrtGraph* graph) {
  BackboneMatch m;
  Ctx c(api, graph);
  if (c.err) { m.reason = "graph introspection failed"; return m; }

  m.patch_conv = FindPatchConv(c, &m.patch_size);
  if (!m.patch_conv) { m.reason = "no patch-embed Conv (kernel==stride, IC=3) found"; return m; }

  std::unordered_set<const OrtNode*> desc;
  if (!c.outs(m.patch_conv).empty()) Descendants(c, c.outs(m.patch_conv)[0], &desc);

  std::vector<const OrtNode*> softmaxes;
  for (const OrtNode* n : c.nodes())
    if (c.op(n) == "Softmax" && desc.count(n)) softmaxes.push_back(n);
  std::sort(softmaxes.begin(), softmaxes.end(),
            [&](const OrtNode* a, const OrtNode* b) { return c.order(a) < c.order(b); });

  for (const OrtNode* sm : softmaxes) {
    MatchedLayer L;
    if (ResolveLayer(c, sm, &L)) m.layers.push_back(L);   // decoder softmaxes fail here and are skipped
  }
  if (m.layers.empty()) { m.reason = "no encoder layer matched the attention+MLP topology"; return m; }

  m.exit_tensor = FindProjectorExit(c, m.patch_conv);
  if (m.exit_tensor.empty()) { m.reason = "no projector output tensor found"; return m; }

  m.ok = true;
  return m;
}

BackboneMatch MatchPlainViT(const OrtApi& api, const OrtGraph* graph) {
  BackboneMatch m;
  Ctx c(api, graph);
  if (c.err) { m.reason = "graph introspection failed"; return m; }

  m.patch_conv = FindPatchConv(c, &m.patch_size);
  if (!m.patch_conv) { m.reason = "plain-vit: no patch-embed Conv (kernel==stride, IC=3)"; return m; }

  std::unordered_set<const OrtNode*> desc;
  if (!c.outs(m.patch_conv).empty()) Descendants(c, c.outs(m.patch_conv)[0], &desc);

  std::vector<const OrtNode*> softmaxes;
  for (const OrtNode* n : c.nodes())
    if (c.op(n) == "Softmax" && desc.count(n)) softmaxes.push_back(n);
  std::sort(softmaxes.begin(), softmaxes.end(),
            [&](const OrtNode* a, const OrtNode* b) { return c.order(a) < c.order(b); });

  for (const OrtNode* sm : softmaxes) {
    MatchedLayer L;
    if (ResolvePlainLayer(c, sm, &L)) m.layers.push_back(L);   // MAP-head softmax fails (norm1 null)
  }
  if (m.layers.empty()) { m.reason = "plain-vit: no encoder layer matched the pre-norm attention+MLP topology"; return m; }

  // Exit: the last layer's second residual, then optionally the encoder post-LayerNorm over the
  // sequence. SigLIP normalizes the whole sequence -> exit at that post-LN (== last_hidden_state).
  // CLIP does not (its post_layernorm applies only to the pooled cls row, behind a Gather the layout
  // walk does not cross) -> the post-LN is absent here and the raw last residual IS last_hidden_state.
  const OrtNode* res2 = SecondResidual(c, m.layers.back());
  if (!res2 || c.outs(res2).empty()) { m.reason = "plain-vit: last encoder layer residual not found"; return m; }
  m.post_ln = FwdFirst(c, c.outs(res2)[0], [&](const OrtNode* n) { return c.op(n) == "LayerNormalization"; });
  if (m.post_ln && !c.outs(m.post_ln).empty()) {
    m.exit_tensor = c.outs(m.post_ln)[0];     // SigLIP: normalized sequence
  } else {
    m.post_ln = nullptr;
    m.exit_tensor = c.outs(res2)[0];          // CLIP: un-normed last residual
  }

  const OrtNode* pre_ln = nullptr;
  std::string pos, cls;
  if (!FindStem(c, m.layers[0].norm1, &pos, &cls, &pre_ln)) {
    m.reason = "plain-vit: stem embeddings Add (tokens + position) not found"; return m;
  }
  if (pos.empty()) { m.reason = "plain-vit: position-embedding initializer not found"; return m; }
  m.pos_embed = pos;
  m.cls_embed = cls;           // empty for SigLIP; the class embedding for CLIP
  m.pre_ln = pre_ln;           // null for SigLIP; pre_layrnorm for CLIP
  m.quick_gelu = DetectQuickGelu(c, m.layers[0]);

  m.ok = true;
  return m;
}

// ---- SAM ViT-Det walks: fused qkv + decomposed rel-pos + windowing ---------------------
namespace {

// window_partition inserts a Pad before qkv; window_unpartition a Slice before the residual. Both
// carry data on input[0], so the SAM walks cross them like layout ops.
bool IsWinLayout(const std::string& op) { return IsLayout(op) || op == "Pad" || op == "Slice"; }

// Back through layout + Pad/Slice to a LayerNormalization (norm1 sits behind window_partition's Pad).
const OrtNode* NormBackWin(const Ctx& c, const std::string& t0) {
  std::string t = t0; std::unordered_set<std::string> seen;
  while (!t.empty() && !seen.count(t)) {
    seen.insert(t);
    const OrtNode* p = c.prod(t);
    if (!p) return nullptr;
    if (c.op(p) == "LayerNormalization") return p;
    if (!IsWinLayout(c.op(p))) return nullptr;
    t = c.act_input(p);
  }
  return nullptr;
}

// BFS forward crossing layout + Pad/Slice, returning the first consumer satisfying pred.
const OrtNode* FwdFirstWin(const Ctx& c, const std::string& t0,
                           const std::function<bool(const OrtNode*)>& pred) {
  std::vector<std::string> fr{t0}; std::unordered_set<std::string> seen; int visited = 0;
  while (!fr.empty() && visited++ < 512) {
    std::string x = fr.front(); fr.erase(fr.begin());
    if (seen.count(x)) continue;
    seen.insert(x);
    for (const OrtNode* cc : c.cons(x)) {
      if (pred(cc)) return cc;
      if (IsWinLayout(c.op(cc))) for (const std::string& o : c.outs(cc)) fr.push_back(o);
    }
  }
  return nullptr;
}

// The decomposed rel-pos Einsum pair on an attention Softmax: the scores Add = Add(QK^T, Reshape(
// Add(Unsqueeze(Einsum_h), Unsqueeze(Einsum_w)))). Walk the non-MatMul operand branch, collecting
// Einsum nodes crossing layout+Add. Returns true with *rh,*rw set when two distinct Einsums are found.
bool RelPosEinsums(const Ctx& c, const OrtNode* sm, const OrtNode** rh, const OrtNode** rw) {
  *rh = nullptr; *rw = nullptr;
  if (c.ins(sm).empty()) return false;
  // cross layout back to the scores Add feeding the softmax
  std::string t = c.ins(sm)[0]; std::unordered_set<std::string> seen;
  const OrtNode* add = c.prod(t);
  while (add && IsLayout(c.op(add)) && !seen.count(t)) {
    seen.insert(t); t = c.act_input(add); add = c.prod(t);
  }
  if (!add || c.op(add) != "Add") return false;
  std::vector<const OrtNode*> found;
  for (const std::string& x : c.ins(add)) {
    std::vector<std::string> fr{x}; std::unordered_set<std::string> s2; int d = 0;
    while (!fr.empty() && d++ < 24) {
      std::string y = fr.front(); fr.erase(fr.begin());
      if (y.empty() || s2.count(y)) continue;
      s2.insert(y);
      const OrtNode* p = c.prod(y);
      if (!p) continue;
      if (c.op(p) == "Einsum") {
        if (std::find(found.begin(), found.end(), p) == found.end()) found.push_back(p);
        continue;
      }
      if (IsLayout(c.op(p)) || c.op(p) == "Add")
        for (const std::string& in : c.ins(p)) if (!in.empty()) fr.push_back(in);
    }
  }
  if (found.size() < 2) return false;
  *rh = found[0]; *rw = found[1];
  return true;
}

// Windowed iff a Pad node lies between norm1's output and the qkv MatMul (window_partition's F.pad).
bool IsWindowedLayer(const Ctx& c, const OrtNode* norm1, const OrtNode* qkv) {
  if (c.outs(norm1).empty()) return false;
  std::vector<std::string> fr{c.outs(norm1)[0]}; std::unordered_set<std::string> seen; int d = 0;
  while (!fr.empty() && d++ < 512) {
    std::string x = fr.front(); fr.erase(fr.begin());
    if (x.empty() || seen.count(x)) continue;
    seen.insert(x);
    for (const OrtNode* cc : c.cons(x)) {
      if (cc == qkv) continue;
      if (c.op(cc) == "Pad") return true;
      if (c.op(cc) != "MatMul") for (const std::string& o : c.outs(cc)) fr.push_back(o);
    }
  }
  return false;
}

// One SAM ViT-Det encoder layer from its attention Softmax. Fused qkv (q==k==v), mandatory rel-pos
// Einsum pair, no LayerScale, windowing crossed by the Win walks.
bool ResolveSamLayer(const Ctx& c, const OrtNode* sm, MatchedLayer* L) {
  if (c.ins(sm).empty()) return false;
  const OrtNode* scores = ProjMatmulBack(c, c.ins(sm)[0]);   // crosses rel-pos Add + scale Mul -> QK^T
  if (!scores || c.ins(scores).size() < 2) return false;
  const OrtNode* q = ProjMatmulBack(c, c.ins(scores)[0]);    // both back-walk (across Split) to qkv
  const OrtNode* k = ProjMatmulBack(c, c.ins(scores)[1]);
  const OrtNode* ctx = c.outs(sm).empty() ? nullptr : FwdToMatmul(c, c.outs(sm)[0]);
  if (!ctx || c.ins(ctx).size() < 2) return false;
  auto reaches_sm = [&](std::string t) {
    std::unordered_set<std::string> seen; int d = 8;
    while (!t.empty() && !seen.count(t) && d-- > 0) {
      seen.insert(t);
      const OrtNode* p = c.prod(t);
      if (!p) return false;
      if (p == sm) return true;
      if (!IsLayout(c.op(p))) return false;
      t = c.act_input(p);
    }
    return false;
  };
  const std::string& v_in = reaches_sm(c.ins(ctx)[0]) ? c.ins(ctx)[1] : c.ins(ctx)[0];
  const OrtNode* v = ProjMatmulBack(c, v_in);
  const OrtNode* o = c.outs(ctx).empty() ? nullptr : FwdToMatmul(c, c.outs(ctx)[0]);   // proj
  if (!q || !k || !v || !o) return false;
  if (!(q == k && k == v)) return false;                    // require the fused qkv (one MatMul)
  const OrtNode* qkv = q;
  const OrtNode* norm1 = NormBackWin(c, c.act_input(qkv));
  if (!norm1) return false;
  const OrtNode *rh, *rw;
  if (!RelPosEinsums(c, sm, &rh, &rw)) return false;         // the SAM signature: rel-pos is mandatory
  const OrtNode* res1 = FwdFirstWin(c, AfterBias(c, o),
      [&](const OrtNode* n) { return c.op(n) == "Add" && !c.has_init(n); });
  if (!res1) return false;
  const OrtNode* norm2 = c.outs(res1).empty() ? nullptr :
      FwdFirst(c, c.outs(res1)[0], [&](const OrtNode* n) { return c.op(n) == "LayerNormalization"; });
  if (!norm2) return false;
  const OrtNode *fc1, *fc2;
  FindMlp(c, c.outs(norm2).empty() ? std::string() : c.outs(norm2)[0], &fc1, &fc2);
  if (!fc1 || !fc2) return false;
  L->q = qkv; L->k = qkv; L->v = qkv; L->o = o; L->fc1 = fc1; L->fc2 = fc2;
  L->bq = BiasAddOf(c, qkv); L->bo = BiasAddOf(c, o);
  L->bf1 = BiasAddOf(c, fc1); L->bf2 = BiasAddOf(c, fc2);
  L->norm1 = norm1; L->norm2 = norm2; L->ls1 = nullptr; L->ls2 = nullptr;
  L->rel_h = rh; L->rel_w = rw; L->fused_qkv = true;
  L->windowed = IsWindowedLayer(c, norm1, qkv);
  return L->bq && L->bo && L->bf1 && L->bf2;   // one fused-qkv bias (bk/bv share it, left null)
}

}  // namespace

BackboneMatch MatchSamVitDet(const OrtApi& api, const OrtGraph* graph) {
  BackboneMatch m;
  Ctx c(api, graph);
  if (c.err) { m.reason = "graph introspection failed"; return m; }

  m.patch_conv = FindPatchConv(c, &m.patch_size);
  if (!m.patch_conv) { m.reason = "sam: no patch-embed Conv (kernel==stride, IC=3)"; return m; }

  std::unordered_set<const OrtNode*> desc;
  if (!c.outs(m.patch_conv).empty()) Descendants(c, c.outs(m.patch_conv)[0], &desc);

  std::vector<const OrtNode*> softmaxes;
  for (const OrtNode* n : c.nodes())
    if (c.op(n) == "Softmax" && desc.count(n)) softmaxes.push_back(n);
  std::sort(softmaxes.begin(), softmaxes.end(),
            [&](const OrtNode* a, const OrtNode* b) { return c.order(a) < c.order(b); });

  for (const OrtNode* sm : softmaxes) {
    MatchedLayer L;
    if (ResolveSamLayer(c, sm, &L)) m.layers.push_back(L);
  }
  // Require every attention layer to resolve as SAM (fused qkv + rel-pos) -- a partial match means
  // this is not a SAM graph, so leave it to the other families.
  if (m.layers.empty() || m.layers.size() != softmaxes.size()) {
    m.reason = "sam: not all attention layers present the fused-qkv + decomposed-rel-pos topology";
    return m;
  }

  // Exit: the conv neck output (last non-patch Conv's finishing chain) == last_hidden_state.
  m.exit_tensor = FindProjectorExit(c, m.patch_conv);
  if (m.exit_tensor.empty()) { m.reason = "sam: no neck output tensor found"; return m; }

  // Stem: the 2D pos_embed initializer added after the patch conv (no cls, no pre-LN).
  const OrtNode* pre_ln = nullptr;
  std::string pos, cls;
  if (!FindStem(c, m.layers[0].norm1, &pos, &cls, &pre_ln) || pos.empty()) {
    m.reason = "sam: stem pos_embed not found"; return m;
  }
  m.pos_embed = pos;

  m.ok = true;
  return m;
}

// ---- Depth Anything v2 walks: fused qkv (key,value,query) + LayerScale + a DPT multi-tap exit ----
namespace {

// Which Split output block does `t` descend from? Back-walks the projection chain (layout, bias
// Add, scale Mul) looking for one of `split`'s outputs. Returns the block index, or -1. This is what
// makes the qkv role assignment DERIVED rather than an assumed (q,k,v) ordering.
int SplitBlockOf(const Ctx& c, const std::string& t0, const OrtNode* split) {
  std::unordered_map<std::string, int> blk;
  for (size_t i = 0; i < c.outs(split).size(); i++)
    if (!c.outs(split)[i].empty()) blk[c.outs(split)[i]] = static_cast<int>(i);
  std::vector<std::string> fr{t0};
  std::unordered_set<std::string> seen;
  int visited = 0;
  while (!fr.empty() && visited++ < 64) {
    std::string x = fr.front(); fr.erase(fr.begin());
    if (x.empty() || seen.count(x)) continue;
    seen.insert(x);
    auto it = blk.find(x);
    if (it != blk.end()) return it->second;
    const OrtNode* p = c.prod(x);
    if (!p) continue;
    const std::string& o = c.op(p);
    if (!(IsLayout(o) || o == "Add" || o == "Mul" || o == "Split")) continue;
    for (const std::string& y : c.ins(p))
      if (!y.empty() && !c.resolves_to_init(y)) fr.push_back(y);
  }
  return -1;
}

// The bias Add consuming a given Split output block (each of DA v2's three branches keeps its own
// [D] bias Add -- onnxsim concatenated the weights but not the biases).
const OrtNode* SplitBlockBias(const Ctx& c, const OrtNode* split, int blk) {
  if (blk < 0 || blk >= static_cast<int>(c.outs(split).size())) return nullptr;
  return FwdFirst(c, c.outs(split)[blk],
                  [&](const OrtNode* n) { return c.op(n) == "Add" && c.has_init(n); });
}

// One Depth Anything v2 encoder layer from its attention Softmax. Fused qkv with derived per-role
// column blocks, MANDATORY LayerScale on both residual branches (this is what separates the family
// from SAM/plain-ViT), erf-GELU MLP. `res2_out` receives the layer's second residual Add, which the
// tap walk below anchors on.
bool ResolveDepthLayer(const Ctx& c, const OrtNode* sm, MatchedLayer* L, const OrtNode** res2_out) {
  if (c.ins(sm).empty()) return false;
  const OrtNode* scores = ProjMatmulBack(c, c.ins(sm)[0]);   // crosses the 1/sqrt(dh) scale Mul
  if (!scores || c.ins(scores).size() < 2) return false;
  const OrtNode* q = ProjMatmulBack(c, c.ins(scores)[0]);
  const OrtNode* k = ProjMatmulBack(c, c.ins(scores)[1]);
  const OrtNode* ctx = c.outs(sm).empty() ? nullptr : FwdToMatmul(c, c.outs(sm)[0]);
  if (!ctx || c.ins(ctx).size() < 2) return false;
  auto reaches_sm = [&](std::string t) {
    std::unordered_set<std::string> seen; int d = 8;
    while (!t.empty() && !seen.count(t) && d-- > 0) {
      seen.insert(t);
      const OrtNode* p = c.prod(t);
      if (!p) return false;
      if (p == sm) return true;
      if (!IsLayout(c.op(p))) return false;
      t = c.act_input(p);
    }
    return false;
  };
  const std::string& v_in = reaches_sm(c.ins(ctx)[0]) ? c.ins(ctx)[1] : c.ins(ctx)[0];
  const OrtNode* v = ProjMatmulBack(c, v_in);
  const OrtNode* o = c.outs(ctx).empty() ? nullptr : FwdToMatmul(c, c.outs(ctx)[0]);
  if (!q || !k || !v || !o) return false;
  if (!(q == k && k == v)) return false;                     // require the fused qkv (one MatMul)
  const OrtNode* qkv = q;
  const OrtNode* split = c.outs(qkv).empty() ? nullptr :
      FwdFirst(c, c.outs(qkv)[0], [&](const OrtNode* n) { return c.op(n) == "Split"; });
  if (!split || c.outs(split).size() != 3) return false;
  const int bq = SplitBlockOf(c, c.ins(scores)[0], split);
  const int bk = SplitBlockOf(c, c.ins(scores)[1], split);
  const int bv = SplitBlockOf(c, v_in, split);
  if (bq < 0 || bk < 0 || bv < 0 || bq == bk || bk == bv || bq == bv) return false;
  const OrtNode* norm1 = NormBack(c, c.act_input(qkv));
  if (!norm1) return false;
  const OrtNode *ls1, *res1;
  LayerScaleAndResidual(c, AfterBias(c, o), &ls1, &res1);
  if (!ls1 || !res1) return false;                           // LayerScale is mandatory here
  const OrtNode* norm2 = c.outs(res1).empty() ? nullptr :
      FwdFirst(c, c.outs(res1)[0], [&](const OrtNode* n) { return c.op(n) == "LayerNormalization"; });
  if (!norm2) return false;
  const OrtNode *fc1, *fc2;
  FindMlp(c, c.outs(norm2).empty() ? std::string() : c.outs(norm2)[0], &fc1, &fc2);
  if (!fc1 || !fc2) return false;
  const OrtNode *ls2, *res2;
  LayerScaleAndResidual(c, AfterBias(c, fc2), &ls2, &res2);
  if (!ls2 || !res2) return false;
  L->q = qkv; L->k = qkv; L->v = qkv; L->o = o; L->fc1 = fc1; L->fc2 = fc2;
  L->bq = SplitBlockBias(c, split, bq);
  L->bk = SplitBlockBias(c, split, bk);
  L->bv = SplitBlockBias(c, split, bv);
  L->bo = BiasAddOf(c, o); L->bf1 = BiasAddOf(c, fc1); L->bf2 = BiasAddOf(c, fc2);
  L->norm1 = norm1; L->norm2 = norm2; L->ls1 = ls1; L->ls2 = ls2;
  L->fused_qkv = true;
  L->qkv_blk_q = bq; L->qkv_blk_k = bk; L->qkv_blk_v = bv;
  *res2_out = res2;
  return L->bq && L->bk && L->bv && L->bo && L->bf1 && L->bf2;
}

// Does a tap's normalized output land in the DPT reassemble? Forward through layout and the
// cls-stripping Slice to a Conv.
//
// This has to answer correctly in BOTH of the matcher's call contexts, which see different graphs.
// In GetCapability the whole model is visible and the answer is that Conv. In Compile the graph is
// the FUSED SUBGRAPH -- the claim stops at the taps, so the head is not in it and the only Conv
// present is the stem's. There the taps are the subgraph's own outputs and have no consumer at all,
// which is what an unconsumed tensor means here. Anything else (an encoder norm1, whose output feeds
// the next layer's qkv MatMul in either context) is not a tap.
bool TapFeedsReassemble(const Ctx& c, const std::string& t0) {
  if (c.cons(t0).empty()) return true;         // fused-subgraph context: the tap IS an exit
  std::vector<std::string> fr{t0};
  std::unordered_set<std::string> seen;
  int visited = 0;
  while (!fr.empty() && visited++ < 64) {
    std::string x = fr.front(); fr.erase(fr.begin());
    if (x.empty() || seen.count(x)) continue;
    seen.insert(x);
    for (const OrtNode* cc : c.cons(x)) {
      if (c.op(cc) == "Conv") return true;
      if (IsLayout(c.op(cc)) || c.op(cc) == "Slice")
        for (const std::string& o : c.outs(cc)) fr.push_back(o);
    }
  }
  return false;
}

// The DPT taps. A tap is a LayerNormalization that reads a resolved encoder layer's second residual
// (through layout only) and whose output feeds the reassemble. Anchoring on the ENCODER side rather
// than walking back from the head's Convs is what makes this work in the fused-subgraph context too,
// where the head is not present. Purely structural; the layer index each tap reads is DERIVED, never
// assumed. Fills tap_layers / tap_norms / tap_tensors in graph order.
bool FindDptTaps(const Ctx& c, const std::vector<const OrtNode*>& res2, BackboneMatch* m) {
  std::unordered_map<const OrtNode*, int> layer_of;
  for (size_t i = 0; i < res2.size(); i++) layer_of[res2[i]] = static_cast<int>(i);
  struct Tap { int order, layer; const OrtNode* ln; };
  std::vector<Tap> taps;
  for (const OrtNode* n : c.nodes()) {
    if (c.op(n) != "LayerNormalization" || c.outs(n).empty()) continue;
    // back to the producing node through layout ops only: a tap norm reads the residual directly
    std::string t = c.act_input(n);
    std::unordered_set<std::string> seen;
    const OrtNode* p = nullptr;
    while (!t.empty() && !seen.count(t)) {
      seen.insert(t);
      p = c.prod(t);
      if (!p || !IsLayout(c.op(p))) break;
      t = c.act_input(p);
    }
    if (!p) continue;
    auto it = layer_of.find(p);
    if (it == layer_of.end()) continue;                    // not reading a layer output
    if (!TapFeedsReassemble(c, c.outs(n)[0])) continue;     // the next layer's norm1 lands here
    taps.push_back({c.order(n), it->second, n});
  }
  if (taps.size() < 2) return false;   // one tap is a plain encoder exit, not a DPT head
  std::sort(taps.begin(), taps.end(), [](const Tap& a, const Tap& b) { return a.order < b.order; });
  for (const Tap& t : taps) {
    if (!m->tap_layers.empty() && t.layer <= m->tap_layers.back()) return false;   // must ascend
    if (c.outs(t.ln).empty()) return false;
    m->tap_layers.push_back(t.layer);
    m->tap_norms.push_back(t.ln);
    m->tap_tensors.push_back(c.outs(t.ln)[0]);
  }
  return true;
}

}  // namespace

BackboneMatch MatchDepthAnythingDpt(const OrtApi& api, const OrtGraph* graph) {
  BackboneMatch m;
  Ctx c(api, graph);
  if (c.err) { m.reason = "graph introspection failed"; return m; }

  m.patch_conv = FindPatchConv(c, &m.patch_size);
  if (!m.patch_conv) { m.reason = "no patch-embed Conv (kernel==stride, IC=3)"; return m; }

  std::unordered_set<const OrtNode*> desc;
  if (!c.outs(m.patch_conv).empty()) Descendants(c, c.outs(m.patch_conv)[0], &desc);

  std::vector<const OrtNode*> softmaxes;
  for (const OrtNode* n : c.nodes())
    if (c.op(n) == "Softmax" && desc.count(n)) softmaxes.push_back(n);
  std::sort(softmaxes.begin(), softmaxes.end(),
            [&](const OrtNode* a, const OrtNode* b) { return c.order(a) < c.order(b); });

  std::vector<const OrtNode*> res2;
  for (const OrtNode* sm : softmaxes) {
    MatchedLayer L;
    const OrtNode* r2 = nullptr;
    if (!ResolveDepthLayer(c, sm, &L, &r2)) continue;
    m.layers.push_back(L);
    res2.push_back(r2);
  }
  // Every attention layer must present the fused-qkv + LayerScale topology: a partial match means
  // this is a different family (RF-DETR's separate q/k/v, or SAM/plain-ViT without LayerScale).
  if (m.layers.empty() || m.layers.size() != softmaxes.size()) {
    m.reason = "not all attention layers present the fused-qkv + LayerScale topology";
    return m;
  }

  // The discriminator: a DPT head reading several normalized intermediate encoder outputs.
  if (!FindDptTaps(c, res2, &m)) {
    m.reason = "no multi-tap DPT head (>=2 normalized intermediate encoder taps)";
    return m;
  }
  m.exit_tensor = m.tap_tensors.back();   // the deepest tap; the claim seeds on all of them

  // Stem: the folded [1,ntok,d] interpolated position embedding plus the prepended cls token.
  const OrtNode* pre_ln = nullptr;
  std::string pos, cls;
  if (!FindStem(c, m.layers[0].norm1, &pos, &cls, &pre_ln) || pos.empty()) {
    m.reason = "stem position embedding not found"; return m;
  }
  if (cls.empty()) { m.reason = "stem cls token not found"; return m; }
  m.pos_embed = pos;
  m.cls_embed = cls;
  m.pre_ln = pre_ln;   // DINOv2 has none; carried so a variant that grows one is not silently dropped

  m.ok = true;
  return m;
}

// ---------------------------------------------------------------------------------------------
// ModernBERT (Family::ModernBert)
// ---------------------------------------------------------------------------------------------
namespace {

// The initializer shape behind a weight edge, following a QDQ DequantizeLinear.
const std::vector<int64_t>* WeightShape(const Ctx& c, const std::string& t) {
  if (c.is_init(t)) return c.init_shape(t);
  const OrtNode* p = c.prod(t);
  if (p && c.op(p) == "DequantizeLinear" && !c.ins(p).empty() && c.is_init(c.ins(p)[0]))
    return c.init_shape(c.ins(p)[0]);
  return nullptr;
}

bool IsGraphInput(const Ctx& c, const std::string& t) {
  return !t.empty() && !c.prod(t) && !c.is_init(t);
}

// A MatMul reading activation `t` whose weight is a 2D initializer [a, b] (b < 0: any b).
const OrtNode* MatMulInit(const Ctx& c, const std::string& t, int64_t a, int64_t b, int64_t* got_b) {
  for (const OrtNode* n : c.cons(t)) {
    if (c.op(n) != "MatMul" || c.ins(n).size() != 2 || c.ins(n)[0] != t) continue;
    const std::vector<int64_t>* s = WeightShape(c, c.ins(n)[1]);
    if (!s || s->size() != 2 || (*s)[0] != a || (b >= 0 && (*s)[1] != b)) continue;
    if (got_b) *got_b = (*s)[1];
    return n;
  }
  return nullptr;
}

// A LayerNormalization reading `t`, with (want_bias) or without a bias input.
const OrtNode* LnOf(const Ctx& c, const std::string& t, bool want_bias) {
  for (const OrtNode* n : c.cons(t)) {
    if (c.op(n) != "LayerNormalization" || c.ins(n).empty() || c.ins(n)[0] != t) continue;
    const bool has_b = c.ins(n).size() >= 3 && !c.ins(n)[2].empty();
    if (has_b == want_bias) return n;
  }
  return nullptr;
}

// The residual Add reading `x` whose other operand is a MatMul with a [k, d] weight, optionally
// behind a bias Add. Sets *mm and *bias_add (null when there is none).
const OrtNode* ResidualAdd(const Ctx& c, const std::string& x, int64_t k, int64_t d,
                           const OrtNode** mm, const OrtNode** bias_add) {
  for (const OrtNode* n : c.cons(x)) {
    if (c.op(n) != "Add" || c.ins(n).size() != 2) continue;
    const std::string& other = c.ins(n)[0] == x ? c.ins(n)[1] : c.ins(n)[0];
    const OrtNode* p = c.prod(other);
    const OrtNode* ba = nullptr;
    if (p && c.op(p) == "Add" && c.has_init(p)) { ba = p; p = c.prod(c.act_input(p)); }
    if (!p || c.op(p) != "MatMul" || c.ins(p).size() != 2) continue;
    const std::vector<int64_t>* s = WeightShape(c, c.ins(p)[1]);
    if (!s || s->size() != 2 || (*s)[1] != d || (k >= 0 && (*s)[0] != k)) continue;
    *mm = p; *bias_add = ba;
    return n;
  }
  return nullptr;
}

// Bounded backward walk from `from`, never expanding a stop tensor, the stop node, or a Shape
// (a shape computation may read the residual, and following it would reach earlier layers).
void WalkBack(const Ctx& c, const std::string& from, const std::unordered_set<std::string>& stop_t,
              const OrtNode* stop_n, const std::function<void(const OrtNode*)>& visit) {
  std::vector<std::string> st{from};
  std::unordered_set<const OrtNode*> seen;
  int budget = 600;
  while (!st.empty() && budget-- > 0) {
    std::string t = st.back(); st.pop_back();
    if (stop_t.count(t)) continue;
    const OrtNode* p = c.prod(t);
    if (!p || p == stop_n || seen.count(p)) continue;
    seen.insert(p);
    visit(p);
    if (c.op(p) == "Shape") continue;
    for (const std::string& in : c.ins(p))
      if (!in.empty() && !c.is_init(in)) st.push_back(in);
  }
}

// A scalar initializer operand of a node, "" if none.
std::string ScalarInit(const Ctx& c, const OrtNode* n) {
  for (const std::string& t : c.ins(n)) {
    if (!c.is_init(t)) continue;
    const std::vector<int64_t>* s = c.init_shape(t);
    size_t e = 1;
    if (s) for (int64_t v : *s) e *= (size_t)v;
    if (s && e == 1) return t;
  }
  return std::string();
}

// The attention details of one layer: the RoPE table (via its Cos), the scale, the sliding band.
bool ResolveAttention(const Ctx& c, const std::string& ctx_in, const std::unordered_set<std::string>& stop,
                      const OrtNode* stop_n, MbLayer* L, std::string* why) {
  std::vector<const OrtNode*> coss;
  WalkBack(c, ctx_in, stop, stop_n, [&](const OrtNode* p) {
    const std::string& op = c.op(p);
    if (op == "Cos" && std::find(coss.begin(), coss.end(), p) == coss.end()) coss.push_back(p);
    else if (op == "LessOrEqual") { std::string w = ScalarInit(c, p); if (!w.empty()) L->window = w; }
    else if (op == "Mul" && L->scale.empty()) L->scale = ScalarInit(c, p);
  });
  if (coss.size() != 1) { *why = "a layer reads " + std::to_string(coss.size()) + " RoPE Cos tables, expected 1"; return false; }
  // Cos <- (Concat/Transpose) <- MatMul(inverse frequencies [.., half, ..], positions)
  WalkBack(c, c.ins(coss[0])[0], {}, nullptr, [&](const OrtNode* p) {
    if (!L->inv_freq.empty() || c.op(p) != "MatMul") return;
    for (const std::string& t : c.ins(p)) {
      const std::vector<int64_t>* s = c.is_init(t) ? c.init_shape(t) : nullptr;
      if (!s) continue;
      int big = 0;
      for (int64_t v : *s) big += v > 1;
      if (big == 1) { L->inv_freq = t; return; }
    }
  });
  if (L->inv_freq.empty()) { *why = "no RoPE inverse-frequency initializer behind the layer's Cos"; return false; }
  if (L->scale.empty()) { *why = "no attention-scale constant in the layer"; return false; }
  return true;
}

// One encoder layer starting at residual `x`. Returns false with *why empty when `x` does not open
// a layer (the encoder ended), or with *why set on a malformed layer.
bool WalkMbLayer(const Ctx& c, const std::string& x, int64_t d, MbLayer* L, std::string* xnext,
                 std::string* why) {
  std::string h = x;
  const OrtNode* wqkv = MatMulInit(c, x, d, 3 * d, nullptr);
  if (!wqkv) {
    const OrtNode* ln = LnOf(c, x, false);
    if (!ln) return false;
    wqkv = MatMulInit(c, c.outs(ln)[0], d, 3 * d, nullptr);
    if (!wqkv) return false;
    L->attn_norm = ln; h = c.outs(ln)[0];
  }
  L->wqkv = wqkv;
  const OrtNode* ba = nullptr;
  const OrtNode* a1 = ResidualAdd(c, x, d, d, &L->wo, &ba);
  if (!a1 || ba) { *why = "no [d,d] attention output projection into the residual"; return false; }
  const std::string xm = c.outs(a1)[0];
  L->mlp_norm = LnOf(c, xm, false);
  if (!L->mlp_norm) { *why = "no bias-less MLP LayerNorm"; return false; }
  int64_t n2 = 0;
  L->wi = MatMulInit(c, c.outs(L->mlp_norm)[0], d, -1, &n2);
  if (!L->wi || n2 % 2) { *why = "no [d, 2 d_ff] MLP up-projection"; return false; }
  bool split = false;
  for (const OrtNode* s : c.cons(c.outs(L->wi)[0])) split |= c.op(s) == "Split" && c.outs(s).size() == 2;
  if (!split) { *why = "the MLP up-projection is not split into a GeGLU pair"; return false; }
  const OrtNode* a2 = ResidualAdd(c, xm, n2 / 2, d, &L->wo_mlp, &ba);
  if (!a2 || ba) { *why = "no [d_ff, d] MLP down-projection into the residual"; return false; }
  *xnext = c.outs(a2)[0];
  return ResolveAttention(c, c.act_input(L->wo), {x, h}, wqkv, L, why);
}

// A torch pre-norm head projection's weight: the MatMul's weight edge, or Transpose(Split(init)).
bool ResolveProj(const Ctx& c, const OrtNode* mm, ProjWeight* w) {
  const std::string& t = c.ins(mm)[1];
  if (WeightShape(c, t)) { w->tensor = t; w->rows_of = -1; }
  else {
    const OrtNode* tr = c.prod(t);
    const OrtNode* sp = tr && c.op(tr) == "Transpose" ? c.prod(c.ins(tr)[0]) : nullptr;
    if (!sp || c.op(sp) != "Split" || !WeightShape(c, c.ins(sp)[0])) return false;
    const auto& so = c.outs(sp);
    auto it = std::find(so.begin(), so.end(), c.ins(tr)[0]);
    w->tensor = c.ins(sp)[0]; w->rows_of = (int)(it - so.begin()); w->nsplit = (int)so.size();
  }
  for (const OrtNode* a : c.cons(c.outs(mm)[0])) {
    if (c.op(a) != "Add") continue;
    for (const std::string& b : c.ins(a)) {
      if (b == c.outs(mm)[0]) continue;
      if (c.is_init(b)) { w->bias = b; return true; }
      const OrtNode* sp = c.prod(b);
      if (sp && c.op(sp) == "Split" && c.is_init(c.ins(sp)[0])) {
        const auto& so = c.outs(sp);
        w->bias = c.ins(sp)[0]; w->bias_rows_of = (int)(std::find(so.begin(), so.end(), b) - so.begin());
        return true;
      }
    }
  }
  return true;   // no bias
}

// One torch nn.TransformerEncoderLayer(norm_first) starting at residual `x`.
bool WalkMbPost(const Ctx& c, const std::string& x, int64_t d, MbPost* P, std::string* xnext) {
  P->ln1 = LnOf(c, x, true);
  if (!P->ln1) return false;
  const std::string h = c.outs(P->ln1)[0];
  // q/k/v: MatMuls fed by h (possibly through a layout Transpose), weights split from one init
  std::vector<const OrtNode*> qkv;
  std::vector<std::string> feeds{h};
  for (const OrtNode* n : c.cons(h)) if (IsLayout(c.op(n))) feeds.push_back(c.outs(n)[0]);
  for (const std::string& f : feeds)
    for (const OrtNode* n : c.cons(f))
      if (c.op(n) == "MatMul" && c.ins(n).size() == 2 && c.ins(n)[0] == f) qkv.push_back(n);
  if (qkv.size() != 3) return false;
  ProjWeight pw[3];
  for (int i = 0; i < 3; i++) if (!ResolveProj(c, qkv[i], &pw[i]) || pw[i].rows_of < 0) return false;
  for (int i = 0; i < 3; i++) {
    if (pw[i].rows_of == 0) P->q = pw[i];
    else if (pw[i].rows_of == 1) P->k = pw[i];
    else if (pw[i].rows_of == 2) P->v = pw[i];
  }
  if (P->q.tensor.empty() || P->k.tensor.empty() || P->v.tensor.empty()) return false;
  const OrtNode *omm = nullptr, *oba = nullptr;
  const OrtNode* a1 = ResidualAdd(c, x, d, d, &omm, &oba);
  if (!a1 || !ResolveProj(c, omm, &P->o)) return false;
  const std::string xm = c.outs(a1)[0];
  P->ln2 = LnOf(c, xm, true);
  if (!P->ln2) return false;
  int64_t dff = 0;
  const OrtNode* w1 = MatMulInit(c, c.outs(P->ln2)[0], d, -1, &dff);
  if (!w1 || !ResolveProj(c, w1, &P->w1)) return false;
  const OrtNode *w2 = nullptr, *b2 = nullptr;
  const OrtNode* a2 = ResidualAdd(c, xm, dff, d, &w2, &b2);
  if (!a2 || !ResolveProj(c, w2, &P->w2)) return false;
  // the activation between w1's bias Add and w2: ReLU, or erf GELU
  bool relu = false, erf = false;
  WalkBack(c, c.act_input(w2), {c.outs(P->ln2)[0]}, w1, [&](const OrtNode* p) {
    relu |= c.op(p) == "Relu"; erf |= c.op(p) == "Erf";
  });
  if (relu == erf) return false;
  P->gelu = erf;
  WalkBack(c, c.act_input(omm), {h, x}, nullptr, [&](const OrtNode* p) {
    if (c.op(p) == "Mul" && P->scale.empty()) P->scale = ScalarInit(c, p);
  });
  *xnext = c.outs(a2)[0];
  return true;
}

}  // namespace

BackboneMatch MatchModernBert(const OrtApi& api, const OrtGraph* graph) {
  BackboneMatch m;
  Ctx c(api, graph);
  if (c.err) { m.reason = "graph read failed"; return m; }
  ModernBertMatch& mb = m.mb;
  // Stem: Gather(table [V, d], graph input) -> bias-less LayerNormalization.
  for (const OrtNode* n : c.nodes()) {
    if (c.op(n) != "Gather" || c.ins(n).size() != 2 || !IsGraphInput(c, c.ins(n)[1])) continue;
    const std::vector<int64_t>* s = WeightShape(c, c.ins(n)[0]);
    if (!s || s->size() != 2) continue;
    const OrtNode* ln = LnOf(c, c.outs(n)[0], false);
    if (!ln) continue;
    mb.tok_gather = n; mb.emb_norm = ln; mb.ids_input = c.ins(n)[1];
    break;
  }
  if (!mb.tok_gather) { m.reason = "no token-embedding Gather into a LayerNorm"; return m; }
  const int64_t d = (*WeightShape(c, c.ins(mb.tok_gather)[0]))[1];

  std::string x = c.outs(mb.emb_norm)[0];
  for (;;) {
    MbLayer L;
    std::string xn, why;
    if (!WalkMbLayer(c, x, d, &L, &xn, &why)) {
      if (!why.empty()) { m.reason = "layer " + std::to_string(mb.layers.size()) + ": " + why; return m; }
      break;
    }
    mb.layers.push_back(L);
    x = xn;
  }
  if (mb.layers.size() < 2) { m.reason = "fewer than 2 ModernBERT layers"; return m; }
  mb.final_norm = LnOf(c, x, false);
  if (!mb.final_norm) { m.reason = "no final LayerNorm after the last layer"; return m; }
  x = c.outs(mb.final_norm)[0];
  // Optional post bias: Add(final, Unsqueeze(Gather(type table, graph input))).
  for (const OrtNode* a : c.cons(x)) {
    if (c.op(a) != "Add" || c.ins(a).size() != 2) continue;
    const OrtNode* p = c.prod(c.ins(a)[0] == x ? c.ins(a)[1] : c.ins(a)[0]);
    while (p && IsLayout(c.op(p))) p = c.prod(c.ins(p)[0]);
    if (!p || c.op(p) != "Gather" || c.ins(p).size() != 2 || !IsGraphInput(c, c.ins(p)[1])) continue;
    const std::vector<int64_t>* s = WeightShape(c, c.ins(p)[0]);
    if (!s || s->size() != 2 || (*s)[1] != d) continue;
    mb.type_gather = p; mb.type_input = c.ins(p)[1];
    x = c.outs(a)[0];
    break;
  }
  for (;;) {
    MbPost P;
    std::string xn;
    if (!WalkMbPost(c, x, d, &P, &xn)) break;
    mb.post.push_back(P);
    x = xn;
  }
  m.exit_tensor = x;
  m.ok = true;
  return m;
}

BackboneMatch MatchAny(const OrtApi& api, const OrtGraph* graph) {
  struct Entry { Family family; const char* name; BackboneMatch (*fn)(const OrtApi&, const OrtGraph*); };
  // Registered family matchers, tried in order. Adding a target is: register its signature here,
  // and wire a Compile marshaling path in rocket_ep.cc keyed on the family this returns. Order is by
  // decreasing strictness: Depth Anything (fused qkv + LayerScale + a multi-tap DPT exit) first;
  // then DINOv2/RF-DETR (LayerScale + a CSP-projector Conv exit); then SAM (fused qkv + a mandatory
  // decomposed-rel-pos Einsum pair on every layer); then plain-ViT last -- the plain-ViT matcher
  // would loosely claim SAM's global-attention layers (and any encoder's layers without a
  // projector), so every stricter family must be tried before it.
  //
  // Depth Anything MUST precede DINOv2/RF-DETR, and the reason is a trap rather than a preference:
  // MatchBackbone misses a Depth Anything graph only by accident. Its ClassifyAttention uses the
  // layout-only MatmulBackLayout for the scores walk, and DA v2 puts a Mul(QK^T, 1/sqrt(dh)) in the
  // way. Relax that one walk to ProjMatmulBack -- which every other family already uses, so it reads
  // as an obvious cleanup -- and MatchBackbone claims all 12 encoder layers and hands back
  // /head/conv3/Conv_output_0 as a "projector exit", silently swallowing the entire DPT head into a
  // subgraph the DINOv2 marshaler would then compute as something else entirely. Over-claiming in
  // GetCapability corrupts the output with no fallback, so do not rely on that miss for separation.
  // ModernBERT leads: its stem (a token Gather from a graph input into a bias-less LayerNorm) is
  // one no vision family has, and every vision matcher needs a patch Conv a text graph lacks.
  static const Entry kRegistry[] = {
    {Family::ModernBert,       "modernbert",    &MatchModernBert},
    {Family::DepthAnythingDpt, "depth-dpt",     &MatchDepthAnythingDpt},
    {Family::DinoV2RfDetr,     "dinov2-rfdetr", &MatchBackbone},
    {Family::SamVitDet,        "sam-vitdet",    &MatchSamVitDet},
    {Family::SiglipVit,        "plain-vit",     &MatchPlainViT},
  };
  std::string reasons;
  for (const Entry& e : kRegistry) {
    BackboneMatch m = e.fn(api, graph);
    if (m.ok) { m.family = e.family; return m; }
    if (!reasons.empty()) reasons += "; ";
    reasons += std::string(e.name) + ": " + m.reason;   // accumulate each family's miss reason
  }
  BackboneMatch none;
  none.reason = reasons.empty() ? "no registered family matched" : reasons;
  return none;
}

}  // namespace rocket_match
