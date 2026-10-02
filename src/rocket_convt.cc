// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_convt: the per-node ConvTranspose handler. See rocket_convt.h for what is claimed.
//
// Per call and per batch image the handler converts the fp32 NCHW input plane to fp16, runs
// rocket_conv_transpose2d_fp16_prepacked (one resident GEMM, the input read as IC planes of IH*IW
// pixels, then a host scatter-add of the taps), and widens the fp16 output back to fp32. The
// layouts need no transpose: ONNX's X [N][C][H][W], W [C][OC][KH][KW] and Y [N][OC][OH][OW] are the
// entry's own for one batch image and group 1.

#include "rocket_convt.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rocket_conv.h"
#include "rocket_matmul.h"

namespace rocket_convt {
namespace {

inline void Discard(const OrtApi& api, OrtStatus* st) {
  if (st) api.ReleaseStatus(st);
}

void Log(const OrtApi& api, const OrtLogger& lg, OrtLoggingLevel lvl, const std::string& msg) {
  Discard(api, api.Logger_LogMessage(&lg, lvl, msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
}

// An attribute read: *present is false when the node does not carry it (ONNX Runtime reports an unset
// optional attribute either as a null attribute or as ORT_NOT_FOUND). Returns false when the attribute
// is present but not of the expected type.
bool ReadInts(const OrtApi& api, const OrtNode* n, const char* name, std::vector<int64_t>* v,
              bool* present) {
  *present = false;
  v->clear();
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return true; }
  if (!attr) return true;
  *present = true;
  int64_t buf[16];
  size_t out = 0;   // bytes
  st = api.ReadOpAttr(attr, ORT_OP_ATTR_INTS, buf, sizeof(buf), &out);
  if (st) { api.ReleaseStatus(st); return false; }
  if (out > sizeof(buf) || out % sizeof(int64_t)) return false;
  v->assign(buf, buf + out / sizeof(int64_t));
  return true;
}

bool ReadInt(const OrtApi& api, const OrtNode* n, const char* name, int64_t def, int64_t* v) {
  *v = def;
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return true; }
  if (!attr) return true;
  size_t out = 0;
  st = api.ReadOpAttr(attr, ORT_OP_ATTR_INT, v, sizeof(*v), &out);
  if (st) { api.ReleaseStatus(st); return false; }
  return true;
}

bool ReadString(const OrtApi& api, const OrtNode* n, const char* name, const char* def,
                std::string* v) {
  *v = def;
  const OrtOpAttr* attr = nullptr;
  OrtStatus* st = api.Node_GetAttributeByName(n, name, &attr);
  if (st) { api.ReleaseStatus(st); return true; }
  if (!attr) return true;
  char buf[64];
  size_t out = 0;   // bytes, no terminator
  st = api.ReadOpAttr(attr, ORT_OP_ATTR_STRING, buf, sizeof(buf), &out);
  if (st) { api.ReleaseStatus(st); return false; }
  if (out > sizeof(buf)) return false;
  v->assign(buf, out);
  return true;
}

// Element type and dimensions of a value (-1 for a symbolic dim). *rank is -1 when the shape is
// unknown.
OrtStatus* ValueShape(const OrtApi& api, const OrtValueInfo* vi, ONNXTensorElementDataType* et,
                      std::vector<int64_t>* dims, int* rank) {
  *et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  dims->clear();
  *rank = -1;
  const OrtTypeInfo* ti = nullptr;   // owned by the graph
  if (OrtStatus* st = api.GetValueInfoTypeInfo(vi, &ti)) return st;
  const OrtTensorTypeAndShapeInfo* tsi = nullptr;
  if (OrtStatus* st = api.CastTypeInfoToTensorInfo(ti, &tsi)) return st;
  if (!tsi) return nullptr;
  if (OrtStatus* st = api.GetTensorElementType(tsi, et)) return st;
  size_t nd = 0;
  if (OrtStatus* st = api.GetDimensionsCount(tsi, &nd)) return st;
  if (nd == 0) return nullptr;   // unknown (a ConvTranspose operand is never a scalar)
  dims->resize(nd);
  if (OrtStatus* st = api.GetDimensions(tsi, dims->data(), nd)) return st;
  for (auto& d : *dims)
    if (d <= 0) d = -1;
  *rank = static_cast<int>(nd);
  return nullptr;
}

// A constant fp32 initializer's value and dimensions. *ok is false when the value is not a constant
// initializer or not fp32.
OrtStatus* ConstF32(const OrtApi& api, const OrtValueInfo* vi, const float** data,
                    std::vector<int64_t>* dims, bool* ok) {
  *ok = false;
  *data = nullptr;
  dims->clear();
  bool is_const = false;
  if (OrtStatus* st = api.ValueInfo_IsConstantInitializer(vi, &is_const)) return st;
  if (!is_const) return nullptr;
  const OrtValue* v = nullptr;
  if (OrtStatus* st = api.ValueInfo_GetInitializerValue(vi, &v)) return st;
  if (!v) return nullptr;
  OrtTensorTypeAndShapeInfo* info = nullptr;
  if (OrtStatus* st = api.GetTensorTypeAndShape(v, &info)) return st;
  ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t nd = 0;
  OrtStatus* st = api.GetTensorElementType(info, &et);
  if (!st) st = api.GetDimensionsCount(info, &nd);
  if (!st) { dims->resize(nd); st = api.GetDimensions(info, dims->data(), nd); }
  api.ReleaseTensorTypeAndShapeInfo(info);
  if (st) return st;
  if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return nullptr;
  const void* raw = nullptr;
  if (OrtStatus* s2 = api.GetTensorData(v, &raw)) return s2;
  *data = static_cast<const float*>(raw);
  *ok = true;
  return nullptr;
}

rocket_conv_transpose2d_desc MakeDesc(const Spec& s, int ih, int iw) {
  rocket_conv_transpose2d_desc d{};
  d.ic = s.ic; d.ih = ih; d.iw = iw; d.oc = s.oc;
  d.kh = s.kh; d.kw = s.kw;
  d.stride_y = s.sy; d.stride_x = s.sx;
  d.pad_top = s.pt; d.pad_left = s.pl;
  // The entry's output extent is (I-1)s - 2*pad + d(K-1) + opad + 1 with one pad per axis; ONNX's
  // subtracts the leading and the trailing pad. A trailing pad smaller than the leading one keeps
  // more trailing rows, which is the entry's output_padding: the scatter-add places every tap at
  // i*s - pad_top + k*d and keeps what lands inside the extent, so the two agree element for element.
  d.opad_y = s.opy + s.pt - s.pb;
  d.opad_x = s.opx + s.pl - s.pr;
  d.dil_y = s.dy; d.dil_x = s.dx;
  d.depthwise = 0;
  return d;
}

std::string PlanReason(int rc) {
  if (rc == ROCKET_E_UNSUPPORTED) return "plan: unsupported (not the RK3588)";
  if (rc == ROCKET_E_SHAPE) return "plan: invalid descriptor";
  if (rc == -3) return "plan: empty output";
  if (rc == -4) return "plan: past the entry's size bound";
  return "plan: refused (" + std::to_string(rc) + ")";
}

double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The host path for a spatial size the entry's plan refuses, reached only by a node whose input size
// was symbolic at GetCapability. A direct fp32 scatter, the definition itself.
void HostConvT(const Spec& s, int ih, int iw, const float* x, const float* w, const float* b,
               float* y) {
  const int OH = s.OH(ih), OW = s.OW(iw);
  const size_t plane = static_cast<size_t>(OH) * OW;
  for (int oc = 0; oc < s.oc; oc++) {
    const float bv = b ? b[oc] : 0.0f;
    for (size_t i = 0; i < plane; i++) y[oc * plane + i] = bv;
  }
  for (int ic = 0; ic < s.ic; ic++)
    for (int oc = 0; oc < s.oc; oc++) {
      const float* wk = w + (static_cast<size_t>(ic) * s.oc + oc) * s.kh * s.kw;
      float* yo = y + oc * plane;
      for (int y0 = 0; y0 < ih; y0++)
        for (int x0 = 0; x0 < iw; x0++) {
          const float v = x[(static_cast<size_t>(ic) * ih + y0) * iw + x0];
          for (int ky = 0; ky < s.kh; ky++) {
            const int ph = y0 * s.sy - s.pt + ky * s.dy;
            if (ph < 0 || ph >= OH) continue;
            for (int kx = 0; kx < s.kw; kx++) {
              const int pw = x0 * s.sx - s.pl + kx * s.dx;
              if (pw < 0 || pw >= OW) continue;
              yo[static_cast<size_t>(ph) * OW + pw] += v * wk[ky * s.kw + kx];
            }
          }
        }
    }
}

// One compiled ConvTranspose node.
struct State {
  explicit State(const OrtApi& a) : api(a) {}
  ~State();

  const OrtApi& api;
  std::shared_ptr<Shared> sh;
  Spec s;
  std::string name;
  size_t x_idx = 0;
  // The fp32 weight and bias, kept only for a node whose spatial size is symbolic: a new size packs
  // from them, and a size the plan refuses runs HostConvT on them.
  std::vector<float> w32, b32;
  struct Pack {
    int ih, iw;
    rocket_conv_transpose2d_weights* h;   // null: this size runs on the host path
  };
  std::vector<Pack> packs;
  std::vector<_Float16> s_in, s_out;
  bool prof = false;
  long calls = 0, host_calls = 0;
  double t_in = 0, t_npu = 0, t_out = 0, t_host = 0;
};

State::~State() {
  if (prof && calls)
    std::fprintf(stderr,
                 "ort-rocket prof ConvTranspose %s: %ld calls (%ld host), per call in %.3f ms, "
                 "npu entry %.3f ms, out %.3f ms, host %.3f ms\n",
                 name.c_str(), calls, host_calls, t_in / calls, t_npu / calls, t_out / calls,
                 t_host / calls);
  for (auto& p : packs)
    if (p.h) rocket_conv_transpose2d_weights_free(sh->ctx, p.h);
}

// Packs the weight for one spatial size. Returns 0, the plan's (negative) code on a refusal, or
// kPackFailed when the plan accepted and the pack itself failed; *h is null unless 0.
constexpr int kPackFailed = 1;
int PackAt(State* st, const float* w, const float* b, int ih, int iw,
           rocket_conv_transpose2d_weights** h) {
  *h = nullptr;
  const rocket_conv_transpose2d_desc d = MakeDesc(st->s, ih, iw);
  const int rc = rocket_conv_transpose2d_prepacked_plan(&d);
  if (rc) return rc;
  const size_t nw = static_cast<size_t>(st->s.ic) * st->s.oc * st->s.kh * st->s.kw;
  std::vector<_Float16> w16(nw), b16;
  for (size_t i = 0; i < nw; i++) w16[i] = static_cast<_Float16>(w[i]);
  if (b) {
    b16.resize(st->s.oc);
    for (int i = 0; i < st->s.oc; i++) b16[i] = static_cast<_Float16>(b[i]);
  }
  *h = rocket_conv_transpose2d_weights_pack(st->sh->ctx, &d, w16.data(), b ? b16.data() : nullptr);
  return *h ? 0 : kPackFailed;
}

struct ConvTComputeInfo : ComputeInfoBase {
  explicit ConvTComputeInfo(State* st) : state(st) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }
  ~ConvTComputeInfo() override { delete state; }
  State* state;

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* /*ctx*/,
                                                 void** compute_state) noexcept {
    *compute_state = static_cast<ConvTComputeInfo*>(this_ptr)->state;
    return nullptr;
  }
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                            void* /*compute_state*/) noexcept {}

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/, void* compute_state,
                                             OrtKernelContext* kctx) noexcept {
    auto* st = static_cast<State*>(compute_state);
    const OrtApi& api = st->api;
    const Spec& s = st->s;
    std::lock_guard<std::mutex> guard(st->sh->mu);

    const OrtValue* xv = nullptr;
    if (OrtStatus* e = api.KernelContext_GetInput(kctx, st->x_idx, &xv)) return e;
    std::vector<int64_t> xd;
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    {
      OrtTensorTypeAndShapeInfo* info = nullptr;
      if (OrtStatus* e = api.GetTensorTypeAndShape(xv, &info)) return e;
      size_t nd = 0;
      OrtStatus* e = api.GetTensorElementType(info, &et);
      if (!e) e = api.GetDimensionsCount(info, &nd);
      if (!e) { xd.resize(nd); e = api.GetDimensions(info, xd.data(), nd); }
      api.ReleaseTensorTypeAndShapeInfo(info);
      if (e) return e;
    }
    if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || xd.size() != 4 || xd[1] != s.ic)
      return api.CreateStatus(ORT_INVALID_ARGUMENT,
          ("ort-rocket: ConvTranspose " + st->name + ": input must be fp32 [N," +
           std::to_string(s.ic) + ",H,W]").c_str());
    const int64_t N = xd[0];
    const int ih = static_cast<int>(xd[2]), iw = static_cast<int>(xd[3]);
    const int OH = s.OH(ih), OW = s.OW(iw);
    if (N < 0 || ih <= 0 || iw <= 0 || OH <= 0 || OW <= 0)
      return api.CreateStatus(ORT_INVALID_ARGUMENT,
          ("ort-rocket: ConvTranspose " + st->name + ": empty output for input [" +
           std::to_string(N) + "," + std::to_string(s.ic) + "," + std::to_string(ih) + "," +
           std::to_string(iw) + "]").c_str());
    const int64_t yshape[4] = {N, s.oc, OH, OW};
    OrtValue* yv = nullptr;
    if (OrtStatus* e = api.KernelContext_GetOutput(kctx, 0, yshape, 4, &yv)) return e;
    void* yraw = nullptr;
    if (OrtStatus* e = api.GetTensorMutableData(yv, &yraw)) return e;
    const void* xraw = nullptr;
    if (OrtStatus* e = api.GetTensorData(xv, &xraw)) return e;
    const float* x = static_cast<const float*>(xraw);
    float* y = static_cast<float*>(yraw);
    if (N == 0) return nullptr;

    State::Pack* pk = nullptr;
    for (auto& p : st->packs)
      if (p.ih == ih && p.iw == iw) { pk = &p; break; }
    if (!pk) {
      if (st->w32.empty())   // a static node sees only its compiled size
        return api.CreateStatus(ORT_INVALID_ARGUMENT,
            ("ort-rocket: ConvTranspose " + st->name + ": input size " + std::to_string(ih) + "x" +
             std::to_string(iw) + " differs from the size it was compiled for").c_str());
      rocket_conv_transpose2d_weights* h = nullptr;
      const int rc = PackAt(st, st->w32.data(), s.has_bias ? st->b32.data() : nullptr, ih, iw, &h);
      if (rc == kPackFailed)
        return api.CreateStatus(ORT_FAIL,
            ("ort-rocket: ConvTranspose " + st->name + ": weight pack failed at " +
             std::to_string(ih) + "x" + std::to_string(iw)).c_str());
      st->packs.push_back({ih, iw, h});   // h null: the plan refused this size, run it on the host
      pk = &st->packs.back();
    }

    const size_t in_n = static_cast<size_t>(s.ic) * ih * iw;
    const size_t out_n = static_cast<size_t>(s.oc) * OH * OW;
    if (!pk->h) {
      const double t0 = st->prof ? NowMs() : 0;
      for (int64_t n = 0; n < N; n++)
        HostConvT(s, ih, iw, x + n * in_n, st->w32.data(), s.has_bias ? st->b32.data() : nullptr,
                  y + n * out_n);
      if (st->prof) { st->t_host += NowMs() - t0; st->calls++; st->host_calls++; }
      return nullptr;
    }
    st->s_in.resize(in_n);
    st->s_out.resize(out_n);
    _Float16* in16 = st->s_in.data();
    _Float16* out16 = st->s_out.data();
    for (int64_t n = 0; n < N; n++) {
      const double t0 = st->prof ? NowMs() : 0;
      const float* xn = x + n * in_n;
      for (size_t i = 0; i < in_n; i++) in16[i] = static_cast<_Float16>(xn[i]);
      const double t1 = st->prof ? NowMs() : 0;
      const int rc = rocket_conv_transpose2d_fp16_prepacked(st->sh->ctx, pk->h, in16, out16);
      if (rc)
        return api.CreateStatus(ORT_FAIL,
            ("ort-rocket: ConvTranspose " + st->name + ": the NPU entry failed rc=" +
             std::to_string(rc)).c_str());
      const double t2 = st->prof ? NowMs() : 0;
      float* yn = y + n * out_n;
      for (size_t i = 0; i < out_n; i++) yn[i] = static_cast<float>(out16[i]);
      if (st->prof) { st->t_in += t1 - t0; st->t_npu += t2 - t1; st->t_out += NowMs() - t2; }
    }
    if (st->prof) st->calls++;
    return nullptr;
  }
};

}  // namespace

Shared::~Shared() {
  if (ctx) rocket_ctx_free(ctx);
}

std::shared_ptr<Shared> OpenShared(int nthreads, std::string* why) {
  // Not std::make_shared: its type tag is a GNU unique symbol, which makes the .so impossible to
  // dlclose, so unregistering the EP library would no longer unload it.
  std::shared_ptr<Shared> sh(new Shared);
  sh->ctx = rocket_ctx_create(nthreads);
  if (!sh->ctx) {
    *why = "rocket_ctx_create(" + std::to_string(nthreads) + ") failed; is /dev/accel/accel0 "
           "present and privileged?";
    return nullptr;
  }
  return sh;
}

bool Enabled() {
  const char* e = std::getenv("ROCKET_ORT_CONVTRANSPOSE");
  return !(e && std::atoi(e) == 0);
}

OrtStatus* Check(const OrtApi& api, const OrtNode* node, Spec* s, std::string* why) {
  *s = Spec{};
  why->clear();
  const char* dom = nullptr;
  if (OrtStatus* st = api.Node_GetDomain(node, &dom)) return st;
  if (dom && *dom && std::strcmp(dom, "ai.onnx") != 0) { *why = "not the ONNX domain"; return nullptr; }

  size_t ni = 0, no = 0;
  if (OrtStatus* st = api.Node_GetNumInputs(node, &ni)) return st;
  if (OrtStatus* st = api.Node_GetNumOutputs(node, &no)) return st;
  std::vector<const OrtValueInfo*> ins(ni), outs(no);
  if (OrtStatus* st = api.Node_GetInputs(node, ins.data(), ni)) return st;
  if (OrtStatus* st = api.Node_GetOutputs(node, outs.data(), no)) return st;
  if (ni < 2 || !ins[0] || !ins[1] || no < 1 || !outs[0]) { *why = "inputs"; return nullptr; }
  auto name_of = [&](const OrtValueInfo* vi, std::string* out) -> OrtStatus* {
    const char* nm = nullptr;
    if (OrtStatus* st = api.GetValueInfoName(vi, &nm)) return st;
    *out = nm ? nm : "";
    return nullptr;
  };
  if (OrtStatus* st = name_of(ins[0], &s->x_name)) return st;
  if (OrtStatus* st = name_of(ins[1], &s->w_name)) return st;

  // X and Y: fp32; X rank 4 when known.
  ONNXTensorElementDataType xet, yet;
  std::vector<int64_t> xd, yd;
  int xr = -1, yr = -1;
  if (OrtStatus* st = ValueShape(api, ins[0], &xet, &xd, &xr)) return st;
  if (OrtStatus* st = ValueShape(api, outs[0], &yet, &yd, &yr)) return st;
  if (xet != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || yet != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    *why = "not fp32";
    return nullptr;
  }
  if (xr != -1 && xr != 4) { *why = "not 2-D"; return nullptr; }

  // W: a constant fp32 initializer [IC][OC/group][KH][KW].
  const float* wdata = nullptr;
  std::vector<int64_t> wd;
  bool wok = false;
  if (OrtStatus* st = ConstF32(api, ins[1], &wdata, &wd, &wok)) return st;
  if (!wok) { *why = "weight not a constant fp32 initializer"; return nullptr; }
  if (wd.size() != 4) { *why = "not 2-D"; return nullptr; }
  for (int64_t d : wd)
    if (d <= 0 || d > (1 << 24)) { *why = "weight shape"; return nullptr; }

  int64_t group = 1;
  std::string auto_pad;
  std::vector<int64_t> ks, strides, dil, pads, opad, oshape;
  bool has_ks, has_st, has_dil, has_pads, has_opad, has_os;
  if (!ReadInt(api, node, "group", 1, &group) ||
      !ReadString(api, node, "auto_pad", "NOTSET", &auto_pad) ||
      !ReadInts(api, node, "kernel_shape", &ks, &has_ks) ||
      !ReadInts(api, node, "strides", &strides, &has_st) ||
      !ReadInts(api, node, "dilations", &dil, &has_dil) ||
      !ReadInts(api, node, "pads", &pads, &has_pads) ||
      !ReadInts(api, node, "output_padding", &opad, &has_opad) ||
      !ReadInts(api, node, "output_shape", &oshape, &has_os)) {
    *why = "attribute type";
    return nullptr;
  }
  if (group != 1) {
    *why = (group == wd[0] && wd[1] == 1) ? "depthwise" : "grouped";
    return nullptr;
  }
  if (has_os && !oshape.empty()) { *why = "output_shape"; return nullptr; }
  if (auto_pad != "NOTSET" && auto_pad != "VALID") { *why = "auto_pad SAME"; return nullptr; }

  s->ic = static_cast<int>(wd[0]);
  s->oc = static_cast<int>(wd[1]);
  s->kh = static_cast<int>(wd[2]);
  s->kw = static_cast<int>(wd[3]);
  if (has_ks && !ks.empty() && (ks.size() != 2 || ks[0] != wd[2] || ks[1] != wd[3])) {
    *why = "kernel_shape";
    return nullptr;
  }
  if ((has_st && !strides.empty() && strides.size() != 2) ||
      (has_dil && !dil.empty() && dil.size() != 2) ||
      (has_pads && !pads.empty() && pads.size() != 4) ||
      (has_opad && !opad.empty() && opad.size() != 2)) {
    *why = "not 2-D";
    return nullptr;
  }
  if (!strides.empty()) { s->sy = static_cast<int>(strides[0]); s->sx = static_cast<int>(strides[1]); }
  if (!dil.empty()) { s->dy = static_cast<int>(dil[0]); s->dx = static_cast<int>(dil[1]); }
  if (!pads.empty() && auto_pad == "NOTSET") {
    s->pt = static_cast<int>(pads[0]); s->pl = static_cast<int>(pads[1]);
    s->pb = static_cast<int>(pads[2]); s->pr = static_cast<int>(pads[3]);
  }
  if (!opad.empty()) { s->opy = static_cast<int>(opad[0]); s->opx = static_cast<int>(opad[1]); }
  if (s->sy < 1 || s->sx < 1 || s->dy < 1 || s->dx < 1 || s->pt < 0 || s->pl < 0 || s->pb < 0 ||
      s->pr < 0 || s->opy < 0 || s->opx < 0 || s->sy > 64 || s->sx > 64 || s->dy > 64 ||
      s->dx > 64 || s->pt > 4096 || s->pl > 4096 || s->pb > 4096 || s->pr > 4096 ||
      s->opy > 64 || s->opx > 64) {
    *why = "attribute range";
    return nullptr;
  }
  if (s->pb > s->pt + s->opy || s->pr > s->pl + s->opx) {
    *why = "trailing pad past the leading pad";
    return nullptr;
  }

  // B: optional, a constant fp32 initializer [OC].
  if (ni >= 3 && ins[2]) {
    if (OrtStatus* st = name_of(ins[2], &s->b_name)) return st;
    const float* bdata = nullptr;
    std::vector<int64_t> bd;
    bool bok = false;
    if (OrtStatus* st = ConstF32(api, ins[2], &bdata, &bd, &bok)) return st;
    if (!bok) { *why = "bias not a constant fp32 initializer"; return nullptr; }
    if (bd.size() != 1 || bd[0] != s->oc) { *why = "bias shape"; return nullptr; }
    s->has_bias = true;
  }

  if (xr == 4) {
    s->n = xd[0];
    s->ih = xd[2];
    s->iw = xd[3];
    if (xd[1] != -1 && xd[1] != s->ic) { *why = "input channels differ from the weight's"; return nullptr; }
  }
  // The plan at the static size, or at a nominal one when the size is symbolic: then it checks
  // everything the node fixes, and Compute plans each size it meets.
  const bool fixed = s->ih > 0 && s->iw > 0;
  const int ih = fixed ? static_cast<int>(s->ih) : 16, iw = fixed ? static_cast<int>(s->iw) : 16;
  if (fixed && (s->OH(ih) <= 0 || s->OW(iw) <= 0)) { *why = "empty output"; return nullptr; }
  const rocket_conv_transpose2d_desc d = MakeDesc(*s, ih, iw);
  const int rc = rocket_conv_transpose2d_prepacked_plan(&d);
  if (rc && (fixed || rc != -3)) { *why = PlanReason(rc); return nullptr; }
  return nullptr;
}

bool IsConvTGraph(const OrtApi& api, const OrtGraph* graph) {
  size_t nn = 0;
  if (OrtStatus* st = api.Graph_GetNumNodes(graph, &nn)) { api.ReleaseStatus(st); return false; }
  if (nn != 1) return false;
  const OrtNode* n = nullptr;
  if (OrtStatus* st = api.Graph_GetNodes(graph, &n, 1)) { api.ReleaseStatus(st); return false; }
  const char* op = nullptr;
  if (OrtStatus* st = api.Node_GetOperatorType(n, &op)) { api.ReleaseStatus(st); return false; }
  return op && std::strcmp(op, "ConvTranspose") == 0;
}

OrtStatus* Compile(const OrtApi& api, const OrtLogger& logger, std::shared_ptr<Shared> shared,
                   const OrtGraph* graph, const OrtNode* fused_node, OrtNodeComputeInfo** info) {
  *info = nullptr;
  if (!shared || !shared->ctx)
    return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: a ConvTranspose was claimed without an open context");
  const OrtNode* node = nullptr;
  if (OrtStatus* st = api.Graph_GetNodes(graph, &node, 1)) return st;
  auto st = std::make_unique<State>(api);
  st->sh = std::move(shared);
  std::string why;
  if (OrtStatus* e = Check(api, node, &st->s, &why)) return e;
  if (!why.empty())
    return api.CreateStatus(ORT_EP_FAIL,
        ("ort-rocket: a claimed ConvTranspose no longer passes its check at Compile (" + why +
         "); this is an internal inconsistency").c_str());
  const char* nm = nullptr;
  if (OrtStatus* e = api.Node_GetName(node, &nm)) return e;
  st->name = nm && *nm ? nm : "(input " + st->s.x_name + ")";
  const char* pz = std::getenv("ROCKET_ORT_PROF");
  st->prof = pz && std::atoi(pz) != 0;

  // The weight and bias, read from the node's own input edges.
  size_t ni = 0;
  if (OrtStatus* e = api.Node_GetNumInputs(node, &ni)) return e;
  std::vector<const OrtValueInfo*> ins(ni);
  if (OrtStatus* e = api.Node_GetInputs(node, ins.data(), ni)) return e;
  const float* w = nullptr;
  const float* b = nullptr;
  std::vector<int64_t> dims;
  bool ok = false;
  if (OrtStatus* e = ConstF32(api, ins[1], &w, &dims, &ok)) return e;
  if (!ok) return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: ConvTranspose weight unreadable at Compile");
  if (st->s.has_bias) {
    if (OrtStatus* e = ConstF32(api, ins[2], &b, &dims, &ok)) return e;
    if (!ok) return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: ConvTranspose bias unreadable at Compile");
  }

  // The fused node's activation input. Constant initializers were dropped from its inputs at
  // GetCapability, so it is the one input named as the node's X.
  size_t fni = 0;
  if (OrtStatus* e = api.Node_GetNumInputs(fused_node, &fni)) return e;
  std::vector<const OrtValueInfo*> fins(fni);
  if (OrtStatus* e = api.Node_GetInputs(fused_node, fins.data(), fni)) return e;
  bool found = false;
  for (size_t i = 0; i < fni; i++) {
    if (!fins[i]) continue;
    const char* in_nm = nullptr;
    if (OrtStatus* e = api.GetValueInfoName(fins[i], &in_nm)) return e;
    if (in_nm && st->s.x_name == in_nm) { st->x_idx = i; found = true; break; }
  }
  if (!found)
    return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: ConvTranspose input not found on the fused node");

  const Spec& s = st->s;
  const size_t nw = static_cast<size_t>(s.ic) * s.oc * s.kh * s.kw;
  const bool fixed = s.ih > 0 && s.iw > 0;
  if (fixed) {
    rocket_conv_transpose2d_weights* h = nullptr;
    const int rc = PackAt(st.get(), w, b, static_cast<int>(s.ih), static_cast<int>(s.iw), &h);
    if (!h)
      return api.CreateStatus(ORT_EP_FAIL,
          ("ort-rocket: ConvTranspose " + st->name + ": weight pack failed at Compile (" +
           std::to_string(rc) + ")").c_str());
    st->packs.push_back({static_cast<int>(s.ih), static_cast<int>(s.iw), h});
  } else {
    st->w32.assign(w, w + nw);
    if (b) st->b32.assign(b, b + s.oc);
  }
  Log(api, logger, ORT_LOGGING_LEVEL_INFO,
      "ort-rocket: ConvTranspose " + st->name + " compiled: " + std::to_string(s.ic) + "->" +
          std::to_string(s.oc) + " k" + std::to_string(s.kh) + "x" + std::to_string(s.kw) + " s" +
          std::to_string(s.sy) + "x" + std::to_string(s.sx) + " d" + std::to_string(s.dy) + "x" +
          std::to_string(s.dx) + " pads " + std::to_string(s.pt) + "," + std::to_string(s.pl) + "," +
          std::to_string(s.pb) + "," + std::to_string(s.pr) + " opad " + std::to_string(s.opy) +
          "," + std::to_string(s.opx) + (s.has_bias ? " +bias" : "") + ", input " +
          (fixed ? std::to_string(s.ih) + "x" + std::to_string(s.iw) + " (packed)"
                 : std::string("size symbolic (packed per size at Compute)")));
  *info = new ConvTComputeInfo(st.release());
  return nullptr;
}

}  // namespace rocket_convt
