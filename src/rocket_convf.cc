// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_convf: the per-node forward Conv handler. See rocket_convf.h for what is claimed.
//
// Per call and per batch image the handler converts the fp32 NCHW input to fp16, runs
// rocket_conv2d_fp16_prepacked (a threaded host unfold and one resident GEMM), and widens the fp16
// output back to fp32 with the bias and the folded activation applied in fp32. The layouts need no
// transpose: ONNX's X [N][C][H][W], W [OC][IC][KH][KW] and Y [N][OC][OH][OW] are the entry's own for
// one batch image and group 1. The weight is packed without its bias, so the bias is added to the
// fp32 widening of the GEMM's fp16 output and rounds once, into fp32.

#include "rocket_convf.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "rocket_conv.h"
#include "rocket_matmul.h"
#include "rocket_node_attr.h"

namespace rocket_convf {
namespace {

using rocket_node::ConstF32;
using rocket_node::ReadFloat;
using rocket_node::ReadInt;
using rocket_node::ReadInts;
using rocket_node::ReadString;
using rocket_node::ValueShape;

inline void Discard(const OrtApi& api, OrtStatus* st) {
  if (st) api.ReleaseStatus(st);
}

void Log(const OrtApi& api, const OrtLogger& lg, OrtLoggingLevel lvl, const std::string& msg) {
  Discard(api, api.Logger_LogMessage(&lg, lvl, msg.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
}

double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string OpType(const OrtApi& api, const OrtNode* n) {
  const char* op = nullptr;
  OrtStatus* st = api.Node_GetOperatorType(n, &op);
  if (st) { api.ReleaseStatus(st); return std::string(); }
  return op ? std::string(op) : std::string();
}

bool OnnxDomain(const OrtApi& api, const OrtNode* n) {
  const char* dom = nullptr;
  OrtStatus* st = api.Node_GetDomain(n, &dom);
  if (st) { api.ReleaseStatus(st); return false; }
  return !dom || !*dom || std::strcmp(dom, "ai.onnx") == 0;
}

rocket_conv2d_desc MakeDesc(const Spec& s) {
  rocket_conv2d_desc d{};
  d.ic = s.ic; d.ih = s.ih; d.iw = s.iw; d.oc = s.oc;
  d.kh = s.kh; d.kw = s.kw;
  d.stride_y = s.sy; d.stride_x = s.sx;
  d.pad_top = s.pt; d.pad_left = s.pl;
  d.dil_y = s.dy; d.dil_x = s.dx;
  // The trailing pads are expressed as the output extent: the entry reads every tap past the plane
  // as zero, so an explicit extent is exactly ONNX's leading-plus-trailing pad.
  d.oh = s.oh; d.ow = s.ow;
  return d;
}

std::string PlanReason(int rc) {
  if (rc == ROCKET_E_UNSUPPORTED) return "plan: unsupported (not the RK3588)";
  if (rc == ROCKET_E_SHAPE) return "plan: invalid descriptor";
  if (rc == -3) return "plan: empty output";
  if (rc == -4) return "plan: past the entry's size bound";
  return "plan: refused (" + std::to_string(rc) + ")";
}

// One axis of auto_pad SAME: the total pad that makes the output ceil(I / s), split with the odd one
// at the end (SAME_UPPER) or at the start (SAME_LOWER).
void SamePads(int in, int k, int s, int d, bool upper, int* begin, int* end) {
  const int out = (in + s - 1) / s;
  int total = (out - 1) * s + (k - 1) * d + 1 - in;
  if (total < 0) total = 0;
  *begin = upper ? total / 2 : total - total / 2;
  *end = total - *begin;
}

// x (fp32) -> y (fp16), n elements.
void ToF16(const float* x, _Float16* y, size_t n) {
  size_t i = 0;
#if defined(__aarch64__)
  for (; i + 8 <= n; i += 8)
    vst1q_f16(reinterpret_cast<__fp16*>(y + i),
              vcombine_f16(vcvt_f16_f32(vld1q_f32(x + i)), vcvt_f16_f32(vld1q_f32(x + i + 4))));
#endif
  for (; i < n; i++) y[i] = static_cast<_Float16>(x[i]);
}

// y = act(float(x) + b) over one output plane of n elements.
void Widen(const _Float16* x, float* y, size_t n, float b, Act act, float alpha) {
  size_t i = 0;
#if defined(__aarch64__)
  const float32x4_t vb = vdupq_n_f32(b), z = vdupq_n_f32(0.0f), va = vdupq_n_f32(alpha);
  for (; i + 8 <= n; i += 8) {
    const float16x8_t h = vld1q_f16(reinterpret_cast<const __fp16*>(x + i));
    float32x4_t v[2] = {vaddq_f32(vcvt_f32_f16(vget_low_f16(h)), vb),
                        vaddq_f32(vcvt_high_f32_f16(h), vb)};
    for (int q = 0; q < 2; q++) {
      if (act == Act::kRelu) v[q] = vmaxq_f32(v[q], z);
      else if (act == Act::kLeakyRelu) v[q] = vbslq_f32(vcgeq_f32(v[q], z), v[q], vmulq_f32(v[q], va));
      vst1q_f32(y + i + 4 * q, v[q]);
    }
  }
#endif
  for (; i < n; i++) {
    float v = static_cast<float>(x[i]) + b;
    if (act == Act::kRelu) v = v > 0.0f ? v : 0.0f;
    else if (act == Act::kLeakyRelu) v = v >= 0.0f ? v : v * alpha;
    y[i] = v;
  }
}

// One compiled Conv node.
struct State {
  explicit State(const OrtApi& a) : api(a) {}
  ~State();

  const OrtApi& api;
  std::shared_ptr<rocket_convt::Shared> sh;
  Spec s;
  std::string name;
  size_t x_idx = 0;
  rocket_conv2d_weights* h = nullptr;
  std::vector<float> bias;   // OC entries, zeros without a bias
  std::vector<_Float16> s_in, s_out;
  bool prof = false;
  long calls = 0;
  double t_in = 0, t_npu = 0, t_out = 0;
  std::vector<double> npu_ms;   // every call's NPU entry, so the first call can be told apart
};

State::~State() {
  if (prof && calls)
    std::fprintf(stderr,
                 "ort-rocket prof Conv %s: %ld calls, per call in %.3f ms, npu entry %.3f ms, "
                 "out %.3f ms; npu entry first %.3f ms, median after it %.3f ms\n",
                 name.c_str(), calls, t_in / calls, t_npu / calls, t_out / calls,
                 rocket_convt::FirstMs(npu_ms), rocket_convt::MedianAfterFirstMs(npu_ms));
  if (h) rocket_conv2d_weights_free(sh->ctx, h);
}

struct ConvComputeInfo : rocket_convt::ComputeInfoBase {
  explicit ConvComputeInfo(State* st) : state(st) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }
  ~ConvComputeInfo() override { delete state; }
  State* state;

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* /*ctx*/,
                                                 void** compute_state) noexcept {
    *compute_state = static_cast<ConvComputeInfo*>(this_ptr)->state;
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
    if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || xd.size() != 4 || xd[1] != s.ic ||
        xd[2] != s.ih || xd[3] != s.iw || xd[0] < 0)
      return api.CreateStatus(ORT_INVALID_ARGUMENT,
          ("ort-rocket: Conv " + st->name + ": input must be fp32 [N," + std::to_string(s.ic) + "," +
           std::to_string(s.ih) + "," + std::to_string(s.iw) + "], the size it was compiled for").c_str());
    const int64_t N = xd[0];
    const int64_t yshape[4] = {N, s.oc, s.oh, s.ow};
    OrtValue* yv = nullptr;
    if (OrtStatus* e = api.KernelContext_GetOutput(kctx, 0, yshape, 4, &yv)) return e;
    void* yraw = nullptr;
    if (OrtStatus* e = api.GetTensorMutableData(yv, &yraw)) return e;
    const void* xraw = nullptr;
    if (OrtStatus* e = api.GetTensorData(xv, &xraw)) return e;
    const float* x = static_cast<const float*>(xraw);
    float* y = static_cast<float*>(yraw);
    if (N == 0) return nullptr;

    const size_t in_n = static_cast<size_t>(s.ic) * s.ih * s.iw;
    const size_t plane = static_cast<size_t>(s.oh) * s.ow;
    const size_t out_n = static_cast<size_t>(s.oc) * plane;
    st->s_in.resize(in_n);
    st->s_out.resize(out_n);
    for (int64_t n = 0; n < N; n++) {
      const double t0 = st->prof ? NowMs() : 0;
      ToF16(x + n * in_n, st->s_in.data(), in_n);
      const double t1 = st->prof ? NowMs() : 0;
      const int rc = rocket_conv2d_fp16_prepacked(st->sh->ctx, st->h, st->s_in.data(), st->s_out.data());
      if (rc)
        return api.CreateStatus(ORT_FAIL, ("ort-rocket: Conv " + st->name +
                                           ": the NPU entry failed rc=" + std::to_string(rc)).c_str());
      const double t2 = st->prof ? NowMs() : 0;
      float* yn = y + n * out_n;
      for (int oc = 0; oc < s.oc; oc++)
        Widen(st->s_out.data() + oc * plane, yn + oc * plane, plane, st->bias[oc], s.act, s.alpha);
      if (st->prof) {
        st->t_in += t1 - t0; st->t_npu += t2 - t1; st->t_out += NowMs() - t2;
        st->npu_ms.push_back(t2 - t1);
      }
    }
    if (st->prof) st->calls++;
    return nullptr;
  }
};

}  // namespace

Mode GetMode() {
  const char* e = std::getenv("ROCKET_ORT_CONV");
  if (!e) return Mode::kCost;
  if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0) return Mode::kOff;
  if (std::strcmp(e, "all") == 0) return Mode::kAll;
  return Mode::kCost;
}

OrtStatus* Check(const OrtApi& api, const OrtNode* node, Spec* s, std::string* why) {
  *s = Spec{};
  why->clear();
  if (!OnnxDomain(api, node)) { *why = "not the ONNX domain"; return nullptr; }

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
  if (OrtStatus* st = name_of(outs[0], &s->y_name)) return st;

  // X and Y: fp32; X rank 4 with a static spatial size.
  ONNXTensorElementDataType xet, yet;
  std::vector<int64_t> xd, yd;
  int xr = -1, yr = -1;
  if (OrtStatus* st = ValueShape(api, ins[0], &xet, &xd, &xr)) return st;
  if (OrtStatus* st = ValueShape(api, outs[0], &yet, &yd, &yr)) return st;
  if (xet != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || yet != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    *why = "not fp32";
    return nullptr;
  }
  if (xr != 4) { *why = xr == -1 ? "input shape unknown" : "not 2-D"; return nullptr; }
  if (xd[2] <= 0 || xd[3] <= 0) { *why = "input size symbolic"; return nullptr; }
  if (xd[2] > (1 << 20) || xd[3] > (1 << 20)) { *why = "input size"; return nullptr; }

  // W: a constant fp32 initializer [OC][IC/group][KH][KW].
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
  std::vector<int64_t> ks, strides, dil, pads;
  bool has_ks, has_st, has_dil, has_pads;
  if (!ReadInt(api, node, "group", 1, &group) ||
      !ReadString(api, node, "auto_pad", "NOTSET", &auto_pad) ||
      !ReadInts(api, node, "kernel_shape", &ks, &has_ks) ||
      !ReadInts(api, node, "strides", &strides, &has_st) ||
      !ReadInts(api, node, "dilations", &dil, &has_dil) ||
      !ReadInts(api, node, "pads", &pads, &has_pads)) {
    *why = "attribute type";
    return nullptr;
  }
  if (group != 1) {
    *why = (wd[1] == 1 && group == wd[0]) ? "depthwise" : "grouped";
    return nullptr;
  }
  s->oc = static_cast<int>(wd[0]);
  s->ic = static_cast<int>(wd[1]);
  s->kh = static_cast<int>(wd[2]);
  s->kw = static_cast<int>(wd[3]);
  if (xd[1] != -1 && xd[1] != s->ic) { *why = "input channels differ from the weight's"; return nullptr; }
  if (has_ks && !ks.empty() && (ks.size() != 2 || ks[0] != wd[2] || ks[1] != wd[3])) {
    *why = "kernel_shape";
    return nullptr;
  }
  if ((has_st && !strides.empty() && strides.size() != 2) ||
      (has_dil && !dil.empty() && dil.size() != 2) ||
      (has_pads && !pads.empty() && pads.size() != 4)) {
    *why = "not 2-D";
    return nullptr;
  }
  if (!strides.empty()) { s->sy = static_cast<int>(strides[0]); s->sx = static_cast<int>(strides[1]); }
  if (!dil.empty()) { s->dy = static_cast<int>(dil[0]); s->dx = static_cast<int>(dil[1]); }
  if (s->sy < 1 || s->sx < 1 || s->dy < 1 || s->dx < 1 || s->sy > 64 || s->sx > 64 || s->dy > 64 ||
      s->dx > 64) {
    *why = "attribute range";
    return nullptr;
  }
  s->n = xd[0];
  s->ih = static_cast<int>(xd[2]);
  s->iw = static_cast<int>(xd[3]);
  if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    const bool upper = auto_pad == "SAME_UPPER";
    SamePads(s->ih, s->kh, s->sy, s->dy, upper, &s->pt, &s->pb);
    SamePads(s->iw, s->kw, s->sx, s->dx, upper, &s->pl, &s->pr);
  } else if (auto_pad == "NOTSET") {
    if (!pads.empty()) {
      s->pt = static_cast<int>(pads[0]); s->pl = static_cast<int>(pads[1]);
      s->pb = static_cast<int>(pads[2]); s->pr = static_cast<int>(pads[3]);
    }
  } else if (auto_pad != "VALID") {
    *why = "auto_pad";
    return nullptr;
  }
  if (s->pt < 0 || s->pl < 0 || s->pb < 0 || s->pr < 0 || s->pt > 4096 || s->pl > 4096 ||
      s->pb > 4096 || s->pr > 4096) {
    *why = "attribute range";
    return nullptr;
  }
  s->oh = (s->ih + s->pt + s->pb - ((s->kh - 1) * s->dy + 1)) / s->sy + 1;
  s->ow = (s->iw + s->pl + s->pr - ((s->kw - 1) * s->dx + 1)) / s->sx + 1;
  if (s->ih + s->pt + s->pb < (s->kh - 1) * s->dy + 1 || s->iw + s->pl + s->pr < (s->kw - 1) * s->dx + 1 ||
      s->oh <= 0 || s->ow <= 0) {
    *why = "empty output";
    return nullptr;
  }
  if (yr == 4 && ((yd[2] > 0 && yd[2] != s->oh) || (yd[3] > 0 && yd[3] != s->ow))) {
    *why = "output size differs from the attributes'";
    return nullptr;
  }

  // B: optional, a constant fp32 initializer [OC].
  if (ni >= 3 && ins[2]) {
    const float* bdata = nullptr;
    std::vector<int64_t> bd;
    bool bok = false;
    if (OrtStatus* st = ConstF32(api, ins[2], &bdata, &bd, &bok)) return st;
    if (!bok) { *why = "bias not a constant fp32 initializer"; return nullptr; }
    if (bd.size() != 1 || bd[0] != s->oc) { *why = "bias shape"; return nullptr; }
    s->has_bias = true;
  }

  const rocket_conv2d_desc d = MakeDesc(*s);
  const int rc = rocket_conv2d_prepacked_plan(&d);
  if (rc) { *why = PlanReason(rc); return nullptr; }
  return nullptr;
}

OrtStatus* FoldableActivation(const OrtApi& api, const OrtNode* conv, Spec* s, const OrtNode** act_node) {
  *act_node = nullptr;
  s->act = Act::kNone;
  size_t no = 0;
  if (OrtStatus* st = api.Node_GetNumOutputs(conv, &no)) return st;
  if (no != 1) return nullptr;
  const OrtValueInfo* y = nullptr;
  if (OrtStatus* st = api.Node_GetOutputs(conv, &y, 1)) return st;
  if (!y) return nullptr;
  bool is_out = false;
  if (OrtStatus* st = api.ValueInfo_IsGraphOutput(y, &is_out)) return st;
  if (is_out) return nullptr;
  size_t nc = 0;
  if (OrtStatus* st = api.ValueInfo_GetValueNumConsumers(y, &nc)) return st;
  if (nc != 1) return nullptr;
  const OrtNode* c = nullptr;
  int64_t idx = -1;
  if (OrtStatus* st = api.ValueInfo_GetValueConsumers(y, &c, &idx, 1)) return st;
  if (!c || idx != 0 || !OnnxDomain(api, c)) return nullptr;
  const std::string op = OpType(api, c);
  if (op == "Relu") {
    s->act = Act::kRelu;
  } else if (op == "LeakyRelu") {
    float a = 0.01f;
    if (!ReadFloat(api, c, "alpha", 0.01f, &a)) return nullptr;
    s->act = Act::kLeakyRelu;
    s->alpha = a;
  } else {
    return nullptr;
  }
  *act_node = c;
  return nullptr;
}

// ONNX Runtime's CPU kernel for one Conv node at 4 intra-op threads on the A76s, in ms: a linear
// model in its MACs, its fp32 weight bytes and its input plus output elements, fitted (relative
// error) to 29 nodes of pix2pix's encoder and Depth Anything v2-small's neck and head, profiled at
// ORT 1.27.0 with allow_spinning 0 [HW, mainline RK1, 2026-10-08]. It lands within 0.71-1.23x of
// the measured time at all but one node: the 3x3 stride-2 reassemble conv, where the CPU kernel is
// 2.7x slower than the model says. The EP cannot read the session's thread count, so the rule is
// fitted at the configuration the README recommends; fewer threads make the CPU slower, which only
// makes a refused node a missed win. The fit ran on the ondemand governor; under performance the same
// kernels take 0.93-0.97x its time, which leans the rule toward claiming.
double CpuEstimateMs(const Spec& s) {
  const double out = static_cast<double>(s.oc) * s.oh * s.ow;
  const double gmac = out * s.ic * s.kh * s.kw / 1e9;
  const double wmb = static_cast<double>(s.oc) * s.ic * s.kh * s.kw * 4 / 1e6;
  const double io = (static_cast<double>(s.ic) * s.ih * s.iw + out) / 1e6;
  return 0.1021 + 20.1358 * gmac + 0.0966 * wmb + 0.3402 * io;
}

// The rule admits a node whose estimated NPU call plus conversions is at most kMargin of the CPU
// estimate. The NPU estimate is fitted to back-to-back calls. Inside a session the claimed nodes run
// 1.01-1.04x that time on Depth Anything's neck and 0.95-1.20x on pix2pix's encoder, since
// librocketnpu creates its workers on their own cores; before that, the encoder's caller pin put
// every worker on one core and the same nodes ran 1.27-1.29x. At 0.9 Depth Anything's wall is
// 0.976-0.979x the claim off under the performance and the ondemand governor, against 0.990-0.999x
// at 0.75; pix2pix claims the same seven convs at both [HW, mainline RK1, the 600 MHz request
// (~816 MHz), five rotated passes, 2026-10-09 UTC]. ROCKET_ORT_CONV_MARGIN overrides it, for
// measuring the rule against a model.
constexpr double kMargin = 0.9;

double Margin() {
  static const double m = [] {
    const char* e = std::getenv("ROCKET_ORT_CONV_MARGIN");
    const double v = e ? std::atof(e) : 0.0;
    return v > 0 ? v : kMargin;
  }();
  return m;
}

bool CostRule(const Spec& s, std::string* note) {
  const rocket_conv2d_desc d = MakeDesc(s);
  const double npu = rocket_conv2d_prepacked_estimate_ms(&d);
  // The fp32 -> fp16 input conversion and the fp16 -> fp32 widening, one thread, ~0.3 ns an
  // element measured on the ConvTranspose claim.
  const double conv = 0.3e-6 * (static_cast<double>(s.ic) * s.ih * s.iw +
                                static_cast<double>(s.oc) * s.oh * s.ow);
  const double cpu = CpuEstimateMs(s);
  char buf[160];
  std::snprintf(buf, sizeof(buf), "npu %.2f + conversions %.2f ms against cpu %.2f ms", npu, conv, cpu);
  *note = buf;
  return npu >= 0 && npu + conv <= Margin() * cpu;
}

bool IsConvGraph(const OrtApi& api, const OrtGraph* graph) {
  size_t nn = 0;
  if (OrtStatus* st = api.Graph_GetNumNodes(graph, &nn)) { api.ReleaseStatus(st); return false; }
  if (nn != 1 && nn != 2) return false;
  const OrtNode* n[2] = {nullptr, nullptr};
  if (OrtStatus* st = api.Graph_GetNodes(graph, n, nn)) { api.ReleaseStatus(st); return false; }
  int convs = 0, acts = 0;
  for (size_t i = 0; i < nn; i++) {
    const std::string op = OpType(api, n[i]);
    if (op == "Conv") convs++;
    else if (op == "Relu" || op == "LeakyRelu") acts++;
  }
  return convs == 1 && convs + acts == static_cast<int>(nn);
}

OrtStatus* Compile(const OrtApi& api, const OrtLogger& logger, std::shared_ptr<rocket_convt::Shared> shared,
                   const OrtGraph* graph, const OrtNode* fused_node, OrtNodeComputeInfo** info) {
  *info = nullptr;
  if (!shared || !shared->ctx)
    return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: a Conv was claimed without an open context");
  size_t nn = 0;
  if (OrtStatus* e = api.Graph_GetNumNodes(graph, &nn)) return e;
  const OrtNode* nodes[2] = {nullptr, nullptr};
  if (nn < 1 || nn > 2) return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: a Conv claim of an unexpected size");
  if (OrtStatus* e = api.Graph_GetNodes(graph, nodes, nn)) return e;
  const OrtNode* node = nullptr;
  const OrtNode* act = nullptr;
  for (size_t i = 0; i < nn; i++) {
    if (OpType(api, nodes[i]) == "Conv") node = nodes[i];
    else act = nodes[i];
  }
  if (!node) return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: a Conv claim without a Conv");

  auto st = std::make_unique<State>(api);
  st->sh = std::move(shared);
  std::string why;
  if (OrtStatus* e = Check(api, node, &st->s, &why)) return e;
  if (!why.empty())
    return api.CreateStatus(ORT_EP_FAIL,
        ("ort-rocket: a claimed Conv no longer passes its check at Compile (" + why +
         "); this is an internal inconsistency").c_str());
  if (act) {
    const std::string op = OpType(api, act);
    if (op == "Relu") {
      st->s.act = Act::kRelu;
    } else {
      float a = 0.01f;
      if (!ReadFloat(api, act, "alpha", 0.01f, &a))
        return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: a folded LeakyRelu's alpha is unreadable");
      st->s.act = Act::kLeakyRelu;
      st->s.alpha = a;
    }
  }
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
  if (!ok) return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: Conv weight unreadable at Compile");
  if (st->s.has_bias) {
    if (OrtStatus* e = ConstF32(api, ins[2], &b, &dims, &ok)) return e;
    if (!ok) return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: Conv bias unreadable at Compile");
  }

  // The fused node's activation input. Constant initializers were dropped from its inputs at
  // GetCapability, so it is the one input named as the conv's X.
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
  if (!found) return api.CreateStatus(ORT_EP_FAIL, "ort-rocket: Conv input not found on the fused node");

  const Spec& s = st->s;
  const size_t nw = static_cast<size_t>(s.oc) * s.ic * s.kh * s.kw;
  std::vector<_Float16> w16(nw);
  for (size_t i = 0; i < nw; i++) w16[i] = static_cast<_Float16>(w[i]);
  const rocket_conv2d_desc d = MakeDesc(s);
  st->h = rocket_conv2d_weights_pack(st->sh->ctx, &d, w16.data(), nullptr);
  if (!st->h)
    return api.CreateStatus(ORT_EP_FAIL, ("ort-rocket: Conv " + st->name + ": weight pack failed at Compile").c_str());
  st->bias.assign(s.oc, 0.0f);
  if (b) st->bias.assign(b, b + s.oc);

  std::string note;
  CostRule(s, &note);
  Log(api, logger, ORT_LOGGING_LEVEL_INFO,
      "ort-rocket: Conv " + st->name + " compiled: " + std::to_string(s.ic) + "x" + std::to_string(s.ih) +
          "x" + std::to_string(s.iw) + " -> " + std::to_string(s.oc) + "x" + std::to_string(s.oh) + "x" +
          std::to_string(s.ow) + " k" + std::to_string(s.kh) + "x" + std::to_string(s.kw) + " s" +
          std::to_string(s.sy) + "x" + std::to_string(s.sx) + " d" + std::to_string(s.dy) + "x" +
          std::to_string(s.dx) + " pads " + std::to_string(s.pt) + "," + std::to_string(s.pl) + "," +
          std::to_string(s.pb) + "," + std::to_string(s.pr) + (s.has_bias ? " +bias" : "") +
          (s.act == Act::kRelu ? " +Relu" : s.act == Act::kLeakyRelu ? " +LeakyRelu" : "") + " (" + note + ")");
  *info = new ConvComputeInfo(st.release());
  return nullptr;
}

}  // namespace rocket_convf
