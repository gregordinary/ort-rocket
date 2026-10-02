// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_convt: the ConvTranspose handler. Unlike the families, which claim one encoder subgraph by
// structure, this claims ConvTranspose NODES one at a time, each as its own fused node, and runs
// each on librocketnpu's resident transposed convolution (col2im over the resident fp16 matmul).
//
// A node is claimed when GetCapability can see that the entry will run it: a 2-D ConvTranspose with
// group 1, fp32 input and output, a constant fp32 weight and an optional constant fp32 bias,
// auto_pad NOTSET or VALID, no output_shape attribute, trailing pads no larger than the leading pad
// plus output_padding, and a descriptor rocket_conv_transpose2d_prepacked_plan accepts. Everything
// else stays on ONNX Runtime's CPU kernel, including every depthwise and grouped transpose.
//
// The weight is packed once per node at Compile, on one rocket_ctx shared by every ConvTranspose
// node of the EP instance. A node whose input height or width is symbolic packs at the first
// Compute that sees each spatial size, and runs a size the entry's plan refuses on a host fp32
// scatter, so a claimed node never fails a Run for its shape.

#ifndef ORT_ROCKET_CONVT_H
#define ORT_ROCKET_CONVT_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "onnxruntime_c_api.h"

struct rocket_ctx;

namespace rocket_convt {

// The base of every compute info this EP hands ONNX Runtime, so ReleaseNodeComputeInfos can free
// either kind through one pointer type.
struct ComputeInfoBase : OrtNodeComputeInfo {
  ComputeInfoBase() : OrtNodeComputeInfo{} {}
  virtual ~ComputeInfoBase() = default;
};

// The one resident-matmul context every ConvTranspose node of an EP instance packs on and runs on.
// The context is not thread-safe, so every Compute holds `mu` for its whole body. Each compiled node
// holds a reference, so the context outlives every weight packed on it whatever order ONNX Runtime
// releases the EP and the compute infos in.
struct Shared {
  rocket_ctx* ctx = nullptr;
  std::mutex mu;
  ~Shared();
};

// Opens the context with `nthreads` worker fds. Returns null (and sets *why) when the device does not
// open, in which case GetCapability claims no ConvTranspose node.
std::shared_ptr<Shared> OpenShared(int nthreads, std::string* why);

// The handler's switch: ROCKET_ORT_CONVTRANSPOSE=0 turns it off, so every ConvTranspose runs on the
// CPU kernel.
bool Enabled();

// A parsed ConvTranspose node. Dimensions of the input that are symbolic in the graph are -1.
struct Spec {
  int ic = 0, oc = 0, kh = 0, kw = 0;
  int sy = 1, sx = 1, dy = 1, dx = 1;
  int pt = 0, pl = 0, pb = 0, pr = 0;   // ONNX pads [top, left, bottom, right]
  int opy = 0, opx = 0;                 // output_padding
  bool has_bias = false;
  int64_t n = -1, ih = -1, iw = -1;
  std::string x_name, w_name, b_name;
  int OH(int ih_) const { return (ih_ - 1) * sy - pt - pb + dy * (kh - 1) + opy + 1; }
  int OW(int iw_) const { return (iw_ - 1) * sx - pl - pr + dx * (kw - 1) + opx + 1; }
};

// Parses one node and decides whether the EP claims it. Returns an error status only for an ONNX
// Runtime API failure. On return *why is empty when the node is claimable, else the refusal's class
// (a short fixed string, so GetCapability can count refusals by reason).
OrtStatus* Check(const OrtApi& api, const OrtNode* node, Spec* spec, std::string* why);

// True when a fused subgraph is one ConvTranspose node, which is how Compile tells a ConvTranspose
// claim from the family claim.
bool IsConvTGraph(const OrtApi& api, const OrtGraph* graph);

// Compiles one claimed ConvTranspose: reads its weight and bias, packs the weight on `shared` when
// the input's spatial size is static, and hands back the node's compute info.
OrtStatus* Compile(const OrtApi& api, const OrtLogger& logger, std::shared_ptr<Shared> shared,
                   const OrtGraph* graph, const OrtNode* fused_node, OrtNodeComputeInfo** info);

}  // namespace rocket_convt

#endif  // ORT_ROCKET_CONVT_H
