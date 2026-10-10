// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_convf: the forward Conv handler. Like rocket_convt, it claims NODES one at a time, each as
// its own fused node, and runs each on librocketnpu's resident forward convolution (a host im2col
// into channel planes feeding the resident fp16 matmul), on the context the ConvTranspose claim
// shares.
//
// A node is claimed when GetCapability can see that the entry will run it AND that it is worth
// running there: a 2-D Conv with group 1, fp32 input and output, a static spatial size, a constant
// fp32 weight and an optional constant fp32 bias, any auto_pad, a descriptor
// rocket_conv2d_prepacked_plan accepts, and a cost rule that estimates the NPU call plus the fp32
// conversions at no more than 0.75x ONNX Runtime's 4-thread CPU kernel (see CostRule). Everything else
// stays on ONNX Runtime's CPU kernel, including every depthwise and grouped conv.
//
// A Relu or LeakyRelu that is the conv's only consumer, and whose input is not a graph output, is
// claimed into the same fused node and applied in the fp16 -> fp32 widening pass, so the claim does
// not undo the CPU EP's own Conv + activation fusion.

#ifndef ORT_ROCKET_CONVF_H
#define ORT_ROCKET_CONVF_H

#include <cstdint>
#include <string>

#include "onnxruntime_c_api.h"
#include "rocket_convt.h"

namespace rocket_convf {

// The handler's switch, ROCKET_ORT_CONV: 0 turns it off, so every Conv runs on the CPU kernel; "all"
// claims every node the entry can run, skipping the cost rule (for measurement); anything else, or
// unset, applies the cost rule.
enum class Mode { kOff, kCost, kAll };
Mode GetMode();

enum class Act { kNone, kRelu, kLeakyRelu };

// A parsed Conv node at its static size.
struct Spec {
  int ic = 0, oc = 0, kh = 0, kw = 0;
  int sy = 1, sx = 1, dy = 1, dx = 1;
  int pt = 0, pl = 0, pb = 0, pr = 0;   // resolved pads [top, left, bottom, right]
  int ih = 0, iw = 0, oh = 0, ow = 0;
  int64_t n = -1;                       // batch, -1 when symbolic
  bool has_bias = false;
  std::string x_name, y_name;           // the conv's input and output
  Act act = Act::kNone;
  float alpha = 0.01f;                  // LeakyRelu's slope
};

// Parses one Conv node and decides whether the entry can run it. Returns an error status only for an
// ONNX Runtime API failure. On return *why is empty when the node is runnable, else the refusal's
// class (a short fixed string, so GetCapability can count refusals by reason). The cost rule is
// separate (CostRule), and so is the activation fold (FoldableActivation).
OrtStatus* Check(const OrtApi& api, const OrtNode* node, Spec* spec, std::string* why);

// The activation node that consumes the conv's output when it can be folded into the claim, else
// null: a Relu, or a LeakyRelu with a float alpha, in the ONNX domain, the only consumer of the
// conv's output, which is not a graph output. Sets spec->act and spec->alpha.
OrtStatus* FoldableActivation(const OrtApi& api, const OrtNode* conv, Spec* spec,
                              const OrtNode** act_node);

// The cost rule: true when the NPU call plus the fp32 <-> fp16 conversions is estimated at no more
// than 0.75x ONNX Runtime's CPU kernel for the same node. *note gets the two estimates.
bool CostRule(const Spec& s, std::string* note);

// True when a fused subgraph is a claimed Conv (with or without its folded activation), which is how
// Compile tells this claim from the others.
bool IsConvGraph(const OrtApi& api, const OrtGraph* graph);

// Compiles one claimed Conv: reads its weight and bias, packs the weight on `shared`, and hands back
// the node's compute info.
OrtStatus* Compile(const OrtApi& api, const OrtLogger& logger, std::shared_ptr<rocket_convt::Shared> shared,
                   const OrtGraph* graph, const OrtNode* fused_node, OrtNodeComputeInfo** info);

}  // namespace rocket_convf

#endif  // ORT_ROCKET_CONVF_H
