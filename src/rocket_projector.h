// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_projector -- the RF-DETR CSP projector (single-scale feature fusion) on the rocket
// NPU, assembled from the validated librocketnpu primitives. This is the stage between the
// DINOv2 backbone (rocket_backbone.h) and the deformable decoder: it fuses the 4 multi-scale
// feature taps into the single [1,256,grid,grid] map the transformer samples. The same chain
// is graded bit-faithful to the ORT CPU EP (cos(npu,ort) = 0.999997). Spatial geometry
// (grid, np) comes from the shared
// rocket_backbone_geom; the channel dims are fixed.
//
// The block (/backbone/backbone.0/projector/stages.0):
//   concat(4 taps [576,384]) -> [576,1536]
//   cv1: matmul 1536->256 -> chLN -> SiLU -> split -> a[576,128], b[576,128]
//   m.0/m.1/m.2 bottlenecks on b (chained): conv3x3 128 -> chLN -> SiLU, twice each
//   concat(a,b,m0,m1,m2) [576,640] -> cv2: matmul 640->256 -> chLN -> SiLU
//   final chLN (stages.0.1) -> [576,256]
//
// State is TOKEN-major [576,C] (chLN/SiLU/1x1 natural there); the 3x3 convs transpose to
// the [C,24,24] cube and back. 1x1 convs are matmuls; 3x3 convs are rocket_conv2d_fp16;
// chLN is rocket_layernorm_fp16; SiLU is the DPU LUT. All weights fp16 in native layout
// (conv weights [OC,IC,KH,KW]; the 1x1 weights [OC,IC]); marshaling is the EP's job.

#ifndef ORT_ROCKET_PROJECTOR_H
#define ORT_ROCKET_PROJECTOR_H

#include "rocket_backbone.h"   // rocket_backbone_geom (shared spatial geometry: grid, np, ...)

#ifdef __cplusplus
extern "C" {
#endif

// Projector channel dims. The RF-DETR CSP projector is tied to the DINOv2 ViT-S backbone, so its
// tap channel dim is fixed at the ViT-S d=384 (it must equal the backbone geom->d it consumes; a
// non-ViT-S backbone would pair with a different neck, not this one). The spatial dims (grid, np)
// are per-variant and read from the shared rocket_backbone_geom.
enum {
  RPJ_DIN = 384,                       // tap channels = DINOv2 ViT-S d (== backbone geom->d)
  RPJ_NTAP = 4,
  RPJ_CIN = RPJ_NTAP * RPJ_DIN,        // 1536 = 4 concatenated taps
  RPJ_CO = 256,                        // cv1/cv2/final output channels
  RPJ_CM = 128,                        // bottleneck channels
  RPJ_CCAT = 5 * RPJ_CM                // 640 = [a,b,m0,m1,m2]
};

// The projector's fp16 weights (owned by the caller / the EP's compiled state). `geom` must be
// set (the same rocket_backbone_geom as the backbone weights) before any run.
typedef struct {
  const rocket_backbone_geom *geom;    // shared variant geometry (grid, np)
  const _Float16 *cv1_w;               // [CO*CIN]  1x1 conv as [OC,IC]
  const _Float16 *cv1_g, *cv1_b;       // [CO]
  const _Float16 *cv2_w;               // [CO*CCAT]
  const _Float16 *cv2_g, *cv2_b;       // [CO]
  const _Float16 *fin_g, *fin_b;       // [CO]      final LayerNorm (no activation)
  const _Float16 *m_w[3][2];           // [CM*CM*9] 3x3 conv [OC,IC,KH,KW]
  const _Float16 *m_g[3][2], *m_b[3][2]; // [CM]
  float eps;                           // 1e-6
} rocket_projector_weights;

// De-window a backbone feature-norm output into a spatial projector tap. `feat` is the
// [ntok*RPJ_DIN] fp16 backbone output (nw windows x tw tokens, cls at each window's row 0);
// `tap` receives [np*RPJ_DIN] fp16 spatial token-major (position s = ph*grid+pw), the inverse
// of the stem window-partition. Pure host layout, no hardware.
void rocket_projector_dewindow(const rocket_backbone_geom *g, const _Float16 *feat, _Float16 *tap);

// Run the projector on the NPU. `taps[i]` (i in 0..3) are the 4 de-windowed spatial taps
// [RPJ_NP*RPJ_DIN] fp16; `y_out` receives [RPJ_NP*RPJ_CO] fp16 token-major (position-major).
// fd >= 0 runs on the NPU. Returns 0, <0 on error. Scratch is malloc'd internally.
//
// One-shot path (re-packs the 1x1 weights and re-allocates conv BOs per call); the resident
// context below is the latency path and this stays the ROCKET_ORT_RESIDENT=0 fallback.
int rocket_projector_run(int fd, const rocket_projector_weights *w,
                         const _Float16 *const taps[RPJ_NTAP], _Float16 *y_out);

// ---- resident (prepacked, resident-BO conv) projector -- the latency path -------------
// Packs the two 1x1 conv matmuls (cv1, cv2) into resident NPU BOs once at create; the six
// 3x3 bottleneck convs run through a resident-BO conv context (no per-call BO alloc/free).
// chLN, SiLU, and the final chLN run threaded on the host. The weights struct `w` must
// outlive the context. Bit-faithful to rocket_projector_run within fp16. Returns NULL on
// create failure.
typedef struct rocket_projector_ctx rocket_projector_ctx;

rocket_projector_ctx *rocket_projector_ctx_create(const rocket_projector_weights *w, int nthreads);
void                  rocket_projector_ctx_free(rocket_projector_ctx *c);
int  rocket_projector_run_ctx(rocket_projector_ctx *c,
                              const _Float16 *const taps[RPJ_NTAP], _Float16 *y_out);

#ifdef __cplusplus
}
#endif
#endif  // ORT_ROCKET_PROJECTOR_H
