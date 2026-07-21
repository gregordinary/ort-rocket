// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_backbone -- the RF-DETR DINOv2 ViT-S backbone (stem + windowed/global encoder + 4
// multi-scale feature norms) on the rocket NPU, assembled from the validated librocketnpu
// primitives. This is the compute the ONNX Runtime EP's Compile/Compute wires up; the same
// chain is graded bit-faithful to the ORT CPU EP (cos(npu,ort) >= 0.99999 on every feature
// output).
//
// The backbone is a DINOv2 ViT-S: d=384, 6 heads, d_ff=1536, 4 multi-scale feature taps sharing
// one final LayerNorm, windowed attention interleaved with periodic global attention. Every
// dimension -- the transformer dims (d, nhead, dff, dhead) as well as the resolution, patch
// size, window grid, token/layer counts, and per-layer global/tap masks -- lives in a
// rocket_backbone_geom the EP selects, so a non-ViT-S backbone (a future larger-ViT target)
// sets its own without touching this datapath. The RF-DETR checkpoints are all ViT-S. All
// weights are fp16 in rocket layout (LayerScale pre-folded into Wo/bo and Wf2/bf2, projections
// transposed to [out,in]); marshaling is the EP's job.
//
//   variant  img  patch  grid  win  tw   ntok  layers  global      taps
//   nano     384   16     24    2   145   580    12    {3,6,9}     {2,5,8,11}
//   base     560   14     40    4   101  1616    11    {2,5,8}     {1,4,7,10}

#ifndef ORT_ROCKET_BACKBONE_H
#define ORT_ROCKET_BACKBONE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compile-time bounds (the sizing constants that must be fixed for the per-layer arrays and the
// stem stack scratch). The live per-variant transformer dims (d, nhead, dff, dhead) are in
// rocket_backbone_geom, not here.
enum {
  RBB_IC = 3,                 // RGB input channels (invariant)
  RBB_NTAP = 4,               // feature outputs the projector concatenates (RF-DETR CSP)
  RBB_NLAYER_MAX = 32,        // per-layer array bound (a ViT-H backbone has 32 encoder layers)
  RBB_D_MAX = 1536,           // max model dim; sizes the stem's cls_pe stack scratch.
                              // rocket_backbone_geom_check enforces geom->d <= this.
};

// Per-variant geometry the EP selects and the run functions read. The derived fields (nw, tw,
// np, pk, ntok, pos) are precomputed in the table entries.
typedef struct {
  int img;                    // input side px
  int patch;                  // patch conv kernel/stride px
  int grid;                   // patch grid per side (img/patch)
  int win;                    // windows per side
  int nw;                     // win*win windows
  int tw;                     // tokens per window = (grid/win)^2 + 1
  int np;                     // grid*grid patches
  int pk;                     // IC*patch*patch im2col contraction (real columns)
  int pk_pad;                 // pk rounded up to a multiple of 32 (matmul K%32; zero tail)
  int ntok;                   // nw*tw total tokens
  int pos;                    // position table rows = np + 1
  // Transformer dims. Per-variant so a non-ViT-S backbone sets its own; the RF-DETR checkpoints
  // are all ViT-S (d=384, nhead=6, dhead=64, dff=1536). Invariants (rocket_backbone_geom_check):
  // d and dff are %32 (int8 projection N), dhead*nhead == d, d <= RBB_D_MAX.
  int d;                      // model / embedding dim
  int nhead;                  // attention heads
  int dhead;                  // head dim = d / nhead (precomputed)
  int dff;                    // FFN hidden dim
  int nlayer;                 // encoder layers (<= RBB_NLAYER_MAX)
  int global[RBB_NLAYER_MAX]; // per-layer global-attention mask (1 = global, 0 = windowed)
  int tap[RBB_NLAYER_MAX];    // per-layer feature-tap mask (exactly RBB_NTAP entries set)
} rocket_backbone_geom;

// Select the geometry for a patch-conv kernel size (16 -> nano, 14 -> base); NULL if unknown.
const rocket_backbone_geom *rocket_backbone_geom_select(int patch_size);

// Check the matmul-alignment invariants a geometry must satisfy or the resident/one-shot matmul
// silently miscomputes: M = ntok and M = np must both be %4 (the prepacked matmul returns garbage
// at rc=0 for M%4 != 0), and the int8 projection N must be %32. Both shipping variants pass -- this
// guards a future/mis-derived variant table entry that would otherwise run and return wrong
// detections with no error. Cheap; call once at Compile. Returns 0 if valid, else a negative code,
// and sets *why (when non-NULL) to a static reason string.
int rocket_backbone_geom_check(const rocket_backbone_geom *g, const char **why);

// One encoder layer's fp16 weights, in rocket layout. Projection weights are [out,in];
// wo/bo and wf2/bf2 already have LayerScale (lambda1/lambda2) folded in.
typedef struct {
  const _Float16 *ln1_g, *ln1_b, *ln2_g, *ln2_b;   // [D]
  const _Float16 *wq, *wk, *wv, *wo;               // [D*D]
  const _Float16 *bq, *bk, *bv, *bo;               // [D]
  const _Float16 *wf1;                             // [DFF*D]
  const _Float16 *bf1;                             // [DFF]
  const _Float16 *wf2;                             // [D*DFF]
  const _Float16 *bf2;                             // [D]
} rocket_backbone_layer;

// Optional native-int8 encoder projections (opt-in ROCKET_ORT_INT8). When rocket_backbone_int8
// is attached, the resident ctx runs the 5 per-token GEMMs (patch-embed, Q/K/V, O, fc1, fc2) as
// W8A8 on the NPU: raw int8 weights in rocket [out,in] layout with a per-out-channel fp32 scale
// (symmetric, zero-point 0 -- QDQ weight quant is symmetric), and dynamic per-row int8
// activation quant per call. Everything else (attention, LN, GELU, bias/residual, projector,
// feature norm) stays fp16, identical to the fp16 resident path. This mode is slower than fp16
// on this stack (int8 matmul reads int32 back to the host -- there is no on-chip int32 accum in
// mainline rocket), so it is a faithfulness/experimentation mode, not the default.
// One int8 projection: weight codes w[out*in] (rocket [out,in]) + per-out-channel scale s[out].
// For the encoder GEMMs the activation is quantized with the model's STATIC per-tensor scale
// a_scale + zero-point a_zp (asymmetric, matching the QDQ CPU path -- dynamic per-row quant
// amplifies ViT channel-outliers and collapses accuracy); rowsum[out] = sum_in w (int32) folds
// the zero-point correction. a_scale <= 0 selects dynamic per-row symmetric quant (used for the
// patch-embed, whose input image has no outliers).
typedef struct {
  const int8_t *w; const float *s;        // weight codes + per-out-channel scale
  float a_scale; int a_zp;                // input-activation static scale + zero-point (0 => dynamic)
  const int32_t *rowsum;                  // [out] sum_in w, for the zero-point fold (may be NULL)
} rocket_i8_proj;
typedef struct {
  rocket_i8_proj patch;                            // [D * pk_pad], s[D]
  rocket_i8_proj q[RBB_NLAYER_MAX], k[RBB_NLAYER_MAX], v[RBB_NLAYER_MAX];   // [D*D],   s[D]
  rocket_i8_proj o[RBB_NLAYER_MAX];                // [D*D],    s[D]
  rocket_i8_proj f1[RBB_NLAYER_MAX];               // [DFF*D],  s[DFF]
  rocket_i8_proj f2[RBB_NLAYER_MAX];               // [D*DFF],  s[D]
} rocket_backbone_int8;

// The whole backbone's fp16 weights (owned by the caller / the EP's compiled state). `geom`
// must be set (via rocket_backbone_geom_select) before any run and must outlive the weights.
typedef struct {
  const rocket_backbone_geom *geom;   // selected variant geometry
  const _Float16 *patch_w;   // [D*PK]  im2col patch projection (conv reshaped [oc][ic*kh*kw])
  const _Float16 *patch_b;   // [D]
  const _Float16 *cls;       // [D]     cls token
  const _Float16 *pos;       // [POS*D] position table
  const _Float16 *fn_g;      // [D]     shared feature-norm gamma
  const _Float16 *fn_b;      // [D]     shared feature-norm beta
  rocket_backbone_layer layer[RBB_NLAYER_MAX];
  float eps;                 // 1e-6
  const rocket_backbone_int8 *i8;   // NULL => fp16 path; non-NULL => native-int8 projections
} rocket_backbone_weights;

// Run the backbone on one preprocessed image. `img_chw` is [IC*IMG*IMG] fp16. `feats[i]`
// (i in 0..3) each receive a [NTOK*D] fp16 feature-norm output for tap layer {2,5,8,11}.
// fd >= 0 runs on the NPU. Returns 0, <0 on error. Scratch is malloc'd internally.
//
// This is the one-shot path: every static weight is re-scattered and every BO re-allocated
// on each call. Correct and simple, but pack/dispatch-bound. The resident context below is
// the latency path; this stays as the ROCKET_ORT_RESIDENT=0 fallback and the A/B reference.
int rocket_backbone_run(int fd, const rocket_backbone_weights *w,
                        const _Float16 *img_chw, _Float16 *const feats[RBB_NTAP]);

// ---- resident (prepacked, multicore) backbone -- the latency path ---------------------
// The one-shot re-packs the static GEMMs (patch proj + per-layer Q/K/V/O/fc1/fc2, one set
// per encoder layer) on every image. The resident context packs them into NPU BOs ONCE at
// create and fans them
// across the 3 cores; per image only the activations are packed. Attention runs on the NPU
// via the flash-attention fan-out (windowed layers as nw*nh independent groups of TW tokens,
// global layers as nh groups of NTOK tokens); LayerNorm, bias/residual adds, and GELU run
// threaded on the host, where the data already lives de-tiled after the NPU readback. The
// weights struct `w` must outlive the context (its bias/gamma/beta pointers are read per
// call; only the GEMM matrices are copied into resident BOs). Bit-faithful to
// rocket_backbone_run within fp16 (the gate cross-checks cosine). Returns NULL on create
// failure (e.g. a weight pack hit the IOVA limit).
typedef struct rocket_backbone_ctx rocket_backbone_ctx;

rocket_backbone_ctx *rocket_backbone_ctx_create(const rocket_backbone_weights *w, int nthreads);
void                 rocket_backbone_ctx_free(rocket_backbone_ctx *c);
int  rocket_backbone_run_ctx(rocket_backbone_ctx *c, const _Float16 *img_chw,
                             _Float16 *const feats[RBB_NTAP]);

#ifdef __cplusplus
}
#endif
#endif  // ORT_ROCKET_BACKBONE_H
