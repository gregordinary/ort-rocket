// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_backbone -- the NPU chain behind the ort-rocket EP. See rocket_backbone.h.
// The im2col patch-embed, stem glue (cls-per-window prepend + position add + window
// partition), the windowed/global encoder chain, and the 4 feature-norm taps are the
// validated NPU datapath. Geometry
// -- resolution, patch size, window grid, token/layer counts, global/tap masks, AND the
// transformer dims (d, nhead, dhead, dff) -- is read from the rocket_backbone_geom the EP
// selects, so this datapath is not pinned to ViT-S. The RF-DETR checkpoints are all ViT-S.

#define _POSIX_C_SOURCE 199309L   // clock_gettime / CLOCK_MONOTONIC (ROCKET_ORT_PROF)
#include "rocket_backbone.h"

#include <stdlib.h>
#include <string.h>

#include "rocket_npu.h"
#include "rocket_matmul.h"
#include "rocket_encoder.h"
#include "rocket_norm.h"
#include "rocket_attn.h"
#include "rocket_ort_host.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

// ---- variant geometry table --------------------------------------------------------------
// Derived fields precomputed: nw=win*win, tw=(grid/win)^2+1, np=grid*grid, pk=IC*patch*patch,
// ntok=nw*tw, pos=np+1. global[]/tap[] are indexed by encoder layer.
static const rocket_backbone_geom RBB_GEOM_NANO = {
  .img = 384, .patch = 16, .grid = 24, .win = 2,
  .nw = 4, .tw = 145, .np = 576, .pk = 768, .pk_pad = 768, .ntok = 580, .pos = 577,
  .d = 384, .nhead = 6, .dhead = 64, .dff = 1536, .nlayer = 12,   // DINOv2 ViT-S
  .global = {0,0,0,1,0,0,1,0,0,1,0,0},   // global attention on layers 3,6,9
  .tap    = {0,0,1,0,0,1,0,0,1,0,0,1},   // feature taps on layers 2,5,8,11
};
static const rocket_backbone_geom RBB_GEOM_BASE = {
  .img = 560, .patch = 14, .grid = 40, .win = 4,
  .nw = 16, .tw = 101, .np = 1600, .pk = 588, .pk_pad = 608, .ntok = 1616, .pos = 1601,
  .d = 384, .nhead = 6, .dhead = 64, .dff = 1536, .nlayer = 11,   // DINOv2 ViT-S
  .global = {0,0,1,0,0,1,0,0,1,0,0},     // global attention on layers 2,5,8
  .tap    = {0,1,0,0,1,0,0,1,0,0,1},     // feature taps on layers 1,4,7,10
};

const rocket_backbone_geom *rocket_backbone_geom_select(int patch_size) {
  if (patch_size == 16) return &RBB_GEOM_NANO;
  if (patch_size == 14) return &RBB_GEOM_BASE;
  return NULL;
}

int rocket_backbone_geom_check(const rocket_backbone_geom *g, const char **why) {
  if (!g)                { if (why) *why = "null geometry"; return -1; }
  // M = ntok (encoder GEMMs) and M = np (projector GEMMs) both drive the prepacked matmul, whose
  // height must be a multiple of 4 -- M%4 != 0 returns garbage at rc=0.
  if (g->ntok % 4 != 0)  { if (why) *why = "ntok is not a multiple of 4 (matmul M%4 constraint)"; return -2; }
  if (g->np   % 4 != 0)  { if (why) *why = "np is not a multiple of 4 (matmul M%4 constraint)";   return -3; }
  // The int8 projection output width N must be %32; the transformer dims must be self-consistent
  // and fit the compile-time stem scratch. All hold for ViT-S; the guard trips only on a
  // mis-derived variant table entry (which would otherwise mis-compute silently).
  if (g->d    % 32 != 0) { if (why) *why = "d is not a multiple of 32 (int8 projection N%32)";    return -4; }
  if (g->dff  % 32 != 0) { if (why) *why = "dff is not a multiple of 32 (int8 projection N%32)";  return -5; }
  if (g->nhead <= 0 || g->dhead * g->nhead != g->d)
                         { if (why) *why = "dhead*nhead != d (inconsistent head split)";          return -6; }
  if (g->d > RBB_D_MAX)  { if (why) *why = "d exceeds RBB_D_MAX (stem scratch bound)";            return -7; }
  if (g->nlayer > RBB_NLAYER_MAX)
                         { if (why) *why = "nlayer exceeds RBB_NLAYER_MAX (per-layer array bound)"; return -8; }
  return 0;
}

// im2col: patches[p][ic*P*P + r*P + c] = img[ic][ph*P+r][pw*P+c], p = ph*GRID+pw. Rows are
// laid out at stride pk_pad (>= pk); the [pk, pk_pad) tail must be pre-zeroed by the caller
// (the matmul K is pk_pad, a multiple of 32; the zero tail leaves the product unchanged).
static void im2col(const rocket_backbone_geom *g, const _Float16 *img, _Float16 *patches) {
  const int P = g->patch, GRID = g->grid, IMG = g->img, PKP = g->pk_pad;
  for (int ph = 0; ph < GRID; ph++)
    for (int pw = 0; pw < GRID; pw++) {
      int p = ph * GRID + pw;
      _Float16 *row = patches + (size_t)p * PKP;
      for (int ic = 0; ic < RBB_IC; ic++)
        for (int r = 0; r < P; r++)
          for (int c = 0; c < P; c++)
            row[ic * P * P + r * P + c] =
                img[((size_t)ic * IMG + (ph * P + r)) * IMG + (pw * P + c)];
    }
}

// Build the [NW*TW,D] entry buffer from pe[NP,D] (patch embeddings, pre-position), cls[D],
// pos[POS,D]. cls_pe = cls + pos[0]; patch token t gets pos[1+t]; window w=(wh,ww), within
// j=(ih,iw) maps to patch ph=wh*iwn+ih, pw=ww*iwn+iw, t=ph*GRID+pw. Row w*TW = cls_pe.
static void stem(const rocket_backbone_geom *g, const _Float16 *pe, const _Float16 *cls,
                 const _Float16 *pos, _Float16 *buf) {
  const int D = g->d;
  _Float16 cls_pe[RBB_D_MAX];   // D <= RBB_D_MAX is enforced by rocket_backbone_geom_check
  for (int h = 0; h < D; h++) cls_pe[h] = (_Float16)((float)cls[h] + (float)pos[h]);
  const int iwn = g->grid / g->win;
  for (int w = 0; w < g->nw; w++) {
    int wh = w / g->win, ww = w % g->win;
    _Float16 *wbase = buf + (size_t)w * g->tw * D;
    memcpy(wbase, cls_pe, (size_t)D * sizeof(_Float16));
    for (int j = 0; j < iwn * iwn; j++) {
      int ih = j / iwn, iw = j % iwn;
      int ph = wh * iwn + ih, pw = ww * iwn + iw;
      int t = ph * g->grid + pw;
      const _Float16 *src = pe + (size_t)t * D;
      const _Float16 *prow = pos + (size_t)(1 + t) * D;
      _Float16 *dst = wbase + (size_t)(1 + j) * D;
      for (int h = 0; h < D; h++) dst[h] = (_Float16)((float)src[h] + (float)prow[h]);
    }
  }
}

static int run_block(int fd, const rocket_backbone_geom *g, int is_global,
                     const rocket_backbone_layer *L, float eps,
                     const _Float16 *x, _Float16 *y) {
  if (is_global)
    return rocket_encoder_block_fp16(fd, g->ntok, g->d, g->nhead, g->dff, x,
        L->ln1_g, L->ln1_b, L->wq, L->bq, L->wk, L->bk, L->wv, L->bv, L->wo, L->bo,
        L->ln2_g, L->ln2_b, L->wf1, L->bf1, L->wf2, L->bf2, eps, y);
  for (int w = 0; w < g->nw; w++) {
    int rc = rocket_encoder_block_fp16(fd, g->tw, g->d, g->nhead, g->dff,
        x + (size_t)w * g->tw * g->d,
        L->ln1_g, L->ln1_b, L->wq, L->bq, L->wk, L->bk, L->wv, L->bv, L->wo, L->bo,
        L->ln2_g, L->ln2_b, L->wf1, L->bf1, L->wf2, L->bf2, eps,
        y + (size_t)w * g->tw * g->d);
    if (rc) return rc;
  }
  return 0;
}

int rocket_backbone_run(int fd, const rocket_backbone_weights *w,
                        const _Float16 *img_chw, _Float16 *const feats[RBB_NTAP]) {
  const rocket_backbone_geom *g = w->geom;
  const int D = g->d;
  int rc = 0;
  _Float16 *patches = malloc((size_t)g->np * g->pk_pad * sizeof(_Float16));
  _Float16 *pe = malloc((size_t)g->np * D * sizeof(_Float16));
  _Float16 *buf = malloc((size_t)g->ntok * D * sizeof(_Float16));
  _Float16 *tmp = malloc((size_t)g->ntok * D * sizeof(_Float16));
  if (!patches || !pe || !buf || !tmp) { rc = -1; goto out; }

  // patch embed: patches[NP,pk_pad] . patch_w[D,pk_pad]^T -> pe[NP,D], + per-channel bias.
  // Zero the buffer so the [pk, pk_pad) column tail is 0 before im2col fills the real columns.
  memset(patches, 0, (size_t)g->np * g->pk_pad * sizeof(_Float16));
  im2col(g, img_chw, patches);
  rc = rocket_matmul_fp16(fd, g->np, g->pk_pad, D, patches, w->patch_w, pe);
  if (rc) goto out;
  for (int p = 0; p < g->np; p++)
    for (int h = 0; h < D; h++) {
      size_t i = (size_t)p * D + h;
      pe[i] = (_Float16)((float)pe[i] + (float)w->patch_b[h]);
    }

  // stem glue -> flat [NTOK,D] entry buffer.
  stem(g, pe, w->cls, w->pos, buf);

  // encoder chain with feature taps.
  for (int l = 0; l < g->nlayer; l++) {
    rc = run_block(fd, g, g->global[l], &w->layer[l], w->eps, buf, tmp);
    if (rc) goto out;
    _Float16 *s = buf; buf = tmp; tmp = s;   // block wrote into tmp; make it current
    if (g->tap[l]) {
      int fi = 0; for (int k = 0; k <= l; k++) fi += g->tap[k]; fi -= 1;
      rc = rocket_layernorm_fp16(fd, g->ntok, D, buf, w->fn_g, w->fn_b, w->eps, feats[fi]);
      if (rc) goto out;
    }
  }

out:
  free(patches); free(pe); free(buf); free(tmp);
  return rc;
}

// ===========================================================================================
// Resident (prepacked, multicore) backbone. See rocket_backbone.h. The per-token ops (patch
// proj, Q/K/V/O proj, fc1/fc2) run on the whole flat [NTOK,D] buffer via prepacked matmuls --
// the windowing only changes WHICH tokens attend, not the projections -- so only the
// attention is window-aware. LayerNorm/bias/residual/GELU run threaded on the host.
// ===========================================================================================

// The head dim (geom->dhead) and the fused Q|K|V projection width (3*geom->d) are read from the
// selected geometry per call, not compile-time constants.

struct rocket_backbone_ctx {
  const rocket_backbone_weights *w;
  const rocket_backbone_geom *g;
  rocket_ctx     *mm;                  // prepacked static weights, multicore
  rocket_fa_ctx  *fa;                  // multicore MHA (NULL => single-fd fallback)
  int             aux_fd;              // single-fd flash-attn fallback
  int             ht;                  // host worker threads (LN/GELU/bias/residual)
  rocket_weights *w_patch;
  // Q|K|V are fused into one [3D,D] prepacked weight per layer -- one matmul dispatch instead
  // of three, bit-exact (the projection is per-token and column-independent). bqkv is the
  // concatenated [bq|bk|bv] bias, added in one pass over the [NTOK,3D] output.
  rocket_weights *wqkv[RBB_NLAYER_MAX], *wo[RBB_NLAYER_MAX];
  rocket_weights *wf1[RBB_NLAYER_MAX], *wf2[RBB_NLAYER_MAX];
  _Float16       *bqkv[RBB_NLAYER_MAX];    // owned [3D] concatenated bias
  // Per-image scratch, owned by the ctx and allocated ONCE (not malloc'd per image; the ctx
  // is single-Run like the rest of librocketnpu's ctx APIs, so one scratch set is safe).
  _Float16 *patches, *pe, *buf, *ln, *q, *cc, *ff, *qkv, *Qh, *Kh, *Vh, *Oh;
  // Optional native-int8 path (w->i8 != NULL). The 5 per-token GEMMs run W8A8 on the NPU via
  // the group-wise resident int8 matmul (group=K, so per-row activation x per-out-channel weight
  // scale, fp32 Cf out narrowed to fp16). Weights packed once here; b_scales read from w->i8.
  const rocket_backbone_int8 *i8;          // == w->i8 (borrowed)
  rocket_i8_ctx  *i8ctx;
  rocket_i8_weights *w_patch8, *wq8[RBB_NLAYER_MAX], *wk8[RBB_NLAYER_MAX], *wv8[RBB_NLAYER_MAX];
  rocket_i8_weights *wo8[RBB_NLAYER_MAX], *wf18[RBB_NLAYER_MAX], *wf28[RBB_NLAYER_MAX];
  int8_t   *A8;                            // [ntok*DFF] int8 activation scratch
  float    *aScale;                        // [ntok] per-row activation scale
  float    *Cf;                            // [ntok*DFF] fp32 dequantized matmul output scratch
  int       i8mask;                        // which GEMMs run int8: 1=patch 2=qkv 4=o 8=fc1 16=fc2
};

static double bb_now_ms(void);   // defined below (ROCKET_ORT_PROF timing)

// Attention host relayout, threaded over the ng independent (window,head) groups (g =
// wi*NHEAD + h). The QKV bias is folded into the gather -- one fused pass instead of a
// separate add_bias over the whole [NTOK,3D] projection -- bit-exact (the same fp16 add).
// Qd/Kd = [T][dh]; Vd = [dh][T] (the AV B-operand, V transposed).
typedef struct {
  const _Float16 *qkv, *bqkv;     // bqkv NULL => plain gather (no bias fold)
  _Float16 *Qh, *Kh, *Vh;
  int T, tw;                      // T = tokens attending together; tw = per-window stride
  int nhead, dh, d;               // attention geometry (dh = head dim, d = model dim, row stride 3*d)
} bb_gather_arg;
static void bb_gather_groups(void *a, int lo, int hi) {
  bb_gather_arg *s = a;
  const int T = s->T, NH = s->nhead, DH = s->dh, D = s->d, D3 = 3 * D;
  for (int gi = lo; gi < hi; gi++) {
    int wi = gi / NH, h = gi % NH, off = h * DH, rbase = wi * s->tw;
    _Float16 *Qd = s->Qh + (size_t)gi * T * DH;
    _Float16 *Kd = s->Kh + (size_t)gi * T * DH;
    _Float16 *Vd = s->Vh + (size_t)gi * DH * T;
    const _Float16 *bq = s->bqkv ? s->bqkv + off : NULL;
    const _Float16 *bk = s->bqkv ? s->bqkv + D + off : NULL;
    const _Float16 *bv = s->bqkv ? s->bqkv + 2 * D + off : NULL;
    for (int t = 0; t < T; t++) {
      const _Float16 *row = s->qkv + (size_t)(rbase + t) * D3;
      const _Float16 *qr = row + off, *kr = row + D + off, *vr = row + 2 * D + off;
      _Float16 *qd = Qd + (size_t)t * DH, *kd = Kd + (size_t)t * DH;
      if (s->bqkv)
        for (int cx = 0; cx < DH; cx++) {
          qd[cx] = (_Float16)((float)qr[cx] + (float)bq[cx]);
          kd[cx] = (_Float16)((float)kr[cx] + (float)bk[cx]);
          Vd[(size_t)cx * T + t] = (_Float16)((float)vr[cx] + (float)bv[cx]);
        }
      else {
        memcpy(qd, qr, DH * sizeof(_Float16));
        memcpy(kd, kr, DH * sizeof(_Float16));
        for (int cx = 0; cx < DH; cx++) Vd[(size_t)cx * T + t] = vr[cx];
      }
    }
  }
}

typedef struct { _Float16 *cc; const _Float16 *Oh; int T, tw; int nhead, dh, d; } bb_scatter_arg;
static void bb_scatter_groups(void *a, int lo, int hi) {
  bb_scatter_arg *s = a;
  const int T = s->T, NH = s->nhead, DH = s->dh, D = s->d;
  for (int gi = lo; gi < hi; gi++) {
    int wi = gi / NH, h = gi % NH, off = h * DH, rbase = wi * s->tw;
    const _Float16 *Od = s->Oh + (size_t)gi * T * DH;
    for (int t = 0; t < T; t++)
      memcpy(s->cc + (size_t)(rbase + t) * D + off, Od + (size_t)t * DH,
             DH * sizeof(_Float16));
  }
}

// Head/window-major relayout + flash-attention + scatter back. For a global layer the whole
// NTOK-token sequence is nh independent heads; for a windowed layer each of the NW windows'
// TW tokens is a separate attention problem, so the groups are the NW*NHEAD (window,head)
// pairs (TW tokens each) fanned across the cores in one call. `qkv` is the fused [NTOK,3D]
// projection with bias NOT yet applied; `bqkv` [3D] is folded into the gather. cc receives
// [NTOK,D]. Qh/Kh/Vh/Oh are ctx scratch. The gather and scatter thread over the ng groups.
// scale = 1/sqrt(dh) applied in FA. tsplit (or NULL): [0] += host relayout ms, [1] += NPU
// flash-attn ms.
static int bb_attn(rocket_backbone_ctx *c, int is_global,
                   const _Float16 *qkv, const _Float16 *bqkv, _Float16 *cc,
                   _Float16 *Qh, _Float16 *Kh, _Float16 *Vh, _Float16 *Oh, double *tsplit) {
  const rocket_backbone_geom *g = c->g;
  const int DH = g->dhead, NH = g->nhead;
  const float scale = 1.f / sqrtf((float)DH);
  const int T  = is_global ? g->ntok : g->tw;
  const int nw = is_global ? 1 : g->nw;
  const int ng = nw * NH;   // attention groups
  double t0 = 0;

  bb_gather_arg ga = { qkv, bqkv, Qh, Kh, Vh, T, g->tw, NH, DH, g->d };
  if (tsplit) t0 = bb_now_ms();
  rocket_host_parallel_for(ng, c->ht, bb_gather_groups, &ga);
  if (tsplit) tsplit[0] += bb_now_ms() - t0;

  int rc;
  if (tsplit) t0 = bb_now_ms();
  if (c->fa) rc = rocket_flash_attn_fp16_ctx(c->fa, T, T, DH, DH, ng, ng,
                                             scale, 0.f, Qh, Kh, Vh, NULL, Oh);
  else       rc = rocket_flash_attn_fp16(c->aux_fd, T, T, DH, DH, ng, ng,
                                         scale, 0.f, Qh, Kh, Vh, NULL, Oh);
  if (tsplit) tsplit[1] += bb_now_ms() - t0;
  if (rc) return rc;

  bb_scatter_arg sa = { cc, Oh, T, g->tw, NH, DH, g->d };
  if (tsplit) t0 = bb_now_ms();
  rocket_host_parallel_for(ng, c->ht, bb_scatter_groups, &sa);
  if (tsplit) tsplit[0] += bb_now_ms() - t0;
  return 0;
}

// Native-int8 GEMM (group=K, so per-row activation x per-out-channel weight scale -> fp32
// Cf[M,N]). `p` carries the weight scale s[N], the activation quant params, and rowsum for the
// zero-point fold; `w` is the packed weight. Result is narrowed into `out` at row stride
// `out_stride` (== N, or 3D for a QKV slice). Two activation-quant modes:
//   static asymmetric (p->a_scale > 0): A_int8 = clamp(round(x/a_scale) + a_zp), a_scale[m] const;
//     the matmul dequant uses the RAW A_int8, so subtract the zero-point term
//     a_scale*s[n]*a_zp*rowsum[n] from Cf afterwards. Matches the QDQ CPU path.
//   dynamic per-row symmetric (p->a_scale <= 0): a_scale[m] = max|x[m,:]|/127, a_zp = 0.
static int i8_gemm(rocket_backbone_ctx *c, int M, int K, int N, const _Float16 *in,
                   const rocket_i8_proj *p, rocket_i8_weights *w, _Float16 *out, int out_stride) {
  const int stat = (p->a_scale > 0.f);
  const float as = p->a_scale;
  const int   az = p->a_zp;
  const float inv = stat ? 1.f / as : 0.f;
  for (int m = 0; m < M; m++) {
    const _Float16 *row = in + (size_t)m * K;
    int8_t *dst = c->A8 + (size_t)m * K;
    if (stat) {
      c->aScale[m] = as;
      for (int k = 0; k < K; k++) {
        long qv = lrintf((float)row[k] * inv) + az;
        dst[k] = (int8_t)(qv < -128 ? -128 : (qv > 127 ? 127 : qv));
      }
    } else {
      float amax = 0.f;
      for (int k = 0; k < K; k++) { float a = fabsf((float)row[k]); if (a > amax) amax = a; }
      float s = (amax > 0.f) ? amax / 127.f : 1.f;
      c->aScale[m] = s;
      float dinv = 1.f / s;
      for (int k = 0; k < K; k++) {
        long qv = lrintf((float)row[k] * dinv);
        dst[k] = (int8_t)(qv < -127 ? -127 : (qv > 127 ? 127 : qv));
      }
    }
  }
  int rc = rocket_matmul_int8_prepacked_gw(c->i8ctx, M, K, N, c->A8, c->aScale, p->s, c->Cf, w);
  if (rc) return rc;
  if (stat && az != 0 && p->rowsum) {
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++)
        out[(size_t)m * out_stride + n] =
            (_Float16)(c->Cf[(size_t)m * N + n] - as * p->s[n] * (float)az * (float)p->rowsum[n]);
  } else {
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++)
        out[(size_t)m * out_stride + n] = (_Float16)c->Cf[(size_t)m * N + n];
  }
  return 0;
}

rocket_backbone_ctx *rocket_backbone_ctx_create(const rocket_backbone_weights *w, int nthreads) {
  if (!w || !w->geom) return NULL;
  rocket_backbone_ctx *c = calloc(1, sizeof(*c));
  if (!c) return NULL;
  c->w = w;
  c->g = w->geom;
  const rocket_backbone_geom *g = c->g;
  const int D = g->d, DFF = g->dff, D3 = 3 * g->d;   // fused Q|K|V width = 3*d
  c->aux_fd = -1;
  c->ht = (nthreads <= 1) ? 1 : 4;   // host LN/GELU across the 4 A76 big cores
  { const char *e = getenv("ROCKET_ORT_HT");   // tuning override for the host-op fan-out
    if (e) { int v = atoi(e); if (v >= 1 && v <= 8) c->ht = v; } }

  c->mm = rocket_ctx_create(nthreads);
  c->aux_fd = rocket_open();
  if (!c->mm || c->aux_fd < 0) goto fail;

  c->w_patch = rocket_weights_pack(c->mm, g->np, g->pk_pad, D, w->patch_w);
  if (!c->w_patch) goto fail;
  _Float16 *qkv_tmp = malloc((size_t)D3 * D * sizeof(_Float16));
  if (!qkv_tmp) goto fail;
  for (int l = 0; l < g->nlayer; l++) {
    const rocket_backbone_layer *L = &w->layer[l];
    // fuse [Wq;Wk;Wv] row-wise into one [3D,D] weight; concat [bq|bk|bv] into bqkv[3D].
    memcpy(qkv_tmp,                     L->wq, (size_t)D * D * sizeof(_Float16));
    memcpy(qkv_tmp + (size_t)D * D,     L->wk, (size_t)D * D * sizeof(_Float16));
    memcpy(qkv_tmp + (size_t)2 * D * D, L->wv, (size_t)D * D * sizeof(_Float16));
    c->wqkv[l] = rocket_weights_pack(c->mm, g->ntok, D, D3, qkv_tmp);
    c->bqkv[l] = malloc((size_t)D3 * sizeof(_Float16));
    if (c->bqkv[l]) {
      memcpy(c->bqkv[l],         L->bq, D * sizeof(_Float16));
      memcpy(c->bqkv[l] + D,     L->bk, D * sizeof(_Float16));
      memcpy(c->bqkv[l] + 2 * D, L->bv, D * sizeof(_Float16));
    }
    c->wo[l]  = rocket_weights_pack(c->mm, g->ntok, D,   D,   L->wo);
    c->wf1[l] = rocket_weights_pack(c->mm, g->ntok, D,   DFF, L->wf1);
    c->wf2[l] = rocket_weights_pack(c->mm, g->ntok, DFF, D,   L->wf2);
    if (!c->wqkv[l] || !c->bqkv[l] || !c->wo[l] || !c->wf1[l] || !c->wf2[l]) { free(qkv_tmp); goto fail; }
  }
  free(qkv_tmp);

  // Optional native-int8 projections (ROCKET_ORT_INT8). Pack the 5 GEMM weights + patch into
  // resident int8 BOs (group = K, one quant group spanning K -> per-out-channel weight scale);
  // the fp16 GEMM weights above stay packed but unused in this mode (attention/bias/norms remain
  // fp16). b_scales are borrowed from w->i8 and read per call.
  c->i8 = w->i8;
  c->i8mask = 0x1F;   // all 5 GEMMs int8 by default; ROCKET_ORT_I8_MASK isolates a subset (debug)
  { const char *e = getenv("ROCKET_ORT_I8_MASK"); if (e) c->i8mask = (int)strtol(e, NULL, 0); }
  if (c->i8) {
    c->i8ctx = rocket_i8_ctx_create(nthreads);
    if (!c->i8ctx) goto fail;
    c->w_patch8 = rocket_i8_weights_pack_gw(c->i8ctx, g->np, g->pk_pad, D, c->i8->patch.w, g->pk_pad);
    if (!c->w_patch8) goto fail;
    for (int l = 0; l < g->nlayer; l++) {
      c->wq8[l]  = rocket_i8_weights_pack_gw(c->i8ctx, g->ntok, D,   D,   c->i8->q[l].w,  D);
      c->wk8[l]  = rocket_i8_weights_pack_gw(c->i8ctx, g->ntok, D,   D,   c->i8->k[l].w,  D);
      c->wv8[l]  = rocket_i8_weights_pack_gw(c->i8ctx, g->ntok, D,   D,   c->i8->v[l].w,  D);
      c->wo8[l]  = rocket_i8_weights_pack_gw(c->i8ctx, g->ntok, D,   D,   c->i8->o[l].w,  D);
      c->wf18[l] = rocket_i8_weights_pack_gw(c->i8ctx, g->ntok, D,   DFF, c->i8->f1[l].w, D);
      c->wf28[l] = rocket_i8_weights_pack_gw(c->i8ctx, g->ntok, DFF, D,   c->i8->f2[l].w, DFF);
      if (!c->wq8[l] || !c->wk8[l] || !c->wv8[l] || !c->wo8[l] || !c->wf18[l] || !c->wf28[l]) goto fail;
    }
    // A8 holds A_int8[M*K] (max K = DFF at fc2); Cf holds one GEMM's fp32 out [M*N] (max N = DFF
    // at fc1, which exceeds 3D). Size both to ntok*DFF, the max over the 5 GEMMs.
    c->A8     = malloc((size_t)g->ntok * DFF);
    c->aScale = malloc((size_t)g->ntok * sizeof(float));
    c->Cf     = malloc((size_t)g->ntok * DFF * sizeof(float));
    if (!c->A8 || !c->aScale || !c->Cf) goto fail;
  }

  // Per-image scratch, allocated once (the ctx is single-Run).
  const size_t Nd = (size_t)g->ntok * D;
  c->patches = malloc((size_t)g->np * g->pk_pad * sizeof(_Float16));
  c->pe  = malloc((size_t)g->np * D * sizeof(_Float16));
  c->buf = malloc(Nd * sizeof(_Float16));
  c->ln  = malloc(Nd * sizeof(_Float16));
  c->q   = malloc(Nd * sizeof(_Float16));
  c->cc  = malloc(Nd * sizeof(_Float16));
  c->ff  = malloc((size_t)g->ntok * DFF * sizeof(_Float16));
  c->qkv = malloc((size_t)g->ntok * D3 * sizeof(_Float16));
  c->Qh  = malloc(Nd * sizeof(_Float16));
  c->Kh  = malloc(Nd * sizeof(_Float16));
  c->Vh  = malloc(Nd * sizeof(_Float16));
  c->Oh  = malloc(Nd * sizeof(_Float16));
  if (!c->patches || !c->pe || !c->buf || !c->ln || !c->q || !c->cc || !c->ff ||
      !c->qkv || !c->Qh || !c->Kh || !c->Vh || !c->Oh) goto fail;

  // multicore MHA via the flash-attention fan-out (single-fd fallback if it can't create).
  if (nthreads > 1) c->fa = rocket_fa_ctx_create(nthreads);
  return c;
fail:
  rocket_backbone_ctx_free(c);
  return NULL;
}

void rocket_backbone_ctx_free(rocket_backbone_ctx *c) {
  if (!c) return;
  if (c->mm) {
    if (c->w_patch) rocket_weights_free(c->mm, c->w_patch);
    for (int l = 0; l < RBB_NLAYER_MAX; l++) {
      if (c->wqkv[l]) rocket_weights_free(c->mm, c->wqkv[l]);
      if (c->wo[l])  rocket_weights_free(c->mm, c->wo[l]);
      if (c->wf1[l]) rocket_weights_free(c->mm, c->wf1[l]);
      if (c->wf2[l]) rocket_weights_free(c->mm, c->wf2[l]);
    }
  }
  for (int l = 0; l < RBB_NLAYER_MAX; l++) free(c->bqkv[l]);
  if (c->i8ctx) {
    if (c->w_patch8) rocket_i8_weights_free(c->i8ctx, c->w_patch8);
    for (int l = 0; l < RBB_NLAYER_MAX; l++) {
      if (c->wq8[l])  rocket_i8_weights_free(c->i8ctx, c->wq8[l]);
      if (c->wk8[l])  rocket_i8_weights_free(c->i8ctx, c->wk8[l]);
      if (c->wv8[l])  rocket_i8_weights_free(c->i8ctx, c->wv8[l]);
      if (c->wo8[l])  rocket_i8_weights_free(c->i8ctx, c->wo8[l]);
      if (c->wf18[l]) rocket_i8_weights_free(c->i8ctx, c->wf18[l]);
      if (c->wf28[l]) rocket_i8_weights_free(c->i8ctx, c->wf28[l]);
    }
    rocket_i8_ctx_free(c->i8ctx);
  }
  free(c->A8); free(c->aScale); free(c->Cf);
  free(c->patches); free(c->pe); free(c->buf); free(c->ln); free(c->q);
  free(c->cc); free(c->ff); free(c->qkv); free(c->Qh); free(c->Kh); free(c->Vh); free(c->Oh);
  if (c->fa) rocket_fa_ctx_free(c->fa);
  if (c->mm) rocket_ctx_free(c->mm);
  if (c->aux_fd >= 0) rocket_close(c->aux_fd);
  free(c);
}

// Optional per-phase timing breakdown (ROCKET_ORT_PROF=1). Accumulated per call, printed at
// the end -- run the bench with a few iters and read the warm ones.
static double bb_now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int rocket_backbone_run_ctx(rocket_backbone_ctx *c, const _Float16 *img_chw,
                            _Float16 *const feats[RBB_NTAP]) {
  if (!c || !img_chw) return -1;
  rocket_host_pin_self();   // keep the compute thread's serial parts on an A76, not an A55
  const rocket_backbone_weights *w = c->w;
  const rocket_backbone_geom *g = c->g;
  const int D = g->d, DFF = g->dff, D3 = 3 * g->d;   // fused Q|K|V width = 3*d
  const float eps = w->eps;
  int rc = -2;
  int prof = getenv("ROCKET_ORT_PROF") != NULL;
  double tb[8] = {0}, t0 = 0;
#define TIC (t0 = bb_now_ms())
#define TOC(i) do { if (prof) tb[i] += bb_now_ms() - t0; } while (0)

  // Scratch is ctx-owned (allocated once at create), not malloc'd per image.
  _Float16 *patches = c->patches, *pe = c->pe, *buf = c->buf, *ln = c->ln,
           *q = c->q, *cc = c->cc, *ff = c->ff, *qkv = c->qkv,
           *Qh = c->Qh, *Kh = c->Kh, *Vh = c->Vh, *Oh = c->Oh;
  double att[2] = {0, 0};   // ROCKET_ORT_PROF split: [0] host relayout, [1] NPU flash-attn

  // patch embed: im2col -> prepacked matmul -> + per-channel bias. Zero the [pk, pk_pad) column
  // tail (ctx scratch is reused across images; im2col only writes the real pk columns).
  TIC;
  memset(patches, 0, (size_t)g->np * g->pk_pad * sizeof(_Float16));
  im2col(g, img_chw, patches);
  if (c->i8 && (c->i8mask & 1)) rc = i8_gemm(c, g->np, g->pk_pad, D, patches, &c->i8->patch, c->w_patch8, pe, D);
  else                          rc = rocket_matmul_fp16_prepacked(c->mm, g->np, g->pk_pad, D, patches, pe, c->w_patch);
  if (rc) goto out;
  for (int p = 0; p < g->np; p++)
    for (int h = 0; h < D; h++) {
      size_t i = (size_t)p * D + h;
      pe[i] = (_Float16)((float)pe[i] + (float)w->patch_b[h]);
    }
  stem(g, pe, w->cls, w->pos, buf);
  TOC(0);

  for (int l = 0; l < g->nlayer; l++) {
    const rocket_backbone_layer *L = &w->layer[l];

    // attention sublayer: buf += Wo * MHA(LN1(buf)). Q|K|V are one fused matmul; the QKV bias
    // is folded into the attention gather (no separate add_bias pass over [NTOK,3D]).
    TIC; rocket_host_layernorm(g->ntok, D, buf, L->ln1_g, L->ln1_b, eps, ln, c->ht); TOC(1);
    TIC;
    if (c->i8) {   // unfused Q/K/V int8 GEMMs writing into the [NTOK,3D] qkv slices
      if (c->i8mask & 2) {
        if ((rc = i8_gemm(c, g->ntok, D, D, ln, &c->i8->q[l], c->wq8[l], qkv,         D3))) goto out;
        if ((rc = i8_gemm(c, g->ntok, D, D, ln, &c->i8->k[l], c->wk8[l], qkv + D,     D3))) goto out;
        if ((rc = i8_gemm(c, g->ntok, D, D, ln, &c->i8->v[l], c->wv8[l], qkv + 2 * D, D3))) goto out;
      } else if ((rc = rocket_matmul_fp16_prepacked(c->mm, g->ntok, D, D3, ln, qkv, c->wqkv[l]))) goto out;
    } else if ((rc = rocket_matmul_fp16_prepacked(c->mm, g->ntok, D, D3, ln, qkv, c->wqkv[l]))) goto out;
    TOC(2);
    if ((rc = bb_attn(c, g->global[l], qkv, c->bqkv[l], cc, Qh, Kh, Vh, Oh, prof ? att : NULL))) goto out;
    TIC;
    if (c->i8 && (c->i8mask & 4)) { if ((rc = i8_gemm(c, g->ntok, D, D, cc, &c->i8->o[l], c->wo8[l], q, D))) goto out; }
    else if ((rc = rocket_matmul_fp16_prepacked(c->mm, g->ntok, D, D, cc, q, c->wo[l]))) goto out;
    TOC(2);
    // buf += (Wo*ctx + bo), one fused threaded pass (bo/Wo carry LayerScale lambda1).
    TIC; rocket_host_bias_residual(g->ntok, D, buf, q, L->bo, c->ht); TOC(5);

    // FFN sublayer: buf += Wf2 * GELU(Wf1 * LN2(buf)); the fc1 bias is fused into the GELU.
    TIC; rocket_host_layernorm(g->ntok, D, buf, L->ln2_g, L->ln2_b, eps, ln, c->ht); TOC(1);
    TIC;
    if (c->i8 && (c->i8mask & 8)) { if ((rc = i8_gemm(c, g->ntok, D, DFF, ln, &c->i8->f1[l], c->wf18[l], ff, DFF))) goto out; }
    else if ((rc = rocket_matmul_fp16_prepacked(c->mm, g->ntok, D, DFF, ln, ff, c->wf1[l]))) goto out;
    TOC(2);
    TIC; rocket_host_bias_gelu(g->ntok, DFF, ff, L->bf1, c->ht); TOC(4);
    TIC;
    if (c->i8 && (c->i8mask & 16)) { if ((rc = i8_gemm(c, g->ntok, DFF, D, ff, &c->i8->f2[l], c->wf28[l], cc, D))) goto out; }
    else if ((rc = rocket_matmul_fp16_prepacked(c->mm, g->ntok, DFF, D, ff, cc, c->wf2[l]))) goto out;
    TOC(2);
    // buf += (Wf2*gelu + bf2), one fused threaded pass (bf2/Wf2 carry LayerScale lambda2).
    TIC; rocket_host_bias_residual(g->ntok, D, buf, cc, L->bf2, c->ht); TOC(5);

    if (g->tap[l]) {
      int fi = 0; for (int kk = 0; kk <= l; kk++) fi += g->tap[kk]; fi -= 1;
      TIC; rocket_host_layernorm(g->ntok, D, buf, w->fn_g, w->fn_b, eps, feats[fi], c->ht); TOC(6);
    }
  }
  rc = 0;
  if (prof) {
    tb[3] = att[1];   // NPU flash-attn
    tb[7] = att[0];   // host attention relayout (gather + scatter)
    const char *nm[8] = {"patch+stem","host-LN","proj-matmul","attn-FA(NPU)","host-GELU",
                         "host-bias/res","feat-norm","attn-relayout"};
    double tot = 0; for (int i = 0; i < 8; i++) tot += tb[i];
    fprintf(stderr, "[bb prof] total %.1f ms:\n", tot);
    for (int i = 0; i < 8; i++)
      fprintf(stderr, "    %-14s %7.1f ms (%4.1f%%)\n", nm[i], tb[i], 100.0 * tb[i] / (tot + 1e-9));
  }
#undef TIC
#undef TOC

out:
  return rc;   // scratch is ctx-owned; nothing to free per call
}
