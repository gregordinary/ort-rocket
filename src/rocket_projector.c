// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_projector -- the NPU chain behind the ort-rocket EP's projector stage. See
// rocket_projector.h. The validated projector NPU datapath (cos(npu,ort) = 0.999997): concat
// the 4 taps, cv1 (1x1 matmul) -> chLN -> SiLU
// -> split, 3 chained conv3x3 bottlenecks, concat, cv2 (1x1 matmul) -> chLN -> SiLU, a
// final channel LayerNorm. State is carried token-major [np,C]; the 3x3 convs transpose
// to the [C,grid,grid] cube. Spatial dims (grid, np) come from the shared geometry.

#include "rocket_projector.h"

#include <stdlib.h>
#include <string.h>

#include "rocket_matmul.h"
#include "rocket_conv.h"
#include "rocket_norm.h"
#include "rocket_activation.h"
#include "rocket_npu.h"
#include "rocket_ort_host.h"

// token[np,C] -> cube[C,np]  (cube[c*np+s] = tok[s*C+c]); np = grid*grid, row-major HxW.
static void tok_to_cube(const _Float16 *tok, _Float16 *cube, int C, int np) {
  for (int s = 0; s < np; s++)
    for (int c = 0; c < C; c++) cube[(size_t)c * np + s] = tok[(size_t)s * C + c];
}
static void cube_to_tok(const _Float16 *cube, _Float16 *tok, int C, int np) {
  for (int s = 0; s < np; s++)
    for (int c = 0; c < C; c++) tok[(size_t)s * C + c] = cube[(size_t)c * np + s];
}

// concat token-major columns: dst[np,total] row s = [src0[s], src1[s], ...].
static void concat_cols(_Float16 *dst, int total, const _Float16 *const *srcs,
                        const int *cols, int nsrc, int np) {
  for (int s = 0; s < np; s++) {
    _Float16 *row = dst + (size_t)s * total;
    int off = 0;
    for (int i = 0; i < nsrc; i++) {
      memcpy(row + off, srcs[i] + (size_t)s * cols[i], cols[i] * sizeof(_Float16));
      off += cols[i];
    }
  }
}

void rocket_projector_dewindow(const rocket_backbone_geom *g, const _Float16 *feat, _Float16 *tap) {
  const int D = g->d;   // backbone feature stride (== RPJ_DIN for the RF-DETR ViT-S backbone)
  const int iwn = g->grid / g->win;
  for (int w = 0; w < g->nw; w++) {
    int wh = w / g->win, ww = w % g->win;
    const _Float16 *wbase = feat + (size_t)w * g->tw * D;   // window's tw rows
    for (int j = 0; j < iwn * iwn; j++) {                       // patch tokens (cls excluded)
      int ih = j / iwn, iw = j % iwn;
      int ph = wh * iwn + ih, pw = ww * iwn + iw;
      int t = ph * g->grid + pw;                               // spatial position
      const _Float16 *src = wbase + (size_t)(1 + j) * D;   // row 0 is cls, skip
      memcpy(tap + (size_t)t * D, src, (size_t)D * sizeof(_Float16));
    }
  }
}

// conv3x3 pad1 stride1 over the grid x grid map, CM->CM channels.
static int conv3x3(int fd, const _Float16 *in, const _Float16 *W, _Float16 *out, int grid) {
  rocket_conv2d_desc d = { .ic = RPJ_CM, .ih = grid, .iw = grid, .oc = RPJ_CM,
                           .kh = 3, .kw = 3, .stride_y = 1, .stride_x = 1,
                           .pad_top = 1, .pad_left = 1, .dil_y = 1, .dil_x = 1, .depthwise = 0 };
  return rocket_conv2d_fp16(fd, &d, in, W, out);
}

// A bottleneck: t_in[np,128] -> (conv3x3 -> chLN -> SiLU) x2 -> m_out[np,128].
static int bottleneck(int fd, int np, int grid, float eps, const _Float16 *t_in,
                      const _Float16 *w1, const _Float16 *g1, const _Float16 *b1,
                      const _Float16 *w2, const _Float16 *g2, const _Float16 *b2,
                      _Float16 *m_out, _Float16 *cube, _Float16 *convo,
                      _Float16 *tok, _Float16 *ln) {
  int rc;
  tok_to_cube(t_in, cube, RPJ_CM, np);
  if ((rc = conv3x3(fd, cube, w1, convo, grid))) return rc;
  cube_to_tok(convo, tok, RPJ_CM, np);
  if ((rc = rocket_layernorm_fp16(fd, np, RPJ_CM, tok, g1, b1, eps, ln))) return rc;
  if ((rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_SILU, ln, tok, np * RPJ_CM))) return rc;
  tok_to_cube(tok, cube, RPJ_CM, np);
  if ((rc = conv3x3(fd, cube, w2, convo, grid))) return rc;
  cube_to_tok(convo, tok, RPJ_CM, np);
  if ((rc = rocket_layernorm_fp16(fd, np, RPJ_CM, tok, g2, b2, eps, ln))) return rc;
  if ((rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_SILU, ln, m_out, np * RPJ_CM))) return rc;
  return 0;
}

int rocket_projector_run(int fd, const rocket_projector_weights *w,
                         const _Float16 *const taps[RPJ_NTAP], _Float16 *y_out) {
  const rocket_backbone_geom *g = w->geom;
  const int np = g->np, grid = g->grid;
  int rc = 0;
  const float eps = w->eps;
  _Float16 *x0 = malloc((size_t)np * RPJ_CIN * sizeof(_Float16));
  _Float16 *cv1 = malloc((size_t)np * RPJ_CO * sizeof(_Float16));
  _Float16 *cv1a = malloc((size_t)np * RPJ_CO * sizeof(_Float16));
  _Float16 *a = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  _Float16 *b = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  _Float16 *m0 = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  _Float16 *m1 = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  _Float16 *m2 = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  _Float16 *cat = malloc((size_t)np * RPJ_CCAT * sizeof(_Float16));
  _Float16 *cv2 = malloc((size_t)np * RPJ_CO * sizeof(_Float16));
  _Float16 *cv2a = malloc((size_t)np * RPJ_CO * sizeof(_Float16));
  _Float16 *cube = malloc((size_t)RPJ_CM * np * sizeof(_Float16));
  _Float16 *convo = malloc((size_t)RPJ_CM * np * sizeof(_Float16));
  _Float16 *btok = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  _Float16 *bln = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  if (!x0 || !cv1 || !cv1a || !a || !b || !m0 || !m1 || !m2 || !cat || !cv2 || !cv2a ||
      !cube || !convo || !btok || !bln) { rc = -1; goto out; }

  // concat 4 taps -> x0[np,1536]
  { const _Float16 *srcs[4] = { taps[0], taps[1], taps[2], taps[3] };
    int cols[4] = { RPJ_DIN, RPJ_DIN, RPJ_DIN, RPJ_DIN };
    concat_cols(x0, RPJ_CIN, srcs, cols, 4, np); }

  // cv1: matmul -> chLN -> SiLU -> split
  if ((rc = rocket_matmul_fp16(fd, np, RPJ_CIN, RPJ_CO, x0, w->cv1_w, cv1))) goto out;
  if ((rc = rocket_layernorm_fp16(fd, np, RPJ_CO, cv1, w->cv1_g, w->cv1_b, eps, cv1a))) goto out;
  if ((rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_SILU, cv1a, cv1, np * RPJ_CO))) goto out;
  for (int s = 0; s < np; s++) {
    memcpy(a + (size_t)s * RPJ_CM, cv1 + (size_t)s * RPJ_CO, RPJ_CM * sizeof(_Float16));
    memcpy(b + (size_t)s * RPJ_CM, cv1 + (size_t)s * RPJ_CO + RPJ_CM, RPJ_CM * sizeof(_Float16));
  }

  // 3 chained bottlenecks on b
  if ((rc = bottleneck(fd, np, grid, eps, b,  w->m_w[0][0], w->m_g[0][0], w->m_b[0][0],
                       w->m_w[0][1], w->m_g[0][1], w->m_b[0][1], m0, cube, convo, btok, bln))) goto out;
  if ((rc = bottleneck(fd, np, grid, eps, m0, w->m_w[1][0], w->m_g[1][0], w->m_b[1][0],
                       w->m_w[1][1], w->m_g[1][1], w->m_b[1][1], m1, cube, convo, btok, bln))) goto out;
  if ((rc = bottleneck(fd, np, grid, eps, m1, w->m_w[2][0], w->m_g[2][0], w->m_b[2][0],
                       w->m_w[2][1], w->m_g[2][1], w->m_b[2][1], m2, cube, convo, btok, bln))) goto out;

  // concat [a,b,m0,m1,m2] -> cat[np,640]
  { const _Float16 *srcs[5] = { a, b, m0, m1, m2 };
    int cols[5] = { RPJ_CM, RPJ_CM, RPJ_CM, RPJ_CM, RPJ_CM };
    concat_cols(cat, RPJ_CCAT, srcs, cols, 5, np); }

  // cv2: matmul -> chLN -> SiLU
  if ((rc = rocket_matmul_fp16(fd, np, RPJ_CCAT, RPJ_CO, cat, w->cv2_w, cv2))) goto out;
  if ((rc = rocket_layernorm_fp16(fd, np, RPJ_CO, cv2, w->cv2_g, w->cv2_b, eps, cv2a))) goto out;
  if ((rc = rocket_activation_fp16(fd, ROCKET_ACTIVATION_SILU, cv2a, cv2, np * RPJ_CO))) goto out;
  // final chLN (no activation)
  if ((rc = rocket_layernorm_fp16(fd, np, RPJ_CO, cv2, w->fin_g, w->fin_b, eps, y_out))) goto out;

out:
  free(x0); free(cv1); free(cv1a); free(a); free(b); free(m0); free(m1); free(m2);
  free(cat); free(cv2); free(cv2a); free(cube); free(convo); free(btok); free(bln);
  return rc;
}

// ===========================================================================================
// Resident (prepacked 1x1 + resident-BO conv) projector. See rocket_projector.h. The two 1x1
// convs are prepacked matmuls; the six 3x3 bottleneck convs run through a resident-BO conv
// context; chLN/SiLU/final chLN run threaded on the host.
// ===========================================================================================

struct rocket_projector_ctx {
  const rocket_projector_weights *w;
  const rocket_backbone_geom *g;
  rocket_ctx      *mm;        // prepacked cv1/cv2
  rocket_conv_ctx *conv;      // resident-BO 3x3 convs
  int              conv_fd;   // fd the conv ctx borrows (owned here)
  int              ht;        // host worker threads (chLN/SiLU)
  rocket_weights  *w_cv1, *w_cv2;
  // Per-image scratch, owned by the ctx and allocated ONCE (not malloc'd per image; single-Run).
  _Float16 *x0, *cv1, *a, *b, *m0, *m1, *m2, *cat, *cv2, *cube, *convo, *btok, *bln;
};

// conv3x3 pad1 stride1 CM->CM over the grid x grid map, on the resident conv ctx.
static int conv3x3_ctx(rocket_conv_ctx *cx, const _Float16 *in, const _Float16 *W,
                       _Float16 *out, int grid) {
  rocket_conv2d_desc d = { .ic = RPJ_CM, .ih = grid, .iw = grid, .oc = RPJ_CM,
                           .kh = 3, .kw = 3, .stride_y = 1, .stride_x = 1,
                           .pad_top = 1, .pad_left = 1, .dil_y = 1, .dil_x = 1, .depthwise = 0 };
  return rocket_conv2d_fp16_ctx(cx, &d, in, W, out);
}

// A bottleneck on the resident conv ctx: t_in[np,128] -> (conv3x3 -> chLN -> SiLU) x2.
static int bottleneck_ctx(rocket_projector_ctx *c, const _Float16 *t_in,
                          const _Float16 *w1, const _Float16 *g1, const _Float16 *b1,
                          const _Float16 *w2, const _Float16 *g2, const _Float16 *b2,
                          _Float16 *m_out, _Float16 *cube, _Float16 *convo,
                          _Float16 *tok, _Float16 *lnb) {
  int rc;
  const int np = c->g->np, grid = c->g->grid;
  const float eps = c->w->eps;
  tok_to_cube(t_in, cube, RPJ_CM, np);
  if ((rc = conv3x3_ctx(c->conv, cube, w1, convo, grid))) return rc;
  cube_to_tok(convo, tok, RPJ_CM, np);
  rocket_host_layernorm(np, RPJ_CM, tok, g1, b1, eps, lnb, c->ht);
  rocket_host_silu((size_t)np * RPJ_CM, lnb, c->ht);   // lnb <- SiLU(chLN(conv))
  tok_to_cube(lnb, cube, RPJ_CM, np);
  if ((rc = conv3x3_ctx(c->conv, cube, w2, convo, grid))) return rc;
  cube_to_tok(convo, tok, RPJ_CM, np);
  rocket_host_layernorm(np, RPJ_CM, tok, g2, b2, eps, m_out, c->ht);
  rocket_host_silu((size_t)np * RPJ_CM, m_out, c->ht);
  return 0;
}

rocket_projector_ctx *rocket_projector_ctx_create(const rocket_projector_weights *w, int nthreads) {
  if (!w || !w->geom) return NULL;
  rocket_projector_ctx *c = calloc(1, sizeof(*c));
  if (!c) return NULL;
  c->w = w;
  c->g = w->geom;
  const int np = c->g->np;
  c->conv_fd = -1;
  c->ht = (nthreads <= 1) ? 1 : 4;

  c->mm = rocket_ctx_create(nthreads);
  c->conv_fd = rocket_open();
  if (!c->mm || c->conv_fd < 0) goto fail;
  c->conv = rocket_conv_ctx_create(c->conv_fd);
  if (!c->conv) goto fail;

  c->w_cv1 = rocket_weights_pack(c->mm, np, RPJ_CIN,  RPJ_CO, w->cv1_w);
  c->w_cv2 = rocket_weights_pack(c->mm, np, RPJ_CCAT, RPJ_CO, w->cv2_w);
  if (!c->w_cv1 || !c->w_cv2) goto fail;

  // Per-image scratch, allocated once (the ctx is single-Run).
  c->x0    = malloc((size_t)np * RPJ_CIN * sizeof(_Float16));
  c->cv1   = malloc((size_t)np * RPJ_CO * sizeof(_Float16));
  c->a     = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  c->b     = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  c->m0    = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  c->m1    = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  c->m2    = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  c->cat   = malloc((size_t)np * RPJ_CCAT * sizeof(_Float16));
  c->cv2   = malloc((size_t)np * RPJ_CO * sizeof(_Float16));
  c->cube  = malloc((size_t)RPJ_CM * np * sizeof(_Float16));
  c->convo = malloc((size_t)RPJ_CM * np * sizeof(_Float16));
  c->btok  = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  c->bln   = malloc((size_t)np * RPJ_CM * sizeof(_Float16));
  if (!c->x0 || !c->cv1 || !c->a || !c->b || !c->m0 || !c->m1 || !c->m2 || !c->cat ||
      !c->cv2 || !c->cube || !c->convo || !c->btok || !c->bln) goto fail;
  return c;
fail:
  rocket_projector_ctx_free(c);
  return NULL;
}

void rocket_projector_ctx_free(rocket_projector_ctx *c) {
  if (!c) return;
  if (c->mm) {
    if (c->w_cv1) rocket_weights_free(c->mm, c->w_cv1);
    if (c->w_cv2) rocket_weights_free(c->mm, c->w_cv2);
  }
  if (c->conv) rocket_conv_ctx_free(c->conv);
  if (c->mm) rocket_ctx_free(c->mm);
  if (c->conv_fd >= 0) rocket_close(c->conv_fd);
  free(c->x0); free(c->cv1); free(c->a); free(c->b); free(c->m0); free(c->m1); free(c->m2);
  free(c->cat); free(c->cv2); free(c->cube); free(c->convo); free(c->btok); free(c->bln);
  free(c);
}

int rocket_projector_run_ctx(rocket_projector_ctx *c,
                             const _Float16 *const taps[RPJ_NTAP], _Float16 *y_out) {
  if (!c || !y_out) return -1;
  rocket_host_pin_self();   // keep the compute thread's serial parts on an A76, not an A55
  const rocket_projector_weights *w = c->w;
  const int np = c->g->np;
  const float eps = w->eps;
  int rc = 0;
  // Scratch is ctx-owned (allocated once at create), not malloc'd per image.
  _Float16 *x0 = c->x0, *cv1 = c->cv1, *a = c->a, *b = c->b, *m0 = c->m0, *m1 = c->m1,
           *m2 = c->m2, *cat = c->cat, *cv2 = c->cv2, *cube = c->cube, *convo = c->convo,
           *btok = c->btok, *bln = c->bln;

  // concat 4 taps -> x0[np,1536]
  { const _Float16 *srcs[4] = { taps[0], taps[1], taps[2], taps[3] };
    int cols[4] = { RPJ_DIN, RPJ_DIN, RPJ_DIN, RPJ_DIN };
    concat_cols(x0, RPJ_CIN, srcs, cols, 4, np); }

  // cv1: prepacked matmul -> chLN -> SiLU -> split
  if ((rc = rocket_matmul_fp16_prepacked(c->mm, np, RPJ_CIN, RPJ_CO, x0, cv1, c->w_cv1))) goto out;
  rocket_host_layernorm(np, RPJ_CO, cv1, w->cv1_g, w->cv1_b, eps, cv1, c->ht);
  rocket_host_silu((size_t)np * RPJ_CO, cv1, c->ht);
  for (int s = 0; s < np; s++) {
    memcpy(a + (size_t)s * RPJ_CM, cv1 + (size_t)s * RPJ_CO, RPJ_CM * sizeof(_Float16));
    memcpy(b + (size_t)s * RPJ_CM, cv1 + (size_t)s * RPJ_CO + RPJ_CM, RPJ_CM * sizeof(_Float16));
  }

  // 3 chained bottlenecks on b
  if ((rc = bottleneck_ctx(c, b,  w->m_w[0][0], w->m_g[0][0], w->m_b[0][0],
                           w->m_w[0][1], w->m_g[0][1], w->m_b[0][1], m0, cube, convo, btok, bln))) goto out;
  if ((rc = bottleneck_ctx(c, m0, w->m_w[1][0], w->m_g[1][0], w->m_b[1][0],
                           w->m_w[1][1], w->m_g[1][1], w->m_b[1][1], m1, cube, convo, btok, bln))) goto out;
  if ((rc = bottleneck_ctx(c, m1, w->m_w[2][0], w->m_g[2][0], w->m_b[2][0],
                           w->m_w[2][1], w->m_g[2][1], w->m_b[2][1], m2, cube, convo, btok, bln))) goto out;

  // concat [a,b,m0,m1,m2] -> cat[np,640]
  { const _Float16 *srcs[5] = { a, b, m0, m1, m2 };
    int cols[5] = { RPJ_CM, RPJ_CM, RPJ_CM, RPJ_CM, RPJ_CM };
    concat_cols(cat, RPJ_CCAT, srcs, cols, 5, np); }

  // cv2: prepacked matmul -> chLN -> SiLU -> final chLN (no activation)
  if ((rc = rocket_matmul_fp16_prepacked(c->mm, np, RPJ_CCAT, RPJ_CO, cat, cv2, c->w_cv2))) goto out;
  rocket_host_layernorm(np, RPJ_CO, cv2, w->cv2_g, w->cv2_b, eps, cv2, c->ht);
  rocket_host_silu((size_t)np * RPJ_CO, cv2, c->ht);
  rocket_host_layernorm(np, RPJ_CO, cv2, w->fin_g, w->fin_b, eps, y_out, c->ht);

out:
  return rc;   // scratch is ctx-owned; nothing to free per call
}
