// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_ort_host -- host-side elementwise helpers shared by the resident (prepacked,
// multicore) backbone and projector paths. The resident chain keeps the big GEMMs and the
// attention on the NPU (packed once, fanned across cores) but runs the memory-bound glue --
// LayerNorm over the channel axis, bias adds, residual adds, and the GELU/SiLU pointwise
// activations -- on the host A76 cores, where the data already lives de-tiled after the NPU
// readback. These are fp32-accumulate (LayerNorm) and fp16-exact LUT (activations), so they
// are at least as faithful as the on-NPU path they replace.

#ifndef ORT_ROCKET_HOST_H
#define ORT_ROCKET_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Parallel-for over [0,n): split into <=nt contiguous chunks across the big cores. The host
// ops here are memory-bound and embarrassingly parallel over rows/elements. Each worker is
// pinned to an A76 big core (else the scheduler scatters them onto the A55s and the join
// barrier stalls on the slow straggler).
void rocket_host_parallel_for(int n, int nt, void (*fn)(void *, int, int), void *arg);

// Pin the CALLING thread (the per-image compute thread) to a big core, so its serial parts --
// im2col/stem, the matmul input-pack and NEON de-tile between NPU submits -- run on an A76 and
// not an A55. Respects the librocketnpu affinity base (set per-process for a stream pool).
// No-op if the library was built without affinity support.
void rocket_host_pin_self(void);

// Channel-axis LayerNorm: out[i, :] = (x[i,:] - mean) / sqrt(var + eps) * g + b, over the
// last axis of length d, for M rows. fp32 accumulate. Threaded across nt host workers.
void rocket_host_layernorm(int M, int d, const _Float16 *x, const _Float16 *g,
                           const _Float16 *b, float eps, _Float16 *out, int nt);

// Broadcast bias add in place: C[i, j] += b[j], over M rows of width N. b may be NULL (no-op).
// Threaded across nt host workers (memory-bound, embarrassingly parallel over rows).
void rocket_host_add_bias(int M, int N, _Float16 *C, const _Float16 *b, int nt);

// Residual add in place: acc[i] += add[i], over n elements. Threaded across nt workers.
void rocket_host_residual(size_t n, _Float16 *acc, const _Float16 *add, int nt);

// Fused bias + residual in place: acc[i,j] += y[i,j] + b[j], over M rows of width N (one pass
// instead of a bias pass then a residual pass). b may be NULL. Threaded across nt workers.
void rocket_host_bias_residual(int M, int N, _Float16 *acc, const _Float16 *y,
                               const _Float16 *b, int nt);

// Exact-erf GELU in place over n elements: x <- 0.5*x*(1+erf(x/sqrt(2))). Uses a 65536-entry
// fp16 LUT (bit-exact to the scalar path over every fp16 pattern), built once on first use.
// Threaded across nt host workers.
void rocket_host_gelu(size_t n, _Float16 *x, int nt);

// Fused bias + GELU in place: x[i,j] <- GELU(x[i,j] + b[j]), over M rows of width N (one pass
// instead of a bias pass then a GELU pass). b may be NULL (plain GELU). Bit-exact to the two
// passes -- the same fp16 rounding of x+b feeds the same LUT. Threaded across nt workers.
void rocket_host_bias_gelu(int M, int N, _Float16 *x, const _Float16 *b, int nt);

// SiLU in place over n elements: x <- x*sigmoid(x). fp16 LUT, built once. Threaded.
void rocket_host_silu(size_t n, _Float16 *x, int nt);

#ifdef __cplusplus
}
#endif
#endif  // ORT_ROCKET_HOST_H
