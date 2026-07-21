// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ort-rocket authors
//
// rocket_ort_host -- host elementwise helpers for the resident backbone/projector paths.
// See rocket_ort_host.h. The GELU/SiLU LUTs mirror the SigLIP resident encoder's approach:
// the activation is a function of a single 16-bit value, so all 65536 outputs fit one table
// and the hot per-element transcendental becomes a load.

#include "rocket_ort_host.h"

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

// ---- parallel-for ---------------------------------------------------------------------
// Pin each host worker to a big (A76) core, exactly as the NPU fan-out workers are pinned.
// Without this the scheduler scatters the workers onto the A55 little cores and the join
// barrier stalls on the slow straggler -- measured ~1.7x slower than pinned on the ViT-S
// backbone. rocket_pin_worker (librocketnpu) auto-detects the big cluster by max frequency
// and is a no-op if the library was built without affinity support. rocket_affinity_set_base
// shifts this thread's rotation so the worker pins to big_core[(base + idx) % nbig].
void rocket_pin_worker(int worker_idx);
void rocket_affinity_set_base(int base);

// Big-core pinning is default-on (the measured single-stream latency win). A stream pool has two
// escapes from every process crowding the same big cores: ROCKET_ORT_PIN=0 gives up the pin and
// lets the scheduler spread across all 8 cores, or -- keeping the pin -- ROCKET_ORT_AFFINITY_BASE
// =<rank> per process rotates that process's workers onto a different big core, so P<=nbig streams
// partition the cluster (with ROCKET_ORT_THREADS=1, exactly one A76 per stream) instead of stacking
// on it. The base is thread-local, so each pinned thread sets it just before pinning.
static int g_pin = 1;
static int g_base = 0;   // ROCKET_ORT_AFFINITY_BASE: per-process big-cluster rotation (pool spreading)
static pthread_once_t g_pin_once = PTHREAD_ONCE_INIT;
static void pin_init(void)
{
    const char *e = getenv("ROCKET_ORT_PIN"); if (e && atoi(e) == 0) g_pin = 0;
    const char *b = getenv("ROCKET_ORT_AFFINITY_BASE"); if (b) { int v = atoi(b); g_base = v < 0 ? 0 : v; }
}
static void pin_thread(int idx)
{
    pthread_once(&g_pin_once, pin_init);
    if (!g_pin) return;
    rocket_affinity_set_base(g_base);
    rocket_pin_worker(idx);
}

void rocket_host_pin_self(void) { pin_thread(0); }

// ---- persistent worker pool -----------------------------------------------------------
// The resident backbone issues ~100 parallel-fors per image (LN/gather/scatter/bias-residual/
// bias-gelu across 12 layers plus the projector), so a pthread_create+join per call was ~400
// spawn/join pairs per inference of pure host overhead on this dispatch/host-glue-bound path.
// Instead each compute thread keeps a persistent pool: the workers are spawned once, park on a
// condvar, and are woken per job. The pool is thread-local, so process-per-stream (the benchmarked
// deployment) gets one pool per process and an in-process stream pool gets one per compute thread,
// both contention-free; combined with the one-Run-per-session serialization, only the owning
// thread ever submits to its pool. A submitted job splits [0,n) into nt lanes: the submitter runs
// lane 0 inline (on its already-pinned big core) and workers run lanes 1..nt-1. The passed fn must
// not itself call a host parallel-for (no nested submit on one thread) -- none of the callers do.
#define HOSTPOOL_MAXW      7        // up to 8 lanes total (submitter lane 0 + 7 workers)
#define HOST_PAR_MIN_WORK  32768    // below this many element-ops, run inline (a pool wake costs more)

struct hostpool;
typedef struct { struct hostpool *pool; int lane; } hostpool_warg;
typedef struct hostpool {
    pthread_mutex_t mtx;
    pthread_cond_t  cv_work;        // workers wait here for a job (gen advance) or shutdown
    pthread_cond_t  cv_done;        // submitter waits here for completion / creation handshake
    void (*fn)(void *, int, int);
    void *arg;
    int n, nt;                      // current job: range [0,n) split into nt lanes
    unsigned gen;                   // bumped per job; a worker acts when gen advances past its seen
    int remaining;                  // workers yet to finish the current job
    int ready;                      // workers parked and ready (creation handshake)
    int nworkers;
    int shutdown;
    pthread_t th[HOSTPOOL_MAXW];
    hostpool_warg warg[HOSTPOOL_MAXW];
} hostpool;

// Run one lane's contiguous chunk. Job fields are stable for the job's duration (set under lock
// before the gen bump, untouched until remaining hits 0), so reading them unlocked is safe.
static void hostpool_do_chunk(hostpool *p, int lane)
{
    if (lane >= p->nt) return;
    int chunk = (p->n + p->nt - 1) / p->nt;
    int lo = lane * chunk, hi = lo + chunk;
    if (lo >= p->n) return;
    if (hi > p->n) hi = p->n;
    p->fn(p->arg, lo, hi);
}

static void *hostpool_worker(void *pv)
{
    hostpool_warg *w = pv;
    hostpool *p = w->pool;
    int lane = w->lane;
    pin_thread(lane);                              // pin this worker to its big-core lane, once
    pthread_mutex_lock(&p->mtx);
    p->ready++;
    pthread_cond_signal(&p->cv_done);              // tell hostpool_ensure we are parked
    unsigned seen = p->gen;
    for (;;) {
        while (p->gen == seen && !p->shutdown) pthread_cond_wait(&p->cv_work, &p->mtx);
        if (p->shutdown) break;
        seen = p->gen;
        pthread_mutex_unlock(&p->mtx);
        hostpool_do_chunk(p, lane);
        pthread_mutex_lock(&p->mtx);
        if (--p->remaining == 0) pthread_cond_signal(&p->cv_done);
    }
    pthread_mutex_unlock(&p->mtx);
    return NULL;
}

// Grow the pool to `want` workers, blocking until the new ones are parked (so a subsequent submit
// counts them in `remaining`). Only the owning thread calls this, never during a live job.
static void hostpool_ensure(hostpool *p, int want)
{
    if (want > HOSTPOOL_MAXW) want = HOSTPOOL_MAXW;
    if (want <= p->nworkers) return;
    pthread_mutex_lock(&p->mtx);
    while (p->nworkers < want) {
        int k = p->nworkers;
        p->warg[k].pool = p; p->warg[k].lane = k + 1;   // workers take lanes 1..; lane 0 is submitter
        if (pthread_create(&p->th[k], NULL, hostpool_worker, &p->warg[k]) != 0) break;
        p->nworkers++;
    }
    while (p->ready < p->nworkers) pthread_cond_wait(&p->cv_done, &p->mtx);
    pthread_mutex_unlock(&p->mtx);
}

static void hostpool_submit(hostpool *p, int n, int nt, void (*fn)(void *, int, int), void *arg)
{
    hostpool_ensure(p, nt - 1);
    pthread_mutex_lock(&p->mtx);
    p->fn = fn; p->arg = arg; p->n = n; p->nt = nt;
    p->remaining = p->nworkers;                    // every parked worker decrements once
    p->gen++;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mtx);
    hostpool_do_chunk(p, 0);                        // submitter runs lane 0 on its pinned core
    pthread_mutex_lock(&p->mtx);
    while (p->remaining != 0) pthread_cond_wait(&p->cv_done, &p->mtx);
    pthread_mutex_unlock(&p->mtx);
}

static pthread_key_t   g_pool_key;
static pthread_once_t  g_pool_key_once = PTHREAD_ONCE_INIT;
static void hostpool_destroy(void *pv)             // pthread_key destructor: runs at thread exit
{
    hostpool *p = pv;
    pthread_mutex_lock(&p->mtx);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mtx);
    for (int k = 0; k < p->nworkers; k++) pthread_join(p->th[k], NULL);
    pthread_mutex_destroy(&p->mtx);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    free(p);
}
static void pool_key_init(void) { pthread_key_create(&g_pool_key, hostpool_destroy); }

static hostpool *pool_get(void)
{
    pthread_once(&g_pool_key_once, pool_key_init);
    hostpool *p = pthread_getspecific(g_pool_key);
    if (!p) {
        p = calloc(1, sizeof *p);
        if (!p) return NULL;
        pthread_mutex_init(&p->mtx, NULL);
        pthread_cond_init(&p->cv_work, NULL);
        pthread_cond_init(&p->cv_done, NULL);
        pthread_setspecific(g_pool_key, p);
    }
    return p;
}

// Dispatch [0,n) across nt lanes via the persistent pool (caller has already decided to thread).
static void pfor_pool(int n, int nt, void (*fn)(void *, int, int), void *arg)
{
    if (nt > HOSTPOOL_MAXW + 1) nt = HOSTPOOL_MAXW + 1;
    hostpool *p = pool_get();
    if (!p) { fn(arg, 0, n); return; }             // OOM: run serially
    hostpool_submit(p, n, nt, fn, arg);
}

// Public entry: threshold on row count (the caller cannot express per-item cost here -- used by the
// gather/scatter fan-outs, whose per-group work is large even when the group count is small).
void rocket_host_parallel_for(int n, int nt, void (*fn)(void *, int, int), void *arg)
{
    if (nt < 2 || n < 2 * nt) { fn(arg, 0, n); return; }
    pfor_pool(n, nt, fn, arg);
}

// Work-aware entry (in-file helpers, which know their inner width): gate on estimated element-ops
// so a small elementwise op runs inline instead of paying the pool wake. `work` ~ n * inner_width.
static void pfor_work(int n, int nt, long work, void (*fn)(void *, int, int), void *arg)
{
    if (nt < 2 || n < 2 * nt || work < HOST_PAR_MIN_WORK) { fn(arg, 0, n); return; }
    pfor_pool(n, nt, fn, arg);
}

// ---- LayerNorm (fp32 accumulate, threaded over rows) ----------------------------------
typedef struct { int d; const _Float16 *x, *g, *b; float eps; _Float16 *out; } ln_arg;
static void ln_rows(void *a, int lo, int hi)
{
    ln_arg *s = a; int d = s->d;
    for (int i = lo; i < hi; i++) {
        const _Float16 *xr = s->x + (size_t)i * d;
        double mean = 0; for (int j = 0; j < d; j++) mean += (double)xr[j]; mean /= d;
        double var = 0; for (int j = 0; j < d; j++) { double t = (double)xr[j] - mean; var += t * t; }
        double inv = 1.0 / sqrt(var / d + s->eps);
        _Float16 *o = s->out + (size_t)i * d;
        for (int j = 0; j < d; j++)
            o[j] = (_Float16)(((double)xr[j] - mean) * inv * (double)s->g[j] + (double)s->b[j]);
    }
}
void rocket_host_layernorm(int M, int d, const _Float16 *x, const _Float16 *g,
                           const _Float16 *b, float eps, _Float16 *out, int nt)
{
    ln_arg a = { d, x, g, b, eps, out };
    pfor_work(M, nt, (long)M * d, ln_rows, &a);
}

// ---- bias / residual (threaded, memory-bound) -----------------------------------------
typedef struct { int N; _Float16 *C; const _Float16 *y, *b; } br_arg;
static void bias_rows(void *a, int lo, int hi)
{
    br_arg *s = a; int N = s->N;
    for (int i = lo; i < hi; i++) {
        _Float16 *r = s->C + (size_t)i * N;
        for (int j = 0; j < N; j++) r[j] = (_Float16)((float)r[j] + (float)s->b[j]);
    }
}
void rocket_host_add_bias(int M, int N, _Float16 *C, const _Float16 *b, int nt)
{
    if (!b) return;
    br_arg a = { N, C, NULL, b };
    pfor_work(M, nt, (long)M * N, bias_rows, &a);
}

typedef struct { _Float16 *acc; const _Float16 *add; } res_arg;
static void res_range(void *a, int lo, int hi)
{
    res_arg *s = a;
    for (int i = lo; i < hi; i++) s->acc[i] = (_Float16)((float)s->acc[i] + (float)s->add[i]);
}
void rocket_host_residual(size_t n, _Float16 *acc, const _Float16 *add, int nt)
{
    res_arg a = { acc, add };
    pfor_work((int)n, nt, (long)n, res_range, &a);
}

static void bias_res_rows(void *a, int lo, int hi)
{
    br_arg *s = a; int N = s->N;
    for (int i = lo; i < hi; i++) {
        _Float16 *r = s->C + (size_t)i * N;
        const _Float16 *yr = s->y + (size_t)i * N;
        if (s->b) for (int j = 0; j < N; j++) r[j] = (_Float16)((float)r[j] + (float)yr[j] + (float)s->b[j]);
        else      for (int j = 0; j < N; j++) r[j] = (_Float16)((float)r[j] + (float)yr[j]);
    }
}
void rocket_host_bias_residual(int M, int N, _Float16 *acc, const _Float16 *y,
                               const _Float16 *b, int nt)
{
    br_arg a = { N, acc, y, b };
    pfor_work(M, nt, (long)M * N, bias_res_rows, &a);
}

// ---- activation LUTs (built once, bit-exact to the scalar formula) --------------------
static inline _Float16 gelu1(_Float16 xv)
{
    const float inv_sqrt2 = 0.70710678118654752440f;   // 1/sqrt(2)
    float v = (float)xv;
    return (_Float16)(0.5f * v * (1.f + erff(v * inv_sqrt2)));
}
static inline _Float16 silu1(_Float16 xv)
{
    float v = (float)xv;
    return (_Float16)(v / (1.f + expf(-v)));
}

static _Float16 g_gelu_lut[65536];
static _Float16 g_silu_lut[65536];
static pthread_once_t g_gelu_once = PTHREAD_ONCE_INIT;
static pthread_once_t g_silu_once = PTHREAD_ONCE_INIT;
static void gelu_lut_build(void)
{
    for (int u = 0; u < 65536; u++) {
        uint16_t bits = (uint16_t)u; _Float16 h; memcpy(&h, &bits, sizeof h);
        g_gelu_lut[u] = gelu1(h);
    }
}
static void silu_lut_build(void)
{
    for (int u = 0; u < 65536; u++) {
        uint16_t bits = (uint16_t)u; _Float16 h; memcpy(&h, &bits, sizeof h);
        g_silu_lut[u] = silu1(h);
    }
}

static void gelu_range(void *a, int lo, int hi)
{
    _Float16 *x = a;
    for (int i = lo; i < hi; i++) { uint16_t b; memcpy(&b, &x[i], sizeof b); x[i] = g_gelu_lut[b]; }
}
static void silu_range(void *a, int lo, int hi)
{
    _Float16 *x = a;
    for (int i = lo; i < hi; i++) { uint16_t b; memcpy(&b, &x[i], sizeof b); x[i] = g_silu_lut[b]; }
}

void rocket_host_gelu(size_t n, _Float16 *x, int nt)
{
    pthread_once(&g_gelu_once, gelu_lut_build);
    pfor_work((int)n, nt, (long)n, gelu_range, x);
}

// Fused bias + GELU: one pass over the [M,N] tile instead of an add_bias pass then a gelu
// pass. Bit-exact -- v = fp16(x+b) is the same value the separate add_bias would store, and
// it indexes the same LUT.
typedef struct { int N; _Float16 *x; const _Float16 *b; } bg_arg;
static void bias_gelu_rows(void *a, int lo, int hi)
{
    bg_arg *s = a; int N = s->N;
    for (int i = lo; i < hi; i++) {
        _Float16 *r = s->x + (size_t)i * N;
        if (s->b)
            for (int j = 0; j < N; j++) {
                _Float16 v = (_Float16)((float)r[j] + (float)s->b[j]);
                uint16_t bits; memcpy(&bits, &v, sizeof bits); r[j] = g_gelu_lut[bits];
            }
        else
            for (int j = 0; j < N; j++) {
                uint16_t bits; memcpy(&bits, &r[j], sizeof bits); r[j] = g_gelu_lut[bits];
            }
    }
}
void rocket_host_bias_gelu(int M, int N, _Float16 *x, const _Float16 *b, int nt)
{
    pthread_once(&g_gelu_once, gelu_lut_build);
    bg_arg a = { N, x, b };
    pfor_work(M, nt, (long)M * N, bias_gelu_rows, &a);
}
void rocket_host_silu(size_t n, _Float16 *x, int nt)
{
    pthread_once(&g_silu_once, silu_lut_build);
    pfor_work((int)n, nt, (long)n, silu_range, x);
}
