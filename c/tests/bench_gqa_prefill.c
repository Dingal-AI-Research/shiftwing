/* Full-attention prefill microbenchmark at real Ornith397 shapes.
 *
 * Why this exists: the 8,451-token qualification spends 421.8 s in full
 * attention, but a full-model arm costs ~17 minutes, so every attention
 * hypothesis so far has been argued rather than measured.  One layer of
 * synthetic weights at the model's real dimensions reproduces the same kernel
 * sequence in seconds.
 *
 * What it measures: the engine's current prefill behaviour, which is the
 * single-token decode entry point called once per position
 * (qwen.c:4105 -> attn_forward -> cuda_attn_try ->
 * coli_cuda_gqa_decode_q4_f16).  Weight values are synthetic, so the numbers
 * are timings, not outputs; the checksum only guards against a degenerate run
 * that computed nothing.
 *
 * It also probes the dynamic-shared-memory ceiling.  gqa_decode_kernel sizes
 * its score buffer at (position+1)*4 bytes and nothing raises the 48 KB
 * default cap, so the launch is expected to fail somewhere near 12.25k
 * positions -- which in the engine disables CUDA entirely.
 */
#include "../backend_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HIDDEN = 4096, QHEADS = 32, KVHEADS = 2, HEAD_DIM = 256,
    ROTARY = 64, GROUP = 128,
    QROWS = QHEADS * HEAD_DIM * 2,   /* q_proj fuses [q|gate] per head */
    KVROWS = KVHEADS * HEAD_DIM,
    CTXN = QHEADS * HEAD_DIM
};

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Deterministic filler.  Scales stay small so softmax cannot saturate and
 * turn the run into a NaN benchmark. */
static void fill_q4(unsigned char *q, float *s, int rows, int cols) {
    int ng = cols / GROUP, rb = ng * (GROUP / 2);
    for (size_t i = 0; i < (size_t)rows * rb; i++)
        q[i] = (unsigned char)((i * 29u + 17u) & 0xffu);
    for (size_t i = 0; i < (size_t)rows * ng; i++)
        s[i] = ldexpf(1.0f, -12 + (int)(i & 1u));
}

static void fill_q8(signed char *q, float *s, int rows, int cols) {
    for (size_t i = 0; i < (size_t)rows * cols; i++)
        q[i] = (signed char)((int)((i * 37u + 11u) & 0x7fu) - 64);
    for (int r = 0; r < rows; r++) s[r] = ldexpf(1.0f, -14);
}

struct Weights {
    void *q, *k, *v, *o;
    void *qs, *ks, *vs, *os;
    void *q_norm, *k_norm;
    void *k_cache, *v_cache;
};

static void free_weights(ColiCuda *ctx, struct Weights *w) {
    coli_cuda_free(ctx, w->q); coli_cuda_free(ctx, w->k);
    coli_cuda_free(ctx, w->v); coli_cuda_free(ctx, w->o);
    coli_cuda_free(ctx, w->qs); coli_cuda_free(ctx, w->ks);
    coli_cuda_free(ctx, w->vs); coli_cuda_free(ctx, w->os);
    coli_cuda_free(ctx, w->q_norm); coli_cuda_free(ctx, w->k_norm);
    coli_cuda_free(ctx, w->k_cache); coli_cuda_free(ctx, w->v_cache);
    memset(w, 0, sizeof(*w));
}

#define OK(call) do { if ((call) != 0) { \
    fprintf(stderr, "%s failed: %s\n", #call, coli_cuda_last_error()); \
    goto fail; } } while (0)

static int upload_q4(ColiCuda *ctx, void **dq, void **ds, int rows, int cols) {
    int ng = cols / GROUP, rb = ng * (GROUP / 2), rc = 0;
    unsigned char *hq = malloc((size_t)rows * rb);
    float *hs = malloc((size_t)rows * ng * sizeof(float));
    if (!hq || !hs) { rc = -1; goto out; }
    fill_q4(hq, hs, rows, cols);
    if (coli_cuda_malloc(ctx, dq, (size_t)rows * rb) ||
        coli_cuda_malloc(ctx, ds, (size_t)rows * ng * sizeof(float)) ||
        coli_cuda_upload(ctx, *dq, hq, (size_t)rows * rb) ||
        coli_cuda_upload(ctx, *ds, hs, (size_t)rows * ng * sizeof(float)))
        rc = -1;
out:
    free(hq); free(hs);
    return rc;
}

static int upload_q8(ColiCuda *ctx, void **dq, void **ds, int rows, int cols) {
    int rc = 0;
    signed char *hq = malloc((size_t)rows * cols);
    float *hs = malloc((size_t)rows * sizeof(float));
    if (!hq || !hs) { rc = -1; goto out; }
    fill_q8(hq, hs, rows, cols);
    if (coli_cuda_malloc(ctx, dq, (size_t)rows * cols) ||
        coli_cuda_malloc(ctx, ds, (size_t)rows * sizeof(float)) ||
        coli_cuda_upload(ctx, *dq, hq, (size_t)rows * cols) ||
        coli_cuda_upload(ctx, *ds, hs, (size_t)rows * sizeof(float)))
        rc = -1;
out:
    free(hq); free(hs);
    return rc;
}

/* One layer of prefill the way the engine does it today: T sequential
 * single-token calls, each ending in a host stream synchronize. */
static double run_per_token(ColiCuda *ctx, struct Weights *w, int T,
                            const float *x, float *out, int *failed_at) {
    int q_ng = HIDDEN / GROUP, q_rb = q_ng * (GROUP / 2);
    int o_ng = CTXN / GROUP;
    double t0 = now_s();
    *failed_at = -1;
    for (int t = 0; t < T; t++) {
        if (coli_cuda_gqa_decode_q4_f16(
                ctx, out + (size_t)t * HIDDEN, x + (size_t)t * HIDDEN,
                (const unsigned char *)w->q, (const float *)w->qs, q_rb, q_ng,
                (const unsigned char *)w->k, (const float *)w->ks, q_rb, q_ng,
                (const unsigned char *)w->v, (const float *)w->vs, q_rb, q_ng,
                w->o, (const float *)w->os, 1, CTXN, o_ng,
                (const float *)w->q_norm, (const float *)w->k_norm,
                (float *)w->k_cache, (float *)w->v_cache, t, HIDDEN,
                QHEADS, KVHEADS, HEAD_DIM, ROTARY, GROUP, 5000000.0f, 1e-6f)) {
            *failed_at = t;
            break;
        }
    }
    return now_s() - t0;
}

/* The batched form: one call for the whole block. */
static double run_batched(ColiCuda *ctx, struct Weights *w, int T,
                          int max_seq, const float *x, float *out, int *bad) {
    int q_ng = HIDDEN / GROUP, q_rb = q_ng * (GROUP / 2);
    int o_ng = CTXN / GROUP;
    double t0 = now_s();
    *bad = coli_cuda_gqa_prefill_q4_f16(
        ctx, out, x, T, 0,
        (const unsigned char *)w->q, (const float *)w->qs, q_rb, q_ng,
        (const unsigned char *)w->k, (const float *)w->ks, q_rb, q_ng,
        (const unsigned char *)w->v, (const float *)w->vs, q_rb, q_ng,
        w->o, (const float *)w->os, 1, CTXN, o_ng,
        (const float *)w->q_norm, (const float *)w->k_norm,
        (float *)w->k_cache, (float *)w->v_cache, max_seq, HIDDEN,
        QHEADS, KVHEADS, HEAD_DIM, ROTARY, GROUP, 5000000.0f, 1e-6f);
    return now_s() - t0;
}

/* Bit-identical is the acceptance bar, so this counts differing floats rather
 * than reporting a tolerance. */
static size_t count_diff(const float *a, const float *b, size_t n,
                         float *max_abs) {
    size_t bad = 0;
    *max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            float d = fabsf(a[i] - b[i]);
            if (d > *max_abs) *max_abs = d;
            bad++;
        }
    }
    return bad;
}

int main(int argc, char **argv) {
    ColiCuda *ctx = NULL;
    struct Weights w;
    memset(&w, 0, sizeof(w));
    float *x = NULL, *ref = NULL, *got = NULL, *norm = NULL;
    float *kref = NULL, *vref = NULL, *kgot = NULL, *vgot = NULL;
    int max_seq = 16384, rc = 1;
    int sweep[8], n_sweep = 0;

    if (argc > 1)
        for (int i = 1; i < argc && n_sweep < 8; i++) {
            int v = atoi(argv[i]);
            if (v > 0) sweep[n_sweep++] = v;
        }
    if (!n_sweep) {
        sweep[n_sweep++] = 512;
        sweep[n_sweep++] = 2335;
        sweep[n_sweep++] = 8451;
        sweep[n_sweep++] = 16384;
    }
    int max_T = 0;
    for (int i = 0; i < n_sweep; i++) if (sweep[i] > max_T) max_T = sweep[i];
    if (max_T > max_seq) max_seq = max_T;

    if (coli_cuda_create(&ctx, 0)) {
        fprintf(stderr, "no CUDA device: %s\n", coli_cuda_last_error());
        return 77;
    }
    int sm = 0, shared = 0, major = 0, minor = 0;
    coli_cuda_device_limits(ctx, &sm, &shared);
    coli_cuda_compute_capability(ctx, &major, &minor);
    printf("device %s  sm_%d%d  multiprocessors=%d  sharedMemPerBlock=%d B\n",
           coli_cuda_device_name(ctx), major, minor, sm, shared);
    printf("shape hidden=%d q_heads=%d kv_heads=%d head_dim=%d rotary=%d "
           "group=%d o_fmt=int8\n",
           HIDDEN, QHEADS, KVHEADS, HEAD_DIM, ROTARY, GROUP);
    printf("per-token path launches <<<%d blocks, %d threads>>> on %d "
           "multiprocessors\n", QHEADS, HEAD_DIM, sm);
    printf("shared-memory ceiling of the per-token path: position+1 > %d\n\n",
           (shared - 128) / 4);

    size_t span = (size_t)max_T * HIDDEN;
    size_t kvspan = (size_t)max_seq * KVROWS;
    norm = malloc((size_t)HEAD_DIM * sizeof(float));
    x = malloc(span * sizeof(float));
    ref = malloc(span * sizeof(float));
    got = malloc(span * sizeof(float));
    kref = malloc(kvspan * sizeof(float));
    vref = malloc(kvspan * sizeof(float));
    kgot = malloc(kvspan * sizeof(float));
    vgot = malloc(kvspan * sizeof(float));
    if (!norm || !x || !ref || !got || !kref || !vref || !kgot || !vgot)
        goto fail;
    for (int i = 0; i < HEAD_DIM; i++) norm[i] = 0.01f * sinf((float)i);
    for (size_t i = 0; i < span; i++)
        x[i] = 0.05f * sinf((float)(i + 1) * 0.031f);

    OK(upload_q4(ctx, &w.q, &w.qs, QROWS, HIDDEN));
    OK(upload_q4(ctx, &w.k, &w.ks, KVROWS, HIDDEN));
    OK(upload_q4(ctx, &w.v, &w.vs, KVROWS, HIDDEN));
    OK(upload_q8(ctx, &w.o, &w.os, HIDDEN, CTXN));
    OK(coli_cuda_malloc(ctx, &w.q_norm, (size_t)HEAD_DIM * sizeof(float)));
    OK(coli_cuda_malloc(ctx, &w.k_norm, (size_t)HEAD_DIM * sizeof(float)));
    OK(coli_cuda_upload(ctx, w.q_norm, norm, (size_t)HEAD_DIM * sizeof(float)));
    OK(coli_cuda_upload(ctx, w.k_norm, norm, (size_t)HEAD_DIM * sizeof(float)));
    OK(coli_cuda_malloc(ctx, &w.k_cache, kvspan * sizeof(float)));
    OK(coli_cuda_malloc(ctx, &w.v_cache, kvspan * sizeof(float)));

    for (int i = 0; i < n_sweep; i++) {
        int T = sweep[i], failed_at = -1, bad = 0;
        size_t out_n = (size_t)T * HIDDEN, kv_n = (size_t)T * KVROWS;

        OK(coli_cuda_memset(ctx, w.k_cache, 0, kvspan * sizeof(float)));
        OK(coli_cuda_memset(ctx, w.v_cache, 0, kvspan * sizeof(float)));
        OK(coli_cuda_sync(ctx));
        double per_token = run_per_token(ctx, &w, T, x, ref, &failed_at);
        if (failed_at < 0) {
            OK(coli_cuda_download(ctx, kref, w.k_cache, kv_n * sizeof(float)));
            OK(coli_cuda_download(ctx, vref, w.v_cache, kv_n * sizeof(float)));
        }

        OK(coli_cuda_memset(ctx, w.k_cache, 0, kvspan * sizeof(float)));
        OK(coli_cuda_memset(ctx, w.v_cache, 0, kvspan * sizeof(float)));
        OK(coli_cuda_sync(ctx));
        double batched = run_batched(ctx, &w, T, max_seq, x, got, &bad);
        if (bad) {
            printf("  T=%-6d batched FAILED: %s\n", T, coli_cuda_last_error());
            continue;
        }
        OK(coli_cuda_download(ctx, kgot, w.k_cache, kv_n * sizeof(float)));
        OK(coli_cuda_download(ctx, vgot, w.v_cache, kv_n * sizeof(float)));

        double pairs = (double)T * ((double)T + 1.0) / 2.0;
        double flop = 4.0 * (double)QHEADS * (double)HEAD_DIM * pairs;

        if (failed_at >= 0) {
            printf("  T=%-6d per-token FAILED at position %d (%s)\n",
                   T, failed_at, coli_cuda_last_error());
            printf("           batched  %8.3f s  %7.3f TFLOP/s  "
                   "15-layer %8.2f s (%5.2f min)  [no reference to compare]\n",
                   batched, flop / batched / 1e12, batched * 15.0,
                   batched * 15.0 / 60.0);
            continue;
        }

        float mo = 0.0f, mk = 0.0f, mv = 0.0f;
        size_t bo = count_diff(ref, got, out_n, &mo);
        size_t bk = count_diff(kref, kgot, kv_n, &mk);
        size_t bv = count_diff(vref, vgot, kv_n, &mv);

        printf("  T=%-6d per-token %8.3f s (%6.3f TFLOP/s)   "
               "batched %8.3f s (%6.3f TFLOP/s)   speedup %6.2fx\n",
               T, per_token, flop / per_token / 1e12,
               batched, flop / batched / 1e12, per_token / batched);
        printf("           15-layer projection %8.2f s -> %8.2f s "
               "(%5.2f min -> %5.2f min)\n",
               per_token * 15.0, batched * 15.0,
               per_token * 15.0 / 60.0, batched * 15.0 / 60.0);
        printf("           exactness: output %s (%zu/%zu differ, max %.3g)  "
               "K %zu differ  V %zu differ\n",
               bo == 0 ? "BIT-IDENTICAL" : "DIVERGED", bo, out_n, (double)mo,
               bk, bv);
        if (bo || bk || bv) rc = 2;
    }
    if (rc != 2) rc = 0;

fail:
    free_weights(ctx, &w);
    free(norm); free(x); free(ref); free(got);
    free(kref); free(vref); free(kgot); free(vgot);
    if (ctx) coli_cuda_destroy(ctx);
    return rc;
}
