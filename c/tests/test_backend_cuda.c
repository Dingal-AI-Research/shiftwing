#include "../backend_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_OK(call) do { if ((call) != 0) { \
    fprintf(stderr, "%s failed: %s\n", #call, coli_cuda_last_error()); \
    goto fail; \
} } while (0)

static float max_diff(const float *a, const float *b, int n) {
    float out = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > out) out = d;
    }
    return out;
}

static void set_q3(unsigned char *row, int index, unsigned code) {
    int bit = 3 * index, byte = bit >> 3, shift = bit & 7;
    unsigned value = (code & 7u) << shift;
    row[byte] |= (unsigned char)value;
    if (shift > 5) row[byte + 1] |= (unsigned char)(value >> 8);
}

/* Compare the two CUDA representations used by the Ornith397 release path at
 * the model's real routed-expert dimensions.  Earlier tests exercised q3 and
 * q4 independently at small shapes, so a shape-dependent expansion/kernel
 * regression could still pass both suites. */
static int check_ornith_q3_q4_equivalence(ColiCuda *ctx) {
    enum { H = 4096, I = 1024, GS = 128, K = 10 };
    const int hng = H / GS, ing = I / GS;
    const int hrb3 = H * 3 / 8, hrb4 = H / 2;
    const int irb3 = I * 3 / 8, irb4 = I / 2;
    const size_t hidden_q3_bytes = (size_t)I * hrb3;
    const size_t hidden_q4_bytes = (size_t)I * hrb4;
    const size_t down_q3_bytes = (size_t)H * irb3;
    const size_t down_q4_bytes = (size_t)H * irb4;
    uint8_t *hq3 = calloc(hidden_q3_bytes, 1);
    uint8_t *hq4 = calloc(hidden_q4_bytes, 1);
    uint8_t *dq3 = calloc(down_q3_bytes, 1);
    uint8_t *dq4 = calloc(down_q4_bytes, 1);
    float *hs = calloc((size_t)I * hng, sizeof(*hs));
    float *ds = calloc((size_t)H * ing, sizeof(*ds));
    float *x = calloc(H, sizeof(*x));
    float *ref = calloc(H, sizeof(*ref));
    float *got = calloc(H, sizeof(*got));
    void *d_hq3 = NULL, *d_hq4 = NULL, *d_dq3 = NULL, *d_dq4 = NULL;
    void *d_hs = NULL, *d_ds = NULL, *d_x = NULL, *d_x16 = NULL, *d_y = NULL;
    int ok = 0;
    if (!hq3 || !hq4 || !dq3 || !dq4 || !hs || !ds || !x || !ref || !got)
        goto out;
    for (int r = 0; r < I; r++) {
        for (int g = 0; g < hng; g++)
            hs[(size_t)r * hng + g] = ldexpf(1.0f, -10 + ((r + g) & 1));
        for (int i = 0; i < H; i++) {
            unsigned code = (unsigned)((r * 5 + i * 7) & 7);
            set_q3(hq3 + (size_t)r * hrb3, i, code);
            uint8_t *p = hq4 + (size_t)r * hrb4 + i / 2;
            if (i & 1) *p |= (uint8_t)((code + 4) << 4);
            else *p = (uint8_t)(code + 4);
        }
    }
    for (int r = 0; r < H; r++) {
        for (int g = 0; g < ing; g++)
            ds[(size_t)r * ing + g] = ldexpf(1.0f, -10 + ((r + g) & 1));
        for (int i = 0; i < I; i++) {
            unsigned code = (unsigned)((r * 3 + i * 5) & 7);
            set_q3(dq3 + (size_t)r * irb3, i, code);
            uint8_t *p = dq4 + (size_t)r * irb4 + i / 2;
            if (i & 1) *p |= (uint8_t)((code + 4) << 4);
            else *p = (uint8_t)(code + 4);
        }
    }
    for (int i = 0; i < H; i++) x[i] = sinf((float)(i * 13 + 1)) * 0.125f;
#define ORNITH_CUDA(call) do { if ((call) != 0) goto out; } while (0)
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_hq3, hidden_q3_bytes));
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_hq4, hidden_q4_bytes));
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_dq3, down_q3_bytes));
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_dq4, down_q4_bytes));
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_hs, (size_t)I * hng * sizeof(float)));
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_ds, (size_t)H * ing * sizeof(float)));
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_x, (size_t)H * sizeof(float)));
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_x16, (size_t)H * sizeof(uint16_t)));
    ORNITH_CUDA(coli_cuda_malloc(ctx, &d_y, (size_t)H * sizeof(float)));
    ORNITH_CUDA(coli_cuda_upload(ctx, d_hq3, hq3, hidden_q3_bytes));
    ORNITH_CUDA(coli_cuda_upload(ctx, d_hq4, hq4, hidden_q4_bytes));
    ORNITH_CUDA(coli_cuda_upload(ctx, d_dq3, dq3, down_q3_bytes));
    ORNITH_CUDA(coli_cuda_upload(ctx, d_dq4, dq4, down_q4_bytes));
    ORNITH_CUDA(coli_cuda_upload(ctx, d_hs, hs, (size_t)I * hng * sizeof(float)));
    ORNITH_CUDA(coli_cuda_upload(ctx, d_ds, ds, (size_t)H * ing * sizeof(float)));
    ORNITH_CUDA(coli_cuda_upload(ctx, d_x, x, (size_t)H * sizeof(float)));
    ORNITH_CUDA(coli_cuda_f32_to_f16(ctx, d_x16, d_x, H));
    const unsigned char *h3[K], *h4[K], *d3[K], *d4[K];
    const float *hscale[K], *dscale[K];
    float route[K];
    for (int k = 0; k < K; k++) {
        h3[k] = d_hq3; h4[k] = d_hq4; d3[k] = d_dq3; d4[k] = d_dq4;
        hscale[k] = d_hs; dscale[k] = d_ds; route[k] = 1.0f / K;
    }
    ORNITH_CUDA(coli_cuda_grouped_q3_mlp_f16(
        ctx, d_y, d_x16, h3, hscale, h3, hscale, d3, dscale, route, K,
        H, I, GS, hrb3, hng, irb3, ing));
    ORNITH_CUDA(coli_cuda_download(ctx, ref, d_y, (size_t)H * sizeof(float)));
    ORNITH_CUDA(coli_cuda_sync(ctx));
    ORNITH_CUDA(coli_cuda_grouped_q4_mlp_f16(
        ctx, d_y, d_x16, h4, hscale, h4, hscale, d4, dscale, route, K,
        H, I, GS, hrb4, hng, irb4, ing));
    ORNITH_CUDA(coli_cuda_download(ctx, got, d_y, (size_t)H * sizeof(float)));
    ORNITH_CUDA(coli_cuda_sync(ctx));
    {
        float diff = max_diff(ref, got, H);
        printf("Ornith q3/native vs expanded-q4 maxdiff=%.3g\n", diff);
        ok = diff < 2e-5f;
    }
#undef ORNITH_CUDA
out:
    coli_cuda_free(ctx, d_hq3); coli_cuda_free(ctx, d_hq4);
    coli_cuda_free(ctx, d_dq3); coli_cuda_free(ctx, d_dq4);
    coli_cuda_free(ctx, d_hs); coli_cuda_free(ctx, d_ds);
    coli_cuda_free(ctx, d_x); coli_cuda_free(ctx, d_x16); coli_cuda_free(ctx, d_y);
    free(hq3); free(hq4); free(dq3); free(dq4); free(hs); free(ds);
    free(x); free(ref); free(got);
    return ok;
}

int main(void) {
    enum { N = 2053, ROWS = 37, COLS = 1027, GS = 128 };
    const int ng = (COLS + GS - 1) / GS;
    const int rb4 = ng * (GS / 2);
    const int rb3 = ng * (GS * 3 / 8);
    const int rb8 = COLS + 5;
    int ok = 0, major = 0, minor = 0;
    ColiCuda *ctx = NULL;
    float *x = calloc(N, sizeof(*x));
    float *w = calloc(N, sizeof(*w));
    float *up = calloc(N, sizeof(*up));
    float *ref = calloc(N > ROWS ? N : ROWS, sizeof(*ref));
    float *got = calloc(N > ROWS ? N : ROWS, sizeof(*got));
    int8_t *q8 = calloc((size_t)ROWS * rb8, 1);
    uint8_t *q4 = calloc((size_t)ROWS * rb4, 1);
    uint8_t *q3 = calloc((size_t)ROWS * rb3, 1);
    float *s8 = calloc(ROWS, sizeof(*s8));
    float *s4 = calloc((size_t)ROWS * ng, sizeof(*s4));
    void *pinned = NULL, *dx = NULL, *dw = NULL, *dup = NULL, *dy = NULL;
    void *dx16 = NULL, *dq8 = NULL, *dq4 = NULL, *dq3 = NULL;
    void *ds8 = NULL, *ds4 = NULL;
    void *dqdown = NULL, *dq3down = NULL, *dsdown = NULL;
    void *dq8down = NULL, *ds8down = NULL;
    void *dbx = NULL, *dby = NULL, *dfw = NULL, *dfg = NULL, *dfs = NULL;
    void *dshared_scale = NULL;
    if (!x || !w || !up || !ref || !got || !q8 || !q4 || !q3 || !s8 || !s4) {
        fprintf(stderr, "host allocation failed\n");
        goto fail;
    }
    setenv("CUDA_PROFILE_STAGES", "1", 1);
    CUDA_OK(coli_cuda_create(&ctx, 0));
    CUDA_OK(coli_cuda_compute_capability(ctx, &major, &minor));
    printf("CUDA device: %s (sm_%d%d)\n", coli_cuda_device_name(ctx), major, minor);
    CUDA_OK(coli_cuda_malloc_host(ctx, &pinned, 4096));
    memset(pinned, 0x5a, 4096);

    CUDA_OK(coli_cuda_malloc(ctx, &dx, N * sizeof(float)));
    CUDA_OK(coli_cuda_malloc(ctx, &dx16, N * sizeof(uint16_t)));
    CUDA_OK(coli_cuda_malloc(ctx, &dw, N * sizeof(float)));
    CUDA_OK(coli_cuda_malloc(ctx, &dup, N * sizeof(float)));
    CUDA_OK(coli_cuda_malloc(ctx, &dy, (N > ROWS ? N : ROWS) * sizeof(float)));
    for (int i = 0; i < N; i++) {
        x[i] = sinf((float)(i * 13 + 1)) * 0.125f;
        w[i] = cosf((float)(i * 7 + 2)) * 0.0625f;
        up[i] = sinf((float)(i * 17 + 4)) * 0.25f;
    }

    CUDA_OK(coli_cuda_upload(ctx, dx, x, N * sizeof(float)));
    CUDA_OK(coli_cuda_upload(ctx, dw, w, N * sizeof(float)));
    CUDA_OK(coli_cuda_upload(ctx, dup, up, N * sizeof(float)));
    CUDA_OK(coli_cuda_rmsnorm_zero(ctx, (float *)dy, (float *)dx,
                                   (float *)dw, N, 1e-6f));
    CUDA_OK(coli_cuda_download(ctx, got, dy, N * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    float ms = 0.0f;
    for (int i = 0; i < N; i++) ms += x[i] * x[i];
    float inv = 1.0f / sqrtf(ms / N + 1e-6f);
    for (int i = 0; i < N; i++) ref[i] = x[i] * inv * (1.0f + w[i]);
    float rms_diff = max_diff(ref, got, N);

    enum { NB = 3, FROWS = 5 };
    float bx[NB * N], bref[NB * N], bgot[NB * N];
    float fw[FROWS * N], fg_ref[NB * FROWS], fg_got[NB * FROWS];
    float fs_ref[NB * FROWS], fs_got[NB * FROWS];
    for (int b = 0; b < NB; b++) {
        float ss = 0.0f;
        for (int i = 0; i < N; i++) {
            bx[b * N + i] = x[i] + 0.125f * (float)b;
            ss += bx[b * N + i] * bx[b * N + i];
        }
        float binv = 1.0f / sqrtf(ss / N + 1e-6f);
        for (int i = 0; i < N; i++)
            bref[b * N + i] =
                bx[b * N + i] * binv * (1.0f + w[i]);
    }
    for (int r = 0; r < FROWS; r++)
        for (int i = 0; i < N; i++)
            fw[r * N + i] = 0.03125f * (float)(r - (i % 11));
    for (int b = 0; b < NB; b++)
        for (int r = 0; r < FROWS; r++) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++)
                sum += bx[b * N + i] * fw[r * N + i];
            fg_ref[b * FROWS + r] = sum;
            fs_ref[b * FROWS + r] = 1.0f / (1.0f + expf(-sum));
        }
    CUDA_OK(coli_cuda_malloc(ctx, &dbx, sizeof(bx)));
    CUDA_OK(coli_cuda_malloc(ctx, &dby, sizeof(bgot)));
    CUDA_OK(coli_cuda_malloc(ctx, &dfw, sizeof(fw)));
    CUDA_OK(coli_cuda_malloc(ctx, &dfg, sizeof(fg_got)));
    CUDA_OK(coli_cuda_malloc(ctx, &dfs, sizeof(fs_got)));
    CUDA_OK(coli_cuda_upload(ctx, dbx, bx, sizeof(bx)));
    CUDA_OK(coli_cuda_upload(ctx, dfw, fw, sizeof(fw)));
    CUDA_OK(coli_cuda_rmsnorm_zero_batch(
        ctx, (float *)dby, (float *)dbx, (float *)dw, NB, N, 1e-6f));
    CUDA_OK(coli_cuda_f32_gemm(
        ctx, (float *)dfg, (float *)dbx, (float *)dfw, NB, FROWS, N));
    CUDA_OK(coli_cuda_sigmoid(
        ctx, (float *)dfs, (float *)dfg, NB * FROWS));
    CUDA_OK(coli_cuda_download(ctx, bgot, dby, sizeof(bgot)));
    CUDA_OK(coli_cuda_download(ctx, fg_got, dfg, sizeof(fg_got)));
    CUDA_OK(coli_cuda_download(ctx, fs_got, dfs, sizeof(fs_got)));
    CUDA_OK(coli_cuda_sync(ctx));
    float batch_rms_diff = max_diff(bref, bgot, NB * N);
    float f32_gemm_diff = max_diff(fg_ref, fg_got, NB * FROWS);
    float sigmoid_diff = max_diff(fs_ref, fs_got, NB * FROWS);

    CUDA_OK(coli_cuda_silu_mul(ctx, (float *)dy, (float *)dw, (float *)dup, N));
    CUDA_OK(coli_cuda_download(ctx, got, dy, N * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    for (int i = 0; i < N; i++) ref[i] = w[i] / (1.0f + expf(-w[i])) * up[i];
    float silu_diff = max_diff(ref, got, N);
    CUDA_OK(coli_cuda_axpy(ctx, (float *)dy, (const float *)dup, 0.375f, N));
    CUDA_OK(coli_cuda_download(ctx, got, dy, N * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    for (int i = 0; i < N; i++) ref[i] += 0.375f * up[i];
    float axpy_diff = max_diff(ref, got, N);

    coli_cuda_free(ctx, dx); dx = NULL;
    CUDA_OK(coli_cuda_malloc(ctx, &dx, COLS * sizeof(float)));
    CUDA_OK(coli_cuda_upload(ctx, dx, x, COLS * sizeof(float)));
    for (int r = 0; r < ROWS; r++) {
        s8[r] = ldexpf(1.0f, (r % 7) - 9);
        for (int i = 0; i < COLS; i++)
            q8[(size_t)r * rb8 + i] = (int8_t)(((r * 29 + i * 11) % 255) - 127);
        for (int g = 0; g < ng; g++) s4[(size_t)r * ng + g] = ldexpf(1.0f, ((r + g) % 7) - 7);
        for (int i = 0; i < rb4 * 2; i++) {
            int q = (r * 3 + i * 13) % 16;
            uint8_t *p = &q4[(size_t)r * rb4 + i / 2];
            if (i & 1) *p |= (uint8_t)(q << 4); else *p = (uint8_t)q;
        }
        for (int i = 0; i < ng * GS; i++)
            set_q3(q3 + (size_t)r * rb3, i,
                   (unsigned)((r * 3 + i * 13) % 8));
    }
    CUDA_OK(coli_cuda_malloc(ctx, &dq8, (size_t)ROWS * rb8));
    CUDA_OK(coli_cuda_malloc(ctx, &dq4, (size_t)ROWS * rb4));
    CUDA_OK(coli_cuda_malloc(ctx, &dq3, (size_t)ROWS * rb3));
    CUDA_OK(coli_cuda_malloc(ctx, &ds8, ROWS * sizeof(float)));
    CUDA_OK(coli_cuda_malloc(ctx, &ds4, (size_t)ROWS * ng * sizeof(float)));
    CUDA_OK(coli_cuda_upload(ctx, dq8, q8, (size_t)ROWS * rb8));
    CUDA_OK(coli_cuda_upload(ctx, dq4, q4, (size_t)ROWS * rb4));
    CUDA_OK(coli_cuda_upload(ctx, dq3, q3, (size_t)ROWS * rb3));
    CUDA_OK(coli_cuda_upload(ctx, ds8, s8, ROWS * sizeof(float)));
    CUDA_OK(coli_cuda_upload(ctx, ds4, s4, (size_t)ROWS * ng * sizeof(float)));

    CUDA_OK(coli_cuda_q8_gemv(ctx, (float *)dy, (float *)dx,
                              (const signed char *)dq8, (float *)ds8,
                              ROWS, COLS, rb8));
    CUDA_OK(coli_cuda_download(ctx, got, dy, ROWS * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    for (int r = 0; r < ROWS; r++) {
        float sum = 0.0f;
        for (int i = 0; i < COLS; i++) sum += x[i] * q8[(size_t)r * rb8 + i];
        ref[r] = sum * s8[r];
    }
    float q8_diff = max_diff(ref, got, ROWS);

    CUDA_OK(coli_cuda_q4_gemv(ctx, (float *)dy, (float *)dx,
                              (const unsigned char *)dq4, (float *)ds4,
                              ROWS, COLS, GS, rb4, ng));
    CUDA_OK(coli_cuda_download(ctx, got, dy, ROWS * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    for (int r = 0; r < ROWS; r++) {
        float sum = 0.0f;
        for (int i = 0; i < COLS; i++) {
            uint8_t p = q4[(size_t)r * rb4 + i / 2];
            int q = ((i & 1) ? p >> 4 : p & 15) - 8;
            sum += x[i] * q * s4[(size_t)r * ng + i / GS];
        }
        ref[r] = sum;
    }
    float q4_diff = max_diff(ref, got, ROWS);

    CUDA_OK(coli_cuda_f32_to_f16(ctx, (unsigned short *)dx16,
                                 (const float *)dx, COLS));
    CUDA_OK(coli_cuda_q4_gemv_f16(ctx, (float *)dy,
                                  (const unsigned short *)dx16,
                                  (const unsigned char *)dq4, (float *)ds4,
                                  ROWS, COLS, GS, rb4, ng));
    CUDA_OK(coli_cuda_download(ctx, got, dy, ROWS * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    float q4_f16_diff = max_diff(ref, got, ROWS);

    CUDA_OK(coli_cuda_q3_gemv(ctx, (float *)dy, (float *)dx,
                              (const unsigned char *)dq3, (float *)ds4,
                              ROWS, COLS, GS, rb3, ng));
    CUDA_OK(coli_cuda_download(ctx, got, dy, ROWS * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    for (int r = 0; r < ROWS; r++) {
        float sum = 0.0f;
        for (int i = 0; i < COLS; i++) {
            int bit = 3 * i, byte = bit >> 3, shift = bit & 7;
            unsigned word = q3[(size_t)r * rb3 + byte];
            if (shift > 5)
                word |= (unsigned)q3[(size_t)r * rb3 + byte + 1] << 8;
            int quant = (int)((word >> shift) & 7u) - 4;
            sum += x[i] * quant * s4[(size_t)r * ng + i / GS];
        }
        ref[r] = sum;
    }
    float q3_diff = max_diff(ref, got, ROWS);
    CUDA_OK(coli_cuda_q3_gemv_f16(
        ctx, (float *)dy, (const unsigned short *)dx16,
        (const unsigned char *)dq3, (float *)ds4, ROWS, COLS, GS, rb3, ng));
    CUDA_OK(coli_cuda_download(ctx, got, dy, ROWS * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    float q3_f16_diff = max_diff(ref, got, ROWS);

    /* The speculative verifier uses the same grouped/shared expert equations
     * over 2--8 activations. Compare its two-token transaction directly with
     * the established single-token kernels, including fp16 activation and
     * SwiGLU rounding. */
    enum { BATCH = 2, TOPK = 2, BH = 1026, BI = ROWS };
    const int down_ng = (BI + GS - 1) / GS;
    const int down_rb = down_ng * (GS / 2);
    const int down_rb3 = down_ng * (GS * 3 / 8);
    uint8_t *qdown = calloc((size_t)BH * down_rb, 1);
    uint8_t *q3down = calloc((size_t)BH * down_rb3, 1);
    float *sdown = calloc((size_t)BH * down_ng, sizeof(float));
    const int down_rb8 = BI + 3;
    int8_t *q8down = calloc((size_t)BH * down_rb8, 1);
    float *s8down = calloc(BH, sizeof(float));
    if (!qdown || !q3down || !sdown || !q8down || !s8down) {
        fprintf(stderr, "batch host allocation failed\n");
        free(qdown); free(q3down); free(sdown); free(q8down); free(s8down);
        goto fail;
    }
    for (int r = 0; r < BH; r++) {
        sdown[(size_t)r * down_ng] = ldexpf(1.0f, (r % 5) - 8);
        s8down[r] = ldexpf(1.0f, (r % 7) - 9);
        for (int i = 0; i < BI; i++)
            q8down[(size_t)r * down_rb8 + i] =
                (int8_t)(((r * 17 + i * 23) % 255) - 127);
        for (int i = 0; i < down_rb * 2; i++) {
            int q = (r * 5 + i * 7) % 16;
            uint8_t *p = &qdown[(size_t)r * down_rb + i / 2];
            if (i & 1) *p |= (uint8_t)(q << 4); else *p = (uint8_t)q;
        }
        for (int i = 0; i < down_ng * GS; i++)
            set_q3(q3down + (size_t)r * down_rb3, i,
                   (unsigned)((r * 5 + i * 7) % 8));
    }
    CUDA_OK(coli_cuda_malloc(ctx, &dqdown, (size_t)BH * down_rb));
    CUDA_OK(coli_cuda_malloc(ctx, &dq3down, (size_t)BH * down_rb3));
    CUDA_OK(coli_cuda_malloc(ctx, &dsdown,
                             (size_t)BH * down_ng * sizeof(float)));
    CUDA_OK(coli_cuda_malloc(ctx, &dq8down, (size_t)BH * down_rb8));
    CUDA_OK(coli_cuda_malloc(ctx, &ds8down, (size_t)BH * sizeof(float)));
    CUDA_OK(coli_cuda_upload(ctx, dqdown, qdown, (size_t)BH * down_rb));
    CUDA_OK(coli_cuda_upload(ctx, dq3down, q3down,
                             (size_t)BH * down_rb3));
    CUDA_OK(coli_cuda_upload(ctx, dsdown, sdown,
                             (size_t)BH * down_ng * sizeof(float)));
    CUDA_OK(coli_cuda_upload(ctx, dq8down, q8down,
                             (size_t)BH * down_rb8));
    CUDA_OK(coli_cuda_upload(ctx, ds8down, s8down,
                             (size_t)BH * sizeof(float)));
    for (int b = 0; b < BATCH; b++)
        for (int i = 0; i < BH; i++)
            x[(size_t)b * BH + i] =
                (float)(((b + 1) * 11 + i * 3) % 31 - 15) / 64.0f;
    coli_cuda_free(ctx, dx); dx = NULL;
    CUDA_OK(coli_cuda_malloc(ctx, &dx, BATCH * BH * sizeof(float)));
    CUDA_OK(coli_cuda_upload(ctx, dx, x, BATCH * BH * sizeof(float)));
    CUDA_OK(coli_cuda_f32_to_f16(ctx, (unsigned short *)dx16,
                                 (const float *)dx, BATCH * BH));
    const unsigned char *gq_one[TOPK] = {
        (const unsigned char *)dq4, (const unsigned char *)dq4};
    const float *gs_one[TOPK] = {(const float *)ds4, (const float *)ds4};
    const unsigned char *uq_one[TOPK] = {
        (const unsigned char *)dq4, (const unsigned char *)dq4};
    const float *us_one[TOPK] = {(const float *)ds4, (const float *)ds4};
    const unsigned char *dq_one[TOPK] = {
        (const unsigned char *)dqdown, (const unsigned char *)dqdown};
    const float *ds_one[TOPK] = {
        (const float *)dsdown, (const float *)dsdown};
    float route_weight[BATCH * TOPK] = {0.625f, 0.375f, 0.25f, 0.75f};
    for (int b = 0; b < BATCH; b++) {
        CUDA_OK(coli_cuda_grouped_q4_mlp_f16(
            ctx, (float *)dy, (const unsigned short *)dx16 + (size_t)b * BH,
            gq_one, gs_one, uq_one, us_one, dq_one, ds_one,
            route_weight + b * TOPK, TOPK, BH, BI, GS, rb4, ng, down_rb,
            down_ng));
        CUDA_OK(coli_cuda_shared_q4_mlp_f16(
            ctx, (float *)dy, (const unsigned short *)dx16 + (size_t)b * BH,
            (const unsigned char *)dq4, (const float *)ds4,
            (const unsigned char *)dq4, (const float *)ds4,
            (const unsigned char *)dqdown, (const float *)dsdown,
            (const unsigned char *)dq4, (const float *)ds4, BH, BI, GS, rb4,
            ng, down_rb, down_ng, rb4, ng));
        CUDA_OK(coli_cuda_download(ctx, ref + (size_t)b * BH, dy,
                                   BH * sizeof(float)));
    }
    CUDA_OK(coli_cuda_sync(ctx));
    coli_cuda_profile_reset(ctx);
    const unsigned char *gq_batch[BATCH * TOPK] = {
        gq_one[0], gq_one[1], gq_one[0], gq_one[1]};
    const float *gs_batch[BATCH * TOPK] = {
        gs_one[0], gs_one[1], gs_one[0], gs_one[1]};
    const unsigned char *uq_batch[BATCH * TOPK] = {
        uq_one[0], uq_one[1], uq_one[0], uq_one[1]};
    const float *us_batch[BATCH * TOPK] = {
        us_one[0], us_one[1], us_one[0], us_one[1]};
    const unsigned char *dq_batch[BATCH * TOPK] = {
        dq_one[0], dq_one[1], dq_one[0], dq_one[1]};
    const float *ds_batch[BATCH * TOPK] = {
        ds_one[0], ds_one[1], ds_one[0], ds_one[1]};
    CUDA_OK(coli_cuda_grouped_q4_mlp_batch_f16(
        ctx, (float *)dy, (const unsigned short *)dx16, gq_batch, gs_batch,
        uq_batch, us_batch, dq_batch, ds_batch, route_weight, BATCH, TOPK, BH,
        BI, GS, rb4, ng, down_rb, down_ng));
    CUDA_OK(coli_cuda_shared_q4_mlp_batch_f16(
        ctx, (float *)dy, (const unsigned short *)dx16,
        (const unsigned char *)dq4, (const float *)ds4,
        (const unsigned char *)dq4, (const float *)ds4,
        (const unsigned char *)dqdown, (const float *)dsdown,
        (const unsigned char *)dq4, (const float *)ds4, BATCH, BH, BI, GS,
        rb4, ng, down_rb, down_ng, rb4, ng));
    CUDA_OK(coli_cuda_download(ctx, got, dy, BATCH * BH * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    unsigned long long cuda_transactions = 0;
    double cuda_stage_ms[8] = {0};
    CUDA_OK(coli_cuda_profile_snapshot(
        ctx, &cuda_transactions, cuda_stage_ms));
    float batch_moe_diff = max_diff(ref, got, BATCH * BH);

    const unsigned char *gq3_one[TOPK] = {
        (const unsigned char *)dq3, (const unsigned char *)dq3};
    const unsigned char *uq3_one[TOPK] = {
        (const unsigned char *)dq3, (const unsigned char *)dq3};
    const unsigned char *dq3_one[TOPK] = {
        (const unsigned char *)dq3down, (const unsigned char *)dq3down};
    for (int b = 0; b < BATCH; b++) {
        CUDA_OK(coli_cuda_grouped_q3_mlp_f16(
            ctx, (float *)dy, (const unsigned short *)dx16 + (size_t)b * BH,
            gq3_one, gs_one, uq3_one, us_one, dq3_one, ds_one,
            route_weight + b * TOPK, TOPK, BH, BI, GS, rb3, ng, down_rb3,
            down_ng));
        CUDA_OK(coli_cuda_download(ctx, ref + (size_t)b * BH, dy,
                                   BH * sizeof(float)));
    }
    CUDA_OK(coli_cuda_sync(ctx));
    const unsigned char *gq3_batch[BATCH * TOPK] = {
        gq3_one[0], gq3_one[1], gq3_one[0], gq3_one[1]};
    const unsigned char *uq3_batch[BATCH * TOPK] = {
        uq3_one[0], uq3_one[1], uq3_one[0], uq3_one[1]};
    const unsigned char *dq3_batch[BATCH * TOPK] = {
        dq3_one[0], dq3_one[1], dq3_one[0], dq3_one[1]};
    CUDA_OK(coli_cuda_grouped_q3_mlp_batch_f16(
        ctx, (float *)dy, (const unsigned short *)dx16, gq3_batch, gs_batch,
        uq3_batch, us_batch, dq3_batch, ds_batch, route_weight, BATCH, TOPK,
        BH, BI, GS, rb3, ng, down_rb3, down_ng));
    CUDA_OK(coli_cuda_download(ctx, got, dy, BATCH * BH * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    float batch_q3_moe_diff = max_diff(ref, got, BATCH * BH);
    float shared_scale[BATCH] = {0.25f, 0.75f};
    for (int b = 0; b < BATCH; b++) {
        const float *xb = (const float *)dx + (size_t)b * BH;
        CUDA_OK(coli_cuda_q8_gemv(
            ctx, (float *)dw, xb, (const signed char *)dq8,
            (const float *)ds8, BI, BH, rb8));
        CUDA_OK(coli_cuda_q8_gemv(
            ctx, (float *)dup, xb, (const signed char *)dq8,
            (const float *)ds8, BI, BH, rb8));
        CUDA_OK(coli_cuda_silu_mul(
            ctx, (float *)dw, (const float *)dw, (const float *)dup, BI));
        CUDA_OK(coli_cuda_q8_gemv(
            ctx, (float *)dy, (const float *)dw,
            (const signed char *)dq8down, (const float *)ds8down, BH, BI,
            down_rb8));
        CUDA_OK(coli_cuda_download(
            ctx, ref + (size_t)b * BH, dy, BH * sizeof(float)));
    }
    CUDA_OK(coli_cuda_sync(ctx));
    for (int b = 0; b < BATCH; b++)
        for (int h = 0; h < BH; h++)
            ref[(size_t)b * BH + h] *= shared_scale[b];
    CUDA_OK(coli_cuda_memset(ctx, dy, 0, BATCH * BH * sizeof(float)));
    CUDA_OK(coli_cuda_shared_q8_mlp_batch(
        ctx, (float *)dy, (const float *)dx, shared_scale,
        (const signed char *)dq8, (const float *)ds8, rb8,
        (const signed char *)dq8, (const float *)ds8, rb8,
        (const signed char *)dq8down, (const float *)ds8down, down_rb8,
        BATCH, BH, BI));
    CUDA_OK(coli_cuda_download(ctx, got, dy, BATCH * BH * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    float batch_q8_shared_diff = max_diff(ref, got, BATCH * BH);
    CUDA_OK(coli_cuda_malloc(ctx, &dshared_scale, sizeof(shared_scale)));
    CUDA_OK(coli_cuda_upload(ctx, dshared_scale, shared_scale,
                             sizeof(shared_scale)));
    CUDA_OK(coli_cuda_memset(ctx, dy, 0, BATCH * BH * sizeof(float)));
    CUDA_OK(coli_cuda_shared_q8_mlp_batch_device_scale(
        ctx, (float *)dy, (const float *)dx, (const float *)dshared_scale,
        (const signed char *)dq8, (const float *)ds8, rb8,
        (const signed char *)dq8, (const float *)ds8, rb8,
        (const signed char *)dq8down, (const float *)ds8down, down_rb8,
        BATCH, BH, BI));
    CUDA_OK(coli_cuda_download(ctx, got, dy, BATCH * BH * sizeof(float)));
    CUDA_OK(coli_cuda_sync(ctx));
    float batch_q8_device_scale_diff = max_diff(ref, got, BATCH * BH);
    free(qdown); free(q3down); free(sdown); free(q8down); free(s8down);
    printf("rmsnorm maxdiff=%.3g batch-rmsnorm=%.3g f32-gemm=%.3g "
           "sigmoid=%.3g silu maxdiff=%.3g axpy maxdiff=%.3g "
           "q8 maxdiff=%.3g q4 maxdiff=%.3g q4f16 maxdiff=%.3g "
           "q3 maxdiff=%.3g q3f16 maxdiff=%.3g "
           "batch-moe maxdiff=%.3g batch-q3-moe maxdiff=%.3g "
           "batch-q8-shared maxdiff=%.3g "
           "batch-q8-device-scale maxdiff=%.3g\n",
           rms_diff, batch_rms_diff, f32_gemm_diff, sigmoid_diff,
           silu_diff, axpy_diff, q8_diff, q4_diff, q4_f16_diff,
           q3_diff, q3_f16_diff, batch_moe_diff, batch_q3_moe_diff,
           batch_q8_shared_diff,
           batch_q8_device_scale_diff);
    int ornith_q3_q4_ok = check_ornith_q3_q4_equivalence(ctx);
    ok = rms_diff < 2e-6f && batch_rms_diff < 2e-6f &&
         f32_gemm_diff < 2e-4f && sigmoid_diff < 2e-6f &&
         silu_diff < 1e-7f && axpy_diff < 1e-7f && q8_diff < 2e-4f &&
         q4_diff < 2e-4f && q4_f16_diff < 4e-3f &&
         q3_diff < 2e-4f && q3_f16_diff < 4e-3f &&
         batch_moe_diff < 1e-6f && batch_q3_moe_diff < 1e-6f &&
         batch_q8_shared_diff < 1e-6f &&
         batch_q8_device_scale_diff < 1e-6f &&
         cuda_transactions == 1 && cuda_stage_ms[0] >= 0.0 &&
         cuda_stage_ms[7] >= 0.0 && ornith_q3_q4_ok;

fail:
    coli_cuda_free(ctx, dx); coli_cuda_free(ctx, dw); coli_cuda_free(ctx, dup);
    coli_cuda_free(ctx, dy); coli_cuda_free(ctx, dx16);
    coli_cuda_free(ctx, dq8); coli_cuda_free(ctx, dq4); coli_cuda_free(ctx, dq3);
    coli_cuda_free(ctx, ds8); coli_cuda_free(ctx, ds4);
    coli_cuda_free(ctx, dqdown); coli_cuda_free(ctx, dq3down);
    coli_cuda_free(ctx, dsdown);
    coli_cuda_free(ctx, dq8down); coli_cuda_free(ctx, ds8down);
    coli_cuda_free(ctx, dbx); coli_cuda_free(ctx, dby);
    coli_cuda_free(ctx, dfw); coli_cuda_free(ctx, dfg);
    coli_cuda_free(ctx, dfs);
    coli_cuda_free(ctx, dshared_scale);
    coli_cuda_free_host(ctx, pinned);
    coli_cuda_destroy(ctx);
    free(x); free(w); free(up); free(ref); free(got); free(q8); free(q4); free(q3);
    free(s8); free(s4);
    return ok ? 0 : 1;
}
