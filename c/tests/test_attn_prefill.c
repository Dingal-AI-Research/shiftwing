#define QWEN_NO_MAIN
#include "../qwen.c"

/* attn_prefill_layer must be indistinguishable from the per-token loop it
 * replaces -- not close to it.  The qualification harness binds generated text
 * by SHA-256, so a single changed bit invalidates the bound baseline and every
 * quality gate behind it.  This test therefore compares with memcmp and counts
 * differing floats rather than reporting a tolerance, unlike test_attn_slots
 * whose slot batching is only claimed to agree to 1e-6.
 *
 * Two shapes are covered:
 *   base=0    the prefill call site (forward_prefill_core)
 *   base>0    the speculative target-block verifier (forward_decode_block),
 *             where a divergence would not fail loudly -- it would quietly
 *             collapse the MTP accept rate.
 */

static Layer *full_layer(Model *m) {
    for (int i = 0; i < m->c.n_layers; i++)
        if (m->layer[i].type == LT_FULL) return &m->layer[i];
    return NULL;
}

static void reset_state(Model *m, AttnW *w) {
    int kvrows = m->c.n_kv_heads * m->c.head_dim;
    size_t n = (size_t)m->max_seq * kvrows;
    if (w->k_cache) memset(w->k_cache, 0, n * sizeof(float));
    if (w->v_cache) memset(w->v_cache, 0, n * sizeof(float));
    if (w->k_cache16) memset(w->k_cache16, 0, n * sizeof(uint16_t));
    if (w->v_cache16) memset(w->v_cache16, 0, n * sizeof(uint16_t));
#ifdef COLI_CUDA
    w->cuda_state_pos = 0;
#endif
    m->pos = 0;
}

static size_t differing(const float *a, const float *b, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) bad++;
    return bad;
}

/* Sequential reference: exactly the loop attn_prefill_layer replaced. */
static void sequential(Model *m, Layer *l, const float *x, int rows,
                       float *out) {
    int H = m->c.hidden;
    for (int t = 0; t < rows; t++) {
        m->pos = t;
        attn_forward(m, l, x + (int64_t)t * H, out + (int64_t)t * H);
    }
}

static int compare_case(Model *m, Layer *l, const float *x, int base, int T,
                        const char *label, int expect_cuda) {
    int H = m->c.hidden, rows = base + T;
    float *ref = falloc((int64_t)rows * H);
    float *got = falloc((int64_t)rows * H);

    reset_state(m, &l->attn);
    sequential(m, l, x, rows, ref);

    reset_state(m, &l->attn);
    if (base > 0) sequential(m, l, x, base, got);
    m->pos = base;
#ifdef COLI_CUDA
    unsigned long long before = cuda_rt.attn_prefill_calls;
#endif
    attn_prefill_layer(m, l, x + (int64_t)base * H, T, base,
                       got + (int64_t)base * H);
#ifdef COLI_CUDA
    int engaged = cuda_rt.attn_prefill_calls > before;
#else
    int engaged = 0;
#endif

    size_t span = (size_t)T * H;
    size_t bad = differing(ref + (int64_t)base * H, got + (int64_t)base * H,
                           span);
    /* The block's own rows attend to each other, so a wrong key or value in
     * row i shows up in the output of every row after it.  Comparing the whole
     * block therefore covers the cache without downloading it from the GPU. */
    printf("  %-28s base=%-4d rows=%-4d  %s (%zu/%zu floats differ)  "
           "batched-cuda=%s\n",
           label, base, T, bad ? "DIVERGED" : "bit-identical", bad, span,
           engaged ? "yes" : "no");
    /* A case that was supposed to take the batched path but silently fell back
     * would compare itself against itself and always pass. */
    if (expect_cuda && !engaged) {
        printf("    expected the batched CUDA path, but it did not engage\n");
        bad++;
    }
    if (!expect_cuda && engaged) {
        printf("    expected the per-token fallback, but the batched path ran\n");
        bad++;
    }
    if (m->pos != base + T - 1) {
        printf("    position left at %d, expected %d\n", m->pos,
               base + T - 1);
        bad++;
    }
    free(ref);
    free(got);
    return bad != 0;
}

int main(void) {
#ifdef COLI_CUDA
    /* Without these the engine never starts the backend and every case below
     * would pass through the CPU fallback -- a green test proving nothing. */
    setenv("COLI_CUDA", "1", 1);
    setenv("CUDA_DENSE", "1", 1);
    setenv("CUDA_F16", "1", 1);
#endif
    setenv("MTP", "0", 1);
    setenv("KV16", "0", 1);
    setenv("EXPERT_RAM", "4", 1);
    setenv("PREFETCH_THREADS", "0", 1);
    static Model m;
    model_init(&m, "qwen_tiny_i4");
#ifdef COLI_CUDA
    /* main() does this; QWEN_NO_MAIN builds must start the backend themselves. */
    cuda_backend_start();
#endif
    Layer *l = full_layer(&m);
    if (!l) {
        fprintf(stderr, "tiny model has no full-attention layer\n");
        return 1;
    }
    int H = m.c.hidden, rows = 96;
    if (rows > m.max_seq) rows = m.max_seq;
    float *x = falloc((int64_t)rows * H);
    for (int64_t i = 0; i < (int64_t)rows * H; i++)
        x[i] = sinf((float)(i + 1) * 0.023f) * 0.2f;

#ifdef COLI_CUDA
    printf("attention prefill exactness (CUDA build)\n");
#else
    printf("attention prefill exactness (CPU build)\n");
#endif
    int fail = 0;
#ifdef COLI_CUDA
    int cuda = cuda_rt.active && cuda_rt.use_f16;
    if (!cuda)
        printf("  NOTE: CUDA backend inactive; cases run the CPU fallback\n");
#else
    int cuda = 0;
#endif
    fail |= compare_case(&m, l, x, 0, 64, "prefill block", cuda);
    fail |= compare_case(&m, l, x, 24, 40, "verifier block, mid-sequence", cuda);
    fail |= compare_case(&m, l, x, 0, 8, "short block (below threshold)", 0);
    fail |= compare_case(&m, l, x, 60, 4, "speculative block, 4 rows", 0);
    free(x);
    return fail;
}
