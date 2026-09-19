/* Included after Mat, rms, swiglu_clamped and mlp3. CPU code remains available
 * in every build. GPU handles belong to the backend, never to expert slots. */
static int g_batch_prefill = 0;
#ifdef GLM53_CUDA_BACKEND
static GlmCuda *g_glm_cuda = NULL;
static int g_cuda_warned = 0;
static GlmCudaMatrix gpu_matrix(const Mat *m, int resident) {
    GlmCudaMatrix d = { m->fmt, m->rows, m->columns, m->gs,
        m->fmt == 4 ? (const void *)m->q4 : m->fmt == 1 ? (const void *)m->q8 : (const void *)m->f,
        m->s, resident ? &((Mat *)m)->gpu : NULL };
    return d;
}
static void gpu_fallback_warning(void) {
    if (!g_cuda_warned) {
        fprintf(stderr, "GLM53 CUDA: operation fell back to CPU (%s)\n", glm_cuda_error());
        g_cuda_warned = 1;
    }
}
#endif
static void gpu_init(void) {
#ifdef GLM53_CUDA_BACKEND
    const char *enable = getenv("GLM53_CUDA");
    if (!enable || atoi(enable) != 0) {
        const char *setting = getenv("GLM53_CUDA_MB");
        long mb = setting ? strtol(setting, NULL, 10) : 12288;
        const char *device = getenv("GLM53_CUDA_DEVICE");
        if (mb > 0 && (unsigned long)mb <= SIZE_MAX / (1024 * 1024))
            g_glm_cuda = glm_cuda_create(device ? atoi(device) : 0, (size_t)mb * 1024 * 1024);
        if (g_glm_cuda) {
            GlmCudaStats s; glm_cuda_stats(g_glm_cuda, &s);
            fprintf(stderr, "GLM53 CUDA: batched FP32, budget %.0f MiB, compact resident weights + reusable expert buffers\n",
                    (double)s.budget_bytes / (1024 * 1024));
        } else fprintf(stderr, "GLM53 CUDA: unavailable (%s); using CPU\n", glm_cuda_error());
    }
    g_batch_prefill = g_glm_cuda != NULL;
#endif
    const char *batch = getenv("GLM53_BATCH_MATH");
    if (batch) g_batch_prefill = atoi(batch) != 0;
}
static void gpu_release(void) {
#ifdef GLM53_CUDA_BACKEND
    if (g_glm_cuda) {
        GlmCudaStats s; glm_cuda_stats(g_glm_cuda, &s);
        fprintf(stderr, "GLM53 CUDA stats: calls=%llu batched=%llu mlp=%llu max_batch=%d uploaded_mb=%.3f resident_mb=%.3f peak_mb=%.3f budget_mb=%.3f fallbacks=%llu\n",
                s.calls, s.batched_calls, s.mlp_calls, s.max_batch,
                (double)s.uploaded_bytes / 1e6, (double)s.resident_bytes / 1e6,
                (double)s.peak_bytes / 1e6, (double)s.budget_bytes / 1e6, s.fallbacks);
        glm_cuda_destroy(g_glm_cuda); g_glm_cuda = NULL;
    }
    g_cuda_warned = 0;
#endif
    g_batch_prefill = 0;
}
static float *batch_floats(size_t count) {
    if (count > SIZE_MAX / sizeof(float)) { fprintf(stderr, "GLM batch size overflow\n"); exit(1); }
    float *p = malloc(count * sizeof(float));
    if (!p) { fprintf(stderr, "OOM in GLM batched projections\n"); exit(1); }
    return p;
}
static void cpu_mm_rows(float *out, const Mat *w, const float *x, int batch, int first, int rows) {
    if (w->fmt == 4)
        matmul_i4_grouped(out, x, w->q4 + (size_t)first * ((w->columns + 1) / 2),
            w->s + (size_t)first * ((w->columns + w->gs - 1) / w->gs), batch, w->columns, rows, w->gs);
    else if (w->fmt == 1)
        matmul_q(out, x, w->q8 + (size_t)first * w->columns, w->s + first, batch, w->columns, rows);
    else matmul(out, x, w->f + (size_t)first * w->columns, batch, w->columns, rows);
}
static void mm_rows(float *out, const Mat *w, const float *x, int batch, int first, int rows) {
#ifdef GLM53_CUDA_BACKEND
    if (g_glm_cuda && batch > 1) {
        GlmCudaMatrix d = gpu_matrix(w, 1);
        if (glm_cuda_mm(g_glm_cuda, out, x, batch, &d, first, rows)) return;
        gpu_fallback_warning();
    }
#endif
    cpu_mm_rows(out, w, x, batch, first, rows);
}
static void mm(float *out, const Mat *w, const float *x, int batch) {
    mm_rows(out, w, x, batch, 0, w->rows);
}
static void mlp3_batch(float *out, const float *x, int batch, const Mat *g, const Mat *u,
                        const Mat *d, float limit, int resident) {
#ifdef GLM53_CUDA_BACKEND
    if (g_glm_cuda && batch > 1) {
        GlmCudaMatrix mg = gpu_matrix(g, resident), mu = gpu_matrix(u, resident), md = gpu_matrix(d, resident);
        if (glm_cuda_mlp(g_glm_cuda, out, x, batch, &mg, &mu, &md, limit)) return;
        gpu_fallback_warning();
    }
#endif
    /* CPU fallback also batches tokens. No transient expert acquires a
     * resident handle for a host slot that can be overwritten. */
    float *sg = batch_floats((size_t)batch * g->rows);
    float *su = batch_floats((size_t)batch * u->rows);
    cpu_mm_rows(sg, g, x, batch, 0, g->rows);
    cpu_mm_rows(su, u, x, batch, 0, u->rows);
    for (int t = 0; t < batch; t++)
        swiglu_clamped(sg + (size_t)t * g->rows, su + (size_t)t * u->rows, g->rows, limit);
    cpu_mm_rows(out, d, sg, batch, 0, d->rows);
    free(su); free(sg);
}
/* Each head has its own matrix row range. Gather all tokens for a head and
 * scatter the result back; matrix weights are uploaded once for all heads. */
static void mm_heads(float *out, const Mat *w, const float *x, int batch, int heads, int in, int width) {
    float *a = batch_floats((size_t)batch * in), *b = batch_floats((size_t)batch * width);
    for (int h = 0; h < heads; h++) {
        for (int t = 0; t < batch; t++)
            memcpy(a + (size_t)t * in, x + ((size_t)t * heads + h) * in, (size_t)in * sizeof(float));
        mm_rows(b, w, a, batch, h * width, width);
        for (int t = 0; t < batch; t++)
            memcpy(out + ((size_t)t * heads + h) * width, b + (size_t)t * width, (size_t)width * sizeof(float));
    }
    free(b); free(a);
}
static void kda_layer_batch(const Cfg *c, const GLayer *l, const float *x, int tokens,
                           float *out, float *state, float *window, float *scratch) {
    const int P = c->kda_proj, H = c->kda_heads, D = c->kda_hd;
    float *q = batch_floats((size_t)tokens * P), *k = batch_floats((size_t)tokens * P);
    float *v = batch_floats((size_t)tokens * P), *decay = batch_floats((size_t)tokens * P);
    float *gate = batch_floats((size_t)tokens * P), *beta = batch_floats((size_t)tokens * H);
    float *low = batch_floats((size_t)tokens * D), *normed = batch_floats((size_t)tokens * P);
    float *qkv = batch_floats((size_t)3 * P), *core = batch_floats(P);
    mm(q, &l->kq, x, tokens); mm(k, &l->kk, x, tokens); mm(v, &l->kv, x, tokens);
    mm(low, &l->kfa, x, tokens); mm(decay, &l->kfb, low, tokens);
    mm(beta, &l->kb, x, tokens);
    mm(low, &l->kga, x, tokens); mm(gate, &l->kgb, low, tokens);
    for (int t = 0; t < tokens; t++) {
        float *dt = decay + (size_t)t * P, *bt = beta + (size_t)t * H;
        for (int h = 0; h < H; h++) {
            bt[h] = sigmoidf_(bt[h]);
            for (int d = 0; d < D; d++) {
                int i = h * D + d;
                dt[i] = c->gate_lb * sigmoidf_(expf(l->alog[h]) * (dt[i] + l->dt[i]));
            }
        }
        memcpy(qkv, q + (size_t)t * P, (size_t)P * sizeof(float));
        memcpy(qkv + P, k + (size_t)t * P, (size_t)P * sizeof(float));
        memcpy(qkv + 2 * P, v + (size_t)t * P, (size_t)P * sizeof(float));
        /* Recurrence and convolution retain their exact token order. */
        coli_kda_step(core, state, window, qkv, l->conv, dt, bt, H, D, D, c->conv_k, 1e-6f, scratch);
        for (int h = 0; h < H; h++) {
            float square = 0;
            for (int d = 0; d < D; d++) square += core[h * D + d] * core[h * D + d];
            float inverse = 1.0f / sqrtf(square / D + c->eps);
            for (int d = 0; d < D; d++) {
                int i = h * D + d;
                normed[(size_t)t * P + i] = core[i] * inverse * l->onorm[d] * sigmoidf_(gate[(size_t)t * P + i]);
            }
        }
    }
    mm(out, &l->ko, normed, tokens);
    free(core); free(qkv); free(normed); free(low); free(beta); free(gate); free(decay); free(v); free(k); free(q);
}
