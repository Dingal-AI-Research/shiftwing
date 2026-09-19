/* GLM-specific FP32 batched math. Quantized weights stay compact in VRAM;
 * bounded row tiles are expanded for cuBLAS rather than expanding the model.
 * Independent of the Qwen runtime and its precision policy. */
#include "glm53_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <cstdint>
#include <initializer_list>

struct Buffer { void *ptr = nullptr; size_t bytes = 0; };
struct Weight {
    Buffer data;
    int fmt = 0, rows = 0, cols = 0, group = 0;
    size_t scale_offset = 0;
    Weight *next = nullptr;
};
struct GlmCuda {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    int device = 0;
    bool failed = false;
    Weight *resident = nullptr;
    Weight transient[3];
    Buffer input, output, gate, up, tile;
    GlmCudaStats stats = {};
};
static thread_local char last_error[256] = "";
static void error(const char *where, const char *why) {
    std::snprintf(last_error, sizeof(last_error), "%s: %s", where, why);
}
static bool cu(GlmCuda *c, cudaError_t rc, const char *where) {
    if (rc == cudaSuccess) return true;
    error(where, cudaGetErrorString(rc)); c->failed = true; return false;
}
static bool blas(GlmCuda *c, cublasStatus_t rc, const char *where) {
    if (rc == CUBLAS_STATUS_SUCCESS) return true;
    std::snprintf(last_error, sizeof(last_error), "%s: cuBLAS status %d", where, (int)rc);
    c->failed = true; return false;
}
static bool product(size_t a, size_t b, size_t *out) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    *out = a * b; return true;
}
static void release(GlmCuda *c, Buffer &b) {
    if (b.ptr) { cudaFree(b.ptr); c->stats.allocated_bytes -= b.bytes; }
    b = Buffer{};
}
static bool reserve(GlmCuda *c, Buffer &b, size_t bytes) {
    if (bytes <= b.bytes) return true;
    release(c, b);
    if (bytes > c->stats.budget_bytes - c->stats.allocated_bytes) {
        error("GPU allocation", "configured memory budget reached"); return false;
    }
    cudaError_t rc = cudaMalloc(&b.ptr, bytes);
    if (rc != cudaSuccess) { b.ptr = nullptr; cudaGetLastError(); error("GPU allocation", cudaGetErrorString(rc)); return false; }
    b.bytes = bytes; c->stats.allocated_bytes += bytes;
    c->stats.peak_bytes = std::max(c->stats.peak_bytes, c->stats.allocated_bytes);
    return true;
}
static bool upload(GlmCuda *c, void *dst, const void *src, size_t bytes) {
    if (!cu(c, cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, c->stream), "upload")) return false;
    c->stats.uploaded_bytes += bytes; return true;
}
static bool valid(const GlmCudaMatrix *m) {
    return m && m->weights && m->rows > 0 && m->cols > 0 &&
        (m->fmt == 0 || (m->scales && (m->fmt == 1 || (m->fmt == 4 && m->group > 0))));
}
static Weight *weights(GlmCuda *c, const GlmCudaMatrix *m, int scratch) {
    if (!valid(m)) { error("matrix", "invalid shape or precision"); return nullptr; }
    if (m->cache && *m->cache) return static_cast<Weight *>(*m->cache);
    size_t qbytes, sbytes = 0;
    const size_t stride = m->fmt == 4 ? ((size_t)m->cols + 1) / 2 :
                          (size_t)m->cols * (m->fmt == 0 ? sizeof(float) : 1);
    if (!product(m->rows, stride, &qbytes)) return nullptr;
    if (m->fmt != 0 && !product(m->rows, m->fmt == 1 ? sizeof(float) :
            (((size_t)m->cols + m->group - 1) / m->group) * sizeof(float), &sbytes)) return nullptr;
    const size_t at = (qbytes + 3) & ~(size_t)3;
    if (at < qbytes || sbytes > SIZE_MAX - at) return nullptr;
    const size_t slack = std::min(c->stats.budget_bytes / 4, (size_t)512 << 20);
    if (m->cache && at + sbytes + slack > c->stats.budget_bytes - c->stats.allocated_bytes) {
        error("resident weights", "leaving GPU workspace headroom"); return nullptr;
    }
    Weight *w = m->cache ? new (std::nothrow) Weight : &c->transient[scratch];
    if (!w) return nullptr;
    if (!reserve(c, w->data, at + sbytes)) { if (m->cache) delete w; return nullptr; }
    w->fmt = m->fmt; w->rows = m->rows; w->cols = m->cols; w->group = m->group; w->scale_offset = at;
    bool ok = upload(c, w->data.ptr, m->weights, qbytes) &&
        (!sbytes || upload(c, static_cast<char *>(w->data.ptr) + at, m->scales, sbytes));
    if (!ok) { if (m->cache) { release(c, w->data); delete w; } return nullptr; }
    if (m->cache) {
        w->next = c->resident; c->resident = w; *m->cache = w;
        c->stats.resident_bytes += w->data.bytes;
    }
    return w;
}
__global__ static void unpack(float *out, const unsigned char *q, const float *scale,
                             int fmt, int cols, int group, int first, int rows) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * cols) return;
    int row = (int)(i / cols) + first, col = (int)(i % cols);
    if (fmt == 0) { out[i] = reinterpret_cast<const float *>(q)[(size_t)row * cols + col]; return; }
    if (fmt == 1) { out[i] = reinterpret_cast<const signed char *>(q)[(size_t)row * cols + col] * scale[row]; return; }
    unsigned char b = q[(size_t)row * ((cols + 1) / 2) + col / 2];
    int value = ((col & 1) ? b >> 4 : b & 15) - 8;
    out[i] = value * scale[(size_t)row * ((cols + group - 1) / group) + col / group];
}
static bool gemm(GlmCuda *c, float *out, const float *x, int batch,
                 const Weight *w, int first, int rows) {
    const int tile_rows = std::min(rows, std::max(1, (int)((size_t)32 * 1024 * 1024 / ((size_t)w->cols * sizeof(float)))));
    size_t tile_bytes;
    if (!product((size_t)tile_rows * w->cols, sizeof(float), &tile_bytes) || !reserve(c, c->tile, tile_bytes)) return false;
    const float alpha = 1, beta = 0;
    const auto *q = static_cast<const unsigned char *>(w->data.ptr);
    const auto *s = reinterpret_cast<const float *>(q + w->scale_offset);
    for (int at = 0; at < rows; at += tile_rows) {
        int n = std::min(tile_rows, rows - at);
        unpack<<<(unsigned)(((size_t)n * w->cols + 255) / 256), 256, 0, c->stream>>>
            (static_cast<float *>(c->tile.ptr), q, s, w->fmt, w->cols, w->group, first + at, n);
        if (!cu(c, cudaGetLastError(), "unpack")) return false;
        /* y=x W^T in row-major; ldc stays at the full output width for tiles. */
        if (!blas(c, cublasSgemm(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, batch, w->cols,
                &alpha, static_cast<const float *>(c->tile.ptr), w->cols, x, w->cols,
                &beta, out + at, rows), "FP32 batched GEMM")) return false;
    }
    return true;
}
__global__ static void activate(float *gate, const float *up, size_t n, float limit) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i] > limit ? limit : gate[i];
    float u = up[i] < -limit ? -limit : (up[i] > limit ? limit : up[i]);
    gate[i] = (g / (1.0f + expf(-g))) * u;
}
static int finish(GlmCuda *c, float *y, int batch, int rows) {
    size_t bytes = (size_t)batch * rows * sizeof(float);
    bool copied = cu(c, cudaMemcpyAsync(y, c->output.ptr, bytes, cudaMemcpyDeviceToHost, c->stream), "download");
    bool synced = cu(c, cudaStreamSynchronize(c->stream), "finish");
    if (!copied || !synced) return 0;
    c->stats.calls++; c->stats.batched_calls += batch > 1;
    c->stats.max_batch = std::max(c->stats.max_batch, batch);
    return 1;
}
extern "C" const char *glm_cuda_error(void) { return last_error; }
extern "C" GlmCuda *glm_cuda_create(int device, size_t budget) {
    GlmCuda *c = new (std::nothrow) GlmCuda;
    if (!c) return nullptr;
    c->device = device;
    size_t free_bytes = 0, total = 0;
    if (!cu(c, cudaSetDevice(device), "select device") || !cu(c, cudaMemGetInfo(&free_bytes, &total), "device memory")) { delete c; return nullptr; }
    size_t headroom = std::min((size_t)1024 << 20, free_bytes / 4);
    c->stats.budget_bytes = std::min(budget, free_bytes - headroom);
    if (!c->stats.budget_bytes || !cu(c, cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "stream") ||
        !blas(c, cublasCreate(&c->blas), "create cuBLAS") ||
        !blas(c, cublasSetStream(c->blas, c->stream), "cuBLAS stream") ||
        !blas(c, cublasSetMathMode(c->blas, CUBLAS_PEDANTIC_MATH), "FP32 precision") ||
        !blas(c, cublasSetAtomicsMode(c->blas, CUBLAS_ATOMICS_NOT_ALLOWED), "atomics policy")) {
        glm_cuda_destroy(c); return nullptr;
    }
    return c;
}
extern "C" void glm_cuda_stats(const GlmCuda *c, GlmCudaStats *out) { if (c && out) *out = c->stats; }
extern "C" void glm_cuda_destroy(GlmCuda *c) {
    if (!c) return;
    cudaSetDevice(c->device);
    if (c->stream) cudaStreamSynchronize(c->stream);
    while (c->resident) { Weight *w = c->resident; c->resident = w->next; release(c, w->data); delete w; }
    for (auto &w : c->transient) release(c, w.data);
    for (Buffer *b : {&c->input, &c->output, &c->gate, &c->up, &c->tile}) release(c, *b);
    if (c->blas) cublasDestroy(c->blas);
    if (c->stream) cudaStreamDestroy(c->stream);
    delete c;
}
static int fallback(GlmCuda *c) { cudaStreamSynchronize(c->stream); return 0; }
extern "C" int glm_cuda_mm(GlmCuda *c, float *y, const float *x, int batch,
                            const GlmCudaMatrix *m, int first, int rows) {
    if (!c) return 0;
    c->stats.fallbacks++;
    if (c->failed) return 0;
    if (!x || !y || !valid(m) || batch <= 0 || first < 0 || rows <= 0 || first > m->rows - rows) return fallback(c);
    size_t xb, yb;
    if (!product((size_t)batch * m->cols, sizeof(float), &xb) || !product((size_t)batch * rows, sizeof(float), &yb)) return fallback(c);
    Weight *w = weights(c, m, 0);
    if (!w || !reserve(c, c->input, xb) || !reserve(c, c->output, yb) ||
        !upload(c, c->input.ptr, x, xb) ||
        !gemm(c, static_cast<float *>(c->output.ptr), static_cast<const float *>(c->input.ptr), batch, w, first, rows)) return fallback(c);
    int ok = finish(c, y, batch, rows); if (ok) c->stats.fallbacks--; return ok;
}
extern "C" int glm_cuda_mlp(GlmCuda *c, float *y, const float *x, int batch,
                             const GlmCudaMatrix *g, const GlmCudaMatrix *u,
                             const GlmCudaMatrix *d, float limit) {
    if (!c) return 0;
    c->stats.fallbacks++;
    if (c->failed) return 0;
    if (!x || !y || !valid(g) || !valid(u) || !valid(d) || batch <= 0 ||
        g->cols != u->cols || g->rows != u->rows || d->cols != g->rows || d->rows != g->cols || !std::isfinite(limit) || limit <= 0) return fallback(c);
    size_t xb, hb;
    if (!product((size_t)batch * g->cols, sizeof(float), &xb) || !product((size_t)batch * g->rows, sizeof(float), &hb)) return fallback(c);
    Weight *wg = weights(c, g, 0), *wu = weights(c, u, 1), *wd = weights(c, d, 2);
    if (!wg || !wu || !wd || !reserve(c, c->input, xb) || !reserve(c, c->output, xb) ||
        !reserve(c, c->gate, hb) || !reserve(c, c->up, hb) || !upload(c, c->input.ptr, x, xb)) return fallback(c);
    if (!gemm(c, static_cast<float *>(c->gate.ptr), static_cast<const float *>(c->input.ptr), batch, wg, 0, g->rows) ||
        !gemm(c, static_cast<float *>(c->up.ptr), static_cast<const float *>(c->input.ptr), batch, wu, 0, u->rows)) return fallback(c);
    activate<<<(unsigned)(((size_t)batch * g->rows + 255) / 256), 256, 0, c->stream>>>
        (static_cast<float *>(c->gate.ptr), static_cast<const float *>(c->up.ptr), (size_t)batch * g->rows, limit);
    if (!cu(c, cudaGetLastError(), "SwiGLU") ||
        !gemm(c, static_cast<float *>(c->output.ptr), static_cast<const float *>(c->gate.ptr), batch, wd, 0, d->rows)) return fallback(c);
    int ok = finish(c, y, batch, d->rows);
    if (ok) { c->stats.fallbacks--; c->stats.mlp_calls++; } return ok;
}
