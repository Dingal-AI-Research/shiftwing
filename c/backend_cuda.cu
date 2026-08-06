/* colib CUDA backend bootstrap. The public seam is deliberately C-compatible;
 * Phase 5 can replace individual kernels without leaking CUDA types into
 * qwen.c. */
#include "backend_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ColiCuda {
    int device;
    int major, minor;
    int async_alloc;
    char name[256];
    cudaStream_t stream;
    void **group_ptrs;
    float *group_weight, *group_gate, *group_up;
    float *group_route_out;
    __half *group_gate16;
    size_t group_ptr_cap, group_weight_cap, group_hidden_cap, group_up_cap,
           group_hidden16_cap, group_route_out_cap;
    float *gdn_x, *gdn_raw, *gdn_mix, *gdn_z, *gdn_a, *gdn_b, *gdn_core;
    __half *gdn_x16, *gdn_core16;
    size_t gdn_x_cap, gdn_raw_cap, gdn_mix_cap, gdn_z_cap, gdn_a_cap,
           gdn_b_cap, gdn_core_cap, gdn_x16_cap, gdn_core16_cap;
    float *attn_x,*attn_qp,*attn_q,*attn_gate,*attn_k,*attn_v,*attn_ctx;
    __half *attn_x16,*attn_ctx16;
    size_t attn_x_cap,attn_qp_cap,attn_q_cap,attn_gate_cap,attn_k_cap,
           attn_v_cap,attn_ctx_cap,attn_x16_cap,attn_ctx16_cap;
    int profile_stages, profile_pending, profile_shared_marked,
        profile_download_marked;
    cudaEvent_t profile_event[9];
    double profile_ms[8];
    unsigned long long profile_transactions;
};

static thread_local char coli_cuda_error[512];

static int fail_cuda(const char *where, cudaError_t error) {
    snprintf(coli_cuda_error, sizeof(coli_cuda_error), "%s: %s", where,
             cudaGetErrorString(error));
    return -1;
}

static int fail_arg(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(coli_cuda_error, sizeof(coli_cuda_error), fmt, ap);
    va_end(ap);
    return -1;
}

static int launch_status(const char *where) {
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : fail_cuda(where, error);
}

extern "C" const char *coli_cuda_last_error(void) {
    return coli_cuda_error[0] ? coli_cuda_error : "no CUDA error";
}

extern "C" int coli_cuda_create(ColiCuda **out, int device) {
    if (!out) return fail_arg("coli_cuda_create: null output pointer");
    *out = NULL;
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) return fail_cuda("cudaGetDeviceCount", error);
    if (device < 0 || device >= count)
        return fail_arg("CUDA device %d is outside [0,%d)", device, count);
    error = cudaSetDevice(device);
    if (error != cudaSuccess) return fail_cuda("cudaSetDevice", error);
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaSuccess) return fail_cuda("cudaGetDeviceProperties", error);
    ColiCuda *ctx = (ColiCuda *)calloc(1, sizeof(*ctx));
    if (!ctx) return fail_arg("out of memory creating CUDA context");
    ctx->device = device;
    ctx->major = prop.major;
    ctx->minor = prop.minor;
    snprintf(ctx->name, sizeof(ctx->name), "%s", prop.name);
    error = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
        free(ctx);
        return fail_cuda("cudaStreamCreateWithFlags", error);
    }
    const char *async_env = getenv("CUDA_ASYNC_ALLOC");
    if (!async_env || atoi(async_env) != 0) {
        int supported = 0;
        error = cudaDeviceGetAttribute(
            &supported, cudaDevAttrMemoryPoolsSupported, device);
        if (error == cudaSuccess && supported) {
            cudaMemPool_t pool;
            error = cudaDeviceGetDefaultMemPool(&pool, device);
            if (error == cudaSuccess) {
                uint64_t threshold = UINT64_MAX;
                error = cudaMemPoolSetAttribute(
                    pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            }
            if (error == cudaSuccess) ctx->async_alloc = 1;
        }
        if (!ctx->async_alloc)
            fprintf(stderr,
                    "[CUDA] stream-ordered allocator unavailable; "
                    "using synchronous cudaMalloc/cudaFree\n");
    }
    const char *profile_env = getenv("CUDA_PROFILE_STAGES");
    if (profile_env && atoi(profile_env) != 0) {
        ctx->profile_stages = 1;
        for (int i = 0; i < 9; i++) {
            error = cudaEventCreate(&ctx->profile_event[i]);
            if (error != cudaSuccess) {
                for (int j = 0; j < i; j++)
                    cudaEventDestroy(ctx->profile_event[j]);
                memset(ctx->profile_event, 0, sizeof(ctx->profile_event));
                ctx->profile_stages = 0;
                break;
            }
        }
    }
    coli_cuda_error[0] = 0;
    *out = ctx;
    return 0;
}

extern "C" void coli_cuda_destroy(ColiCuda *ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->device);
    cudaStreamSynchronize(ctx->stream);
    cudaFree(ctx->group_ptrs);
    cudaFree(ctx->group_weight);
    cudaFree(ctx->group_gate);
    cudaFree(ctx->group_up);
    cudaFree(ctx->group_route_out);
    cudaFree(ctx->group_gate16);
    cudaFree(ctx->gdn_x);
    cudaFree(ctx->gdn_raw);
    cudaFree(ctx->gdn_mix);
    cudaFree(ctx->gdn_z);
    cudaFree(ctx->gdn_a);
    cudaFree(ctx->gdn_b);
    cudaFree(ctx->gdn_core);
    cudaFree(ctx->gdn_x16);
    cudaFree(ctx->gdn_core16);
    cudaFree(ctx->attn_x);cudaFree(ctx->attn_qp);cudaFree(ctx->attn_q);
    cudaFree(ctx->attn_gate);cudaFree(ctx->attn_k);cudaFree(ctx->attn_v);
    cudaFree(ctx->attn_ctx);cudaFree(ctx->attn_x16);cudaFree(ctx->attn_ctx16);
    if (ctx->profile_transactions) {
        fprintf(stderr,
                "[CUDA_MOE_STAGE] transactions=%llu setup_ms=%.3f "
                "routed_hidden_ms=%.3f routed_down_ms=%.3f "
                "route_reduce_ms=%.3f shared_hidden_ms=%.3f "
                "shared_scale_ms=%.3f shared_down_ms=%.3f "
                "download_ms=%.3f\n",
                ctx->profile_transactions, ctx->profile_ms[0],
                ctx->profile_ms[1], ctx->profile_ms[2], ctx->profile_ms[3],
                ctx->profile_ms[4], ctx->profile_ms[5], ctx->profile_ms[6],
                ctx->profile_ms[7]);
    }
    for (int i = 0; i < 9; i++)
        if (ctx->profile_event[i]) cudaEventDestroy(ctx->profile_event[i]);
    cudaStreamDestroy(ctx->stream);
    free(ctx);
}

extern "C" const char *coli_cuda_device_name(const ColiCuda *ctx) {
    return ctx ? ctx->name : "";
}

extern "C" int coli_cuda_compute_capability(const ColiCuda *ctx, int *major,
                                               int *minor) {
    if (!ctx || !major || !minor)
        return fail_arg("coli_cuda_compute_capability: null argument");
    *major = ctx->major;
    *minor = ctx->minor;
    return 0;
}

extern "C" int coli_cuda_async_alloc_enabled(const ColiCuda *ctx) {
    return ctx ? ctx->async_alloc : 0;
}

extern "C" int coli_cuda_memory_info(ColiCuda *ctx, size_t *free_bytes,
                                      size_t *total_bytes) {
    if (!ctx || !free_bytes || !total_bytes)
        return fail_arg("coli_cuda_memory_info: null argument");
    cudaError_t error = cudaMemGetInfo(free_bytes, total_bytes);
    return error == cudaSuccess ? 0 : fail_cuda("cudaMemGetInfo", error);
}

extern "C" int coli_cuda_malloc(ColiCuda *ctx, void **ptr, size_t bytes) {
    if (!ctx || !ptr) return fail_arg("coli_cuda_malloc: null argument");
    cudaError_t error = ctx->async_alloc
                            ? cudaMallocAsync(ptr, bytes, ctx->stream)
                            : cudaMalloc(ptr, bytes);
    return error == cudaSuccess
               ? 0
               : fail_cuda(ctx->async_alloc ? "cudaMallocAsync" : "cudaMalloc",
                           error);
}

extern "C" void coli_cuda_free(ColiCuda *ctx, void *ptr) {
    if (!ctx || !ptr) return;
    cudaSetDevice(ctx->device);
    if (ctx->async_alloc) cudaFreeAsync(ptr, ctx->stream);
    else cudaFree(ptr);
}

extern "C" int coli_cuda_malloc_host(ColiCuda *ctx, void **ptr, size_t bytes) {
    if (!ctx || !ptr) return fail_arg("coli_cuda_malloc_host: null argument");
    cudaError_t error = cudaHostAlloc(ptr, bytes, cudaHostAllocDefault);
    return error == cudaSuccess ? 0 : fail_cuda("cudaHostAlloc", error);
}

extern "C" void coli_cuda_free_host(ColiCuda *ctx, void *ptr) {
    if (!ctx || !ptr) return;
    cudaFreeHost(ptr);
}

extern "C" int coli_cuda_upload(ColiCuda *ctx, void *dst, const void *src,
                                  size_t bytes) {
    if (!ctx || !dst || !src) return fail_arg("coli_cuda_upload: null argument");
    cudaError_t error = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                        ctx->stream);
    return error == cudaSuccess ? 0 : fail_cuda("cudaMemcpyAsync H2D", error);
}

extern "C" int coli_cuda_download(ColiCuda *ctx, void *dst, const void *src,
                                    size_t bytes) {
    if (!ctx || !dst || !src)
        return fail_arg("coli_cuda_download: null argument");
    if (ctx->profile_pending && !ctx->profile_shared_marked) {
        cudaEventRecord(ctx->profile_event[5], ctx->stream);
        cudaEventRecord(ctx->profile_event[6], ctx->stream);
        cudaEventRecord(ctx->profile_event[7], ctx->stream);
    }
    cudaError_t error = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                        ctx->stream);
    if (error == cudaSuccess && ctx->profile_pending) {
        cudaEventRecord(ctx->profile_event[8], ctx->stream);
        ctx->profile_download_marked = 1;
    }
    return error == cudaSuccess ? 0 : fail_cuda("cudaMemcpyAsync D2H", error);
}

extern "C" int coli_cuda_copy(ColiCuda *ctx, void *dst, const void *src,
                                size_t bytes) {
    if (!ctx || !dst || !src) return fail_arg("coli_cuda_copy: null argument");
    cudaError_t error = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                                        ctx->stream);
    return error == cudaSuccess ? 0 : fail_cuda("cudaMemcpyAsync D2D", error);
}

extern "C" int coli_cuda_memset(ColiCuda *ctx, void *dst, int value,
                                  size_t bytes) {
    if (!ctx || !dst) return fail_arg("coli_cuda_memset: null argument");
    cudaError_t error = cudaMemsetAsync(dst, value, bytes, ctx->stream);
    return error == cudaSuccess ? 0 : fail_cuda("cudaMemsetAsync", error);
}

extern "C" int coli_cuda_sync(ColiCuda *ctx) {
    if (!ctx) return fail_arg("coli_cuda_sync: null context");
    cudaError_t error = cudaStreamSynchronize(ctx->stream);
    if (error == cudaSuccess && ctx->profile_pending &&
        ctx->profile_download_marked) {
        for (int i = 0; i < 8; i++) {
            float elapsed = 0.0f;
            cudaError_t event_error = cudaEventElapsedTime(
                &elapsed, ctx->profile_event[i], ctx->profile_event[i + 1]);
            if (event_error == cudaSuccess) ctx->profile_ms[i] += elapsed;
        }
        ctx->profile_transactions++;
        ctx->profile_pending = 0;
        ctx->profile_shared_marked = 0;
        ctx->profile_download_marked = 0;
    }
    return error == cudaSuccess ? 0 : fail_cuda("cudaStreamSynchronize", error);
}

extern "C" void coli_cuda_profile_reset(ColiCuda *ctx) {
    if (!ctx || !ctx->profile_stages) return;
    cudaStreamSynchronize(ctx->stream);
    memset(ctx->profile_ms, 0, sizeof(ctx->profile_ms));
    ctx->profile_transactions = 0;
    ctx->profile_pending = 0;
    ctx->profile_shared_marked = 0;
    ctx->profile_download_marked = 0;
}

extern "C" int coli_cuda_profile_snapshot(ColiCuda *ctx,
                                           unsigned long long *transactions,
                                           double stage_ms[8]) {
    if (!ctx || !transactions || !stage_ms)
        return fail_arg("coli_cuda_profile_snapshot: null argument");
    if (!ctx->profile_stages) {
        *transactions = 0;
        memset(stage_ms, 0, sizeof(ctx->profile_ms));
        return 1;
    }
    if (coli_cuda_sync(ctx)) return -1;
    *transactions = ctx->profile_transactions;
    memcpy(stage_ms, ctx->profile_ms, sizeof(ctx->profile_ms));
    return 0;
}

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__inline__ __device__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0) value = warp_sum(value);
    return value;
}

__global__ static void rmsnorm_zero_kernel(float *y, const float *x,
                                            const float *w, int n, float eps) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[i] * x[i];
    sum = block_sum(sum);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / (float)n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[i] = x[i] * inv_rms * (1.0f + w[i]);
}

__global__ static void rmsnorm_zero_batch_kernel(
    float *y, const float *x, const float *w, int n, float eps) {
    int sample = blockIdx.x;
    const float *xs = x + (size_t)sample * n;
    float *ys = y + (size_t)sample * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += xs[i] * xs[i];
    sum = block_sum(sum);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sum / (float)n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ys[i] = xs[i] * inv_rms * (1.0f + w[i]);
}

__global__ static void f32_gemm_kernel(float *y, const float *x,
                                        const float *w, int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, sample = blockIdx.y;
    if (row >= rows) return;
    const float *xs = x + (size_t)sample * cols;
    const float *ws = w + (size_t)row * cols;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32) sum += xs[i] * ws[i];
    sum = warp_sum(sum);
    if (lane == 0) y[(size_t)sample * rows + row] = sum;
}

__global__ static void sigmoid_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ static void silu_mul_kernel(float *y, const float *gate,
                                        const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

__global__ static void axpy_kernel(float *y, const float *x, float scale,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += scale * x[i];
}

__global__ static void sigmoid_axpy_kernel(float *y, const float *x,
                                            const float *logit, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float gate = 1.0f / (1.0f + expf(-logit[0]));
    if (i < n) y[i] += gate * x[i];
}

__global__ static void f32_to_f16_kernel(__half *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}

__global__ static void q8_gemv_kernel(float *y, const float *x,
                                       const signed char *q,
                                       const float *scales, int rows, int cols,
                                       int row_bytes) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const signed char *qr = q + (size_t)row * row_bytes;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32)
        sum += x[i] * (float)qr[i];
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum * scales[row];
}

__global__ static void q8_gemm_kernel(float *y, const float *x,
                                      const signed char *q,
                                      const float *scales, int batch,
                                      int rows, int cols, int row_bytes) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, sample = blockIdx.y;
    if (row >= rows || sample >= batch) return;
    const float *xs = x + (size_t)sample * cols;
    const signed char *qr = q + (size_t)row * row_bytes;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32)
        sum += xs[i] * (float)qr[i];
    sum = warp_sum(sum);
    if (lane == 0)
        y[(size_t)sample * rows + row] = sum * scales[row];
}

__global__ static void q8_gemm_scaled_add_kernel(
    float *y, const float *x, const signed char *q, const float *scales,
    const float *sample_scale, int batch, int rows, int cols,
    int row_bytes) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, sample = blockIdx.y;
    if (row >= rows || sample >= batch) return;
    const float *xs = x + (size_t)sample * cols;
    const signed char *qr = q + (size_t)row * row_bytes;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32)
        sum += xs[i] * (float)qr[i];
    sum = warp_sum(sum);
    if (lane == 0) {
        float shared = sum * scales[row];
        size_t out = (size_t)sample * rows + row;
        y[out] += shared * sample_scale[sample];
    }
}

__global__ static void q4_gemv_kernel(float *y, const float *x,
                                       const unsigned char *q,
                                       const float *scales, int rows, int cols,
                                       int group_size, int row_bytes,
                                       int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const unsigned char *qr = q + (size_t)row * row_bytes;
    const float *sr = scales + (size_t)row * groups_per_row;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32) {
        unsigned char packed = qr[i >> 1];
        int quant = ((i & 1) ? (packed >> 4) : (packed & 15)) - 8;
        sum += x[i] * (float)quant * sr[i / group_size];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

__device__ static inline unsigned q3_word24(const unsigned char *q) {
    return (unsigned)q[0] | ((unsigned)q[1] << 8) |
           ((unsigned)q[2] << 16);
}

__global__ static void q3_gemv_kernel(float *y, const float *x,
                                       const unsigned char *q,
                                       const float *scales, int rows, int cols,
                                       int group_size, int row_bytes,
                                       int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const unsigned char *qr = q + (size_t)row * row_bytes;
    const float *sr = scales + (size_t)row * groups_per_row;
    float sum = 0.0f;
    int triplets = (cols + 7) / 8;
    for (int triplet = lane; triplet < triplets; triplet += 32) {
        int base = triplet * 8;
        unsigned word = q3_word24(qr + triplet * 3);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            int i = base + k;
            if (i < cols) {
                int quant = (int)((word >> (3 * k)) & 7u) - 4;
                sum += x[i] * (float)quant * sr[i / group_size];
            }
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ static void q4_gemm_kernel(float *y, const float *x,
                                      const unsigned char *q,
                                      const float *scales, int batch,
                                      int rows, int cols, int group_size,
                                      int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, sample = blockIdx.y;
    if (row >= rows || sample >= batch) return;
    const float *xs = x + (size_t)sample * cols;
    const unsigned char *qr = q + (size_t)row * row_bytes;
    const float *sr = scales + (size_t)row * groups_per_row;
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32) {
        unsigned char packed = qr[i >> 1];
        int quant = ((i & 1) ? (packed >> 4) : (packed & 15)) - 8;
        sum += xs[i] * (float)quant * sr[i / group_size];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[(size_t)sample * rows + row] = sum;
}

__global__ static void q4_gemm_f16_kernel(float *y, const __half *x,
                                           const unsigned char *q,
                                           const float *scales, int batch,
                                           int rows, int cols, int group_size,
                                           int row_bytes,
                                           int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, sample = blockIdx.y;
    if (row >= rows || sample >= batch) return;
    const __half *xs = x + (size_t)sample * cols;
    const unsigned char *qr = q + (size_t)row * row_bytes;
    const float *sr = scales + (size_t)row * groups_per_row;
    float sum = 0.0f;
    int pairs = (cols + 1) / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        int i = pair * 2;unsigned char packed = qr[pair];
        sum += __half2float(xs[i]) * (float)((packed & 15) - 8) *
               sr[i / group_size];
        if (i + 1 < cols)
            sum += __half2float(xs[i + 1]) * (float)((packed >> 4) - 8) *
                   sr[(i + 1) / group_size];
    }
    sum = warp_sum(sum);
    if (lane == 0) y[(size_t)sample * rows + row] = sum;
}

__global__ static void q4_gemv_f16_kernel(float *y, const __half *x,
                                           const unsigned char *q,
                                           const float *scales, int rows,
                                           int cols, int group_size,
                                           int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const unsigned char *qr = q + (size_t)row * row_bytes;
    const float *sr = scales + (size_t)row * groups_per_row;
    float sum = 0.0f;
    int pairs = (cols + 1) / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        int i = pair * 2;
        unsigned char packed = qr[pair];
        int q0 = (packed & 15) - 8;
        sum += __half2float(x[i]) * (float)q0 * sr[i / group_size];
        if (i + 1 < cols) {
            int q1 = (packed >> 4) - 8;
            sum += __half2float(x[i + 1]) * (float)q1 * sr[(i + 1) / group_size];
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ static void q3_gemv_f16_kernel(float *y, const __half *x,
                                           const unsigned char *q,
                                           const float *scales, int rows,
                                           int cols, int group_size,
                                           int row_bytes,
                                           int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const unsigned char *qr = q + (size_t)row * row_bytes;
    const float *sr = scales + (size_t)row * groups_per_row;
    float sum = 0.0f;
    int triplets = (cols + 7) / 8;
    for (int triplet = lane; triplet < triplets; triplet += 32) {
        int base = triplet * 8;
        unsigned word = q3_word24(qr + triplet * 3);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            int i = base + k;
            if (i < cols) {
                int quant = (int)((word >> (3 * k)) & 7u) - 4;
                sum += __half2float(x[i]) * (float)quant *
                       sr[i / group_size];
            }
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ static void grouped_hidden_q4_f16_kernel(
    __half *gate_out, const __half *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    int intermediate, int hidden, int group_size, int row_bytes,
    int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, expert = blockIdx.y;
    if (row >= intermediate) return;
    const unsigned char *gq = gate_q[expert] + (size_t)row * row_bytes;
    const unsigned char *uq = up_q[expert] + (size_t)row * row_bytes;
    const float *gs = gate_s[expert] + (size_t)row * groups_per_row;
    const float *us = up_s[expert] + (size_t)row * groups_per_row;
    float gate = 0.0f, up = 0.0f;
    int pairs = (hidden + 1) / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        int i = pair * 2;
        unsigned char gp = gq[pair], uqp = uq[pair];
        float x0 = __half2float(x[i]);
        gate += x0 * (float)((gp & 15) - 8) * gs[i / group_size];
        up += x0 * (float)((uqp & 15) - 8) * us[i / group_size];
        if (i + 1 < hidden) {
            float x1 = __half2float(x[i + 1]);
            gate += x1 * (float)((gp >> 4) - 8) * gs[(i + 1) / group_size];
            up += x1 * (float)((uqp >> 4) - 8) * us[(i + 1) / group_size];
        }
    }
    gate = warp_sum(gate);
    up = warp_sum(up);
    if (lane == 0) {
        size_t out = (size_t)expert * intermediate + row;
        float g = gate;
        gate_out[out] = __float2half_rn(g / (1.0f + expf(-g)) * up);
    }
}

__global__ static void grouped_hidden_q3_f16_kernel(
    __half *gate_out, const __half *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    int intermediate, int hidden, int group_size, int row_bytes,
    int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, expert = blockIdx.y;
    if (row >= intermediate) return;
    const unsigned char *gq = gate_q[expert] + (size_t)row * row_bytes;
    const unsigned char *uq = up_q[expert] + (size_t)row * row_bytes;
    const float *gs = gate_s[expert] + (size_t)row * groups_per_row;
    const float *us = up_s[expert] + (size_t)row * groups_per_row;
    float gate = 0.0f, up = 0.0f;
    int triplets = (hidden + 7) / 8;
    for (int triplet = lane; triplet < triplets; triplet += 32) {
        int base = triplet * 8;
        unsigned gw = q3_word24(gq + triplet * 3);
        unsigned uw = q3_word24(uq + triplet * 3);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            int i = base + k;
            if (i < hidden) {
                float value = __half2float(x[i]);
                gate += value * (float)((int)((gw >> (3 * k)) & 7u) - 4) *
                        gs[i / group_size];
                up += value * (float)((int)((uw >> (3 * k)) & 7u) - 4) *
                      us[i / group_size];
            }
        }
    }
    gate = warp_sum(gate); up = warp_sum(up);
    if (lane == 0) {
        size_t out = (size_t)expert * intermediate + row;
        gate_out[out] =
            __float2half_rn(gate / (1.0f + expf(-gate)) * up);
    }
}

__global__ static void shared_hidden_q4_f16_kernel(
    __half *hidden_out, const __half *x, const unsigned char *gate_q,
    const float *gate_s, const unsigned char *up_q, const float *up_s,
    int intermediate, int hidden, int group_size, int row_bytes,
    int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= intermediate) return;
    const unsigned char *gq = gate_q + (size_t)row * row_bytes;
    const unsigned char *uq = up_q + (size_t)row * row_bytes;
    const float *gs = gate_s + (size_t)row * groups_per_row;
    const float *us = up_s + (size_t)row * groups_per_row;
    float gate = 0.0f, up = 0.0f;
    int pairs = (hidden + 1) / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        int i = pair * 2;
        unsigned char gp = gq[pair], uqp = uq[pair];
        float x0 = __half2float(x[i]);
        gate += x0 * (float)((gp & 15) - 8) * gs[i / group_size];
        up += x0 * (float)((uqp & 15) - 8) * us[i / group_size];
        if (i + 1 < hidden) {
            float x1 = __half2float(x[i + 1]);
            gate += x1 * (float)((gp >> 4) - 8) * gs[(i + 1) / group_size];
            up += x1 * (float)((uqp >> 4) - 8) * us[(i + 1) / group_size];
        }
    }
    gate = warp_sum(gate);
    up = warp_sum(up);
    if (lane == 0)
        hidden_out[row] = __float2half_rn(gate / (1.0f + expf(-gate)) * up);
}

__global__ static void q4_gemv_f16_sigmoid_axpy_kernel(
    float *y, const __half *x, const unsigned char *q, const float *scales,
    const float *logit, int rows, int cols, int group_size, int row_bytes,
    int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const unsigned char *qr = q + (size_t)row * row_bytes;
    const float *sr = scales + (size_t)row * groups_per_row;
    float sum = 0.0f;
    int pairs = (cols + 1) / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        int i = pair * 2;
        unsigned char packed = qr[pair];
        sum += __half2float(x[i]) * (float)((packed & 15) - 8) *
               sr[i / group_size];
        if (i + 1 < cols)
            sum += __half2float(x[i + 1]) * (float)((packed >> 4) - 8) *
                   sr[(i + 1) / group_size];
    }
    sum = warp_sum(sum);
    if (lane == 0)
        y[row] += (1.0f / (1.0f + expf(-logit[0]))) * sum;
}

__global__ static void grouped_down_q4_f16_kernel(
    float *y, const __half *hidden_values,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int experts, int hidden, int intermediate,
    int group_size, int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= hidden) return;
    float total = 0.0f;
    for (int expert = 0; expert < experts; expert++) {
        const unsigned char *qr = down_q[expert] + (size_t)row * row_bytes;
        const float *sr = down_s[expert] + (size_t)row * groups_per_row;
        const __half *x = hidden_values + (size_t)expert * intermediate;
        float sum = 0.0f;
        int pairs = (intermediate + 1) / 2;
        for (int pair = lane; pair < pairs; pair += 32) {
            int i = pair * 2;
            unsigned char packed = qr[pair];
            sum += __half2float(x[i]) * (float)((packed & 15) - 8) * sr[i / group_size];
            if (i + 1 < intermediate)
                sum += __half2float(x[i + 1]) * (float)((packed >> 4) - 8) *
                       sr[(i + 1) / group_size];
        }
        sum = warp_sum(sum);
        if (lane == 0) total += route_weight[expert] * sum;
    }
    if (lane == 0) y[row] = total;
}

__global__ static void grouped_down_q3_f16_kernel(
    float *y, const __half *hidden_values,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int experts, int hidden, int intermediate,
    int group_size, int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= hidden) return;
    float total = 0.0f;
    for (int expert = 0; expert < experts; expert++) {
        const unsigned char *qr = down_q[expert] + (size_t)row * row_bytes;
        const float *sr = down_s[expert] + (size_t)row * groups_per_row;
        const __half *x = hidden_values + (size_t)expert * intermediate;
        float sum = 0.0f;
        int triplets = (intermediate + 7) / 8;
        for (int triplet = lane; triplet < triplets; triplet += 32) {
            int base = triplet * 8;
            unsigned word = q3_word24(qr + triplet * 3);
#pragma unroll
            for (int k = 0; k < 8; k++) {
                int i = base + k;
                if (i < intermediate)
                    sum += __half2float(x[i]) *
                           (float)((int)((word >> (3 * k)) & 7u) - 4) *
                           sr[i / group_size];
            }
        }
        sum = warp_sum(sum);
        if (lane == 0) total += route_weight[expert] * sum;
    }
    if (lane == 0) y[row] = total;
}

__global__ static void grouped_hidden_batch_q4_f16_kernel(
    __half *gate_out, const __half *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const int *route_map, int unique_experts, int batch, int intermediate,
    int hidden,
    int group_size, int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, expert = blockIdx.y;
    if (row >= intermediate || expert >= unique_experts) return;
    const unsigned char *gq = gate_q[expert] + (size_t)row * row_bytes;
    const unsigned char *uq = up_q[expert] + (size_t)row * row_bytes;
    const float *gs = gate_s[expert] + (size_t)row * groups_per_row;
    const float *us = up_s[expert] + (size_t)row * groups_per_row;
    float gate[8] = {0.0f}, up[8] = {0.0f};
    int pairs = (hidden + 1) / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        int i = pair * 2;
        unsigned char gp = gq[pair], uqp = uq[pair];
        for (int sample = 0; sample < batch; sample++) {
            int route = route_map[expert * batch + sample];
            if (route < 0) continue;
            const __half *xs = x + (size_t)sample * hidden;
            float x0 = __half2float(xs[i]);
            gate[sample] +=
                x0 * (float)((gp & 15) - 8) * gs[i / group_size];
            up[sample] +=
                x0 * (float)((uqp & 15) - 8) * us[i / group_size];
            if (i + 1 < hidden) {
                float x1 = __half2float(xs[i + 1]);
                gate[sample] +=
                    x1 * (float)((gp >> 4) - 8) *
                    gs[(i + 1) / group_size];
                up[sample] +=
                    x1 * (float)((uqp >> 4) - 8) *
                    us[(i + 1) / group_size];
            }
        }
    }
    for (int sample = 0; sample < batch; sample++) {
        int route = route_map[expert * batch + sample];
        if (route < 0) continue;
        float g = warp_sum(gate[sample]);
        float u = warp_sum(up[sample]);
        if (lane == 0)
            gate_out[(size_t)route * intermediate + row] =
                __float2half_rn(g / (1.0f + expf(-g)) * u);
    }
}

__global__ static void grouped_hidden_batch_q3_f16_kernel(
    __half *gate_out, const __half *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const int *route_map, int unique_experts, int batch, int intermediate,
    int hidden, int group_size, int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, expert = blockIdx.y;
    if (row >= intermediate || expert >= unique_experts) return;
    const unsigned char *gq = gate_q[expert] + (size_t)row * row_bytes;
    const unsigned char *uq = up_q[expert] + (size_t)row * row_bytes;
    const float *gs = gate_s[expert] + (size_t)row * groups_per_row;
    const float *us = up_s[expert] + (size_t)row * groups_per_row;
    float gate[8] = {0.0f}, up[8] = {0.0f};
    int triplets = (hidden + 7) / 8;
    for (int triplet = lane; triplet < triplets; triplet += 32) {
        int base = triplet * 8;
        unsigned gw = q3_word24(gq + triplet * 3);
        unsigned uw = q3_word24(uq + triplet * 3);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            int i = base + k;
            if (i >= hidden) continue;
            float gweight = (float)((int)((gw >> (3 * k)) & 7u) - 4) *
                            gs[i / group_size];
            float uweight = (float)((int)((uw >> (3 * k)) & 7u) - 4) *
                            us[i / group_size];
            for (int sample = 0; sample < batch; sample++) {
                int route = route_map[expert * batch + sample];
                if (route < 0) continue;
                float value = __half2float(x[(size_t)sample * hidden + i]);
                gate[sample] += value * gweight;
                up[sample] += value * uweight;
            }
        }
    }
    for (int sample = 0; sample < batch; sample++) {
        int route = route_map[expert * batch + sample];
        if (route < 0) continue;
        float g = warp_sum(gate[sample]);
        float u = warp_sum(up[sample]);
        if (lane == 0)
            gate_out[(size_t)route * intermediate + row] =
                __float2half_rn(g / (1.0f + expf(-g)) * u);
    }
}

__global__ static void grouped_down_batch_q4_f16_kernel(
    float *route_out, const __half *hidden_values,
    const unsigned char *const *down_q, const float *const *down_s,
    const int *route_map, int unique_experts, int batch, int hidden,
    int intermediate, int group_size, int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= hidden) return;
    for (int expert = 0; expert < unique_experts; expert++) {
        const unsigned char *qr =
            down_q[expert] + (size_t)row * row_bytes;
        const float *sr = down_s[expert] + (size_t)row * groups_per_row;
        float sum[8] = {0.0f};
        int pairs = (intermediate + 1) / 2;
        for (int pair = lane; pair < pairs; pair += 32) {
            int i = pair * 2;
            unsigned char packed = qr[pair];
            for (int sample = 0; sample < batch; sample++) {
                int route = route_map[expert * batch + sample];
                if (route < 0) continue;
                const __half *xs =
                    hidden_values + (size_t)route * intermediate;
                sum[sample] +=
                    __half2float(xs[i]) * (float)((packed & 15) - 8) *
                    sr[i / group_size];
                if (i + 1 < intermediate)
                    sum[sample] +=
                        __half2float(xs[i + 1]) *
                        (float)((packed >> 4) - 8) *
                        sr[(i + 1) / group_size];
            }
        }
        for (int sample = 0; sample < batch; sample++) {
            int route = route_map[expert * batch + sample];
            if (route < 0) continue;
            float value = warp_sum(sum[sample]);
            if (lane == 0)
                route_out[(size_t)route * hidden + row] = value;
        }
    }
}

__global__ static void grouped_down_batch_q3_f16_kernel(
    float *route_out, const __half *hidden_values,
    const unsigned char *const *down_q, const float *const *down_s,
    const int *route_map, int unique_experts, int batch, int hidden,
    int intermediate, int group_size, int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= hidden) return;
    for (int expert = 0; expert < unique_experts; expert++) {
        const unsigned char *qr = down_q[expert] + (size_t)row * row_bytes;
        const float *sr = down_s[expert] + (size_t)row * groups_per_row;
        float sum[8] = {0.0f};
        int triplets = (intermediate + 7) / 8;
        for (int triplet = lane; triplet < triplets; triplet += 32) {
            int base = triplet * 8;
            unsigned word = q3_word24(qr + triplet * 3);
#pragma unroll
            for (int k = 0; k < 8; k++) {
                int i = base + k;
                if (i >= intermediate) continue;
                float weight =
                    (float)((int)((word >> (3 * k)) & 7u) - 4) *
                    sr[i / group_size];
                for (int sample = 0; sample < batch; sample++) {
                    int route = route_map[expert * batch + sample];
                    if (route < 0) continue;
                    sum[sample] +=
                        __half2float(hidden_values[
                            (size_t)route * intermediate + i]) * weight;
                }
            }
        }
        for (int sample = 0; sample < batch; sample++) {
            int route = route_map[expert * batch + sample];
            if (route < 0) continue;
            float value = warp_sum(sum[sample]);
            if (lane == 0)
                route_out[(size_t)route * hidden + row] = value;
        }
    }
}

__global__ static void grouped_reduce_batch_kernel(
    float *y, const float *route_out, const float *route_weight, int batch,
    int experts_per_token, int hidden) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = blockIdx.y;
    if (row >= hidden || sample >= batch) return;
    float total = 0.0f;
    int first = sample * experts_per_token;
    for (int j = 0; j < experts_per_token; j++)
        total += route_weight[first + j] *
                 route_out[(size_t)(first + j) * hidden + row];
    y[(size_t)sample * hidden + row] = total;
}

__global__ static void shared_hidden_batch_q4_f16_kernel(
    __half *hidden_out, const __half *x, const unsigned char *gate_q,
    const float *gate_s, const unsigned char *up_q, const float *up_s,
    int batch, int intermediate, int hidden, int group_size, int row_bytes,
    int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= intermediate) return;
    const unsigned char *gq = gate_q + (size_t)row * row_bytes;
    const unsigned char *uq = up_q + (size_t)row * row_bytes;
    const float *gs = gate_s + (size_t)row * groups_per_row;
    const float *us = up_s + (size_t)row * groups_per_row;
    float gate[8] = {0.0f}, up[8] = {0.0f};
    int pairs = (hidden + 1) / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        int i = pair * 2;
        unsigned char gp = gq[pair], uqp = uq[pair];
        float gw0 = (float)((gp & 15) - 8) * gs[i / group_size];
        float uw0 = (float)((uqp & 15) - 8) * us[i / group_size];
        float gw1 = 0.0f, uw1 = 0.0f;
        if (i + 1 < hidden) {
            gw1 = (float)((gp >> 4) - 8) *
                  gs[(i + 1) / group_size];
            uw1 = (float)((uqp >> 4) - 8) *
                  us[(i + 1) / group_size];
        }
        for (int sample = 0; sample < batch; sample++) {
            const __half *xs = x + (size_t)sample * hidden;
            float x0 = __half2float(xs[i]);
            gate[sample] += x0 * gw0;
            up[sample] += x0 * uw0;
            if (i + 1 < hidden) {
                float x1 = __half2float(xs[i + 1]);
                gate[sample] += x1 * gw1;
                up[sample] += x1 * uw1;
            }
        }
    }
    for (int sample = 0; sample < batch; sample++) {
        float g = warp_sum(gate[sample]);
        float u = warp_sum(up[sample]);
        if (lane == 0)
            hidden_out[(size_t)sample * intermediate + row] =
                __float2half_rn(g / (1.0f + expf(-g)) * u);
    }
}

__global__ static void shared_down_batch_q4_f16_kernel(
    float *y, const __half *hidden_values, const unsigned char *down_q,
    const float *down_s, const float *scale_logit, int batch, int hidden,
    int intermediate, int group_size, int row_bytes, int groups_per_row) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    if (row >= hidden) return;
    const unsigned char *qr = down_q + (size_t)row * row_bytes;
    const float *sr = down_s + (size_t)row * groups_per_row;
    float sum[8] = {0.0f};
    int pairs = (intermediate + 1) / 2;
    for (int pair = lane; pair < pairs; pair += 32) {
        int i = pair * 2;
        unsigned char packed = qr[pair];
        float w0 = (float)((packed & 15) - 8) * sr[i / group_size];
        float w1 = 0.0f;
        if (i + 1 < intermediate)
            w1 = (float)((packed >> 4) - 8) *
                 sr[(i + 1) / group_size];
        for (int sample = 0; sample < batch; sample++) {
            const __half *xs =
                hidden_values + (size_t)sample * intermediate;
            sum[sample] += __half2float(xs[i]) * w0;
            if (i + 1 < intermediate)
                sum[sample] += __half2float(xs[i + 1]) * w1;
        }
    }
    for (int sample = 0; sample < batch; sample++) {
        float value = warp_sum(sum[sample]);
        if (lane == 0)
            y[(size_t)sample * hidden + row] +=
                (1.0f / (1.0f + expf(-scale_logit[sample]))) * value;
    }
}

__global__ static void gdn_conv_kernel(float *mix, const float *conv_state,
                                        const float *weight, int position,
                                        int channels, int kernel) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;
    float sum = 0.0f;
    for (int tap = 0; tap < kernel; tap++) {
        int source = position - (kernel - 1 - tap);
        if (source >= 0)
            sum += weight[(size_t)ch * kernel + tap] *
                   conv_state[(size_t)(source % kernel) * channels + ch];
    }
    mix[ch] = sum / (1.0f + expf(-sum));
}

__device__ static float stable_softplus(float x) {
    return x > 20.0f ? x : (x < -20.0f ? expf(x) : log1pf(expf(x)));
}

__global__ static void gdn_recurrent_kernel(
    float *core, float *state, const float *mix, const float *z,
    const float *a, const float *b, const float *a_log,
    const float *dt_bias, const float *norm, int key_heads, int value_heads,
    int key_dim, int value_dim, float eps) {
    int h = blockIdx.x, j = threadIdx.x;
    if (h >= value_heads || j >= value_dim) return;
    int key_head = h / (value_heads / key_heads);
    int key_total = key_heads * key_dim;
    const float *q = mix + (size_t)key_head * key_dim;
    const float *k = mix + key_total + (size_t)key_head * key_dim;
    const float *v = mix + 2 * key_total + (size_t)h * value_dim;
    float q2 = 0.0f, k2 = 0.0f;
    if (j < key_dim) { q2 = q[j] * q[j]; k2 = k[j] * k[j]; }
    q2 = block_sum(q2);
    __shared__ float q_inv, k_inv, decay, beta;
    if (j == 0) q_inv = rsqrtf(q2 + 1e-6f) * rsqrtf((float)key_dim);
    __syncthreads();
    k2 = block_sum(k2);
    if (j == 0) {
        k_inv = rsqrtf(k2 + 1e-6f);
        beta = 1.0f / (1.0f + expf(-b[h]));
        decay = expf(-expf(a_log[h]) * stable_softplus(a[h] + dt_bias[h]));
    }
    __syncthreads();
    float *column = state + (size_t)h * key_dim * value_dim + j;
    float memory = 0.0f;
    for (int i = 0; i < key_dim; i++) {
        float old = column[(size_t)i * value_dim] * decay;
        column[(size_t)i * value_dim] = old;
        memory += old * (k[i] * k_inv);
    }
    float delta = (v[j] - memory) * beta;
    float result = 0.0f;
    for (int i = 0; i < key_dim; i++) {
        float updated = column[(size_t)i * value_dim] +
                        (k[i] * k_inv) * delta;
        column[(size_t)i * value_dim] = updated;
        result += updated * (q[i] * q_inv);
    }
    core[(size_t)h * value_dim + j] = result;
    float square = block_sum(result * result);
    __shared__ float rms_inv;
    if (j == 0) rms_inv = rsqrtf(square / value_dim + eps);
    __syncthreads();
    float zv = z[(size_t)h * value_dim + j];
    core[(size_t)h * value_dim + j] =
        result * rms_inv * norm[j] * (zv / (1.0f + expf(-zv)));
}

__global__ static void qnorm_rope_kernel(float *q, float *gate,
                                          const float *packed,
                                          const float *weight, int heads,
                                          int head_dim, int rotary_dim,
                                          int position, float theta,
                                          float eps) {
    int h=blockIdx.x,j=threadIdx.x;if(h>=heads||j>=head_dim)return;
    const float*src=packed+(size_t)h*2*head_dim;float v=src[j];
    float ms=block_sum(v*v);__shared__ float inv;if(j==0)inv=rsqrtf(ms/head_dim+eps);__syncthreads();
    q[(size_t)h*head_dim+j]=v*inv*(1.f+weight[j]);gate[(size_t)h*head_dim+j]=src[head_dim+j];__syncthreads();
    int half=rotary_dim/2;if(j<half){float angle=position*powf(theta,-2.f*j/rotary_dim),cs=cosf(angle),sn=sinf(angle);size_t base=(size_t)h*head_dim;float a=q[base+j],b=q[base+half+j];q[base+j]=a*cs-b*sn;q[base+half+j]=b*cs+a*sn;}
}

__global__ static void knorm_rope_kernel(float *k,const float *weight,
                                          int heads,int head_dim,int rotary_dim,
                                          int position,float theta,float eps){
    int h=blockIdx.x,j=threadIdx.x;if(h>=heads||j>=head_dim)return;size_t base=(size_t)h*head_dim;float v=k[base+j];float ms=block_sum(v*v);__shared__ float inv;if(j==0)inv=rsqrtf(ms/head_dim+eps);__syncthreads();k[base+j]=v*inv*(1.f+weight[j]);__syncthreads();int half=rotary_dim/2;if(j<half){float angle=position*powf(theta,-2.f*j/rotary_dim),cs=cosf(angle),sn=sinf(angle);float a=k[base+j],b=k[base+half+j];k[base+j]=a*cs-b*sn;k[base+half+j]=b*cs+a*sn;}
}

__global__ static void gqa_decode_kernel(float*ctx,const float*q,const float*gate,
        const float*kcache,const float*vcache,int position,int qheads,int kvheads,int hd){
    int h=blockIdx.x,j=threadIdx.x;if(h>=qheads||j>=hd)return;int hk=h/(qheads/kvheads),T=position+1,kvrows=kvheads*hd;extern __shared__ float score[];
    for(int t=0;t<T;t++){size_t off=(size_t)t*kvrows+(size_t)hk*hd;float p=q[(size_t)h*hd+j]*kcache[off+j];float sum=block_sum(p);if(j==0)score[t]=sum*rsqrtf((float)hd);__syncthreads();}
    if(j==0){float mx=score[0];for(int t=1;t<T;t++)mx=fmaxf(mx,score[t]);float den=0.f;for(int t=0;t<T;t++){score[t]=expf(score[t]-mx);den+=score[t];}for(int t=0;t<T;t++)score[t]/=den;}__syncthreads();
    float sum=0.f;for(int t=0;t<T;t++)sum+=score[t]*vcache[(size_t)t*kvrows+(size_t)hk*hd+j];float g=gate[(size_t)h*hd+j];ctx[(size_t)h*hd+j]=sum/(1.f+expf(-g));
}

static int ensure_cuda_buffer(void **ptr, size_t *capacity, size_t bytes,
                              const char *where) {
    if (*capacity >= bytes) return 0;
    cudaFree(*ptr);
    *ptr = NULL;
    *capacity = 0;
    cudaError_t error = cudaMalloc(ptr, bytes);
    if (error != cudaSuccess) return fail_cuda(where, error);
    *capacity = bytes;
    return 0;
}

extern "C" int coli_cuda_rmsnorm_zero(ColiCuda *ctx, float *y, const float *x,
                                        const float *w, int n, float eps) {
    if (!ctx || !y || !x || !w || n <= 0)
        return fail_arg("coli_cuda_rmsnorm_zero: invalid argument");
    rmsnorm_zero_kernel<<<1, 256, 0, ctx->stream>>>(y, x, w, n, eps);
    return launch_status("rmsnorm_zero_kernel");
}

extern "C" int coli_cuda_rmsnorm_zero_batch(
    ColiCuda *ctx, float *y, const float *x, const float *w, int batch,
    int n, float eps) {
    if (!ctx || !y || !x || !w || batch <= 0 || n <= 0)
        return fail_arg("coli_cuda_rmsnorm_zero_batch: invalid argument");
    rmsnorm_zero_batch_kernel<<<batch, 256, 0, ctx->stream>>>(
        y, x, w, n, eps);
    return launch_status("rmsnorm_zero_batch_kernel");
}

extern "C" int coli_cuda_f32_gemm(ColiCuda *ctx, float *y, const float *x,
                                    const float *w, int batch, int rows,
                                    int cols) {
    if (!ctx || !y || !x || !w || batch <= 0 || rows <= 0 || cols <= 0)
        return fail_arg("coli_cuda_f32_gemm: invalid argument");
    dim3 grid((rows + 7) / 8, batch);
    f32_gemm_kernel<<<grid, 256, 0, ctx->stream>>>(
        y, x, w, rows, cols);
    return launch_status("f32_gemm_kernel");
}

extern "C" int coli_cuda_sigmoid(ColiCuda *ctx, float *y, const float *x,
                                   int n) {
    if (!ctx || !y || !x || n <= 0)
        return fail_arg("coli_cuda_sigmoid: invalid argument");
    sigmoid_kernel<<<(n + 255) / 256, 256, 0, ctx->stream>>>(y, x, n);
    return launch_status("sigmoid_kernel");
}

extern "C" int coli_cuda_silu_mul(ColiCuda *ctx, float *y, const float *gate,
                                    const float *up, int n) {
    if (!ctx || !y || !gate || !up || n <= 0)
        return fail_arg("coli_cuda_silu_mul: invalid argument");
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, ctx->stream>>>(y, gate, up, n);
    return launch_status("silu_mul_kernel");
}

extern "C" int coli_cuda_axpy(ColiCuda *ctx, float *y, const float *x,
                               float scale, int n) {
    if (!ctx || !y || !x || n <= 0)
        return fail_arg("coli_cuda_axpy: invalid argument");
    axpy_kernel<<<(n + 255) / 256, 256, 0, ctx->stream>>>(y, x, scale, n);
    return launch_status("axpy_kernel");
}

extern "C" int coli_cuda_sigmoid_axpy(ColiCuda *ctx, float *y,
                                        const float *x, const float *logit,
                                        int n) {
    if (!ctx || !y || !x || !logit || n <= 0)
        return fail_arg("coli_cuda_sigmoid_axpy: invalid argument");
    sigmoid_axpy_kernel<<<(n + 255) / 256, 256, 0, ctx->stream>>>(
        y, x, logit, n);
    return launch_status("sigmoid_axpy_kernel");
}

extern "C" int coli_cuda_f32_to_f16(ColiCuda *ctx, unsigned short *y,
                                      const float *x, int n) {
    if (!ctx || !y || !x || n <= 0)
        return fail_arg("coli_cuda_f32_to_f16: invalid argument");
    static_assert(sizeof(__half) == sizeof(unsigned short), "unexpected half size");
    f32_to_f16_kernel<<<(n + 255) / 256, 256, 0, ctx->stream>>>(
        (__half *)y, x, n);
    return launch_status("f32_to_f16_kernel");
}

extern "C" int coli_cuda_q8_gemv(ColiCuda *ctx, float *y, const float *x,
                                   const signed char *q, const float *scales,
                                   int rows, int cols, int row_bytes) {
    if (!ctx || !y || !x || !q || !scales || rows <= 0 || cols <= 0 ||
        row_bytes < cols)
        return fail_arg("coli_cuda_q8_gemv: invalid argument");
    q8_gemv_kernel<<<(rows + 7) / 8, 256, 0, ctx->stream>>>(y, x, q, scales, rows, cols,
                                                  row_bytes);
    return launch_status("q8_gemv_kernel");
}

extern "C" int coli_cuda_q8_gemm(ColiCuda *ctx, float *y, const float *x,
                                   int batch, const signed char *q,
                                   const float *scales, int rows, int cols,
                                   int row_bytes) {
    if (!ctx || !y || !x || !q || !scales || batch <= 0 || rows <= 0 ||
        cols <= 0 || row_bytes < cols)
        return fail_arg("coli_cuda_q8_gemm: invalid argument");
    dim3 grid((rows + 7) / 8, batch);
    q8_gemm_kernel<<<grid, 256, 0, ctx->stream>>>(
        y, x, q, scales, batch, rows, cols, row_bytes);
    return launch_status("q8_gemm_kernel");
}

extern "C" int coli_cuda_q4_gemv(ColiCuda *ctx, float *y, const float *x,
                                   const unsigned char *q, const float *scales,
                                   int rows, int cols, int group_size,
                                   int row_bytes, int groups_per_row) {
    int need_groups = (cols + group_size - 1) / group_size;
    int need_bytes = (cols + 1) / 2;
    if (!ctx || !y || !x || !q || !scales || rows <= 0 || cols <= 0 ||
        group_size <= 0 || groups_per_row < need_groups ||
        row_bytes < need_bytes)
        return fail_arg("coli_cuda_q4_gemv: invalid argument");
    q4_gemv_kernel<<<(rows + 7) / 8, 256, 0, ctx->stream>>>(
        y, x, q, scales, rows, cols, group_size, row_bytes, groups_per_row);
    return launch_status("q4_gemv_kernel");
}

extern "C" int coli_cuda_q4_gemm(ColiCuda *ctx, float *y, const float *x,
                                   int batch, const unsigned char *q,
                                   const float *scales, int rows, int cols,
                                   int group_size, int row_bytes,
                                   int groups_per_row) {
    int need_groups = (cols + group_size - 1) / group_size;
    int need_bytes = (cols + 1) / 2;
    if (!ctx || !y || !x || !q || !scales || batch <= 0 || rows <= 0 ||
        cols <= 0 || group_size <= 0 || groups_per_row < need_groups ||
        row_bytes < need_bytes)
        return fail_arg("coli_cuda_q4_gemm: invalid argument");
    dim3 grid((rows + 7) / 8, batch);
    q4_gemm_kernel<<<grid, 256, 0, ctx->stream>>>(
        y, x, q, scales, batch, rows, cols, group_size, row_bytes,
        groups_per_row);
    return launch_status("q4_gemm_kernel");
}

extern "C" int coli_cuda_q4_gemv_f16(ColiCuda *ctx, float *y,
                                       const unsigned short *x,
                                       const unsigned char *q,
                                       const float *scales, int rows, int cols,
                                       int group_size, int row_bytes,
                                       int groups_per_row) {
    int need_groups = (cols + group_size - 1) / group_size;
    int need_bytes = (cols + 1) / 2;
    if (!ctx || !y || !x || !q || !scales || rows <= 0 || cols <= 0 ||
        group_size <= 0 || groups_per_row < need_groups ||
        row_bytes < need_bytes)
        return fail_arg("coli_cuda_q4_gemv_f16: invalid argument");
    q4_gemv_f16_kernel<<<(rows + 7) / 8, 256, 0, ctx->stream>>>(
        y, (const __half *)x, q, scales, rows, cols, group_size, row_bytes,
        groups_per_row);
    return launch_status("q4_gemv_f16_kernel");
}

extern "C" int coli_cuda_q3_gemv(ColiCuda *ctx, float *y, const float *x,
                                   const unsigned char *q,
                                   const float *scales, int rows, int cols,
                                   int group_size, int row_bytes,
                                   int groups_per_row) {
    int need_groups = (cols + group_size - 1) / group_size;
    int need_bytes = ((cols + 7) / 8) * 3;
    if (!ctx || !y || !x || !q || !scales || rows <= 0 || cols <= 0 ||
        group_size <= 0 || groups_per_row < need_groups ||
        row_bytes < need_bytes)
        return fail_arg("coli_cuda_q3_gemv: invalid argument");
    q3_gemv_kernel<<<(rows + 7) / 8, 256, 0, ctx->stream>>>(
        y, x, q, scales, rows, cols, group_size, row_bytes, groups_per_row);
    return launch_status("q3_gemv_kernel");
}

extern "C" int coli_cuda_q3_gemv_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *q, const float *scales, int rows, int cols,
    int group_size, int row_bytes, int groups_per_row) {
    int need_groups = (cols + group_size - 1) / group_size;
    int need_bytes = ((cols + 7) / 8) * 3;
    if (!ctx || !y || !x || !q || !scales || rows <= 0 || cols <= 0 ||
        group_size <= 0 || groups_per_row < need_groups ||
        row_bytes < need_bytes)
        return fail_arg("coli_cuda_q3_gemv_f16: invalid argument");
    q3_gemv_f16_kernel<<<(rows + 7) / 8, 256, 0, ctx->stream>>>(
        y, (const __half *)x, q, scales, rows, cols, group_size, row_bytes,
        groups_per_row);
    return launch_status("q3_gemv_f16_kernel");
}

static int grouped_qx_mlp_f16(
    int bits, ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int experts, int hidden, int intermediate,
    int group_size, int hidden_row_bytes, int hidden_groups,
    int down_row_bytes, int down_groups) {
    if (!ctx || !y || !x || !gate_q || !gate_s || !up_q || !up_s ||
        !down_q || !down_s || !route_weight || experts <= 0 || hidden <= 0 ||
        intermediate <= 0 || group_size <= 0)
        return fail_arg("grouped qx MLP: invalid argument");
    size_t ptr_bytes = (size_t)experts * 6 * sizeof(void *);
    size_t weight_bytes = (size_t)experts * sizeof(float);
    if (ensure_cuda_buffer((void **)&ctx->group_ptrs, &ctx->group_ptr_cap,
                           ptr_bytes + weight_bytes, "grouped setup allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_gate16,
                           &ctx->group_hidden16_cap,
                           (size_t)experts * intermediate * sizeof(__half),
                           "grouped fp16 hidden allocation"))
        return -1;
    if (experts > 32) return fail_arg("grouped expert count exceeds 32");
    unsigned char setup[32 * 6 * sizeof(void *) + 32 * sizeof(float)];
    const void *host_ptrs[6] = {gate_q, gate_s, up_q, up_s, down_q, down_s};
    for (int i = 0; i < 6; i++)
        memcpy(setup + (size_t)i * experts * sizeof(void *), host_ptrs[i],
               (size_t)experts * sizeof(void *));
    memcpy(setup + ptr_bytes, route_weight, weight_bytes);
    cudaError_t error = cudaMemcpyAsync(ctx->group_ptrs, setup,
                                        ptr_bytes + weight_bytes,
                                        cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess) return fail_cuda("grouped setup upload", error);
    const unsigned char **gq = (const unsigned char **)ctx->group_ptrs;
    const float **gs = (const float **)(ctx->group_ptrs + experts);
    const unsigned char **uq = (const unsigned char **)(ctx->group_ptrs + 2 * experts);
    const float **us = (const float **)(ctx->group_ptrs + 3 * experts);
    const unsigned char **dq = (const unsigned char **)(ctx->group_ptrs + 4 * experts);
    const float **ds = (const float **)(ctx->group_ptrs + 5 * experts);
    dim3 hidden_grid((unsigned)((intermediate + 7) / 8), (unsigned)experts);
    if (bits == 3)
        grouped_hidden_q3_f16_kernel<<<hidden_grid, 256, 0, ctx->stream>>>(
            ctx->group_gate16, (const __half *)x, gq, gs, uq, us,
            intermediate, hidden, group_size, hidden_row_bytes,
            hidden_groups);
    else
        grouped_hidden_q4_f16_kernel<<<hidden_grid, 256, 0, ctx->stream>>>(
            ctx->group_gate16, (const __half *)x, gq, gs, uq, us,
            intermediate, hidden, group_size, hidden_row_bytes,
            hidden_groups);
    if (launch_status(bits == 3 ? "grouped_hidden_q3_f16_kernel" :
                                 "grouped_hidden_q4_f16_kernel")) return -1;
    const float *device_weight = (const float *)((const unsigned char *)ctx->group_ptrs + ptr_bytes);
    if (bits == 3)
        grouped_down_q3_f16_kernel<<<(hidden + 7) / 8, 256, 0,
                                      ctx->stream>>>(
            y, ctx->group_gate16, dq, ds, device_weight, experts, hidden,
            intermediate, group_size, down_row_bytes, down_groups);
    else
        grouped_down_q4_f16_kernel<<<(hidden + 7) / 8, 256, 0,
                                      ctx->stream>>>(
            y, ctx->group_gate16, dq, ds, device_weight, experts, hidden,
            intermediate, group_size, down_row_bytes, down_groups);
    return launch_status(bits == 3 ? "grouped_down_q3_f16_kernel" :
                                     "grouped_down_q4_f16_kernel");
}

extern "C" int coli_cuda_grouped_q4_mlp_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int experts, int hidden, int intermediate,
    int group_size, int hidden_row_bytes, int hidden_groups,
    int down_row_bytes, int down_groups) {
    return grouped_qx_mlp_f16(4, ctx, y, x, gate_q, gate_s, up_q, up_s,
        down_q, down_s, route_weight, experts, hidden, intermediate,
        group_size, hidden_row_bytes, hidden_groups, down_row_bytes,
        down_groups);
}

extern "C" int coli_cuda_grouped_q3_mlp_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int experts, int hidden, int intermediate,
    int group_size, int hidden_row_bytes, int hidden_groups,
    int down_row_bytes, int down_groups) {
    return grouped_qx_mlp_f16(3, ctx, y, x, gate_q, gate_s, up_q, up_s,
        down_q, down_s, route_weight, experts, hidden, intermediate,
        group_size, hidden_row_bytes, hidden_groups, down_row_bytes,
        down_groups);
}

static int grouped_qx_mlp_batch_f16(
    int bits, ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int batch, int experts_per_token, int hidden,
    int intermediate, int group_size, int hidden_row_bytes,
    int hidden_groups, int down_row_bytes, int down_groups) {
    int routes = batch * experts_per_token;
    if (!ctx || !y || !x || !gate_q || !gate_s || !up_q || !up_s ||
        !down_q || !down_s || !route_weight || batch <= 0 || batch > 8 ||
        experts_per_token <= 0 || routes > 64 || hidden <= 0 ||
        intermediate <= 0 || group_size <= 0)
        return fail_arg(
            "grouped qx batch MLP: invalid argument");

    const unsigned char *compact_gq[64], *compact_uq[64], *compact_dq[64];
    const float *compact_gs[64], *compact_us[64], *compact_ds[64];
    int route_map[64 * 8];
    for (int i = 0; i < 64 * 8; i++) route_map[i] = -1;
    int unique = 0;
    for (int route = 0; route < routes; route++) {
        int expert = -1;
        for (int u = 0; u < unique; u++)
            if (compact_gq[u] == gate_q[route] &&
                compact_gs[u] == gate_s[route] &&
                compact_uq[u] == up_q[route] &&
                compact_us[u] == up_s[route] &&
                compact_dq[u] == down_q[route] &&
                compact_ds[u] == down_s[route] &&
                route_map[u * batch + route / experts_per_token] < 0) {
                expert = u;
                break;
            }
        if (expert < 0) {
            expert = unique++;
            compact_gq[expert] = gate_q[route];
            compact_gs[expert] = gate_s[route];
            compact_uq[expert] = up_q[route];
            compact_us[expert] = up_s[route];
            compact_dq[expert] = down_q[route];
            compact_ds[expert] = down_s[route];
        }
        int sample = route / experts_per_token;
        route_map[expert * batch + sample] = route;
    }

    size_t ptr_bytes = (size_t)unique * 6 * sizeof(void *);
    size_t map_bytes = (size_t)unique * batch * sizeof(int);
    size_t weight_bytes = (size_t)routes * sizeof(float);
    size_t setup_bytes = ptr_bytes + map_bytes + weight_bytes;
    if (ensure_cuda_buffer((void **)&ctx->group_ptrs, &ctx->group_ptr_cap,
                           setup_bytes, "grouped batch setup allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_gate16,
                           &ctx->group_hidden16_cap,
                           (size_t)routes * intermediate * sizeof(__half),
                           "grouped batch fp16 hidden allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_route_out,
                           &ctx->group_route_out_cap,
                           (size_t)routes * hidden * sizeof(float),
                           "grouped batch route output allocation"))
        return -1;

    unsigned char setup[64 * 6 * sizeof(void *) +
                        64 * 8 * sizeof(int) + 64 * sizeof(float)];
    const void *host_ptrs[6] = {compact_gq, compact_gs, compact_uq,
                                compact_us, compact_dq, compact_ds};
    for (int i = 0; i < 6; i++)
        memcpy(setup + (size_t)i * unique * sizeof(void *), host_ptrs[i],
               (size_t)unique * sizeof(void *));
    memcpy(setup + ptr_bytes, route_map, map_bytes);
    memcpy(setup + ptr_bytes + map_bytes, route_weight, weight_bytes);
    if (ctx->profile_stages) {
        cudaEventRecord(ctx->profile_event[0], ctx->stream);
        ctx->profile_pending = 0;
        ctx->profile_shared_marked = 0;
        ctx->profile_download_marked = 0;
    }
    cudaError_t error = cudaMemcpyAsync(ctx->group_ptrs, setup, setup_bytes,
                                        cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
        return fail_cuda("grouped batch setup upload", error);
    if (ctx->profile_stages)
        cudaEventRecord(ctx->profile_event[1], ctx->stream);

    const unsigned char **gq = (const unsigned char **)ctx->group_ptrs;
    const float **gs = (const float **)(ctx->group_ptrs + unique);
    const unsigned char **uq =
        (const unsigned char **)(ctx->group_ptrs + 2 * unique);
    const float **us = (const float **)(ctx->group_ptrs + 3 * unique);
    const unsigned char **dq =
        (const unsigned char **)(ctx->group_ptrs + 4 * unique);
    const float **ds = (const float **)(ctx->group_ptrs + 5 * unique);
    const int *device_route_map =
        (const int *)((const unsigned char *)ctx->group_ptrs + ptr_bytes);
    const float *device_weight = (const float *)(
        (const unsigned char *)ctx->group_ptrs + ptr_bytes + map_bytes);

    dim3 hidden_grid((unsigned)((intermediate + 7) / 8), (unsigned)unique);
    if (bits == 3)
        grouped_hidden_batch_q3_f16_kernel<<<hidden_grid, 256, 0,
                                             ctx->stream>>>(
            ctx->group_gate16, (const __half *)x, gq, gs, uq, us,
            device_route_map, unique, batch, intermediate, hidden,
            group_size, hidden_row_bytes, hidden_groups);
    else
        grouped_hidden_batch_q4_f16_kernel<<<hidden_grid, 256, 0,
                                             ctx->stream>>>(
            ctx->group_gate16, (const __half *)x, gq, gs, uq, us,
            device_route_map, unique, batch, intermediate, hidden,
            group_size, hidden_row_bytes, hidden_groups);
    if (launch_status(bits == 3 ? "grouped_hidden_batch_q3_f16_kernel" :
                                 "grouped_hidden_batch_q4_f16_kernel"))
        return -1;
    if (ctx->profile_stages)
        cudaEventRecord(ctx->profile_event[2], ctx->stream);
    if (bits == 3)
        grouped_down_batch_q3_f16_kernel<<<(hidden + 7) / 8, 256, 0,
                                           ctx->stream>>>(
            ctx->group_route_out, ctx->group_gate16, dq, ds,
            device_route_map, unique, batch, hidden, intermediate,
            group_size, down_row_bytes, down_groups);
    else
        grouped_down_batch_q4_f16_kernel<<<(hidden + 7) / 8, 256, 0,
                                           ctx->stream>>>(
            ctx->group_route_out, ctx->group_gate16, dq, ds,
            device_route_map, unique, batch, hidden, intermediate,
            group_size, down_row_bytes, down_groups);
    if (ctx->profile_stages)
        cudaEventRecord(ctx->profile_event[3], ctx->stream);
    dim3 reduce_grid((unsigned)((hidden + 255) / 256), (unsigned)batch);
    grouped_reduce_batch_kernel<<<reduce_grid, 256, 0, ctx->stream>>>(
        y, ctx->group_route_out, device_weight, batch, experts_per_token,
        hidden);
    int status = launch_status("grouped weight-reuse batch kernels");
    if (!status && ctx->profile_stages) {
        cudaEventRecord(ctx->profile_event[4], ctx->stream);
        ctx->profile_pending = 1;
    }
    return status;
}

extern "C" int coli_cuda_grouped_q4_mlp_batch_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int batch, int experts_per_token, int hidden,
    int intermediate, int group_size, int hidden_row_bytes,
    int hidden_groups, int down_row_bytes, int down_groups) {
    return grouped_qx_mlp_batch_f16(4, ctx, y, x, gate_q, gate_s, up_q, up_s,
        down_q, down_s, route_weight, batch, experts_per_token, hidden,
        intermediate, group_size, hidden_row_bytes, hidden_groups,
        down_row_bytes, down_groups);
}

extern "C" int coli_cuda_grouped_q3_mlp_batch_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int batch, int experts_per_token, int hidden,
    int intermediate, int group_size, int hidden_row_bytes,
    int hidden_groups, int down_row_bytes, int down_groups) {
    return grouped_qx_mlp_batch_f16(3, ctx, y, x, gate_q, gate_s, up_q, up_s,
        down_q, down_s, route_weight, batch, experts_per_token, hidden,
        intermediate, group_size, hidden_row_bytes, hidden_groups,
        down_row_bytes, down_groups);
}

extern "C" int coli_cuda_shared_q4_mlp_batch_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *gate_q, const float *gate_s,
    const unsigned char *up_q, const float *up_s,
    const unsigned char *down_q, const float *down_s,
    const unsigned char *scale_q, const float *scale_s,
    int batch, int hidden, int intermediate, int group_size,
    int hidden_row_bytes, int hidden_groups, int down_row_bytes,
    int down_groups, int scale_row_bytes, int scale_groups) {
    if (!ctx || !y || !x || !gate_q || !gate_s || !up_q || !up_s ||
        !down_q || !down_s || !scale_q || !scale_s || batch <= 0 ||
        batch > 8 || hidden <= 0 || intermediate <= 0 || group_size <= 0)
        return fail_arg("coli_cuda_shared_q4_mlp_batch_f16: invalid argument");
    if (ensure_cuda_buffer((void **)&ctx->group_gate16,
                           &ctx->group_hidden16_cap,
                           (size_t)batch * intermediate * sizeof(__half),
                           "shared batch hidden allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_gate, &ctx->group_hidden_cap,
                           (size_t)batch * sizeof(float),
                           "shared batch gate allocation"))
        return -1;
    dim3 hidden_grid((unsigned)((intermediate + 7) / 8));
    shared_hidden_batch_q4_f16_kernel<<<hidden_grid, 256, 0, ctx->stream>>>(
        ctx->group_gate16, (const __half *)x, gate_q, gate_s, up_q, up_s,
        batch, intermediate, hidden, group_size, hidden_row_bytes,
        hidden_groups);
    if (ctx->profile_pending)
        cudaEventRecord(ctx->profile_event[5], ctx->stream);
    dim3 scale_grid(1, (unsigned)batch);
    q4_gemm_f16_kernel<<<scale_grid, 256, 0, ctx->stream>>>(
        ctx->group_gate, (const __half *)x, scale_q, scale_s, batch, 1,
        hidden, group_size, scale_row_bytes, scale_groups);
    if (ctx->profile_pending)
        cudaEventRecord(ctx->profile_event[6], ctx->stream);
    dim3 down_grid((unsigned)((hidden + 7) / 8));
    shared_down_batch_q4_f16_kernel<<<down_grid, 256, 0, ctx->stream>>>(
        y, ctx->group_gate16, down_q, down_s, ctx->group_gate, batch,
        hidden, intermediate, group_size, down_row_bytes, down_groups);
    int status = launch_status("shared batch q4 MLP kernels");
    if (!status && ctx->profile_pending) {
        cudaEventRecord(ctx->profile_event[7], ctx->stream);
        ctx->profile_shared_marked = 1;
    }
    return status;
}

static int shared_q8_mlp_batch_impl(
    ColiCuda *ctx, float *y, const float *x, const float *scale,
    const signed char *gate_q, const float *gate_s, int gate_row_bytes,
    const signed char *up_q, const float *up_s, int up_row_bytes,
    const signed char *down_q, const float *down_s, int down_row_bytes,
    int batch, int hidden, int intermediate, int device_scale) {
    if (!ctx || !y || !x || !scale || !gate_q || !gate_s || !up_q || !up_s ||
        !down_q || !down_s || batch <= 0 || batch > 8 || hidden <= 0 ||
        intermediate <= 0 || gate_row_bytes < hidden ||
        up_row_bytes < hidden || down_row_bytes < intermediate)
        return fail_arg("coli_cuda_shared_q8_mlp_batch: invalid argument");
    if (ensure_cuda_buffer((void **)&ctx->group_gate,
                           &ctx->group_hidden_cap,
                           (size_t)batch * intermediate * sizeof(float),
                           "shared q8 gate allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_up, &ctx->group_up_cap,
                           (size_t)batch * intermediate * sizeof(float),
                           "shared q8 up allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_weight,
                           &ctx->group_weight_cap,
                           (size_t)batch * sizeof(float),
                           "shared q8 scale allocation"))
        return -1;
    dim3 hidden_grid((unsigned)((intermediate + 7) / 8),
                     (unsigned)batch);
    q8_gemm_kernel<<<hidden_grid, 256, 0, ctx->stream>>>(
        ctx->group_gate, x, gate_q, gate_s, batch, intermediate, hidden,
        gate_row_bytes);
    q8_gemm_kernel<<<hidden_grid, 256, 0, ctx->stream>>>(
        ctx->group_up, x, up_q, up_s, batch, intermediate, hidden,
        up_row_bytes);
    silu_mul_kernel<<<((batch * intermediate) + 255) / 256, 256, 0,
                       ctx->stream>>>(ctx->group_gate, ctx->group_gate,
                                     ctx->group_up, batch * intermediate);
    if (ctx->profile_pending)
        cudaEventRecord(ctx->profile_event[5], ctx->stream);
    const float *scale_input = scale;
    cudaError_t error = cudaSuccess;
    if (!device_scale) {
        error = cudaMemcpyAsync(
            ctx->group_weight, scale, (size_t)batch * sizeof(float),
            cudaMemcpyHostToDevice, ctx->stream);
        if (error != cudaSuccess)
            return fail_cuda("shared q8 scale upload", error);
        scale_input = ctx->group_weight;
    }
    if (ctx->profile_pending)
        cudaEventRecord(ctx->profile_event[6], ctx->stream);
    dim3 down_grid((unsigned)((hidden + 7) / 8), (unsigned)batch);
    q8_gemm_scaled_add_kernel<<<down_grid, 256, 0, ctx->stream>>>(
        y, ctx->group_gate, down_q, down_s, scale_input, batch, hidden,
        intermediate, down_row_bytes);
    int status = launch_status("shared batch q8 MLP kernels");
    if (!status && ctx->profile_pending) {
        cudaEventRecord(ctx->profile_event[7], ctx->stream);
        ctx->profile_shared_marked = 1;
    }
    return status;
}

extern "C" int coli_cuda_shared_q8_mlp_batch(
    ColiCuda *ctx, float *y, const float *x, const float *scale,
    const signed char *gate_q, const float *gate_s, int gate_row_bytes,
    const signed char *up_q, const float *up_s, int up_row_bytes,
    const signed char *down_q, const float *down_s, int down_row_bytes,
    int batch, int hidden, int intermediate) {
    return shared_q8_mlp_batch_impl(
        ctx,y,x,scale,gate_q,gate_s,gate_row_bytes,up_q,up_s,up_row_bytes,
        down_q,down_s,down_row_bytes,batch,hidden,intermediate,0);
}

extern "C" int coli_cuda_shared_q8_mlp_batch_device_scale(
    ColiCuda *ctx, float *y, const float *x, const float *device_scale,
    const signed char *gate_q, const float *gate_s, int gate_row_bytes,
    const signed char *up_q, const float *up_s, int up_row_bytes,
    const signed char *down_q, const float *down_s, int down_row_bytes,
    int batch, int hidden, int intermediate) {
    return shared_q8_mlp_batch_impl(
        ctx,y,x,device_scale,gate_q,gate_s,gate_row_bytes,up_q,up_s,
        up_row_bytes,down_q,down_s,down_row_bytes,batch,hidden,intermediate,1);
}

extern "C" int coli_cuda_shared_q4_mlp_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *gate_q, const float *gate_s,
    const unsigned char *up_q, const float *up_s,
    const unsigned char *down_q, const float *down_s,
    const unsigned char *scale_q, const float *scale_s,
    int hidden, int intermediate, int group_size, int hidden_row_bytes,
    int hidden_groups, int down_row_bytes, int down_groups,
    int scale_row_bytes, int scale_groups) {
    if (!ctx || !y || !x || !gate_q || !gate_s || !up_q || !up_s ||
        !down_q || !down_s || !scale_q || !scale_s || hidden <= 0 ||
        intermediate <= 0 || group_size <= 0)
        return fail_arg("coli_cuda_shared_q4_mlp_f16: invalid argument");
    if (ensure_cuda_buffer((void **)&ctx->group_gate16,
                           &ctx->group_hidden16_cap,
                           (size_t)intermediate * sizeof(__half),
                           "shared hidden allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_gate, &ctx->group_hidden_cap,
                           sizeof(float), "shared gate allocation"))
        return -1;
    shared_hidden_q4_f16_kernel<<<(intermediate + 7) / 8, 256, 0,
                                   ctx->stream>>>(
        ctx->group_gate16, (const __half *)x, gate_q, gate_s, up_q, up_s,
        intermediate, hidden, group_size, hidden_row_bytes, hidden_groups);
    q4_gemv_f16_kernel<<<1, 256, 0, ctx->stream>>>(
        ctx->group_gate, (const __half *)x, scale_q, scale_s, 1, hidden,
        group_size, scale_row_bytes, scale_groups);
    q4_gemv_f16_sigmoid_axpy_kernel<<<(hidden + 7) / 8, 256, 0,
                                      ctx->stream>>>(
        y, ctx->group_gate16, down_q, down_s, ctx->group_gate, hidden,
        intermediate, group_size, down_row_bytes, down_groups);
    return launch_status("shared q4 MLP kernels");
}

extern "C" int coli_cuda_gdn_decode_q4_f16(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    const unsigned char *qkv_q, const float *qkv_s, int qkv_row_bytes,
    int qkv_groups, const unsigned char *z_q, const float *z_s,
    int z_row_bytes, int z_groups, const unsigned char *out_q,
    const float *out_s, int out_row_bytes, int out_groups,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *norm_weight, float *conv_state, float *recurrent_state,
    int position, int hidden, int key_heads, int value_heads, int key_dim,
    int value_dim, int conv_kernel, int group_size, float eps) {
    if (!ctx || !out || !x || !a || !b || !qkv_q || !qkv_s || !z_q ||
        !z_s || !out_q || !out_s || !conv_weight || !a_log || !dt_bias ||
        !norm_weight || !conv_state || !recurrent_state || position < 0 ||
        hidden <= 0 || key_heads <= 0 || value_heads <= 0 || key_dim <= 0 ||
        value_dim <= 0 || value_dim > 256 || value_heads % key_heads ||
        conv_kernel <= 0 || group_size <= 0)
        return fail_arg("coli_cuda_gdn_decode_q4_f16: invalid argument");
    int key_total = key_heads * key_dim;
    int value_total = value_heads * value_dim;
    int channels = 2 * key_total + value_total;
#define GDN_RESERVE(member, cap, bytes, label) \
    ensure_cuda_buffer((void **)&ctx->member, &ctx->cap, (bytes), (label))
    if (GDN_RESERVE(gdn_x, gdn_x_cap, (size_t)hidden * sizeof(float),
                    "GDN input allocation") ||
        GDN_RESERVE(gdn_x16, gdn_x16_cap,
                    (size_t)hidden * sizeof(__half), "GDN fp16 input allocation") ||
        GDN_RESERVE(gdn_raw, gdn_raw_cap,
                    (size_t)channels * sizeof(float), "GDN raw allocation") ||
        GDN_RESERVE(gdn_mix, gdn_mix_cap,
                    (size_t)channels * sizeof(float), "GDN mix allocation") ||
        GDN_RESERVE(gdn_z, gdn_z_cap,
                    (size_t)value_total * sizeof(float), "GDN z allocation") ||
        GDN_RESERVE(gdn_a, gdn_a_cap,
                    (size_t)value_heads * sizeof(float), "GDN a allocation") ||
        GDN_RESERVE(gdn_b, gdn_b_cap,
                    (size_t)value_heads * sizeof(float), "GDN b allocation") ||
        GDN_RESERVE(gdn_core, gdn_core_cap,
                    (size_t)value_total * sizeof(float), "GDN core allocation") ||
        GDN_RESERVE(gdn_core16, gdn_core16_cap,
                    (size_t)value_total * sizeof(__half),
                    "GDN fp16 core allocation"))
        return -1;
#undef GDN_RESERVE
    cudaError_t error = cudaMemcpyAsync(ctx->gdn_x, x,
                                        (size_t)hidden * sizeof(float),
                                        cudaMemcpyHostToDevice, ctx->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(ctx->gdn_a, a,
                                (size_t)value_heads * sizeof(float),
                                cudaMemcpyHostToDevice, ctx->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(ctx->gdn_b, b,
                                (size_t)value_heads * sizeof(float),
                                cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess) return fail_cuda("GDN input upload", error);
    f32_to_f16_kernel<<<(hidden + 255) / 256, 256, 0, ctx->stream>>>(
        ctx->gdn_x16, ctx->gdn_x, hidden);
    q4_gemv_f16_kernel<<<(channels + 7) / 8, 256, 0, ctx->stream>>>(
        ctx->gdn_raw, ctx->gdn_x16, qkv_q, qkv_s, channels, hidden,
        group_size, qkv_row_bytes, qkv_groups);
    q4_gemv_f16_kernel<<<(value_total + 7) / 8, 256, 0, ctx->stream>>>(
        ctx->gdn_z, ctx->gdn_x16, z_q, z_s, value_total, hidden, group_size,
        z_row_bytes, z_groups);
    error = cudaMemcpyAsync(conv_state +
                                (size_t)(position % conv_kernel) * channels,
                            ctx->gdn_raw, (size_t)channels * sizeof(float),
                            cudaMemcpyDeviceToDevice, ctx->stream);
    if (error != cudaSuccess) return fail_cuda("GDN convolution state", error);
    gdn_conv_kernel<<<(channels + 255) / 256, 256, 0, ctx->stream>>>(
        ctx->gdn_mix, conv_state, conv_weight, position, channels, conv_kernel);
    gdn_recurrent_kernel<<<value_heads, value_dim, 0, ctx->stream>>>(
        ctx->gdn_core, recurrent_state, ctx->gdn_mix, ctx->gdn_z, ctx->gdn_a,
        ctx->gdn_b, a_log, dt_bias, norm_weight, key_heads, value_heads,
        key_dim, value_dim, eps);
    f32_to_f16_kernel<<<(value_total + 255) / 256, 256, 0, ctx->stream>>>(
        ctx->gdn_core16, ctx->gdn_core, value_total);
    q4_gemv_f16_kernel<<<(hidden + 7) / 8, 256, 0, ctx->stream>>>(
        ctx->gdn_x, ctx->gdn_core16, out_q, out_s, hidden, value_total,
        group_size, out_row_bytes, out_groups);
    if (launch_status("GDN decode kernels")) return -1;
    error = cudaMemcpyAsync(out, ctx->gdn_x, (size_t)hidden * sizeof(float),
                            cudaMemcpyDeviceToHost, ctx->stream);
    if (error != cudaSuccess) return fail_cuda("GDN output download", error);
    error = cudaStreamSynchronize(ctx->stream);
    return error == cudaSuccess ? 0 : fail_cuda("GDN synchronize", error);
}

extern "C" int coli_cuda_gdn_block_q4_f16(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    int batch, const unsigned char *qkv_q, const float *qkv_s,
    int qkv_row_bytes, int qkv_groups, const unsigned char *z_q,
    const float *z_s, int z_row_bytes, int z_groups,
    const unsigned char *out_q, const float *out_s, int out_row_bytes,
    int out_groups, const float *conv_weight, const float *a_log,
    const float *dt_bias, const float *norm_weight, float *conv_state,
    float *recurrent_state, int position, int hidden, int key_heads,
    int value_heads, int key_dim, int value_dim, int conv_kernel,
    int group_size, float eps) {
    if (!ctx || !out || !x || !a || !b || batch < 1 || batch > 8 ||
        !qkv_q || !qkv_s || !z_q || !z_s || !out_q || !out_s ||
        !conv_weight || !a_log || !dt_bias || !norm_weight || !conv_state ||
        !recurrent_state || position < 0 || hidden <= 0 || key_heads <= 0 ||
        value_heads <= 0 || key_dim <= 0 || value_dim <= 0 ||
        conv_kernel <= 0 || group_size <= 0)
        return fail_arg("coli_cuda_gdn_block_q4_f16: invalid argument");
    int key_total = key_heads * key_dim;
    int value_total = value_heads * value_dim;
    int channels = 2 * key_total + value_total;
#define GDN_BLOCK_RESERVE(member, cap, bytes, label) \
    ensure_cuda_buffer((void **)&ctx->member, &ctx->cap, (bytes), (label))
    if (GDN_BLOCK_RESERVE(gdn_x, gdn_x_cap,
                          (size_t)batch * hidden * sizeof(float),
                          "GDN block input allocation") ||
        GDN_BLOCK_RESERVE(gdn_x16, gdn_x16_cap,
                          (size_t)batch * hidden * sizeof(__half),
                          "GDN block fp16 input allocation") ||
        GDN_BLOCK_RESERVE(gdn_raw, gdn_raw_cap,
                          (size_t)batch * channels * sizeof(float),
                          "GDN block raw allocation") ||
        GDN_BLOCK_RESERVE(gdn_mix, gdn_mix_cap,
                          (size_t)channels * sizeof(float),
                          "GDN block mix allocation") ||
        GDN_BLOCK_RESERVE(gdn_z, gdn_z_cap,
                          (size_t)batch * value_total * sizeof(float),
                          "GDN block z allocation") ||
        GDN_BLOCK_RESERVE(gdn_a, gdn_a_cap,
                          (size_t)batch * value_heads * sizeof(float),
                          "GDN block a allocation") ||
        GDN_BLOCK_RESERVE(gdn_b, gdn_b_cap,
                          (size_t)batch * value_heads * sizeof(float),
                          "GDN block b allocation") ||
        GDN_BLOCK_RESERVE(gdn_core, gdn_core_cap,
                          (size_t)batch * value_total * sizeof(float),
                          "GDN block core allocation") ||
        GDN_BLOCK_RESERVE(gdn_core16, gdn_core16_cap,
                          (size_t)batch * value_total * sizeof(__half),
                          "GDN block fp16 core allocation"))
        return -1;
#undef GDN_BLOCK_RESERVE
    cudaError_t error = cudaMemcpyAsync(ctx->gdn_x, x,
        (size_t)batch * hidden * sizeof(float), cudaMemcpyHostToDevice,
        ctx->stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(ctx->gdn_a, a,
        (size_t)batch * value_heads * sizeof(float), cudaMemcpyHostToDevice,
        ctx->stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(ctx->gdn_b, b,
        (size_t)batch * value_heads * sizeof(float), cudaMemcpyHostToDevice,
        ctx->stream);
    if (error != cudaSuccess) return fail_cuda("GDN block input upload", error);
    int input_count = batch * hidden;
    f32_to_f16_kernel<<<(input_count + 255) / 256, 256, 0, ctx->stream>>>(
        ctx->gdn_x16, ctx->gdn_x, input_count);
    dim3 qkv_grid((channels + 7) / 8, batch);
    dim3 z_grid((value_total + 7) / 8, batch);
    q4_gemm_f16_kernel<<<qkv_grid, 256, 0, ctx->stream>>>(
        ctx->gdn_raw, ctx->gdn_x16, qkv_q, qkv_s, batch, channels, hidden,
        group_size, qkv_row_bytes, qkv_groups);
    q4_gemm_f16_kernel<<<z_grid, 256, 0, ctx->stream>>>(
        ctx->gdn_z, ctx->gdn_x16, z_q, z_s, batch, value_total, hidden,
        group_size, z_row_bytes, z_groups);
    for (int t = 0; t < batch; t++) {
        int pos = position + t;
        error = cudaMemcpyAsync(conv_state +
                    (size_t)(pos % conv_kernel) * channels,
                    ctx->gdn_raw + (size_t)t * channels,
                    (size_t)channels * sizeof(float), cudaMemcpyDeviceToDevice,
                    ctx->stream);
        if (error != cudaSuccess) return fail_cuda("GDN block convolution state", error);
        gdn_conv_kernel<<<(channels + 255) / 256, 256, 0, ctx->stream>>>(
            ctx->gdn_mix, conv_state, conv_weight, pos, channels, conv_kernel);
        gdn_recurrent_kernel<<<value_heads, value_dim, 0, ctx->stream>>>(
            ctx->gdn_core + (size_t)t * value_total, recurrent_state,
            ctx->gdn_mix, ctx->gdn_z + (size_t)t * value_total,
            ctx->gdn_a + (size_t)t * value_heads,
            ctx->gdn_b + (size_t)t * value_heads, a_log, dt_bias, norm_weight,
            key_heads, value_heads, key_dim, value_dim, eps);
    }
    int core_count = batch * value_total;
    f32_to_f16_kernel<<<(core_count + 255) / 256, 256, 0, ctx->stream>>>(
        ctx->gdn_core16, ctx->gdn_core, core_count);
    dim3 out_grid((hidden + 7) / 8, batch);
    q4_gemm_f16_kernel<<<out_grid, 256, 0, ctx->stream>>>(
        ctx->gdn_x, ctx->gdn_core16, out_q, out_s, batch, hidden, value_total,
        group_size, out_row_bytes, out_groups);
    if (launch_status("GDN block kernels")) return -1;
    error = cudaMemcpyAsync(out, ctx->gdn_x,
                            (size_t)batch * hidden * sizeof(float),
                            cudaMemcpyDeviceToHost, ctx->stream);
    if (error != cudaSuccess) return fail_cuda("GDN block output download", error);
    error = cudaStreamSynchronize(ctx->stream);
    return error == cudaSuccess ? 0 : fail_cuda("GDN block synchronize", error);
}

static int gdn_slots_q4_f16_impl(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    int batch, const int *slot, const int *position,
    const unsigned char *qkv_q, const float *qkv_s, int qkv_row_bytes,
    int qkv_groups, const unsigned char *z_q, const float *z_s,
    int z_row_bytes, int z_groups, const unsigned char *out_q,
    const float *out_s, int out_row_bytes, int out_groups,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *norm_weight, float *conv_state, float *recurrent_state,
    int slots, int hidden, int key_heads, int value_heads, int key_dim,
    int value_dim, int conv_kernel, int group_size, float eps,
    int device_io) {
    if (!ctx || !out || !x || !a || !b || batch < 1 || batch > 16 ||
        !slot || !position || !qkv_q || !qkv_s || !z_q || !z_s || !out_q ||
        !out_s || !conv_weight || !a_log || !dt_bias || !norm_weight ||
        !conv_state || !recurrent_state || slots < 1 || hidden <= 0 ||
        key_heads <= 0 || value_heads <= 0 || key_dim <= 0 ||
        value_dim <= 0 || conv_kernel <= 0 || group_size <= 0)
        return fail_arg("coli_cuda_gdn_slots_q4_f16: invalid argument");
    for (int row = 0; row < batch; row++)
        if (slot[row] < 0 || slot[row] >= slots || position[row] < 0)
            return fail_arg("coli_cuda_gdn_slots_q4_f16: invalid slot");
    int key_total = key_heads * key_dim;
    int value_total = value_heads * value_dim;
    int channels = 2 * key_total + value_total;
    size_t conv_stride = (size_t)conv_kernel * channels;
    size_t recurrent_stride = (size_t)value_heads * key_dim * value_dim;
#define GDN_SLOTS_RESERVE(member, cap, bytes, label) \
    ensure_cuda_buffer((void **)&ctx->member, &ctx->cap, (bytes), (label))
    if (GDN_SLOTS_RESERVE(gdn_x, gdn_x_cap,
                          (size_t)batch * hidden * sizeof(float),
                          "GDN slots input allocation") ||
        GDN_SLOTS_RESERVE(gdn_x16, gdn_x16_cap,
                          (size_t)batch * hidden * sizeof(__half),
                          "GDN slots fp16 input allocation") ||
        GDN_SLOTS_RESERVE(gdn_raw, gdn_raw_cap,
                          (size_t)batch * channels * sizeof(float),
                          "GDN slots raw allocation") ||
        GDN_SLOTS_RESERVE(gdn_mix, gdn_mix_cap,
                          (size_t)batch * channels * sizeof(float),
                          "GDN slots mix allocation") ||
        GDN_SLOTS_RESERVE(gdn_z, gdn_z_cap,
                          (size_t)batch * value_total * sizeof(float),
                          "GDN slots z allocation") ||
        GDN_SLOTS_RESERVE(gdn_a, gdn_a_cap,
                          (size_t)batch * value_heads * sizeof(float),
                          "GDN slots a allocation") ||
        GDN_SLOTS_RESERVE(gdn_b, gdn_b_cap,
                          (size_t)batch * value_heads * sizeof(float),
                          "GDN slots b allocation") ||
        GDN_SLOTS_RESERVE(gdn_core, gdn_core_cap,
                          (size_t)batch * value_total * sizeof(float),
                          "GDN slots core allocation") ||
        GDN_SLOTS_RESERVE(gdn_core16, gdn_core16_cap,
                          (size_t)batch * value_total * sizeof(__half),
                          "GDN slots fp16 core allocation"))
        return -1;
#undef GDN_SLOTS_RESERVE
    cudaMemcpyKind input_kind =
        device_io ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaError_t error = cudaMemcpyAsync(
        ctx->gdn_x, x, (size_t)batch * hidden * sizeof(float),
        input_kind, ctx->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(
            ctx->gdn_a, a, (size_t)batch * value_heads * sizeof(float),
            input_kind, ctx->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(
            ctx->gdn_b, b, (size_t)batch * value_heads * sizeof(float),
            input_kind, ctx->stream);
    if (error != cudaSuccess)
        return fail_cuda("GDN slots input upload", error);
    int input_count = batch * hidden;
    f32_to_f16_kernel<<<(input_count + 255) / 256, 256, 0, ctx->stream>>>(
        ctx->gdn_x16, ctx->gdn_x, input_count);
    dim3 qkv_grid((channels + 7) / 8, batch);
    dim3 z_grid((value_total + 7) / 8, batch);
    q4_gemm_f16_kernel<<<qkv_grid, 256, 0, ctx->stream>>>(
        ctx->gdn_raw, ctx->gdn_x16, qkv_q, qkv_s, batch, channels, hidden,
        group_size, qkv_row_bytes, qkv_groups);
    q4_gemm_f16_kernel<<<z_grid, 256, 0, ctx->stream>>>(
        ctx->gdn_z, ctx->gdn_x16, z_q, z_s, batch, value_total, hidden,
        group_size, z_row_bytes, z_groups);
    for (int row = 0; row < batch; row++) {
        float *slot_conv = conv_state + (size_t)slot[row] * conv_stride;
        float *slot_recurrent =
            recurrent_state + (size_t)slot[row] * recurrent_stride;
        int pos = position[row];
        error = cudaMemcpyAsync(
            slot_conv + (size_t)(pos % conv_kernel) * channels,
            ctx->gdn_raw + (size_t)row * channels,
            (size_t)channels * sizeof(float), cudaMemcpyDeviceToDevice,
            ctx->stream);
        if (error != cudaSuccess)
            return fail_cuda("GDN slots convolution state", error);
        gdn_conv_kernel<<<(channels + 255) / 256, 256, 0, ctx->stream>>>(
            ctx->gdn_mix + (size_t)row * channels, slot_conv, conv_weight,
            pos, channels, conv_kernel);
        gdn_recurrent_kernel<<<value_heads, value_dim, 0, ctx->stream>>>(
            ctx->gdn_core + (size_t)row * value_total, slot_recurrent,
            ctx->gdn_mix + (size_t)row * channels,
            ctx->gdn_z + (size_t)row * value_total,
            ctx->gdn_a + (size_t)row * value_heads,
            ctx->gdn_b + (size_t)row * value_heads, a_log, dt_bias,
            norm_weight, key_heads, value_heads, key_dim, value_dim, eps);
    }
    int core_count = batch * value_total;
    f32_to_f16_kernel<<<(core_count + 255) / 256, 256, 0, ctx->stream>>>(
        ctx->gdn_core16, ctx->gdn_core, core_count);
    dim3 out_grid((hidden + 7) / 8, batch);
    q4_gemm_f16_kernel<<<out_grid, 256, 0, ctx->stream>>>(
        ctx->gdn_x, ctx->gdn_core16, out_q, out_s, batch, hidden,
        value_total, group_size, out_row_bytes, out_groups);
    if (launch_status("GDN slots kernels")) return -1;
    error = cudaMemcpyAsync(
        out, ctx->gdn_x, (size_t)batch * hidden * sizeof(float),
        device_io ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost,
        ctx->stream);
    if (error != cudaSuccess)
        return fail_cuda("GDN slots output download", error);
    if (device_io) return 0;
    error = cudaStreamSynchronize(ctx->stream);
    return error == cudaSuccess ? 0 : fail_cuda("GDN slots synchronize", error);
}

extern "C" int coli_cuda_gdn_slots_q4_f16(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    int batch, const int *slot, const int *position,
    const unsigned char *qkv_q, const float *qkv_s, int qkv_row_bytes,
    int qkv_groups, const unsigned char *z_q, const float *z_s,
    int z_row_bytes, int z_groups, const unsigned char *out_q,
    const float *out_s, int out_row_bytes, int out_groups,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *norm_weight, float *conv_state, float *recurrent_state,
    int slots, int hidden, int key_heads, int value_heads, int key_dim,
    int value_dim, int conv_kernel, int group_size, float eps) {
    return gdn_slots_q4_f16_impl(
        ctx, out, x, a, b, batch, slot, position, qkv_q, qkv_s,
        qkv_row_bytes, qkv_groups, z_q, z_s, z_row_bytes, z_groups, out_q,
        out_s, out_row_bytes, out_groups, conv_weight, a_log, dt_bias,
        norm_weight, conv_state, recurrent_state, slots, hidden, key_heads,
        value_heads, key_dim, value_dim, conv_kernel, group_size, eps, 0);
}

extern "C" int coli_cuda_gdn_slots_q4_f16_device(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    int batch, const int *slot, const int *position,
    const unsigned char *qkv_q, const float *qkv_s, int qkv_row_bytes,
    int qkv_groups, const unsigned char *z_q, const float *z_s,
    int z_row_bytes, int z_groups, const unsigned char *out_q,
    const float *out_s, int out_row_bytes, int out_groups,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *norm_weight, float *conv_state, float *recurrent_state,
    int slots, int hidden, int key_heads, int value_heads, int key_dim,
    int value_dim, int conv_kernel, int group_size, float eps) {
    return gdn_slots_q4_f16_impl(
        ctx, out, x, a, b, batch, slot, position, qkv_q, qkv_s,
        qkv_row_bytes, qkv_groups, z_q, z_s, z_row_bytes, z_groups, out_q,
        out_s, out_row_bytes, out_groups, conv_weight, a_log, dt_bias,
        norm_weight, conv_state, recurrent_state, slots, hidden, key_heads,
        value_heads, key_dim, value_dim, conv_kernel, group_size, eps, 1);
}

extern "C" int coli_cuda_gqa_decode_q4_f16(
    ColiCuda *ctx, float *out, const float *x,
    const unsigned char *q_q, const float *q_s, int q_rb, int q_ng,
    const unsigned char *k_q, const float *k_s, int k_rb, int k_ng,
    const unsigned char *v_q, const float *v_s, int v_rb, int v_ng,
    const void *o_q, const float *o_s, int o_fmt, int o_rb, int o_ng,
    const float *q_norm, const float *k_norm, float *k_cache,
    float *v_cache, int position, int hidden, int query_heads,
    int kv_heads, int head_dim, int rotary_dim, int group_size,
    float theta, float eps) {
    if(!ctx||!out||!x||!q_q||!q_s||!k_q||!k_s||!v_q||!v_s||!o_q||!o_s||!q_norm||!k_norm||!k_cache||!v_cache||position<0||hidden<=0||query_heads<=0||kv_heads<=0||head_dim<=0||head_dim>256||query_heads%kv_heads||(o_fmt!=1&&o_fmt!=4))return fail_arg("coli_cuda_gqa_decode_q4_f16: invalid argument");
    int qrows=query_heads*head_dim*2,kvrows=kv_heads*head_dim,ctxn=query_heads*head_dim;
#define ATTN_RESERVE(member,cap,bytes,label) ensure_cuda_buffer((void**)&ctx->member,&ctx->cap,(bytes),(label))
    if(ATTN_RESERVE(attn_x,attn_x_cap,(size_t)hidden*4,"attention input")||ATTN_RESERVE(attn_x16,attn_x16_cap,(size_t)hidden*2,"attention fp16 input")||ATTN_RESERVE(attn_qp,attn_qp_cap,(size_t)qrows*4,"attention qp")||ATTN_RESERVE(attn_q,attn_q_cap,(size_t)ctxn*4,"attention q")||ATTN_RESERVE(attn_gate,attn_gate_cap,(size_t)ctxn*4,"attention gate")||ATTN_RESERVE(attn_k,attn_k_cap,(size_t)kvrows*4,"attention k")||ATTN_RESERVE(attn_v,attn_v_cap,(size_t)kvrows*4,"attention v")||ATTN_RESERVE(attn_ctx,attn_ctx_cap,(size_t)ctxn*4,"attention ctx")||ATTN_RESERVE(attn_ctx16,attn_ctx16_cap,(size_t)ctxn*2,"attention fp16 ctx"))return -1;
#undef ATTN_RESERVE
    cudaError_t error=cudaMemcpyAsync(ctx->attn_x,x,(size_t)hidden*4,cudaMemcpyHostToDevice,ctx->stream);if(error!=cudaSuccess)return fail_cuda("attention input upload",error);
    f32_to_f16_kernel<<<(hidden+255)/256,256,0,ctx->stream>>>(ctx->attn_x16,ctx->attn_x,hidden);
    q4_gemv_f16_kernel<<<(qrows+7)/8,256,0,ctx->stream>>>(ctx->attn_qp,ctx->attn_x16,q_q,q_s,qrows,hidden,group_size,q_rb,q_ng);
    q4_gemv_f16_kernel<<<(kvrows+7)/8,256,0,ctx->stream>>>(ctx->attn_k,ctx->attn_x16,k_q,k_s,kvrows,hidden,group_size,k_rb,k_ng);
    q4_gemv_f16_kernel<<<(kvrows+7)/8,256,0,ctx->stream>>>(ctx->attn_v,ctx->attn_x16,v_q,v_s,kvrows,hidden,group_size,v_rb,v_ng);
    qnorm_rope_kernel<<<query_heads,head_dim,0,ctx->stream>>>(ctx->attn_q,ctx->attn_gate,ctx->attn_qp,q_norm,query_heads,head_dim,rotary_dim,position,theta,eps);
    knorm_rope_kernel<<<kv_heads,head_dim,0,ctx->stream>>>(ctx->attn_k,k_norm,kv_heads,head_dim,rotary_dim,position,theta,eps);
    error=cudaMemcpyAsync(k_cache+(size_t)position*kvrows,ctx->attn_k,(size_t)kvrows*4,cudaMemcpyDeviceToDevice,ctx->stream);if(error==cudaSuccess)error=cudaMemcpyAsync(v_cache+(size_t)position*kvrows,ctx->attn_v,(size_t)kvrows*4,cudaMemcpyDeviceToDevice,ctx->stream);if(error!=cudaSuccess)return fail_cuda("attention KV append",error);
    gqa_decode_kernel<<<query_heads,head_dim,(size_t)(position+1)*4,ctx->stream>>>(ctx->attn_ctx,ctx->attn_q,ctx->attn_gate,k_cache,v_cache,position,query_heads,kv_heads,head_dim);
    if(o_fmt==4){f32_to_f16_kernel<<<(ctxn+255)/256,256,0,ctx->stream>>>(ctx->attn_ctx16,ctx->attn_ctx,ctxn);q4_gemv_f16_kernel<<<(hidden+7)/8,256,0,ctx->stream>>>(ctx->attn_x,ctx->attn_ctx16,(const unsigned char*)o_q,o_s,hidden,ctxn,group_size,o_rb,o_ng);}else q8_gemv_kernel<<<(hidden+7)/8,256,0,ctx->stream>>>(ctx->attn_x,ctx->attn_ctx,(const signed char*)o_q,o_s,hidden,ctxn,o_rb);
    if(launch_status("GQA decode kernels"))return -1;error=cudaMemcpyAsync(out,ctx->attn_x,(size_t)hidden*4,cudaMemcpyDeviceToHost,ctx->stream);if(error!=cudaSuccess)return fail_cuda("attention output download",error);error=cudaStreamSynchronize(ctx->stream);return error==cudaSuccess?0:fail_cuda("attention synchronize",error);
}

static int gqa_slots_q4_f16_impl(
    ColiCuda *ctx, float *out, const float *x, int batch, const int *slot,
    const int *position, const unsigned char *q_q, const float *q_s,
    int q_rb, int q_ng, const unsigned char *k_q, const float *k_s,
    int k_rb, int k_ng, const unsigned char *v_q, const float *v_s,
    int v_rb, int v_ng, const void *o_q, const float *o_s, int o_fmt,
    int o_rb, int o_ng, const float *q_norm, const float *k_norm,
    float *k_cache, float *v_cache, int slots, int max_seq, int hidden,
    int query_heads, int kv_heads, int head_dim, int rotary_dim,
    int group_size, float theta, float eps, int device_io) {
    if (!ctx || !out || !x || batch < 1 || batch > 16 || !slot ||
        !position || !q_q || !q_s || !k_q || !k_s || !v_q || !v_s ||
        !o_q || !o_s || !q_norm || !k_norm || !k_cache || !v_cache ||
        slots < 1 || max_seq < 1 || hidden <= 0 || query_heads <= 0 ||
        kv_heads <= 0 || head_dim <= 0 || head_dim > 256 ||
        query_heads % kv_heads || (o_fmt != 1 && o_fmt != 4))
        return fail_arg("coli_cuda_gqa_slots_q4_f16: invalid argument");
    for (int row = 0; row < batch; row++)
        if (slot[row] < 0 || slot[row] >= slots || position[row] < 0 ||
            position[row] >= max_seq)
            return fail_arg("coli_cuda_gqa_slots_q4_f16: invalid slot");
    int qrows = query_heads * head_dim * 2;
    int kvrows = kv_heads * head_dim;
    int ctxn = query_heads * head_dim;
#define ATTN_SLOTS_RESERVE(member, cap, bytes, label) \
    ensure_cuda_buffer((void **)&ctx->member, &ctx->cap, (bytes), (label))
    if (ATTN_SLOTS_RESERVE(attn_x, attn_x_cap,
                           (size_t)batch * hidden * sizeof(float),
                           "attention slots input") ||
        ATTN_SLOTS_RESERVE(attn_x16, attn_x16_cap,
                           (size_t)batch * hidden * sizeof(__half),
                           "attention slots fp16 input") ||
        ATTN_SLOTS_RESERVE(attn_qp, attn_qp_cap,
                           (size_t)batch * qrows * sizeof(float),
                           "attention slots qp") ||
        ATTN_SLOTS_RESERVE(attn_q, attn_q_cap,
                           (size_t)batch * ctxn * sizeof(float),
                           "attention slots q") ||
        ATTN_SLOTS_RESERVE(attn_gate, attn_gate_cap,
                           (size_t)batch * ctxn * sizeof(float),
                           "attention slots gate") ||
        ATTN_SLOTS_RESERVE(attn_k, attn_k_cap,
                           (size_t)batch * kvrows * sizeof(float),
                           "attention slots k") ||
        ATTN_SLOTS_RESERVE(attn_v, attn_v_cap,
                           (size_t)batch * kvrows * sizeof(float),
                           "attention slots v") ||
        ATTN_SLOTS_RESERVE(attn_ctx, attn_ctx_cap,
                           (size_t)batch * ctxn * sizeof(float),
                           "attention slots context") ||
        ATTN_SLOTS_RESERVE(attn_ctx16, attn_ctx16_cap,
                           (size_t)batch * ctxn * sizeof(__half),
                           "attention slots fp16 context"))
        return -1;
#undef ATTN_SLOTS_RESERVE
    cudaError_t error = cudaMemcpyAsync(
        ctx->attn_x, x, (size_t)batch * hidden * sizeof(float),
        device_io ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
        ctx->stream);
    if (error != cudaSuccess)
        return fail_cuda("attention slots input upload", error);
    int input_count = batch * hidden;
    f32_to_f16_kernel<<<(input_count + 255) / 256, 256, 0, ctx->stream>>>(
        ctx->attn_x16, ctx->attn_x, input_count);
    dim3 q_grid((qrows + 7) / 8, batch);
    dim3 kv_grid((kvrows + 7) / 8, batch);
    q4_gemm_f16_kernel<<<q_grid, 256, 0, ctx->stream>>>(
        ctx->attn_qp, ctx->attn_x16, q_q, q_s, batch, qrows, hidden,
        group_size, q_rb, q_ng);
    q4_gemm_f16_kernel<<<kv_grid, 256, 0, ctx->stream>>>(
        ctx->attn_k, ctx->attn_x16, k_q, k_s, batch, kvrows, hidden,
        group_size, k_rb, k_ng);
    q4_gemm_f16_kernel<<<kv_grid, 256, 0, ctx->stream>>>(
        ctx->attn_v, ctx->attn_x16, v_q, v_s, batch, kvrows, hidden,
        group_size, v_rb, v_ng);
    size_t slot_stride = (size_t)max_seq * kvrows;
    for (int row = 0; row < batch; row++) {
        float *qr = ctx->attn_q + (size_t)row * ctxn;
        float *gr = ctx->attn_gate + (size_t)row * ctxn;
        float *kr = ctx->attn_k + (size_t)row * kvrows;
        float *vr = ctx->attn_v + (size_t)row * kvrows;
        float *cr = ctx->attn_ctx + (size_t)row * ctxn;
        float *slot_k = k_cache + (size_t)slot[row] * slot_stride;
        float *slot_v = v_cache + (size_t)slot[row] * slot_stride;
        int pos = position[row];
        qnorm_rope_kernel<<<query_heads, head_dim, 0, ctx->stream>>>(
            qr, gr, ctx->attn_qp + (size_t)row * qrows, q_norm, query_heads,
            head_dim, rotary_dim, pos, theta, eps);
        knorm_rope_kernel<<<kv_heads, head_dim, 0, ctx->stream>>>(
            kr, k_norm, kv_heads, head_dim, rotary_dim, pos, theta, eps);
        error = cudaMemcpyAsync(slot_k + (size_t)pos * kvrows, kr,
                                (size_t)kvrows * sizeof(float),
                                cudaMemcpyDeviceToDevice, ctx->stream);
        if (error == cudaSuccess)
            error = cudaMemcpyAsync(slot_v + (size_t)pos * kvrows, vr,
                                    (size_t)kvrows * sizeof(float),
                                    cudaMemcpyDeviceToDevice, ctx->stream);
        if (error != cudaSuccess)
            return fail_cuda("attention slots KV append", error);
        gqa_decode_kernel<<<query_heads, head_dim,
                            (size_t)(pos + 1) * sizeof(float), ctx->stream>>>(
            cr, qr, gr, slot_k, slot_v, pos, query_heads, kv_heads, head_dim);
    }
    dim3 out_grid((hidden + 7) / 8, batch);
    if (o_fmt == 4) {
        int context_count = batch * ctxn;
        f32_to_f16_kernel<<<(context_count + 255) / 256, 256, 0,
                             ctx->stream>>>(
            ctx->attn_ctx16, ctx->attn_ctx, context_count);
        q4_gemm_f16_kernel<<<out_grid, 256, 0, ctx->stream>>>(
            ctx->attn_x, ctx->attn_ctx16, (const unsigned char *)o_q, o_s,
            batch, hidden, ctxn, group_size, o_rb, o_ng);
    } else {
        q8_gemm_kernel<<<out_grid, 256, 0, ctx->stream>>>(
            ctx->attn_x, ctx->attn_ctx, (const signed char *)o_q, o_s, batch,
            hidden, ctxn, o_rb);
    }
    if (launch_status("GQA slots kernels")) return -1;
    error = cudaMemcpyAsync(
        out, ctx->attn_x, (size_t)batch * hidden * sizeof(float),
        device_io ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost,
        ctx->stream);
    if (error != cudaSuccess)
        return fail_cuda("attention slots output download", error);
    if (device_io) return 0;
    error = cudaStreamSynchronize(ctx->stream);
    return error == cudaSuccess ? 0
                                : fail_cuda("attention slots synchronize", error);
}

extern "C" int coli_cuda_gqa_slots_q4_f16(
    ColiCuda *ctx, float *out, const float *x, int batch, const int *slot,
    const int *position, const unsigned char *q_q, const float *q_s,
    int q_rb, int q_ng, const unsigned char *k_q, const float *k_s,
    int k_rb, int k_ng, const unsigned char *v_q, const float *v_s,
    int v_rb, int v_ng, const void *o_q, const float *o_s, int o_fmt,
    int o_rb, int o_ng, const float *q_norm, const float *k_norm,
    float *k_cache, float *v_cache, int slots, int max_seq, int hidden,
    int query_heads, int kv_heads, int head_dim, int rotary_dim,
    int group_size, float theta, float eps) {
    return gqa_slots_q4_f16_impl(
        ctx, out, x, batch, slot, position, q_q, q_s, q_rb, q_ng, k_q, k_s,
        k_rb, k_ng, v_q, v_s, v_rb, v_ng, o_q, o_s, o_fmt, o_rb, o_ng,
        q_norm, k_norm, k_cache, v_cache, slots, max_seq, hidden,
        query_heads, kv_heads, head_dim, rotary_dim, group_size, theta, eps,
        0);
}

extern "C" int coli_cuda_gqa_slots_q4_f16_device(
    ColiCuda *ctx, float *out, const float *x, int batch, const int *slot,
    const int *position, const unsigned char *q_q, const float *q_s,
    int q_rb, int q_ng, const unsigned char *k_q, const float *k_s,
    int k_rb, int k_ng, const unsigned char *v_q, const float *v_s,
    int v_rb, int v_ng, const void *o_q, const float *o_s, int o_fmt,
    int o_rb, int o_ng, const float *q_norm, const float *k_norm,
    float *k_cache, float *v_cache, int slots, int max_seq, int hidden,
    int query_heads, int kv_heads, int head_dim, int rotary_dim,
    int group_size, float theta, float eps) {
    return gqa_slots_q4_f16_impl(
        ctx, out, x, batch, slot, position, q_q, q_s, q_rb, q_ng, k_q, k_s,
        k_rb, k_ng, v_q, v_s, v_rb, v_ng, o_q, o_s, o_fmt, o_rb, o_ng,
        q_norm, k_norm, k_cache, v_cache, slots, max_seq, hidden,
        query_heads, kv_heads, head_dim, rotary_dim, group_size, theta, eps,
        1);
}
