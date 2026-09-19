#ifndef COLIB_DEEPSEEK_V4_H
#define COLIB_DEEPSEEK_V4_H

/* Scalar, dependency-free DeepSeek-V4 reference math.  These routines favor
 * auditability over speed and are the correctness oracle for the SM120
 * kernels.  Tensor layouts match the pinned 0731 checkpoint. */

#include "deepseek_v4_limits.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define DSV4_HC_MULT 4
#define DSV4_HC_MIX ((2 + DSV4_HC_MULT) * DSV4_HC_MULT)
#define DSV4_FP8_BLOCK 128
#define DSV4_FP4_BLOCK 32
#define DSV4_PI 3.14159265358979323846f

static inline float dsv4_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float z = expf(x);
    return z / (1.0f + z);
}

static inline float dsv4_softplus(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

static inline float dsv4_fp4_e2m1(uint8_t code) {
    static const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                       2.0f, 3.0f, 4.0f, 6.0f};
    float v = magnitude[code & 7u];
    return (code & 8u) ? -v : v;
}

/* Exact IEEE bit construction avoids scalar libm calls in hot conversion
 * loops. E4M3 values and UE8M0 scales are unchanged, including signed zero. */
static inline float dsv4_fp8_e4m3fn(uint8_t code) {
    uint32_t sign=(uint32_t)(code&128u)<<24;
    unsigned exponent=(code>>3)&15u,mantissa=code&7u;
    if (exponent==15 && mantissa==7) return NAN;
    if (!exponent) {
        float value=(float)mantissa*0.001953125f;
        return sign?-value:value;
    }
    uint32_t bits=sign|((exponent+120u)<<23)|(mantissa<<20);
    float value;memcpy(&value,&bits,sizeof(value));return value;
}

static inline float dsv4_ue8m0(uint8_t code) {
    if (code==255u) return NAN;
    uint32_t bits=code?(uint32_t)code<<23:UINT32_C(0x00400000);
    float value;memcpy(&value,&bits,sizeof(value));return value;
}

/* Reference encoder used only by fixture quantization.  Ties select the even
 * raw mantissa, matching round-to-nearest-even casts used by the GPU path. */
static inline uint8_t dsv4_fp4_encode(float value) {
    if (isnan(value)) return 0;
    int best=0;float error=INFINITY,target=fminf(fabsf(value),6.0f);
    for (int code=0;code<8;code++) {
        float e=fabsf(target-dsv4_fp4_e2m1((uint8_t)code));
        if (e<error || (e==error && !(code&1) && (best&1))) {error=e;best=code;}
    }
    return (uint8_t)(best | (signbit(value)?8:0));
}

static inline uint8_t dsv4_fp8_encode(float x) {
    uint32_t bits;memcpy(&bits,&x,sizeof(bits));
    uint32_t magnitude=bits&UINT32_C(0x7fffffff);
    unsigned sign=(bits>>24)&128u;
    if (magnitude>UINT32_C(0x7f800000)) return 0x7f;
    if (magnitude>UINT32_C(0x43e00000)) magnitude=UINT32_C(0x43e00000);
    unsigned raw;
    if (magnitude<UINT32_C(0x3c800000)) {
        float target;memcpy(&target,&magnitude,sizeof(target));
        float scaled=target*512.0f;
        raw=(unsigned)scaled;float fraction=scaled-raw;
        if (fraction>0.5f || (fraction==0.5f && (raw&1))) raw++;
    } else {
        raw=(magnitude>>20)-960u;
        unsigned remainder=magnitude&UINT32_C(0x000fffff);
        if (remainder>UINT32_C(0x00080000) ||
            (remainder==UINT32_C(0x00080000) && (raw&1))) raw++;
    }
    if (raw>0x7e) raw=0x7e;
    return (uint8_t)(raw|sign);
}

/* Official MXFP activation quantization: groups of 128, amax floor 1e-4,
 * and a power-of-two ceil(amax/448) scale stored as UE8M0. */
static inline int dsv4_act_quant_mxfp(const float *x, int n,
                                     uint8_t *q, uint8_t *scales) {
    if (!x || !q || !scales || n <= 0 || n % DSV4_FP8_BLOCK) return 0;
    for (int block = 0; block < n / DSV4_FP8_BLOCK; block++) {
        float amax = 1e-4f;
        for (int j = 0; j < DSV4_FP8_BLOCK; j++)
            amax = fmaxf(amax, fabsf(x[block * DSV4_FP8_BLOCK + j]));
        int exponent = (int)ceilf(log2f(amax / 448.0f));
        if (exponent < -127) exponent = -127;
        if (exponent > 127) exponent = 127;
        float scale = ldexpf(1.0f, exponent);
        scales[block] = (uint8_t)(exponent + 127);
        for (int j = 0; j < DSV4_FP8_BLOCK; j++) {
            float v = x[block * DSV4_FP8_BLOCK + j] / scale;
            q[block * DSV4_FP8_BLOCK + j] = dsv4_fp8_encode(v);
        }
    }
    return 1;
}

static inline int dsv4_act_quant_mxfp_batch(const float *input,int batch,int columns,
    uint8_t *activation,uint8_t *scales) {
    if (!input || !activation || !scales || batch<1 || columns<1 || columns%DSV4_FP8_BLOCK) return 0;
    int ok=1;
#ifdef _OPENMP
#pragma omp parallel for if(batch>=16) num_threads(6) reduction(&:ok)
#endif
    for (int b=0;b<batch;b++)
        ok &= dsv4_act_quant_mxfp(input+(size_t)b*columns,columns,activation+(size_t)b*columns,
                                 scales+(size_t)b*(columns/DSV4_FP8_BLOCK));
    return ok;
}

/* FP8 A[M,K] x FP8 B[N,K]^T with one activation scale per 128 K and
 * one weight scale per 128x128 output/K tile. */
static inline int dsv4_fp8_simulate(float *values, int count,
                                      int block) {
    if (!values || count <= 0 || block <= 0 || count % block) return 0;
    for (int base = 0; base < count; base += block) {
        float maximum = 1e-4f;
        for (int index = 0; index < block; index++)
            maximum = fmaxf(maximum, fabsf(values[base + index]));
        int exponent = (int)ceilf(log2f(maximum / 448.0f));
        if (exponent < -127) exponent = -127;
        if (exponent > 127) exponent = 127;
        float scale = ldexpf(1.0f, exponent);
        for (int index = 0; index < block; index++) {
            uint8_t quantized = dsv4_fp8_encode(values[base + index] / scale);
            values[base + index] = dsv4_fp8_e4m3fn(quantized) * scale;
        }
    }
    return 1;
}

static inline int dsv4_hadamard(float *values, int count) {
    if (!values || count <= 0 || (count & (count - 1))) return 0;
    for (int stride = 1; stride < count; stride <<= 1)
        for (int base = 0; base < count; base += 2 * stride)
            for (int index = 0; index < stride; index++) {
                float left = values[base + index];
                float right = values[base + stride + index];
                values[base + index] = left + right;
                values[base + stride + index] = left - right;
            }
    float scale = 1.0f / sqrtf((float)count);
    for (int index = 0; index < count; index++) values[index] *= scale;
    return 1;
}

static inline int dsv4_fp4_simulate(float *values, int count, int block) {
    if (!values || count <= 0 || block <= 0 || count % block) return 0;
    for (int base = 0; base < count; base += block) {
        float maximum = 6.0f*0x1p-126f;
        for (int index = 0; index < block; index++)
            maximum = fmaxf(maximum, fabsf(values[base + index]));
        int exponent = (int)ceilf(log2f(maximum / 6.0f));
        if (exponent < -127) exponent = -127;
        if (exponent > 127) exponent = 127;
        float scale = ldexpf(1.0f, exponent);
        for (int index = 0; index < block; index++) {
            uint8_t quantized = dsv4_fp4_encode(values[base + index] / scale);
            values[base + index] = dsv4_fp4_e2m1(quantized) * scale;
        }
    }
    return 1;
}

static inline int dsv4_fp8_gemm(float *out, const uint8_t *a,
                               const uint8_t *a_scale,
                               const uint8_t *weight,
                               const uint8_t *weight_scale,
                               int batch, int rows, int cols) {
    if (!out || !a || !a_scale || !weight || !weight_scale || batch <= 0 ||
        rows <= 0 || cols <= 0 || cols % DSV4_FP8_BLOCK) return 0;
    int kblocks = cols / DSV4_FP8_BLOCK;
    for (int b = 0; b < batch; b++) for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int kb = 0; kb < kblocks; kb++) {
            float inner = 0.0f;
            for (int j = 0; j < DSV4_FP8_BLOCK; j++) {
                int k = kb * DSV4_FP8_BLOCK + j;
                inner += dsv4_fp8_e4m3fn(a[(size_t)b * cols + k]) *
                         dsv4_fp8_e4m3fn(weight[(size_t)r * cols + k]);
            }
            float as = dsv4_ue8m0(a_scale[(size_t)b * kblocks + kb]);
            float ws = dsv4_ue8m0(weight_scale[(size_t)(r / 128) * kblocks + kb]);
            sum += inner * as * ws;
        }
        out[(size_t)b * rows + r] = sum;
    }
    return 1;
}

/* FP8 A[M,K] x packed-E2M1 B[N,K]^T.  Checkpoint weights use low nibble
 * first along K and one UE8M0 weight scale per row/32 K. */
static inline int dsv4_fp4_gemm(float *out, const uint8_t *a,
                               const uint8_t *a_scale,
                               const uint8_t *weight,
                               const uint8_t *weight_scale,
                               int batch, int rows, int cols) {
    if (!out || !a || !a_scale || !weight || !weight_scale || batch <= 0 ||
        rows <= 0 || cols <= 0 || cols % DSV4_FP8_BLOCK) return 0;
    int ablocks = cols / DSV4_FP8_BLOCK, wblocks = cols / DSV4_FP4_BLOCK;
    for (int b = 0; b < batch; b++) for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int wb = 0; wb < wblocks; wb++) {
            float inner = 0.0f;
            for (int j = 0; j < DSV4_FP4_BLOCK; j++) {
                int k = wb * DSV4_FP4_BLOCK + j;
                uint8_t packed = weight[(size_t)r * (cols / 2) + k / 2];
                uint8_t nibble = (k & 1) ? packed >> 4 : packed & 15u;
                inner += dsv4_fp8_e4m3fn(a[(size_t)b * cols + k]) *
                         dsv4_fp4_e2m1(nibble);
            }
            float as = dsv4_ue8m0(a_scale[(size_t)b * ablocks + wb / 4]);
            float ws = dsv4_ue8m0(weight_scale[(size_t)r * wblocks + wb]);
            sum += inner * as * ws;
        }
        out[(size_t)b * rows + r] = sum;
    }
    return 1;
}

static inline float dsv4_bf16(uint16_t raw) {
    uint32_t bits = (uint32_t)raw << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline uint16_t dsv4_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
        if (bits & UINT32_C(0x007fffff)) bits |= UINT32_C(0x00010000);
        return (uint16_t)(bits >> 16);
    }
    bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static inline float dsv4_round_bf16(float value) {
    return dsv4_bf16(dsv4_float_to_bf16(value));
}

static inline void dsv4_round_bf16_array(float *values, size_t count) {
    for (size_t index = 0; index < count; index++)
        values[index] = dsv4_round_bf16(values[index]);
}

static inline void dsv4_rmsnorm(float *out, const float *x,
                                const uint16_t *weight, int n, float eps) {
    /* Accumulate the exactly representable BF16 squares without the long
     * serial-FP32 reduction drift, then retain the upstream FP32 mean/RMS. */
    double squares = 0.0;
    for (int i = 0; i < n; i++) squares += (double)x[i] * x[i];
    float inverse = 1.0f / sqrtf((float)(squares / n) + eps);
    for (int i = 0; i < n; i++)
        out[i] = dsv4_round_bf16(
            x[i] * inverse * dsv4_bf16(weight[i]));
}

static inline void dsv4_query_norm_bf16(float *query,int dim,float epsilon) {
    float square=0;
    for (int i=0;i<dim;i++) square+=dsv4_round_bf16(query[i]*query[i]);
    float mean=dsv4_round_bf16(square/dim);
    float inverse=dsv4_round_bf16(1.0f/sqrtf(dsv4_round_bf16(mean+epsilon)));
    for (int i=0;i<dim;i++) query[i]=dsv4_round_bf16(query[i]*inverse);
}

static inline void dsv4_bf16_gemm(float *out, const float *x,
                                  const uint16_t *weight,
                                  int batch, int rows, int cols) {
    for (int b = 0; b < batch; b++) for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int k = 0; k < cols; k++)
            sum += x[(size_t)b * cols + k] * dsv4_bf16(weight[(size_t)r * cols + k]);
        out[(size_t)b * rows + r] = sum;
    }
}

static inline void dsv4_f32_gemm(float *out, const float *x,
                                 const float *weight,
                                 int batch, int rows, int cols) {
    for (int b = 0; b < batch; b++) for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int k = 0; k < cols; k++)
            sum += x[(size_t)b * cols + k] * weight[(size_t)r * cols + k];
        out[(size_t)b * rows + r] = sum;
    }
}

static inline int dsv4_linear_fp8(float *out, const float *x,
                                  const uint8_t *weight,
                                  const uint8_t *weight_scale,
                                  int batch, int rows, int cols,
                                  uint8_t *act, uint8_t *act_scale) {
    for (int b = 0; b < batch; b++)
        if (!dsv4_act_quant_mxfp(x + (size_t)b * cols, cols,
                                 act + (size_t)b * cols,
                                 act_scale + (size_t)b * (cols / 128))) return 0;
    if (!dsv4_fp8_gemm(out, act, act_scale, weight, weight_scale,
                         batch, rows, cols))
        return 0;
    dsv4_round_bf16_array(out, (size_t)batch * rows);
    return 1;
}

static inline int dsv4_linear_fp4(float *out, const float *x,
                                  const uint8_t *weight,
                                  const uint8_t *weight_scale,
                                  int batch, int rows, int cols,
                                  uint8_t *act, uint8_t *act_scale) {
    for (int b = 0; b < batch; b++)
        if (!dsv4_act_quant_mxfp(x + (size_t)b * cols, cols,
                                 act + (size_t)b * cols,
                                 act_scale + (size_t)b * (cols / 128))) return 0;
    if (!dsv4_fp4_gemm(out, act, act_scale, weight, weight_scale,
                         batch, rows, cols))
        return 0;
    dsv4_round_bf16_array(out, (size_t)batch * rows);
    return 1;
}

static inline float dsv4_clamped_swiglu(float gate, float up);

/* One native routed/shared expert. Scratch sizes are intermediate*2 floats,
 * max(hidden,intermediate) activation bytes, and that length/128 scales. */
static inline int dsv4_expert_fp4(float *out, const float *x,
                                  const uint8_t *w1, const uint8_t *s1,
                                  const uint8_t *w2, const uint8_t *s2,
                                  const uint8_t *w3, const uint8_t *s3,
                                  int hidden, int intermediate, float route,
                                  float *gate, float *up,
                                  uint8_t *act, uint8_t *act_scale) {
    if (!dsv4_linear_fp4(gate, x, w1, s1, 1, intermediate, hidden,
                         act, act_scale) ||
        !dsv4_linear_fp4(up, x, w3, s3, 1, intermediate, hidden,
                         act, act_scale)) return 0;
    for (int i = 0; i < intermediate; i++)
        gate[i] = dsv4_round_bf16(
            route * dsv4_clamped_swiglu(gate[i], up[i]));
    return dsv4_linear_fp4(out, gate, w2, s2, 1, hidden, intermediate,
                           act, act_scale);
}

static inline int dsv4_expert_fp8(float *out, const float *x,
                                  const uint8_t *w1, const uint8_t *s1,
                                  const uint8_t *w2, const uint8_t *s2,
                                  const uint8_t *w3, const uint8_t *s3,
                                  int hidden, int intermediate,
                                  float *gate, float *up,
                                  uint8_t *act, uint8_t *act_scale) {
    if (!dsv4_linear_fp8(gate, x, w1, s1, 1, intermediate, hidden,
                         act, act_scale) ||
        !dsv4_linear_fp8(up, x, w3, s3, 1, intermediate, hidden,
                         act, act_scale)) return 0;
    for (int index = 0; index < intermediate; index++)
        gate[index] = dsv4_round_bf16(
            dsv4_clamped_swiglu(gate[index], up[index]));
    return dsv4_linear_fp8(out, gate, w2, s2, 1, hidden, intermediate,
                           act, act_scale);
}

static inline int dsv4_route_topk(const float *logits, const float *bias,
                                  int experts, const int *hash_indices,
                                  int topk, int *indices, float *weights) {
    if (!logits || !indices || !weights || experts < topk || topk < 1 ||
        topk > 256 || experts > 256)
        return 0;
    float original[256], selected[256];
    for (int index = 0; index < experts; index++) {
        original[index] = sqrtf(dsv4_softplus(logits[index]));
        selected[index] = original[index] + (bias ? bias[index] : 0.0f);
    }
    if (hash_indices) {
        for (int route = 0; route < topk; route++) {
            if (hash_indices[route] < 0 || hash_indices[route] >= experts)
                return 0;
            indices[route] = hash_indices[route];
        }
    } else {
        uint8_t used[256] = {0};
        for (int route = 0; route < topk; route++) {
            int best = -1;
            for (int index = 0; index < experts; index++)
                if (!used[index] &&
                    (best < 0 || selected[index] > selected[best]))
                    best = index;
            indices[route] = best;
            used[best] = 1;
        }
    }
    float total = 0.0f;
    for (int route = 0; route < topk; route++)
        total += original[indices[route]];
    if (!(total > 0.0f)) return 0;
    for (int route = 0; route < topk; route++)
        weights[route] = (original[indices[route]] / total) * 1.5f;
    return 1;
}

static inline int dsv4_route_top6(const float *logits, const float *bias,
                                  int experts, const int *hash_indices,
                                  int indices[6], float weights[6]) {
    return dsv4_route_topk(logits, bias, experts, hash_indices, 6,
                           indices, weights);
}

static inline float dsv4_clamped_swiglu(float gate, float up) {
    up = fminf(10.0f, fmaxf(-10.0f, up));
    gate = fminf(10.0f, gate);
    return (gate / (1.0f + expf(-gate))) * up;
}

static inline void dsv4_hc_split_sinkhorn(const float mixes[DSV4_HC_MIX],
                                           const float scale[3],
                                           const float base[DSV4_HC_MIX],
                                           float eps, int iterations,
                                           float pre[4], float post[4],
                                           float comb[16]) {
    for (int j = 0; j < 4; j++) {
        pre[j] = dsv4_sigmoid(mixes[j] * scale[0] + base[j]) + eps;
        post[j] = 2.0f * dsv4_sigmoid(mixes[4 + j] * scale[1] + base[4 + j]);
    }
    for (int j = 0; j < 4; j++) {
        float row_max = -FLT_MAX, row_sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            comb[j * 4 + k] = mixes[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k];
            row_max = fmaxf(row_max, comb[j * 4 + k]);
        }
        for (int k = 0; k < 4; k++) row_sum += expf(comb[j * 4 + k] - row_max);
        for (int k = 0; k < 4; k++) comb[j * 4 + k] = expf(comb[j * 4 + k] - row_max) / row_sum + eps;
    }
    for (int it = 0; it < iterations; it++) {
        for (int k = 0; k < 4; k++) {
            float sum = 0.0f;
            for (int j = 0; j < 4; j++) sum += comb[j * 4 + k];
            for (int j = 0; j < 4; j++) comb[j * 4 + k] /= sum + eps;
        }
        if (it + 1 == iterations) break;
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) sum += comb[j * 4 + k];
            for (int k = 0; k < 4; k++) comb[j * 4 + k] /= sum + eps;
        }
    }
}

static inline void dsv4_hc_pre(const float *residual, int dim,
                               const float pre[4], float *out) {
    for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int h = 0; h < 4; h++) sum += pre[h] * residual[(size_t)h * dim + d];
        out[d] = sum;
    }
}

static inline void dsv4_hc_forward_pre(
    const float *residual, const float *function_weight, int dim,
    const float scale[3], const float base[DSV4_HC_MIX], float norm_eps,
    float *reduced, float post[4], float comb[16], float *mixes_out) {
    double square_sum = 0.0;float mixes[DSV4_HC_MIX], pre[4];
    int residual_dim = DSV4_HC_MULT * dim;
    for (int index = 0; index < residual_dim; index++)
        square_sum += (double)residual[index] * residual[index];
    float inverse_rms = 1.0f / sqrtf((float)(square_sum / residual_dim) + norm_eps);
    for (int mix = 0; mix < DSV4_HC_MIX; mix++) {
        /* HC function weights are FP32. Retain their full precision while
         * avoiding a 16384-term serial FP32 reduction error. */
        double value = 0.0;
        for (int index = 0; index < residual_dim; index++)
            value += (double)function_weight[(size_t)mix * residual_dim + index] *
                     residual[index];
        mixes[mix] = (float)value * inverse_rms;
    }
    dsv4_hc_split_sinkhorn(mixes, scale, base, 1e-6f, 20,
                            pre, post, comb);
    dsv4_hc_pre(residual, dim, pre, reduced);
    if (mixes_out) memcpy(mixes_out, mixes, sizeof(mixes));
}

static inline void dsv4_hc_post(const float *x, const float *residual, int dim,
                                const float post[4], const float comb[16],
                                float *out) {
    for (int to = 0; to < 4; to++) for (int d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (int from = 0; from < 4; from++)
            sum += comb[from * 4 + to] * residual[(size_t)from * dim + d];
        out[(size_t)to * dim + d] = post[to]*x[d]+sum;
    }
}

typedef int (*dsv4_hc_module_fn)(void *context, const float *input,
                                  float *output);

static inline int dsv4_hc_stage(
    float *hc_state, int dim, const float *function_weight,
    const float scale[3], const float base[DSV4_HC_MIX],
    const uint16_t *norm_weight, float epsilon, dsv4_hc_module_fn module,
    void *module_context, float *reduced, float *post, float *combination,
    float *mixes, float *module_output, float *expanded) {
    if (!hc_state || dim <= 0 || !function_weight || !scale || !base ||
        !norm_weight || !module || !reduced || !post || !combination ||
        !mixes || !module_output || !expanded)
        return 0;
    dsv4_hc_forward_pre(hc_state, function_weight, dim, scale, base, epsilon,
                        reduced, post, combination, mixes);
    dsv4_round_bf16_array(reduced, (size_t)dim);
    dsv4_rmsnorm(reduced, reduced, norm_weight, dim, epsilon);
    if (!module(module_context, reduced, module_output)) return 0;
    dsv4_hc_post(module_output, hc_state, dim, post, combination, expanded);
    dsv4_round_bf16_array(expanded, (size_t)DSV4_HC_MULT * dim);
    memcpy(hc_state, expanded,
           (size_t)DSV4_HC_MULT * dim * sizeof(*hc_state));
    return 1;
}

static inline float dsv4_yarn_frequency(int pair, int rotary_dim,
                                        int original_seq_len, float base,
                                        float factor, int beta_fast,
                                        int beta_slow) {
    /* Match the pinned FP32 reciprocal-after-power operation. A negative
     * exponent rounds frequencies differently, magnified at long positions. */
    float freq = 1.0f / powf(base, (float)(2 * pair) / rotary_dim);
    if (original_seq_len <= 0) return freq;
    float denom = 2.0f * logf(base);
    float low_f = rotary_dim * logf(original_seq_len / (beta_fast * 2.0f * DSV4_PI)) / denom;
    float high_f = rotary_dim * logf(original_seq_len / (beta_slow * 2.0f * DSV4_PI)) / denom;
    int low = (int)floorf(low_f), high = (int)ceilf(high_f);
    if (low < 0) low = 0;
    if (high > rotary_dim - 1) high = rotary_dim - 1;
    float ramp = high == low ? (pair - low) / 0.001f : (float)(pair - low) / (high - low);
    ramp = fminf(1.0f, fmaxf(0.0f, ramp));
    float smooth = 1.0f - ramp;
    return freq / factor * (1.0f - smooth) + freq * smooth;
}

static inline void dsv4_rope(float *x, int rotary_dim, int position,
                             int original_seq_len, float base, float factor,
                             int beta_fast, int beta_slow, int inverse) {
    for (int pair = 0; pair < rotary_dim / 2; pair++) {
        float angle = position * dsv4_yarn_frequency(pair, rotary_dim,
            original_seq_len, base, factor, beta_fast, beta_slow);
        if (inverse) angle = -angle;
        float c = cosf(angle), s = sinf(angle), a = x[2 * pair], b = x[2 * pair + 1];
        x[2 * pair] = a * c - b * s;
        x[2 * pair + 1] = a * s + b * c;
    }
}

static inline int dsv4_window_indices(int window, int seqlen, int start_pos,
                                      int row, int *out) {
    if (!out || window <= 0 || seqlen <= 0 || row < 0 || row >= seqlen) return 0;
    int count = start_pos == 0 ? (seqlen < window ? seqlen : window) : window;
    if (start_pos >= window - 1) {
        int p = start_pos % window;
        for (int i = 0; i < window; i++) out[i] = (p + 1 + i) % window;
    } else if (start_pos > 0) {
        for (int i = 0; i < window; i++) out[i] = i <= start_pos ? i : -1;
    } else {
        int begin = row - window + 1;
        if (begin < 0) begin = 0;
        for (int i = 0; i < count; i++) {
            int v = begin + i;
            out[i] = v <= row ? v : -1;
        }
    }
    return count;
}

static inline int dsv4_compress_indices(int ratio, int seqlen, int start_pos,
                                        int offset, int row, int *out) {
    if (!out || ratio <= 0 || seqlen <= 0 || row < 0 || row >= seqlen) return 0;
    if (start_pos > 0) {
        int count = (start_pos + 1) / ratio;
        for (int i = 0; i < count; i++) out[i] = i + offset;
        return count;
    }
    int count = seqlen / ratio;
    int visible = (row + 1) / ratio;
    for (int i = 0; i < count; i++) out[i] = i < visible ? i + offset : -1;
    return count;
}

/* Learned compressor pooling for one completed window. kv/score/ape are
 * [tokens,dim]; softmax is independently normalized for every feature. */
static inline void dsv4_compress_pool(float *out, const float *kv,
                                      const float *score, const float *ape,
                                      int tokens, int dim) {
    for (int d = 0; d < dim; d++) {
        float maximum = -FLT_MAX, denom = 0.0f, sum = 0.0f;
        for (int t = 0; t < tokens; t++)
            maximum = fmaxf(maximum, score[(size_t)t * dim + d] + ape[(size_t)t * dim + d]);
        for (int t = 0; t < tokens; t++) {
            float weight = expf(score[(size_t)t * dim + d] +
                                ape[(size_t)t * dim + d] - maximum);
            denom += weight;
            sum += kv[(size_t)t * dim + d] * weight;
        }
        out[d] = sum / denom;
    }
}

typedef struct {
    int ratio;
    int dim;
    int overlap;
    int next_position;
    float *kv_state;
    float *score_state;
} dsv4_compressor_state;

static inline size_t dsv4_compressor_state_floats(int ratio, int dim,
                                                   int overlap) {
    if (ratio <= 0 || dim <= 0) return 0;
    size_t coefficient = overlap ? 2u : 1u;
    if ((size_t)ratio > SIZE_MAX / coefficient ||
        (size_t)dim > SIZE_MAX / coefficient ||
        (size_t)ratio * coefficient >
            SIZE_MAX / ((size_t)dim * coefficient))
        return 0;
    return (size_t)ratio * coefficient * (size_t)dim * coefficient;
}

static inline int dsv4_compressor_state_init(dsv4_compressor_state *state,
                                              int ratio, int dim, int overlap,
                                              float *kv_state,
                                              float *score_state) {
    size_t count = dsv4_compressor_state_floats(ratio, dim, overlap);
    if (!state || !count || !kv_state || !score_state) return 0;
    state->ratio = ratio;
    state->dim = dim;
    state->overlap = overlap != 0;
    state->next_position = 0;
    state->kv_state = kv_state;
    state->score_state = score_state;
    for (size_t index = 0; index < count; index++) {
        kv_state[index] = 0.0f;
        score_state[index] = -INFINITY;
    }
    return 1;
}

/* Incremental decode form of Compressor.forward. kv/score are the learned
 * projections for one token, ape is [ratio, coefficient*dim]. Return 1 when a
 * complete compressed value was emitted, 0 while accumulating, and -1 for an
 * invalid/non-contiguous transaction. The raw pooled output is normalized,
 * RoPE-transformed, and quantized by the caller exactly as in the reference. */
static inline int dsv4_compressor_step(dsv4_compressor_state *state,
                                       int position, const float *kv,
                                       const float *score, const float *ape,
                                       float *output) {
    if (!state || !kv || !score || !ape || !output || position < 0 ||
        position != state->next_position || state->ratio <= 0 ||
        state->dim <= 0 || !state->kv_state || !state->score_state)
        return -1;
    int ratio = state->ratio, dim = state->dim;
    int coefficient = state->overlap ? 2 : 1;
    int width = coefficient * dim;
    int slot = position % ratio;
    int target = state->overlap ? ratio + slot : slot;
    for (int axis = 0; axis < width; axis++) {
        state->kv_state[(size_t)target * width + axis] = kv[axis];
        state->score_state[(size_t)target * width + axis] =
            score[axis] + ape[(size_t)slot * width + axis];
    }
    state->next_position++;
    if (slot + 1 != ratio) return 0;
    for (int axis = 0; axis < dim; axis++) {
        float maximum = -INFINITY, denominator = 0.0f, sum = 0.0f;
        int samples = state->overlap ? 2 * ratio : ratio;
        for (int sample = 0; sample < samples; sample++) {
            int row = sample;
            int column = axis;
            if (state->overlap && sample >= ratio) {
                row = sample;
                column = dim + axis;
            }
            maximum = fmaxf(maximum,
                state->score_state[(size_t)row * width + column]);
        }
        for (int sample = 0; sample < samples; sample++) {
            int row = sample;
            int column = axis;
            if (state->overlap && sample >= ratio) {
                row = sample;
                column = dim + axis;
            }
            float weight = expf(
                state->score_state[(size_t)row * width + column] - maximum);
            denominator += weight;
            sum += state->kv_state[(size_t)row * width + column] * weight;
        }
        output[axis] = sum / denominator;
    }
    if (state->overlap) {
        memmove(state->kv_state,
                state->kv_state + (size_t)ratio * width,
                (size_t)ratio * width * sizeof(float));
        memmove(state->score_state,
                state->score_state + (size_t)ratio * width,
                (size_t)ratio * width * sizeof(float));
    }
    return 1;
}

static inline int dsv4_dspark_indices(int window, int block, int start_pos,
                                      int *out) {
    if (!out || window <= 0 || block <= 0 || start_pos <= 0) return 0;
    int visible = start_pos + 1 < window ? start_pos + 1 : window;
    for (int i = 0; i < visible; i++) out[i] = i;
    for (int i = 0; i < block; i++) out[visible + i] = window + i;
    return visible + block;
}

/* Learned indexer score from Indexer.forward: ReLU(q_h dot kv_t),
 * weighted across heads, then top-k over visible compressed positions. The
 * caller provides `workspace[tokens]`; selected indices include cache offset. */
static inline int dsv4_indexer_select(const float *workspace,int tokens,int topk,int offset,int *indices) {
    if (!workspace || !indices || tokens<1 || topk<1 || topk>tokens || offset<0) return 0;
    /* Keep the best k in a heap whose root is the worst retained item.
     * Equal scores prefer the lower source index, exactly as the scalar scan.
     * This avoids a quadratic scan of already selected indices at every token. */
    int retained=0;
    for (int token=0;token<tokens;token++) {
        if (isnan(workspace[token]) || workspace[token]==-INFINITY) continue;
        int at;
        if (retained<topk) { at=retained++;indices[at]=token+offset; }
        else {
            int worst=indices[0]-offset;
            if (workspace[token]<workspace[worst] || (workspace[token]==workspace[worst] && token>worst)) continue;
            indices[0]=token+offset;at=0;
        }
        if (at) {
            while (at>0) {
                int parent=(at-1)/2,a=indices[at]-offset,b=indices[parent]-offset;
                if (!(workspace[a]<workspace[b] || (workspace[a]==workspace[b] && a>b))) break;
                int temp=indices[at];indices[at]=indices[parent];indices[parent]=temp;at=parent;
            }
        } else {
            for (;;) {
                int child=at*2+1;if (child>=retained) break;
                if (child+1<retained) {
                    int a=indices[child+1]-offset,b=indices[child]-offset;
                    if (workspace[a]<workspace[b] || (workspace[a]==workspace[b] && a>b)) child++;
                }
                int a=indices[child]-offset,b=indices[at]-offset;
                if (!(workspace[a]<workspace[b] || (workspace[a]==workspace[b] && a>b))) break;
                int temp=indices[at];indices[at]=indices[child];indices[child]=temp;at=child;
            }
        }
    }
    if (retained!=topk) return 0;
    for (int end=retained-1;end>0;end--) {
        int temp=indices[0];indices[0]=indices[end];indices[end]=temp;
        for (int at=0;;) {
            int child=at*2+1;if (child>=end) break;
            if (child+1<end) {
                int a=indices[child+1]-offset,b=indices[child]-offset;
                if (workspace[a]<workspace[b] || (workspace[a]==workspace[b] && a>b)) child++;
            }
            int a=indices[child]-offset,b=indices[at]-offset;
            if (!(workspace[a]<workspace[b] || (workspace[a]==workspace[b] && a>b))) break;
            temp=indices[at];indices[at]=indices[child];indices[child]=temp;at=child;
        }
    }
    return topk;
}

static inline int dsv4_indexer_topk(const float *query, const float *kv,
                                    const float *head_weights, int heads,
                                    int dim, int tokens, int topk, int offset,
                                    float *workspace, int *indices) {
    if (!query || !kv || !head_weights || !workspace || !indices ||
        heads <= 0 || dim <= 0 || tokens <= 0 || topk <= 0 ||
        topk > tokens || offset < 0)
        return 0;
    for (int token = 0; token < tokens; token++) {
        float score = 0.0f;
        for (int head = 0; head < heads; head++) {
            float dot = 0.0f;
            for (int axis = 0; axis < dim; axis++)
                dot += query[(size_t)head * dim + axis] *
                       kv[(size_t)token * dim + axis];
            score += dsv4_round_bf16(fmaxf(dsv4_round_bf16(dot), 0.0f) * head_weights[head]);
        }
        workspace[token] = dsv4_round_bf16(score);
    }
    return dsv4_indexer_select(workspace,tokens,topk,offset,indices);
}

/* Pinned sparse_attn_kernel: online softmax in 64-key blocks, FP32
 * denominator, BF16 probabilities for the value GEMM, and BF16 output. */
static inline void dsv4_sparse_attention(float *out, const float *q,
    const float *kv, int heads, int dim, const int *indices,
    int topk, const float *sink, float scale) {
    for (int h=0;h<heads;h++) {
        float maximum=-INFINITY,denominator=0.0f;
        float *output=out+(size_t)h*dim;
        for (int d=0;d<dim;d++) output[d]=0;
        for (int start=0;start<topk;start+=64) {
            int count=topk-start;if (count>64) count=64;
            float scores[64],previous=maximum;
            for (int t=0;t<count;t++) {
                int index=indices[start+t];float dot=0;
                if (index>=0) for (int d=0;d<dim;d++) dot+=q[(size_t)h*dim+d]*kv[(size_t)index*dim+d];
                scores[t]=index>=0?dot*scale:-INFINITY;
                maximum=fmaxf(maximum,scores[t]);
            }
            if (maximum==-INFINITY) continue;
            float factor=expf(previous-maximum),sum=0;
            for (int d=0;d<dim;d++) output[d]*=factor;
            for (int t=0;t<count;t++) {
                float probability=expf(scores[t]-maximum);sum+=probability;
                scores[t]=dsv4_round_bf16(probability);
            }
            denominator=denominator*factor+sum;
            for (int t=0;t<count;t++) if (indices[start+t]>=0)
                for (int d=0;d<dim;d++) output[d]+=scores[t]*kv[(size_t)indices[start+t]*dim+d];
        }
        denominator+=expf(sink[h]-maximum);
        for (int d=0;d<dim;d++) output[d]=dsv4_round_bf16(output[d]/denominator);
    }
}

#endif
