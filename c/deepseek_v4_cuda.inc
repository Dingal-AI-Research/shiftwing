/* Native DeepSeek-V4 FP8/FP4 CUDA kernels.
 *
 * Checkpoint layouts and rounding boundaries match deepseek_v4.h.  The
 * generic projections write BF16-rounded values into fp32 buffers.  Routed
 * experts preserve the official sequence:
 *   FP4 projection -> BF16 -> route*clamped-SwiGLU -> BF16
 *   -> MXFP8 activation quantization -> FP4 down -> BF16 -> FP32 reduction.
 */

__device__ __forceinline__ float dsv4_cuda_fp8(unsigned char code) {
    int sign = code >> 7;
    int exponent = (code >> 3) & 15;
    int mantissa = code & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float value = exponent
        ? ldexpf(1.0f + (float)mantissa * 0.125f, exponent - 7)
        : ldexpf((float)mantissa, -9);
    return sign ? -value : value;
}

__device__ __forceinline__ float dsv4_cuda_fp4(unsigned char code) {
    int magnitude = code & 7;
    float value;
    switch (magnitude) {
        case 0: value = 0.0f; break;
        case 1: value = 0.5f; break;
        case 2: value = 1.0f; break;
        case 3: value = 1.5f; break;
        case 4: value = 2.0f; break;
        case 5: value = 3.0f; break;
        case 6: value = 4.0f; break;
        default: value = 6.0f; break;
    }
    return (code & 8) ? -value : value;
}

__device__ __forceinline__ float dsv4_cuda_ue8m0(unsigned char code) {
    return code == 255 ? NAN : ldexpf(1.0f, (int)code - 127);
}

__device__ __forceinline__ float dsv4_cuda_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ unsigned char dsv4_cuda_fp8_encode(float value) {
    if (isnan(value)) return 0x7f;
    bool negative = signbit(value);
    float target = fminf(fabsf(value), 448.0f);
    int raw;
    if (target < 0.015625f) {
        raw = (int)nearbyintf(target * 512.0f);
        raw = raw < 0 ? 0 : (raw > 8 ? 8 : raw);
    } else {
        int exponent;
        (void)frexpf(target, &exponent);
        int unbiased = exponent - 1;
        int encoded_exponent = unbiased + 7;
        float normalized = ldexpf(target, -unbiased);
        int mantissa = (int)nearbyintf((normalized - 1.0f) * 8.0f);
        if (mantissa == 8) {
            encoded_exponent++;
            mantissa = 0;
        }
        raw = (encoded_exponent << 3) | mantissa;
        if (raw > 0x7e) raw = 0x7e;
    }
    return (unsigned char)(raw | (negative ? 0x80 : 0));
}

__device__ __forceinline__ float dsv4_cuda_warp_max(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

__device__ __forceinline__ float dsv4_cuda_block_max(float value) {
    __shared__ float warp_values[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = dsv4_cuda_warp_max(value);
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32
        ? warp_values[lane] : -INFINITY;
    if (warp == 0) value = dsv4_cuda_warp_max(value);
    return value;
}
__global__ static void dsv4_bf16_gemm_kernel(
    float *output, const float *input, const unsigned short *weight,
    int rows, int columns) {
    int row = blockIdx.x;
    int sample = blockIdx.y;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns;
         column += blockDim.x) {
        unsigned int bits =
            (unsigned int)weight[(size_t)row * columns + column] << 16;
        sum += input[(size_t)sample * columns + column] *
               __uint_as_float(bits);
    }
    float total = block_sum(sum);
    if (threadIdx.x == 0)
        output[(size_t)sample * rows + row] = total;
}


__global__ static void dsv4_fp8_gemm_kernel(
    float *out, const unsigned char *act, const unsigned char *act_scale,
    const unsigned char *weight, const unsigned char *weight_scale,
    int rows, int cols) {
    int row = blockIdx.x;
    int sample = blockIdx.y;
    if (row >= rows) return;
    int blocks = cols / 128;
    float total = 0.0f;
    for (int block = 0; block < blocks; block++) {
        int column = block * 128 + threadIdx.x;
        float product = threadIdx.x < 128
            ? dsv4_cuda_fp8(act[(size_t)sample * cols + column]) *
              dsv4_cuda_fp8(weight[(size_t)row * cols + column])
            : 0.0f;
        float inner = block_sum(product);
        if (threadIdx.x == 0) {
            float as = dsv4_cuda_ue8m0(
                act_scale[(size_t)sample * blocks + block]);
            float ws = dsv4_cuda_ue8m0(
                weight_scale[(size_t)(row / 128) * blocks + block]);
            total += inner * as * ws;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[(size_t)sample * rows + row] = dsv4_cuda_bf16(total);
}

__global__ static void dsv4_fp4_gemm_kernel(
    float *out, const unsigned char *act, const unsigned char *act_scale,
    const unsigned char *weight, const unsigned char *weight_scale,
    int rows, int cols) {
    int row = blockIdx.x;
    int sample = blockIdx.y;
    if (row >= rows) return;
    int activation_blocks = cols / 128;
    int weight_blocks = cols / 32;
    float total = 0.0f;
    for (int block = 0; block < weight_blocks; block++) {
        int column = block * 32 + threadIdx.x;
        float product = 0.0f;
        if (threadIdx.x < 32) {
            unsigned char packed =
                weight[(size_t)row * (cols / 2) + column / 2];
            unsigned char code =
                (column & 1) ? (packed >> 4) : (packed & 15);
            product = dsv4_cuda_fp8(
                act[(size_t)sample * cols + column]) * dsv4_cuda_fp4(code);
        }
        float inner = block_sum(product);
        if (threadIdx.x == 0) {
            float as = dsv4_cuda_ue8m0(
                act_scale[(size_t)sample * activation_blocks + block / 4]);
            float ws = dsv4_cuda_ue8m0(
                weight_scale[(size_t)row * weight_blocks + block]);
            total += inner * as * ws;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[(size_t)sample * rows + row] = dsv4_cuda_bf16(total);
}

__global__ static void dsv4_clamped_swiglu_kernel(
    float *out, const float *gate, const float *up, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float g = fminf(10.0f, gate[index]);
    float u = fminf(10.0f, fmaxf(-10.0f, up[index]));
    out[index] = (g / (1.0f + expf(-g))) * u;
}

__global__ static void dsv4_group_hidden_kernel(
    float *gate, float *up, const unsigned char *act,
    const unsigned char *act_scale, const unsigned char *const *w1,
    const unsigned char *const *s1, const unsigned char *const *w3,
    const unsigned char *const *s3, int hidden, int intermediate) {
    int row = blockIdx.x;
    int expert = blockIdx.y;
    if (row >= intermediate) return;
    int weight_blocks = hidden / 32;
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (int block = 0; block < weight_blocks; block++) {
        int column = block * 32 + threadIdx.x;
        float gate_product = 0.0f, up_product = 0.0f;
        if (threadIdx.x < 32) {
            float a = dsv4_cuda_fp8(act[column]);
            unsigned char gate_packed =
                w1[expert][(size_t)row * (hidden / 2) + column / 2];
            unsigned char up_packed =
                w3[expert][(size_t)row * (hidden / 2) + column / 2];
            unsigned char gate_code =
                (column & 1) ? gate_packed >> 4 : gate_packed & 15;
            unsigned char up_code =
                (column & 1) ? up_packed >> 4 : up_packed & 15;
            gate_product = a * dsv4_cuda_fp4(gate_code);
            up_product = a * dsv4_cuda_fp4(up_code);
        }
        float gate_inner = block_sum(gate_product);
        __shared__ float saved_gate;
        if (threadIdx.x == 0) saved_gate = gate_inner;
        __syncthreads();
        float up_inner = block_sum(up_product);
        if (threadIdx.x == 0) {
            float as = dsv4_cuda_ue8m0(act_scale[block / 4]);
            gate_total += saved_gate * as *
                dsv4_cuda_ue8m0(s1[expert][(size_t)row * weight_blocks + block]);
            up_total += up_inner * as *
                dsv4_cuda_ue8m0(s3[expert][(size_t)row * weight_blocks + block]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        size_t index = (size_t)expert * intermediate + row;
        gate[index] = dsv4_cuda_bf16(gate_total);
        up[index] = dsv4_cuda_bf16(up_total);
    }
}

__global__ static void dsv4_group_middle_quant_kernel(
    float *middle, const float *up, unsigned char *quantized,
    unsigned char *scales, const float *route_weight, int intermediate) {
    int block = blockIdx.x;
    int expert = blockIdx.y;
    int column = block * 128 + threadIdx.x;
    size_t index = (size_t)expert * intermediate + column;
    float g = middle[index];
    float u = fminf(10.0f, fmaxf(-10.0f, up[index]));
    g = fminf(10.0f, g);
    float value = dsv4_cuda_bf16(
        route_weight[expert] * (g / (1.0f + expf(-g))) * u);
    middle[index] = value;
    float maximum = dsv4_cuda_block_max(fmaxf(1e-4f, fabsf(value)));
    __shared__ float scale;
    if (threadIdx.x == 0) {
        int exponent = (int)ceilf(log2f(maximum / 448.0f));
        exponent = exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
        scale = ldexpf(1.0f, exponent);
        scales[(size_t)expert * (intermediate / 128) + block] =
            (unsigned char)(exponent + 127);
    }
    __syncthreads();
    quantized[index] = dsv4_cuda_fp8_encode(value / scale);
}

__global__ static void dsv4_group_down_kernel(
    float *route_out, const unsigned char *act,
    const unsigned char *act_scale, const unsigned char *const *w2,
    const unsigned char *const *s2, int hidden, int intermediate) {
    int row = blockIdx.x;
    int expert = blockIdx.y;
    if (row >= hidden) return;
    int activation_blocks = intermediate / 128;
    int weight_blocks = intermediate / 32;
    float total = 0.0f;
    for (int block = 0; block < weight_blocks; block++) {
        int column = block * 32 + threadIdx.x;
        float product = 0.0f;
        if (threadIdx.x < 32) {
            unsigned char packed =
                w2[expert][(size_t)row * (intermediate / 2) + column / 2];
            unsigned char code =
                (column & 1) ? packed >> 4 : packed & 15;
            product = dsv4_cuda_fp8(
                act[(size_t)expert * intermediate + column]) *
                dsv4_cuda_fp4(code);
        }
        float inner = block_sum(product);
        if (threadIdx.x == 0) {
            float as = dsv4_cuda_ue8m0(
                act_scale[(size_t)expert * activation_blocks + block / 4]);
            float ws = dsv4_cuda_ue8m0(
                s2[expert][(size_t)row * weight_blocks + block]);
            total += inner * as * ws;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        route_out[(size_t)expert * hidden + row] =
            dsv4_cuda_bf16(total);
}

__global__ static void dsv4_group_reduce_kernel(
    float *out, const float *route_out, int experts, int hidden) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= hidden) return;
    float sum = 0.0f;
    for (int expert = 0; expert < experts; expert++)
        sum += route_out[(size_t)expert * hidden + row];
    out[row] = sum;
}
extern "C" int coli_cuda_dsv4_bf16_gemm(
    ColiCuda *ctx, float *output, const float *input,
    const unsigned short *weight, int batch, int rows, int columns) {
    if (!ctx || !output || !input || !weight || batch <= 0 ||
        rows <= 0 || columns <= 0)
        return fail_arg("DeepSeek BF16 GEMM: invalid argument");
    dim3 grid((unsigned)rows, (unsigned)batch);
    dsv4_bf16_gemm_kernel<<<grid, 128, 0, ctx->stream>>>(
        output, input, weight, rows, columns);
    return launch_status("dsv4_bf16_gemm_kernel");
}


extern "C" int coli_cuda_dsv4_fp8_gemm(
    ColiCuda *ctx, float *out, const unsigned char *act,
    const unsigned char *act_scale, const unsigned char *weight,
    const unsigned char *weight_scale, int batch, int rows, int cols) {
    if (!ctx || !out || !act || !act_scale || !weight || !weight_scale ||
        batch <= 0 || rows <= 0 || cols <= 0 || cols % 128)
        return fail_arg("DeepSeek FP8 GEMM: invalid argument");
    dim3 grid((unsigned)rows, (unsigned)batch);
    dsv4_fp8_gemm_kernel<<<grid, 128, 0, ctx->stream>>>(
        out, act, act_scale, weight, weight_scale, rows, cols);
    return launch_status("dsv4_fp8_gemm_kernel");
}

extern "C" int coli_cuda_dsv4_fp4_gemm(
    ColiCuda *ctx, float *out, const unsigned char *act,
    const unsigned char *act_scale, const unsigned char *weight,
    const unsigned char *weight_scale, int batch, int rows, int cols) {
    if (!ctx || !out || !act || !act_scale || !weight || !weight_scale ||
        batch <= 0 || rows <= 0 || cols <= 0 || cols % 128)
        return fail_arg("DeepSeek FP4 GEMM: invalid argument");
    dim3 grid((unsigned)rows, (unsigned)batch);
    dsv4_fp4_gemm_kernel<<<grid, 128, 0, ctx->stream>>>(
        out, act, act_scale, weight, weight_scale, rows, cols);
    return launch_status("dsv4_fp4_gemm_kernel");
}

extern "C" int coli_cuda_dsv4_clamped_swiglu(
    ColiCuda *ctx, float *out, const float *gate, const float *up, int count) {
    if (!ctx || !out || !gate || !up || count <= 0)
        return fail_arg("DeepSeek clamped SwiGLU: invalid argument");
    dsv4_clamped_swiglu_kernel<<<(count + 255) / 256, 256, 0, ctx->stream>>>(
        out, gate, up, count);
    return launch_status("dsv4_clamped_swiglu_kernel");
}

extern "C" int coli_cuda_dsv4_grouped_fp4_experts(
    ColiCuda *ctx, float *out, const unsigned char *act,
    const unsigned char *act_scale,
    const unsigned char *const *w1, const unsigned char *const *s1,
    const unsigned char *const *w2, const unsigned char *const *s2,
    const unsigned char *const *w3, const unsigned char *const *s3,
    const float *route_weight, int experts, int hidden, int intermediate) {
    if (!ctx || !out || !act || !act_scale || !w1 || !s1 || !w2 || !s2 ||
        !w3 || !s3 || !route_weight || experts <= 0 || experts > 32 ||
        hidden <= 0 || intermediate <= 0 || hidden % 128 ||
        intermediate % 128)
        return fail_arg("DeepSeek grouped FP4 experts: invalid argument");

    size_t pointer_bytes = (size_t)experts * 6 * sizeof(void *);
    size_t route_bytes = (size_t)experts * sizeof(float);
    size_t hidden_bytes = (size_t)experts * intermediate * sizeof(float);
    size_t output_bytes = (size_t)experts * hidden * sizeof(float);
    size_t act_bytes = (size_t)experts * intermediate;
    size_t scale_bytes = (size_t)experts * (intermediate / 128);
    if (ensure_cuda_buffer((void **)&ctx->group_ptrs, &ctx->group_ptr_cap,
                           pointer_bytes + route_bytes,
                           "DeepSeek grouped setup allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_gate, &ctx->group_hidden_cap,
                           hidden_bytes, "DeepSeek grouped gate allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_up, &ctx->group_up_cap,
                           hidden_bytes, "DeepSeek grouped up allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_route_out,
                           &ctx->group_route_out_cap, output_bytes,
                           "DeepSeek grouped output allocation") ||
        ensure_cuda_buffer((void **)&ctx->dsv4_group_act,
                           &ctx->dsv4_group_act_cap, act_bytes,
                           "DeepSeek grouped activation allocation") ||
        ensure_cuda_buffer((void **)&ctx->dsv4_group_scale,
                           &ctx->dsv4_group_scale_cap, scale_bytes,
                           "DeepSeek grouped scale allocation"))
        return -1;

    unsigned char setup[32 * 6 * sizeof(void *) + 32 * sizeof(float)];
    const void *host_pointers[6] = {w1, s1, w2, s2, w3, s3};
    for (int set = 0; set < 6; set++)
        memcpy(setup + (size_t)set * experts * sizeof(void *),
               host_pointers[set], (size_t)experts * sizeof(void *));
    memcpy(setup + pointer_bytes, route_weight, route_bytes);
    if (ctx->profile_stages) {
        cudaEventRecord(ctx->profile_event[0], ctx->stream);
        ctx->profile_pending = 0;
        ctx->profile_shared_marked = 0;
        ctx->profile_download_marked = 0;
    }
    cudaError_t error = cudaMemcpyAsync(
        ctx->group_ptrs, setup, pointer_bytes + route_bytes,
        cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
        return fail_cuda("DeepSeek grouped setup upload", error);
    if (ctx->profile_stages)
        cudaEventRecord(ctx->profile_event[1], ctx->stream);

    const unsigned char **device_w1 =
        (const unsigned char **)ctx->group_ptrs;
    const unsigned char **device_s1 =
        (const unsigned char **)(ctx->group_ptrs + experts);
    const unsigned char **device_w2 =
        (const unsigned char **)(ctx->group_ptrs + 2 * experts);
    const unsigned char **device_s2 =
        (const unsigned char **)(ctx->group_ptrs + 3 * experts);
    const unsigned char **device_w3 =
        (const unsigned char **)(ctx->group_ptrs + 4 * experts);
    const unsigned char **device_s3 =
        (const unsigned char **)(ctx->group_ptrs + 5 * experts);
    const float *device_route = (const float *)(
        (const unsigned char *)ctx->group_ptrs + pointer_bytes);

    dim3 hidden_grid((unsigned)intermediate, (unsigned)experts);
    dsv4_group_hidden_kernel<<<hidden_grid, 128, 0, ctx->stream>>>(
        ctx->group_gate, ctx->group_up, act, act_scale, device_w1, device_s1,
        device_w3, device_s3, hidden, intermediate);
    if (launch_status("dsv4_group_hidden_kernel")) return -1;
    if (ctx->profile_stages)
        cudaEventRecord(ctx->profile_event[2], ctx->stream);

    dim3 quant_grid((unsigned)(intermediate / 128), (unsigned)experts);
    dsv4_group_middle_quant_kernel<<<quant_grid, 128, 0, ctx->stream>>>(
        ctx->group_gate, ctx->group_up, ctx->dsv4_group_act,
        ctx->dsv4_group_scale, device_route, intermediate);
    if (launch_status("dsv4_group_middle_quant_kernel")) return -1;

    dim3 down_grid((unsigned)hidden, (unsigned)experts);
    dsv4_group_down_kernel<<<down_grid, 128, 0, ctx->stream>>>(
        ctx->group_route_out, ctx->dsv4_group_act, ctx->dsv4_group_scale,
        device_w2, device_s2, hidden, intermediate);
    if (ctx->profile_stages)
        cudaEventRecord(ctx->profile_event[3], ctx->stream);

    dsv4_group_reduce_kernel<<<(hidden + 255) / 256, 256, 0, ctx->stream>>>(
        out, ctx->group_route_out, experts, hidden);
    int status = launch_status("DeepSeek grouped FP4 expert kernels");
    if (!status && ctx->profile_stages) {
        cudaEventRecord(ctx->profile_event[4], ctx->stream);
        ctx->profile_pending = 1;
        ctx->profile_transactions++;
    }
    return status;
}
