#include <cuda_bf16.h>
#include <mma.h>
/* Native DeepSeek-V4 FP8/FP4 CUDA kernels.
 *
 * Checkpoint layouts and rounding boundaries match deepseek_v4.h.  The
 * generic projections write BF16-rounded values into fp32 buffers.  Routed
 * experts preserve the official sequence:
 *   FP4 projection -> BF16 -> route*clamped-SwiGLU -> BF16
 *   -> MXFP8 activation quantization -> FP4 down -> BF16 -> FP32 reduction.
 */

__device__ __forceinline__ float dsv4_cuda_fp8(unsigned char code) {
    unsigned exponent=(code>>3)&15u,mantissa=code&7u,sign=(unsigned)(code&128u)<<24;
    if (exponent==15 && mantissa==7) return NAN;
    if (!exponent) {float value=(float)mantissa*0.001953125f;return sign?-value:value;}
    return __uint_as_float(sign|((exponent+120u)<<23)|(mantissa<<20));
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
    return code == 255 ? NAN : __uint_as_float(code ? (unsigned)code<<23 : 0x00400000u);
}

__device__ __forceinline__ float dsv4_cuda_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ unsigned char dsv4_cuda_fp8_encode(float value) {
    unsigned bits=__float_as_uint(value),magnitude=bits&0x7fffffffu,sign=(bits>>24)&128u;
    if (magnitude>0x7f800000u) return 0x7f;
    if (magnitude>0x43e00000u) magnitude=0x43e00000u;
    unsigned raw;
    if (magnitude<0x3c800000u) raw=__float2uint_rn(__uint_as_float(magnitude)*512.0f);
    else {
        raw=(magnitude>>20)-960u;
        unsigned remainder=magnitude&0x000fffffu;
        if (remainder>0x00080000u || (remainder==0x00080000u && (raw&1))) raw++;
    }
    return (unsigned char)((raw>0x7e?0x7e:raw)|sign);
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

/* Bounded tensor-core tiles. Decode native FP8/FP4 values into shared half
 * tiles only; preserve the scale-block accumulation order and BF16 boundary.
 * FP8/FP4 values are exactly representable in half, so this adds no weight
 * quantization and never expands a complete matrix. */
template <bool FP4>
__global__ static void dsv4_native_tiled_gemm_kernel(
    float *out,const unsigned char *act,const unsigned char *act_scale,
    const unsigned char *weight,const unsigned char *weight_scale,
    int batch,int rows,int cols) {
    using namespace nvcuda;
    constexpr int K=FP4?32:128;
    __shared__ __align__(32) half a_tile[16*K];
    __shared__ __align__(32) half b_tile[64*K];
    __shared__ __align__(32) float products[4*256];
    const int tid=threadIdx.x,warp=tid/32;
    const int sample_base=blockIdx.y*16,row_base=blockIdx.x*64;
    float totals[8]={0};
    for (int block=0;block<cols/K;block++) {
        for (int i=tid;i<16*K;i+=128) {
            int sample=sample_base+i/K,col=block*K+i%K;
            a_tile[i]=__float2half_rn(sample<batch?dsv4_cuda_fp8(act[(size_t)sample*cols+col]):0);
        }
        for (int i=tid;i<64*K;i+=128) {
            int row=row_base+i/K,col=block*K+i%K;
            float value=0;
            if (row<rows) {
                if (FP4) {
                    unsigned char packed=weight[(size_t)row*(cols/2)+col/2];
                    value=dsv4_cuda_fp4((col&1)?packed>>4:packed&15);
                } else value=dsv4_cuda_fp8(weight[(size_t)row*cols+col]);
            }
            b_tile[i]=__float2half_rn(value);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::fill_fragment(c,0.0f);
        for (int k=0;k<K;k+=16) {
            wmma::load_matrix_sync(a,a_tile+k,K);
            wmma::load_matrix_sync(b,b_tile+warp*16*K+k,K);
            wmma::mma_sync(c,a,b,c);
        }
        wmma::store_matrix_sync(products+warp*256,c,16,wmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for (int j=0;j<8;j++) {
            int i=tid+j*128,sample=sample_base+i/64,row=row_base+i%64;
            if (sample<batch && row<rows) {
                float av=dsv4_cuda_ue8m0(act_scale[(size_t)sample*(cols/128)+(block*K)/128]);
                float wv=dsv4_cuda_ue8m0(weight_scale[FP4 ? (size_t)row*(cols/32)+block : (size_t)(row/128)*(cols/128)+block]);
                totals[j]+=products[(i%64)/16*256+(i/64)*16+i%16]*av*wv;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j=0;j<8;j++) {
        int i=tid+j*128,sample=sample_base+i/64,row=row_base+i%64;
        if (sample<batch && row<rows) out[(size_t)sample*rows+row]=dsv4_cuda_bf16(totals[j]);
    }
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
        route_weight[expert] * ((g / (1.0f + expf(-g))) * u));
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


/* Match the released 64-key online softmax and BF16 probability cast.
 * Dot products and value accumulation retain their original token order. */
__global__ static void dsv4_sparse_attention_kernel(float *out,const float *query,const float *kv,
    const int *indices,const float *sink,int heads,int dim,int selected,float scale,
    const int *counts,int stride) {
    if (counts) {
        query+=(size_t)blockIdx.y*heads*dim;out+=(size_t)blockIdx.y*heads*dim;
        indices+=(size_t)blockIdx.y*stride;selected=counts[blockIdx.y];
    }
    extern __shared__ float scores[];
    __shared__ float denominator,maximum,factor;
    int head=blockIdx.x;
    for (int t=threadIdx.x;t<selected;t+=blockDim.x) {
        int index=indices[t];float value=0;
        if (index>=0) for (int d=0;d<dim;d++) value+=query[(size_t)head*dim+d]*kv[(size_t)index*dim+d];
        scores[t]=index>=0?value*scale:-INFINITY;
    }
    for (int d=threadIdx.x;d<dim;d+=blockDim.x) out[(size_t)head*dim+d]=0;
    if (!threadIdx.x) {denominator=0;maximum=-INFINITY;}
    __syncthreads();
    for (int start=0;start<selected;start+=64) {
        int end=start+64;if (end>selected) end=selected;
        if (!threadIdx.x) {
            float previous=maximum,sum=0;
            for (int t=start;t<end;t++) maximum=fmaxf(maximum,scores[t]);
            factor=maximum==-INFINITY?1:expf(previous-maximum);
            for (int t=start;t<end;t++) {
                float p=indices[t]>=0?expf(scores[t]-maximum):0;
                sum+=p;scores[t]=dsv4_cuda_bf16(p);
            }
            denominator=denominator*factor+sum;
        }
        __syncthreads();
        for (int d=threadIdx.x;d<dim;d+=blockDim.x) {
            float value=out[(size_t)head*dim+d]*factor;
            for (int t=start;t<end;t++) if (indices[t]>=0) value+=scores[t]*kv[(size_t)indices[t]*dim+d];
            out[(size_t)head*dim+d]=value;
        }
        __syncthreads();
    }
    if (!threadIdx.x) denominator+=expf(sink[head]-maximum);
    __syncthreads();
    for (int d=threadIdx.x;d<dim;d+=blockDim.x)
        out[(size_t)head*dim+d]=dsv4_cuda_bf16(out[(size_t)head*dim+d]/denominator);
}
__global__ static void dsv4_index_scores_kernel(float *scores,const float *query,const float *kv,
    const float *weights,int heads,int dim,int tokens) {
    __shared__ float head_scores[128];
    int token=blockIdx.x,head=threadIdx.x;
    if (head<heads) {
        float dot=0;
        for (int d=0;d<dim;d++) dot+=query[(size_t)head*dim+d]*kv[(size_t)token*dim+d];
        head_scores[head]=dsv4_cuda_bf16(fmaxf(dsv4_cuda_bf16(dot),0)*weights[head]);
    }
    __syncthreads();
    if (!head) {
        float sum=0;for (int h=0;h<heads;h++) sum+=head_scores[h];scores[token]=dsv4_cuda_bf16(sum);
    }
}
extern "C" int coli_cuda_dsv4_sparse_attention(ColiCuda *ctx,float *out,const float *query,const float *kv,
    const int *indices,const float *sink,int heads,int dim,int selected,float scale) {
    if (!ctx || !out || !query || !kv || !indices || !sink || heads<1 || dim<1 || selected<1 || selected>2048)
        return fail_arg("DeepSeek sparse attention: invalid argument");
    dsv4_sparse_attention_kernel<<<heads,128,selected*sizeof(float),ctx->stream>>>(out,query,kv,indices,sink,heads,dim,selected,scale,NULL,selected);
    return launch_status("dsv4_sparse_attention_kernel");
}
extern "C" int coli_cuda_dsv4_sparse_attention_batch(ColiCuda *ctx,float *out,const float *query,const float *kv,
    const int *indices,const int *counts,const float *sink,int batch,int heads,int dim,int stride,float scale) {
    if (!ctx || !out || !query || !kv || !indices || !counts || !sink || batch<1 || batch>2048 || heads<1 || dim<1 || stride<1 || stride>2048)
        return fail_arg("DeepSeek batched sparse attention: invalid argument");
    dsv4_sparse_attention_kernel<<<dim3(heads,batch),128,stride*sizeof(float),ctx->stream>>>(
        out,query,kv,indices,sink,heads,dim,stride,scale,counts,stride);
    return launch_status("dsv4_sparse_attention_batch_kernel");
}
extern "C" int coli_cuda_dsv4_index_scores(ColiCuda *ctx,float *scores,const float *query,const float *kv,
    const float *weights,int heads,int dim,int tokens) {
    if (!ctx || !scores || !query || !kv || !weights || heads<1 || heads>128 || dim<1 || tokens<1)
        return fail_arg("DeepSeek index scores: invalid argument");
    dsv4_index_scores_kernel<<<tokens,128,0,ctx->stream>>>(scores,query,kv,weights,heads,dim,tokens);
    return launch_status("dsv4_index_scores_kernel");
}

/* Released wo_a is stored as FP8 but evaluated with BF16 activations and
 * dequantized BF16 weights. Decode only a shared tile; do not quantize input. */
__global__ static void dsv4_fp8_weight_bf16_vector_kernel(float *out,const float *input,
    const unsigned char *weight,const unsigned char *scale,int rows,int cols) {
    int row=blockIdx.x,sample=blockIdx.y;float sum=0;
    for (int k=threadIdx.x;k<cols;k+=blockDim.x) {
        float w=dsv4_cuda_bf16(dsv4_cuda_fp8(weight[(size_t)row*cols+k])*dsv4_cuda_ue8m0(scale[(size_t)(row/128)*(cols/128)+k/128]));
        sum+=dsv4_cuda_bf16(input[(size_t)sample*cols+k])*w;
    }
    float total=block_sum(sum);if (!threadIdx.x) out[(size_t)sample*rows+row]=dsv4_cuda_bf16(total);
}
__global__ static void dsv4_fp8_weight_bf16_tiled_kernel(float *out,const float *input,
    const unsigned char *weight,const unsigned char *scale,int batch,int rows,int cols) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 a_tile[16*32],b_tile[64*32];
    __shared__ __align__(32) float products[4*256];
    int tid=threadIdx.x,warp=tid/32,sample_base=blockIdx.y*16,row_base=blockIdx.x*64;
    wmma::fragment<wmma::accumulator,16,16,16,float> c;wmma::fill_fragment(c,0);
    for (int block=0;block<cols/32;block++) {
        for (int i=tid;i<16*32;i+=128) {int sample=sample_base+i/32,k=block*32+i%32;a_tile[i]=__float2bfloat16_rn(sample<batch?input[(size_t)sample*cols+k]:0);}
        for (int i=tid;i<64*32;i+=128) {
            int row=row_base+i/32,k=block*32+i%32;float w=0;
            if (row<rows) w=dsv4_cuda_fp8(weight[(size_t)row*cols+k])*dsv4_cuda_ue8m0(scale[(size_t)(row/128)*(cols/128)+k/128]);
            b_tile[i]=__float2bfloat16_rn(w);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
        for (int k=0;k<32;k+=16) {wmma::load_matrix_sync(a,a_tile+k,32);wmma::load_matrix_sync(b,b_tile+warp*16*32+k,32);wmma::mma_sync(c,a,b,c);}
        __syncthreads();
    }
    wmma::store_matrix_sync(products+warp*256,c,16,wmma::mem_row_major);__syncthreads();
    for (int i=tid;i<16*64;i+=128) {
        int sample=sample_base+i/64,row=row_base+i%64;
        if (sample<batch && row<rows) out[(size_t)sample*rows+row]=dsv4_cuda_bf16(products[(i%64)/16*256+(i/64)*16+i%16]);
    }
}
extern "C" int coli_cuda_dsv4_fp8_weight_bf16_gemm(ColiCuda *ctx,float *out,const float *input,
    const unsigned char *weight,const unsigned char *scale,int batch,int rows,int cols) {
    if (!ctx || !out || !input || !weight || !scale || batch<1 || rows<1 || cols<1 || cols%128)
        return fail_arg("DeepSeek FP8-weight BF16 GEMM: invalid argument");
    if (batch<8) {dim3 grid(rows,batch);dsv4_fp8_weight_bf16_vector_kernel<<<grid,128,0,ctx->stream>>>(out,input,weight,scale,rows,cols);}
    else {dim3 grid((rows+63)/64,(batch+15)/16);dsv4_fp8_weight_bf16_tiled_kernel<<<grid,128,0,ctx->stream>>>(out,input,weight,scale,batch,rows,cols);}
    return launch_status("dsv4_fp8_weight_bf16_gemm");
}

extern "C" int coli_cuda_dsv4_fp8_gemm(
    ColiCuda *ctx, float *out, const unsigned char *act,
    const unsigned char *act_scale, const unsigned char *weight,
    const unsigned char *weight_scale, int batch, int rows, int cols) {
    if (!ctx || !out || !act || !act_scale || !weight || !weight_scale ||
        batch <= 0 || rows <= 0 || cols <= 0 || cols % 128)
        return fail_arg("DeepSeek FP8 GEMM: invalid argument");
    if (batch>=8) {
        dim3 tiles((rows+63)/64,(batch+15)/16);
        dsv4_native_tiled_gemm_kernel<false><<<tiles,128,0,ctx->stream>>>(out,act,act_scale,weight,weight_scale,batch,rows,cols);
        return launch_status("dsv4_fp8_tiled_gemm_kernel");
    }
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
    if (batch>=8) {
        dim3 tiles((rows+63)/64,(batch+15)/16);
        dsv4_native_tiled_gemm_kernel<true><<<tiles,128,0,ctx->stream>>>(out,act,act_scale,weight,weight_scale,batch,rows,cols);
        return launch_status("dsv4_fp4_tiled_gemm_kernel");
    }
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

/* Split form of the grouped experts for serving. The middle stage -- route
 * weight, clamped SwiGLU, BF16 rounding and MXFP quantization -- runs on the
 * host between the two halves with the same code as the per-expert path, so
 * the result is byte-identical to it. The fused kernel above evaluates expf
 * on the device, which differs from libm by one ulp in about a third of
 * arguments and flips the BF16 middle value roughly once per 4 million
 * elements: rare, but enough to move a near-tie token in a long decode. */
extern "C" int coli_cuda_dsv4_grouped_fp4_hidden(
    ColiCuda *ctx, float *gate_host, float *up_host,
    const unsigned char *act, const unsigned char *act_scale,
    const unsigned char *const *w1, const unsigned char *const *s1,
    const unsigned char *const *w3, const unsigned char *const *s3,
    int experts, int hidden, int intermediate) {
    if (!ctx || !gate_host || !up_host || !act || !act_scale || !w1 || !s1 ||
        !w3 || !s3 || experts <= 0 || experts > 32 || hidden <= 0 ||
        intermediate <= 0 || hidden % 128 || intermediate % 128)
        return fail_arg("DeepSeek grouped FP4 hidden: invalid argument");
    size_t pointer_bytes = (size_t)experts * 4 * sizeof(void *);
    size_t hidden_bytes = (size_t)experts * intermediate * sizeof(float);
    if (ensure_cuda_buffer((void **)&ctx->group_ptrs, &ctx->group_ptr_cap,
                           pointer_bytes, "DeepSeek grouped setup allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_gate, &ctx->group_hidden_cap,
                           hidden_bytes, "DeepSeek grouped gate allocation") ||
        ensure_cuda_buffer((void **)&ctx->group_up, &ctx->group_up_cap,
                           hidden_bytes, "DeepSeek grouped up allocation"))
        return -1;
    unsigned char setup[32 * 4 * sizeof(void *)];
    const void *host_pointers[4] = {w1, s1, w3, s3};
    for (int set = 0; set < 4; set++)
        memcpy(setup + (size_t)set * experts * sizeof(void *),
               host_pointers[set], (size_t)experts * sizeof(void *));
    cudaError_t error = cudaMemcpyAsync(ctx->group_ptrs, setup, pointer_bytes,
                                        cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
        return fail_cuda("DeepSeek grouped hidden setup upload", error);
    const unsigned char **device_w1 = (const unsigned char **)ctx->group_ptrs;
    const unsigned char **device_s1 = (const unsigned char **)(ctx->group_ptrs + experts);
    const unsigned char **device_w3 = (const unsigned char **)(ctx->group_ptrs + 2 * experts);
    const unsigned char **device_s3 = (const unsigned char **)(ctx->group_ptrs + 3 * experts);
    dim3 hidden_grid((unsigned)intermediate, (unsigned)experts);
    dsv4_group_hidden_kernel<<<hidden_grid, 128, 0, ctx->stream>>>(
        ctx->group_gate, ctx->group_up, act, act_scale, device_w1, device_s1,
        device_w3, device_s3, hidden, intermediate);
    if (launch_status("dsv4_group_hidden_kernel")) return -1;
    error = cudaMemcpyAsync(gate_host, ctx->group_gate, hidden_bytes,
                            cudaMemcpyDeviceToHost, ctx->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(up_host, ctx->group_up, hidden_bytes,
                                cudaMemcpyDeviceToHost, ctx->stream);
    return error == cudaSuccess ? 0 : fail_cuda("DeepSeek grouped hidden download", error);
}

extern "C" int coli_cuda_dsv4_grouped_fp4_down(
    ColiCuda *ctx, float *out, const unsigned char *middle_host,
    const unsigned char *middle_scale_host,
    const unsigned char *const *w2, const unsigned char *const *s2,
    int experts, int hidden, int intermediate) {
    if (!ctx || !out || !middle_host || !middle_scale_host || !w2 || !s2 ||
        experts <= 0 || experts > 32 || hidden <= 0 || intermediate <= 0 ||
        hidden % 128 || intermediate % 128)
        return fail_arg("DeepSeek grouped FP4 down: invalid argument");
    size_t pointer_bytes = (size_t)experts * 2 * sizeof(void *);
    size_t output_bytes = (size_t)experts * hidden * sizeof(float);
    size_t act_bytes = (size_t)experts * intermediate;
    size_t scale_bytes = (size_t)experts * (intermediate / 128);
    if (ensure_cuda_buffer((void **)&ctx->group_ptrs, &ctx->group_ptr_cap,
                           pointer_bytes, "DeepSeek grouped setup allocation") ||
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
    unsigned char setup[32 * 2 * sizeof(void *)];
    memcpy(setup, w2, (size_t)experts * sizeof(void *));
    memcpy(setup + (size_t)experts * sizeof(void *), s2, (size_t)experts * sizeof(void *));
    cudaError_t error = cudaMemcpyAsync(ctx->group_ptrs, setup, pointer_bytes,
                                        cudaMemcpyHostToDevice, ctx->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(ctx->dsv4_group_act, middle_host, act_bytes,
                                cudaMemcpyHostToDevice, ctx->stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(ctx->dsv4_group_scale, middle_scale_host, scale_bytes,
                                cudaMemcpyHostToDevice, ctx->stream);
    if (error != cudaSuccess)
        return fail_cuda("DeepSeek grouped down upload", error);
    const unsigned char **device_w2 = (const unsigned char **)ctx->group_ptrs;
    const unsigned char **device_s2 = (const unsigned char **)(ctx->group_ptrs + experts);
    dim3 down_grid((unsigned)hidden, (unsigned)experts);
    dsv4_group_down_kernel<<<down_grid, 128, 0, ctx->stream>>>(
        ctx->group_route_out, ctx->dsv4_group_act, ctx->dsv4_group_scale,
        device_w2, device_s2, hidden, intermediate);
    if (launch_status("dsv4_group_down_kernel")) return -1;
    dsv4_group_reduce_kernel<<<(hidden + 255) / 256, 256, 0, ctx->stream>>>(
        out, ctx->group_route_out, experts, hidden);
    return launch_status("DeepSeek grouped FP4 down kernels");
}

extern "C" int coli_cuda_dsv4_memory_stats(ColiCuda *ctx,unsigned long long *used_peak,unsigned long long *reserved_peak,
    unsigned long long *scratch_bytes,size_t *available,size_t *total) {
    if (!ctx || !used_peak || !reserved_peak || !scratch_bytes || !available || !total)
        return fail_arg("DeepSeek memory statistics: invalid argument");
    *used_peak=*reserved_peak=0;
    *scratch_bytes=ctx->group_ptr_cap+ctx->group_hidden_cap+ctx->group_up_cap+ctx->group_route_out_cap+
        ctx->dsv4_group_act_cap+ctx->dsv4_group_scale_cap;
    cudaError_t error=cudaMemGetInfo(available,total);
    if (error!=cudaSuccess) return fail_cuda("DeepSeek memory statistics",error);
    if (!ctx->async_alloc) return 0;
    cudaMemPool_t pool;error=cudaDeviceGetDefaultMemPool(&pool,ctx->device);
    if (error==cudaSuccess) error=cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemHigh,used_peak);
    if (error==cudaSuccess) error=cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemHigh,reserved_peak);
    return error==cudaSuccess?0:fail_cuda("DeepSeek pool statistics",error);
}
