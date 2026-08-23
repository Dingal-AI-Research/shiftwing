#ifndef SHIFTWING_BACKEND_CUDA_H
#define SHIFTWING_BACKEND_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiCuda ColiCuda;

/* All functions return zero on success. The error string is thread-local. */
const char *coli_cuda_last_error(void);
int coli_cuda_create(ColiCuda **out, int device);
void coli_cuda_destroy(ColiCuda *ctx);
const char *coli_cuda_device_name(const ColiCuda *ctx);
int coli_cuda_compute_capability(const ColiCuda *ctx, int *major, int *minor);
int coli_cuda_async_alloc_enabled(const ColiCuda *ctx);
int coli_cuda_memory_info(ColiCuda *ctx, size_t *free_bytes,
                          size_t *total_bytes);

int coli_cuda_malloc(ColiCuda *ctx, void **ptr, size_t bytes);
void coli_cuda_free(ColiCuda *ctx, void *ptr);
int coli_cuda_malloc_host(ColiCuda *ctx, void **ptr, size_t bytes);
void coli_cuda_free_host(ColiCuda *ctx, void *ptr);
int coli_cuda_upload(ColiCuda *ctx, void *dst, const void *src, size_t bytes);
int coli_cuda_download(ColiCuda *ctx, void *dst, const void *src, size_t bytes);
int coli_cuda_copy(ColiCuda *ctx, void *dst, const void *src, size_t bytes);
int coli_cuda_memset(ColiCuda *ctx, void *dst, int value, size_t bytes);
int coli_cuda_sync(ColiCuda *ctx);
void coli_cuda_profile_reset(ColiCuda *ctx);
/* Returns 0 with a snapshot, 1 when stage profiling was not enabled, and -1
 * on error. */
int coli_cuda_profile_snapshot(ColiCuda *ctx,
                               unsigned long long *transactions,
                               double stage_ms[8]);

/* Initial correctness kernels. Buffers are device pointers and use fp32.
 * rmsnorm_zero implements y=x*rsqrt(mean(x^2)+eps)*(1+w).
 * q4 uses the container layout: low nibble first, signed value nibble-8,
 * row stride rb bytes, and one fp32 scale per row/group. */
int coli_cuda_rmsnorm_zero(ColiCuda *ctx, float *y, const float *x,
                           const float *w, int n, float eps);
int coli_cuda_rmsnorm_zero_batch(ColiCuda *ctx, float *y, const float *x,
                                 const float *w, int batch, int n, float eps);
int coli_cuda_f32_gemm(ColiCuda *ctx, float *y, const float *x,
                       const float *w, int batch, int rows, int cols);
int coli_cuda_sigmoid(ColiCuda *ctx, float *y, const float *x, int n);
int coli_cuda_silu_mul(ColiCuda *ctx, float *y, const float *gate,
                       const float *up, int n);
int coli_cuda_axpy(ColiCuda *ctx, float *y, const float *x, float scale, int n);
int coli_cuda_sigmoid_axpy(ColiCuda *ctx, float *y, const float *x,
                           const float *logit, int n);
int coli_cuda_f32_to_f16(ColiCuda *ctx, unsigned short *y, const float *x,
                         int n);
int coli_cuda_q8_gemv(ColiCuda *ctx, float *y, const float *x,
                      const signed char *q, const float *scales,
                      int rows, int cols, int row_bytes);
int coli_cuda_q8_gemm(ColiCuda *ctx, float *y, const float *x, int batch,
                      const signed char *q, const float *scales,
                      int rows, int cols, int row_bytes);
int coli_cuda_q4_gemv(ColiCuda *ctx, float *y, const float *x,
                      const unsigned char *q, const float *scales,
                      int rows, int cols, int group_size,
                      int row_bytes, int groups_per_row);
int coli_cuda_q4_gemm(ColiCuda *ctx, float *y, const float *x, int batch,
                      const unsigned char *q, const float *scales,
                      int rows, int cols, int group_size, int row_bytes,
                      int groups_per_row);
int coli_cuda_q4_gemv_f16(ColiCuda *ctx, float *y, const unsigned short *x,
                          const unsigned char *q, const float *scales,
                          int rows, int cols, int group_size,
                          int row_bytes, int groups_per_row);
int coli_cuda_q3_gemv(ColiCuda *ctx, float *y, const float *x,
                      const unsigned char *q, const float *scales,
                      int rows, int cols, int group_size,
                      int row_bytes, int groups_per_row);
int coli_cuda_q3_gemv_f16(ColiCuda *ctx, float *y,
                          const unsigned short *x,
                          const unsigned char *q, const float *scales,
                          int rows, int cols, int group_size,
                          int row_bytes, int groups_per_row);
int coli_cuda_grouped_q4_mlp_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int experts, int hidden, int intermediate,
    int group_size, int hidden_row_bytes, int hidden_groups,
    int down_row_bytes, int down_groups);
int coli_cuda_grouped_q3_mlp_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int experts, int hidden, int intermediate,
    int group_size, int hidden_row_bytes, int hidden_groups,
    int down_row_bytes, int down_groups);
int coli_cuda_grouped_q4_mlp_batch_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int batch, int experts_per_token, int hidden,
    int intermediate, int group_size, int hidden_row_bytes,
    int hidden_groups, int down_row_bytes, int down_groups);
int coli_cuda_grouped_q3_mlp_batch_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *const *gate_q, const float *const *gate_s,
    const unsigned char *const *up_q, const float *const *up_s,
    const unsigned char *const *down_q, const float *const *down_s,
    const float *route_weight, int batch, int experts_per_token, int hidden,
    int intermediate, int group_size, int hidden_row_bytes,
    int hidden_groups, int down_row_bytes, int down_groups);
int coli_cuda_shared_q4_mlp_batch_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *gate_q, const float *gate_s,
    const unsigned char *up_q, const float *up_s,
    const unsigned char *down_q, const float *down_s,
    const unsigned char *scale_q, const float *scale_s,
    int batch, int hidden, int intermediate, int group_size,
    int hidden_row_bytes, int hidden_groups, int down_row_bytes,
    int down_groups, int scale_row_bytes, int scale_groups);
int coli_cuda_shared_q8_mlp_batch(
    ColiCuda *ctx, float *y, const float *x, const float *scale,
    const signed char *gate_q, const float *gate_s, int gate_row_bytes,
    const signed char *up_q, const float *up_s, int up_row_bytes,
    const signed char *down_q, const float *down_s, int down_row_bytes,
    int batch, int hidden, int intermediate);
int coli_cuda_shared_q8_mlp_batch_device_scale(
    ColiCuda *ctx, float *y, const float *x, const float *device_scale,
    const signed char *gate_q, const float *gate_s, int gate_row_bytes,
    const signed char *up_q, const float *up_s, int up_row_bytes,
    const signed char *down_q, const float *down_s, int down_row_bytes,
    int batch, int hidden, int intermediate);
int coli_cuda_shared_q4_mlp_f16(
    ColiCuda *ctx, float *y, const unsigned short *x,
    const unsigned char *gate_q, const float *gate_s,
    const unsigned char *up_q, const float *up_s,
    const unsigned char *down_q, const float *down_s,
    const unsigned char *scale_q, const float *scale_s,
    int hidden, int intermediate, int group_size, int hidden_row_bytes,
    int hidden_groups, int down_row_bytes, int down_groups,
    int scale_row_bytes, int scale_groups);
int coli_cuda_gdn_decode_q4_f16(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    const unsigned char *qkv_q, const float *qkv_s, int qkv_row_bytes,
    int qkv_groups, const unsigned char *z_q, const float *z_s,
    int z_row_bytes, int z_groups, const unsigned char *out_q,
    const float *out_s, int out_row_bytes, int out_groups,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *norm_weight, float *conv_state, float *recurrent_state,
    int position, int hidden, int key_heads, int value_heads, int key_dim,
    int value_dim, int conv_kernel, int group_size, float eps);
int coli_cuda_gdn_block_q4_f16(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    int batch, const unsigned char *qkv_q, const float *qkv_s,
    int qkv_row_bytes, int qkv_groups, const unsigned char *z_q,
    const float *z_s, int z_row_bytes, int z_groups,
    const unsigned char *out_q, const float *out_s, int out_row_bytes,
    int out_groups, const float *conv_weight, const float *a_log,
    const float *dt_bias, const float *norm_weight, float *conv_state,
    float *recurrent_state, int position, int hidden, int key_heads,
    int value_heads, int key_dim, int value_dim, int conv_kernel,
    int group_size, float eps);
int coli_cuda_gdn_slots_q4_f16(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    int batch, const int *slot, const int *position,
    const unsigned char *qkv_q, const float *qkv_s, int qkv_row_bytes,
    int qkv_groups, const unsigned char *z_q, const float *z_s,
    int z_row_bytes, int z_groups, const unsigned char *out_q,
    const float *out_s, int out_row_bytes, int out_groups,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *norm_weight, float *conv_state, float *recurrent_state,
    int slots, int hidden, int key_heads, int value_heads, int key_dim,
    int value_dim, int conv_kernel, int group_size, float eps);
/* Device-I/O form of the resident slot operation. x/a/b/out are CUDA
 * pointers; weights and slot metadata retain the same ownership as above. */
int coli_cuda_gdn_slots_q4_f16_device(
    ColiCuda *ctx, float *out, const float *x, const float *a, const float *b,
    int batch, const int *slot, const int *position,
    const unsigned char *qkv_q, const float *qkv_s, int qkv_row_bytes,
    int qkv_groups, const unsigned char *z_q, const float *z_s,
    int z_row_bytes, int z_groups, const unsigned char *out_q,
    const float *out_s, int out_row_bytes, int out_groups,
    const float *conv_weight, const float *a_log, const float *dt_bias,
    const float *norm_weight, float *conv_state, float *recurrent_state,
    int slots, int hidden, int key_heads, int value_heads, int key_dim,
    int value_dim, int conv_kernel, int group_size, float eps);
int coli_cuda_gqa_decode_q4_f16(
    ColiCuda *ctx, float *out, const float *x,
    const unsigned char *q_q, const float *q_s, int q_rb, int q_ng,
    const unsigned char *k_q, const float *k_s, int k_rb, int k_ng,
    const unsigned char *v_q, const float *v_s, int v_rb, int v_ng,
    const void *o_q, const float *o_s, int o_fmt, int o_rb, int o_ng,
    const float *q_norm, const float *k_norm, float *k_cache,
    float *v_cache, int position, int hidden, int query_heads,
    int kv_heads, int head_dim, int rotary_dim, int group_size,
    float theta, float eps);
int coli_cuda_gqa_slots_q4_f16(
    ColiCuda *ctx, float *out, const float *x, int batch, const int *slot,
    const int *position, const unsigned char *q_q, const float *q_s,
    int q_rb, int q_ng, const unsigned char *k_q, const float *k_s,
    int k_rb, int k_ng, const unsigned char *v_q, const float *v_s,
    int v_rb, int v_ng, const void *o_q, const float *o_s, int o_fmt,
    int o_rb, int o_ng, const float *q_norm, const float *k_norm,
    float *k_cache, float *v_cache, int slots, int max_seq, int hidden,
    int query_heads, int kv_heads, int head_dim, int rotary_dim,
    int group_size, float theta, float eps);
/* Device-I/O form: x/out are CUDA pointers. */
int coli_cuda_gqa_slots_q4_f16_device(
    ColiCuda *ctx, float *out, const float *x, int batch, const int *slot,
    const int *position, const unsigned char *q_q, const float *q_s,
    int q_rb, int q_ng, const unsigned char *k_q, const float *k_s,
    int k_rb, int k_ng, const unsigned char *v_q, const float *v_s,
    int v_rb, int v_ng, const void *o_q, const float *o_s, int o_fmt,
    int o_rb, int o_ng, const float *q_norm, const float *k_norm,
    float *k_cache, float *v_cache, int slots, int max_seq, int hidden,
    int query_heads, int kv_heads, int head_dim, int rotary_dim,
    int group_size, float theta, float eps);

#ifdef __cplusplus
}
#endif
#endif
