#ifndef SHIFTWING_GLM53_CUDA_H
#define SHIFTWING_GLM53_CUDA_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct GlmCuda GlmCuda;
/* fmt: 0=f32, 1=row-scaled i8, 4=group-scaled i4; low nibble first,
 * value=nibble-8. Cache handles belong only to immutable resident weights.
 * Streamed expert slots MUST pass cache=NULL: their host addresses are reused. */
typedef struct {
    int fmt, rows, cols, group;
    const void *weights;
    const float *scales;
    void **cache;
} GlmCudaMatrix;
typedef struct {
    unsigned long long calls, batched_calls, mlp_calls, uploaded_bytes, fallbacks;
    size_t budget_bytes, allocated_bytes, peak_bytes, resident_bytes;
    int max_batch;
} GlmCudaStats;
GlmCuda *glm_cuda_create(int device, size_t budget_bytes);
void glm_cuda_destroy(GlmCuda *ctx);
const char *glm_cuda_error(void);
void glm_cuda_stats(const GlmCuda *ctx, GlmCudaStats *out);
/* Return 1 only when the whole host output is ready, otherwise recompute on
 * CPU. x=[batch,cols], y=[batch,row_count]. Row ranges share a resident cache. */
int glm_cuda_mm(GlmCuda *, float *y, const float *x, int batch,
                const GlmCudaMatrix *, int first_row, int row_count);
/* Gate/up/down stay on device through clamped SwiGLU. Transient matrices are
 * uploaded on EVERY call, independent of their host addresses. */
int glm_cuda_mlp(GlmCuda *, float *y, const float *x, int batch,
                 const GlmCudaMatrix *gate, const GlmCudaMatrix *up,
                 const GlmCudaMatrix *down, float limit);
#ifdef __cplusplus
}
#endif
#endif
