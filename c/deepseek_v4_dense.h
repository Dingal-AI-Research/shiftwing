#ifndef COLIB_DEEPSEEK_V4_DENSE_H
#define COLIB_DEEPSEEK_V4_DENSE_H

/* Before this header pulls in any system header: glibc locks its
 * feature-test macros at the first one it sees, and st.h defining
 * _GNU_SOURCE further down the include chain is then too late -- O_DIRECT
 * stays invisible and every expert read silently falls back to buffered.
 * Guarded so a translation unit that already defined it is untouched. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
/* One-time host-RAM arena for non-routed weights. The 0731 physical layout is
 * 8.238 GiB for the base dense set and 0.554 GiB for optional DSpark dense
 * weights; the 146.625 GiB of routed experts remain in the bounded tier.
 * Payloads are preserved byte-for-byte and 64-byte aligned for CPU/CUDA use. */

#include "deepseek_v4_model.h"
#include <sys/mman.h>
#include <time.h>
#ifdef COLI_CUDA
#include <pthread.h>
#include "backend_cuda.h"
#endif

#define DSV4_BASE_DENSE_RECORDS 1564
#define DSV4_BASE_DENSE_BYTES UINT64_C(8845959388)
#define DSV4_DSPARK_DENSE_RECORDS 97
#define DSV4_DSPARK_DENSE_BYTES UINT64_C(595190812)
#define DSV4_DENSE_ALIGNMENT 64u
#ifndef DSV4_EXPERT_HIDDEN
#define DSV4_EXPERT_HIDDEN DSV4_DIM
#endif
#ifndef DSV4_MOE_INTERMEDIATE
#define DSV4_MOE_INTERMEDIATE 2048
#endif
#ifndef DSV4_BASE_LAYERS
#define DSV4_BASE_LAYERS 43
#endif

typedef struct {
    dsv4_store *store;
    unsigned char *arena;
    size_t arena_bytes;
    size_t *offsets;
    int records;
    int include_dspark;
    uint64_t payload_bytes;
    /* Host bytes returned to the OS once the device holds the arena. */
    uint64_t released_bytes;
#ifdef COLI_CUDA
    ColiCuda *cuda;
    unsigned char *cuda_arena;
    void *cuda_activation;
    void *cuda_activation_scale;
    void *cuda_output;
    size_t cuda_activation_cap;
    size_t cuda_activation_scale_cap;
    size_t cuda_output_cap;
    void *cuda_attention_cache, *cuda_index_cache, *cuda_attention_query, *cuda_attention_out, *cuda_attention_indices, *cuda_attention_aux;
    size_t cuda_attention_cache_cap, cuda_index_cache_cap, cuda_attention_query_cap, cuda_attention_out_cap, cuda_attention_indices_cap, cuda_attention_aux_cap;
    const float *cuda_attention_owner, *cuda_index_owner;
    int cuda_attention_position, cuda_index_position;
    uint64_t cuda_calls;
    uint64_t cuda_upload_bytes;
    uint64_t cuda_download_bytes;
    uint64_t cuda_pinned_upload_bytes;
    uint64_t cuda_pageable_upload_bytes;
    double cuda_fp8_seconds;
    double cuda_bf16_seconds;
    int cuda_lock_initialized;
    pthread_mutex_t cuda_lock;
#endif

    char error[256];
} dsv4_dense_arena;

static inline double dsv4_dense_now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static inline int dsv4_dense_is_expert(const char *name) {
    return name && strstr(name, ".ffn.experts.") != NULL;
}

/* FP8 projection payloads (attention wq/wkv/wo, indexer wq_b, shared experts)
 * are consumed only through the device dispatch once CUDA is enabled; the
 * host address then merely locates the device copy. Their whole pages can be
 * released after upload. This is the conservative pre-load estimate of those
 * bytes (each record's interior less its two edge pages), which the memory
 * plan subtracts from the steady-state dense footprint. */
static inline uint64_t dsv4_dense_fp8_page_bytes(const dsv4_store *store, int include_dspark) {
    uint64_t bytes = 0;
    if (!store || !store->descriptors) return 0;
    for (int index = 0; index < store->raw.n; index++) {
        const st_tensor *tensor = &store->raw.t[index];
        if (dsv4_dense_is_expert(tensor->name) ||
            (!include_dspark && !strncmp(tensor->name, "mtp.", 4)) ||
            store->descriptors[index].dtype != DSV4_DTYPE_FP8_E4M3 ||
            tensor->nbytes <= 2 * 4096)
            continue;
        bytes += (uint64_t)tensor->nbytes - 2 * 4096;
    }
    return bytes;
}

/* Give the FP8 projection pages back to the OS after the device upload and
 * trap host reads of them (PROT_NONE), so a path that wrongly touches host
 * FP8 bytes faults instead of computing on zeros. Only whole pages inside a
 * record are affected; edge pages shared with neighbouring records stay.
 * release=0 restores access before the arena is freed. */
static inline uint64_t dsv4_dense_arena_release_fp8(dsv4_dense_arena *dense,
                                                     int release) {
    uint64_t bytes = 0;
    if (!dense || !dense->arena || !dense->offsets || !dense->store) return 0;
    const dsv4_store *store = dense->store;
    for (int index = 0; index < store->raw.n; index++) {
        if (dense->offsets[index] == SIZE_MAX ||
            store->descriptors[index].dtype != DSV4_DTYPE_FP8_E4M3)
            continue;
        uintptr_t start = (uintptr_t)dense->arena + dense->offsets[index];
        uintptr_t end = start + (uintptr_t)store->raw.t[index].nbytes;
        uintptr_t lo = (start + 4095) & ~(uintptr_t)4095, hi = end & ~(uintptr_t)4095;
        if (hi <= lo) continue;
        if (release) {
            if (madvise((void *)lo, hi - lo, MADV_DONTNEED)) continue;
            (void)mprotect((void *)lo, hi - lo, PROT_NONE);
        } else {
            (void)mprotect((void *)lo, hi - lo, PROT_READ | PROT_WRITE);
        }
        bytes += hi - lo;
    }
    dense->released_bytes = release ? bytes : 0;
    return bytes;
}

static inline int dsv4_dense_is_dspark(const char *name) {
    return name && !strncmp(name, "mtp.", 4);
}

static inline size_t dsv4_dense_align(size_t value) {
    return (value + DSV4_DENSE_ALIGNMENT - 1) &
           ~(size_t)(DSV4_DENSE_ALIGNMENT - 1);
}

static inline int dsv4_dense_arena_init(dsv4_dense_arena *dense,
                                         dsv4_store *store,
                                         int include_dspark, int strict) {
    if (!dense || !store || store->raw.n < 1) return 0;
    memset(dense, 0, sizeof(*dense));
    dense->store = store;
    dense->include_dspark = include_dspark != 0;
    dense->offsets = (size_t *)malloc((size_t)store->raw.n *
                                      sizeof(*dense->offsets));
    if (!dense->offsets) {
        snprintf(dense->error, sizeof(dense->error), "out of memory");
        return 0;
    }
    size_t cursor = 0;
    for (int index = 0; index < store->raw.n; index++) {
        st_tensor *tensor = &store->raw.t[index];
        int selected = !dsv4_dense_is_expert(tensor->name) &&
            (dense->include_dspark || !dsv4_dense_is_dspark(tensor->name));
        dense->offsets[index] = SIZE_MAX;
        if (!selected) continue;
        if (tensor->nbytes < 0 ||
            (uint64_t)tensor->nbytes > SIZE_MAX - cursor) {
            snprintf(dense->error, sizeof(dense->error),
                     "dense arena size overflow at %s", tensor->name);
            free(dense->offsets); dense->offsets = NULL;
            return 0;
        }
        cursor = dsv4_dense_align(cursor);
        dense->offsets[index] = cursor;
        cursor += (size_t)tensor->nbytes;
        dense->payload_bytes += (uint64_t)tensor->nbytes;
        dense->records++;
    }
    dense->arena_bytes = dsv4_dense_align(cursor);
    int expected_records = DSV4_BASE_DENSE_RECORDS +
        (dense->include_dspark ? DSV4_DSPARK_DENSE_RECORDS : 0);
    uint64_t expected_bytes = DSV4_BASE_DENSE_BYTES +
        (dense->include_dspark ? DSV4_DSPARK_DENSE_BYTES : 0);
    if (strict && (dense->records != expected_records ||
                   dense->payload_bytes != expected_bytes)) {
        snprintf(dense->error, sizeof(dense->error),
                 "dense contract expected %d/%llu records/bytes, got %d/%llu",
                 expected_records, (unsigned long long)expected_bytes,
                 dense->records, (unsigned long long)dense->payload_bytes);
        free(dense->offsets); dense->offsets = NULL;
        return 0;
    }
    if (!dense->arena_bytes ||
        posix_memalign((void **)&dense->arena, DSV4_DENSE_ALIGNMENT,
                       dense->arena_bytes) != 0) {
        snprintf(dense->error, sizeof(dense->error),
                 "cannot allocate %zu-byte dense arena", dense->arena_bytes);
        free(dense->offsets); dense->offsets = NULL;
        return 0;
    }
    for (int index = 0; index < store->raw.n; index++)
        if (dense->offsets[index] != SIZE_MAX) {
            unsigned char *data=dense->arena+dense->offsets[index];
            st_read_raw(&store->raw, store->raw.t[index].name, data, 1);
            if (!dsv4_store_verify_record(store,&store->raw.t[index],data)) {
                snprintf(dense->error,sizeof(dense->error),"%.255s",store->error);
                free(dense->arena); free(dense->offsets);
                dense->arena=NULL; dense->offsets=NULL;
                return 0;
            }
        }
    return 1;
}
#ifdef COLI_CUDA
static inline int dsv4_dense_cuda_buffer(
    dsv4_dense_arena *dense, void **buffer, size_t *capacity,
    size_t bytes, const char *label) {
    if (*capacity >= bytes) return 1;
    if (*buffer) coli_cuda_free(dense->cuda, *buffer);
    *buffer = NULL;
    *capacity = 0;
    if (coli_cuda_malloc(dense->cuda, buffer, bytes)) {
        snprintf(dense->error, sizeof(dense->error), "%s: %s", label,
                 coli_cuda_last_error());
        return 0;
    }
    *capacity = bytes;
    return 1;
}

static inline int dsv4_dense_arena_enable_cuda(
    dsv4_dense_arena *dense, ColiCuda *cuda) {
    if (!dense || !dense->arena || !dense->arena_bytes || dense->cuda ||
        !cuda)
        return 0;
    unsigned char *device_arena = NULL;
    int upload_error =
        coli_cuda_malloc(cuda, (void **)&device_arena, dense->arena_bytes);
    void *staging = NULL;
    size_t staging_bytes = 64u * 1024u * 1024u;
    const char *staging_mb = getenv("DSV4_DENSE_PINNED_MB");
    if (staging_mb && *staging_mb) {
        long requested = strtol(staging_mb, NULL, 10);
        if (requested >= 16 && requested <= 512)
            staging_bytes = (size_t)requested * 1024u * 1024u;
    }
    const char *pinned_env = getenv("DSV4_PINNED_UPLOAD");
    int use_pinned = !pinned_env || atoi(pinned_env) != 0;
    int pinned = use_pinned && !upload_error &&
        !coli_cuda_malloc_host(cuda, &staging, staging_bytes);
    if (!upload_error && pinned) {
        for (size_t offset = 0; offset < dense->arena_bytes;) {
            size_t bytes = dense->arena_bytes - offset;
            if (bytes > staging_bytes) bytes = staging_bytes;
            memcpy(staging, dense->arena + offset, bytes);
            if (coli_cuda_upload(
                    cuda, device_arena + offset, staging, bytes) ||
                coli_cuda_sync(cuda)) {
                upload_error = 1;
                break;
            }
            offset += bytes;
        }
    } else if (!upload_error) {
        upload_error =
            coli_cuda_upload(cuda, device_arena, dense->arena,
                             dense->arena_bytes) ||
            coli_cuda_sync(cuda);
    }
    if (staging) coli_cuda_free_host(cuda, staging);
    if (upload_error) {
        if (device_arena) coli_cuda_free(cuda, device_arena);
        coli_cuda_sync(cuda);
        snprintf(dense->error, sizeof(dense->error),
                 "CUDA dense arena upload: %s", coli_cuda_last_error());
        return 0;
    }
    if (pthread_mutex_init(&dense->cuda_lock, NULL)) {
        coli_cuda_free(cuda, device_arena);
        coli_cuda_sync(cuda);
        snprintf(dense->error, sizeof(dense->error),
                 "cannot initialize CUDA dense-arena lock");
        return 0;
    }
    dense->cuda = cuda;
    dense->cuda_arena = device_arena;
    dense->cuda_lock_initialized = 1;
    dense->cuda_upload_bytes = dense->arena_bytes;
    dense->cuda_pinned_upload_bytes =
        pinned ? dense->arena_bytes : 0;
    dense->cuda_pageable_upload_bytes =
        pinned ? 0 : dense->arena_bytes;
    const char *release = getenv("DSV4_DENSE_RELEASE");
    if (!release || !*release || atoi(release) != 0)
        dsv4_dense_arena_release_fp8(dense, 1);
    return 1;
}

static inline const unsigned char *dsv4_dense_cuda_pointer(
    const dsv4_dense_arena *dense, const void *host, size_t bytes) {
    uintptr_t begin = (uintptr_t)dense->arena;
    uintptr_t value = (uintptr_t)host;
    if (value < begin || value - begin > dense->arena_bytes ||
        bytes > dense->arena_bytes - (value - begin))
        return NULL;
    return dense->cuda_arena + (value - begin);
}
#endif

static inline int dsv4_dense_linear_fp8(
    const dsv4_dense_arena *view, float *output, const float *input,
    const uint8_t *weight, const uint8_t *weight_scale,
    int batch, int rows, int columns, uint8_t *activation,
    uint8_t *activation_scale) {
    if (!view || !output || !input || !weight || !weight_scale ||
        !activation || !activation_scale || batch < 1 || rows < 1 ||
        columns < 1 || columns % 128)
        return 0;
#ifndef COLI_CUDA
    return dsv4_linear_fp8(
        output, input, weight, weight_scale, batch, rows, columns,
        activation, activation_scale);
#else
    if (!view->cuda)
        return dsv4_linear_fp8(
            output, input, weight, weight_scale, batch, rows, columns,
            activation, activation_scale);
    double cuda_start = dsv4_dense_now_seconds();
    if ((size_t)batch > SIZE_MAX / (size_t)columns ||
        (size_t)rows > SIZE_MAX / (size_t)columns ||
        (size_t)batch > SIZE_MAX / (size_t)rows)
        return 0;
    size_t activation_bytes = (size_t)batch * columns;
    size_t activation_scale_bytes =
        (size_t)batch * (size_t)(columns / 128);
    size_t output_bytes = (size_t)batch * rows * sizeof(float);
    size_t weight_bytes = (size_t)rows * columns;
    size_t weight_scale_bytes =
        (size_t)((rows + 127) / 128) * (size_t)(columns / 128);
    const unsigned char *device_weight =
        dsv4_dense_cuda_pointer(view, weight, weight_bytes);
    const unsigned char *device_scale =
        dsv4_dense_cuda_pointer(view, weight_scale, weight_scale_bytes);
    if (!device_weight || !device_scale) return 0;
    if (!dsv4_act_quant_mxfp_batch(input,batch,columns,activation,activation_scale)) return 0;
    dsv4_dense_arena *dense = (dsv4_dense_arena *)view;
    pthread_mutex_lock(&dense->cuda_lock);
    dense->error[0] = '\0';
    int ok = dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_activation,
                 &dense->cuda_activation_cap, activation_bytes,
                 "CUDA dense activation allocation") &&
             dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_activation_scale,
                 &dense->cuda_activation_scale_cap,
                 activation_scale_bytes,
                 "CUDA dense scale allocation") &&
             dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_output, &dense->cuda_output_cap,
                 output_bytes, "CUDA dense output allocation");
    if (ok)
        ok = !coli_cuda_upload(
                 dense->cuda, dense->cuda_activation,
                 activation, activation_bytes) &&
             !coli_cuda_upload(
                 dense->cuda, dense->cuda_activation_scale,
                 activation_scale, activation_scale_bytes) &&
             !coli_cuda_dsv4_fp8_gemm(
                 dense->cuda, (float *)dense->cuda_output,
                 (const unsigned char *)dense->cuda_activation,
                 (const unsigned char *)dense->cuda_activation_scale,
                 device_weight, device_scale, batch, rows, columns) &&
             !coli_cuda_download(
                 dense->cuda, output, dense->cuda_output, output_bytes) &&
             !coli_cuda_sync(dense->cuda);
    if (ok) {
        dense->cuda_calls++;
        dense->cuda_upload_bytes +=
            activation_bytes + activation_scale_bytes;
        dense->cuda_download_bytes += output_bytes;
    } else if (!dense->error[0]) {
        snprintf(dense->error, sizeof(dense->error),
                 "CUDA dense FP8 projection: %s",
                 coli_cuda_last_error());
    }
    dense->cuda_fp8_seconds += dsv4_dense_now_seconds() - cuda_start;
    pthread_mutex_unlock(&dense->cuda_lock);
    if (ok)
        dsv4_round_bf16_array(output, (size_t)batch * rows);
    return ok;
#endif
}
static inline int dsv4_dense_fp8_weight_bf16(const dsv4_dense_arena *view,float *output,const float *input,
    const uint8_t *weight,const uint8_t *scale,int batch,int rows,int cols) {
    if (!view || !output || !input || !weight || !scale || batch<1 || rows<1 || cols<1 || cols%128) return 0;
#ifdef COLI_CUDA
    if (view->cuda) {
        dsv4_dense_arena *d=(dsv4_dense_arena*)view;
        const unsigned char *w=dsv4_dense_cuda_pointer(d,weight,(size_t)rows*cols);
        const unsigned char *ws=dsv4_dense_cuda_pointer(d,scale,(size_t)((rows+127)/128)*(cols/128));
        if (!w || !ws) return 0;
        pthread_mutex_lock(&d->cuda_lock);double start=dsv4_dense_now_seconds();
        size_t in=(size_t)batch*cols*sizeof(float),out=(size_t)batch*rows*sizeof(float);
        int ok=dsv4_dense_cuda_buffer(d,&d->cuda_activation,&d->cuda_activation_cap,in,"wo_a input") &&
            dsv4_dense_cuda_buffer(d,&d->cuda_output,&d->cuda_output_cap,out,"wo_a output") &&
            !coli_cuda_upload(d->cuda,d->cuda_activation,input,in) &&
            !coli_cuda_dsv4_fp8_weight_bf16_gemm(d->cuda,d->cuda_output,d->cuda_activation,w,ws,batch,rows,cols) &&
            !coli_cuda_download(d->cuda,output,d->cuda_output,out) && !coli_cuda_sync(d->cuda);
        d->cuda_bf16_seconds+=dsv4_dense_now_seconds()-start;
        if (ok) {d->cuda_calls++;d->cuda_upload_bytes+=in;d->cuda_download_bytes+=out;}
        pthread_mutex_unlock(&d->cuda_lock);return ok;
    }
#endif
    for (int b=0;b<batch;b++) for (int r=0;r<rows;r++) {
        float sum=0;
        for (int k=0;k<cols;k++) {
            float w=dsv4_round_bf16(dsv4_fp8_e4m3fn(weight[(size_t)r*cols+k])*dsv4_ue8m0(scale[(size_t)(r/128)*(cols/128)+k/128]));
            sum+=dsv4_round_bf16(input[(size_t)b*cols+k])*w;
        }
        output[(size_t)b*rows+r]=dsv4_round_bf16(sum);
    }
    return 1;
}

static inline int dsv4_dense_linear_bf16(
    const dsv4_dense_arena *view, float *output, const float *input,
    const uint16_t *weight, int batch, int rows, int columns) {
    if (!view || !output || !input || !weight || batch < 1 ||
        rows < 1 || columns < 1)
        return 0;
#ifndef COLI_CUDA
    dsv4_bf16_gemm(output, input, weight, batch, rows, columns);
    return 1;
#else
    if (!view->cuda) {
        dsv4_bf16_gemm(output, input, weight, batch, rows, columns);
        return 1;
    }
    double cuda_start = dsv4_dense_now_seconds();
    if ((size_t)rows > SIZE_MAX / (size_t)columns ||
        (size_t)batch > SIZE_MAX / (size_t)columns ||
        (size_t)batch > SIZE_MAX /
            ((size_t)rows * sizeof(float)))
        return 0;
    size_t input_bytes =
        (size_t)batch * columns * sizeof(float);
    size_t output_bytes =
        (size_t)batch * rows * sizeof(float);
    size_t weight_bytes =
        (size_t)rows * columns * sizeof(uint16_t);
    const unsigned char *device_weight =
        dsv4_dense_cuda_pointer(view, weight, weight_bytes);
    if (!device_weight) return 0;
    dsv4_dense_arena *dense = (dsv4_dense_arena *)view;
    pthread_mutex_lock(&dense->cuda_lock);
    dense->error[0] = '\0';
    int ok = dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_activation,
                 &dense->cuda_activation_cap, input_bytes,
                 "CUDA dense BF16 input allocation") &&
             dsv4_dense_cuda_buffer(
                 dense, &dense->cuda_output,
                 &dense->cuda_output_cap, output_bytes,
                 "CUDA dense BF16 output allocation");
    if (ok)
        ok = !coli_cuda_upload(
                 dense->cuda, dense->cuda_activation,
                 input, input_bytes) &&
             !coli_cuda_dsv4_bf16_gemm(
                 dense->cuda, (float *)dense->cuda_output,
                 (const float *)dense->cuda_activation,
                 (const unsigned short *)device_weight,
                 batch, rows, columns) &&
             !coli_cuda_download(
                 dense->cuda, output, dense->cuda_output,
                 output_bytes) &&
             !coli_cuda_sync(dense->cuda);
    if (ok) {
        dense->cuda_calls++;
        dense->cuda_upload_bytes += input_bytes;
        dense->cuda_download_bytes += output_bytes;
    } else if (!dense->error[0]) {
        snprintf(dense->error, sizeof(dense->error),
                 "CUDA dense BF16 projection: %s",
                 coli_cuda_last_error());
    }
    dense->cuda_bf16_seconds += dsv4_dense_now_seconds() - cuda_start;
    pthread_mutex_unlock(&dense->cuda_lock);
    return ok;
#endif
}



static inline const void *dsv4_dense_find(const dsv4_dense_arena *dense,
                                           const char *name,
                                           const dsv4_tensor_desc **descriptor) {
    if (!dense || !dense->arena || !name) return NULL;
    st_tensor *tensor = st_find(&dense->store->raw, name);
    if (!tensor) return NULL;
    ptrdiff_t index = tensor - dense->store->raw.t;
    if (index < 0 || index >= dense->store->raw.n ||
        dense->offsets[index] == SIZE_MAX)
        return NULL;
    if (descriptor) *descriptor = &dense->store->descriptors[index];
    return dense->arena + dense->offsets[index];
}

static inline int dsv4_dense_layer_prefix(char *output, size_t capacity,
                                           int layer) {
    if (!output || !capacity || layer < 0) return 0;
    int length = layer < DSV4_BASE_LAYERS
        ? snprintf(output, capacity, "layers.%d", layer)
        : snprintf(output, capacity, "mtp.%d", layer - DSV4_BASE_LAYERS);
    return length >= 0 && (size_t)length < capacity;
}

static inline const void *dsv4_dense_named(
    const dsv4_dense_arena *dense, const char *prefix, const char *suffix,
    dsv4_dtype dtype, int64_t rows, int64_t columns) {
    char name[192];
    int length = snprintf(name, sizeof(name), "%s%s", prefix, suffix);
    if (length < 0 || length >= (int)sizeof(name)) return NULL;
    const dsv4_tensor_desc *descriptor = NULL;
    const void *value = dsv4_dense_find(dense, name, &descriptor);
    return value && dsv4_model_tensor_shape(descriptor, dtype, 2,
                                             rows, columns) ? value : NULL;
}

static inline int dsv4_dense_embed_token(
    const dsv4_dense_arena *dense, int token, float *hc_output) {
    if (!dense || !hc_output || token < 0 || token >= DSV4_VOCAB)
        return 0;
    const uint16_t *embedding = (const uint16_t *)dsv4_dense_named(
        dense, "embed", ".weight", DSV4_DTYPE_BF16,
        DSV4_VOCAB, DSV4_DIM);
    if (!embedding) return 0;
    const uint16_t *row = embedding + (size_t)token * DSV4_DIM;
    for (int copy = 0; copy < DSV4_HC_MULT; copy++)
        for (int index = 0; index < DSV4_DIM; index++)
            hc_output[(size_t)copy * DSV4_DIM + index] =
                dsv4_bf16(row[index]);
    return 1;
}

static inline int dsv4_dense_fp8_pair(
    const dsv4_dense_arena *dense, const char *prefix,
    int64_t rows, int64_t columns, const uint8_t **weight,
    const uint8_t **scale) {
    *weight = (const uint8_t *)dsv4_dense_named(
        dense, prefix, ".weight", DSV4_DTYPE_FP8_E4M3, rows, columns);
    *scale = (const uint8_t *)dsv4_dense_named(
        dense, prefix, ".scale", DSV4_DTYPE_UE8M0,
        (rows + 127) / 128, (columns + 127) / 128);
    return *weight && *scale;
}

static inline int dsv4_dense_route_logits(const dsv4_dense_arena *dense,int layer,int token,
    const char *gate_prefix,const float *logits,int *indices,float *weights) {
    int hash_indices[DSV4_TOPK];
    const int *hash = NULL;
    const float *bias = NULL;
    if (layer < 3) {
        const int64_t *table = (const int64_t *)dsv4_dense_named(
            dense, gate_prefix, ".tid2eid", DSV4_DTYPE_I64,
            DSV4_VOCAB, DSV4_TOPK);
        if (!table) return 0;
        for (int route = 0; route < DSV4_TOPK; route++) {
            int64_t expert = table[(size_t)token * DSV4_TOPK + route];
            if (expert < 0 || expert >= DSV4_EXPERTS) return 0;
            hash_indices[route] = (int)expert;
        }
        hash = hash_indices;
    } else {
        char name[128];
        if (snprintf(name, sizeof(name), "%s.bias", gate_prefix) >=
            (int)sizeof(name)) return 0;
        const dsv4_tensor_desc *descriptor = NULL;
        bias = (const float *)dsv4_dense_find(dense, name, &descriptor);
        if (!bias || !descriptor || descriptor->dtype != DSV4_DTYPE_F32 ||
            descriptor->rank != 1 || descriptor->shape[0] != DSV4_EXPERTS)
            return 0;
    }
    return dsv4_route_topk(logits, bias, DSV4_EXPERTS, hash,
                            DSV4_TOPK, indices, weights);
}
static inline int dsv4_dense_route(
    const dsv4_dense_arena *dense, int layer, int token, const float *input,
    int *indices, float *weights, float *logits) {
    if (!dense || !input || !indices || !weights || !logits || token < 0 ||
        token >= DSV4_VOCAB)
        return 0;
    char layer_prefix[64], gate_prefix[96];
    if (!dsv4_dense_layer_prefix(layer_prefix, sizeof(layer_prefix), layer) ||
        snprintf(gate_prefix, sizeof(gate_prefix), "%s.ffn.gate",
                 layer_prefix) >= (int)sizeof(gate_prefix))
        return 0;
    const uint16_t *gate = (const uint16_t *)dsv4_dense_named(
        dense, gate_prefix, ".weight", DSV4_DTYPE_BF16,
        DSV4_EXPERTS, DSV4_EXPERT_HIDDEN);
    if (!gate) return 0;
    dsv4_bf16_gemm(logits, input, gate, 1, DSV4_EXPERTS,
                    DSV4_EXPERT_HIDDEN);
    return dsv4_dense_route_logits(dense,layer,token,gate_prefix,logits,indices,weights);
}

static inline int dsv4_dense_route_batch(const dsv4_dense_arena *dense,int layer,const int *tokens,
    const float *input,int count,int *indices,float *weights,float *logits) {
    if (!dense || !tokens || !input || !indices || !weights || !logits || count<1 || count>2048) return 0;
    char lp[64],prefix[96];
    if (!dsv4_dense_layer_prefix(lp,sizeof(lp),layer) || snprintf(prefix,sizeof(prefix),"%s.ffn.gate",lp)>=(int)sizeof(prefix)) return 0;
    const uint16_t *gate=(const uint16_t*)dsv4_dense_named(dense,prefix,".weight",DSV4_DTYPE_BF16,DSV4_EXPERTS,DSV4_EXPERT_HIDDEN);
    if (!gate || !dsv4_dense_linear_bf16(dense,logits,input,gate,count,DSV4_EXPERTS,DSV4_EXPERT_HIDDEN)) return 0;
    for (int b=0;b<count;b++)
        if (tokens[b]<0 || tokens[b]>=DSV4_VOCAB || !dsv4_dense_route_logits(dense,layer,tokens[b],prefix,
            logits+(size_t)b*DSV4_EXPERTS,indices+(size_t)b*DSV4_TOPK,weights+(size_t)b*DSV4_TOPK)) return 0;
    return 1;
}

static inline int dsv4_dense_shared_expert_batch(
    const dsv4_dense_arena *dense, int layer, const float *input,
    float *output, int batch, float *gate, float *up, uint8_t *activation,
    uint8_t *activation_scale) {
    char layer_prefix[64], prefix[128];
    if (!dsv4_dense_layer_prefix(layer_prefix, sizeof(layer_prefix), layer))
        return 0;
    const char *projection[3] = {"w1", "w2", "w3"};
    const uint8_t *weight[3], *scale[3];
    const int64_t rows[3] = {DSV4_MOE_INTERMEDIATE,
                             DSV4_EXPERT_HIDDEN,
                             DSV4_MOE_INTERMEDIATE};
    const int64_t columns[3] = {DSV4_EXPERT_HIDDEN,
                                DSV4_MOE_INTERMEDIATE,
                                DSV4_EXPERT_HIDDEN};
    for (int index = 0; index < 3; index++) {
        int length = snprintf(prefix, sizeof(prefix),
                              "%s.ffn.shared_experts.%s", layer_prefix,
                              projection[index]);
        if (length < 0 || length >= (int)sizeof(prefix) ||
            !dsv4_dense_fp8_pair(dense, prefix, rows[index], columns[index],
                                  &weight[index], &scale[index]))
            return 0;
    }
    if (!dsv4_dense_linear_fp8(
            dense, gate, input, weight[0], scale[0], batch,
            DSV4_MOE_INTERMEDIATE, DSV4_EXPERT_HIDDEN,
            activation, activation_scale) ||
        !dsv4_dense_linear_fp8(
            dense, up, input, weight[2], scale[2], batch,
            DSV4_MOE_INTERMEDIATE, DSV4_EXPERT_HIDDEN,
            activation, activation_scale))
        return 0;
    for (int index = 0; index < batch * DSV4_MOE_INTERMEDIATE; index++)
        gate[index] = dsv4_round_bf16(
            dsv4_clamped_swiglu(gate[index], up[index]));
    return dsv4_dense_linear_fp8(
        dense, output, gate, weight[1], scale[1], batch,
        DSV4_EXPERT_HIDDEN, DSV4_MOE_INTERMEDIATE,
        activation, activation_scale);
}

static inline int dsv4_dense_shared_expert(
    const dsv4_dense_arena *dense, int layer, const float *input,
    float *output, float *gate, float *up, uint8_t *activation,
    uint8_t *activation_scale) {
    return dsv4_dense_shared_expert_batch(dense, layer, input, output, 1,
                                        gate, up, activation, activation_scale);
}

static inline void dsv4_dense_arena_close(dsv4_dense_arena *dense) {
    if (!dense) return;
#ifdef COLI_CUDA
    if (dense->cuda) {
        if (dense->cuda_lock_initialized)
            pthread_mutex_lock(&dense->cuda_lock);
        if (dense->cuda_arena)
            coli_cuda_free(dense->cuda, dense->cuda_arena);
        if (dense->cuda_activation)
            coli_cuda_free(dense->cuda, dense->cuda_activation);
        if (dense->cuda_activation_scale)
            coli_cuda_free(dense->cuda, dense->cuda_activation_scale);
        if (dense->cuda_output)
            coli_cuda_free(dense->cuda, dense->cuda_output);
        coli_cuda_free(dense->cuda,dense->cuda_attention_cache);coli_cuda_free(dense->cuda,dense->cuda_index_cache);
        coli_cuda_free(dense->cuda,dense->cuda_attention_query);coli_cuda_free(dense->cuda,dense->cuda_attention_out);
        coli_cuda_free(dense->cuda,dense->cuda_attention_indices);coli_cuda_free(dense->cuda,dense->cuda_attention_aux);
        coli_cuda_sync(dense->cuda);
        if (dense->cuda_lock_initialized) {
            pthread_mutex_unlock(&dense->cuda_lock);
            pthread_mutex_destroy(&dense->cuda_lock);
        }
    }
#endif

    if (dense->released_bytes) dsv4_dense_arena_release_fp8(dense, 0);
    free(dense->arena); free(dense->offsets);
    memset(dense, 0, sizeof(*dense));
}

#endif
