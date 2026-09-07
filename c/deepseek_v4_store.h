#ifndef COLIB_DEEPSEEK_V4_STORE_H
#define COLIB_DEEPSEEK_V4_STORE_H

/* Adapt the lossless DeepSeek model-manifest to the established raw shard
 * reader. This preserves the direct-I/O, persistent io_uring, counters, and
 * batch-read behavior already qualified by the Qwen/Ornith engine. */

#include <limits.h>
#include <stddef.h>
#include <sys/stat.h>

#include "st.h"

#define DSV4_MANIFEST_SCHEMA "colib.deepseek-v4.model-manifest.v1"
#define DSV4_PINNED_REVISION "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
#define DSV4_MAX_DIMS 8

typedef enum {
    DSV4_DTYPE_INVALID = 0,
    DSV4_DTYPE_BF16,
    DSV4_DTYPE_F32,
    DSV4_DTYPE_FP8_E4M3,
    DSV4_DTYPE_UE8M0,
    DSV4_DTYPE_I8,
    DSV4_DTYPE_I64,
} dsv4_dtype;

typedef struct {
    dsv4_dtype dtype;
    int rank;
    int64_t shape[DSV4_MAX_DIMS];
    int layer;
    int expert;
    char projection[16];
} dsv4_tensor_desc;

typedef struct {
    shards raw;
    dsv4_tensor_desc *descriptors;
    int records;
    int segments;
    char error[512];
} dsv4_store;

static char *dsv4_store_read_file(const char *path, size_t *size_out) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) { if (file) fclose(file); return NULL; }
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET)) { fclose(file); return NULL; }
    char *text = calloc((size_t)size + 1, 1);
    if (!text || fread(text, 1, (size_t)size, file) != (size_t)size) {
        free(text); fclose(file); return NULL;
    }
    fclose(file);
    if (size_out) *size_out = (size_t)size;
    return text;
}

static int dsv4_safe_relative(const char *name) {
    if (!name || !*name || name[0] == '/' || name[0] == '\\') return 0;
    if (strstr(name, "..") || strchr(name, '\\')) return 0;
    return 1;
}

static dsv4_dtype dsv4_store_dtype(const char *name) {
    if (!name) return DSV4_DTYPE_INVALID;
    if (!strcmp(name, "BF16")) return DSV4_DTYPE_BF16;
    if (!strcmp(name, "F32")) return DSV4_DTYPE_F32;
    if (!strcmp(name, "F8_E4M3")) return DSV4_DTYPE_FP8_E4M3;
    if (!strcmp(name, "F8_E8M0")) return DSV4_DTYPE_UE8M0;
    if (!strcmp(name, "I8")) return DSV4_DTYPE_I8;
    if (!strcmp(name, "I64")) return DSV4_DTYPE_I64;
    return DSV4_DTYPE_INVALID;
}

static int dsv4_store_dtype_bytes(dsv4_dtype dtype) {
    switch (dtype) {
        case DSV4_DTYPE_BF16: return 2;
        case DSV4_DTYPE_F32: return 4;
        case DSV4_DTYPE_FP8_E4M3:
        case DSV4_DTYPE_UE8M0:
        case DSV4_DTYPE_I8: return 1;
        case DSV4_DTYPE_I64: return 8;
        default: return 0;
    }
}

static const dsv4_tensor_desc *dsv4_store_find(
    dsv4_store *store, const char *name) {
    if (!store || !name || !store->descriptors) return NULL;
    st_tensor *tensor = st_find(&store->raw, name);
    if (!tensor) return NULL;
    ptrdiff_t index = tensor - store->raw.t;
    if (index < 0 || index >= store->raw.n) return NULL;
    return &store->descriptors[index];
}

static int dsv4_store_prepare_hash(shards *store, int records) {
    store->hcap = 1;
    while (store->hcap < records * 2) store->hcap <<= 1;
    store->hidx = malloc((size_t)store->hcap * sizeof(int));
    if (!store->hidx) return 0;
    for (int index = 0; index < store->hcap; index++)
        store->hidx[index] = -1;
    return 1;
}

static void dsv4_store_hash_insert(shards *store, int index) {
    uint64_t hash = st_hash(store->t[index].name) & (store->hcap - 1);
    while (store->hidx[hash] >= 0)
        hash = (hash + 1) & (store->hcap - 1);
    store->hidx[hash] = index;
}

static int dsv4_store_init(dsv4_store *store, const char *snapshot) {
    if (!store || !snapshot) return 0;
    memset(store, 0, sizeof(*store));
    char manifest_path[PATH_MAX];
    if (snprintf(manifest_path, sizeof(manifest_path), "%s/model-manifest.json",
                 snapshot) >= (int)sizeof(manifest_path)) {
        snprintf(store->error, sizeof(store->error), "manifest path is too long");
        return 0;
    }
    char *text = dsv4_store_read_file(manifest_path, NULL);
    if (!text) {
        snprintf(store->error, sizeof(store->error), "cannot read model-manifest.json");
        return 0;
    }
    jval *root = json_parse(text, NULL);
    jval *schema = json_get(root, "schema"), *source = json_get(root, "source");
    jval *revision = json_get(source, "revision"), *inventory = json_get(root, "inventory");
    if (!root || root->t != J_OBJ || !schema || schema->t != J_STR ||
        strcmp(schema->str, DSV4_MANIFEST_SCHEMA) || !revision ||
        revision->t != J_STR || strcmp(revision->str, DSV4_PINNED_REVISION) ||
        !inventory || inventory->t != J_OBJ) {
        snprintf(store->error, sizeof(store->error), "manifest identity/schema mismatch");
        free(text); return 0;
    }
    int count = 0;
    for (int group = 0; group < inventory->len; group++) {
        jval *records = inventory->kids[group];
        if (!records || records->t != J_ARR) {
            snprintf(store->error, sizeof(store->error), "inventory group is not an array");
            free(text); return 0;
        }
        count += records->len;
    }
    store->raw.cap = count > 0 ? count : 1;
    store->raw.t = calloc((size_t)store->raw.cap, sizeof(st_tensor));
    store->descriptors = calloc((size_t)store->raw.cap, sizeof(dsv4_tensor_desc));
    if (!store->raw.t || !store->descriptors ||
        !dsv4_store_prepare_hash(&store->raw, count)) {
        snprintf(store->error, sizeof(store->error), "out of memory");
        free(text); return 0;
    }
    for (int group = 0; group < inventory->len; group++) {
        jval *records = inventory->kids[group];
        for (int index = 0; index < records->len; index++) {
            jval *record = records->kids[index];
            jval *name = json_get(record, "name"), *file = json_get(record, "file");
            jval *offset = json_get(record, "offset"), *nbytes = json_get(record, "nbytes");
            jval *shape = json_get(record, "shape"), *dtype = json_get(record, "dtype");
            jval *layer = json_get(record, "layer"), *expert = json_get(record, "expert");
            jval *projection = json_get(record, "projection");
            dsv4_dtype storage_dtype = dtype && dtype->t == J_STR
                ? dsv4_store_dtype(dtype->str) : DSV4_DTYPE_INVALID;
            if (!name || name->t != J_STR || !file || file->t != J_STR ||
                !dsv4_safe_relative(file->str) || !offset || offset->t != J_NUM ||
                !nbytes || nbytes->t != J_NUM || offset->num < 0 || nbytes->num < 0 ||
                !shape || shape->t != J_ARR || shape->len > DSV4_MAX_DIMS ||
                !dtype || dtype->t != J_STR || storage_dtype == DSV4_DTYPE_INVALID) {
                snprintf(store->error, sizeof(store->error), "malformed inventory record %d", index);
                free(text); return 0;
            }
            char path[PATH_MAX];
            if (snprintf(path, sizeof(path), "%s/%s", snapshot, file->str) >= (int)sizeof(path)) {
                snprintf(store->error, sizeof(store->error), "segment path is too long");
                free(text); return 0;
            }
            int fd = st_open_fd(&store->raw, path);
            struct stat status;
            int64_t off = (int64_t)offset->num, bytes = (int64_t)nbytes->num;
            if (fstat(fd, &status) || off > status.st_size || bytes > status.st_size - off) {
                snprintf(store->error, sizeof(store->error), "record %s exceeds segment", name->str);
                free(text); return 0;
            }
            int64_t numel = 1;
            dsv4_tensor_desc descriptor = {
                .dtype = storage_dtype, .rank = shape->len, .layer = -1, .expert = -1
            };
            for (int axis = 0; axis < shape->len; axis++) {
                if (!shape->kids[axis] || shape->kids[axis]->t != J_NUM ||
                    shape->kids[axis]->num < 0 ||
                    shape->kids[axis]->num > (double)INT64_MAX) {
                    snprintf(store->error, sizeof(store->error), "invalid shape for %s", name->str);
                    free(text); return 0;
                }
                int64_t dimension = (int64_t)shape->kids[axis]->num;
                if ((double)dimension != shape->kids[axis]->num ||
                    (dimension && numel > INT64_MAX / dimension)) {
                    snprintf(store->error, sizeof(store->error), "shape overflow for %s", name->str);
                    free(text); return 0;
                }
                descriptor.shape[axis] = dimension;
                numel *= dimension;
            }
            int element_bytes = dsv4_store_dtype_bytes(storage_dtype);
            if (numel > INT64_MAX / element_bytes || numel * element_bytes != bytes) {
                snprintf(store->error, sizeof(store->error), "dtype/shape byte mismatch for %s", name->str);
                free(text); return 0;
            }
            if (st_has(&store->raw, name->str)) {
                snprintf(store->error, sizeof(store->error), "duplicate tensor %s", name->str);
                free(text); return 0;
            }
            if (layer && layer->t == J_NUM) descriptor.layer = (int)layer->num;
            if (expert && expert->t == J_NUM) descriptor.expert = (int)expert->num;
            if (projection && projection->t == J_STR)
                snprintf(descriptor.projection, sizeof(descriptor.projection), "%s", projection->str);
            int tensor_index = store->raw.n++;
            st_tensor *tensor = &store->raw.t[tensor_index];
            tensor->name = strdup(name->str); tensor->fd = fd; tensor->off = off;
            tensor->nbytes = bytes; tensor->dtype = 3; tensor->numel = numel;
            if (!tensor->name) {
                snprintf(store->error, sizeof(store->error), "out of memory");
                free(text); return 0;
            }
            store->descriptors[tensor_index] = descriptor;
            dsv4_store_hash_insert(&store->raw, tensor_index);
        }
    }
    store->records = store->raw.n;
    store->segments = store->raw.nfd;
    free(text);
    return 1;
}

static int dsv4_store_read_slice(dsv4_store *store, const char *name,
                                  int64_t offset, int64_t bytes, void *output,
                                  int direct) {
    if (!store || !name || !output || offset < 0 || bytes < 0) return 0;
    st_tensor *tensor = st_find(&store->raw, name);
    if (!tensor || offset > tensor->nbytes || bytes > tensor->nbytes - offset)
        return 0;
    if (!bytes) return 1;
    st_tensor slice = *tensor;
    slice.off += offset;
    slice.nbytes = bytes;
    if (direct && st_pread_direct_try(&store->raw, &slice, output) == 0)
        return 1;
    st_pread_full(slice.fd, output, bytes, slice.off, name);
    __atomic_fetch_add(&store->raw.read_bytes, (uint64_t)bytes,
                       __ATOMIC_RELAXED);
#if defined(POSIX_FADV_DONTNEED)
    posix_fadvise(slice.fd, slice.off, bytes, POSIX_FADV_DONTNEED);
#endif
    return 1;
}

static void dsv4_store_close(dsv4_store *store) {
    if (!store) return;
    for (int i = 0; i < store->raw.n; i++) free(store->raw.t[i].name);
    for (int i = 0; i < store->raw.nfd; i++) {
        if (store->raw.fds[i] >= 0) close(store->raw.fds[i]);
        if (store->raw.dfds[i] >= 0) close(store->raw.dfds[i]);
        free(store->raw.paths[i]);
    }
    free(store->raw.t); free(store->raw.hidx); free(store->descriptors);
    memset(store, 0, sizeof(*store));
}

#endif
