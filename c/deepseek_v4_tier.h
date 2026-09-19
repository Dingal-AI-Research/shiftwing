#ifndef COLIB_DEEPSEEK_V4_TIER_H
#define COLIB_DEEPSEEK_V4_TIER_H

/* Bounded host tier for converter-native routed experts. Each cache entry owns
 * one contiguous w1/s1/w2/s2/w3/s3 buffer, while the six manifest records are
 * fetched together through the established direct-I/O/io_uring batch reader.
 * Frequency survives eviction; recency breaks ties. Active references cannot
 * be evicted, which makes the API safe for concurrent scheduler rows. */

#include <pthread.h>
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

#include "deepseek_v4_dense.h"
#include "tier.h"

#ifndef DSV4_EXPERT_HIDDEN
#define DSV4_EXPERT_HIDDEN DSV4_DIM
#endif
#ifndef DSV4_MOE_INTERMEDIATE
#define DSV4_MOE_INTERMEDIATE 2048
#endif
#ifndef DSV4_BASE_LAYERS
#define DSV4_BASE_LAYERS 43
#endif
#ifndef DSV4_DSPARK_LAYERS
#define DSV4_DSPARK_LAYERS 3
#endif

typedef struct {
    int valid;
    int layer;
    int expert;
    int references;
    /* Storage page-locked for direct upload DMA (see dsv4_expert_pin). */
    int pinned;
    unsigned char *storage;
    uint8_t *w1, *s1, *w2, *s2, *w3, *s3;
} dsv4_expert_entry;

typedef struct {
    dsv4_store *store;
    dsv4_expert_entry *entries;
    dsv4_expert_entry *prefill_entries;
    int prefill_layer;
    uint32_t *heat;
    uint32_t *last;
    int layers;
    int capacity_per_layer;
    uint32_t clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    double last_route_seconds;
    double last_routed_seconds;
    double last_shared_seconds;
    /* Cumulative miss-path attribution, accumulated under the cache lock:
     * storage read, record hashing, staging-to-entry copy, host-to-device
     * staging plus upload, and expert kernel through its sync. */
    double read_seconds, hash_seconds, copy_seconds, pin_seconds,
        upload_seconds, kernel_seconds;
    uint64_t read_experts;
    void *read_staging;
    size_t read_staging_bytes;
    int read_staging_pinned;
    pthread_mutex_t lock;
    char error[256];
#ifdef COLI_CUDA
    ColiCuda *cuda;
    unsigned char **cuda_storage;
    int *cuda_layer;
    int *cuda_expert;
    int *cuda_references;
    uint32_t *cuda_stamp;
    /* Byte offsets of w1/s1/w2/s2/w3/s3 inside each slot, recorded at upload
     * so device pointers never depend on how the host entry is laid out now. */
    uint32_t (*cuda_offsets)[6];
    int cuda_slots;
    uint32_t cuda_clock;
    uint64_t cuda_hits;
    uint64_t cuda_misses;
    uint64_t cuda_evictions;
    uint64_t cuda_upload_bytes;
    uint64_t cuda_pinned_upload_bytes;
    void *cuda_upload_staging;
    size_t cuda_upload_staging_cap;
    void *cuda_activation;
    void *cuda_activation_scale;
    void *cuda_output;
    size_t cuda_activation_cap;
    size_t cuda_activation_scale_cap;
    size_t cuda_output_cap;
    /* Host side of the grouped middle stage: gate/up for every routed expert
     * come down, the quantized middle activations go back up. */
    float *group_gate_host, *group_up_host;
    uint8_t *group_middle, *group_middle_scale;
    int cuda_lock_initialized;
    pthread_mutex_t cuda_lock;
    /* Register entry storage with the driver after its first fill so uploads
     * DMA straight from it (28 GB/s measured) instead of a 13 MB memcpy into
     * pinned staging first (10 GB/s effective). DSV4_PINNED_CACHE=0 keeps the
     * staging route; a registration failure also falls back to it. */
    int pinned_cache;
    uint64_t cuda_direct_uploads, pin_failures;
#endif
} dsv4_expert_cache;

static inline size_t dsv4_expert_payload_bytes(void) {
    size_t hidden = DSV4_EXPERT_HIDDEN;
    size_t intermediate = DSV4_MOE_INTERMEDIATE;
    return intermediate * hidden / 2 + intermediate * (hidden / 32) +
           hidden * intermediate / 2 + hidden * (intermediate / 32) +
           intermediate * hidden / 2 + intermediate * (hidden / 32);
}

static inline int dsv4_expert_prefix(char *output, size_t capacity,
                                      int layer, int expert) {
    if (!output || capacity == 0 || layer < 0 ||
        layer >= DSV4_BASE_LAYERS + DSV4_DSPARK_LAYERS || expert < 0 ||
        expert >= DSV4_EXPERTS)
        return 0;
    int length = layer < DSV4_BASE_LAYERS
        ? snprintf(output, capacity, "layers.%d.ffn.experts.%d", layer,
                   expert)
        : snprintf(output, capacity, "mtp.%d.ffn.experts.%d",
                   layer - DSV4_BASE_LAYERS, expert);
    return length >= 0 && (size_t)length < capacity;
}

static inline int dsv4_expert_record_names(char names[6][192],
                                            int layer, int expert) {
    char prefix[128];
    if (!dsv4_expert_prefix(prefix, sizeof(prefix), layer, expert)) return 0;
    const char *suffix[6] = {".w1.weight", ".w1.scale", ".w2.weight",
                             ".w2.scale", ".w3.weight", ".w3.scale"};
    for (int index = 0; index < 6; index++) {
        int length = snprintf(names[index], 192, "%s%s", prefix,
                              suffix[index]);
        if (length < 0 || length >= 192) return 0;
    }
    return 1;
}

/* Entry storage is page-aligned so a packed on-disk expert can be read into
 * it directly with O_DIRECT. The default record pointers are the packed
 * w1/s1/w2/s2/w3/s3 layout; an in-place read re-points them to the on-disk
 * order instead, so every consumer must go through the pointers. */
static inline int dsv4_expert_entry_storage(dsv4_expert_entry *entry) {
    if (!entry) return 0;
    if (!entry->storage) {
        void *storage = NULL;
        if (posix_memalign(&storage, 4096, dsv4_expert_payload_bytes())) return 0;
        entry->storage = (unsigned char *)storage;
    }
    size_t w1 = (size_t)DSV4_MOE_INTERMEDIATE * DSV4_EXPERT_HIDDEN / 2;
    size_t s1 = (size_t)DSV4_MOE_INTERMEDIATE *
                (DSV4_EXPERT_HIDDEN / 32);
    size_t w2 = (size_t)DSV4_EXPERT_HIDDEN * DSV4_MOE_INTERMEDIATE / 2;
    size_t s2 = (size_t)DSV4_EXPERT_HIDDEN *
                (DSV4_MOE_INTERMEDIATE / 32);
    size_t w3 = w1;
    entry->w1 = entry->storage;
    entry->s1 = entry->w1 + w1;
    entry->w2 = entry->s1 + s1;
    entry->s2 = entry->w2 + w2;
    entry->w3 = entry->s2 + s2;
    entry->s3 = entry->w3 + w3;
    return entry->s3 + s1 == entry->storage + dsv4_expert_payload_bytes();
}

static inline void dsv4_expert_entry_free(dsv4_expert_cache *cache,
                                          dsv4_expert_entry *entry) {
    if (!entry) return;
#ifdef COLI_CUDA
    if (entry->pinned && cache->cuda)
        coli_cuda_host_unregister(cache->cuda, entry->storage);
#else
    (void)cache;
#endif
    free(entry->storage);
    memset(entry, 0, sizeof(*entry));
}

/* Page-lock a freshly filled entry once; later fills reuse the registration.
 * Registering touched pages costs under a millisecond per entry. */
static inline void dsv4_expert_pin(dsv4_expert_cache *cache,
                                   dsv4_expert_entry *entry) {
#ifdef COLI_CUDA
    if (!cache->cuda || !cache->pinned_cache || entry->pinned) return;
    if (coli_cuda_host_register(cache->cuda, entry->storage,
                                dsv4_expert_payload_bytes())) {
        cache->pin_failures++;
        cache->pinned_cache = 0;
        return;
    }
    entry->pinned = 1;
#else
    (void)cache; (void)entry;
#endif
}

static inline int dsv4_expert_descriptors(dsv4_expert_cache *cache,
                                           char names[6][192]) {
    const dsv4_dtype dtype[6] = {
        DSV4_DTYPE_I8, DSV4_DTYPE_UE8M0, DSV4_DTYPE_I8,
        DSV4_DTYPE_UE8M0, DSV4_DTYPE_I8, DSV4_DTYPE_UE8M0,
    };
    const int64_t rows[6] = {
        DSV4_MOE_INTERMEDIATE, DSV4_MOE_INTERMEDIATE,
        DSV4_EXPERT_HIDDEN, DSV4_EXPERT_HIDDEN,
        DSV4_MOE_INTERMEDIATE, DSV4_MOE_INTERMEDIATE,
    };
    const int64_t columns[6] = {
        DSV4_EXPERT_HIDDEN / 2, DSV4_EXPERT_HIDDEN / 32,
        DSV4_MOE_INTERMEDIATE / 2, DSV4_MOE_INTERMEDIATE / 32,
        DSV4_EXPERT_HIDDEN / 2, DSV4_EXPERT_HIDDEN / 32,
    };
    for (int index = 0; index < 6; index++) {
        const dsv4_tensor_desc *descriptor =
            dsv4_store_find(cache->store, names[index]);
        if (!dsv4_model_tensor_shape(descriptor, dtype[index], 2,
                                      rows[index], columns[index])) {
            snprintf(cache->error, sizeof(cache->error),
                     "expert tensor contract mismatch: %s", names[index]);
            return 0;
        }
    }
    return 1;
}

/* Read all six records of each expert in one bounded extent, including any
 * alignment gaps. The converter packs an expert's six records contiguously on
 * page boundaries, so the extent is normally the payload itself: it is then
 * read straight into the entry's page-aligned storage and hashed there, with
 * the record pointers following the on-disk order. An extent that is not
 * packed lands in staging and each record is copied into the packed layout.
 * Either way the bytes behind w1..s3 are identical. */
static inline int dsv4_expert_read_extents(dsv4_expert_cache *cache,
                                         char (*names)[192], dsv4_expert_entry **entries, int count) {
    if (count < 1 || count > DSV4_TOPK) return 0;
    st_tensor extents[DSV4_TOPK], *ptrs[DSV4_TOPK], *records[DSV4_TOPK*6];
    void *buffers[DSV4_TOPK] = {0};
    int in_place[DSV4_TOPK] = {0}, staged = 0, ok = 0;
    size_t payload = dsv4_expert_payload_bytes();
    for (int e=0;e<count;e++) {
        int64_t lo=INT64_MAX, hi=0; int fd=-1;
        for (int j=0;j<6;j++) {
            st_tensor *t=st_find(&cache->store->raw,names[e*6+j]);
            if (!t || t->off<0 || t->nbytes<=0 || t->off>INT64_MAX-t->nbytes ||
                (fd>=0 && t->fd!=fd)) goto done;
            for (int k=0;k<j;k++) {
                st_tensor *u=records[e*6+k];
                if (t->off<u->off+u->nbytes && u->off<t->off+t->nbytes) goto done;
            }
            records[e*6+j]=t; fd=t->fd;
            if (t->off<lo) lo=t->off;
            if (t->off+t->nbytes>hi) hi=t->off+t->nbytes;
        }
        if ((uint64_t)(hi-lo)>payload+6*4096u) goto done;
        extents[e]=*records[e*6]; extents[e].off=lo; extents[e].nbytes=hi-lo;
        ptrs[e]=&extents[e];
        in_place[e] = (uint64_t)(hi-lo)==payload && !(lo & 4095) &&
            entries[e]->storage && !((uintptr_t)entries[e]->storage & 4095u);
        if (!in_place[e]) staged = 1;
    }
    size_t stride=(payload+6*4096u+4095u)&~(size_t)4095u;
    if (staged && !cache->read_staging) {
        cache->read_staging_bytes=(size_t)DSV4_TOPK*stride;
#ifdef COLI_CUDA
        if (cache->cuda) {
            if (coli_cuda_malloc_host(cache->cuda,&cache->read_staging,cache->read_staging_bytes)) goto done;
            cache->read_staging_pinned=1;
        } else
#endif
        if (posix_memalign(&cache->read_staging,4096,cache->read_staging_bytes)) goto done;
    }
    for (int e=0;e<count;e++)
        buffers[e]=in_place[e] ? (void*)entries[e]->storage : (void*)((char*)cache->read_staging+(size_t)e*stride);
    double read_started=dsv4_dense_now_seconds();
    st_read_extents(&cache->store->raw,ptrs,buffers,count);
    double read_done=dsv4_dense_now_seconds();
    void *record_data[DSV4_TOPK*6];
    for (int e=0;e<count;e++) for (int j=0;j<6;j++)
        record_data[e*6+j]=(char*)buffers[e]+records[e*6+j]->off-extents[e].off;
    int verified=dsv4_store_verify_records(cache->store,records,record_data,count*6);
    double hash_done=dsv4_dense_now_seconds();
    cache->read_seconds+=read_done-read_started;
    cache->hash_seconds+=hash_done-read_done;
    cache->read_experts+=(uint64_t)count;
    if (!verified) goto done;
    for (int e=0;e<count;e++) {
        dsv4_expert_entry *entry=entries[e];
        uint8_t **slot[6]={&entry->w1,&entry->s1,&entry->w2,&entry->s2,&entry->w3,&entry->s3};
        if (in_place[e]) {
            for (int j=0;j<6;j++) *slot[j]=(uint8_t*)record_data[e*6+j];
        } else {
            dsv4_expert_entry_storage(entry);
            for (int j=0;j<6;j++) memcpy(*slot[j],record_data[e*6+j],(size_t)records[e*6+j]->nbytes);
        }
    }
    double copy_done=dsv4_dense_now_seconds();
    cache->copy_seconds+=copy_done-hash_done;
    for (int e=0;e<count;e++) dsv4_expert_pin(cache,entries[e]);
    cache->pin_seconds+=dsv4_dense_now_seconds()-copy_done;
    ok=1;
done:
    if (!ok) snprintf(cache->error,sizeof(cache->error),"%s%.180s",
        cache->store->error[0] ? "" : "invalid or unallocatable expert extent",cache->store->error);
    return ok;
}

static inline int dsv4_expert_load(dsv4_expert_cache *cache,
                                    dsv4_expert_entry *entry,
                                    int layer, int expert) {
    char names[6][192];
    if (!dsv4_expert_record_names(names, layer, expert) ||
        !dsv4_expert_descriptors(cache, names) ||
        !dsv4_expert_entry_storage(entry)) {
        if (!cache->error[0])
            snprintf(cache->error, sizeof(cache->error),
                     "cannot allocate expert %d/%d", layer, expert);
        return 0;
    }
    if (!dsv4_expert_read_extents(cache, names, &entry, 1)) return 0;
    entry->layer = layer;
    entry->expert = expert;
    entry->valid = 1;
    return 1;
}

static inline int dsv4_expert_cache_init(dsv4_expert_cache *cache,
                                          dsv4_store *store,
                                          int capacity_per_layer) {
    if (!cache || !store || capacity_per_layer < DSV4_TOPK ||
        capacity_per_layer > DSV4_EXPERTS)
        return 0;
    memset(cache, 0, sizeof(*cache));
    cache->store = store;
    cache->layers = DSV4_BASE_LAYERS + DSV4_DSPARK_LAYERS;
    cache->capacity_per_layer = capacity_per_layer;
    size_t slots = (size_t)cache->layers * capacity_per_layer;
    size_t keys = (size_t)cache->layers * DSV4_EXPERTS;
    cache->entries = (dsv4_expert_entry *)calloc(slots,
                                                 sizeof(*cache->entries));
    cache->heat = (uint32_t *)calloc(keys, sizeof(*cache->heat));
    cache->last = (uint32_t *)calloc(keys, sizeof(*cache->last));
    if (!cache->entries || !cache->heat || !cache->last) {
        snprintf(cache->error, sizeof(cache->error), "out of memory");
        free(cache->entries); free(cache->heat); free(cache->last);
        memset(cache, 0, sizeof(*cache));
        return 0;
    }
    pthread_mutex_init(&cache->lock, NULL);
    return 1;
}

/* Whole-prompt prefill visits one layer at a time. Reuse one layer's expert
 * storage across every prompt chunk, then recycle that storage for the next
 * layer. Drop the decode cache first so both host budgets never coexist. */
static inline int dsv4_expert_prefill_begin(dsv4_expert_cache *cache) {
    pthread_mutex_lock(&cache->lock);
    if (cache->prefill_entries) { pthread_mutex_unlock(&cache->lock); return 0; }
    size_t slots=(size_t)cache->layers*cache->capacity_per_layer;
    for (size_t i=0;i<slots;i++) if (cache->entries[i].references) {
        pthread_mutex_unlock(&cache->lock);return 0;
    }
    dsv4_expert_entry *window=calloc(DSV4_EXPERTS,sizeof(*window));
    if (!window) { pthread_mutex_unlock(&cache->lock);return 0; }
    for (size_t i=0;i<slots;i++) dsv4_expert_entry_free(cache,&cache->entries[i]);
    cache->prefill_entries=window;cache->prefill_layer=-1;
    pthread_mutex_unlock(&cache->lock);return 1;
}
static inline int dsv4_expert_prefill_layer(dsv4_expert_cache *cache,int layer) {
    pthread_mutex_lock(&cache->lock);
    if (!cache->prefill_entries || layer<0 || layer>=cache->layers) { pthread_mutex_unlock(&cache->lock);return 0; }
    for (int i=0;i<DSV4_EXPERTS;i++) if (cache->prefill_entries[i].references) { pthread_mutex_unlock(&cache->lock);return 0; }
    for (int i=0;i<DSV4_EXPERTS;i++) {
        cache->prefill_entries[i].valid=0;
        cache->prefill_entries[i].layer=layer;
    }
    cache->prefill_layer=layer;
    pthread_mutex_unlock(&cache->lock);return 1;
}
static inline void dsv4_expert_prefill_end(dsv4_expert_cache *cache) {
    pthread_mutex_lock(&cache->lock);
    if (cache->prefill_entries) {
        for (int i=0;i<DSV4_EXPERTS;i++) dsv4_expert_entry_free(cache,&cache->prefill_entries[i]);
        free(cache->prefill_entries);cache->prefill_entries=NULL;
    }
    cache->prefill_layer=-1;
    pthread_mutex_unlock(&cache->lock);
}
static inline dsv4_expert_entry *dsv4_expert_layer_entries(dsv4_expert_cache *cache,int layer,int *capacity) {
    if (cache->prefill_entries) {
        *capacity=DSV4_EXPERTS;
        return layer==cache->prefill_layer ? cache->prefill_entries : NULL;
    }
    *capacity=cache->capacity_per_layer;
    return cache->entries+(size_t)layer*cache->capacity_per_layer;
}

static inline dsv4_expert_entry *dsv4_expert_cache_acquire(
    dsv4_expert_cache *cache, int layer, int expert) {
    if (!cache || !cache->entries || layer < 0 || layer >= cache->layers ||
        expert < 0 || expert >= DSV4_EXPERTS)
        return NULL;
    pthread_mutex_lock(&cache->lock);
    cache->clock++;
    size_t key = (size_t)layer * DSV4_EXPERTS + expert;
    if (cache->heat[key] != UINT32_MAX) cache->heat[key]++;
    cache->last[key] = cache->clock;
    if ((cache->clock & 4095u) == 0)
        tier_decay(cache->heat, cache->layers * DSV4_EXPERTS);
    int layer_capacity;
    dsv4_expert_entry *base=dsv4_expert_layer_entries(cache,layer,&layer_capacity);
    if (!base) { pthread_mutex_unlock(&cache->lock);return NULL; }
    for (int slot = 0; slot < layer_capacity; slot++)
        if (base[slot].valid && base[slot].expert == expert) {
            base[slot].references++;
            cache->hits++;
            pthread_mutex_unlock(&cache->lock);
            return &base[slot];
        }
    int selected = -1;
    uint64_t coldest = UINT64_MAX;
    for (int slot = 0; slot < layer_capacity; slot++) {
        if (!base[slot].valid) { selected = slot; break; }
        if (base[slot].references) continue;
        size_t resident_key = (size_t)layer * DSV4_EXPERTS +
                              base[slot].expert;
        uint64_t score = tier_lfru_score(cache->heat[resident_key],
                                         cache->last[resident_key],
                                         cache->clock);
        if (score < coldest) { coldest = score; selected = slot; }
    }
    if (selected < 0) {
        snprintf(cache->error, sizeof(cache->error),
                 "all expert slots are active for layer %d", layer);
        pthread_mutex_unlock(&cache->lock);
        return NULL;
    }
    dsv4_expert_entry *entry = &base[selected];
    if (entry->valid) cache->evictions++;
    entry->valid = 0;
    if (!dsv4_expert_load(cache, entry, layer, expert)) {
        pthread_mutex_unlock(&cache->lock);
        return NULL;
    }
    entry->references = 1;
    cache->misses++;
    pthread_mutex_unlock(&cache->lock);
    return entry;
}

static inline int dsv4_expert_cache_release(dsv4_expert_cache *cache,
                                             dsv4_expert_entry *entry) {
    if (!cache || !entry) return 0;
    pthread_mutex_lock(&cache->lock);
    int ok = entry->references > 0;
    if (ok) entry->references--;
    pthread_mutex_unlock(&cache->lock);
    return ok;
}

/* Acquire a deduplicated prompt-expert group in one storage submission. The
 * caller chunks groups to capacity_per_layer and releases every returned entry
 * after the grouped expert computation. */
static inline int dsv4_expert_cache_acquire_many(
    dsv4_expert_cache *cache, int layer, const int *experts, int count,
    dsv4_expert_entry **output) {
    if (!cache || !cache->entries || !experts || !output || count < 1 ||
        count > cache->capacity_per_layer || layer < 0 ||
        layer >= cache->layers)
        return 0;
    for (int left = 0; left < count; left++) {
        if (experts[left] < 0 || experts[left] >= DSV4_EXPERTS) return 0;
        for (int right = 0; right < left; right++)
            if (experts[left] == experts[right]) {
                snprintf(cache->error, sizeof(cache->error),
                         "grouped expert ids must be unique");
                return 0;
            }
    }
    int *miss_input = (int *)calloc((size_t)count, sizeof(int));
    char (*names)[192] =
        (char (*)[192])calloc((size_t)count * 6, sizeof(*names));
    dsv4_expert_entry **loading =
        (dsv4_expert_entry **)calloc((size_t)count, sizeof(*loading));
    if (!miss_input || !names || !loading) {
        free(miss_input); free(names); free(loading);
        return 0;
    }
    memset(output, 0, (size_t)count * sizeof(*output));
    pthread_mutex_lock(&cache->lock);
    int layer_capacity;
    dsv4_expert_entry *base=dsv4_expert_layer_entries(cache,layer,&layer_capacity);
    int misses = 0, acquired = 0;
    if (!base) goto fail;
    for (int item = 0; item < count; item++) {
        cache->clock++;
        size_t key = (size_t)layer * DSV4_EXPERTS + experts[item];
        if (cache->heat[key] != UINT32_MAX) cache->heat[key]++;
        cache->last[key] = cache->clock;
        for (int slot = 0; slot < layer_capacity; slot++)
            if (base[slot].valid && base[slot].expert == experts[item]) {
                base[slot].references++;
                output[item] = &base[slot];
                cache->hits++;
                acquired++;
                break;
            }
        if (output[item]) continue;
        int selected = -1;
        uint64_t coldest = UINT64_MAX;
        for (int slot = 0; slot < layer_capacity; slot++) {
            if (!base[slot].valid && !base[slot].references) {
                selected = slot;
                break;
            }
            if (base[slot].references) continue;
            size_t resident_key = (size_t)layer * DSV4_EXPERTS +
                                  base[slot].expert;
            uint64_t score = tier_lfru_score(cache->heat[resident_key],
                                             cache->last[resident_key],
                                             cache->clock);
            if (score < coldest) { coldest = score; selected = slot; }
        }
        if (selected < 0) {
            snprintf(cache->error, sizeof(cache->error),
                     "insufficient inactive expert slots for grouped layer %d",
                     layer);
            goto fail;
        }
        dsv4_expert_entry *entry = &base[selected];
        if (entry->valid) cache->evictions++;
        entry->valid = 0;
        entry->references = 1;
        entry->layer = layer;
        entry->expert = experts[item];
        output[item] = entry;
        acquired++;
        if (!dsv4_expert_entry_storage(entry)) {
            snprintf(cache->error, sizeof(cache->error), "out of memory");
            goto fail;
        }
        if (!dsv4_expert_record_names(names + (size_t)misses * 6,
                                       layer, experts[item]) ||
            !dsv4_expert_descriptors(cache,
                                      names + (size_t)misses * 6))
            goto fail;
        loading[misses] = entry;
        miss_input[misses++] = item;
    }
    if ((cache->clock & 4095u) == 0)
        tier_decay(cache->heat, cache->layers * DSV4_EXPERTS);
    if (misses) {
        if (!dsv4_expert_read_extents(cache, names, loading, misses)) goto fail;
        for (int miss = 0; miss < misses; miss++)
            output[miss_input[miss]]->valid = 1;
        cache->misses += (uint64_t)misses;
    }
    pthread_mutex_unlock(&cache->lock);
    free(miss_input); free(names); free(loading);
    return acquired == count;

fail:
    for (int item = 0; item < count; item++)
        if (output[item] && output[item]->references > 0) {
            output[item]->references--;
            if (!output[item]->valid) output[item]->expert = -1;
        }
    pthread_mutex_unlock(&cache->lock);
    free(miss_input); free(names); free(loading);
    return 0;
}
#ifdef COLI_CUDA
static inline int dsv4_cuda_cache_buffer(
    dsv4_expert_cache *cache, void **buffer, size_t *capacity,
    size_t bytes, const char *name) {
    if (*capacity >= bytes) return 1;
    if (*buffer) coli_cuda_free(cache->cuda, *buffer);
    *buffer = NULL;
    *capacity = 0;
    if (coli_cuda_malloc(cache->cuda, buffer, bytes)) {
        snprintf(cache->error, sizeof(cache->error), "%s: %s", name,
                 coli_cuda_last_error());
        return 0;
    }
    *capacity = bytes;
    return 1;
}

static inline int dsv4_expert_cache_enable_cuda(
    dsv4_expert_cache *cache, ColiCuda *cuda, size_t budget_bytes) {
    if (!cache || !cache->entries || cache->cuda || !cuda ||
        budget_bytes < (size_t)DSV4_TOPK * dsv4_expert_payload_bytes())
        return 0;
    size_t requested = budget_bytes / dsv4_expert_payload_bytes();
    size_t maximum = (size_t)cache->layers * cache->capacity_per_layer;
    if (requested > maximum) requested = maximum;
    if (requested > INT_MAX) requested = INT_MAX;
    cache->cuda_storage = (unsigned char **)calloc(
        requested, sizeof(*cache->cuda_storage));
    cache->cuda_layer = (int *)malloc(requested * sizeof(*cache->cuda_layer));
    cache->cuda_expert = (int *)malloc(
        requested * sizeof(*cache->cuda_expert));
    cache->cuda_references = (int *)calloc(
        requested, sizeof(*cache->cuda_references));
    cache->cuda_stamp = (uint32_t *)calloc(
        requested, sizeof(*cache->cuda_stamp));
    cache->cuda_offsets = (uint32_t (*)[6])calloc(
        requested, sizeof(*cache->cuda_offsets));
    if (!cache->cuda_storage || !cache->cuda_layer || !cache->cuda_expert ||
        !cache->cuda_references || !cache->cuda_stamp || !cache->cuda_offsets) {
        free(cache->cuda_storage); free(cache->cuda_layer);
        free(cache->cuda_expert); free(cache->cuda_references);
        free(cache->cuda_stamp); free(cache->cuda_offsets);
        cache->cuda_storage = NULL; cache->cuda_layer = NULL;
        cache->cuda_expert = NULL; cache->cuda_references = NULL;
        cache->cuda_stamp = NULL; cache->cuda_offsets = NULL;
        snprintf(cache->error, sizeof(cache->error),
                 "cannot allocate CUDA expert-cache metadata");
        return 0;
    }
    for (size_t slot = 0; slot < requested; slot++) {
        cache->cuda_layer[slot] = -1;
        cache->cuda_expert[slot] = -1;
    }
    cache->cuda = cuda;
    cache->cuda_slots = (int)requested;
    const char *pinned_cache = getenv("DSV4_PINNED_CACHE");
    cache->pinned_cache = !pinned_cache || !*pinned_cache || atoi(pinned_cache) != 0;
    const char *pinned_env = getenv("DSV4_PINNED_UPLOAD");
    int use_pinned = !pinned_env || atoi(pinned_env) != 0;
    cache->cuda_upload_staging_cap = use_pinned
        ? (size_t)DSV4_TOPK * dsv4_expert_payload_bytes() : 0;
    if (use_pinned && coli_cuda_malloc_host(
            cuda, &cache->cuda_upload_staging,
            cache->cuda_upload_staging_cap)) {
        snprintf(cache->error, sizeof(cache->error),
                 "cannot allocate pinned CUDA expert staging: %s",
                 coli_cuda_last_error());
        free(cache->cuda_storage); free(cache->cuda_layer);
        free(cache->cuda_expert); free(cache->cuda_references);
        free(cache->cuda_stamp); free(cache->cuda_offsets);
        cache->cuda_storage = NULL; cache->cuda_layer = NULL;
        cache->cuda_expert = NULL; cache->cuda_references = NULL;
        cache->cuda_stamp = NULL; cache->cuda_offsets = NULL; cache->cuda = NULL;
        cache->cuda_slots = 0;
        cache->cuda_upload_staging_cap = 0;
        return 0;
    }
    if (pthread_mutex_init(&cache->cuda_lock, NULL)) {
        snprintf(cache->error, sizeof(cache->error),
                 "cannot initialize CUDA expert-cache lock");
        free(cache->cuda_storage); free(cache->cuda_layer);
        free(cache->cuda_expert); free(cache->cuda_references);
        free(cache->cuda_stamp); free(cache->cuda_offsets);
        coli_cuda_free_host(cuda, cache->cuda_upload_staging);
        cache->cuda_storage = NULL; cache->cuda_layer = NULL;
        cache->cuda_expert = NULL; cache->cuda_references = NULL;
        cache->cuda_stamp = NULL; cache->cuda_offsets = NULL; cache->cuda = NULL;
        cache->cuda_upload_staging = NULL;
        cache->cuda_upload_staging_cap = 0;
        cache->cuda_slots = 0;
        return 0;
    }
    cache->cuda_lock_initialized = 1;
    /* No active reads while enabling CUDA. The next read obtains reusable
     * pinned staging instead of retaining the earlier pageable allocation. */
    free(cache->read_staging); cache->read_staging=NULL;
    cache->read_staging_bytes=0;
    return 1;
}

static inline int dsv4_cuda_experts_acquire(
    dsv4_expert_cache *cache, dsv4_expert_entry **entries, int count,
    int *slots, const unsigned char **w1, const unsigned char **s1,
    const unsigned char **w2, const unsigned char **s2,
    const unsigned char **w3, const unsigned char **s3) {
    pthread_mutex_lock(&cache->lock);
    double upload_started = dsv4_dense_now_seconds();
    int acquired = 0;
    for (int item = 0; item < count; item++) {
        int selected = -1;
        cache->cuda_clock++;
        for (int slot = 0; slot < cache->cuda_slots; slot++)
            if (cache->cuda_layer[slot] == entries[item]->layer &&
                cache->cuda_expert[slot] == entries[item]->expert) {
                selected = slot;
                cache->cuda_hits++;
                break;
            }
        if (selected < 0) {
            uint64_t coldest = UINT64_MAX;
            for (int slot = 0; slot < cache->cuda_slots; slot++) {
                if (cache->cuda_references[slot]) continue;
                if (cache->cuda_layer[slot] < 0) {
                    selected = slot;
                    break;
                }
                size_t key = (size_t)cache->cuda_layer[slot] *
                    DSV4_EXPERTS + cache->cuda_expert[slot];
                uint64_t score = tier_lfru_score(
                    cache->heat[key], cache->cuda_stamp[slot],
                    cache->cuda_clock);
                if (score < coldest) {
                    coldest = score;
                    selected = slot;
                }
            }
            if (selected < 0) {
                snprintf(cache->error, sizeof(cache->error),
                         "all CUDA expert slots are active");
                goto fail;
            }
            if (!cache->cuda_storage[selected] &&
                coli_cuda_malloc(cache->cuda,
                    (void **)&cache->cuda_storage[selected],
                    dsv4_expert_payload_bytes())) {
                snprintf(cache->error, sizeof(cache->error),
                         "CUDA expert allocation: %s",
                         coli_cuda_last_error());
                goto fail;
            }
            if (cache->cuda_layer[selected] >= 0)
                cache->cuda_evictions++;
            const void *upload_source = entries[item]->storage;
            if (entries[item]->pinned) {
                cache->cuda_direct_uploads++;
            } else if (cache->cuda_upload_staging) {
                unsigned char *staging =
                    (unsigned char *)cache->cuda_upload_staging +
                    (size_t)item * dsv4_expert_payload_bytes();
                memcpy(staging, entries[item]->storage,
                       dsv4_expert_payload_bytes());
                upload_source = staging;
            }
            if (coli_cuda_upload(cache->cuda, cache->cuda_storage[selected],
                                 upload_source,
                                 dsv4_expert_payload_bytes())) {
                snprintf(cache->error, sizeof(cache->error),
                         "CUDA expert upload: %s", coli_cuda_last_error());
                goto fail;
            }
            cache->cuda_layer[selected] = entries[item]->layer;
            cache->cuda_expert[selected] = entries[item]->expert;
            const dsv4_expert_entry *entry = entries[item];
            const uint8_t *host[6] = {entry->w1, entry->s1, entry->w2,
                                      entry->s2, entry->w3, entry->s3};
            for (int record = 0; record < 6; record++)
                cache->cuda_offsets[selected][record] =
                    (uint32_t)(host[record] - entry->storage);
            cache->cuda_misses++;
            cache->cuda_upload_bytes += dsv4_expert_payload_bytes();
            if (entries[item]->pinned || cache->cuda_upload_staging)
                cache->cuda_pinned_upload_bytes +=
                    dsv4_expert_payload_bytes();
        }
        cache->cuda_references[selected]++;
        cache->cuda_stamp[selected] = cache->cuda_clock;
        slots[item] = selected;
        const unsigned char *base = cache->cuda_storage[selected];
        const uint32_t *offset = cache->cuda_offsets[selected];
        w1[item] = base + offset[0]; s1[item] = base + offset[1];
        w2[item] = base + offset[2]; s2[item] = base + offset[3];
        w3[item] = base + offset[4]; s3[item] = base + offset[5];
        acquired++;
    }
    cache->upload_seconds += dsv4_dense_now_seconds() - upload_started;
    pthread_mutex_unlock(&cache->lock);
    return 1;
fail:
    for (int item = 0; item < acquired; item++)
        cache->cuda_references[slots[item]]--;
    pthread_mutex_unlock(&cache->lock);
    return 0;
}

static inline void dsv4_cuda_experts_release(
    dsv4_expert_cache *cache, const int *slots, int count) {
    pthread_mutex_lock(&cache->lock);
    for (int item = 0; item < count; item++)
        if (slots[item] >= 0 &&
            cache->cuda_references[slots[item]] > 0)
            cache->cuda_references[slots[item]]--;
    pthread_mutex_unlock(&cache->lock);
}

static inline int dsv4_routed_experts_forward_cuda(
    dsv4_expert_cache *cache, dsv4_expert_entry **entries,
    const float *route_weights, const float *input, float *output,
    uint8_t *activation, uint8_t *activation_scale) {
    int slots[DSV4_TOPK];
    int cuda_entries_acquired = 0;
    const unsigned char *w1[DSV4_TOPK], *s1[DSV4_TOPK],
        *w2[DSV4_TOPK], *s2[DSV4_TOPK],
        *w3[DSV4_TOPK], *s3[DSV4_TOPK];
    memset(slots, -1, sizeof(slots));
    pthread_mutex_lock(&cache->cuda_lock);
    cache->error[0] = '\0';
    int ok = dsv4_act_quant_mxfp(
        input, DSV4_EXPERT_HIDDEN, activation, activation_scale);
    if (ok) {
        cuda_entries_acquired = dsv4_cuda_experts_acquire(
            cache, entries, DSV4_TOPK, slots,
            w1, s1, w2, s2, w3, s3);
        ok = cuda_entries_acquired;
    }
    if (ok) {
        ok = dsv4_cuda_cache_buffer(
                 cache, &cache->cuda_activation,
                 &cache->cuda_activation_cap, DSV4_EXPERT_HIDDEN,
                 "CUDA routed activation allocation") &&
             dsv4_cuda_cache_buffer(
                 cache, &cache->cuda_activation_scale,
                 &cache->cuda_activation_scale_cap,
                 DSV4_EXPERT_HIDDEN / 128,
                 "CUDA routed scale allocation") &&
             dsv4_cuda_cache_buffer(
                 cache, &cache->cuda_output, &cache->cuda_output_cap,
                 (size_t)DSV4_EXPERT_HIDDEN * sizeof(float),
                 "CUDA routed output allocation");
    }
    size_t middle_n = (size_t)DSV4_TOPK * DSV4_MOE_INTERMEDIATE;
    if (ok && !cache->group_gate_host) {
        cache->group_gate_host = (float *)malloc(middle_n * sizeof(float));
        cache->group_up_host = (float *)malloc(middle_n * sizeof(float));
        cache->group_middle = (uint8_t *)malloc(middle_n);
        cache->group_middle_scale = (uint8_t *)malloc(middle_n / 128);
        if (!cache->group_gate_host || !cache->group_up_host ||
            !cache->group_middle || !cache->group_middle_scale) {
            snprintf(cache->error, sizeof(cache->error),
                     "cannot allocate grouped middle scratch");
            ok = 0;
        }
    }
    double kernel_started = dsv4_dense_now_seconds();
    /* Hidden GEMMs for all routed experts, then the middle stage on the host
     * with the per-expert path's own code (route weight, clamped SwiGLU,
     * BF16, MXFP quantization), then the down GEMMs and the id-ordered
     * reduction. Device expf differs from libm by one ulp often enough to
     * flip a BF16 middle value every few tokens, which the serving output
     * must not depend on. */
    if (ok)
        ok = !coli_cuda_upload(
                 cache->cuda, cache->cuda_activation, activation,
                 DSV4_EXPERT_HIDDEN) &&
             !coli_cuda_upload(
                 cache->cuda, cache->cuda_activation_scale,
                 activation_scale, DSV4_EXPERT_HIDDEN / 128) &&
             !coli_cuda_dsv4_grouped_fp4_hidden(
                 cache->cuda, cache->group_gate_host, cache->group_up_host,
                 (const unsigned char *)cache->cuda_activation,
                 (const unsigned char *)cache->cuda_activation_scale,
                 w1, s1, w3, s3, DSV4_TOPK,
                 DSV4_EXPERT_HIDDEN, DSV4_MOE_INTERMEDIATE) &&
             !coli_cuda_sync(cache->cuda);
    if (ok) {
        for (int route = 0; route < DSV4_TOPK; route++) {
            float *middle = cache->group_gate_host +
                (size_t)route * DSV4_MOE_INTERMEDIATE;
            const float *up = cache->group_up_host +
                (size_t)route * DSV4_MOE_INTERMEDIATE;
            for (int i = 0; i < DSV4_MOE_INTERMEDIATE; i++)
                middle[i] = dsv4_round_bf16(
                    route_weights[route] * dsv4_clamped_swiglu(middle[i], up[i]));
            if (!dsv4_act_quant_mxfp(
                    middle, DSV4_MOE_INTERMEDIATE,
                    cache->group_middle + (size_t)route * DSV4_MOE_INTERMEDIATE,
                    cache->group_middle_scale +
                        (size_t)route * (DSV4_MOE_INTERMEDIATE / 128))) {
                ok = 0;
                break;
            }
        }
    }
    if (ok)
        ok = !coli_cuda_dsv4_grouped_fp4_down(
                 cache->cuda, (float *)cache->cuda_output,
                 cache->group_middle, cache->group_middle_scale, w2, s2,
                 DSV4_TOPK, DSV4_EXPERT_HIDDEN, DSV4_MOE_INTERMEDIATE) &&
             !coli_cuda_download(
                 cache->cuda, output, cache->cuda_output,
                 (size_t)DSV4_EXPERT_HIDDEN * sizeof(float)) &&
             !coli_cuda_sync(cache->cuda);
    cache->kernel_seconds += dsv4_dense_now_seconds() - kernel_started;
    if (!ok && !cache->error[0])
        snprintf(cache->error, sizeof(cache->error),
                 "CUDA routed expert execution: %s",
                 coli_cuda_last_error());
    if (cuda_entries_acquired)
        dsv4_cuda_experts_release(cache, slots, DSV4_TOPK);
    pthread_mutex_unlock(&cache->cuda_lock);
    return ok;
}
#endif

/* Scalar/store-backed routed MoE reduction. Production CUDA consumes the same
 * cache entries with the grouped FP4 kernel; this path is the exact CPU oracle
 * and fallback. Scratch is hidden output, two intermediate vectors, and MXFP
 * activation bytes/scales sized for max(hidden, intermediate). */
static inline int dsv4_routed_experts_forward(
    dsv4_expert_cache *cache, int layer, const int *experts,
    const float *route_weights, const float *input, float *output,
    float *expert_output, float *gate, float *up, uint8_t *activation,
    uint8_t *activation_scale) {
    if (!cache || !experts || !route_weights || !input || !output ||
        !expert_output || !gate || !up || !activation || !activation_scale)
        return 0;
    dsv4_expert_entry *entries[DSV4_TOPK];
    if (!dsv4_expert_cache_acquire_many(
            cache, layer, experts, DSV4_TOPK, entries))
        return 0;
    memset(output, 0, (size_t)DSV4_EXPERT_HIDDEN * sizeof(*output));
    int ok = 1;
#ifdef COLI_CUDA
    if (cache->cuda)
        ok = dsv4_routed_experts_forward_cuda(
            cache, entries, route_weights, input, output, activation,
            activation_scale);
    else
#endif
    {
    for (int route = 0; route < DSV4_TOPK; route++) {
        dsv4_expert_entry *entry = entries[route];
        if (!dsv4_expert_fp4(
                expert_output, input, entry->w1, entry->s1, entry->w2,
                entry->s2, entry->w3, entry->s3, DSV4_EXPERT_HIDDEN,
                DSV4_MOE_INTERMEDIATE, route_weights[route], gate, up,
                activation, activation_scale)) {
            ok = 0;
            break;
        }
        for (int index = 0; index < DSV4_EXPERT_HIDDEN; index++)
            output[index] += expert_output[index];
    }
    }
    for (int route = 0; route < DSV4_TOPK; route++)
        if (!dsv4_expert_cache_release(cache, entries[route])) ok = 0;
    return ok;
}

static inline int dsv4_moe_forward(
    dsv4_expert_cache *cache, const dsv4_dense_arena *dense, int layer,
    int token, const float *input, float *output, float *router_logits,
    float *routed, float *shared, float *expert_output, float *gate,
    float *up, uint8_t *activation, uint8_t *activation_scale) {
    if (!cache || !dense || !input || !output || !router_logits || !routed ||
        !shared || !expert_output || !gate || !up || !activation ||
        !activation_scale)
        return 0;
    int experts[DSV4_TOPK];
    float route_weights[DSV4_TOPK];
    double stage_started = dsv4_dense_now_seconds();
    if (!dsv4_dense_route(dense, layer, token, input, experts,
                           route_weights, router_logits))
        return 0;
    /* Upstream MoE accumulates expert outputs in increasing expert id. */
    for (int i=1;i<DSV4_TOPK;i++) for (int j=i;j>0 && experts[j]<experts[j-1];j--) {
        int id=experts[j];experts[j]=experts[j-1];experts[j-1]=id;
        float weight=route_weights[j];route_weights[j]=route_weights[j-1];route_weights[j-1]=weight;
    }
    cache->last_route_seconds =
        dsv4_dense_now_seconds() - stage_started;
    stage_started = dsv4_dense_now_seconds();
    if (!dsv4_routed_experts_forward(
            cache, layer, experts, route_weights, input, routed,
            expert_output, gate, up, activation, activation_scale))
        return 0;
    cache->last_routed_seconds =
        dsv4_dense_now_seconds() - stage_started;
    stage_started = dsv4_dense_now_seconds();
    if (!dsv4_dense_shared_expert(dense, layer, input, shared, gate, up,
                                   activation, activation_scale))
        return 0;
    cache->last_shared_seconds =
        dsv4_dense_now_seconds() - stage_started;
    for (int index = 0; index < DSV4_EXPERT_HIDDEN; index++)
        output[index] = dsv4_round_bf16(routed[index] + shared[index]);
    return 1;
}

static inline int dsv4_expert_cache_prefetch(dsv4_expert_cache *cache,
                                              int layer, const int *experts,
                                              int count) {
    if (!cache || !experts || count < 0) return 0;
    for (int item = 0; item < count; item++) {
        char names[6][192];
        if (!dsv4_expert_record_names(names, layer, experts[item])) return 0;
        for (int record = 0; record < 6; record++)
            st_prefetch(&cache->store->raw, names[record]);
    }
    return 1;
}

static inline void dsv4_expert_cache_close(dsv4_expert_cache *cache) {
    if (!cache) return;
#ifdef COLI_CUDA
    if (cache->cuda) {
        if (cache->cuda_lock_initialized)
            pthread_mutex_lock(&cache->cuda_lock);
        for (int slot = 0; slot < cache->cuda_slots; slot++)
            if (cache->cuda_storage[slot])
                coli_cuda_free(cache->cuda, cache->cuda_storage[slot]);
        if (cache->cuda_activation)
            coli_cuda_free(cache->cuda, cache->cuda_activation);
        if (cache->cuda_activation_scale)
            coli_cuda_free(cache->cuda, cache->cuda_activation_scale);
        if (cache->cuda_output)
            coli_cuda_free(cache->cuda, cache->cuda_output);
        free(cache->group_gate_host); free(cache->group_up_host);
        free(cache->group_middle); free(cache->group_middle_scale);
        coli_cuda_sync(cache->cuda);
        if (cache->cuda_upload_staging)
            coli_cuda_free_host(
                cache->cuda, cache->cuda_upload_staging);
        if (cache->cuda_lock_initialized) {
            pthread_mutex_unlock(&cache->cuda_lock);
            pthread_mutex_destroy(&cache->cuda_lock);
        }
    }
    free(cache->cuda_storage); free(cache->cuda_layer);
    free(cache->cuda_expert); free(cache->cuda_references);
    free(cache->cuda_stamp); free(cache->cuda_offsets);
#endif
    if (cache->entries) {
        dsv4_expert_prefill_end(cache);
        size_t slots = (size_t)cache->layers * cache->capacity_per_layer;
        for (size_t slot = 0; slot < slots; slot++)
            dsv4_expert_entry_free(cache, &cache->entries[slot]);
        pthread_mutex_destroy(&cache->lock);
    }
#ifdef COLI_CUDA
    if (cache->read_staging_pinned) coli_cuda_free_host(cache->cuda,cache->read_staging);
    else
#endif
    free(cache->read_staging);
    free(cache->entries); free(cache->heat); free(cache->last);
    memset(cache, 0, sizeof(*cache));
}

#endif
