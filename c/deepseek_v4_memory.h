#ifndef COLIB_DEEPSEEK_V4_MEMORY_H
#define COLIB_DEEPSEEK_V4_MEMORY_H
#include "deepseek_v4_prefill.h"

#define DSV4_GIB (UINT64_C(1024)*1024*1024)
#define DSV4_MAX_PAYLOAD (16u*1024u*1024u)

/* Current attention state is host-backed. Keep every persistent scratch buffer
 * in this accounting, including ones shared by compressed attention modes. */
static inline uint64_t dsv4_runtime_scratch_bytes(int context) {
    if (context < 1 || context > DSV4_MAX_CONTEXT) return 0;
    uint64_t query = DSV4_ATTN_HEADS * DSV4_ATTN_HEAD_DIM;
    uint64_t selection = DSV4_ATTN_WINDOW +
        (DSV4_INDEX_TOPK > (context+127)/128 ? DSV4_INDEX_TOPK : (context+127)/128);
    uint64_t act = query > DSV4_ATTN_HIDDEN ? query : DSV4_ATTN_HIDDEN;
    if (act < DSV4_MOE_INTERMEDIATE) act = DSV4_MOE_INTERMEDIATE;
    uint64_t floats = (6+DSV4_HC_MULT)*DSV4_ATTN_HIDDEN +
        DSV4_HC_MULT + DSV4_HC_MULT*DSV4_HC_MULT + DSV4_HC_MIX +
        DSV4_ATTN_Q_RANK + 2*query + DSV4_ATTN_HEAD_DIM +
        DSV4_ATTN_O_GROUPS*DSV4_ATTN_O_RANK + selection +
        DSV4_EXPERTS + 2*DSV4_MOE_INTERMEDIATE +
        6*DSV4_ATTN_HEAD_DIM + DSV4_INDEX_HEADS*DSV4_INDEX_DIM +
        4*DSV4_INDEX_DIM + DSV4_INDEX_HEADS + (context+DSV4_INDEX_RATIO-1)/DSV4_INDEX_RATIO;
    return floats*sizeof(float) + selection*sizeof(int) +
        2*act + (act+127)/128 + (act+31)/32 +
        DSV4_ATTN_Q_RANK + (DSV4_ATTN_Q_RANK+127)/128;
}

/* Upper bound for both dense and expert reusable device projections. Dense
 * attention uses bounded chunks. Reserve the largest dense width and both
 * reusable attention/index caches at the engine maximum context. No
 * full FP4/FP8 weight expansion is performed by the native kernels. */
static inline uint64_t dsv4_device_scratch_bytes(int chunk) {
    if (chunk < 1 || chunk > 2048) return 0;
    uint64_t width = DSV4_ATTN_HEADS*DSV4_ATTN_HEAD_DIM;
    if (width < DSV4_ATTN_HIDDEN) width = DSV4_ATTN_HIDDEN;
    uint64_t dense = (uint64_t)chunk*(width*8 + (width+127)/128) +
        (uint64_t)DSV4_VOCAB*sizeof(float);
    uint64_t expert = (uint64_t)chunk*(DSV4_EXPERT_HIDDEN*5 +
        (DSV4_EXPERT_HIDDEN+31)/32);
    uint64_t grouped = (uint64_t)DSV4_TOPK*(6*sizeof(void*)+sizeof(float)+
        2*DSV4_MOE_INTERMEDIATE*sizeof(float)+DSV4_EXPERT_HIDDEN*sizeof(float)+
        DSV4_MOE_INTERMEDIATE+(DSV4_MOE_INTERMEDIATE+127)/128);
    uint64_t selected = (DSV4_MAX_CONTEXT+DSV4_COMPRESS_RATIO-1)/DSV4_COMPRESS_RATIO;
    if (selected < DSV4_INDEX_TOPK) selected = DSV4_INDEX_TOPK;
    uint64_t attention = ((uint64_t)DSV4_ATTN_WINDOW+(DSV4_MAX_CONTEXT+DSV4_INDEX_RATIO-1)/DSV4_INDEX_RATIO)*DSV4_ATTN_HEAD_DIM*sizeof(float) +
        ((uint64_t)DSV4_MAX_CONTEXT+DSV4_INDEX_RATIO-1)/DSV4_INDEX_RATIO*DSV4_INDEX_DIM*sizeof(float) +
        (uint64_t)chunk*(2*width*sizeof(float)+DSV4_ATTN_HEAD_DIM*sizeof(float)+
            (DSV4_ATTN_WINDOW+selected+1)*sizeof(int))+DSV4_MAX_CONTEXT*sizeof(float)+DSV4_ATTN_HEADS*sizeof(float);
    return dense + expert + grouped + attention;
}

typedef struct {
    int context, chunk, host_capacity;
    uint64_t host_available, device_available;
    /* dense_bytes is the arena as loaded; dense_released is what the arena
     * returns to the host once the device holds it (FP8 payload pages), so
     * the steady-state host footprint is dense_bytes - dense_released. */
    uint64_t dense_bytes, dense_released, state_bytes, host_scratch, device_scratch;
    uint64_t staging_bytes, request_bytes, host_cache, device_cache;
    uint64_t host_headroom, device_headroom, host_required, device_required;
    char error[256];
} dsv4_memory_plan;

/* Pure planning function: no weights or device buffers are allocated here. */
static inline int dsv4_plan_memory(dsv4_memory_plan *p, int context, int chunk,
    uint64_t host_available, uint64_t device_available, int cuda,
    uint64_t host_budget, uint64_t device_budget, uint64_t device_headroom,
    uint64_t host_dense_released) {
    memset(p, 0, sizeof(*p));
    if (context < 1 || context > DSV4_MAX_CONTEXT ||
        (chunk != 256 && chunk != 512 && chunk != 1024 && chunk != 2048)) {
        snprintf(p->error, sizeof(p->error), "invalid context or prefill chunk"); return 0;
    }
    p->context=context; p->host_available=host_available; p->device_available=device_available;
    p->dense_bytes=DSV4_BASE_DENSE_BYTES + (uint64_t)DSV4_BASE_DENSE_RECORDS*DSV4_DENSE_ALIGNMENT;
    p->state_bytes=dsv4_runtime_state_bytes(context);
    p->host_headroom=DSV4_GIB; p->device_headroom=device_headroom;
    /* Dense upload staging (maximum supported override), read extents, direct
     * bounce buffers, and the expert upload staging can coexist. */
    p->staging_bytes=512u*1024u*1024u + (uint64_t)DSV4_TOPK*
        (3*dsv4_expert_payload_bytes()+12u*4096u);
    /* Payload + worst-case byte-bound token IDs + tokenizer/metadata reserve. */
    p->request_bytes=(uint64_t)DSV4_MAX_PAYLOAD*5 + 256u*1024u*1024u +
        (uint64_t)DSV4_VOCAB*(sizeof(float)+16);
    uint64_t layer_bytes=(uint64_t)(DSV4_BASE_LAYERS+DSV4_DSPARK_LAYERS)*dsv4_expert_payload_bytes();
    uint64_t minimum_host=layer_bytes*DSV4_TOPK;
    uint64_t minimum_device=(uint64_t)DSV4_TOPK*dsv4_expert_payload_bytes();
    p->dense_released=cuda && host_dense_released<p->dense_bytes ? host_dense_released : 0;
    /* The whole arena is host-resident while it loads and uploads; runtime
     * state, prefill banks and the expert cache only exist afterwards. */
    uint64_t load_peak=p->dense_bytes+p->staging_bytes+p->request_bytes+p->host_headroom;
    if (load_peak > host_available) {
        snprintf(p->error,sizeof(p->error),
            "minimum native memory budget cannot fit (host_available=%llu device_available=%llu)",
            (unsigned long long)host_available,(unsigned long long)device_available);
        return 0;
    }
    for (; chunk >= 256; chunk /= 2) {
        p->chunk=chunk;
        p->host_scratch=dsv4_runtime_scratch_bytes(context)+dsv4_prefill_host_bytes(chunk)+
            dsv4_prefill_bank_bytes(context<DSV4_REVIEW_INPUT_TOKENS ? context : DSV4_REVIEW_INPUT_TOKENS);
        p->device_scratch=cuda ? dsv4_device_scratch_bytes(chunk) : 0;
        uint64_t fixed=p->dense_bytes-p->dense_released+p->state_bytes+p->host_scratch+
            p->staging_bytes+p->request_bytes+p->host_headroom;
        if (fixed > host_available || minimum_host > host_available-fixed || host_budget < minimum_host) continue;
        uint64_t cache=host_available-fixed;
        if (cache>host_budget) cache=host_budget;
        uint64_t capacity=cache/layer_bytes;
        if (capacity>DSV4_EXPERTS) capacity=DSV4_EXPERTS;
        p->host_capacity=(int)capacity; p->host_cache=capacity*layer_bytes;
        p->host_required=fixed+p->host_cache;
        if (!cuda) return 1;
        if (p->device_scratch>2*DSV4_GIB) continue;
        uint64_t device_fixed=p->dense_bytes+p->device_scratch+device_headroom;
        if (device_fixed>device_available || minimum_device>device_available-device_fixed || device_budget<minimum_device) continue;
        cache=device_available-device_fixed;
        if (cache>device_budget) cache=device_budget;
        p->device_cache=(cache/dsv4_expert_payload_bytes())*dsv4_expert_payload_bytes();
        p->device_required=device_fixed+p->device_cache;
        return 1;
    }
    snprintf(p->error,sizeof(p->error),
        "minimum native memory budget cannot fit (host_available=%llu device_available=%llu)",
        (unsigned long long)host_available,(unsigned long long)device_available);
    return 0;
}

static inline uint64_t dsv4_host_available_bytes(void) {
    FILE *f=fopen("/proc/meminfo","r");
    if (!f) return 0;
    char line[256]; unsigned long long kb=0;
    while (fgets(line,sizeof(line),f)) if (sscanf(line,"MemAvailable: %llu kB",&kb)==1) break;
    fclose(f);
    uint64_t available=(uint64_t)kb*1024;
    /* cgroup v2 may impose a smaller effective memory ceiling. */
    f=fopen("/sys/fs/cgroup/memory.max","r");
    unsigned long long maximum=0,current=0;
    int bounded=f && fscanf(f,"%llu",&maximum)==1;
    if (f) fclose(f);
    if (bounded) {
        f=fopen("/sys/fs/cgroup/memory.current","r");
        if (!f || fscanf(f,"%llu",&current)!=1) { if(f) fclose(f); return 0; }
        fclose(f);
        uint64_t remaining=maximum>current ? maximum-current : 0;
        if (remaining<available) available=remaining;
    }
    return available;
}
#endif
