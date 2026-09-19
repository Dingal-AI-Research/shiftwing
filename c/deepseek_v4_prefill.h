
#ifndef COLIB_DEEPSEEK_V4_PREFILL_H
#define COLIB_DEEPSEEK_V4_PREFILL_H
/* Before this header pulls in any system header: glibc locks its
 * feature-test macros at the first one it sees, and st.h defining
 * _GNU_SOURCE further down the include chain is then too late -- O_DIRECT
 * stays invisible and every expert read silently falls back to buffered.
 * Guarded so a translation unit that already defined it is untouched. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "deepseek_v4_attention_batch.h"

/* Notifications describe committed tokens separately from work in the current
 * chunk. Returning zero cancels and poisons partially advanced attention state. */
typedef int (*dsv4_prefill_progress_fn)(void *, int committed, int chunk,
                                      int layer, int layers, int activity);

static inline int dsv4_prefill_linear(
    dsv4_expert_cache *cache, float *out, const float *input,
    const uint8_t *weight, const uint8_t *scale, int batch, int rows, int cols,
    uint8_t *act, uint8_t *act_scale) {
#ifdef COLI_CUDA
    if (cache->cuda) {
        if (!dsv4_act_quant_mxfp_batch(input,batch,cols,act,act_scale)) return 0;
        size_t n = (size_t)batch * cols, bytes = (size_t)batch * rows * sizeof(float);
        if (!dsv4_cuda_cache_buffer(cache, &cache->cuda_activation,
                &cache->cuda_activation_cap, n, "prefill activation") ||
            !dsv4_cuda_cache_buffer(cache, &cache->cuda_activation_scale,
                &cache->cuda_activation_scale_cap, n / 128, "prefill scales") ||
            !dsv4_cuda_cache_buffer(cache, &cache->cuda_output,
                &cache->cuda_output_cap, bytes, "prefill output")) return 0;
        if (coli_cuda_upload(cache->cuda, cache->cuda_activation, act, n) ||
            coli_cuda_upload(cache->cuda, cache->cuda_activation_scale, act_scale, n / 128) ||
            coli_cuda_dsv4_fp4_gemm(cache->cuda, cache->cuda_output,
                cache->cuda_activation, cache->cuda_activation_scale,
                weight, scale, batch, rows, cols) ||
            coli_cuda_download(cache->cuda, out, cache->cuda_output, bytes) ||
            coli_cuda_sync(cache->cuda)) return 0;
        dsv4_round_bf16_array(out, (size_t)batch * rows);
        return 1;
    }
#endif
    return dsv4_linear_fp4(out, input, weight, scale, batch, rows, cols, act, act_scale);
}

/* One resident expert, many assigned tokens. Routing weights are applied before
 * middle-activation quantization, as in the pinned implementation. */
static inline int dsv4_prefill_expert(
    dsv4_expert_cache *cache, dsv4_expert_entry *entry,
    const float *input, const float *routes, float *out, int batch,
    float *gate, float *up, uint8_t *act, uint8_t *act_scale) {
    const uint8_t *w1 = entry->w1, *s1 = entry->s1, *w2 = entry->w2,
        *s2 = entry->s2, *w3 = entry->w3, *s3 = entry->s3;
    int ok = 1;
#ifdef COLI_CUDA
    int slots[DSV4_TOPK] = {-1};
    int acquired = 0;
    if (cache->cuda) {
        pthread_mutex_lock(&cache->cuda_lock);
        acquired = dsv4_cuda_experts_acquire(cache, &entry, 1, slots,
                                             &w1, &s1, &w2, &s2, &w3, &s3);
        ok = acquired;
    }
#endif
    double kernel_started = dsv4_dense_now_seconds();
    if (ok) ok = dsv4_prefill_linear(cache, gate, input, w1, s1, batch,
                         DSV4_MOE_INTERMEDIATE, DSV4_EXPERT_HIDDEN, act, act_scale) &&
                  dsv4_prefill_linear(cache, up, input, w3, s3, batch,
                         DSV4_MOE_INTERMEDIATE, DSV4_EXPERT_HIDDEN, act, act_scale);
    if (ok) {
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < DSV4_MOE_INTERMEDIATE; i++) {
                size_t j = (size_t)b * DSV4_MOE_INTERMEDIATE + i;
                gate[j] = dsv4_round_bf16(routes[b] * dsv4_clamped_swiglu(gate[j], up[j]));
            }
        ok = dsv4_prefill_linear(cache, out, gate, w2, s2, batch,
                      DSV4_EXPERT_HIDDEN, DSV4_MOE_INTERMEDIATE, act, act_scale);
    }
    cache->kernel_seconds += dsv4_dense_now_seconds() - kernel_started;
#ifdef COLI_CUDA
    if (cache->cuda) {
        if (acquired) dsv4_cuda_experts_release(cache, slots, 1);
        pthread_mutex_unlock(&cache->cuda_lock);
    }
#endif
    return ok;
}

static inline int dsv4_prefill_attention(dsv4_runtime *r, int layer, float *hc) {
    dsv4_block_scratch *s = &r->block;
    dsv4_runtime_layer *state = &r->layers[layer];
    dsv4_block_attention_context sliding = {r->dense, layer,
        &state->attention.sliding, &s->attention};
    dsv4_block_nonoverlap_attention_context nonoverlap = {r->dense, layer,
        &state->attention.nonoverlap, &r->nonoverlap_scratch};
    dsv4_block_overlap_attention_context overlap = {r->dense, layer,
        &state->attention.overlap, &r->overlap_scratch};
    dsv4_hc_module_fn module = dsv4_block_attention_module;
    void *context = &sliding;
    if (state->mode == DSV4_LAYER_OVERLAP) {
        module = dsv4_block_overlap_attention_module; context = &overlap;
    } else if (state->mode == DSV4_LAYER_NONOVERLAP) {
        module = dsv4_block_nonoverlap_attention_module; context = &nonoverlap;
    }
    const float *fn, *scale, *base;
    const uint16_t *norm;
    return dsv4_block_bind_stage(r->dense, layer, "attn", &fn, &scale, &base, &norm) &&
        dsv4_hc_stage(hc, DSV4_ATTN_HIDDEN, fn, scale, base, norm, 1e-6f,
            module, context, s->reduced, s->post, s->combination, s->mixes,
            s->module_output, s->expanded);
}

static inline size_t dsv4_prefill_host_bytes(int batch) {
    if (batch < 1 || batch > 2048) return 0;
    size_t h = DSV4_ATTN_HIDDEN, m = DSV4_HC_MULT;
    return dsv4_attention_batch_bytes(batch) + dsv4_attention_collection_bytes(batch) + (size_t)batch * ((m + 4 + DSV4_TOPK) * h * sizeof(float) +
        2 * DSV4_MOE_INTERMEDIATE * sizeof(float) +
        (m + m*m + DSV4_TOPK + 1) * sizeof(float) +
        (DSV4_TOPK + 1) * sizeof(int) +
        (h > DSV4_MOE_INTERMEDIATE ? h : DSV4_MOE_INTERMEDIATE) * 2);
}

static inline void *dsv4_prefill_alloc(size_t count,size_t width) {
    return count && width && count<=SIZE_MAX/width ? malloc(count*width) : NULL;
}

/* A bounded chunk of one layer. The HC bank is owned by the caller. */
static inline int dsv4_prefill_layer_chunk(dsv4_runtime *r,const int *tokens,int count,int layer,float *hc,
    dsv4_prefill_progress_fn progress,void *opaque) {
    const size_t h=DSV4_ATTN_HIDDEN,m=DSV4_HC_MULT,bh=(size_t)count*h;
    size_t a=(size_t)count*(h>DSV4_MOE_INTERMEDIATE ? h : DSV4_MOE_INTERMEDIATE);
    float *input = dsv4_prefill_alloc(bh, sizeof(float)),
        *post = dsv4_prefill_alloc((size_t)count*m, sizeof(float)),
        *comb = dsv4_prefill_alloc((size_t)count*m*m, sizeof(float)),
        *route_out = dsv4_prefill_alloc(bh*DSV4_TOPK, sizeof(float)),
        *group = dsv4_prefill_alloc(bh, sizeof(float)), *output = dsv4_prefill_alloc(bh, sizeof(float)),
        *gate = dsv4_prefill_alloc((size_t)count*DSV4_MOE_INTERMEDIATE, sizeof(float)),
        *up = dsv4_prefill_alloc((size_t)count*DSV4_MOE_INTERMEDIATE, sizeof(float)),
        *routes = dsv4_prefill_alloc((size_t)count*DSV4_TOPK, sizeof(float)),
        *group_routes = dsv4_prefill_alloc(count, sizeof(float));
    int *ids = dsv4_prefill_alloc((size_t)count*DSV4_TOPK, sizeof(int)),
        *assignments = dsv4_prefill_alloc(count, sizeof(int));
    uint8_t *act = malloc(a), *act_scale = malloc((a+31)/32);
    int ok = hc && input && post && comb && route_out && group && output &&
        gate && up && routes && group_routes && ids && assignments && act && act_scale;
    if (!ok) goto done;
        double phase_started = dsv4_dense_now_seconds();
        if (!dsv4_attention_batch(r,layer,hc,count,progress,opaque)) goto fail;
        if (count == 1) r->decode_attention_seconds += dsv4_dense_now_seconds() - phase_started;
        phase_started = dsv4_dense_now_seconds();
        const float *fn, *scale, *base; const uint16_t *norm;
        if (!dsv4_block_bind_stage(r->dense, layer, "ffn", &fn, &scale, &base, &norm)) goto fail;
        for (int b = 0; b < count; b++) {
            float *x = input+(size_t)b*h;
            dsv4_hc_forward_pre(hc+(size_t)b*h*m, fn, h, scale, base, 1e-6f,
                x, post+(size_t)b*m, comb+(size_t)b*m*m, r->block.mixes);
            dsv4_round_bf16_array(x, h);
            dsv4_rmsnorm(x, x, norm, h, 1e-6f);
            if (!dsv4_dense_route(r->dense, layer, tokens[b], x,
                ids+(size_t)b*DSV4_TOPK, routes+(size_t)b*DSV4_TOPK, r->block.router_logits)) goto fail;
        }
        if (count == 1) r->decode_route_seconds += dsv4_dense_now_seconds() - phase_started;
        if (r->trace && !r->trace(r->trace_context,"ffn_input",layer,input,count,h)) goto fail;
        if (count == 1 && !r->experts->prefill_entries && r->decode_grouped) {
            /* Decode: the six routed experts of this layer go through one
             * grouped submission -- one storage batch (up to six extents, so
             * the drive sees queue depth six and the record hashes run in
             * parallel), one device acquire, one grouped kernel, one sync --
             * exactly the smoke path dsv4_moe_forward takes. Six serial
             * single-expert fetches cost about 10 ms of queue-depth-one read
             * plus a single-threaded hash and eighteen synchronised launches
             * per layer, which was most of the decode step. Expert ids are
             * sorted ascending so the reduction order matches upstream. */
            int sorted[DSV4_TOPK]; float weights[DSV4_TOPK];
            for (int k=0;k<DSV4_TOPK;k++) { sorted[k]=ids[k]; weights[k]=routes[k]; }
            for (int k=1;k<DSV4_TOPK;k++) for (int j=k;j>0 && sorted[j]<sorted[j-1];j--) {
                int id=sorted[j];sorted[j]=sorted[j-1];sorted[j-1]=id;
                float weight=weights[j];weights[j]=weights[j-1];weights[j-1]=weight;
            }
            if (progress && !progress(opaque, r->position, count, layer,
                                      DSV4_RUNTIME_LAYERS, count+sorted[DSV4_TOPK-1])) goto fail;
            phase_started = dsv4_dense_now_seconds();
            if (!dsv4_routed_experts_forward(r->experts, layer, sorted, weights, input,
                    group, route_out, gate, up, act, act_scale)) goto fail;
            r->decode_routed_seconds += dsv4_dense_now_seconds() - phase_started;
            phase_started = dsv4_dense_now_seconds();
            if (!dsv4_dense_shared_expert_batch(r->dense, layer, input, output,
                                                count, gate, up, act, act_scale)) goto fail;
            r->decode_shared_seconds += dsv4_dense_now_seconds() - phase_started;
            for (size_t i = 0; i < h; i++) {
                r->block.module_output[i] = dsv4_round_bf16(group[i] + output[i]);
                group[i] = r->block.module_output[i];
            }
            dsv4_hc_post(r->block.module_output, hc, h, post, comb, r->block.expanded);
            dsv4_round_bf16_array(r->block.expanded, h*m);
            memcpy(hc, r->block.expanded, h*m*sizeof(float));
            if (r->trace && !r->trace(r->trace_context,"ffn",layer,group,count,h)) goto fail;
            goto done;
        }
        if (r->experts->prefill_entries) {
            int needed[DSV4_EXPERTS]={0};
            for (int b=0;b<count*DSV4_TOPK;b++) {
                if (ids[b]<0 || ids[b]>=DSV4_EXPERTS) goto fail;
                needed[ids[b]]=1;
            }
            for (int slot=0;slot<DSV4_EXPERTS;slot++) {
                dsv4_expert_entry *resident=&r->experts->prefill_entries[slot];
                if (resident->valid) needed[resident->expert]=0;
            }
            int batch_ids[DSV4_TOPK],n=0;
            for (int expert=0;expert<=DSV4_EXPERTS;expert++) {
                if (expert<DSV4_EXPERTS && needed[expert]) batch_ids[n++]=expert;
                if (n && (n==DSV4_TOPK || expert==DSV4_EXPERTS)) {
                    dsv4_expert_entry *entries[DSV4_TOPK]={0};
                    if (progress && !progress(opaque,r->position,count,layer,DSV4_RUNTIME_LAYERS,count+batch_ids[n-1])) goto fail;
                    if (!dsv4_expert_cache_acquire_many(r->experts,layer,batch_ids,n,entries)) goto fail;
                    for (int j=0;j<n;j++) if (!dsv4_expert_cache_release(r->experts,entries[j])) goto fail;
                    n=0;
                }
            }
        }
        phase_started = dsv4_dense_now_seconds();
        for (int expert = 0; expert < DSV4_EXPERTS; expert++) {
            int n = 0;
            for (int b = 0; b < count; b++)
                for (int route = 0; route < DSV4_TOPK; route++) {
                    int at = b*DSV4_TOPK+route;
                    if (ids[at] == expert) {
                        if (n >= count) goto fail; /* routing must be unique */
                        assignments[n] = at; group_routes[n] = routes[at];
                        memcpy(group+(size_t)n*h, input+(size_t)b*h, h*sizeof(float)); n++;
                    }
                }
            if (!n) continue;
            if (progress && !progress(opaque, r->position, count, layer,
                                      DSV4_RUNTIME_LAYERS, count+expert)) goto fail;
            dsv4_expert_entry *entry = NULL;
            if (!dsv4_expert_cache_acquire_many(r->experts, layer, &expert, 1, &entry)) goto fail;
            int expert_ok = dsv4_prefill_expert(r->experts, entry, group, group_routes,
                                               output, n, gate, up, act, act_scale);
            int released = dsv4_expert_cache_release(r->experts, entry);
            if (!expert_ok || !released) goto fail;
            for (int i = 0; i < n; i++)
                memcpy(route_out+(size_t)assignments[i]*h, output+(size_t)i*h, h*sizeof(float));
        }
        if (count == 1) r->decode_routed_seconds += dsv4_dense_now_seconds() - phase_started;
        phase_started = dsv4_dense_now_seconds();
        if (!dsv4_dense_shared_expert_batch(r->dense, layer, input, output,
                                            count, gate, up, act, act_scale)) goto fail;
        if (count == 1) r->decode_shared_seconds += dsv4_dense_now_seconds() - phase_started;
        for (int b = 0; b < count; b++) {
            int order[DSV4_TOPK];for (int k=0;k<DSV4_TOPK;k++) order[k]=k;
            for (int k=1;k<DSV4_TOPK;k++) for (int j=k;j>0 && ids[b*DSV4_TOPK+order[j]]<ids[b*DSV4_TOPK+order[j-1]];j--) {
                int value=order[j];order[j]=order[j-1];order[j-1]=value;
            }
            for (size_t i = 0; i < h; i++) {
                float sum = 0;
                for (int route = 0; route < DSV4_TOPK; route++)
                    sum += route_out[((size_t)b*DSV4_TOPK+order[route])*h+i];
                r->block.module_output[i] = dsv4_round_bf16(sum + output[(size_t)b*h+i]);
                group[(size_t)b*h+i]=r->block.module_output[i];
            }
            dsv4_hc_post(r->block.module_output, hc+(size_t)b*h*m, h,
                post+(size_t)b*m, comb+(size_t)b*m*m, r->block.expanded);
            dsv4_round_bf16_array(r->block.expanded, h*m);
            memcpy(hc+(size_t)b*h*m, r->block.expanded, h*m*sizeof(float));
        }
    if (r->trace && !r->trace(r->trace_context,"ffn",layer,group,count,h)) goto fail;
    goto done;
fail:
    ok=0;r->poisoned=1;
    snprintf(r->error,sizeof(r->error),"prefill failed/cancelled at layer %d, committed position %d",layer,r->position);
done:
    free(input); free(post); free(comb); free(route_out); free(group);
    free(output); free(gate); free(up); free(routes); free(group_routes);
    free(ids); free(assignments); free(act); free(act_scale);
    return ok;
}
/* Finish causal attention for a layer before its token-independent MoE. Each
 * routed expert then handles every prompt chunk while its device weights remain
 * resident. Accumulating experts by increasing id matches upstream MoE.forward.
 * Only two extra prompt-sized hidden banks are retained; GEMMs stay chunked. */
static inline int dsv4_prefill_layer_prompt(dsv4_runtime *r,const int *tokens,int count,int chunk,int layer,float *hc,
    dsv4_prefill_progress_fn progress,void *opaque) {
    const size_t h=DSV4_ATTN_HIDDEN,m=DSV4_HC_MULT,n=(size_t)count,cap=(size_t)chunk;
    float *input=dsv4_prefill_alloc(n*h,sizeof(float)),*sum=calloc(n*h,sizeof(float)),
        *post=dsv4_prefill_alloc(n*m,sizeof(float)),*comb=dsv4_prefill_alloc(n*m*m,sizeof(float)),
        *routes=dsv4_prefill_alloc(n*DSV4_TOPK,sizeof(float)),
        *group=dsv4_prefill_alloc(cap*h,sizeof(float)),*output=dsv4_prefill_alloc(cap*h,sizeof(float)),
        *gate=dsv4_prefill_alloc(cap*DSV4_MOE_INTERMEDIATE,sizeof(float)),*up=dsv4_prefill_alloc(cap*DSV4_MOE_INTERMEDIATE,sizeof(float)),
        *group_routes=dsv4_prefill_alloc(cap,sizeof(float)),*logits=dsv4_prefill_alloc(cap*DSV4_EXPERTS,sizeof(float));
    int *ids=dsv4_prefill_alloc(n*DSV4_TOPK,sizeof(int)),*ordered=dsv4_prefill_alloc(n*DSV4_TOPK,sizeof(int)),
        *assigned=dsv4_prefill_alloc(cap,sizeof(int));
    size_t aw=h>DSV4_MOE_INTERMEDIATE?h:DSV4_MOE_INTERMEDIATE;
    uint8_t *act=dsv4_prefill_alloc(cap*aw,1),*ascale=dsv4_prefill_alloc((cap*aw+31)/32,1);
    dsv4_expert_entry *active=NULL;
    int ok=input && sum && post && comb && routes && group && output && gate && up && group_routes && logits && ids && ordered && assigned && act && ascale;
    double phase_start=dsv4_dense_now_seconds(),layer_start=phase_start;
    if (!ok) goto done;
    for (int pos=0;pos<count;pos+=chunk) {
        int take=count-pos;if (take>chunk) take=chunk;r->prefill_chunk_start=r->position+pos;
        if (!dsv4_attention_batch(r,layer,hc+(size_t)pos*h*m,take,progress,opaque)) goto fail;
    }
    r->last_attention_stage_seconds[layer]=dsv4_dense_now_seconds()-phase_start;
    phase_start=dsv4_dense_now_seconds();
    const float *fn,*scale,*base;const uint16_t *norm;
    if (!dsv4_block_bind_stage(r->dense,layer,"ffn",&fn,&scale,&base,&norm)) goto fail;
    for (int pos=0;pos<count;pos+=chunk) {
        int take=count-pos;if (take>chunk) take=chunk;r->prefill_chunk_start=r->position+pos;
        if (progress && !progress(opaque,r->position,take,layer,DSV4_RUNTIME_LAYERS,0)) goto fail;
#ifdef _OPENMP
#pragma omp parallel for if(take>=16) num_threads(6)
#endif
        for (int b=pos;b<pos+take;b++) {
            float *x=input+(size_t)b*h;
            dsv4_hc_forward_pre(hc+(size_t)b*h*m,fn,h,scale,base,1e-6f,x,post+(size_t)b*m,comb+(size_t)b*m*m,NULL);
            dsv4_round_bf16_array(x,h);dsv4_rmsnorm(x,x,norm,h,1e-6f);
        }
        if (!dsv4_dense_route_batch(r->dense,layer,tokens+pos,input+(size_t)pos*h,take,ids+(size_t)pos*DSV4_TOPK,routes+(size_t)pos*DSV4_TOPK,logits)) goto fail;
    }
    r->prefill_chunk_start=r->position;
    if (r->trace && !r->trace(r->trace_context,"ffn_input",layer,input,count,h)) goto fail;
    int offsets[DSV4_EXPERTS+1]={0},cursor[DSV4_EXPERTS];
    for (int b=0;b<count;b++) for (int k=0;k<DSV4_TOPK;k++) {
        int at=b*DSV4_TOPK+k,id=ids[at];if (id<0 || id>=DSV4_EXPERTS) goto fail;
        for (int previous=0;previous<k;previous++) if (ids[b*DSV4_TOPK+previous]==id) goto fail;
        offsets[id+1]++;
    }
    for (int e=0;e<DSV4_EXPERTS;e++) {offsets[e+1]+=offsets[e];cursor[e]=offsets[e];}
    for (int at=0;at<count*DSV4_TOPK;at++) ordered[cursor[ids[at]]++]=at;
    r->last_route_seconds[layer]=dsv4_dense_now_seconds()-phase_start;
    phase_start=dsv4_dense_now_seconds();
    /* Bounded asynchronous reads populate one layer's host expert window. */
    if (r->experts->prefill_entries) {
        int batch_ids[DSV4_TOPK],pending=0;
        for (int e=0;e<=DSV4_EXPERTS;e++) {
            if (e<DSV4_EXPERTS && offsets[e+1]>offsets[e]) batch_ids[pending++]=e;
            if (pending && (pending==DSV4_TOPK || e==DSV4_EXPERTS)) {
                dsv4_expert_entry *entries[DSV4_TOPK]={0};
                if (progress && !progress(opaque,r->position,count<chunk?count:chunk,layer,DSV4_RUNTIME_LAYERS,chunk+batch_ids[pending-1])) goto fail;
                if (!dsv4_expert_cache_acquire_many(r->experts,layer,batch_ids,pending,entries)) goto fail;
                for (int j=0;j<pending;j++) if (!dsv4_expert_cache_release(r->experts,entries[j])) goto fail;
                pending=0;
            }
        }
    }
    r->last_prefill_read_seconds[layer]=dsv4_dense_now_seconds()-phase_start;
    phase_start=dsv4_dense_now_seconds();
    for (int e=0;e<DSV4_EXPERTS;e++) {
        int at=offsets[e],end=offsets[e+1];if (at==end) continue;
        if (!dsv4_expert_cache_acquire_many(r->experts,layer,&e,1,&active)) goto fail;
        while (at<end) {
            /* FFN has no causal dependencies. Pack this expert's ordered
             * assignments across the prompt, while retaining the configured
             * matrix batch bound. This avoids underfilled per-token chunks.
             * Progress counts assigned tokens and gives their first position. */
            int first=ordered[at]/DSV4_TOPK,batch=end-at;if (batch>chunk) batch=chunk;
            for (int b=0;b<batch;b++) {
                int route=ordered[at++],token=route/DSV4_TOPK;
                assigned[b]=token;group_routes[b]=routes[route];
                memcpy(group+(size_t)b*h,input+(size_t)token*h,h*sizeof(float));
            }
            r->prefill_chunk_start=r->position+first;
            if (progress && !progress(opaque,r->position,batch,layer,DSV4_RUNTIME_LAYERS,batch+e)) goto fail;
            if (!dsv4_prefill_expert(r->experts,active,group,group_routes,output,batch,gate,up,act,ascale)) goto fail;
            for (int b=0;b<batch;b++) for (size_t d=0;d<h;d++) sum[(size_t)assigned[b]*h+d]+=output[(size_t)b*h+d];
        }
        if (!dsv4_expert_cache_release(r->experts,active)) goto fail;
        active=NULL;
    }
    r->last_routed_seconds[layer]=dsv4_dense_now_seconds()-phase_start;
    phase_start=dsv4_dense_now_seconds();
    for (int pos=0;pos<count;pos+=chunk) {
        int take=count-pos;if (take>chunk) take=chunk;r->prefill_chunk_start=r->position+pos;
        if (progress && !progress(opaque,r->position,take,layer,DSV4_RUNTIME_LAYERS,take+DSV4_EXPERTS)) goto fail;
        if (!dsv4_dense_shared_expert_batch(r->dense,layer,input+(size_t)pos*h,output,take,gate,up,act,ascale)) goto fail;
        for (int b=0;b<take;b++) {
            float *value=sum+(size_t)(pos+b)*h;
            for (size_t d=0;d<h;d++) value[d]=dsv4_round_bf16(value[d]+output[(size_t)b*h+d]);
            dsv4_hc_post(value,hc+(size_t)(pos+b)*h*m,h,post+(size_t)(pos+b)*m,comb+(size_t)(pos+b)*m*m,r->block.expanded);
            dsv4_round_bf16_array(r->block.expanded,h*m);memcpy(hc+(size_t)(pos+b)*h*m,r->block.expanded,h*m*sizeof(float));
        }
    }
    r->last_shared_seconds[layer]=dsv4_dense_now_seconds()-phase_start;
    r->last_layer_seconds[layer]=dsv4_dense_now_seconds()-layer_start;
    r->last_ffn_stage_seconds[layer]=r->last_layer_seconds[layer]-r->last_attention_stage_seconds[layer];
    r->prefill_chunk_start=r->position;
    if (r->trace && !r->trace(r->trace_context,"ffn",layer,sum,count,h)) goto fail;
    goto done;
fail:
    ok=0;
done:
    if (active) dsv4_expert_cache_release(r->experts,active);
    if (!ok) {r->poisoned=1;snprintf(r->error,sizeof(r->error),"whole-prompt layer %d failed/cancelled at committed position %d",layer,r->position);}
    free(input);free(sum);free(post);free(comb);free(routes);free(group);free(output);free(gate);free(up);free(group_routes);free(logits);
    free(ids);free(ordered);free(assigned);free(act);free(ascale);return ok;
}

static inline int dsv4_prefill_commit(dsv4_runtime *r,const int *tokens,int count,const float *hc,float *logits) {
    size_t h=DSV4_ATTN_HIDDEN,m=DSV4_HC_MULT;
    memcpy(r->hc,hc+(size_t)(count-1)*h*m,h*m*sizeof(float));
    if (logits) {
        const uint16_t *head=(const uint16_t*)dsv4_dense_named(r->dense,"head",".weight",DSV4_DTYPE_BF16,DSV4_VOCAB,h);
        if (!head || !dsv4_runtime_head(r->dense,r->hc,r->head_hidden) ||
            !dsv4_dense_linear_bf16(r->dense,logits,r->head_hidden,head,1,DSV4_VOCAB,h)) return 0;
    }
    memcpy(r->history+r->position,tokens,(size_t)count*sizeof(int));r->position+=count;
    return 1;
}
static inline int dsv4_runtime_prefill_chunk(dsv4_runtime *r,const int *tokens,int count,float *logits,
    dsv4_prefill_progress_fn progress,void *opaque) {
    if (!r || r->poisoned || !tokens || count<1 || count>2048 || count>r->context-r->position) return 0;
    for (int b=0;b<count;b++) if (tokens[b]<0 || tokens[b]>=DSV4_VOCAB) return 0;
    size_t width=(size_t)DSV4_ATTN_HIDDEN*DSV4_HC_MULT;
    float *hc=dsv4_prefill_alloc((size_t)count*width,sizeof(float));if (!hc) return 0;
    double step_started=dsv4_dense_now_seconds();
    int ok=1;
    for (int b=0;ok && b<count;b++) ok=dsv4_dense_embed_token(r->dense,tokens[b],hc+(size_t)b*width);
    r->prefill_chunk_start=r->position;
    for (int layer=0;ok && layer<DSV4_RUNTIME_LAYERS;layer++)
        ok=dsv4_prefill_layer_chunk(r,tokens,count,layer,hc,progress,opaque);
    if (ok) {
        double head_started=dsv4_dense_now_seconds();
        ok=dsv4_prefill_commit(r,tokens,count,hc,logits);
        if (count==1) r->decode_head_seconds+=dsv4_dense_now_seconds()-head_started;
    }
    if (count==1) { r->decode_steps++; r->decode_step_seconds+=dsv4_dense_now_seconds()-step_started; }
    if (!ok) r->poisoned=1;
    free(hc);return ok;
}
static inline uint64_t dsv4_prefill_bank_bytes(int tokens) {
    /* HC state plus bounded full-prompt FFN input/sum and routing metadata. */
    return tokens<1 || tokens>DSV4_REVIEW_INPUT_TOKENS ? 0 : (uint64_t)tokens*((DSV4_HC_MULT+2)*DSV4_ATTN_HIDDEN*sizeof(float)+
        (DSV4_HC_MULT+DSV4_HC_MULT*DSV4_HC_MULT+DSV4_TOPK)*sizeof(float)+2*DSV4_TOPK*sizeof(int));
}
/* Layer-major across the entire prompt, with bounded GEMM chunks. Keeping one
 * layer's experts avoids rereading the entire checkpoint for each 1024 tokens.
 * Partial layer state is never reused after cancellation or reported committed. */
static inline int dsv4_runtime_prefill_prompt(dsv4_runtime *r,const int *tokens,int count,int chunk,float *logits,
    dsv4_prefill_progress_fn progress,void *opaque) {
    if (!r || r->poisoned || !tokens || count<1 || count>DSV4_REVIEW_INPUT_TOKENS || chunk<1 || chunk>2048 || count>r->context-r->position) return 0;
    for (int b=0;b<count;b++) if (tokens[b]<0 || tokens[b]>=DSV4_VOCAB) return 0;
    size_t width=(size_t)DSV4_ATTN_HIDDEN*DSV4_HC_MULT;
    if (!dsv4_expert_prefill_begin(r->experts)) return 0;
    float *hc=dsv4_prefill_alloc((size_t)count*width,sizeof(float));
    int ok=hc!=NULL;
    for (int b=0;ok && b<count;b++) ok=dsv4_dense_embed_token(r->dense,tokens[b],hc+(size_t)b*width);
    for (int layer=0;ok && layer<DSV4_RUNTIME_LAYERS;layer++) {
        ok=dsv4_expert_prefill_layer(r->experts,layer);
        if (ok) ok=dsv4_prefill_layer_prompt(r,tokens,count,chunk,layer,hc,progress,opaque);
    }
    if (ok) ok=dsv4_prefill_commit(r,tokens,count,hc,logits);
    if (!ok) r->poisoned=1;
    free(hc);dsv4_expert_prefill_end(r->experts);return ok;
}
#endif
