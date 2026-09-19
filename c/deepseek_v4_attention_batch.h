#ifndef COLIB_DEEPSEEK_V4_ATTENTION_BATCH_H
#define COLIB_DEEPSEEK_V4_ATTENTION_BATCH_H
#include "deepseek_v4_runtime.h"
/* Preserve the pre-chunk ring, append raw KV in token order, and append the
 * immutable compressed prefix. Every selected index is captured at its causal
 * token before later ring writes occur. No future key becomes selectable. */
typedef struct {
    int start,count,filled,ratio,stride,rows;
    float *query,*kv;
    int *indices,*counts;
    const float *sink;
} dsv4_attention_collection;
static inline uint64_t dsv4_attention_collection_bytes(int count) {
    uint64_t compressed=(DSV4_MAX_CONTEXT+DSV4_INDEX_RATIO-1)/DSV4_INDEX_RATIO;
    uint64_t selected=DSV4_INDEX_TOPK;
    if (selected<(DSV4_MAX_CONTEXT+DSV4_COMPRESS_RATIO-1)/DSV4_COMPRESS_RATIO)
        selected=(DSV4_MAX_CONTEXT+DSV4_COMPRESS_RATIO-1)/DSV4_COMPRESS_RATIO;
    return ((uint64_t)DSV4_ATTN_WINDOW+count+compressed)*DSV4_ATTN_HEAD_DIM*sizeof(float)+
        (uint64_t)count*(DSV4_ATTN_WINDOW+selected+1)*sizeof(int);
}
static inline int dsv4_attention_collect(void *opaque,const float *query,const float *kv,
    const int *indices,int selected,const float *sink,int position) {
    dsv4_attention_collection *c=opaque;
    int row=position-c->start,window=DSV4_ATTN_WINDOW,dim=DSV4_ATTN_HEAD_DIM;
    if (row!=c->filled || row<0 || row>=c->count || selected<1 || selected>c->stride) return 0;
    memcpy(c->query+(size_t)row*DSV4_ATTN_HEADS*dim,query,(size_t)DSV4_ATTN_HEADS*dim*sizeof(float));
    memcpy(c->kv+(size_t)(window+row)*dim,kv+(size_t)(position%window)*dim,(size_t)dim*sizeof(float));
    if (c->ratio && (position+1)%c->ratio==0) {
        int compressed=(position+1)/c->ratio-1;
        memcpy(c->kv+(size_t)(window+c->count+compressed)*dim,kv+(size_t)(window+compressed)*dim,(size_t)dim*sizeof(float));
    }
    for (int i=0;i<selected;i++) {
        int id=indices[i],mapped=id;
        if (id>=0 && id<window) {
            int absolute=position-(position-id+window)%window;
            if (absolute<0) return 0;
            mapped=absolute>=c->start?window+absolute-c->start:id;
        } else if (id>=window) {
            if (!c->ratio || id-window>=(position+1)/c->ratio) return 0;
            mapped=id+c->count;
        }
        if (mapped>=c->rows) return 0;
        c->indices[(size_t)row*c->stride+i]=mapped;
    }
    c->counts[row]=selected;c->sink=sink;c->filled++;
    return 1;
}
/* Dense projections are independent across the chunk. The existing attention
 * steps consume their precomputed values in original token order and own all
 * causal ring/compressor/indexer transitions. Output projections follow them. */
static inline size_t dsv4_attention_batch_activation_width(void) {
    size_t width=DSV4_ATTN_HIDDEN,other=(size_t)DSV4_ATTN_O_GROUPS*DSV4_ATTN_O_RANK;
    if (width<other) width=other;
    other=(size_t)DSV4_ATTN_HEADS*DSV4_ATTN_HEAD_DIM/DSV4_ATTN_O_GROUPS;
    if (width<other) width=other;
    if (width<DSV4_ATTN_Q_RANK) width=DSV4_ATTN_Q_RANK;
    return width;
}
static inline uint64_t dsv4_attention_batch_bytes(int count) {
    uint64_t h=DSV4_ATTN_HIDDEN,q=(uint64_t)DSV4_ATTN_HEADS*DSV4_ATTN_HEAD_DIM;
    uint64_t index=(uint64_t)DSV4_INDEX_HEADS*DSV4_INDEX_DIM;
    uint64_t o=(uint64_t)DSV4_ATTN_O_GROUPS*DSV4_ATTN_O_RANK;
    uint64_t floats=2*h+DSV4_HC_MULT+DSV4_HC_MULT*DSV4_HC_MULT+2*DSV4_ATTN_Q_RANK+2*q+
        5*DSV4_ATTN_HEAD_DIM+index+4*DSV4_INDEX_DIM+DSV4_INDEX_HEADS+o+q/DSV4_ATTN_O_GROUPS+DSV4_ATTN_O_RANK;
    uint64_t a=dsv4_attention_batch_activation_width();
    return (uint64_t)count*(floats*sizeof(float)+a+(a+127)/128);
}
static inline int dsv4_batch_named_fp8(const dsv4_dense_arena *dense,const char *prefix,const char *projection,
    float *out,const float *input,int batch,int rows,int cols,uint8_t *act,uint8_t *scales) {
    char name[192];snprintf(name,sizeof(name),"%s.%s",prefix,projection);
    const uint8_t *weight,*scale;
    return dsv4_dense_fp8_pair(dense,name,rows,cols,&weight,&scale) &&
        dsv4_dense_linear_fp8(dense,out,input,weight,scale,batch,rows,cols,act,scales);
}
static inline int dsv4_batch_named_bf16(const dsv4_dense_arena *dense,const char *prefix,const char *projection,
    float *out,const float *input,int batch,int rows,int cols) {
    char name[192];snprintf(name,sizeof(name),"%s.%s",prefix,projection);
    const uint16_t *weight=(const uint16_t*)dsv4_dense_named(dense,name,".weight",DSV4_DTYPE_BF16,rows,cols);
    return weight && dsv4_dense_linear_bf16(dense,out,input,weight,batch,rows,cols);
}
static inline int dsv4_attention_batch(dsv4_runtime *r,int layer,float *hc,int count,
    int (*progress)(void *,int,int,int,int,int),void *opaque) {
    if (count<1 || count>2048) return 0;
    size_t h=DSV4_ATTN_HIDDEN,m=DSV4_HC_MULT,q=(size_t)DSV4_ATTN_HEADS*DSV4_ATTN_HEAD_DIM;
    size_t index=(size_t)DSV4_INDEX_HEADS*DSV4_INDEX_DIM,o=(size_t)DSV4_ATTN_O_GROUPS*DSV4_ATTN_O_RANK;
    size_t bytes=dsv4_attention_batch_bytes(count);unsigned char *bank=malloc(bytes);if (!bank) return 0;
    float *cursor=(float*)bank;
#define DSV4_BATCH_FIELD(name,width) float *name=cursor;cursor+=(size_t)count*(width)
    DSV4_BATCH_FIELD(input,h);DSV4_BATCH_FIELD(output,h);DSV4_BATCH_FIELD(post,m);DSV4_BATCH_FIELD(comb,m*m);
    DSV4_BATCH_FIELD(qa,DSV4_ATTN_Q_RANK);DSV4_BATCH_FIELD(qnorm,DSV4_ATTN_Q_RANK);
    DSV4_BATCH_FIELD(query,q);DSV4_BATCH_FIELD(context,q);DSV4_BATCH_FIELD(kv,DSV4_ATTN_HEAD_DIM);
    DSV4_BATCH_FIELD(ckv,2*DSV4_ATTN_HEAD_DIM);DSV4_BATCH_FIELD(cscore,2*DSV4_ATTN_HEAD_DIM);
    DSV4_BATCH_FIELD(iquery,index);DSV4_BATCH_FIELD(ickv,2*DSV4_INDEX_DIM);DSV4_BATCH_FIELD(icscore,2*DSV4_INDEX_DIM);
    DSV4_BATCH_FIELD(iweights,DSV4_INDEX_HEADS);DSV4_BATCH_FIELD(orank,o);
    DSV4_BATCH_FIELD(group,q/DSV4_ATTN_O_GROUPS);DSV4_BATCH_FIELD(group_out,DSV4_ATTN_O_RANK);
#undef DSV4_BATCH_FIELD
    uint8_t *act=(uint8_t*)cursor,*scales=act+(size_t)count*dsv4_attention_batch_activation_width();
    dsv4_runtime_layer *state=&r->layers[layer];
    dsv4_attention_scratch *work=state->mode==DSV4_LAYER_SLIDING?&r->block.attention:
        state->mode==DSV4_LAYER_OVERLAP?&r->overlap_scratch.attention:&r->nonoverlap_scratch.attention;
    dsv4_indexer_scratch *is=&r->overlap_scratch.indexer;
    const float *fn,*scale,*base;const uint16_t *norm;
    int ok=0;
    dsv4_attention_collection collection={0};
#ifdef COLI_CUDA
    if (r->dense->cuda && count>1 && (!getenv("DSV4_BATCH_ATTENTION") || strcmp(getenv("DSV4_BATCH_ATTENTION"),"0"))) {
        int ratio=state->mode==DSV4_LAYER_SLIDING?0:state->mode==DSV4_LAYER_OVERLAP?DSV4_INDEX_RATIO:DSV4_COMPRESS_RATIO;
        int start=r->prefill_chunk_start,compressed=ratio?(start+count)/ratio:0;
        int selected=state->mode==DSV4_LAYER_OVERLAP?DSV4_INDEX_TOPK:compressed;
        if (selected>compressed) selected=compressed;
        const float *cache=state->mode==DSV4_LAYER_SLIDING?state->attention.sliding.kv_cache:
            state->mode==DSV4_LAYER_OVERLAP?state->attention.overlap.kv_cache:state->attention.nonoverlap.kv_cache;
        collection=(dsv4_attention_collection){.start=start,.count=count,.ratio=ratio,
            .stride=DSV4_ATTN_WINDOW+selected,.rows=DSV4_ATTN_WINDOW+count+compressed,.query=query};
        collection.kv=calloc((size_t)collection.rows*DSV4_ATTN_HEAD_DIM,sizeof(float));
        collection.indices=calloc((size_t)count*collection.stride,sizeof(int));
        collection.counts=calloc(count,sizeof(int));
        if (!collection.kv || !collection.indices || !collection.counts) goto done;
        memcpy(collection.kv,cache,(size_t)DSV4_ATTN_WINDOW*DSV4_ATTN_HEAD_DIM*sizeof(float));
        if (ratio && start/ratio) memcpy(collection.kv+(size_t)(DSV4_ATTN_WINDOW+count)*DSV4_ATTN_HEAD_DIM,
            cache+(size_t)DSV4_ATTN_WINDOW*DSV4_ATTN_HEAD_DIM,(size_t)(start/ratio)*DSV4_ATTN_HEAD_DIM*sizeof(float));
        work->prefill_collect=dsv4_attention_collect;work->prefill_collect_context=&collection;
    }
#endif
    if (!dsv4_block_bind_stage(r->dense,layer,"attn",&fn,&scale,&base,&norm)) goto done;
    if (progress && !progress(opaque,r->position,count,layer,DSV4_RUNTIME_LAYERS,0)) goto done;
#ifdef _OPENMP
#pragma omp parallel for if(count>=16) num_threads(6)
#endif
    for (int b=0;b<count;b++) {
        float *x=input+(size_t)b*h;
        dsv4_hc_forward_pre(hc+(size_t)b*h*m,fn,h,scale,base,1e-6f,x,post+(size_t)b*m,comb+(size_t)b*m*m,NULL);
        dsv4_round_bf16_array(x,h);dsv4_rmsnorm(x,x,norm,h,1e-6f);
    }
    if (r->trace && !r->trace(r->trace_context,"attention_input",layer,input,count,h)) goto done;
    char lp[64],prefix[96];dsv4_dense_layer_prefix(lp,sizeof(lp),layer);snprintf(prefix,sizeof(prefix),"%s.attn",lp);
    if (!dsv4_batch_named_fp8(r->dense,prefix,"wq_a",qa,input,count,DSV4_ATTN_Q_RANK,h,act,scales)) goto done;
    const uint16_t *qn=(const uint16_t*)dsv4_attention_vector(r->dense,prefix,".q_norm.weight",DSV4_DTYPE_BF16,DSV4_ATTN_Q_RANK);
    if (!qn) goto done;
    for (int b=0;b<count;b++) dsv4_rmsnorm(qnorm+(size_t)b*DSV4_ATTN_Q_RANK,qa+(size_t)b*DSV4_ATTN_Q_RANK,qn,DSV4_ATTN_Q_RANK,1e-6f);
    if (!dsv4_batch_named_fp8(r->dense,prefix,"wq_b",query,qnorm,count,q,DSV4_ATTN_Q_RANK,act,scales) ||
        !dsv4_batch_named_fp8(r->dense,prefix,"wkv",kv,input,count,DSV4_ATTN_HEAD_DIM,h,act,scales)) goto done;
    if (r->trace && (!r->trace(r->trace_context,"qa",layer,qa,count,DSV4_ATTN_Q_RANK) ||
        !r->trace(r->trace_context,"qnorm",layer,qnorm,count,DSV4_ATTN_Q_RANK) ||
        !r->trace(r->trace_context,"query",layer,query,count,q) || !r->trace(r->trace_context,"kv",layer,kv,count,DSV4_ATTN_HEAD_DIM))) goto done;
    int cw=state->mode==DSV4_LAYER_OVERLAP?2*DSV4_ATTN_HEAD_DIM:DSV4_ATTN_HEAD_DIM;
    if (state->mode!=DSV4_LAYER_SLIDING &&
        (!dsv4_batch_named_bf16(r->dense,prefix,"compressor.wkv",ckv,input,count,cw,h) ||
         !dsv4_batch_named_bf16(r->dense,prefix,"compressor.wgate",cscore,input,count,cw,h))) goto done;
    if (state->mode==DSV4_LAYER_OVERLAP &&
        (!dsv4_batch_named_fp8(r->dense,prefix,"indexer.wq_b",iquery,qnorm,count,index,DSV4_ATTN_Q_RANK,act,scales) ||
         !dsv4_batch_named_bf16(r->dense,prefix,"indexer.compressor.wkv",ickv,input,count,2*DSV4_INDEX_DIM,h) ||
         !dsv4_batch_named_bf16(r->dense,prefix,"indexer.compressor.wgate",icscore,input,count,2*DSV4_INDEX_DIM,h) ||
         !dsv4_batch_named_bf16(r->dense,prefix,"indexer.weights_proj",iweights,input,count,DSV4_INDEX_HEADS,h))) goto done;
    for (int b=0;b<count;b++) {
        if (progress && !progress(opaque,r->position,count,layer,DSV4_RUNTIME_LAYERS,b)) goto done;
        work->prefill_q_rank=qa+(size_t)b*DSV4_ATTN_Q_RANK;work->prefill_query=query+(size_t)b*q;
        work->prefill_kv=kv+(size_t)b*DSV4_ATTN_HEAD_DIM;work->prefill_context=context+(size_t)b*q;
        work->prefill_compressor_kv=ckv+(size_t)b*cw;work->prefill_compressor_score=cscore+(size_t)b*cw;
        is->prefill_query=iquery+(size_t)b*index;is->prefill_compressor_kv=ickv+(size_t)b*2*DSV4_INDEX_DIM;
        is->prefill_compressor_score=icscore+(size_t)b*2*DSV4_INDEX_DIM;is->prefill_head_weights=iweights+(size_t)b*DSV4_INDEX_HEADS;
        int step=state->mode==DSV4_LAYER_SLIDING?
            dsv4_attention_decode_sliding(r->dense,layer,input+(size_t)b*h,&state->attention.sliding,work,output):
            state->mode==DSV4_LAYER_OVERLAP?
            dsv4_attention_decode_compressed_overlap(r->dense,layer,input+(size_t)b*h,&state->attention.overlap,&r->overlap_scratch,output):
            dsv4_attention_decode_compressed_nonoverlap(r->dense,layer,input+(size_t)b*h,&state->attention.nonoverlap,&r->nonoverlap_scratch,output);
        if (!step) goto done;
    }
#ifdef COLI_CUDA
    if (collection.kv) {
        if (collection.filled!=count || !dsv4_dense_sparse_attention_batch(r->dense,context,query,collection.kv,
            collection.indices,collection.counts,collection.sink,count,DSV4_ATTN_HEADS,DSV4_ATTN_HEAD_DIM,collection.rows,collection.stride)) goto done;
#ifdef _OPENMP
#pragma omp parallel for if(count>=16) num_threads(6)
#endif
        for (int b=0;b<count;b++) {
            for (int head=0;head<DSV4_ATTN_HEADS;head++)
                dsv4_rope(context+(size_t)b*q+(size_t)head*DSV4_ATTN_HEAD_DIM+DSV4_ATTN_HEAD_DIM-DSV4_ATTN_ROPE_DIM,
                    DSV4_ATTN_ROPE_DIM,collection.start+b,collection.ratio?DSV4_ORIGINAL_CONTEXT:0,
                    collection.ratio?DSV4_COMPRESS_ROPE_THETA:10000.0f,collection.ratio?DSV4_ROPE_FACTOR:1.0f,32,1,1);
            dsv4_round_bf16_array(context+(size_t)b*q,q);
        }
    }
#endif
    if (r->trace && !r->trace(r->trace_context,"context",layer,context,count,q)) goto done;
    char projection[128];snprintf(projection,sizeof(projection),"%s.wo_a",prefix);
    const uint8_t *w,*ws;size_t gw=q/DSV4_ATTN_O_GROUPS;
    if (!dsv4_dense_fp8_pair(r->dense,projection,o,gw,&w,&ws)) goto done;
    for (int g=0;g<DSV4_ATTN_O_GROUPS;g++) {
        for (int b=0;b<count;b++) memcpy(group+(size_t)b*gw,context+(size_t)b*q+g*gw,gw*sizeof(float));
        if (!dsv4_dense_fp8_weight_bf16(r->dense,group_out,group,w+(size_t)g*DSV4_ATTN_O_RANK*gw,
            ws+(size_t)g*((DSV4_ATTN_O_RANK+127)/128)*((gw+127)/128),count,DSV4_ATTN_O_RANK,gw)) goto done;
        for (int b=0;b<count;b++) memcpy(orank+(size_t)b*o+(size_t)g*DSV4_ATTN_O_RANK,group_out+(size_t)b*DSV4_ATTN_O_RANK,DSV4_ATTN_O_RANK*sizeof(float));
    }
    if (!dsv4_batch_named_fp8(r->dense,prefix,"wo_b",output,orank,count,h,o,act,scales)) goto done;
    if (r->trace && (!r->trace(r->trace_context,"orank",layer,orank,count,o) || !r->trace(r->trace_context,"attention",layer,output,count,h))) goto done;
    for (int b=0;b<count;b++) {
        dsv4_hc_post(output+(size_t)b*h,hc+(size_t)b*h*m,h,post+(size_t)b*m,comb+(size_t)b*m*m,r->block.expanded);
        dsv4_round_bf16_array(r->block.expanded,h*m);memcpy(hc+(size_t)b*h*m,r->block.expanded,h*m*sizeof(float));
    }
    if (r->trace && !r->trace(r->trace_context,"attention_hc",layer,hc,count,h*m)) goto done;
    ok=1;
done:
    work->prefill_q_rank=work->prefill_query=work->prefill_kv=NULL;
    work->prefill_compressor_kv=work->prefill_compressor_score=NULL;work->prefill_context=NULL;
    is->prefill_query=is->prefill_compressor_kv=is->prefill_compressor_score=is->prefill_head_weights=NULL;
    work->prefill_collect=NULL;work->prefill_collect_context=NULL;
    free(collection.kv);free(collection.indices);free(collection.counts);
    free(bank);return ok;
}
#endif
