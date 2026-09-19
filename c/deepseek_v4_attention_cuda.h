#ifndef COLIB_DEEPSEEK_V4_ATTENTION_CUDA_H
#define COLIB_DEEPSEEK_V4_ATTENTION_CUDA_H
#include "deepseek_v4_dense.h"
#ifdef COLI_CUDA
/* One layer cache per kind, reused between chunks. New request/layer or any
 * nonconsecutive position reuploads visible state; no stale GPU prefix reuse. */
static inline int dsv4_attention_upload_cache(dsv4_dense_arena *d,const float *cache,
    int dim,int position,int ring,int ratio,int capacity,int indexer) {
    void **buffer=indexer?&d->cuda_index_cache:&d->cuda_attention_cache;
    size_t *cap=indexer?&d->cuda_index_cache_cap:&d->cuda_attention_cache_cap;
    const float **owner=indexer?&d->cuda_index_owner:&d->cuda_attention_owner;
    int *last=indexer?&d->cuda_index_position:&d->cuda_attention_position;
    size_t row=(size_t)dim*sizeof(float);
    if (!dsv4_dense_cuda_buffer(d,buffer,cap,(size_t)capacity*row,"CUDA attention cache")) return 0;
    int visible=ratio?(position+1)/ratio:0;
    if (*owner!=cache || *last+1!=position || position==0) {
        if (coli_cuda_upload(d->cuda,*buffer,cache,(size_t)(ring+visible)*row)) return 0;
    } else {
        if (ring && coli_cuda_upload(d->cuda,(char*)*buffer+(size_t)(position%ring)*row,cache+(size_t)(position%ring)*dim,row)) return 0;
        if (ratio && (position+1)%ratio==0 && coli_cuda_upload(d->cuda,(char*)*buffer+(size_t)(ring+visible-1)*row,cache+(size_t)(ring+visible-1)*dim,row)) return 0;
    }
    *owner=cache;*last=position;return 1;
}
static inline int dsv4_attention_cuda_workspace(dsv4_dense_arena *d,int heads,int dim,int selected,int indexer) {
    return dsv4_dense_cuda_buffer(d,&d->cuda_attention_query,&d->cuda_attention_query_cap,(size_t)heads*dim*sizeof(float),"CUDA attention query") &&
        dsv4_dense_cuda_buffer(d,&d->cuda_attention_out,&d->cuda_attention_out_cap,(size_t)(indexer?selected:heads*dim)*sizeof(float),"CUDA attention output") &&
        dsv4_dense_cuda_buffer(d,&d->cuda_attention_indices,&d->cuda_attention_indices_cap,(size_t)selected*sizeof(int),"CUDA attention indices") &&
        dsv4_dense_cuda_buffer(d,&d->cuda_attention_aux,&d->cuda_attention_aux_cap,(size_t)heads*sizeof(float),"CUDA attention auxiliary");
}
#endif
static inline int dsv4_dense_sparse_attention(const dsv4_dense_arena *view,float *out,const float *query,
    const float *kv,int heads,int dim,const int *indices,int selected,const float *sink,float scale,
    int position,int ring,int ratio,int capacity) {
#ifdef COLI_CUDA
    if (view->cuda) {
        dsv4_dense_arena *d=(dsv4_dense_arena*)view;pthread_mutex_lock(&d->cuda_lock);
        int ok=dsv4_attention_upload_cache(d,kv,dim,position,ring,ratio,capacity,0) &&
            dsv4_attention_cuda_workspace(d,heads,dim,selected,0) &&
            !coli_cuda_upload(d->cuda,d->cuda_attention_query,query,(size_t)heads*dim*sizeof(float)) &&
            !coli_cuda_upload(d->cuda,d->cuda_attention_indices,indices,(size_t)selected*sizeof(int)) &&
            !coli_cuda_upload(d->cuda,d->cuda_attention_aux,sink,(size_t)heads*sizeof(float)) &&
            !coli_cuda_dsv4_sparse_attention(d->cuda,d->cuda_attention_out,d->cuda_attention_query,d->cuda_attention_cache,d->cuda_attention_indices,d->cuda_attention_aux,heads,dim,selected,scale) &&
            !coli_cuda_download(d->cuda,out,d->cuda_attention_out,(size_t)heads*dim*sizeof(float)) && !coli_cuda_sync(d->cuda);
        if (!ok) {d->cuda_attention_owner=NULL;snprintf(d->error,sizeof(d->error),"CUDA sparse attention failed: %.180s",coli_cuda_last_error());}
        pthread_mutex_unlock(&d->cuda_lock);return ok;
    }
#endif
    dsv4_sparse_attention(out,query,kv,heads,dim,indices,selected,sink,scale);return 1;
}
#ifdef COLI_CUDA
static inline int dsv4_dense_sparse_attention_batch(dsv4_dense_arena *d,float *out,const float *query,
    const float *kv,const int *indices,const int *counts,const float *sink,
    int batch,int heads,int dim,int rows,int stride) {
    size_t vectors=(size_t)batch*heads*dim*sizeof(float),ids=(size_t)batch*stride*sizeof(int);
    size_t auxiliary=(size_t)heads*sizeof(float)+(size_t)batch*sizeof(int);
    pthread_mutex_lock(&d->cuda_lock);
    /* Linear chunk KV replaces the rolling layout. Force the next scalar call
     * to upload its own causal state, even when this allocation is reused. */
    d->cuda_attention_owner=NULL;
    int ok=dsv4_dense_cuda_buffer(d,&d->cuda_attention_cache,&d->cuda_attention_cache_cap,(size_t)rows*dim*sizeof(float),"CUDA chunk KV") &&
        dsv4_dense_cuda_buffer(d,&d->cuda_attention_query,&d->cuda_attention_query_cap,vectors,"CUDA chunk queries") &&
        dsv4_dense_cuda_buffer(d,&d->cuda_attention_out,&d->cuda_attention_out_cap,vectors,"CUDA chunk context") &&
        dsv4_dense_cuda_buffer(d,&d->cuda_attention_indices,&d->cuda_attention_indices_cap,ids,"CUDA chunk selections") &&
        dsv4_dense_cuda_buffer(d,&d->cuda_attention_aux,&d->cuda_attention_aux_cap,auxiliary,"CUDA chunk counts") &&
        !coli_cuda_upload(d->cuda,d->cuda_attention_cache,kv,(size_t)rows*dim*sizeof(float)) &&
        !coli_cuda_upload(d->cuda,d->cuda_attention_query,query,vectors) &&
        !coli_cuda_upload(d->cuda,d->cuda_attention_indices,indices,ids) &&
        !coli_cuda_upload(d->cuda,d->cuda_attention_aux,sink,(size_t)heads*sizeof(float)) &&
        !coli_cuda_upload(d->cuda,(char*)d->cuda_attention_aux+(size_t)heads*sizeof(float),counts,(size_t)batch*sizeof(int)) &&
        !coli_cuda_dsv4_sparse_attention_batch(d->cuda,d->cuda_attention_out,d->cuda_attention_query,d->cuda_attention_cache,
            d->cuda_attention_indices,(int*)((char*)d->cuda_attention_aux+(size_t)heads*sizeof(float)),d->cuda_attention_aux,
            batch,heads,dim,stride,1.0f/sqrtf((float)dim)) &&
        !coli_cuda_download(d->cuda,out,d->cuda_attention_out,vectors) && !coli_cuda_sync(d->cuda);
    if (!ok) snprintf(d->error,sizeof(d->error),"CUDA chunk attention failed: %.180s",coli_cuda_last_error());
    pthread_mutex_unlock(&d->cuda_lock);
    return ok;
}
#endif
static inline int dsv4_dense_indexer_topk(const dsv4_dense_arena *view,const float *query,const float *kv,
    const float *weights,int heads,int dim,int visible,int topk,int offset,float *scores,int *indices,
    int position,int ratio,int capacity) {
#ifdef COLI_CUDA
    if (view->cuda) {
        dsv4_dense_arena *d=(dsv4_dense_arena*)view;pthread_mutex_lock(&d->cuda_lock);
        int ok=dsv4_attention_upload_cache(d,kv,dim,position,0,ratio,capacity,1) &&
            dsv4_attention_cuda_workspace(d,heads,dim,visible,1) &&
            !coli_cuda_upload(d->cuda,d->cuda_attention_query,query,(size_t)heads*dim*sizeof(float)) &&
            !coli_cuda_upload(d->cuda,d->cuda_attention_aux,weights,(size_t)heads*sizeof(float)) &&
            !coli_cuda_dsv4_index_scores(d->cuda,d->cuda_attention_out,d->cuda_attention_query,d->cuda_index_cache,d->cuda_attention_aux,heads,dim,visible) &&
            !coli_cuda_download(d->cuda,scores,d->cuda_attention_out,(size_t)visible*sizeof(float)) && !coli_cuda_sync(d->cuda);
        if (!ok) {d->cuda_index_owner=NULL;snprintf(d->error,sizeof(d->error),"CUDA index scoring failed: %.180s",coli_cuda_last_error());}
        pthread_mutex_unlock(&d->cuda_lock);
        return ok?dsv4_indexer_select(scores,visible,topk,offset,indices):0;
    }
#endif
    return dsv4_indexer_topk(query,kv,weights,heads,dim,visible,topk,offset,scores,indices);
}
#endif
