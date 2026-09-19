/* Bounded released-weight expert probe: no dense arena or model inference. */
#include "../deepseek_v4_prefill.h"
int main(int argc,char **argv) {
    if (argc!=7) { fprintf(stderr,"usage: probe SNAP LAYER EXPERT BATCH INPUT OUTPUT\n");return 2; }
    int layer=atoi(argv[2]),expert=atoi(argv[3]),batch=atoi(argv[4]);
    if (layer<0 || layer>=DSV4_BASE_LAYERS || expert<0 || expert>=DSV4_EXPERTS || batch<1 || batch>2048) return 2;
    dsv4_store store;
    if (!dsv4_store_init(&store,argv[1]) || !dsv4_store_require_integrity(&store)) { fprintf(stderr,"%s\n",store.error);return 2; }
    dsv4_expert_cache cache;
    if (!dsv4_expert_cache_init(&cache,&store,DSV4_TOPK)) return 2;
#ifdef COLI_CUDA
    ColiCuda *cuda=NULL;
    if (getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))) {
        if (coli_cuda_create(&cuda,0) || !dsv4_expert_cache_enable_cuda(&cache,cuda,DSV4_TOPK*dsv4_expert_payload_bytes())) return 2;
    }
#endif
    size_t input_count=(size_t)batch*DSV4_EXPERT_HIDDEN,inter=(size_t)batch*DSV4_MOE_INTERMEDIATE;
    float *input=malloc(input_count*sizeof(float)),*output=malloc(input_count*sizeof(float)),*routes=malloc(batch*sizeof(float));
    float *gate=malloc(inter*sizeof(float)),*up=malloc(inter*sizeof(float));
    uint8_t *act=malloc(input_count),*scales=malloc((input_count+31)/32);
    if (!input || !output || !routes || !gate || !up || !act || !scales) return 2;
    FILE *file=fopen(argv[5],"rb");
    if (!file || fread(input,sizeof(float),input_count,file)!=input_count || fread(routes,sizeof(float),batch,file)!=(size_t)batch || fgetc(file)!=EOF) return 2;
    fclose(file);
    dsv4_expert_entry *entry=NULL;
    if (!dsv4_expert_cache_acquire_many(&cache,layer,&expert,1,&entry)) { fprintf(stderr,"%s\n",cache.error);return 2; }
    double start=dsv4_dense_now_seconds();
    int ok=dsv4_prefill_expert(&cache,entry,input,routes,output,batch,gate,up,act,scales);
    double seconds=dsv4_dense_now_seconds()-start;
    if (!ok) { fprintf(stderr,"%s\n",cache.error);return 2; }
    file=fopen(argv[6],"wb");if (!file || fwrite(output,sizeof(float),input_count,file)!=input_count || fclose(file)) return 2;
    printf("DSV4_EXPERT_PROBE layer=%d expert=%d batch=%d seconds=%.6f read_bytes=%llu\n",layer,expert,batch,seconds,(unsigned long long)store.raw.read_bytes);
    dsv4_expert_cache_release(&cache,entry);dsv4_expert_cache_close(&cache);dsv4_store_close(&store);
#ifdef COLI_CUDA
    if (cuda) coli_cuda_destroy(cuda);
#endif
    free(input);free(output);free(routes);free(gate);free(up);free(act);free(scales);return 0;
}
