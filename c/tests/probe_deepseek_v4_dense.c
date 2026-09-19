/* Small released-record probe, isolated from complete model startup. */
#include "../deepseek_v4_dense.h"
int main(int argc,char **argv) {
    if (argc!=6) return 2;
    int batch=atoi(argv[3]);if (batch<1 || batch>2048) return 2;
    dsv4_store store;dsv4_dense_arena dense;
    if (!dsv4_store_init(&store,argv[1]) || !dsv4_store_require_integrity(&store) || !dsv4_dense_arena_init(&dense,&store,0,0)) return 2;
#ifdef COLI_CUDA
    ColiCuda *cuda=NULL;
    if (getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA")))
        if (coli_cuda_create(&cuda,0) || !dsv4_dense_arena_enable_cuda(&dense,cuda)) return 2;
#endif
    char name[256];snprintf(name,sizeof(name),"%s.weight",argv[2]);
    const dsv4_tensor_desc *desc=NULL;const void *weight=dsv4_dense_find(&dense,name,&desc);
    if (!weight || desc->rank!=2) return 2;
    int rows=desc->shape[0],cols=desc->shape[1];
    float *input=malloc((size_t)batch*cols*sizeof(float)),*out=malloc((size_t)batch*rows*sizeof(float));
    uint8_t *act=malloc((size_t)batch*cols),*scale=malloc((size_t)batch*((cols+127)/128));
    FILE *f=fopen(argv[4],"rb");
    if (!f || !input || !out || !act || !scale || fread(input,sizeof(float),(size_t)batch*cols,f)!=(size_t)batch*cols) return 2;fclose(f);
    double start=dsv4_dense_now_seconds();int ok=0;
    if (desc->dtype==DSV4_DTYPE_BF16) ok=dsv4_dense_linear_bf16(&dense,out,input,weight,batch,rows,cols);
    else {
        const uint8_t *w,*ws;
        ok=dsv4_dense_fp8_pair(&dense,argv[2],rows,cols,&w,&ws) &&
            (strstr(argv[2],".wo_a") ? dsv4_dense_fp8_weight_bf16(&dense,out,input,w,ws,batch,rows,cols) : dsv4_dense_linear_fp8(&dense,out,input,w,ws,batch,rows,cols,act,scale));
    }
    if (!ok) return 2;
    printf("DENSE_PROBE batch=%d rows=%d cols=%d seconds=%.6f\n",batch,rows,cols,dsv4_dense_now_seconds()-start);
    f=fopen(argv[5],"wb");if (!f || fwrite(out,sizeof(float),(size_t)batch*rows,f)!=(size_t)batch*rows || fclose(f)) return 2;
    dsv4_dense_arena_close(&dense);dsv4_store_close(&store);
#ifdef COLI_CUDA
    if (cuda) coli_cuda_destroy(cuda);
#endif
    free(input);free(out);free(act);free(scale);return 0;
}
