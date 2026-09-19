#include "../deepseek_v4.h"
#include "../json.h"
#ifdef COLI_CUDA
#include "../backend_cuda.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#define CHECK(c) do {if (!(c)) {fprintf(stderr,"sparse:%d: %s\n",__LINE__,#c);return 1;}} while(0)
int main(void) {
    FILE *f=fopen("tests/fixtures/deepseek_v4_sparse.json","rb");
    if (!f) f=fopen("c/tests/fixtures/deepseek_v4_sparse.json","rb");
    CHECK(f);char text[16384];size_t n=fread(text,1,sizeof(text)-1,f);fclose(f);text[n]=0;
    jval *fixture=json_parse(text,NULL);CHECK(fixture);
    CHECK(!strcmp(json_get(fixture,"reference_kernel_sha256")->str,"59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2"));
    float q[16],kv[137*8],sink[2]={0,-1},actual[16];int ids[137];
    for (int i=0;i<16;i++) q[i]=((i*7)%23-11)/8.0f;
    for (int t=0;t<137;t++) {
        ids[t]=t%13 || !t?t:-1;
        for (int d=0;d<8;d++) kv[t*8+d]=((t*13+d*3)%37-18)/16.0f*(1+t/64);
    }
#ifdef COLI_CUDA
    ColiCuda *cuda=NULL;void *dq,*dkv,*ds,*di,*dout;
    CHECK(!coli_cuda_create(&cuda,0));
#define UPLOAD(dst,src) CHECK(!coli_cuda_malloc(cuda,&dst,sizeof(src)));CHECK(!coli_cuda_upload(cuda,dst,src,sizeof(src)))
    UPLOAD(dq,q);UPLOAD(dkv,kv);UPLOAD(ds,sink);UPLOAD(di,ids);
    CHECK(!coli_cuda_malloc(cuda,&dout,sizeof(actual)));
#endif
    jval *cases=json_get(fixture,"cases");CHECK(cases && cases->len==7);
    for (int c=0;c<cases->len;c++) {
        jval *item=cases->kids[c],*expected=json_get(item,"output");int selected=(int)json_get(item,"selected")->num;
        dsv4_sparse_attention(actual,q,kv,2,8,ids,selected,sink,.125f);
        for (int i=0;i<16;i++) CHECK(actual[i]==(float)expected->kids[i]->num);
#ifdef COLI_CUDA
        CHECK(!coli_cuda_dsv4_sparse_attention(cuda,dout,dq,dkv,di,ds,2,8,selected,.125f));
        CHECK(!coli_cuda_download(cuda,actual,dout,sizeof(actual)));
        for (int i=0;i<16;i++) CHECK(actual[i]==(float)expected->kids[i]->num);
#endif
    }
#ifdef COLI_CUDA
    /* One launch with different causal row lengths; unused row tails must not
     * enter either the dot products or the online-softmax denominator. */
    float queries[7*16],batch_out[7*16];int selections[7*137],counts[7];
    for (int c=0;c<7;c++) {
        memcpy(queries+c*16,q,sizeof(q));
        counts[c]=(int)json_get(cases->kids[c],"selected")->num;
        for (int t=0;t<137;t++) selections[c*137+t]=t<counts[c]?ids[t]:136;
    }
    void *bq,*bo,*bi,*bc;UPLOAD(bq,queries);UPLOAD(bi,selections);UPLOAD(bc,counts);
    CHECK(!coli_cuda_malloc(cuda,&bo,sizeof(batch_out)));
    CHECK(!coli_cuda_dsv4_sparse_attention_batch(cuda,bo,bq,dkv,bi,bc,ds,7,2,8,137,.125f));
    CHECK(!coli_cuda_download(cuda,batch_out,bo,sizeof(batch_out)));
    for (int c=0;c<7;c++) for (int i=0;i<16;i++)
        CHECK(batch_out[c*16+i]==(float)json_get(cases->kids[c],"output")->kids[i]->num);
    coli_cuda_free(cuda,bq);coli_cuda_free(cuda,bo);coli_cuda_free(cuda,bi);coli_cuda_free(cuda,bc);
    coli_cuda_free(cuda,dq);coli_cuda_free(cuda,dkv);coli_cuda_free(cuda,ds);coli_cuda_free(cuda,di);coli_cuda_free(cuda,dout);coli_cuda_destroy(cuda);
#endif
    puts("DeepSeek sparse attention matches independent pinned-kernel equations at all 64-key boundaries");return 0;
}
