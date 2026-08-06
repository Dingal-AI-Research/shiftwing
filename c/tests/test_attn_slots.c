#define QWEN_NO_MAIN
#include "../qwen.c"

int main(void){
    setenv("MTP","0",1);setenv("KV16","0",1);setenv("EXPERT_RAM","4",1);setenv("PREFETCH_THREADS","0",1);
    static Model m;model_init(&m,"qwen_tiny_i4");Layer*l=NULL;
    for(int i=0;i<m.c.n_layers;i++)if(m.layer[i].type==LT_FULL){l=&m.layer[i];break;}
    if(!l){fprintf(stderr,"tiny model has no full-attention layer\n");return 1;}
    enum{B=3};Cfg*c=&m.c;AttnW*w=&l->attn;int H=c->hidden,kvrows=c->n_kv_heads*c->head_dim,S=m.max_seq;
    size_t ks=(size_t)S*kvrows;float*x=falloc((int64_t)B*H),*batched=falloc((int64_t)B*H),*reference=falloc((int64_t)B*H);
    float*k=falloc((int64_t)B*ks),*v=falloc((int64_t)B*ks),*k0=falloc((int64_t)B*ks),*v0=falloc((int64_t)B*ks);
    int pos[B]={3,5,7};
    for(int64_t i=0;i<(int64_t)B*H;i++)x[i]=sinf((float)(i+1)*0.029f)*0.2f;
    for(size_t i=0;i<(size_t)B*ks;i++){k[i]=sinf((float)(i+7)*0.011f)*0.04f;v[i]=cosf((float)(i+9)*0.015f)*0.04f;}
    memcpy(k0,k,(size_t)B*ks*sizeof(float));memcpy(v0,v,(size_t)B*ks*sizeof(float));
    attn_forward_slot_batch(&m,l,x,batched,B,NULL,pos,NULL,B,k,v,NULL,NULL,S);
    float maxdiff=0.f,kdiff=0.f,vdiff=0.f;
    for(int row=0;row<B;row++){
        memcpy(w->k_cache,k0+(size_t)row*ks,ks*sizeof(float));memcpy(w->v_cache,v0+(size_t)row*ks,ks*sizeof(float));m.pos=pos[row];
        attn_forward(&m,l,x+(int64_t)row*H,reference+(int64_t)row*H);
        for(int i=0;i<H;i++){float d=fabsf(batched[(int64_t)row*H+i]-reference[(int64_t)row*H+i]);if(d>maxdiff)maxdiff=d;}
        for(size_t i=0;i<ks;i++){float d=fabsf(k[(size_t)row*ks+i]-w->k_cache[i]);if(d>kdiff)kdiff=d;d=fabsf(v[(size_t)row*ks+i]-w->v_cache[i]);if(d>vdiff)vdiff=d;}
    }
    printf("resident GQA slot batch: rows=%d output=%.3g key=%.3g value=%.3g\n",B,maxdiff,kdiff,vdiff);
    free(x);free(batched);free(reference);free(k);free(v);free(k0);free(v0);
    return(maxdiff>1e-6f||kdiff>1e-7f||vdiff>1e-7f)?1:0;
}
