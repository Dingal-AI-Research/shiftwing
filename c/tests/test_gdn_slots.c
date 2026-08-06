#define QWEN_NO_MAIN
#include "../qwen.c"

int main(void){
    setenv("MTP","0",1);setenv("EXPERT_RAM","4",1);setenv("PREFETCH_THREADS","0",1);
    static Model m;model_init(&m,"qwen_tiny_i4");Layer*l=NULL;
    for(int i=0;i<m.c.n_layers;i++)if(m.layer[i].type==LT_LINEAR){l=&m.layer[i];break;}
    if(!l){fprintf(stderr,"tiny model has no GDN layer\n");return 1;}
    enum{B=3};Cfg*c=&m.c;GdnW*w=&l->gdn;int H=c->hidden,kh=c->lin_k_heads,vh=c->lin_v_heads;
    int dk=c->lin_k_dim,dv=c->lin_v_dim,cd=2*kh*dk+vh*dv,K=c->conv_kernel;size_t cs=(size_t)K*cd,ss=(size_t)vh*dk*dv;
    float*x=falloc((int64_t)B*H),*batched=falloc((int64_t)B*H),*reference=falloc((int64_t)B*H);
    float*conv=falloc((int64_t)B*cs),*state=falloc((int64_t)B*ss),*conv0=falloc((int64_t)B*cs),*state0=falloc((int64_t)B*ss);
    int pos[B]={3,5,7};
    for(int64_t i=0;i<(int64_t)B*H;i++)x[i]=sinf((float)(i+1)*0.031f)*0.2f;
    for(size_t i=0;i<(size_t)B*cs;i++)conv[i]=sinf((float)(i+3)*0.017f)*0.03f;
    for(size_t i=0;i<(size_t)B*ss;i++)state[i]=cosf((float)(i+5)*0.013f)*0.02f;
    memcpy(conv0,conv,(size_t)B*cs*sizeof(float));memcpy(state0,state,(size_t)B*ss*sizeof(float));
    gdn_forward_slot_batch(&m,l,x,batched,B,NULL,pos,NULL,B,conv,state);
    float maxdiff=0.f,state_diff=0.f,conv_diff=0.f;
    for(int row=0;row<B;row++){
        memcpy(w->conv_state,conv0+(size_t)row*cs,cs*sizeof(float));memcpy(w->state,state0+(size_t)row*ss,ss*sizeof(float));m.pos=pos[row];
        gdn_forward(&m,l,x+(int64_t)row*H,reference+(int64_t)row*H);
        for(int i=0;i<H;i++){float d=fabsf(batched[(int64_t)row*H+i]-reference[(int64_t)row*H+i]);if(d>maxdiff)maxdiff=d;}
        for(size_t i=0;i<ss;i++){float d=fabsf(state[(size_t)row*ss+i]-w->state[i]);if(d>state_diff)state_diff=d;}
        for(size_t i=0;i<cs;i++){float d=fabsf(conv[(size_t)row*cs+i]-w->conv_state[i]);if(d>conv_diff)conv_diff=d;}
    }
    printf("resident GDN slot batch: rows=%d output=%.3g state=%.3g conv=%.3g\n",B,maxdiff,state_diff,conv_diff);
    free(x);free(batched);free(reference);free(conv);free(state);free(conv0);free(state0);
    return(maxdiff>1e-6f||state_diff>1e-6f||conv_diff>1e-7f)?1:0;
}
