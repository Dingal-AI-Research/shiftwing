#define QWEN_NO_MAIN
#include "../qwen.c"

int main(void){
    setenv("MTP","0",1);setenv("KV16","0",1);setenv("EXPERT_RAM","4",1);setenv("PREFETCH_THREADS","0",1);
    static Model m;model_init(&m,"qwen_tiny_i4");Layer*l=NULL;
    for(int i=0;i<m.c.n_layers;i++)if(m.layer[i].type==LT_LINEAR){l=&m.layer[i];break;}
    if(!l){fprintf(stderr,"tiny model has no GDN layer\n");return 1;}
    enum{B=2};Cfg*c=&m.c;GdnW*w=&l->gdn;int H=c->hidden,cd=2*c->lin_k_heads*c->lin_k_dim+c->lin_v_heads*c->lin_v_dim;
    size_t cs=(size_t)c->conv_kernel*cd,ss=(size_t)c->lin_v_heads*c->lin_k_dim*c->lin_v_dim;
    float*x=falloc((int64_t)B*H),*ref=falloc((int64_t)B*H),*conv=falloc((int64_t)B*cs),*state=falloc((int64_t)B*ss),*conv0=falloc((int64_t)B*cs),*state0=falloc((int64_t)B*ss);
    int pos[B]={3,5};
    for(int64_t i=0;i<(int64_t)B*H;i++)x[i]=sinf((float)(i+11)*0.023f)*0.15f;
    for(size_t i=0;i<(size_t)B*cs;i++)conv[i]=sinf((float)(i+3)*0.019f)*0.02f;
    for(size_t i=0;i<(size_t)B*ss;i++)state[i]=cosf((float)(i+5)*0.017f)*0.015f;
    memcpy(ref,x,(size_t)B*H*sizeof(float));memcpy(conv0,conv,(size_t)B*cs*sizeof(float));memcpy(state0,state,(size_t)B*ss*sizeof(float));
    layer_forward_slot_batch(&m,l,x,B,NULL,pos,NULL,B,conv,state,NULL,NULL,NULL,NULL,m.max_seq);
    float output_diff=0.f,state_diff=0.f,conv_diff=0.f;
    for(int row=0;row<B;row++){
        memcpy(w->conv_state,conv0+(size_t)row*cs,cs*sizeof(float));memcpy(w->state,state0+(size_t)row*ss,ss*sizeof(float));
        layer_forward_one(&m,l,ref+(int64_t)row*H,pos[row]);
        for(int h=0;h<H;h++){float d=fabsf(x[(int64_t)row*H+h]-ref[(int64_t)row*H+h]);if(d>output_diff)output_diff=d;}
        for(size_t i=0;i<ss;i++){float d=fabsf(state[(size_t)row*ss+i]-w->state[i]);if(d>state_diff)state_diff=d;}
        for(size_t i=0;i<cs;i++){float d=fabsf(conv[(size_t)row*cs+i]-w->conv_state[i]);if(d>conv_diff)conv_diff=d;}
    }
    printf("resident hybrid layer batch: rows=%d output=%.3g state=%.3g conv=%.3g\n",B,output_diff,state_diff,conv_diff);
    free(x);free(ref);free(conv);free(state);free(conv0);free(state0);
    return(output_diff>1e-5f||state_diff>1e-6f||conv_diff>1e-7f)?1:0;
}
