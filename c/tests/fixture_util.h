#ifndef QW_FIXTURE_UTIL_H
#define QW_FIXTURE_UTIL_H

static jval *fixture_load(const char *path,char **buf_out,char **arena_out){
    long n=0;*buf_out=read_file(path,&n);(void)n;*arena_out=NULL;jval *root=json_parse(*buf_out,arena_out);
    if(!root||root->t!=J_OBJ){fprintf(stderr,"bad fixture %s\n",path);exit(1);}return root;
}
static void flatten_f32(jval *v,float *out,int *at){
    if(v->t==J_NUM){out[(*at)++]=(float)v->num;return;}
    if(v->t!=J_ARR){fprintf(stderr,"fixture value is not numeric/array\n");exit(1);}
    for(int i=0;i<v->len;i++)flatten_f32(v->kids[i],out,at);
}
static void flatten_i32(jval *v,int *out,int *at){
    if(v->t==J_NUM){out[(*at)++]=(int)v->num;return;}
    if(v->t!=J_ARR){fprintf(stderr,"fixture value is not numeric/array\n");exit(1);}
    for(int i=0;i<v->len;i++)flatten_i32(v->kids[i],out,at);
}
static float fixture_maxdiff(const float *a,const float *b,int n){float d=0.f;for(int i=0;i<n;i++){float x=fabsf(a[i]-b[i]);if(x>d)d=x;}return d;}

#endif
