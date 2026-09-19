#include "../deepseek_v4.h"
#include <stdio.h>
#include <stdlib.h>
static int failures;
#define CHECK(c) do {if (!(c)) {fprintf(stderr,"selection:%d: %s\n",__LINE__,#c);failures++;}} while(0)
static int oracle(const float *values,int n,int k,int offset,int *out) {
    unsigned char used[4097]={0};
    for (int i=0;i<k;i++) {
        int best=-1;float score=-INFINITY;
        for (int j=0;j<n;j++) if (!used[j] && values[j]>score) {best=j;score=values[j];}
        if (best<0) return 0;used[best]=1;out[i]=best+offset;
    }
    return k;
}
int main(void) {
    float values[4097],copy[4097];int got[4097],want[4097];unsigned state=19;
    for (int trial=0;trial<250;trial++) {
        int n=trial<10?trial+1:257+(trial*31)%3840,k=1+(trial*7)%n,offset=trial%127;
        if (k>512) k=512;
        for (int i=0;i<n;i++) {state=state*1664525u+1013904223u;values[i]=(int)(state%43)-21;}
        if (trial%3==0) values[n/2]=NAN;
        if (trial%4==0) values[n/3]=-INFINITY;
        if (trial%5==0) values[n/4]=INFINITY;
        memcpy(copy,values,sizeof(float)*n);
        int expected=oracle(values,n,k,offset,want),actual=dsv4_indexer_select(values,n,k,offset,got);
        CHECK(actual==expected);if (actual) CHECK(!memcmp(got,want,sizeof(int)*k));
        CHECK(!memcmp(copy,values,sizeof(float)*n));
    }
    float empty[]={NAN,-INFINITY};CHECK(!dsv4_indexer_select(empty,2,1,0,got));
    CHECK(!dsv4_indexer_select(values,3,4,0,got));CHECK(!dsv4_indexer_select(values,3,1,-1,got));
    if (!failures) puts("DeepSeek index heap matches exhaustive top-k with ties, invalid scores and offsets");
    return failures?1:0;
}
