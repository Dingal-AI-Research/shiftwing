#include <assert.h>
#include "../../deepseek_v4_server.h"
#include "../../deepseek_v4_sha256.h"
static dsv4_result run(dsv4_server *s,const dsv4_request *q,void *opaque) {
    int *active=opaque;assert(!*active);*active=1;
    dsv4_result r={.prompt_tokens=3,.completion_tokens=2,.tokens_per_second=1,.length_limited=q->maximum==1};
    dsv4_frame(s,"PREFILL_BEGIN %s 3 0\n",q->id);
    if (q->bytes==4 && !memcmp(q->payload,"hold",4)) {
        struct timespec delay={0,1000000};
        for (int i=0;i<5000 && !dsv4_cancelled(s);i++) nanosleep(&delay,NULL);
    } else if (q->bytes==5 && !memcmp(q->payload,"error",5)) {
        r.error="FORWARD_FAILED";snprintf(r.detail,sizeof(r.detail),"bad\nstate");
    } else {
        dsv4_data(s,q->id,q->payload,q->bytes);
        dsv4_frame(s,"PREFILL_END %s 3 1\n",q->id);
    }
    *active=0;return r;
}
int main(int argc,char **argv) {
    if (argc==2) {
        char hex[65];if (!dsv4_sha256_file(argv[1],hex)) return 2;
        FILE *f=fopen(argv[1],"rb");assert(f);assert(!fseek(f,0,SEEK_END));long n=ftell(f);rewind(f);
        char *data=malloc((size_t)n+1);assert(data);assert(fread(data,1,(size_t)n,f)==(size_t)n);fclose(f);
        char accelerated[65];dsv4_sha256_hex(data,(size_t)n,accelerated);assert(!strcmp(hex,accelerated));free(data);
        puts(hex);return 0;
    }
    int active=0;return dsv4_server_loop(stdin,stdout,run,&active);
}
