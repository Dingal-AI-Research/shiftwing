#ifndef COLIB_DEEPSEEK_V4_SERVER_H
#define COLIB_DEEPSEEK_V4_SERVER_H
/* One active request. The reader keeps accepting CANCEL while the worker is in
 * prefill/decode. A request's runtime must be reset before its terminal frame. */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#ifndef DSV4_MAX_PAYLOAD
#define DSV4_MAX_PAYLOAD (16u*1024u*1024u)
#endif

typedef struct {
    char id[65];
    char *payload;
    size_t bytes;
    int maximum;
    double temperature, top_p;
} dsv4_request;

typedef struct {
    const char *error;
    char detail[256];
    int prompt_tokens, completion_tokens, length_limited;
    double tokens_per_second, cache_hit_percent, rss_gb;
} dsv4_result;

typedef struct dsv4_server dsv4_server;
typedef dsv4_result (*dsv4_request_fn)(dsv4_server *, const dsv4_request *, void *);
struct dsv4_server {
    FILE *input, *output;
    pthread_mutex_t lock;
    pthread_cond_t ready;
    dsv4_request request;
    int active, shutdown, cancelled;
    dsv4_request_fn run;
    void *opaque;
};

static inline double dsv4_server_seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}
static inline int dsv4_cancelled(dsv4_server *s) {
    return __atomic_load_n(&s->cancelled,__ATOMIC_RELAXED);
}
static inline void dsv4_frame(dsv4_server *s, const char *format, ...) {
    flockfile(s->output);
    va_list args; va_start(args,format); vfprintf(s->output,format,args); va_end(args);
    fflush(s->output); funlockfile(s->output);
}
static inline void dsv4_data(dsv4_server *s,const char *id,const void *data,size_t bytes) {
    flockfile(s->output);
    fprintf(s->output,"DATA %s %zu\n",id,bytes);
    fwrite(data,1,bytes,s->output); fputc('\n',s->output); fflush(s->output);
    funlockfile(s->output);
}
static void *dsv4_server_worker(void *opaque) {
    dsv4_server *s=opaque;
    pthread_mutex_lock(&s->lock);
    for (;;) {
        while (!s->active && !s->shutdown) pthread_cond_wait(&s->ready,&s->lock);
        if (!s->active && s->shutdown) break;
        pthread_mutex_unlock(&s->lock);
        dsv4_result r=s->run(s,&s->request,s->opaque);
        pthread_mutex_lock(&s->lock);
        if (dsv4_cancelled(s)) { r.error="CANCELLED"; r.detail[0]=0; }
        /* Details are diagnostic text, never another wire frame. */
        for (size_t i=0;i<sizeof(r.detail) && r.detail[i];i++)
            if ((unsigned char)r.detail[i]<32) r.detail[i]=' ';
        if (r.error) dsv4_frame(s,"ERROR %s %s%s%s\n",s->request.id,r.error,
                                r.detail[0] ? " " : "",r.detail);
        else dsv4_frame(s,"DONE %s STAT %d %.6f %.6f %.6f %d %d\n",s->request.id,
            r.completion_tokens,r.tokens_per_second,r.cache_hit_percent,r.rss_gb,
            r.prompt_tokens,r.length_limited);
        free(s->request.payload); s->request.payload=NULL; s->active=0;
    }
    pthread_mutex_unlock(&s->lock);
    return NULL;
}
static inline int dsv4_request_id(const char *id) {
    size_t n=strlen(id);
    if (!n || n>64) return 0;
    for (size_t i=0;i<n;i++) if (!isalnum((unsigned char)id[i]) && id[i]!='-' && id[i]!='_') return 0;
    return 1;
}
static inline int dsv4_wire_integer(const char *s,unsigned long long maximum,unsigned long long *out) {
    if (!*s) return 0;
    for (const char *p=s;*p;p++) if (*p<'0' || *p>'9') return 0;
    char *end; errno=0; unsigned long long value=strtoull(s,&end,10);
    if (errno || *end || value>maximum) return 0;
    *out=value; return 1;
}
static inline int dsv4_wire_float(const char *s,double *out) {
    char *end; errno=0; double value=strtod(s,&end);
    if (!*s || errno || *end || !isfinite(value)) return 0;
    *out=value; return 1;
}
/* Invalid lengths or framing make resynchronization ambiguous, so close the
 * process after cancelling and joining its worker. Valid bounded frames with
 * unsupported parameters are consumed and rejected without losing alignment. */
static inline int dsv4_server_loop(FILE *input,FILE *output,dsv4_request_fn run,void *opaque) {
    dsv4_server s={.input=input,.output=output,.run=run,.opaque=opaque};
    if (pthread_mutex_init(&s.lock,NULL)) return 2;
    if (pthread_cond_init(&s.ready,NULL)) { pthread_mutex_destroy(&s.lock); return 2; }
    pthread_t worker;
    if (pthread_create(&worker,NULL,dsv4_server_worker,&s)) {
        pthread_cond_destroy(&s.ready); pthread_mutex_destroy(&s.lock); return 2;
    }
    int result=0;
    dsv4_frame(&s,"\x01\x01READY\x01\x01\n");
    char line[512];
    while (fgets(line,sizeof(line),input)) {
        size_t n=strlen(line);
        if (!n || line[n-1]!='\n') { result=2; break; }
        char *words[10], *save=NULL; int count=0;
        for (char *p=strtok_r(line," \r\n",&save); p && count<10; p=strtok_r(NULL," \r\n",&save)) words[count++]=p;
        if (count==2 && !strcmp(words[0],"CANCEL") && dsv4_request_id(words[1])) {
            pthread_mutex_lock(&s.lock);
            if (s.active && !strcmp(words[1],s.request.id)) __atomic_store_n(&s.cancelled,1,__ATOMIC_RELAXED);
            pthread_mutex_unlock(&s.lock); continue;
        }
        unsigned long long slot,bytes,maximum,grammar=0;
        double temperature,top_p;
        if ((count!=7 && count!=8) || strcmp(words[0],"SUBMIT") ||
            !dsv4_request_id(words[1]) ||
            !dsv4_wire_integer(words[2],INT32_MAX,&slot) ||
            !dsv4_wire_integer(words[3],DSV4_MAX_PAYLOAD,&bytes) ||
            !dsv4_wire_integer(words[4],INT32_MAX,&maximum) ||
            !dsv4_wire_float(words[5],&temperature) ||
            !dsv4_wire_float(words[6],&top_p) ||
            (count==8 && !dsv4_wire_integer(words[7],DSV4_MAX_PAYLOAD,&grammar))) { result=2; break; }
        char *payload=malloc((size_t)bytes+1);
        if (!payload) { result=2; break; }
        if (fread(payload,1,(size_t)bytes,input)!=(size_t)bytes) { free(payload); result=2; break; }
        payload[bytes]=0;
        char discard[4096];
        while (grammar) {
            size_t take=grammar>sizeof(discard) ? sizeof(discard) : (size_t)grammar;
            if (fread(discard,1,take,input)!=take) { result=2; break; }
            grammar-=take;
        }
        if (result || fgetc(input)!='\n') { free(payload); result=2; break; }
        pthread_mutex_lock(&s.lock);
        if (s.active && !strcmp(words[1],s.request.id)) {
            free(payload); pthread_mutex_unlock(&s.lock); result=2; break;
        }
        const char *error=NULL;
        if (slot || !maximum || maximum>8192 || temperature<0 || temperature>2 || top_p<=0 || top_p>1 ||
            (count==8 && strcmp(words[7],"0"))) error="BAD_REQUEST";
        else if (s.active) error="BUSY";
        if (error) { dsv4_frame(&s,"ERROR %s %s\n",words[1],error); free(payload); }
        else {
            snprintf(s.request.id,sizeof(s.request.id),"%s",words[1]);
            s.request.payload=payload; s.request.bytes=(size_t)bytes; s.request.maximum=(int)maximum;
            s.request.temperature=temperature; s.request.top_p=top_p;
            __atomic_store_n(&s.cancelled,0,__ATOMIC_RELAXED); s.active=1;
            pthread_cond_signal(&s.ready);
        }
        pthread_mutex_unlock(&s.lock);
    }
    pthread_mutex_lock(&s.lock); s.shutdown=1;
    __atomic_store_n(&s.cancelled,1,__ATOMIC_RELAXED); pthread_cond_signal(&s.ready);
    pthread_mutex_unlock(&s.lock); pthread_join(worker,NULL);
    pthread_cond_destroy(&s.ready); pthread_mutex_destroy(&s.lock);
    return result;
}
#endif
