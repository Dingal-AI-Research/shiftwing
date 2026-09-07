/* Resident-batch equivalence for Qwen4-Exp.
 *
 * LocalForge talks to the server, which decodes through
 * resident_forward_tokens_ex, not forward_token. Those are separate
 * implementations of the same recurrence, and the Qwen4-Exp work touched both:
 * a four-stream residual, the terminal mixer, and the PLE layer. If they drift,
 * the CLI looks correct while the served lane quietly does not.
 *
 * Takes the container from SNAP so it can run against the generated
 * qwen4exp tiny model; skips cleanly when SNAP is unset, so `make test-c`
 * stays green on machines without one.
 *
 *   SNAP=c/qwen4exp_tiny ./tests/test_resident_qwen4_exp
 */
#define QWEN_NO_MAIN
#include "../qwen.c"

int main(void){
    const char *snap=getenv("SNAP");
    if(!snap){printf("resident qwen4-exp: skipped (set SNAP)\n");return 0;}
    setenv("MTP","0",1);setenv("KV16","0",1);setenv("PREFETCH_THREADS","0",1);
    if(!getenv("CTX"))setenv("CTX","128",1);
    static Model m;model_init(&m,snap);
    Cfg *c=&m.c;

    /* A short prompt plus a few decoded steps exercises prefill, the decode
     * recurrence and the terminal collapse on both paths. */
    enum{PROMPT=6,STEPS=6};
    int prompt[PROMPT];
    for(int i=0;i<PROMPT;i++)prompt[i]=(i*37+5)%c->vocab;

    float *logits=falloc(c->vocab);
    int direct[STEPS];
    model_reset(&m);
    prefill_dispatch(&m,prompt,PROMPT,logits);
    for(int s=0;s<STEPS;s++){
        direct[s]=argmax(logits,c->vocab);
        if(s+1<STEPS)forward_token(&m,direct[s],logits);
    }

    ResidentBatchState r={0};
    if(!resident_batch_init(&m,&r,1,1)){fprintf(stderr,"resident init failed\n");return 1;}
    model_reset(&m);
    prefill_dispatch(&m,prompt,PROMPT,logits);
    if(!resident_import_model_slot(&m,&r,0,logits)){fprintf(stderr,"resident import failed\n");return 1;}
    int resident[STEPS],slot=0;
    for(int s=0;s<STEPS;s++){
        resident[s]=argmax(r.logits,c->vocab);
        if(s+1<STEPS&&!resident_forward_tokens_ex(&m,&r,&slot,&resident[s],1,1,0)){
            fprintf(stderr,"resident forward failed at step %d\n",s);return 1;
        }
    }

    int bad=0;
    for(int s=0;s<STEPS;s++)if(direct[s]!=resident[s])bad++;
    printf("resident qwen4-exp: family=%s tokens direct=",c->is_qwen4_exp?"qwen4-exp":"qwen3.5");
    for(int s=0;s<STEPS;s++)printf(" %d",direct[s]);
    printf(" resident=");
    for(int s=0;s<STEPS;s++)printf(" %d",resident[s]);
    printf(" mismatches=%d\n",bad);
    resident_batch_free(&m,&r);free(logits);
    return bad?1:0;
}
