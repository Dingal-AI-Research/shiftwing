#include <assert.h>
#include "../deepseek_v4_memory.h"
int main(void) {
    dsv4_memory_plan p;
    assert(dsv4_plan_memory(&p,40960,1024,29*DSV4_GIB,15*DSV4_GIB,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(p.chunk==1024 && p.host_cache<=8*DSV4_GIB && p.device_cache<=2*DSV4_GIB);
    assert(p.device_scratch<=2*DSV4_GIB && p.host_required<=p.host_available && p.device_required<=p.device_available);
    assert(p.host_capacity>=DSV4_TOPK && p.state_bytes==dsv4_runtime_state_bytes(40960));
    assert(!dsv4_plan_memory(&p,40960,1024,29*DSV4_GIB,2*DSV4_GIB,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(!dsv4_plan_memory(&p,40960,1024,8*DSV4_GIB,15*DSV4_GIB,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(!dsv4_plan_memory(&p,DSV4_MAX_CONTEXT+1,1024,29*DSV4_GIB,15*DSV4_GIB,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(!dsv4_plan_memory(&p,40960,333,29*DSV4_GIB,15*DSV4_GIB,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    uint64_t dense=DSV4_BASE_DENSE_BYTES+(uint64_t)DSV4_BASE_DENSE_RECORDS*DSV4_DENSE_ALIGNMENT;
    uint64_t minimum=(uint64_t)DSV4_TOPK*dsv4_expert_payload_bytes();
    uint64_t tight=dense+3*DSV4_GIB/2+minimum+dsv4_device_scratch_bytes(256);
    assert(dsv4_plan_memory(&p,40960,2048,29*DSV4_GIB,tight,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(p.chunk==256 && p.device_cache==minimum);
    assert(!dsv4_plan_memory(&p,40960,256,29*DSV4_GIB,tight-1,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(dsv4_plan_memory(&p,100352,1024,29*DSV4_GIB,15*DSV4_GIB,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(p.context==100352 && p.host_required<=p.host_available && p.device_required<=p.device_available);
    assert(p.state_bytes>dsv4_runtime_state_bytes(65536));
    /* Reproduce the RAM/VRAM available during the LocalForge startup failure. */
    assert(dsv4_plan_memory(&p,100352,1024,29277696000ULL,15738077184ULL,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(p.host_required<=p.host_available && p.device_required<=p.device_available);
    /* Releasing the FP8 dense pages after upload lowers the steady-state floor:
     * the 2026-09-18 LocalForge launch saw 22,464,114,688 bytes available at
     * the review context and could not fit; with 5.4 GiB of dense pages
     * returned it fits with the full expert budget, and the load-time peak
     * (whole arena resident) is still checked. */
    uint64_t released=5461ULL*1024*1024;
    assert(!dsv4_plan_memory(&p,100352,1024,22464114688ULL,15738077184ULL,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,0));
    assert(dsv4_plan_memory(&p,100352,1024,22464114688ULL,15738077184ULL,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,released));
    assert(p.dense_released==released && p.host_capacity>=DSV4_TOPK && p.host_required<=p.host_available);
    /* After a clean model swap about 27 GB is available: full expert budget. */
    assert(dsv4_plan_memory(&p,100352,1024,27ULL*1000*1000*1000,15738077184ULL,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,released));
    uint64_t layer=(DSV4_BASE_LAYERS+DSV4_DSPARK_LAYERS)*dsv4_expert_payload_bytes();
    assert(p.host_cache==8*DSV4_GIB/layer*layer && p.host_required<=p.host_available);
    assert(!dsv4_plan_memory(&p,100352,1024,dense+DSV4_GIB,15738077184ULL,1,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,released));
    assert(dsv4_plan_memory(&p,100352,1024,29*DSV4_GIB,15*DSV4_GIB,0,8*DSV4_GIB,2*DSV4_GIB,3*DSV4_GIB/2,released) && p.dense_released==0);
    assert(dsv4_prefill_bank_bytes(92160)==90*dsv4_prefill_bank_bytes(1024));
    assert(!dsv4_prefill_bank_bytes(92161));
    puts("DeepSeek memory bounds, occupied GPU rejection and chunk shrink: ok");
    return 0;
}
