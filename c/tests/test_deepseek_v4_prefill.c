#define DSV4_RUNTIME_LAYERS 4
#define DSV4_BASE_LAYERS 4

#define main original_decode_fixture_main
#include "test_deepseek_v4_runtime_decode.c"
#undef main
#include "../deepseek_v4_prefill.h"

static int cancel_progress(void *opaque, int committed, int chunk, int layer,
                           int layers, int activity) {
    (void)opaque; (void)committed; (void)chunk; (void)layer; (void)layers;
    return activity < 2;
}
static int cancel_expert_progress(void *opaque,int committed,int chunk,int layer,int layers,int activity) {
    (void)committed;(void)layer;(void)layers;
    int *calls=opaque;return activity<chunk || ++*calls<3;
}
static unsigned name_seed(const char *name) {
    unsigned seed = 5381;
    for (; *name; name++) seed = seed * 33 + (unsigned char)*name;
    return seed % 97;
}
/* Nonuniform released-format values keyed by tensor name, with
 * token-dependent expert routes, so two fixtures with different record
 * layouts hold identical tensors. This remains a fixture, not the
 * independent released-weight gate. */
static int fill_fixture(const char *root, const layout *plan) {
    char path[1024]; snprintf(path, sizeof(path), "%s/model.bin", root);
    FILE *file = fopen(path, "r+b"); CHECK(file);
    for (int r = 0; r < plan->count; r++) {
        const record *item = &plan->items[r];
        unsigned seed = name_seed(item->name);
        CHECK(!fseek(file, item->offset, SEEK_SET));
        for (size_t i = 0; i < item->bytes / dtype_bytes(item->dtype); i++) {
            unsigned pattern = (unsigned)(i * 17 + seed * 31) % 11;
            if (!strcmp(item->dtype, "BF16")) {
                uint16_t v = strstr(item->name, "norm") ? 0x3f80 :
                    dsv4_float_to_bf16((float)((int)pattern-5) / 32);
                CHECK(fwrite(&v, 2, 1, file) == 1);
            } else if (!strcmp(item->dtype, "F8_E4M3")) {
                uint8_t v = pattern & 1 ? 0x20 : 0xa0;
                CHECK(fwrite(&v, 1, 1, file) == 1);
            } else if (!strcmp(item->dtype, "I8")) {
                uint8_t v = (uint8_t)(0x11 + pattern);
                CHECK(fwrite(&v, 1, 1, file) == 1);
            } else if (!strcmp(item->dtype, "I64")) {
                int64_t v = (i / DSV4_TOPK + i % DSV4_TOPK) % DSV4_EXPERTS;
                CHECK(fwrite(&v, 8, 1, file) == 1);
            } else CHECK(!fseek(file, dtype_bytes(item->dtype), SEEK_CUR));
        }
    }
    CHECK(!fclose(file));
    return 0;
}
static void remove_fixture(const char *root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.bin", root); unlink(path);
    snprintf(path, sizeof(path), "%s/model-manifest.json", root); unlink(path);
    rmdir(root);
}
int main(void) {
    layout plan; CHECK(build_layout(&plan));
    char root[] = "/tmp/colib-dsv4-prefill-XXXXXX";
    CHECK(mkdtemp(root) && write_fixture(root, &plan));
    CHECK(!fill_fixture(root, &plan));
    /* The same tensors in the converter's page-aligned on-disk record order. */
    layout disk_plan; fixture_disk_order = 1; CHECK(build_layout(&disk_plan)); fixture_disk_order = 0;
    char disk_root[] = "/tmp/colib-dsv4-prefill-disk-XXXXXX";
    CHECK(mkdtemp(disk_root) && write_fixture(disk_root, &disk_plan));
    CHECK(!fill_fixture(disk_root, &disk_plan));
    dsv4_store store; CHECK(dsv4_store_init(&store, root));
    dsv4_dense_arena dense; CHECK(dsv4_dense_arena_init(&dense, &store, 0, 0));
    dsv4_expert_cache experts; CHECK(dsv4_expert_cache_init(&experts, &store, DSV4_TOPK));
    int tokens[137];for (int i=0;i<137;i++) tokens[i]=(i*7+i/5)%3;
    float expected[DSV4_VOCAB], actual[DSV4_VOCAB];
    dsv4_runtime scalar, chunked;
    CHECK(dsv4_runtime_init(&scalar, &store, &dense, &experts, 137));
    for (int i = 0; i < 137; i++) CHECK(dsv4_runtime_decode_token(&scalar, tokens[i], expected, 0, 0));
    CHECK(experts.read_staging); /* unaligned packed records go through staging */
    /* Page-aligned disk-order experts are read in place: no staging buffer is
     * ever needed, the entry pointers follow the on-disk order, and the logits
     * are the same bytes. */
    dsv4_store disk_store; CHECK(dsv4_store_init(&disk_store, disk_root));
    dsv4_dense_arena disk_dense; CHECK(dsv4_dense_arena_init(&disk_dense, &disk_store, 0, 0));
    dsv4_expert_cache disk_experts; CHECK(dsv4_expert_cache_init(&disk_experts, &disk_store, DSV4_TOPK));
    {
        float disk_logits[DSV4_VOCAB]; dsv4_runtime disk;
        CHECK(dsv4_runtime_init(&disk, &disk_store, &disk_dense, &disk_experts, 137));
        for (int i = 0; i < 137; i++) CHECK(dsv4_runtime_decode_token(&disk, tokens[i], disk_logits, 0, 0));
        CHECK(!memcmp(expected, disk_logits, sizeof(disk_logits)));
        CHECK(!memcmp(scalar.hc, disk.hc, DSV4_HC_MULT*DSV4_ATTN_HIDDEN*sizeof(float)));
        CHECK(!disk_experts.read_staging && disk_experts.misses > 0);
        dsv4_expert_entry *entry = dsv4_expert_cache_acquire(&disk_experts, 0, 0);
        CHECK(entry && entry->s1 == entry->storage && entry->w1 == entry->s1 + 512 &&
              entry->s3 == entry->w1 + 8192 && entry->w2 == entry->storage + 26112 - 8192);
        CHECK(dsv4_expert_cache_release(&disk_experts, entry));
        dsv4_runtime_close(&disk);
    }
    for (int chunk = 1; chunk <= 137; chunk += 34) {
        CHECK(dsv4_runtime_init(&chunked, &store, &dense, &experts, 137));
        for (int i = 0; i < 137;) {
            int n = 137-i < chunk ? 137-i : chunk;
            CHECK(dsv4_runtime_prefill_chunk(&chunked, tokens+i, n, actual, NULL, NULL));
            i += n;
        }
        CHECK(!memcmp(expected, actual, sizeof(actual)));
        CHECK(!memcmp(scalar.hc, chunked.hc, DSV4_HC_MULT*DSV4_ATTN_HIDDEN*sizeof(float)));
        CHECK(!memcmp(scalar.layers[0].attention.sliding.kv_cache,
                      chunked.layers[0].attention.sliding.kv_cache,
                      DSV4_ATTN_WINDOW*DSV4_ATTN_HEAD_DIM*sizeof(float)));
        CHECK(chunked.position == 137 && !memcmp(tokens, chunked.history, sizeof(tokens)));
        CHECK(!dsv4_runtime_prefill_chunk(&chunked, tokens, 1, actual, NULL, NULL));
        dsv4_runtime_close(&chunked);
    }
    /* Single-token chunks take the grouped decode path by default; the serial
     * per-expert path stays selectable and must produce the same bytes. */
    {
        CHECK(dsv4_runtime_init(&chunked, &store, &dense, &experts, 137));
        CHECK(chunked.decode_grouped); chunked.decode_grouped = 0;
        for (int i = 0; i < 137; i++) CHECK(dsv4_runtime_prefill_chunk(&chunked, tokens+i, 1, actual, NULL, NULL));
        CHECK(!memcmp(expected, actual, sizeof(actual)));
        CHECK(!memcmp(scalar.hc, chunked.hc, DSV4_HC_MULT*DSV4_ATTN_HIDDEN*sizeof(float)));
        CHECK(chunked.decode_steps == 137 && chunked.decode_step_seconds > 0);
        dsv4_runtime_close(&chunked);
    }
    for (int chunk=1;chunk<=137;chunk+=34) {
        CHECK(dsv4_runtime_init(&chunked,&store,&dense,&experts,137));
        uint64_t misses=experts.misses;
        CHECK(dsv4_runtime_prefill_prompt(&chunked,tokens,137,chunk,actual,NULL,NULL));
        CHECK(!memcmp(expected,actual,sizeof(actual)));
        CHECK(!memcmp(scalar.hc,chunked.hc,DSV4_HC_MULT*DSV4_ATTN_HIDDEN*sizeof(float)));
        for (int layer=0;layer<DSV4_RUNTIME_LAYERS;layer++)
            CHECK(!memcmp(scalar.layers[layer].attention.sliding.kv_cache,
                          chunked.layers[layer].attention.sliding.kv_cache,
                          DSV4_ATTN_WINDOW*DSV4_ATTN_HEAD_DIM*sizeof(float)));
        CHECK(experts.misses-misses<=DSV4_RUNTIME_LAYERS*DSV4_EXPERTS);
        CHECK(!experts.prefill_entries && chunked.position==137);
        dsv4_runtime_close(&chunked);
    }
#ifdef COLI_CUDA
    ColiCuda *cuda=NULL;
    CHECK(!coli_cuda_create(&cuda,0));
    CHECK(dsv4_dense_arena_enable_cuda(&dense,cuda));
    CHECK(dsv4_expert_cache_enable_cuda(&experts,cuda,DSV4_TOPK*dsv4_expert_payload_bytes()));
    for (int chunk=1;chunk<=137;chunk+=34) {
        CHECK(dsv4_runtime_init(&chunked,&store,&dense,&experts,137));
        uint64_t uploads=experts.cuda_upload_bytes;
        CHECK(dsv4_runtime_prefill_prompt(&chunked,tokens,137,chunk,actual,NULL,NULL));
        CHECK(experts.cuda_upload_bytes-uploads <= (uint64_t)DSV4_RUNTIME_LAYERS*DSV4_EXPERTS*dsv4_expert_payload_bytes());
        float error=0;
        for (int i=0;i<DSV4_VOCAB;i++) error=fmaxf(error,fabsf(actual[i]-expected[i]));
        printf("DeepSeek CUDA whole-prompt chunk=%d max_error=%.9g\n",chunk,error);
        CHECK(error<0.001f);
        dsv4_runtime_close(&chunked);
    }
    CHECK(experts.read_staging_pinned && experts.cuda_upload_staging);
    /* Decode on the device: one grouped submission per layer must match the
     * serial per-expert GEMM route bit for bit, and both must track the CPU
     * scalar decode within the BF16 projection tolerance. */
    {
        float serial_logits[DSV4_VOCAB]; dsv4_runtime serial;
        CHECK(dsv4_runtime_init(&serial,&store,&dense,&experts,137)); serial.decode_grouped=0;
        for (int i=0;i<137;i++) CHECK(dsv4_runtime_prefill_chunk(&serial,tokens+i,1,serial_logits,NULL,NULL));
        CHECK(dsv4_runtime_init(&chunked,&store,&dense,&experts,137)); CHECK(chunked.decode_grouped);
        uint64_t hits=experts.cuda_hits+experts.cuda_misses;
        for (int i=0;i<137;i++) CHECK(dsv4_runtime_prefill_chunk(&chunked,tokens+i,1,actual,NULL,NULL));
        CHECK(!memcmp(serial_logits,actual,sizeof(actual)));
        CHECK(!memcmp(serial.hc,chunked.hc,DSV4_HC_MULT*DSV4_ATTN_HIDDEN*sizeof(float)));
        CHECK(experts.cuda_hits+experts.cuda_misses-hits==(uint64_t)137*DSV4_RUNTIME_LAYERS*DSV4_TOPK);
        CHECK(chunked.decode_steps==137 && chunked.decode_routed_seconds>0 && experts.kernel_seconds>0);
        float error=0;
        for (int i=0;i<DSV4_VOCAB;i++) error=fmaxf(error,fabsf(actual[i]-expected[i]));
        printf("DeepSeek CUDA grouped decode max_error=%.9g\n",error);
        CHECK(error<0.001f);
        dsv4_runtime_close(&serial);
        /* In-place disk-order entries upload with their own record offsets. */
        CHECK(dsv4_dense_arena_enable_cuda(&disk_dense,cuda));
        CHECK(dsv4_expert_cache_enable_cuda(&disk_experts,cuda,DSV4_TOPK*dsv4_expert_payload_bytes()));
        float disk_logits[DSV4_VOCAB]; dsv4_runtime disk;
        CHECK(dsv4_runtime_init(&disk,&disk_store,&disk_dense,&disk_experts,137));
        for (int i=0;i<137;i++) CHECK(dsv4_runtime_prefill_chunk(&disk,tokens+i,1,disk_logits,NULL,NULL));
        CHECK(!memcmp(actual,disk_logits,sizeof(disk_logits)));
        CHECK(!memcmp(chunked.hc,disk.hc,DSV4_HC_MULT*DSV4_ATTN_HIDDEN*sizeof(float)));
        CHECK(!disk_experts.read_staging && disk_experts.cuda_misses>0);
        dsv4_runtime_close(&disk);dsv4_runtime_close(&chunked);
    }
#endif
    CHECK(dsv4_runtime_init(&chunked,&store,&dense,&experts,137));
    int expert_calls=0;
    CHECK(!dsv4_runtime_prefill_prompt(&chunked,tokens,137,4,actual,cancel_expert_progress,&expert_calls));
    CHECK(expert_calls==3 && chunked.poisoned && chunked.position==0 && !experts.prefill_entries);
#ifdef COLI_CUDA
    for (int slot=0;slot<experts.cuda_slots;slot++) CHECK(!experts.cuda_references[slot]);
#endif
    dsv4_runtime_close(&chunked);
    CHECK(dsv4_runtime_init(&chunked,&store,&dense,&experts,137));
    CHECK(!dsv4_runtime_prefill_prompt(&chunked,tokens,137,4,actual,cancel_progress,NULL));
    CHECK(chunked.poisoned && chunked.position==0 && !experts.prefill_entries);
    dsv4_runtime_close(&chunked);
    CHECK(dsv4_runtime_init(&chunked, &store, &dense, &experts, 137));
    CHECK(!dsv4_runtime_prefill_chunk(&chunked, tokens, 4, actual, cancel_progress, NULL));
    CHECK(chunked.poisoned && chunked.position == 0);
    CHECK(!dsv4_runtime_prefill_chunk(&chunked, tokens, 1, actual, NULL, NULL));
    dsv4_runtime_close(&chunked); dsv4_runtime_close(&scalar);
    dsv4_expert_cache_close(&experts); dsv4_dense_arena_close(&dense); dsv4_store_close(&store);
    dsv4_expert_cache_close(&disk_experts); dsv4_dense_arena_close(&disk_dense); dsv4_store_close(&disk_store);
#ifdef COLI_CUDA
    coli_cuda_destroy(cuda);
#endif
    remove_fixture(root); remove_fixture(disk_root);
    puts("DeepSeek chunk grouping, causal history, cancellation: ok");
    return 0;
}
