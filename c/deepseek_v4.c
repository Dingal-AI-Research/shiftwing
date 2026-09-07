/* DeepSeek-V4-Flash-0731 native engine entry point.
 * LOAD_ONLY=1 validates the converted identity. Serving remains fail-closed
 * until the streamed forward path passes its manifest-bound quality gate. */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "deepseek_v4_runtime.h"
#include "json.h"

#define DSV4_SOURCE_REVISION "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
#define DSV4_MODEL_ID "deepseek-v4-flash-0731-colib"

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END)) { fclose(file); return NULL; }
    long size = ftell(file);
    if (size < 0) { fclose(file); return NULL; }
    rewind(file);
    char *text = (char *)calloc((size_t)size + 1, 1);
    if (!text || fread(text, 1, (size_t)size, file) != (size_t)size) {
        free(text); fclose(file); return NULL;
    }
    fclose(file);
    return text;
}

static int json_int(jval *root, const char *key, int expected) {
    jval *value = json_get(root, key);
    return value && value->t == J_NUM && (int)value->num == expected;
}

static int json_string(jval *root, const char *key, const char *expected,
                       int optional) {
    jval *value = json_get(root, key);
    return (!value && optional) ||
           (value && value->t == J_STR && !strcmp(value->str, expected));
}

static int json_number(jval *root, const char *key, double expected) {
    jval *value = json_get(root, key);
    return value && value->t == J_NUM && value->num == expected;
}

static int validate_compress_ratios(jval *root) {
    jval *ratios = json_get(root, "compress_ratios");
    if (!ratios || ratios->t != J_ARR || ratios->len != 46) return 0;
    for (int index = 0; index < 46; index++) {
        int expected = index < 2 ? 0 :
            (index < 42 ? (index % 2 ? 128 : 4) : (index == 42 ? 4 : 0));
        if (!ratios->kids[index] || ratios->kids[index]->t != J_NUM ||
            (int)ratios->kids[index]->num != expected) return 0;
    }
    return 1;
}

static int validate_dspark_targets(jval *root) {
    jval *targets = json_get(root, "dspark_target_layer_ids");
    if (!targets || targets->t != J_ARR || targets->len != 3) return 0;
    for (int index = 0; index < 3; index++)
        if (!targets->kids[index] || targets->kids[index]->t != J_NUM ||
            (int)targets->kids[index]->num != 40 + index) return 0;
    return 1;
}

static int validate_config(const char *snapshot, int context) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/config.json", snapshot) >= (int)sizeof(path)) {
        fprintf(stderr, "deepseek_v4: snapshot path is too long\n");
        return 0;
    }
    char *text = read_file(path);
    if (!text) {
        fprintf(stderr, "deepseek_v4: cannot read %s: %s\n", path, strerror(errno));
        return 0;
    }
    jval *root = json_parse(text, NULL);
    int ok = root && root->t == J_OBJ &&
        json_string(root, "model_type", "deepseek_v4", 0) &&
        json_int(root, "hidden_size", 4096) &&
        json_int(root, "num_hidden_layers", 43) &&
        json_int(root, "num_attention_heads", 64) &&
        json_int(root, "n_routed_experts", 256) &&
        json_int(root, "num_experts_per_tok", 6) &&
        json_int(root, "num_hash_layers", 3) &&
        json_int(root, "hc_mult", 4) &&
        json_int(root, "sliding_window", 128) &&
        json_int(root, "vocab_size", 129280) &&
        json_int(root, "moe_intermediate_size", 2048) &&
        json_int(root, "n_shared_experts", 1) &&
        json_int(root, "head_dim", 512) &&
        json_int(root, "q_lora_rank", 1024) &&
        json_int(root, "qk_rope_head_dim", 64) &&
        json_int(root, "o_groups", 8) &&
        json_int(root, "o_lora_rank", 1024) &&
        json_int(root, "index_topk", 512) &&
        json_int(root, "max_position_embeddings", 1048576) &&
        json_number(root, "routed_scaling_factor", 1.5) &&
        json_number(root, "swiglu_limit", 10.0) &&
        json_int(root, "dspark_block_size", 5) &&
        json_int(root, "dspark_markov_rank", 256) &&
        json_int(root, "dspark_noise_token_id", 128799) &&
        validate_dspark_targets(root) && validate_compress_ratios(root) &&
        json_string(root, "expert_dtype", "fp4", 0) &&
        json_string(root, "scoring_func", "sqrtsoftplus", 0) &&
        json_string(root, "torch_dtype", "bfloat16", 0) &&
        json_string(root, "colib_model_family", "deepseek-v4", 1) &&
        json_string(root, "colib_source_revision", DSV4_SOURCE_REVISION, 1);
    if (!ok) fprintf(stderr, "deepseek_v4: pinned configuration validation failed\n");
    if (context < 1 || context > 65536) {
        fprintf(stderr, "deepseek_v4: CTX must be in 1..65536\n");
        ok = 0;
    }
    free(text);
    return ok;
}


static double dsv4_now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static double dsv4_env_gb(const char *name, double fallback) {
    const char *text = getenv(name);
    if (!text || !*text) return fallback;
    char *end = NULL;
    double value = strtod(text, &end);
    return end && !*end && value > 0.0 ? value : fallback;
}

static int dsv4_run_smoke(dsv4_store *store, int context, int token) {
    if (!store || context < 1 || context > 65536 ||
        token < 0 || token >= DSV4_VOCAB) {
        fprintf(stderr, "deepseek_v4: invalid smoke token/context\n");
        return 2;
    }
    dsv4_dense_arena dense = {0};
    dsv4_expert_cache experts = {0};
    dsv4_runtime runtime = {0};
    int dense_ready = 0, experts_ready = 0, runtime_ready = 0;
#ifdef COLI_CUDA
    ColiCuda *cuda = NULL;
#endif
    float *logits = NULL;
    int result = 2;
    double init_start = dsv4_now_seconds();
    uint64_t read_start =
        __atomic_load_n(&store->raw.read_bytes, __ATOMIC_RELAXED);

    if (!dsv4_dense_arena_init(&dense, store, 0, 1)) {
        fprintf(stderr, "deepseek_v4: dense initialization failed: %s\n",
                dense.error);
        goto done;
    }
    dense_ready = 1;
    double ram_gb = dsv4_env_gb("RAM_GB", 6.0);
    size_t expert_bytes = dsv4_expert_payload_bytes();
    size_t layer_bytes =
        (size_t)(DSV4_BASE_LAYERS + DSV4_DSPARK_LAYERS) * expert_bytes;
    int capacity = layer_bytes
        ? (int)((ram_gb * 1024.0 * 1024.0 * 1024.0) / (double)layer_bytes)
        : 0;
    if (capacity < DSV4_TOPK) capacity = DSV4_TOPK;
    if (capacity > DSV4_EXPERTS) capacity = DSV4_EXPERTS;
    if (!dsv4_expert_cache_init(&experts, store, capacity)) {
        fprintf(stderr, "deepseek_v4: expert-cache initialization failed: %s\n",
                experts.error);
        goto done;
    }
    experts_ready = 1;
#ifdef COLI_CUDA
    if (getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA")) != 0) {
        if (coli_cuda_create(&cuda, 0)) {
            fprintf(stderr, "deepseek_v4: CUDA initialization failed: %s\n",
                    coli_cuda_last_error());
            goto done;
        }
        if (!dsv4_dense_arena_enable_cuda(&dense, cuda)) {
            fprintf(stderr, "deepseek_v4: %s\n", dense.error);
            goto done;
        }
        size_t cuda_budget = (size_t)(
            dsv4_env_gb("CUDA_EXPERT_GB", 4.0) *
            1024.0 * 1024.0 * 1024.0);
        if (!dsv4_expert_cache_enable_cuda(&experts, cuda, cuda_budget)) {
            fprintf(stderr, "deepseek_v4: CUDA expert cache failed: %s\n",
                    experts.error);
            goto done;
        }
    }
#else
    if (getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA")) != 0) {
        fprintf(stderr,
                "deepseek_v4: COLI_CUDA requested but binary has no CUDA\n");
        goto done;
    }
#endif
    if (!dsv4_runtime_init(&runtime, store, &dense, &experts, context)) {
        fprintf(stderr, "deepseek_v4: runtime initialization failed: %s\n",
                runtime.error);
        goto done;
    }
    runtime_ready = 1;
    logits = (float *)malloc((size_t)DSV4_VOCAB * sizeof(*logits));
    if (!logits) {
        fprintf(stderr, "deepseek_v4: cannot allocate logits\n");
        goto done;
    }
    int tokens = 1;
    const char *tokens_text = getenv("DSV4_SMOKE_TOKENS");
    if (tokens_text && *tokens_text) tokens = atoi(tokens_text);
    if (tokens < 1 || tokens > context) {
        fprintf(stderr, "deepseek_v4: DSV4_SMOKE_TOKENS must be in 1..%d\n",
                context);
        goto done;
    }
    int profile = getenv("DSV4_SMOKE_PROFILE") &&
        atoi(getenv("DSV4_SMOKE_PROFILE")) != 0;
    int current_token = token;
    int top_token = -1;
    float top_logit = -INFINITY;
    double decode_seconds = 0.0;
    double init_seconds = dsv4_now_seconds() - init_start;
    for (int step = 0; step < tokens; step++) {
        uint64_t step_read_start =
            __atomic_load_n(&store->raw.read_bytes, __ATOMIC_RELAXED);
        uint64_t step_direct_start =
            __atomic_load_n(&store->raw.direct_bytes, __ATOMIC_RELAXED);
        uint64_t step_uring_batches =
            __atomic_load_n(&store->raw.uring_batches, __ATOMIC_RELAXED);
        uint64_t step_uring_reads =
            __atomic_load_n(&store->raw.uring_reads, __ATOMIC_RELAXED);
        uint64_t step_cpu_hits = experts.hits;
        uint64_t step_cpu_misses = experts.misses;
#ifdef COLI_CUDA
        uint64_t step_dense_calls = dense.cuda_calls;
        uint64_t step_cuda_hits = experts.cuda_hits;
        uint64_t step_cuda_misses = experts.cuda_misses;
        double step_fp8_seconds = dense.cuda_fp8_seconds;
        double step_bf16_seconds = dense.cuda_bf16_seconds;
#endif
        double decode_start = dsv4_now_seconds();
        if (!dsv4_runtime_decode_token(
                &runtime, current_token, logits, 4096, 1)) {
            fprintf(stderr, "deepseek_v4: real forward failed: %s\n",
                    runtime.error);
            goto done;
        }
        double step_seconds = dsv4_now_seconds() - decode_start;
        decode_seconds += step_seconds;
        top_token = -1;
        top_logit = -INFINITY;
        for (int index = 0; index < DSV4_VOCAB; index++)
            if (isfinite(logits[index]) &&
                (top_token < 0 || logits[index] > top_logit)) {
                top_logit = logits[index];
                top_token = index;
            }
        if (top_token < 0) {
            fprintf(stderr, "deepseek_v4: all output logits are non-finite\n");
            goto done;
        }
        uint64_t dense_calls = 0;
        uint64_t cuda_hits = 0;
        uint64_t cuda_misses = 0;
        double fp8_seconds = 0.0;
        double bf16_seconds = 0.0;
#ifdef COLI_CUDA
        dense_calls = dense.cuda_calls - step_dense_calls;
        cuda_hits = experts.cuda_hits - step_cuda_hits;
        cuda_misses = experts.cuda_misses - step_cuda_misses;
        fp8_seconds = dense.cuda_fp8_seconds - step_fp8_seconds;
        bf16_seconds = dense.cuda_bf16_seconds - step_bf16_seconds;
#endif
        printf("DSV4_SMOKE_TOKEN status=pass step=%d input=%d top=%d "
               "logit=%.9g decode_s=%.6f tok_s=%.6f read_bytes=%llu "
               "direct_bytes=%llu uring_batches=%llu uring_reads=%llu "
               "dense_calls=%llu dense_fp8_s=%.6f dense_bf16_s=%.6f "
               "cpu_hits=%llu cpu_misses=%llu cuda_hits=%llu "
               "cuda_misses=%llu\n",
               step, current_token, top_token, top_logit, step_seconds,
               step_seconds > 0.0 ? 1.0 / step_seconds : 0.0,
               (unsigned long long)(
                   __atomic_load_n(&store->raw.read_bytes, __ATOMIC_RELAXED) -
                   step_read_start),
               (unsigned long long)(
                   __atomic_load_n(&store->raw.direct_bytes,
                                   __ATOMIC_RELAXED) - step_direct_start),
               (unsigned long long)(
                   __atomic_load_n(&store->raw.uring_batches,
                                   __ATOMIC_RELAXED) - step_uring_batches),
               (unsigned long long)(
                   __atomic_load_n(&store->raw.uring_reads,
                                   __ATOMIC_RELAXED) - step_uring_reads),
               (unsigned long long)dense_calls, fp8_seconds, bf16_seconds,
               (unsigned long long)(experts.hits - step_cpu_hits),
               (unsigned long long)(experts.misses - step_cpu_misses),
               (unsigned long long)cuda_hits,
               (unsigned long long)cuda_misses);
        if (profile) {
            double layers = 0.0;
            for (int layer = 0; layer < DSV4_RUNTIME_LAYERS; layer++) {
                layers += runtime.last_layer_seconds[layer];
                printf("DSV4_PROFILE step=%d layer=%d mode=%d seconds=%.6f "
                       "attention_stage_s=%.6f ffn_stage_s=%.6f "
                       "route_s=%.6f routed_s=%.6f shared_s=%.6f\n",
                       step, layer, (int)runtime.layers[layer].mode,
                       runtime.last_layer_seconds[layer],
                       runtime.last_attention_stage_seconds[layer],
                       runtime.last_ffn_stage_seconds[layer],
                       runtime.last_route_seconds[layer],
                       runtime.last_routed_seconds[layer],
                       runtime.last_shared_seconds[layer]);
            }
            printf("DSV4_PROFILE step=%d layers_s=%.6f head_s=%.6f "
                   "unattributed_s=%.6f\n",
                   step, layers, runtime.last_head_seconds,
                   step_seconds - layers - runtime.last_head_seconds);
        }
        fflush(stdout);
        current_token = top_token;
    }
    uint64_t read_bytes =
        __atomic_load_n(&store->raw.read_bytes, __ATOMIC_RELAXED) - read_start;
    printf("DSV4_SMOKE status=pass input=%d top=%d logit=%.9g tokens=%d "
           "context=%d init_s=%.6f decode_s=%.6f tok_s=%.6f "
           "read_bytes=%llu dense_bytes=%llu expert_capacity=%d "
           "cpu_hits=%llu misses=%llu direct_bytes=%llu "
           "direct_fallbacks=%llu uring_batches=%llu uring_reads=%llu "
           "uring_setups=%llu uring_reuses=%llu uring_fallbacks=%llu",
           token, top_token, top_logit, tokens, context,
           init_seconds, decode_seconds,
           decode_seconds > 0.0 ? (double)tokens / decode_seconds : 0.0,
           (unsigned long long)read_bytes,
           (unsigned long long)dense.payload_bytes, capacity,
           (unsigned long long)experts.hits,
           (unsigned long long)experts.misses,
           (unsigned long long)store->raw.direct_bytes,
           (unsigned long long)store->raw.direct_fallbacks,
           (unsigned long long)store->raw.uring_batches,
           (unsigned long long)store->raw.uring_reads,
           (unsigned long long)store->raw.uring_setups,
           (unsigned long long)store->raw.uring_reuses,
           (unsigned long long)store->raw.uring_fallbacks);
#ifdef COLI_CUDA
    printf(" cuda=%d cuda_dense_calls=%llu cuda_expert_hits=%llu "
           "cuda_expert_misses=%llu cuda_upload_bytes=%llu "
           "cuda_pinned_upload_bytes=%llu",
           cuda != NULL,
           (unsigned long long)dense.cuda_calls,
           (unsigned long long)experts.cuda_hits,
           (unsigned long long)experts.cuda_misses,
           (unsigned long long)(
               dense.cuda_upload_bytes + experts.cuda_upload_bytes),
           (unsigned long long)(
               dense.cuda_pinned_upload_bytes +
               experts.cuda_pinned_upload_bytes));
#else
    printf(" cuda=0");
#endif
    printf("\n");
    fflush(stdout);
    result = 0;
done:
    free(logits);
    if (runtime_ready) dsv4_runtime_close(&runtime);
    if (experts_ready) dsv4_expert_cache_close(&experts);
    if (dense_ready) dsv4_dense_arena_close(&dense);
#ifdef COLI_CUDA
    if (cuda) coli_cuda_destroy(cuda);
#endif
    return result;
}
int main(void) {
    const char *snapshot = getenv("SNAP");
    if (!snapshot || !*snapshot) {
        fprintf(stderr, "set SNAP=<DeepSeek-V4 converted model directory>\n");
        return 1;
    }
    const char *context_text = getenv("CTX");
    int context = context_text ? atoi(context_text) : 16384;
    const char *dspark = getenv("DSPARK");
    if (dspark && strcmp(dspark, "auto") && strcmp(dspark, "on") &&
        strcmp(dspark, "off")) {
        fprintf(stderr, "deepseek_v4: DSPARK must be auto, on, or off\n");
        return 2;
    }
    if (!validate_config(snapshot, context)) return 2;
    printf("colib DeepSeek-V4 engine — pinned native backend\n");
    printf("model=%s source=%s ctx=%d dspark=%s\n", DSV4_MODEL_ID,
           DSV4_SOURCE_REVISION, context, dspark ? dspark : "auto");
    char manifest[4096];
    int has_manifest = snprintf(manifest, sizeof(manifest), "%s/model-manifest.json",
                                snapshot) < (int)sizeof(manifest) && access(manifest, R_OK) == 0;
    if (!has_manifest) {
        fprintf(stderr, "deepseek_v4: converted model-manifest.json is required\n");
        return 2;
    }
    dsv4_store store;
    if (!dsv4_store_init(&store, snapshot)) {
        fprintf(stderr, "deepseek_v4: %s\n", store.error);
        return 2;
    }
    dsv4_model_contract contract;
    if (!dsv4_model_bind_contract(&contract, &store)) {
        fprintf(stderr, "deepseek_v4: %s\n", contract.error);
        dsv4_store_close(&store);
        return 2;
    }
    printf("container=%d records/%d segments layers=%d dspark_layers=%d "
           "read_bytes=%llu runtime_state_bytes=%llu\n", store.records,
           store.segments, contract.base_layers, contract.dspark_layers,
           (unsigned long long)store.raw.read_bytes,
           (unsigned long long)dsv4_runtime_state_bytes(context));
    if (getenv("LOAD_ONLY") && atoi(getenv("LOAD_ONLY")) != 0) {
        dsv4_store_close(&store);
        return 0;
    }
    const char *smoke_token = getenv("DSV4_SMOKE_TOKEN");
    if (smoke_token && *smoke_token) {
        int smoke_context = 128;
        const char *smoke_context_text = getenv("DSV4_SMOKE_CONTEXT");
        if (smoke_context_text && *smoke_context_text)
            smoke_context = atoi(smoke_context_text);
        if (smoke_context > context) smoke_context = context;
        int result = dsv4_run_smoke(
            &store, smoke_context, atoi(smoke_token));
        dsv4_store_close(&store);
        return result;
    }
    dsv4_store_close(&store);
    fprintf(stderr,
        "deepseek_v4: serving is locked until the native forward/quality gate passes\n");
    return 78;
}
