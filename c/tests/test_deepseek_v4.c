#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../deepseek_v4.h"
#include "../json.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

static int closef(float a, float b, float tolerance) {
    return fabsf(a - b) <= tolerance;
}

typedef struct { const float *value; int dim; } constant_module;

static int constant_module_forward(void *context, const float *input,
                                   float *output) {
    constant_module *module = (constant_module *)context;
    (void)input;
    memcpy(output, module->value, (size_t)module->dim * sizeof(*output));
    return 1;
}

static char *read_fixture(void) {
    const char *paths[] = {"tests/fixtures/deepseek_v4_reference.json",
                           "c/tests/fixtures/deepseek_v4_reference.json"};
    for (int p = 0; p < 2; p++) {
        FILE *file = fopen(paths[p], "rb");
        if (!file) continue;
        fseek(file, 0, SEEK_END); long size = ftell(file); rewind(file);
        char *text = calloc((size_t)size + 1, 1);
        if (text && fread(text, 1, (size_t)size, file) == (size_t)size) {
            fclose(file); return text;
        }
        free(text); fclose(file);
    }
    return NULL;
}

int main(void) {
    char *fixture_text = read_fixture(); CHECK(fixture_text != NULL);
    jval *fixture = json_parse(fixture_text, NULL); CHECK(fixture && fixture->t == J_OBJ);
    CHECK(!strcmp(json_get(fixture, "source_revision")->str,
                  "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"));
    jval *fp4 = json_get(fixture, "fp4_e2m1"); CHECK(fp4 && fp4->len == 16);
    for (int i = 0; i < 16; i++)
        CHECK(closef(dsv4_fp4_e2m1((uint8_t)i), (float)fp4->kids[i]->num, 0));
    CHECK(closef(dsv4_fp8_e4m3fn(1), 0.001953125f, 0));
    CHECK(closef(dsv4_fp8_e4m3fn(0x7e), 448.0f, 0));
    CHECK(isnan(dsv4_fp8_e4m3fn(0x7f)));
    CHECK(closef(dsv4_ue8m0(0x7e), 0.5f, 0));
    CHECK(closef(dsv4_ue8m0(0x7f), 1.0f, 0));
    float hadamard[4] = {1, 2, 3, 4}, hadamard_original[4];
    memcpy(hadamard_original, hadamard, sizeof(hadamard));
    CHECK(dsv4_hadamard(hadamard, 4) && dsv4_hadamard(hadamard, 4));
    for (int index = 0; index < 4; index++)
        CHECK(closef(hadamard[index], hadamard_original[index], 1e-6f));
    float fp4_simulation[32] = {6.0f, -6.0f, 1.0f, -0.5f};
    CHECK(dsv4_fp4_simulate(fp4_simulation, 32, 32));
    CHECK(fp4_simulation[0] == 6.0f && fp4_simulation[1] == -6.0f &&
          fp4_simulation[2] == 1.0f && fp4_simulation[3] == -0.5f);

    float x[128]; uint8_t q[128], xs[1];
    for (int i = 0; i < 128; i++) x[i] = i == 0 ? 448.0f : 1.0f;
    CHECK(dsv4_act_quant_mxfp(x, 128, q, xs));
    CHECK(xs[0] == 127 && q[0] == 0x7e && q[1] == 0x38);

    uint8_t one[128], one_scale[1] = {127};
    memset(one, 0x38, sizeof(one));
    float gemm[1];
    CHECK(dsv4_fp8_gemm(gemm, one, one_scale, one, one_scale, 1, 1, 128));
    CHECK(closef(gemm[0], 128.0f, 1e-5f));
    uint8_t fp4_one[64]; memset(fp4_one, 0x22, sizeof(fp4_one));
    uint8_t fp4_scale[4] = {127,127,127,127};
    CHECK(dsv4_fp4_gemm(gemm, one, one_scale, fp4_one, fp4_scale, 1, 1, 128));
    CHECK(closef(gemm[0], 128.0f, 1e-5f));

    float norm_x[2] = {3,4}, norm_out[2]; uint16_t norm_w[2] = {0x3f80,0x3f80};
    dsv4_rmsnorm(norm_out, norm_x, norm_w, 2, 0.0f);
    CHECK(closef(norm_out[0], dsv4_round_bf16(3.0f / sqrtf(12.5f)), 1e-6f));
    uint16_t bf16_matrix[4] = {0x3f80,0,0,0x3f80}; float identity_out[2];
    dsv4_bf16_gemm(identity_out, norm_x, bf16_matrix, 1, 2, 2);
    CHECK(identity_out[0] == 3 && identity_out[1] == 4);

    enum { EH=128, EI=128 };
    uint8_t ew1[EI*EH/2], ew2[EH*EI/2], ew3[EI*EH/2];
    uint8_t es1[EI*(EH/32)], es2[EH*(EI/32)], es3[EI*(EH/32)];
    float ex[EH] = {0}, eg[EI], eu[EI], ey[EH];
    uint8_t eact[EH], eas[EH/128];
    ex[0] = 1.0f; memset(ew1,0x22,sizeof(ew1)); memset(ew2,0x22,sizeof(ew2));
    memset(ew3,0x22,sizeof(ew3)); memset(es1,127,sizeof(es1));
    memset(es2,127,sizeof(es2)); memset(es3,127,sizeof(es3));
    CHECK(dsv4_expert_fp4(ey, ex, ew1, es1, ew2, es2, ew3, es3,
                          EH, EI, 0.5f, eg, eu, eact, eas));
    for (int i = 0; i < EH; i++) CHECK(closef(ey[i], 48.0f, 1e-4f));
    uint8_t fw1[EI*EH], fw2[EH*EI], fw3[EI*EH];
    uint8_t fs1[(EI/128)*(EH/128)], fs2[(EH/128)*(EI/128)];
    uint8_t fs3[(EI/128)*(EH/128)];
    memset(fw1,0x38,sizeof(fw1)); memset(fw2,0x38,sizeof(fw2));
    memset(fw3,0x38,sizeof(fw3)); memset(fs1,127,sizeof(fs1));
    memset(fs2,127,sizeof(fs2)); memset(fs3,127,sizeof(fs3));
    CHECK(dsv4_expert_fp8(ey, ex, fw1, fs1, fw2, fs2, fw3, fs3,
                          EH, EI, eg, eu, eact, eas));
    for (int i = 0; i < EH; i++) CHECK(closef(ey[i], 96.0f, 1e-4f));

    float logits[8] = {-4,-3,-2,-1,0,1,2,3}, bias[8] = {0};
    int indices[6]; float weights[6];
    bias[0] = 100.0f;
    CHECK(dsv4_route_top6(logits, bias, 8, NULL, indices, weights));
    CHECK(indices[0] == 0);
    float route_sum = 0;
    for (int i = 0; i < 6; i++) route_sum += weights[i];
    CHECK(closef(route_sum, 1.5f, 2e-6f));
    jval *router = json_get(fixture, "router");
    jval *router_indices = json_get(router, "indices"), *router_weights = json_get(router, "weights");
    for (int i = 0; i < 6; i++) {
        CHECK(indices[i] == (int)router_indices->kids[i]->num);
        CHECK(closef(weights[i], (float)router_weights->kids[i]->num, 2e-6f));
    }
    int hash[6] = {7,6,5,4,3,2};
    CHECK(dsv4_route_top6(logits, NULL, 8, hash, indices, weights));
    CHECK(!memcmp(indices, hash, sizeof(hash)));

    CHECK(closef(dsv4_clamped_swiglu(100.0f, 100.0f),
                 100.0f * dsv4_sigmoid(10.0f), 1e-4f));
    /* Pinned Torch FP32 SiLU underflows through exp(100) to signed zero. */
    CHECK(dsv4_clamped_swiglu(-100.0f, 2.0f) == 0.0f);
    CHECK(signbit(dsv4_clamped_swiglu(-100.0f, 2.0f)));
    CHECK(dsv4_clamped_swiglu(-80.0f, 2.0f) < 0.0f);

    float mixes[24], base[24], scale[3] = {0.5f,0.75f,1.25f};
    for (int i = 0; i < 24; i++) { mixes[i] = (float)(i - 12) / 7; base[i] = (float)(i % 5 - 2) / 11; }
    float pre[4], post[4], comb[16];
    dsv4_hc_split_sinkhorn(mixes, scale, base, 1e-6f, 20, pre, post, comb);
    for (int k = 0; k < 4; k++) {
        float column = 0;
        for (int j = 0; j < 4; j++) column += comb[j * 4 + k];
        CHECK(closef(column, 1.0f, 2e-5f));
        CHECK(pre[k] > 0 && post[k] > 0 && post[k] < 2);
        jval *hc = json_get(fixture, "hc");
        CHECK(closef(pre[k], (float)json_get(hc, "pre")->kids[k]->num, 2e-6f));
        CHECK(closef(post[k], (float)json_get(hc, "post")->kids[k]->num, 2e-6f));
    }
    jval *hc_comb = json_get(json_get(fixture, "hc"), "comb");
    for (int i = 0; i < 16; i++) CHECK(closef(comb[i], (float)hc_comb->kids[i]->num, 3e-6f));
    float residual[8] = {1,2,3,4,5,6,7,8}, reduced[2], expanded[8];
    dsv4_hc_pre(residual, 2, pre, reduced);
    dsv4_hc_post(reduced, residual, 2, post, comb, expanded);
    for (int i = 0; i < 8; i++) CHECK(isfinite(expanded[i]));

    jval *hc_forward = json_get(fixture, "hc_forward");
    CHECK(hc_forward && (int)json_get(hc_forward, "dim")->num == 4);
    float full_residual[16], full_weight[24 * 16], full_scale[3], full_base[24];
    float full_reduced[4], full_post[4], full_comb[16], full_mixes[24];
    float full_module[4], full_expanded[16];
    jval *full_residual_json = json_get(hc_forward, "residual");
    jval *full_weight_json = json_get(hc_forward, "function_weight");
    jval *full_scale_json = json_get(hc_forward, "scale");
    jval *full_base_json = json_get(hc_forward, "base");
    for (int i = 0; i < 16; i++) full_residual[i] = (float)full_residual_json->kids[i]->num;
    for (int i = 0; i < 24 * 16; i++) full_weight[i] = (float)full_weight_json->kids[i]->num;
    for (int i = 0; i < 3; i++) full_scale[i] = (float)full_scale_json->kids[i]->num;
    for (int i = 0; i < 24; i++) full_base[i] = (float)full_base_json->kids[i]->num;
    dsv4_hc_forward_pre(full_residual, full_weight, 4, full_scale, full_base,
                        1e-6f, full_reduced, full_post, full_comb, full_mixes);
    jval *expected_mixes = json_get(hc_forward, "mixes");
    jval *expected_reduced = json_get(hc_forward, "reduced");
    jval *expected_post = json_get(hc_forward, "post");
    jval *expected_comb = json_get(hc_forward, "comb");
    for (int i = 0; i < 24; i++)
        CHECK(closef(full_mixes[i], (float)expected_mixes->kids[i]->num, 3e-6f));
    for (int i = 0; i < 4; i++) {
        CHECK(closef(full_reduced[i], (float)expected_reduced->kids[i]->num, 3e-6f));
        CHECK(closef(full_post[i], (float)expected_post->kids[i]->num, 3e-6f));
        full_module[i] = (float)json_get(hc_forward, "module")->kids[i]->num;
    }
    for (int i = 0; i < 16; i++)
        CHECK(closef(full_comb[i], (float)expected_comb->kids[i]->num, 3e-6f));
    dsv4_hc_post(full_module, full_residual, 4, full_post, full_comb,
                 full_expanded);
    jval *expected_expanded = json_get(hc_forward, "expanded");
    for (int i = 0; i < 16; i++)
        CHECK(closef(full_expanded[i], (float)expected_expanded->kids[i]->num,
                     4e-6f));
    uint16_t stage_norm[4] = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
    float stage_state[16], stage_reduced[4], stage_post[4], stage_comb[16];
    float stage_mixes[24], stage_module_output[4], stage_expanded[16];
    memcpy(stage_state, full_residual, sizeof(stage_state));
    constant_module stage_module = {full_module, 4};
    CHECK(dsv4_hc_stage(
        stage_state, 4, full_weight, full_scale, full_base, stage_norm,
        1e-6f, constant_module_forward, &stage_module, stage_reduced,
        stage_post, stage_comb, stage_mixes, stage_module_output,
        stage_expanded));
    for (int i = 0; i < 16; i++)
        CHECK(closef(stage_state[i], dsv4_round_bf16(full_expanded[i]), 4e-6f));

    float rope[4] = {1,2,3,4}, original[4]; memcpy(original, rope, sizeof(rope));
    dsv4_rope(rope, 4, 1234, 65536, 160000.0f, 16.0f, 32, 1, 0);
    jval *rope_expected = json_get(json_get(fixture, "rope"), "output");
    for (int i = 0; i < 4; i++) CHECK(closef(rope[i], (float)rope_expected->kids[i]->num, 2e-4f));
    dsv4_rope(rope, 4, 1234, 65536, 160000.0f, 16.0f, 32, 1, 1);
    for (int i = 0; i < 4; i++) CHECK(closef(rope[i], original[i], 2e-5f));

    int window[4];
    CHECK(dsv4_window_indices(4, 6, 0, 5, window) == 4);
    CHECK(window[0] == 2 && window[3] == 5);
    CHECK(dsv4_window_indices(4, 1, 5, 0, window) == 4);
    CHECK(window[0] == 2 && window[3] == 1);
    int compressed[8];
    CHECK(dsv4_compress_indices(4, 16, 0, 16, 7, compressed) == 4);
    CHECK(compressed[0] == 16 && compressed[1] == 17 && compressed[2] == -1);
    CHECK(dsv4_compress_indices(4, 1, 15, 128, 0, compressed) == 4);
    CHECK(compressed[0] == 128 && compressed[3] == 131);
    float pool_kv[4] = {1,10,3,30}, pool_score[4] = {0,0,0,0};
    float pool_ape[4] = {0,0,0,0}, pooled[2];
    dsv4_compress_pool(pooled, pool_kv, pool_score, pool_ape, 2, 2);
    CHECK(closef(pooled[0], 2.0f, 1e-6f) && closef(pooled[1], 20.0f, 1e-6f));
    jval *compressor_decode = json_get(fixture, "compressor_decode");
    CHECK(compressor_decode && compressor_decode->t == J_OBJ);
    const char *compressor_modes[2] = {"overlap", "plain"};
    const int compressor_overlap[2] = {1, 0};
    for (int mode = 0; mode < 2; mode++) {
        jval *item = json_get(compressor_decode, compressor_modes[mode]);
        CHECK(item && item->t == J_OBJ);
        int ratio = (int)json_get(item, "ratio")->num;
        int dim = (int)json_get(item, "dim")->num;
        int coefficient = compressor_overlap[mode] ? 2 : 1;
        int width = coefficient * dim;
        size_t state_count = dsv4_compressor_state_floats(
            ratio, dim, compressor_overlap[mode]);
        CHECK(ratio == 4 && dim == 2 && state_count > 0);
        float *kv_state = calloc(state_count, sizeof(float));
        float *score_state = calloc(state_count, sizeof(float));
        float *ape = calloc((size_t)ratio * width, sizeof(float));
        float *kv_input = calloc((size_t)8 * width, sizeof(float));
        float *score_input = calloc((size_t)8 * width, sizeof(float));
        CHECK(kv_state && score_state && ape && kv_input && score_input);
        jval *ape_json = json_get(item, "ape");
        jval *kv_json = json_get(item, "kv");
        jval *score_json = json_get(item, "score");
        CHECK(ape_json->len == ratio * width && kv_json->len == 8 * width &&
              score_json->len == 8 * width);
        for (int index = 0; index < ape_json->len; index++)
            ape[index] = (float)ape_json->kids[index]->num;
        for (int index = 0; index < kv_json->len; index++) {
            kv_input[index] = (float)kv_json->kids[index]->num;
            score_input[index] = (float)score_json->kids[index]->num;
        }
        dsv4_compressor_state compressor;
        CHECK(dsv4_compressor_state_init(
            &compressor, ratio, dim, compressor_overlap[mode], kv_state,
            score_state));
        jval *outputs = json_get(item, "outputs");
        int emitted = 0;
        for (int position = 0; position < 8; position++) {
            float output_value[2];
            int result = dsv4_compressor_step(
                &compressor, position, kv_input + (size_t)position * width,
                score_input + (size_t)position * width, ape, output_value);
            int expected_result = (position + 1) % ratio == 0;
            CHECK(result == expected_result);
            if (result == 1) {
                jval *expected = outputs->kids[emitted++];
                CHECK((int)json_get(expected, "position")->num == position);
                jval *value = json_get(expected, "value");
                for (int axis = 0; axis < dim; axis++)
                    CHECK(closef(output_value[axis],
                                 (float)value->kids[axis]->num, 2e-6f));
            }
        }
        CHECK(emitted == outputs->len &&
              dsv4_compressor_step(&compressor, 7, kv_input, score_input,
                                    ape, pooled) == -1);
        jval *final_kv = json_get(item, "final_kv_state");
        jval *final_score = json_get(item, "final_score_state");
        CHECK((size_t)final_kv->len == state_count &&
              (size_t)final_score->len == state_count);
        for (size_t index = 0; index < state_count; index++) {
            CHECK(closef(kv_state[index], (float)final_kv->kids[index]->num,
                         1e-7f));
            CHECK(closef(score_state[index],
                         (float)final_score->kids[index]->num, 1e-7f));
        }
        free(kv_state); free(score_state); free(ape);
        free(kv_input); free(score_input);
    }
    CHECK(dsv4_dspark_indices(4, 3, 2, compressed) == 6);
    CHECK(compressed[0] == 0 && compressed[2] == 2 && compressed[3] == 4);

    jval *indexer = json_get(fixture, "indexer");
    CHECK(indexer && indexer->t == J_OBJ);
    int index_heads = (int)json_get(indexer, "heads")->num;
    int index_dim = (int)json_get(indexer, "dim")->num;
    int index_tokens = (int)json_get(indexer, "tokens")->num;
    int index_topk = (int)json_get(indexer, "topk")->num;
    float index_query[12], index_kv[24], index_weights[3], index_scores[6];
    int index_selection[3];
    jval *index_query_json = json_get(indexer, "query");
    jval *index_kv_json = json_get(indexer, "kv");
    jval *index_weights_json = json_get(indexer, "weights");
    for (int index = 0; index < 12; index++)
        index_query[index] = (float)index_query_json->kids[index]->num;
    for (int index = 0; index < 24; index++)
        index_kv[index] = (float)index_kv_json->kids[index]->num;
    for (int index = 0; index < 3; index++)
        index_weights[index] = (float)index_weights_json->kids[index]->num;
    CHECK(dsv4_indexer_topk(index_query, index_kv, index_weights,
                             index_heads, index_dim, index_tokens, index_topk,
                             7, index_scores, index_selection) == index_topk);
    jval *index_expected_scores = json_get(indexer, "scores");
    jval *index_expected_selection = json_get(indexer, "indices");
    for (int index = 0; index < index_tokens; index++)
        CHECK(closef(index_scores[index],
                     (float)index_expected_scores->kids[index]->num, 2e-6f));
    for (int index = 0; index < index_topk; index++)
        CHECK(index_selection[index] ==
              (int)index_expected_selection->kids[index]->num + 7);
    CHECK(!dsv4_indexer_topk(index_query, index_kv, index_weights,
                              index_heads, index_dim, index_tokens,
                              index_tokens + 1, 0, index_scores,
                              index_selection));

    float query[1] = {0}, kv[2] = {2,4}, output[1], sink[1] = {0};
    int selected[2] = {0,1};
    dsv4_sparse_attention(output, query, kv, 1, 1, selected, 2, sink, 1);
    CHECK(closef(output[0], 2.0f, 1e-6f));

    free(fixture_text);
    puts("DeepSeek-V4 scalar reference tests: ok");
    return 0;
}
