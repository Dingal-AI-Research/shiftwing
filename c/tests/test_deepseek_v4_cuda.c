#include "../backend_cuda.h"
#include "../deepseek_v4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *message) {
    fprintf(stderr, "test_deepseek_v4_cuda: %s", message);
    const char *cuda = coli_cuda_last_error();
    if (cuda) fprintf(stderr, ": %s", cuda);
    fputc('\n', stderr);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static void *device_copy(ColiCuda *cuda, const void *source, size_t bytes) {
    void *device = NULL;
    if (coli_cuda_malloc(cuda, &device, bytes) ||
        coli_cuda_upload(cuda, device, source, bytes))
        fail("device allocation/upload");
    return device;
}

static void device_zero(ColiCuda *cuda, void **device, size_t bytes,
                        int value) {
    if (coli_cuda_malloc(cuda, device, bytes) ||
        coli_cuda_memset(cuda, *device, value, bytes))
        fail("device allocation/memset");
}

static float compare(const float *actual, const float *expected, size_t count,
                     float *relative) {
    float maximum = 0.0f, maximum_relative = 0.0f;
    for (size_t index = 0; index < count; index++) {
        float difference = fabsf(actual[index] - expected[index]);
        float denominator = fmaxf(1e-6f, fabsf(expected[index]));
        maximum = fmaxf(maximum, difference);
        maximum_relative = fmaxf(maximum_relative, difference / denominator);
    }
    *relative = maximum_relative;
    return maximum;
}

static void fill_fp4(unsigned char *weight, unsigned char *scale,
                     int rows, int cols, int seed) {
    int blocks = cols / 32;
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < cols; column += 2) {
            int left = ((row * 7 + column * 3 + seed) % 13) - 6;
            int right = ((row * 5 + column * 11 + seed + 3) % 13) - 6;
            unsigned char lo = dsv4_fp4_encode((float)left * 0.5f);
            unsigned char hi = dsv4_fp4_encode((float)right * 0.5f);
            weight[(size_t)row * (cols / 2) + column / 2] =
                (unsigned char)(lo | (hi << 4));
        }
        for (int block = 0; block < blocks; block++)
            scale[(size_t)row * blocks + block] =
                (unsigned char)(125 + ((row + block + seed) % 3));
    }
}

static void test_generic(ColiCuda *cuda) {
    enum { batch = 2, rows = 257, cols = 256 };
    const int activation_blocks = cols / 128;
    const int row_tiles = (rows + 127) / 128;
    float *input = (float *)malloc((size_t)batch * cols * sizeof(float));
    unsigned char *act = (unsigned char *)malloc((size_t)batch * cols);
    unsigned char *act_scale =
        (unsigned char *)malloc((size_t)batch * activation_blocks);
    unsigned char *fp8 = (unsigned char *)malloc((size_t)rows * cols);
    unsigned char *fp8_scale =
        (unsigned char *)malloc((size_t)row_tiles * activation_blocks);
    unsigned char *fp4 = (unsigned char *)malloc((size_t)rows * cols / 2);
    unsigned char *fp4_scale =
        (unsigned char *)malloc((size_t)rows * (cols / 32));
    uint16_t *bf16 = (uint16_t *)malloc(
        (size_t)rows * cols * sizeof(uint16_t));
    float *expected = (float *)malloc((size_t)batch * rows * sizeof(float));
    float *actual = (float *)malloc((size_t)batch * rows * sizeof(float));
    require(input && act && act_scale && fp8 && fp8_scale && fp4 &&
            fp4_scale && bf16 && expected && actual, "generic host allocation");

    for (int sample = 0; sample < batch; sample++) {
        for (int column = 0; column < cols; column++)
            input[(size_t)sample * cols + column] =
                sinf((float)(sample * 31 + column) * 0.071f) * 1.75f;
        require(dsv4_act_quant_mxfp(
            input + (size_t)sample * cols, cols,
            act + (size_t)sample * cols,
            act_scale + (size_t)sample * activation_blocks),
            "generic activation quantization");
    }
    for (int row = 0; row < rows; row++)
        for (int column = 0; column < cols; column++)
            fp8[(size_t)row * cols + column] = dsv4_fp8_encode(
                cosf((float)(row * 17 + column * 5) * 0.037f) * 2.5f);
    for (int tile = 0; tile < row_tiles; tile++)
        for (int block = 0; block < activation_blocks; block++)
            fp8_scale[tile * activation_blocks + block] =
                (unsigned char)(125 + ((tile + block) % 3));
    fill_fp4(fp4, fp4_scale, rows, cols, 9);
    for (int row = 0; row < rows; row++)
        for (int column = 0; column < cols; column++)
            bf16[(size_t)row * cols + column] = dsv4_float_to_bf16(
                sinf((float)(row * 13 + column * 7) * 0.019f));

    void *d_act = device_copy(cuda, act, (size_t)batch * cols);
    void *d_act_scale = device_copy(
        cuda, act_scale, (size_t)batch * activation_blocks);
    void *d_fp8 = device_copy(cuda, fp8, (size_t)rows * cols);
    void *d_fp8_scale = device_copy(
        cuda, fp8_scale, (size_t)row_tiles * activation_blocks);
    void *d_fp4 = device_copy(cuda, fp4, (size_t)rows * cols / 2);
    void *d_fp4_scale = device_copy(
        cuda, fp4_scale, (size_t)rows * (cols / 32));
    void *d_input = device_copy(
        cuda, input, (size_t)batch * cols * sizeof(float));
    void *d_bf16 = device_copy(
        cuda, bf16, (size_t)rows * cols * sizeof(uint16_t));
    void *d_out = NULL;
    require(!coli_cuda_malloc(cuda, &d_out,
        (size_t)batch * rows * sizeof(float)), "generic output allocation");

    require(dsv4_fp8_gemm(expected, act, act_scale, fp8, fp8_scale,
                         batch, rows, cols), "CPU FP8 reference");
    dsv4_round_bf16_array(expected, (size_t)batch * rows);
    require(!coli_cuda_dsv4_fp8_gemm(
        cuda, (float *)d_out, (const unsigned char *)d_act,
        (const unsigned char *)d_act_scale, (const unsigned char *)d_fp8,
        (const unsigned char *)d_fp8_scale, batch, rows, cols),
        "CUDA FP8 projection");
    require(!coli_cuda_download(cuda, actual, d_out,
        (size_t)batch * rows * sizeof(float)) && !coli_cuda_sync(cuda),
        "CUDA FP8 result");
    float relative = 0.0f;
    float absolute = compare(actual, expected, (size_t)batch * rows, &relative);
    printf("DeepSeek CUDA generic fp8 absolute=%.9g relative=%.9g\n",
           absolute, relative);
    require(absolute == 0.0f, "FP8 projection differs from BF16 reference");

    require(dsv4_fp4_gemm(expected, act, act_scale, fp4, fp4_scale,
                         batch, rows, cols), "CPU FP4 reference");
    dsv4_round_bf16_array(expected, (size_t)batch * rows);
    require(!coli_cuda_dsv4_fp4_gemm(
        cuda, (float *)d_out, (const unsigned char *)d_act,
        (const unsigned char *)d_act_scale, (const unsigned char *)d_fp4,
        (const unsigned char *)d_fp4_scale, batch, rows, cols),
        "CUDA FP4 projection");
    require(!coli_cuda_download(cuda, actual, d_out,
        (size_t)batch * rows * sizeof(float)) && !coli_cuda_sync(cuda),
        "CUDA FP4 result");
    absolute = compare(actual, expected, (size_t)batch * rows, &relative);
    printf("DeepSeek CUDA generic fp4 absolute=%.9g relative=%.9g\n",
           absolute, relative);
    require(absolute == 0.0f, "FP4 projection differs from BF16 reference");
    dsv4_bf16_gemm(expected, input, bf16, batch, rows, cols);
    require(!coli_cuda_dsv4_bf16_gemm(
        cuda, (float *)d_out, (const float *)d_input,
        (const unsigned short *)d_bf16, batch, rows, cols),
        "CUDA BF16 projection");
    require(!coli_cuda_download(cuda, actual, d_out,
        (size_t)batch * rows * sizeof(float)) && !coli_cuda_sync(cuda),
        "CUDA BF16 result");
    absolute = compare(actual, expected, (size_t)batch * rows, &relative);
    printf("DeepSeek CUDA generic bf16 absolute=%.9g relative=%.9g\n",
           absolute, relative);
    require(absolute < 2e-4f && relative < 2e-4f,
            "BF16 projection exceeds FP32 reduction tolerance");


    coli_cuda_free(cuda, d_act); coli_cuda_free(cuda, d_act_scale);
    coli_cuda_free(cuda, d_fp8); coli_cuda_free(cuda, d_fp8_scale);
    coli_cuda_free(cuda, d_fp4); coli_cuda_free(cuda, d_fp4_scale);
    coli_cuda_free(cuda, d_input); coli_cuda_free(cuda, d_bf16);
    coli_cuda_free(cuda, d_out);
    free(input); free(act); free(act_scale); free(fp8); free(fp8_scale);
    free(fp4); free(fp4_scale); free(bf16); free(expected); free(actual);
}

static void test_swiglu(ColiCuda *cuda) {
    const int count = 257;
    float gate[count], up[count], expected[count], actual[count];
    for (int index = 0; index < count; index++) {
        gate[index] = ((float)(index % 31) - 15.0f) * 1.25f;
        up[index] = ((float)(index % 29) - 14.0f) * 1.5f;
        expected[index] = dsv4_clamped_swiglu(gate[index], up[index]);
    }
    void *d_gate = device_copy(cuda, gate, sizeof(gate));
    void *d_up = device_copy(cuda, up, sizeof(up));
    void *d_out = NULL;
    require(!coli_cuda_malloc(cuda, &d_out, sizeof(actual)),
            "SwiGLU output allocation");
    require(!coli_cuda_dsv4_clamped_swiglu(
        cuda, (float *)d_out, (const float *)d_gate, (const float *)d_up,
        count), "CUDA clamped SwiGLU");
    require(!coli_cuda_download(cuda, actual, d_out, sizeof(actual)) &&
            !coli_cuda_sync(cuda), "CUDA SwiGLU result");
    float relative = 0.0f;
    float absolute = compare(actual, expected, count, &relative);
    printf("DeepSeek CUDA clamped SwiGLU absolute=%.9g relative=%.9g\n",
           absolute, relative);
    require(absolute < 2e-5f, "clamped SwiGLU exceeds tolerance");
    coli_cuda_free(cuda, d_gate); coli_cuda_free(cuda, d_up);
    coli_cuda_free(cuda, d_out);
}

static void test_grouped_fixture(ColiCuda *cuda) {
    enum { experts = 2, hidden = 128, intermediate = 128 };
    float input[hidden], expected[hidden], actual[hidden];
    unsigned char act[hidden], act_scale[hidden / 128];
    const unsigned char *d_w1[experts], *d_s1[experts], *d_w2[experts],
        *d_s2[experts], *d_w3[experts], *d_s3[experts];
    unsigned char *w1[experts], *s1[experts], *w2[experts], *s2[experts],
        *w3[experts], *s3[experts];
    const float route[experts] = {0.625f, 0.375f};
    memset(expected, 0, sizeof(expected));
    for (int column = 0; column < hidden; column++)
        input[column] = sinf((float)column * 0.13f) * 2.0f;
    require(dsv4_act_quant_mxfp(input, hidden, act, act_scale),
            "grouped input quantization");

    for (int expert = 0; expert < experts; expert++) {
        size_t hidden_weight = (size_t)intermediate * hidden / 2;
        size_t down_weight = (size_t)hidden * intermediate / 2;
        size_t hidden_scale = (size_t)intermediate * (hidden / 32);
        size_t down_scale = (size_t)hidden * (intermediate / 32);
        w1[expert] = (unsigned char *)malloc(hidden_weight);
        w3[expert] = (unsigned char *)malloc(hidden_weight);
        w2[expert] = (unsigned char *)malloc(down_weight);
        s1[expert] = (unsigned char *)malloc(hidden_scale);
        s3[expert] = (unsigned char *)malloc(hidden_scale);
        s2[expert] = (unsigned char *)malloc(down_scale);
        require(w1[expert] && w3[expert] && w2[expert] && s1[expert] &&
                s3[expert] && s2[expert], "grouped weight allocation");
        fill_fp4(w1[expert], s1[expert], intermediate, hidden, expert + 1);
        fill_fp4(w3[expert], s3[expert], intermediate, hidden, expert + 7);
        fill_fp4(w2[expert], s2[expert], hidden, intermediate, expert + 13);
        d_w1[expert] = (const unsigned char *)device_copy(
            cuda, w1[expert], hidden_weight);
        d_s1[expert] = (const unsigned char *)device_copy(
            cuda, s1[expert], hidden_scale);
        d_w3[expert] = (const unsigned char *)device_copy(
            cuda, w3[expert], hidden_weight);
        d_s3[expert] = (const unsigned char *)device_copy(
            cuda, s3[expert], hidden_scale);
        d_w2[expert] = (const unsigned char *)device_copy(
            cuda, w2[expert], down_weight);
        d_s2[expert] = (const unsigned char *)device_copy(
            cuda, s2[expert], down_scale);

        float gate[intermediate], up[intermediate], output[hidden];
        unsigned char scratch_act[intermediate];
        unsigned char scratch_scale[intermediate / 128];
        require(dsv4_expert_fp4(
            output, input, w1[expert], s1[expert], w2[expert], s2[expert],
            w3[expert], s3[expert], hidden, intermediate, route[expert],
            gate, up, scratch_act, scratch_scale), "CPU grouped reference");
        for (int row = 0; row < hidden; row++) expected[row] += output[row];
    }

    void *d_act = device_copy(cuda, act, sizeof(act));
    void *d_act_scale = device_copy(cuda, act_scale, sizeof(act_scale));
    void *d_out = NULL;
    require(!coli_cuda_malloc(cuda, &d_out, sizeof(actual)),
            "grouped output allocation");
    require(!coli_cuda_dsv4_grouped_fp4_experts(
        cuda, (float *)d_out, (const unsigned char *)d_act,
        (const unsigned char *)d_act_scale, d_w1, d_s1, d_w2, d_s2,
        d_w3, d_s3, route, experts, hidden, intermediate),
        "CUDA grouped fixture");
    require(!coli_cuda_download(cuda, actual, d_out, sizeof(actual)) &&
            !coli_cuda_sync(cuda), "CUDA grouped fixture result");
    float relative = 0.0f;
    float absolute = compare(actual, expected, hidden, &relative);
    printf("DeepSeek CUDA grouped fixture absolute=%.9g relative=%.9g\n",
           absolute, relative);
    require(absolute == 0.0f, "grouped fixture differs from BF16 reference");

    for (int expert = 0; expert < experts; expert++) {
        coli_cuda_free(cuda, (void *)d_w1[expert]);
        coli_cuda_free(cuda, (void *)d_s1[expert]);
        coli_cuda_free(cuda, (void *)d_w2[expert]);
        coli_cuda_free(cuda, (void *)d_s2[expert]);
        coli_cuda_free(cuda, (void *)d_w3[expert]);
        coli_cuda_free(cuda, (void *)d_s3[expert]);
        free(w1[expert]); free(s1[expert]); free(w2[expert]);
        free(s2[expert]); free(w3[expert]); free(s3[expert]);
    }
    coli_cuda_free(cuda, d_act); coli_cuda_free(cuda, d_act_scale);
    coli_cuda_free(cuda, d_out);
}

static double seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void test_real_shape(ColiCuda *cuda) {
    enum { experts = 6, hidden = 4096, intermediate = 2048 };
    size_t hidden_weight = (size_t)intermediate * hidden / 2;
    size_t down_weight = (size_t)hidden * intermediate / 2;
    size_t hidden_scale = (size_t)intermediate * (hidden / 32);
    size_t down_scale = (size_t)hidden * (intermediate / 32);
    void *act = NULL, *act_scale = NULL, *w1 = NULL, *s1 = NULL;
    void *w2 = NULL, *s2 = NULL, *w3 = NULL, *s3 = NULL, *out = NULL;
    device_zero(cuda, &act, hidden, 0);
    device_zero(cuda, &act_scale, hidden / 128, 127);
    device_zero(cuda, &w1, hidden_weight, 0);
    device_zero(cuda, &s1, hidden_scale, 127);
    device_zero(cuda, &w2, down_weight, 0);
    device_zero(cuda, &s2, down_scale, 127);
    device_zero(cuda, &w3, hidden_weight, 0);
    device_zero(cuda, &s3, hidden_scale, 127);
    require(!coli_cuda_malloc(cuda, &out, (size_t)hidden * sizeof(float)),
            "real-shape output allocation");
    const unsigned char *w1s[experts], *s1s[experts], *w2s[experts],
        *s2s[experts], *w3s[experts], *s3s[experts];
    float route[experts];
    for (int expert = 0; expert < experts; expert++) {
        w1s[expert] = (const unsigned char *)w1;
        s1s[expert] = (const unsigned char *)s1;
        w2s[expert] = (const unsigned char *)w2;
        s2s[expert] = (const unsigned char *)s2;
        w3s[expert] = (const unsigned char *)w3;
        s3s[expert] = (const unsigned char *)s3;
        route[expert] = 1.0f / experts;
    }
    double started = seconds();
    require(!coli_cuda_dsv4_grouped_fp4_experts(
        cuda, (float *)out, (const unsigned char *)act,
        (const unsigned char *)act_scale, w1s, s1s, w2s, s2s, w3s, s3s,
        route, experts, hidden, intermediate), "CUDA real-shape grouped path");
    float *actual = (float *)malloc((size_t)hidden * sizeof(float));
    require(actual && !coli_cuda_download(
        cuda, actual, out, (size_t)hidden * sizeof(float)) &&
        !coli_cuda_sync(cuda), "CUDA real-shape grouped result");
    double elapsed = seconds() - started;
    float maximum = 0.0f;
    for (int row = 0; row < hidden; row++)
        maximum = fmaxf(maximum, fabsf(actual[row]));
    printf("DeepSeek CUDA real 4096x2048 top6 absolute=%.9g elapsed_ms=%.3f\n",
           maximum, elapsed * 1000.0);
    require(maximum == 0.0f, "real-shape zero contract differs");

    free(actual);
    coli_cuda_free(cuda, act); coli_cuda_free(cuda, act_scale);
    coli_cuda_free(cuda, w1); coli_cuda_free(cuda, s1);
    coli_cuda_free(cuda, w2); coli_cuda_free(cuda, s2);
    coli_cuda_free(cuda, w3); coli_cuda_free(cuda, s3);
    coli_cuda_free(cuda, out);
}

int main(void) {
    ColiCuda *cuda = NULL;
    if (coli_cuda_create(&cuda, 0)) fail("CUDA context creation");
    int major = 0, minor = 0;
    require(!coli_cuda_compute_capability(cuda, &major, &minor),
            "compute capability");
    printf("DeepSeek CUDA device=%s sm=%d%d\n",
           coli_cuda_device_name(cuda), major, minor);
    test_generic(cuda);
    test_swiglu(cuda);
    test_grouped_fixture(cuda);
    test_real_shape(cuda);
    coli_cuda_destroy(cuda);
    puts("DeepSeek-V4 native CUDA tests: ok");
    return 0;
}
