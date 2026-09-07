#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../deepseek_v4_session.h"

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
    return 1; \
} } while (0)

int main(void) {
    dsv4_session_identity identity = {0};
    for (int index = 0; index < 32; index++) {
        identity.model[index] = (uint8_t)(index + 1);
        identity.tokenizer_protocol[index] = (uint8_t)(index * 3 + 7);
        identity.engine[index] = (uint8_t)(index * 5 + 11);
    }
    dsv4_session_state source = {
        .position = 4, .context = 16384, .layers = 43, .hidden = 4096,
        .sampler_profile = 1, .temperature = 1.0f, .top_p = 0.95f,
        .history_n = 4, .mhc_n = 16, .window_kv_n = 32,
        .compressed_kv_n = 12, .compressor_n = 20,
    };
    source.history = calloc(source.history_n, sizeof(int32_t));
    source.mhc = calloc(source.mhc_n, sizeof(uint16_t));
    source.window_kv = calloc(source.window_kv_n, sizeof(uint16_t));
    source.compressed_kv = calloc(source.compressed_kv_n, sizeof(uint16_t));
    source.compressor = calloc(source.compressor_n, sizeof(float));
    CHECK(source.history && source.mhc && source.window_kv &&
          source.compressed_kv && source.compressor);
    for (uint64_t i = 0; i < source.history_n; i++) source.history[i] = (int32_t)(100 + i);
    for (uint64_t i = 0; i < source.mhc_n; i++) source.mhc[i] = (uint16_t)(i * 13 + 1);
    for (uint64_t i = 0; i < source.window_kv_n; i++) source.window_kv[i] = (uint16_t)(i * 17 + 2);
    for (uint64_t i = 0; i < source.compressed_kv_n; i++) source.compressed_kv[i] = (uint16_t)(i * 19 + 3);
    for (uint64_t i = 0; i < source.compressor_n; i++) source.compressor[i] = (float)i / 7.0f;
    for (int i = 0; i < 4; i++) source.sampler_rng[i] = 0x123456789abcdef0ull + i;
    char path[256];
    snprintf(path, sizeof(path), "/tmp/colib-dsv4-session-%ld.bin", (long)getpid());
    CHECK(dsv4_session_write(path, &identity, &source));
    dsv4_session_state loaded = {0};
    CHECK(dsv4_session_read(path, &identity, 16384, &loaded));
    CHECK(loaded.position == source.position && loaded.sampler_profile == 1);
    CHECK(!memcmp(loaded.history, source.history, source.history_n * sizeof(int32_t)));
    CHECK(!memcmp(loaded.mhc, source.mhc, source.mhc_n * sizeof(uint16_t)));
    CHECK(!memcmp(loaded.window_kv, source.window_kv,
                  source.window_kv_n * sizeof(uint16_t)));
    CHECK(!memcmp(loaded.compressed_kv, source.compressed_kv,
                  source.compressed_kv_n * sizeof(uint16_t)));
    CHECK(!memcmp(loaded.compressor, source.compressor,
                  source.compressor_n * sizeof(float)));
    dsv4_session_identity wrong = identity; wrong.model[0] ^= 1;
    dsv4_session_state rejected = {0};
    CHECK(!dsv4_session_read(path, &wrong, 16384, &rejected));
    CHECK(!dsv4_session_read(path, &identity, 65536, &rejected));
    FILE *file = fopen(path, "r+b"); CHECK(file != NULL);
    CHECK(fseek(file, -1, SEEK_END) == 0); int value = fgetc(file);
    CHECK(fseek(file, -1, SEEK_END) == 0 && fputc(value ^ 1, file) != EOF);
    CHECK(fclose(file) == 0);
    CHECK(!dsv4_session_read(path, &identity, 16384, &rejected));
    unlink(path);
    dsv4_session_free(&source); dsv4_session_free(&loaded);
    puts("DeepSeek-V4 session snapshot tests: ok");
    return 0;
}
