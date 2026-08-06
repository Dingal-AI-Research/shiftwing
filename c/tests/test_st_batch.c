#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
#ifndef __linux__
    puts("safetensors batch read: skipped (non-Linux)");
    return 0;
#else
    char dir[] = "test_st_batch_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    const int n = 8192;
    const char *hdr =
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[8192],\"data_offsets\":[0,8192]},"
        "\"b\":{\"dtype\":\"U8\",\"shape\":[8192],\"data_offsets\":[8192,16384]}}";
    uint64_t hlen = (uint64_t)strlen(hdr);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(&hlen, 1, sizeof(hlen), f) == sizeof(hlen));
    CHECK(fwrite(hdr, 1, (size_t)hlen, f) == (size_t)hlen);
    for (int i = 0; i < 2 * n; i++) {
        unsigned char v = (unsigned char)(i * 29 + 7);
        CHECK(fwrite(&v, 1, 1, f) == 1);
    }
    CHECK(fclose(f) == 0);

    shards S;
    st_init(&S, dir);
    unsigned char *a = calloc((size_t)n, 1), *b = calloc((size_t)n, 1);
    CHECK(a && b);
    const char *names[2] = {"a", "b"};
    void *outs[2] = {a, b};
    setenv("URING", "1", 1);
    setenv("DIRECT", "1", 1);
    setenv("URING_PERSIST", "1", 1);
    st_read_raw_batch(&S, names, outs, 2, 1);
    for (int i = 0; i < n; i++) {
        CHECK(a[i] == (unsigned char)(i * 29 + 7));
        CHECK(b[i] == (unsigned char)((i + n) * 29 + 7));
    }
    CHECK(S.read_bytes == (uint64_t)(2 * n));
    CHECK(S.uring_batches + S.uring_fallbacks >= 1);
    CHECK(S.direct_bytes == (uint64_t)(2 * n) || S.direct_fallbacks >= 1);
    uint64_t batches = S.uring_batches, setups = S.uring_setups;
    memset(a, 0, (size_t)n); memset(b, 0, (size_t)n);
    st_read_raw_batch(&S, names, outs, 2, 1);
    for (int i = 0; i < n; i++) {
        CHECK(a[i] == (unsigned char)(i * 29 + 7));
        CHECK(b[i] == (unsigned char)((i + n) * 29 + 7));
    }
    CHECK(S.read_bytes == (uint64_t)(4 * n));
    if (S.uring_batches > batches) {
        CHECK(S.uring_setups == setups);
        CHECK(S.uring_reuses >= 1);
    }

    free(a); free(b);
    CHECK(unlink(path) == 0);
    CHECK(rmdir(dir) == 0);
    puts("safetensors batched io_uring/O_DIRECT read: ok");
    return 0;
#endif
}
