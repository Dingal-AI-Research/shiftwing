#include <math.h>
#include <fcntl.h>
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

static int test_production_shard_count(void) {
    enum { BASE_SHARDS = 122, Q3_SHARDS = 480, TOTAL_SHARDS = BASE_SHARDS + Q3_SHARDS };
    char template[] = "/tmp/colib-st-many-XXXXXX";
    char *dir = mkdtemp(template);
    CHECK(dir != NULL);

    for (int i = 0; i < TOTAL_SHARDS; i++) {
        char path[512];
        CHECK(snprintf(path, sizeof(path), "%s/shard-%04d.safetensors", dir, i)
              < (int)sizeof(path));
        int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
        CHECK(fd >= 0);
        uint64_t header_len = 2;
        CHECK(write(fd, &header_len, sizeof(header_len)) == (ssize_t)sizeof(header_len));
        CHECK(write(fd, "{}", 2) == 2);
        CHECK(close(fd) == 0);
    }

    shards storage;
    st_init(&storage, dir);
    CHECK(storage.nfd == TOTAL_SHARDS);
    CHECK(storage.n == 0);
    for (int i = 0; i < storage.nfd; i++) {
        CHECK(close(storage.fds[i]) == 0);
        if (storage.dfds[i] >= 0) CHECK(close(storage.dfds[i]) == 0);
        free(storage.paths[i]);
    }
    free(storage.t);

    for (int i = 0; i < TOTAL_SHARDS; i++) {
        char path[512];
        CHECK(snprintf(path, sizeof(path), "%s/shard-%04d.safetensors", dir, i)
              < (int)sizeof(path));
        CHECK(unlink(path) == 0);
    }
    CHECK(rmdir(dir) == 0);
    return 0;
}

int main(void) {
    CHECK(bf16_to_f32(0x3f80) == 1.0f);
    CHECK(bf16_to_f32(0xc020) == -2.5f);
    CHECK(f16_to_f32(0x3c00) == 1.0f);
    CHECK(f16_to_f32(0xc100) == -2.5f);
    CHECK(f16_to_f32(0x0001) > 0.0f);
    CHECK(isinf(f16_to_f32(0x7c00)));
    CHECK(st_hash("tensor.weight") == st_hash("tensor.weight"));
    CHECK(st_hash("tensor.weight") != st_hash("tensor.bias"));
    CHECK(test_production_shard_count() == 0);

    puts("safetensors primitive + 602-shard production layout tests: ok");
    return 0;
}
