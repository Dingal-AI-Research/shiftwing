#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../deepseek_v4_tier.h"

static double bench_now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int env_int(const char *name, int fallback) {
    const char *text = getenv(name);
    return text && *text ? atoi(text) : fallback;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s CONVERTED_MODEL_DIR\n", argv[0]);
        return 2;
    }
    int layers = env_int("IO_LAYERS", 4);
    int start_layer = env_int("IO_START_LAYER", 0);
    int repetitions = env_int("IO_REPS", 1);
    if (layers < 1 || start_layer < 0 ||
        start_layer + layers > DSV4_BASE_LAYERS ||
        repetitions < 1 || repetitions > 16) {
        fprintf(stderr, "invalid IO_LAYERS/IO_START_LAYER/IO_REPS\n");
        return 2;
    }
    dsv4_store store;
    if (!dsv4_store_init(&store, argv[1])) {
        fprintf(stderr, "store: %s\n", store.error);
        return 1;
    }
    dsv4_expert_cache cache;
    if (!dsv4_expert_cache_init(&cache, &store, DSV4_TOPK)) {
        fprintf(stderr, "cannot initialize expert cache\n");
        dsv4_store_close(&store);
        return 1;
    }
    uint64_t read_start = store.raw.read_bytes;
    uint64_t direct_start = store.raw.direct_bytes;
    uint64_t fallback_start = store.raw.direct_fallbacks;
    uint64_t batch_start = store.raw.uring_batches;
    uint64_t uring_read_start = store.raw.uring_reads;
    uint64_t setup_start = store.raw.uring_setups;
    uint64_t reuse_start = store.raw.uring_reuses;
    uint64_t uring_fallback_start = store.raw.uring_fallbacks;
    uint64_t checksum = 0;
    double started = bench_now_seconds();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        for (int item = 0; item < layers; item++) {
            int layer = start_layer + item;
            int experts[DSV4_TOPK];
            dsv4_expert_entry *entries[DSV4_TOPK] = {0};
            for (int route = 0; route < DSV4_TOPK; route++)
                experts[route] =
                    (layer * 37 + route * 41 + repetition * 17 + 11) %
                    DSV4_EXPERTS;
            double layer_started = bench_now_seconds();
            if (!dsv4_expert_cache_acquire_many(
                    &cache, layer, experts, DSV4_TOPK, entries)) {
                fprintf(stderr, "layer %d: %s\n", layer, cache.error);
                dsv4_expert_cache_close(&cache);
                dsv4_store_close(&store);
                return 1;
            }
            for (int route = 0; route < DSV4_TOPK; route++) {
                size_t bytes = dsv4_expert_payload_bytes();
                checksum = checksum * UINT64_C(1315423911) +
                    entries[route]->storage[0] +
                    entries[route]->storage[bytes - 1];
                if (!dsv4_expert_cache_release(&cache, entries[route])) {
                    fprintf(stderr, "cannot release layer %d route %d\n",
                            layer, route);
                    dsv4_expert_cache_close(&cache);
                    dsv4_store_close(&store);
                    return 1;
                }
            }
            printf("DSV4_IO_LAYER repetition=%d layer=%d seconds=%.6f\n",
                   repetition, layer,
                   bench_now_seconds() - layer_started);
        }
    }
    double seconds = bench_now_seconds() - started;
    uint64_t bytes = store.raw.read_bytes - read_start;
    printf("DSV4_IO status=pass start_layer=%d layers=%d reps=%d "
           "seconds=%.6f gib=%.6f gib_s=%.6f read_bytes=%llu "
           "direct_bytes=%llu direct_fallbacks=%llu uring_batches=%llu "
           "uring_reads=%llu uring_setups=%llu uring_reuses=%llu "
           "uring_fallbacks=%llu checksum=%llu\n",
           start_layer, layers, repetitions, seconds,
           (double)bytes / (1024.0 * 1024.0 * 1024.0),
           seconds > 0.0
               ? (double)bytes / (1024.0 * 1024.0 * 1024.0) / seconds
               : 0.0,
           (unsigned long long)bytes,
           (unsigned long long)(store.raw.direct_bytes - direct_start),
           (unsigned long long)(
               store.raw.direct_fallbacks - fallback_start),
           (unsigned long long)(store.raw.uring_batches - batch_start),
           (unsigned long long)(store.raw.uring_reads - uring_read_start),
           (unsigned long long)(store.raw.uring_setups - setup_start),
           (unsigned long long)(store.raw.uring_reuses - reuse_start),
           (unsigned long long)(
               store.raw.uring_fallbacks - uring_fallback_start),
           (unsigned long long)checksum);
    dsv4_expert_cache_close(&cache);
    dsv4_store_close(&store);
    return 0;
}
