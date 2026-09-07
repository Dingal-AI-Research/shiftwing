#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../backend_cuda.h"

#define BENCH_TOPK 6
#define BENCH_HIDDEN 4096
#define BENCH_INTERMEDIATE 2048

static double bench_now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static size_t expert_bytes(void) {
    size_t hidden = BENCH_HIDDEN;
    size_t intermediate = BENCH_INTERMEDIATE;
    return intermediate * hidden / 2 + intermediate * (hidden / 32) +
           hidden * intermediate / 2 + hidden * (intermediate / 32) +
           intermediate * hidden / 2 + intermediate * (hidden / 32);
}

static double run_uploads(ColiCuda *cuda, void **device,
                          const unsigned char *source, int repetitions) {
    size_t bytes = expert_bytes();
    double started = bench_now_seconds();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        for (int route = 0; route < BENCH_TOPK; route++)
            if (coli_cuda_upload(
                    cuda, device[route], source + (size_t)route * bytes,
                    bytes))
                return -1.0;
        if (coli_cuda_sync(cuda)) return -1.0;
    }
    return bench_now_seconds() - started;
}

int main(void) {
    int repetitions = getenv("UPLOAD_REPS") ?
        atoi(getenv("UPLOAD_REPS")) : 5;
    if (repetitions < 1 || repetitions > 100) return 2;
    ColiCuda *cuda = NULL;
    if (coli_cuda_create(&cuda, 0)) {
        fprintf(stderr, "CUDA create: %s\n", coli_cuda_last_error());
        return 1;
    }
    size_t bytes = expert_bytes();
    size_t total = (size_t)BENCH_TOPK * bytes;
    unsigned char *pageable = (unsigned char *)malloc(total);
    unsigned char *pinned = NULL;
    void *device[BENCH_TOPK] = {0};
    if (!pageable ||
        coli_cuda_malloc_host(cuda, (void **)&pinned, total)) {
        fprintf(stderr, "host allocation: %s\n", coli_cuda_last_error());
        free(pageable);
        coli_cuda_destroy(cuda);
        return 1;
    }
    memset(pageable, 0xa5, total);
    memcpy(pinned, pageable, total);
    double allocation_started = bench_now_seconds();
    for (int route = 0; route < BENCH_TOPK; route++)
        if (coli_cuda_malloc(cuda, &device[route], bytes)) {
            fprintf(stderr, "device allocation: %s\n",
                    coli_cuda_last_error());
            return 1;
        }
    if (coli_cuda_sync(cuda)) {
        fprintf(stderr, "device allocation sync: %s\n",
                coli_cuda_last_error());
        return 1;
    }
    double allocation_seconds = bench_now_seconds() - allocation_started;
    double pageable_seconds =
        run_uploads(cuda, device, pageable, repetitions);
    double pinned_seconds =
        run_uploads(cuda, device, pinned, repetitions);
    double staged_started = bench_now_seconds();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        memcpy(pinned, pageable, total);
        for (int route = 0; route < BENCH_TOPK; route++)
            if (coli_cuda_upload(
                    cuda, device[route],
                    pinned + (size_t)route * bytes, bytes)) {
                fprintf(stderr, "staged upload: %s\n",
                        coli_cuda_last_error());
                return 1;
            }
        if (coli_cuda_sync(cuda)) {
            fprintf(stderr, "staged sync: %s\n", coli_cuda_last_error());
            return 1;
        }
    }
    double staged_seconds = bench_now_seconds() - staged_started;
    double gib = (double)total * repetitions /
                 (1024.0 * 1024.0 * 1024.0);
    printf("DSV4_UPLOAD status=pass expert_bytes=%zu topk=%d reps=%d "
           "allocation_s=%.6f pageable_s=%.6f pageable_gib_s=%.6f "
           "pinned_s=%.6f pinned_gib_s=%.6f staged_s=%.6f "
           "staged_gib_s=%.6f\n",
           bytes, BENCH_TOPK, repetitions, allocation_seconds,
           pageable_seconds, gib / pageable_seconds,
           pinned_seconds, gib / pinned_seconds,
           staged_seconds, gib / staged_seconds);
    for (int route = 0; route < BENCH_TOPK; route++)
        if (device[route]) coli_cuda_free(cuda, device[route]);
    coli_cuda_sync(cuda);
    coli_cuda_free_host(cuda, pinned);
    free(pageable);
    coli_cuda_destroy(cuda);
    return 0;
}
