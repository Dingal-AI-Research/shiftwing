#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../st.h"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int expert_payload_name(const char *name) {
    size_t n = strlen(name);
    return strstr(name, ".mlp.experts.") &&
           n >= 7 && !strcmp(name + n - 7, ".weight");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s SNAPSHOT_DIR\n", argv[0]);
        return 2;
    }
    int limit = getenv("IO_TENSORS") ? atoi(getenv("IO_TENSORS")) : 20;
    int reps = getenv("IO_REPS") ? atoi(getenv("IO_REPS")) : 5;
    if (limit < 1 || limit > 512 || reps < 1 || reps > 100) return 2;
    shards S;
    st_init(&S, argv[1]);
    const char **names = calloc((size_t)limit * 2, sizeof(*names));
    void **out = calloc((size_t)limit * 2, sizeof(*out));
    int n = 0;
    for (int i = 0; i < S.n && n + 2 <= limit * 2; i++) {
        st_tensor *t = &S.t[i];
        if (!expert_payload_name(t->name)) continue;
        char scale[1024];
        snprintf(scale, sizeof(scale), "%s.qs", t->name);
        st_tensor *s = st_find(&S, scale);
        if (!s) continue;
        names[n] = t->name;
        out[n++] = malloc((size_t)t->nbytes);
        names[n] = s->name;
        out[n++] = malloc((size_t)s->nbytes);
    }
    if (n < 2) {
        fprintf(stderr, "no converted expert payloads found\n");
        return 1;
    }
    uint64_t one = 0;
    for (int i = 0; i < n; i++) {
        st_tensor *t = st_find(&S, names[i]);
        one += (uint64_t)t->nbytes;
        posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    }
    double t0 = now_s();
    for (int r = 0; r < reps; r++) st_read_raw_batch(&S, names, out, n, 1);
    double dt = now_s() - t0;
    uint64_t checksum = 0;
    for (int i = 0; i < n; i++) {
        st_tensor *t = st_find(&S, names[i]);
        unsigned char *p = out[i];
        checksum = checksum * 1315423911u + p[0] + p[t->nbytes - 1];
        free(out[i]);
    }
    double gib = (double)one * reps / (1024. * 1024. * 1024.);
    printf("[TIER_IO] tensors=%d reps=%d logical=%.3fGiB time=%.6fs "
           "throughput=%.3fGiB/s direct=%.3fGiB uring-batches=%llu "
           "uring-setups=%llu uring-reuses=%llu uring-fallbacks=%llu "
           "checksum=%llu\n",
           n, reps, gib, dt, gib / dt,
           (double)S.direct_bytes / (1024. * 1024. * 1024.),
           (unsigned long long)S.uring_batches,
           (unsigned long long)S.uring_setups,
           (unsigned long long)S.uring_reuses,
           (unsigned long long)S.uring_fallbacks,
           (unsigned long long)checksum);
    free(names); free(out);
    return 0;
}
