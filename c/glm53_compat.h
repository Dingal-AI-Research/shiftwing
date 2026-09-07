/* Compatibility shims for the vendored GLM-5.3 engine.
 *
 * glm53.c comes from colibri (Apache-2.0; see NOTICE), whose st.h and tok.h are
 * slightly ahead of this project's. Three symbols it uses do not exist here.
 * They live in their own header rather than in the shared ones so the GLM port
 * cannot perturb the Qwen3.5/Ornith engine, which is the lane currently serving
 * LocalForge.
 *
 * Include after st.h / tok.h.
 */
#ifndef GLM53_COMPAT_H
#define GLM53_COMPAT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "st.h"
#include "tok.h"

/* st_read_f32 with the destination capacity stated by the caller.
 *
 * The bare reader trusts the header's element count and writes that many
 * floats. A shard whose declared shape disagrees with the buffer the caller
 * sized would overrun it, so the count is checked instead of assumed. Returns
 * the elements written. */
static int64_t st_read_f32_cap(shards *S, const char *name, float *out,
                               int64_t capacity, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->numel > capacity) {
        fprintf(stderr, "tensor %s holds %lld elements, buffer takes %lld\n",
                name, (long long)t->numel, (long long)capacity);
        exit(1);
    }
    return st_read_f32(S, name, out, drop);
}

/* Release a tokenizer. tok_load allocates the id tables, the added-token list
 * and both hash maps; the engine loads a tokenizer per serve session, so
 * leaking one per session is not acceptable here even though the CLI exits
 * immediately after. The string arena backing vocab/merges keys belongs to the
 * parsed JSON and is freed by its owner, so only the containers are released. */
static void tok_free(Tok *T) {
    if (!T) return;
    free(T->vocab.e);   T->vocab.e = NULL;   T->vocab.cap = 0;
    free(T->merges.e);  T->merges.e = NULL;  T->merges.cap = 0;
    free(T->id2str);    T->id2str = NULL;
    free(T->id_added);  T->id_added = NULL;
    free(T->id_special);T->id_special = NULL;
    free(T->sp);        T->sp = NULL;        T->nsp = 0;
    T->n_ids = 0;
}

/* Put stdio into binary mode for the serve protocol, which frames binary
 * messages on stdin/stdout. A no-op on POSIX, where there is no text mode;
 * kept as a named call so the Windows build has one place to change. */
static void coli_serve_binary_mode(void) {
#if defined(_WIN32)
    extern int _setmode(int, int);
    _setmode(0, 0x8000 /* _O_BINARY */);
    _setmode(1, 0x8000);
#endif
}

/* Release everything st_init acquired.
 *
 * The engine loads a model per serve session, so the shard table cannot leak
 * per session. st_init strdups every tensor name and every shard path, opens a
 * buffered fd per shard plus a lazy O_DIRECT twin, and builds the name->index
 * hash; all of that is freed here. The twin is -1 when the platform has no
 * O_DIRECT and -2 when it has not been attempted, so only non-negative
 * descriptors are closed. Zeroing at the end makes a second call a no-op
 * rather than a double free.
 *
 * The jval arena parsed from each shard header is deliberately not touched:
 * st_init frees the arena and documents the jvals as a one-time startup leak,
 * and it does not hand this struct a pointer to them. */
static void st_destroy(shards *S) {
    if (!S) return;
    for (int i = 0; i < S->n; i++) { free(S->t[i].name); S->t[i].name = NULL; }
    free(S->t);
    free(S->hidx);
    for (int i = 0; i < S->nfd; i++) {
        free(S->paths[i]);
        if (S->fds[i]  >= 0) close(S->fds[i]);
        if (S->dfds[i] >= 0) close(S->dfds[i]);
    }
    memset(S, 0, sizeof(*S));
}

/* O_DIRECT read of a raw (fd, offset, length) range.
 *
 * st.h has this only for an st_tensor; the expert loader works in raw triples
 * because an expert's three matrices are addressed by offset inside a shard.
 *
 * It matters more than it looks. Measured on this host's model shards, with
 * the engine resident and the page cache full:
 *
 *     O_DIRECT sequential   408 MB/s
 *     buffered  sequential  15.9 MB/s
 *
 * Buffered reads have to evict to make room, and under memory pressure that
 * collapses. The expert path is the hottest read in the engine -- roughly
 * 4.5 GB per generated token -- so it is exactly the path that must not go
 * through the page cache. Falls back to the caller's buffered read whenever
 * the twin descriptor is missing or the filesystem refuses O_DIRECT, which is
 * a normal runtime condition rather than an error.
 *
 * Safe to call from an OpenMP parallel region: it allocates its own bounce
 * buffer and only reads shared state. */
static int st_pread_direct_raw(shards *S, int fd, int64_t off, int64_t nbytes,
                               void *out) {
    if (nbytes <= 0) return -1;
    int dfd = st_direct_fd(S, fd);
    if (dfd < 0) return -1;
    int64_t mask = (int64_t)ST_DIRECT_ALIGN - 1;
    int64_t aligned_off = off & ~mask;
    size_t delta = (size_t)(off - aligned_off);
    size_t need = delta + (size_t)nbytes;
    size_t request = (need + ST_DIRECT_ALIGN - 1) & ~(size_t)(ST_DIRECT_ALIGN - 1);
    void *bounce = NULL;
    if (posix_memalign(&bounce, ST_DIRECT_ALIGN, request) != 0) return -1;
    size_t got = 0;
    while (got < need) {
        ssize_t n = pread(dfd, (char *)bounce + got, request - got,
                          aligned_off + (int64_t)got);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { free(bounce); return -1; }
        got += (size_t)n;
    }
    memcpy(out, (char *)bounce + delta, (size_t)nbytes);
    free(bounce);
    __atomic_fetch_add(&S->read_bytes, (uint64_t)nbytes, __ATOMIC_RELAXED);
    __atomic_fetch_add(&S->direct_bytes, (uint64_t)nbytes, __ATOMIC_RELAXED);
    return 0;
}

/* One io_uring submission for many raw (fd, offset, length) ranges.
 *
 * st.h has a batch reader, but it addresses tensors by name and the expert
 * loader works in raw triples: an expert is up to six pieces at known offsets
 * inside shards. This is the same submission pattern, over ranges.
 *
 * O_DIRECT only. The buffered path is what this exists to avoid -- measured
 * 15.9 MB/s against 408 MB/s on this host's shards -- so a missing twin
 * descriptor is a reason to fall back to the caller's per-read path, not a
 * reason to read through the page cache.
 *
 * Returns 0 when every range arrived complete, -1 otherwise. On -1 nothing is
 * written to the caller's buffers and the caller must read them itself; a
 * kernel or filesystem without io_uring is a normal condition. */
typedef struct {
    int    fd;
    int64_t off_aligned;
    size_t io_len;        /* aligned length submitted */
    size_t delta;         /* where the wanted bytes start inside the bounce */
    size_t need;          /* delta + nbytes: the least that must arrive */
    size_t nbytes;        /* wanted bytes */
    void  *bounce;
    void  *out;
} GlmRange;

#define GLM_BATCH_MAX 256

/* Ring and scratch, kept across batches.
 *
 * The first version of this created a ring and freed it per batch, and
 * allocated a bounce buffer per range. Measured on the real model that was
 * *slower* than the per-expert preads it replaced -- 36.2 s cold became 53.5 s
 * -- because 338 io_uring_setup/mmap pairs and some thirty thousand 14 MB
 * allocations cost more than the waiting they removed. st.h reaches the same
 * conclusion with URING_PERSIST and its cached buffers. */
typedef struct {
    ColiUring ring;
    int  ready;                 /* ring initialised */
    int  broken;                /* something failed: stop trying */
    void  *buf;                 /* one aligned scratch for a whole batch */
    size_t cap;
} GlmUring;

static void glm_uring_release(GlmUring *u) {
    if (!u) return;
#ifdef __linux__
    if (u->ready) coli_uring_close(&u->ring);
#endif
    u->ready = 0;
    free(u->buf);
    u->buf = NULL;
    u->cap = 0;
}

static int glm_read_batch_uring(shards *S, GlmUring *u, const int *fds,
                                const int64_t *offs, const int64_t *lens,
                                void **outs, int nreq) {
#ifdef __linux__
    if (nreq <= 0) return 0;
    if (!u || u->broken) return -1;
    if (nreq > GLM_BATCH_MAX) return -1;      /* caller chunks; keeps the ring small */
    GlmRange *req = calloc((size_t)nreq, sizeof(*req));
    if (!req) return -1;

    /* Lay the whole batch out in one aligned buffer, grown only when a batch
     * needs more than the last one did. */
    size_t total = 0;
    int ok = 1;
    for (int i = 0; i < nreq; i++) {
        int dfd = st_direct_fd(S, fds[i]);
        if (dfd < 0 || lens[i] <= 0) { ok = 0; break; }
        int64_t mask = (int64_t)ST_DIRECT_ALIGN - 1;
        int64_t aligned = offs[i] & ~mask;
        size_t delta = (size_t)(offs[i] - aligned);
        size_t need = delta + (size_t)lens[i];
        size_t io_len = (need + ST_DIRECT_ALIGN - 1) & ~(size_t)(ST_DIRECT_ALIGN - 1);
        if (io_len > UINT32_MAX) { ok = 0; break; }
        req[i].fd = dfd;          req[i].off_aligned = aligned;
        req[i].io_len = io_len;   req[i].delta = delta;
        req[i].need = need;       req[i].nbytes = (size_t)lens[i];
        req[i].out = outs[i];
        req[i].bounce = (void *)(uintptr_t)total;   /* offset for now */
        total += io_len;
    }
    if (ok && total > u->cap) {
        void *grown = NULL;
        if (posix_memalign(&grown, ST_DIRECT_ALIGN, total) != 0) ok = 0;
        else { free(u->buf); u->buf = grown; u->cap = total; }
    }
    if (ok) for (int i = 0; i < nreq; i++)
        req[i].bounce = (char *)u->buf + (uintptr_t)req[i].bounce;

    if (ok && !u->ready) {
        if (coli_uring_init(&u->ring, GLM_BATCH_MAX) != 0) ok = 0;
        else {
            u->ready = 1;
            __atomic_fetch_add(&S->uring_setups, 1, __ATOMIC_RELAXED);
            const char *w = getenv("URING_WORKERS");
            unsigned workers = w ? (unsigned)atoi(w) : 4u;
            if (workers) coli_uring_set_workers(&u->ring, workers);
        }
    }
    if (!ok || !u->ready) { free(req); return -1; }
    __atomic_fetch_add(&S->uring_reuses, 1, __ATOMIC_RELAXED);

    for (int i = 0; i < nreq && ok; i++)
        if (coli_uring_prep_read(&u->ring, req[i].fd, req[i].bounce, req[i].io_len,
                                 req[i].off_aligned, (uint64_t)i) != 0) ok = 0;

    int done = 0;
    if (ok && coli_uring_enter(&u->ring, (unsigned)nreq) < 0) ok = 0;
    while (ok && done < nreq) {
        struct io_uring_cqe cqe;
        if (!coli_uring_peek(&u->ring, &cqe)) {
            if (coli_uring_enter(&u->ring, 1) < 0) ok = 0;
            continue;
        }
        int i = (int)cqe.user_data;
        /* A short read is a failure here, not something to patch up: the
         * caller's fallback re-reads the whole range correctly. */
        if (i < 0 || i >= nreq || cqe.res < 0 || (size_t)cqe.res < req[i].need)
            ok = 0;
        done++;
    }

    if (ok) {
        uint64_t bytes = 0;
        /* La copia dal rimbalzo alla destinazione e' grossa -- un blocco pieno
         * muove qualche centinaio di MB -- e nel percorso per esperto la
         * facevano i thread OpenMP, ognuno la sua. Farla in serie qui ha
         * spostato lavoro parallelo su un thread solo, ed e' una delle ragioni
         * per cui il primo batch era piu' lento di quello che sostituiva. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < nreq; i++)
            memcpy(req[i].out, (char *)req[i].bounce + req[i].delta, req[i].nbytes);
        for (int i = 0; i < nreq; i++) bytes += (uint64_t)req[i].nbytes;
        __atomic_fetch_add(&S->read_bytes, bytes, __ATOMIC_RELAXED);
        __atomic_fetch_add(&S->direct_bytes, bytes, __ATOMIC_RELAXED);
        __atomic_fetch_add(&S->uring_batches, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&S->uring_reads, (uint64_t)nreq, __ATOMIC_RELAXED);
    } else {
        /* A half-drained ring cannot be reused safely. */
        __atomic_fetch_add(&S->uring_fallbacks, 1, __ATOMIC_RELAXED);
        glm_uring_release(u);
        u->broken = 1;
    }
    free(req);
    return ok ? 0 : -1;
#else
    (void)S; (void)u; (void)fds; (void)offs; (void)lens; (void)outs; (void)nreq;
    return -1;
#endif
}

#endif /* GLM53_COMPAT_H */
