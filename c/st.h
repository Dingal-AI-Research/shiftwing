/* Indicizzazione e lettura on-demand di tensori da piu' file safetensors.
 * Equivale a Shards in engine.py, ma:
 *   - legge con pread (niente mmap) + posix_fadvise(DONTNEED) -> le pagine NON
 *     restano residenti nel processo. E' la correzione del bug di RSS: cosi' la
 *     RAM di picco resta densa+cache, non l'intero modello. (vedi memoria mmap-rss-bug)
 *   - converte sempre in float32 in uscita (BF16/F16/F32 supportati). */
#ifndef ST_H
#define ST_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include "json.h"
#include "compat.h"
#include "uring.h"

/* tetto sulla dimensione dell'header safetensors: gli header reali sono piccoli
 * (KB..pochi MB). Un file crafted che dichiara un hlen enorme causerebbe una
 * malloc gigante prima ancora di leggere: lo respingiamo. */
#define ST_MAX_HEADER (512ll << 20)
#define ST_MAX_SHARDS 4096

typedef struct {
    char   *name;
    int     fd;
    int64_t off;       /* offset assoluto del dato dentro al file */
    int64_t nbytes;
    int     dtype;     /* 0=BF16 1=F16 2=F32 */
    int64_t numel;
} st_tensor;

typedef struct {
    st_tensor *t;
    int        n, cap;
    int        fds[ST_MAX_SHARDS];
    int        dfds[ST_MAX_SHARDS];  /* gemelli O_DIRECT (aperti pigramente): -2 = non ancora provato */
    char      *paths[ST_MAX_SHARDS];
    int        nfd;
    int       *hidx;      /* hash map nome->indice (open addressing): con ~120k tensori
                           * (GLM: 256 expert x 78 layer x 3 x 2) la scansione lineare
                           * costava decine di secondi/token (misurato sul primo run reale) */
    int        hcap;
    uint64_t   read_bytes, direct_bytes, direct_fallbacks;
    uint64_t   uring_batches, uring_reads, uring_fallbacks;
    uint64_t   uring_setups, uring_reuses;
} shards;
static uint64_t st_hash(const char *s){
    uint64_t h=1469598103934665603ULL;
    while(*s){ h^=(unsigned char)*s++; h*=1099511628211ULL; }
    return h;
}

static int st_dtype_code(const char *s) {
    if (!strcmp(s, "BF16")) return 0;
    if (!strcmp(s, "F16"))  return 1;
    if (!strcmp(s, "F32"))  return 2;
    if (!strcmp(s, "U8"))   return 3;   /* dati quantizzati (int4 packed / int8) */
    if (!strcmp(s, "I8"))   return 3;
    fprintf(stderr, "unsupported dtype: %s\n", s); exit(1);
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {            /* subnormale o zero */
        if (man == 0) u = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; u = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {  /* inf/nan */
        u = sign | 0x7F800000 | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

static int st_open_fd(shards *S, const char *path) {
    for (int i = 0; i < S->nfd; i++) if (!strcmp(S->paths[i], path)) return S->fds[i];
    int fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    S->paths[S->nfd] = strdup(path); S->fds[S->nfd] = fd;
#ifdef O_DIRECT
    S->dfds[S->nfd] = open(path, COMPAT_O_RDONLY | O_DIRECT);   /* eager: lookup poi thread-safe */
#elif defined(__APPLE__) || defined(_WIN32)
    S->dfds[S->nfd] = compat_open_direct(path);          /* macOS: F_NOCACHE; Windows: NO_BUFFERING */
#else
    S->dfds[S->nfd] = -1;                                /* niente equivalente: solo buffered */
#endif
    S->nfd++;
    return fd;
}

/* fd gemello O_DIRECT dello stesso file (bypassa la page cache: il buffered read su
 * ext4-in-VHDX si strozza a ~0.8 GB/s, O_DIRECT arriva a 2.3+; misurato). -1 se non disponibile. */
static int st_direct_fd(shards *S, int fd) {
    for (int i = 0; i < S->nfd; i++) if (S->fds[i] == fd) return S->dfds[i];
    return -1;
}

static int st_env_enabled(const char *name) {
    const char *v = getenv(name);
    return v && atoi(v) != 0;
}

/* O_DIRECT requires aligned file offsets, buffers, and request lengths. Tensor
 * payloads in safetensors are not page aligned, so read the enclosing 4 KiB
 * extent into a bounce buffer and copy only the requested bytes. A filesystem
 * that rejects O_DIRECT is a normal runtime condition: callers fall back to
 * the exact buffered pread path. */
#define ST_DIRECT_ALIGN 4096u
static int st_pread_direct_try(shards *S, st_tensor *t, void *out) {
    int fd = st_direct_fd(S, t->fd);
    if (fd < 0 || t->nbytes <= 0) return -1;
    int64_t mask = (int64_t)ST_DIRECT_ALIGN - 1;
    int64_t aligned_off = t->off & ~mask;
    size_t delta = (size_t)(t->off - aligned_off);
    size_t need = delta + (size_t)t->nbytes;
    size_t request = (need + ST_DIRECT_ALIGN - 1) & ~(size_t)(ST_DIRECT_ALIGN - 1);
    void *bounce = NULL;
    if (posix_memalign(&bounce, ST_DIRECT_ALIGN, request) != 0) return -1;
    size_t got = 0;
    while (got < need) {
        ssize_t n = pread(fd, (char *)bounce + got, request - got,
                          aligned_off + (int64_t)got);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { free(bounce); return -1; }
        got += (size_t)n;
    }
    memcpy(out, (char *)bounce + delta, (size_t)t->nbytes);
    free(bounce);
    __atomic_fetch_add(&S->read_bytes, (uint64_t)t->nbytes, __ATOMIC_RELAXED);
    __atomic_fetch_add(&S->direct_bytes, (uint64_t)t->nbytes, __ATOMIC_RELAXED);
    return 0;
}

/* indicizza tutti i model-*.safetensors in snap_dir */
/* pread completo: chunk-loop (una singola pread si ferma a ~2^31 byte su Linux
 * — i tensori bf16 grandi la superano), riprova su EINTR e riporta un errore
 * ONESTO: perror stampava "Success" su una short-read (errno resta 0), lo
 * stesso sintomo corretto in glm.c per #236. ST_PREAD_CHUNK e' sovrascrivibile
 * per i test. EN: full pread — chunk loop (one pread caps at ~2^31 bytes and
 * big bf16 tensors exceed it), EINTR retry, honest short-read errors.
 * Exits on failure, like every st.h reader. */
#ifndef ST_PREAD_CHUNK
#define ST_PREAD_CHUNK (1u << 30)
#endif
static void st_pread_full(int fd, void *buf, int64_t n, int64_t off, const char *tag) {
    char *p = (char *)buf;
    int64_t got = 0;
    while (got < n) {
        int64_t want = n - got;
        if (want > (int64_t)ST_PREAD_CHUNK) want = ST_PREAD_CHUNK;
        ssize_t r = pread(fd, p + got, (size_t)want, off + got);
        if (r < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "%s: %s (off %lld, %lld/%lld bytes)\n", tag, strerror(errno),
                    (long long)off, (long long)got, (long long)n);
            exit(1);
        }
        if (r == 0) {
            fprintf(stderr, "%s: short read at EOF (off %lld, %lld/%lld bytes) — truncated file?\n",
                    tag, (long long)off, (long long)got, (long long)n);
            exit(1);
        }
        got += r;
    }
}

static void st_init(shards *S, const char *snap_dir) {
    memset(S, 0, sizeof(*S));
    S->cap = 4096; S->t = calloc(S->cap, sizeof(st_tensor));
    /* raccoglie ordinatamente i nomi dei file shard */
    static char files[ST_MAX_SHARDS][1024]; int nf = 0;
    DIR *d = opendir(snap_dir); struct dirent *e;
    if (!d) { perror(snap_dir); exit(1); }
    while ((e = readdir(d))) {
        const char *dot = strrchr(e->d_name, '.');
        if (dot && !strcmp(dot, ".safetensors")) {  /* model.safetensors o model-0000N-of-... */
            if (nf >= ST_MAX_SHARDS) { fprintf(stderr, "too many shards (>%d): raise ST_MAX_SHARDS\n", ST_MAX_SHARDS); exit(1); }
            snprintf(files[nf++], 1024, "%s/%s", snap_dir, e->d_name);
        }
    }
    closedir(d);
    for (int a = 0; a < nf; a++) for (int b = a+1; b < nf; b++)
        if (strcmp(files[a], files[b]) > 0) { char tmp[1024]; strcpy(tmp, files[a]); strcpy(files[a], files[b]); strcpy(files[b], tmp); }

    for (int fi = 0; fi < nf; fi++) {
        int fd = st_open_fd(S, files[fi]);
        struct stat sst;
        if (fstat(fd, &sst) != 0) { perror("fstat shard"); exit(1); }
        int64_t fsz = (int64_t)sst.st_size;
        uint64_t hlen;
        st_pread_full(fd, &hlen, 8, 0, "pread hlen");
        /* file malevolo/troncato: hlen deve stare nel file dopo gli 8 byte di
         * prefisso e sotto il tetto. Senza questo bound hlen+1 puo' andare in
         * overflow (malloc(0) e poi hdr[hlen]=0 fuori limiti) o forzare una
         * malloc gigante. */
        if (fsz < 8 || hlen > (uint64_t)(fsz - 8) || hlen > (uint64_t)ST_MAX_HEADER) {
            fprintf(stderr, "%s: bad safetensors header length %llu (file %lld bytes)\n",
                    files[fi], (unsigned long long)hlen, (long long)fsz); exit(1); }
        char *hdr = malloc(hlen + 1);
        if (!hdr) { perror("malloc safetensors header"); exit(1); }
        st_pread_full(fd, hdr, (int64_t)hlen, 8, "pread hdr");
        hdr[hlen] = 0;
        int64_t data_start = 8 + (int64_t)hlen;
        char *arena = NULL;
        jval *root = json_parse(hdr, &arena);
        if (!root || root->t != J_OBJ) {
            fprintf(stderr, "%s: safetensors header is not a JSON object\n", files[fi]); exit(1); }
        for (int i = 0; i < root->len; i++) {
            const char *name = root->keys[i];
            if (!strcmp(name, "__metadata__")) continue;
            jval *m = root->kids[i];
            jval *dt = json_get(m, "dtype");
            jval *off = json_get(m, "data_offsets");
            jval *shp = json_get(m, "shape");
            /* un header crafted puo' omettere i campi o dare tipi sbagliati:
             * senza questi guard si dereferenzia NULL (json_get) o si legge
             * off->kids[0/1] oltre i limiti dell'array. */
            if (!dt || dt->t != J_STR || !off || off->t != J_ARR || off->len < 2 ||
                !shp || shp->t != J_ARR) {
                fprintf(stderr, "%s: tensor '%s' has malformed dtype/data_offsets/shape\n",
                        files[fi], name); exit(1); }
            int64_t a0 = (int64_t)off->kids[0]->num, b0 = (int64_t)off->kids[1]->num;
            /* offset dichiarati dal file: non-negativi, ordinati e dentro al
             * file. Altrimenti nbytes=b0-a0 diventa negativo -> malloc((size_t))
             * gigante e la memcpy in st_read_f32 sfora il buffer del chiamante;
             * oppure off punta fuori dal file. */
            if (a0 < 0 || b0 < a0 || data_start + b0 > fsz) {
                fprintf(stderr, "%s: tensor '%s' data_offsets [%lld,%lld] out of file bounds (%lld)\n",
                        files[fi], name, (long long)a0, (long long)b0, (long long)fsz); exit(1); }
            int64_t numel = 1; for (int k = 0; k < shp->len; k++) numel *= (int64_t)shp->kids[k]->num;
            if (S->n == S->cap) { S->cap *= 2; S->t = realloc(S->t, S->cap*sizeof(st_tensor)); }
            st_tensor *t = &S->t[S->n++];
            t->name = strdup(name); t->fd = fd; t->off = data_start + a0;
            t->nbytes = b0 - a0; t->dtype = st_dtype_code(dt->str); t->numel = numel;
        }
        free(arena); /* i jval restano leakati: ok, una tantum all'avvio */
        free(hdr);
    }
    /* indice hash costruito a fine indicizzazione (gli indici restano validi dopo i realloc) */
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = malloc(S->hcap * sizeof(int));
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t h = st_hash(S->t[i].name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (S->hcap - 1);
        S->hidx[h] = i;
    }
}

static st_tensor *st_find(shards *S, const char *name) {
    if (S->hidx) {
        uint64_t h = st_hash(name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) {
            st_tensor *t = &S->t[S->hidx[h]];
            if (!strcmp(t->name, name)) return t;
            h = (h + 1) & (S->hcap - 1);
        }
        return NULL;
    }
    for (int i = 0; i < S->n; i++) if (!strcmp(S->t[i].name, name)) return &S->t[i];
    return NULL;
}
static int st_has(shards *S, const char *name) { return st_find(S, name) != NULL; }

/* prefetch ASINCRONO: dice al kernel di iniziare a leggere le pagine del tensore in
 * background (readahead). Serve a sovrapporre l'I/O degli expert col calcolo: si
 * prefetcha tutto il set di expert di un layer, poi le pread sincrone trovano la cache
 * gia' calda. No-op se il tensore non esiste (es. il primo .qs prima della lettura). */
static void st_prefetch(shards *S, const char *name) {
    st_tensor *t = st_find(S, name);
    if (t) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* legge un tensore in un buffer float32 fornito dal chiamante (numel float).
 * drop=1 -> consiglia al kernel di scartare le pagine (per gli expert in streaming). */
static int64_t st_read_f32(shards *S, const char *name, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    void *raw = malloc(t->nbytes);
    if (!raw) { fprintf(stderr, "malloc %lld bytes for tensor %s failed\n", (long long)t->nbytes, name); exit(1); }
    st_pread_full(t->fd, raw, t->nbytes, t->off, "pread data");
    if (t->dtype == 2) {
        memcpy(out, raw, t->nbytes);
    } else if (t->dtype == 0) {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = bf16_to_f32(p[i]);
    } else {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = f16_to_f32(p[i]);
    }
    free(raw);
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    return t->numel;
}

static int64_t st_numel(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->numel : -1;
}
static int64_t st_nbytes(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->nbytes : -1;
}

/* legge i byte GREZZI di un tensore (nessuna conversione di dtype): per i pesi gia'
 * quantizzati int4/int8 del nostro container (dtype U8). drop=1 -> fadvise DONTNEED. */
static void st_read_raw(shards *S, const char *name, void *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    int try_direct = st_env_enabled("DIRECT") && t->nbytes >= ST_DIRECT_ALIGN;
    if (!try_direct || st_pread_direct_try(S, t, out) != 0) {
        if (try_direct)
            __atomic_fetch_add(&S->direct_fallbacks, 1, __ATOMIC_RELAXED);
        st_pread_full(t->fd, out, t->nbytes, t->off, "pread raw");
        __atomic_fetch_add(&S->read_bytes, (uint64_t)t->nbytes, __ATOMIC_RELAXED);
    }
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
}

typedef struct {
    st_tensor *tensor;
    void *out, *io_buf;
    size_t io_len, delta, need;
    int fd;
} st_batch_req;

#ifdef __linux__
/*
 * A routed-expert miss calls the batch reader once per layer. Recreating an
 * io_uring and allocating up to 60 aligned buffers for every call adds
 * synchronization and allocator work to the storage path. Each caller thread
 * owns one reusable ring and a bounded aligned-buffer array. The largest
 * observed batch determines its retained capacity.
 */
typedef struct {
    ColiUring ring;
    unsigned entries;
    int ready, disabled;
    void **buffer;
    size_t *capacity;
    int nbuffer;
} st_uring_thread_cache;

static _Thread_local st_uring_thread_cache st_uring_cache;

static int st_uring_cache_buffers(st_uring_thread_cache *cache, int count) {
    if (cache->nbuffer >= count) return 0;
    void **buffer = calloc((size_t)count, sizeof(*buffer));
    size_t *capacity = calloc((size_t)count, sizeof(*capacity));
    if (!buffer || !capacity) {
        free(buffer); free(capacity); return -1;
    }
    for (int i = 0; i < cache->nbuffer; i++) {
        buffer[i] = cache->buffer[i];
        capacity[i] = cache->capacity[i];
    }
    free(cache->buffer); free(cache->capacity);
    cache->buffer = buffer; cache->capacity = capacity;
    cache->nbuffer = count;
    return 0;
}

static int st_uring_cache_buffer(st_uring_thread_cache *cache, int index,
                                 size_t bytes, void **out) {
    if (st_uring_cache_buffers(cache, index + 1) != 0) return -1;
    if (cache->capacity[index] < bytes) {
        void *next = NULL;
        if (posix_memalign(&next, ST_DIRECT_ALIGN, bytes) != 0) return -1;
        free(cache->buffer[index]);
        cache->buffer[index] = next;
        cache->capacity[index] = bytes;
    }
    *out = cache->buffer[index];
    return 0;
}

static ColiUring *st_uring_cache_ring(shards *S, unsigned entries,
                                      unsigned workers) {
    st_uring_thread_cache *cache = &st_uring_cache;
    if (cache->disabled) return NULL;
    if (cache->ready && cache->entries >= entries) {
        __atomic_fetch_add(&S->uring_reuses, 1, __ATOMIC_RELAXED);
        return &cache->ring;
    }
    if (cache->ready) {
        coli_uring_close(&cache->ring);
        cache->ready = 0;
    }
    if (coli_uring_init(&cache->ring, entries) != 0) {
        cache->disabled = 1;
        return NULL;
    }
    if (workers) coli_uring_set_workers(&cache->ring, workers);
    cache->entries = *cache->ring.sq_entries;
    cache->ready = 1;
    __atomic_fetch_add(&S->uring_setups, 1, __ATOMIC_RELAXED);
    return &cache->ring;
}

static void st_uring_cache_disable(void) {
    st_uring_thread_cache *cache = &st_uring_cache;
    if (cache->ready) coli_uring_close(&cache->ring);
    cache->ready = 0;
    cache->disabled = 1;
}
#endif

/* Read several packed tensors as one storage transaction. This is used for
 * routed experts, whose gate/up/down payloads and scale arrays are independent
 * safetensors entries. io_uring is opportunistic: kernels, containers, and
 * filesystems may disable it, in which case the same requests are replayed
 * through the tested synchronous path. */
static int st_read_raw_batch_uring(shards *S, const char **names, void **outs,
                                   int nreq, int direct) {
#ifdef __linux__
    if (nreq <= 0) return 0;
    int persistent = st_env_enabled("URING_PERSIST");
    st_batch_req *req = calloc((size_t)nreq, sizeof(*req));
    if (!req) return -1;
    int ok = 1;
    for (int i = 0; i < nreq; i++) {
        st_tensor *t = st_find(S, names[i]);
        if (!t || t->nbytes <= 0 || t->nbytes > UINT32_MAX) { ok = 0; break; }
        req[i].tensor = t; req[i].out = outs[i];
        if (direct) {
            int fd = st_direct_fd(S, t->fd);
            int64_t mask = (int64_t)ST_DIRECT_ALIGN - 1;
            int64_t aligned_off = t->off & ~mask;
            size_t delta = (size_t)(t->off - aligned_off);
            size_t need = delta + (size_t)t->nbytes;
            size_t len = (need + ST_DIRECT_ALIGN - 1) &
                         ~(size_t)(ST_DIRECT_ALIGN - 1);
            int alloc_error = persistent
                ? st_uring_cache_buffer(&st_uring_cache, i, len,
                                        &req[i].io_buf)
                : posix_memalign(&req[i].io_buf, ST_DIRECT_ALIGN, len);
            if (fd < 0 || len > UINT32_MAX || alloc_error != 0) {
                ok = 0; break;
            }
            req[i].fd = fd; req[i].io_len = len;
            req[i].delta = delta; req[i].need = need;
            /* Preserve the aligned offset for submission. */
            req[i].tensor = t;
            (void)aligned_off;
        } else {
            req[i].fd = t->fd; req[i].io_buf = outs[i];
            req[i].io_len = (size_t)t->nbytes; req[i].need = req[i].io_len;
        }
    }
    unsigned entries = 1;
    while (entries < (unsigned)nreq) entries <<= 1;
    if (entries < 8) entries = 8;
    ColiUring local_ring, *ring = NULL;
    unsigned workers = getenv("URING_WORKERS") ?
                       (unsigned)atoi(getenv("URING_WORKERS")) : 4u;
    if (ok && persistent) ring = st_uring_cache_ring(S, entries, workers);
    if (ok && !persistent && coli_uring_init(&local_ring, entries) == 0) {
        ring = &local_ring;
        __atomic_fetch_add(&S->uring_setups, 1, __ATOMIC_RELAXED);
        if (workers) coli_uring_set_workers(ring, workers);
    }
    if (!ok || !ring) {
        for (int i = 0; i < nreq; i++)
            if (direct && !persistent) free(req[i].io_buf);
        free(req); return -1;
    }
    for (int i = 0; i < nreq; i++) {
        int64_t off = req[i].tensor->off;
        if (direct) off &= ~((int64_t)ST_DIRECT_ALIGN - 1);
        if (coli_uring_prep_read(ring, req[i].fd, req[i].io_buf,
                                 req[i].io_len, off, (uint64_t)i) != 0) {
            ok = 0; break;
        }
    }
    int done = 0;
    if (ok && coli_uring_enter(ring, (unsigned)nreq) < 0) ok = 0;
    while (ok && done < nreq) {
        struct io_uring_cqe cqe;
        if (!coli_uring_peek(ring, &cqe)) {
            if (coli_uring_enter(ring, 1) < 0) { ok = 0; break; }
            continue;
        }
        int i = (int)cqe.user_data;
        if (i < 0 || i >= nreq || cqe.res < 0 ||
            (size_t)cqe.res < req[i].need) ok = 0;
        done++;
    }
    if (!persistent) coli_uring_close(ring);
    else if (!ok) st_uring_cache_disable();
    if (ok) {
        uint64_t bytes = 0;
        for (int i = 0; i < nreq; i++) {
            if (direct)
                memcpy(req[i].out, (char *)req[i].io_buf + req[i].delta,
                       (size_t)req[i].tensor->nbytes);
            bytes += (uint64_t)req[i].tensor->nbytes;
        }
        __atomic_fetch_add(&S->read_bytes, bytes, __ATOMIC_RELAXED);
        if (direct) __atomic_fetch_add(&S->direct_bytes, bytes, __ATOMIC_RELAXED);
        __atomic_fetch_add(&S->uring_batches, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&S->uring_reads, (uint64_t)nreq, __ATOMIC_RELAXED);
    }
    for (int i = 0; i < nreq; i++)
        if (direct && !persistent) free(req[i].io_buf);
    free(req);
    return ok ? 0 : -1;
#else
    (void)S; (void)names; (void)outs; (void)nreq; (void)direct;
    return -1;
#endif
}

static void st_read_raw_batch(shards *S, const char **names, void **outs,
                              int nreq, int drop) {
    int direct = st_env_enabled("DIRECT");
    int used_uring = st_env_enabled("URING") &&
                     st_read_raw_batch_uring(S, names, outs, nreq, direct) == 0;
    if (!used_uring) {
        if (st_env_enabled("URING"))
            __atomic_fetch_add(&S->uring_fallbacks, 1, __ATOMIC_RELAXED);
        for (int i = 0; i < nreq; i++) st_read_raw(S, names[i], outs[i], drop);
        return;
    }
    if (drop)
        for (int i = 0; i < nreq; i++) {
            st_tensor *t = st_find(S, names[i]);
            if (t) posix_fadvise(t->fd, t->off, t->nbytes,
                                 POSIX_FADV_DONTNEED);
        }
}

/* legge una FETTA di un tensore: n_elems a partire dall'elemento elem_off.
 * Serve per gli expert fusi di GLM (un tensore = blocco [E, ...]): si legge il
 * solo expert richiesto via pread del sotto-range, niente lettura dell'intero blocco. */
static void st_read_slice_f32(shards *S, const char *name, int64_t elem_off, int64_t n_elems, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    int64_t boff = t->off + elem_off * esz, nb = n_elems * esz;
    void *raw = malloc(nb);
    st_pread_full(t->fd, raw, nb, boff, "pread slice");
    if (t->dtype == 2) memcpy(out, raw, nb);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = bf16_to_f32(p[i]); }
    else { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    if (drop) posix_fadvise(t->fd, boff, nb, POSIX_FADV_DONTNEED);
}

#endif
