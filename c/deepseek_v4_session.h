#ifndef COLIB_DEEPSEEK_V4_SESSION_H
#define COLIB_DEEPSEEK_V4_SESSION_H

/* Versioned DeepSeek session snapshots. The caller supplies SHA-256 identity
 * bytes for the model manifest, tokenizer/protocol, and native engine source.
 * Attention/cache arrays stay in their native BF16 representation. */

#include "deepseek_v4_limits.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DSV4_SESSION_VERSION 1u
#define DSV4_SESSION_ENDIAN 0x01020304u
#define DSV4_SESSION_MAX_CONTEXT DSV4_MAX_CONTEXT
#define DSV4_SESSION_MAX_BYTES (2ull << 30)

typedef struct {
    uint8_t model[32];
    uint8_t tokenizer_protocol[32];
    uint8_t engine[32];
} dsv4_session_identity;

typedef struct {
    uint32_t position, context, layers, hidden;
    uint32_t sampler_profile; /* 0=deterministic, 1=agent/tool */
    float temperature, top_p;
    uint64_t sampler_rng[4];
    int32_t *history;
    uint16_t *mhc;
    uint16_t *window_kv;
    uint16_t *compressed_kv;
    float *compressor;
    uint64_t history_n, mhc_n, window_kv_n, compressed_kv_n, compressor_n;
} dsv4_session_state;

typedef struct {
    char magic[8];
    uint32_t version, header_bytes, endian, reserved;
    dsv4_session_identity identity;
    uint32_t position, context, layers, hidden;
    uint32_t sampler_profile;
    float temperature, top_p;
    uint64_t sampler_rng[4];
    uint64_t history_n, mhc_n, window_kv_n, compressed_kv_n, compressor_n;
    uint64_t checksum;
} dsv4_session_header;

static uint64_t dsv4_session_hash(uint64_t hash, const void *data, size_t bytes) {
    const uint8_t *input = (const uint8_t *)data;
    for (size_t index = 0; index < bytes; index++) {
        hash ^= input[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

static int dsv4_session_nonzero(const uint8_t value[32]) {
    uint8_t combined = 0;
    for (int index = 0; index < 32; index++) combined |= value[index];
    return combined != 0;
}

static int dsv4_session_identity_valid(const dsv4_session_identity *identity) {
    return identity && dsv4_session_nonzero(identity->model) &&
        dsv4_session_nonzero(identity->tokenizer_protocol) &&
        dsv4_session_nonzero(identity->engine);
}

static int dsv4_session_add_bytes(uint64_t *total, uint64_t count,
                                  uint64_t element) {
    if (count && element > UINT64_MAX / count) return 0;
    uint64_t bytes = count * element;
    if (*total > UINT64_MAX - bytes) return 0;
    *total += bytes;
    return *total <= DSV4_SESSION_MAX_BYTES;
}

static int dsv4_session_layout_valid(const dsv4_session_state *state,
                                      uint64_t *payload_bytes) {
    if (!state || state->position > state->context || !state->position ||
        state->context > DSV4_SESSION_MAX_CONTEXT || state->layers != 43 ||
        state->hidden != 4096 || state->history_n != state->position ||
        !state->history || !state->mhc || !state->window_kv ||
        state->sampler_profile > 1) return 0;
    if ((state->sampler_profile == 0 &&
         (state->temperature != 0.0f || state->top_p != 1.0f)) ||
        (state->sampler_profile == 1 &&
         (state->temperature != 1.0f || state->top_p != 0.95f))) return 0;
    uint64_t total = 0;
    if (!dsv4_session_add_bytes(&total, state->history_n, sizeof(int32_t)) ||
        !dsv4_session_add_bytes(&total, state->mhc_n, sizeof(uint16_t)) ||
        !dsv4_session_add_bytes(&total, state->window_kv_n, sizeof(uint16_t)) ||
        !dsv4_session_add_bytes(&total, state->compressed_kv_n, sizeof(uint16_t)) ||
        !dsv4_session_add_bytes(&total, state->compressor_n, sizeof(float)))
        return 0;
    if ((state->compressed_kv_n && !state->compressed_kv) ||
        (state->compressor_n && !state->compressor)) return 0;
    if (payload_bytes) *payload_bytes = total;
    return 1;
}

static void dsv4_session_fill_header(dsv4_session_header *header,
                                      const dsv4_session_identity *identity,
                                      const dsv4_session_state *state) {
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, "COLIDSV4", 8);
    header->version = DSV4_SESSION_VERSION;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->endian = DSV4_SESSION_ENDIAN;
    header->identity = *identity;
    header->position = state->position; header->context = state->context;
    header->layers = state->layers; header->hidden = state->hidden;
    header->sampler_profile = state->sampler_profile;
    header->temperature = state->temperature; header->top_p = state->top_p;
    memcpy(header->sampler_rng, state->sampler_rng, sizeof(header->sampler_rng));
    header->history_n = state->history_n; header->mhc_n = state->mhc_n;
    header->window_kv_n = state->window_kv_n;
    header->compressed_kv_n = state->compressed_kv_n;
    header->compressor_n = state->compressor_n;
}

static uint64_t dsv4_session_checksum(dsv4_session_header header,
                                      const dsv4_session_state *state) {
    header.checksum = 0;
    uint64_t hash = dsv4_session_hash(1469598103934665603ull,
                                      &header, sizeof(header));
    hash = dsv4_session_hash(hash, state->history,
                             (size_t)state->history_n * sizeof(int32_t));
    hash = dsv4_session_hash(hash, state->mhc,
                             (size_t)state->mhc_n * sizeof(uint16_t));
    hash = dsv4_session_hash(hash, state->window_kv,
                             (size_t)state->window_kv_n * sizeof(uint16_t));
    hash = dsv4_session_hash(hash, state->compressed_kv,
                             (size_t)state->compressed_kv_n * sizeof(uint16_t));
    return dsv4_session_hash(hash, state->compressor,
                             (size_t)state->compressor_n * sizeof(float));
}

static int dsv4_session_write_array(FILE *file, const void *data,
                                     size_t size, uint64_t count) {
    return !count || fwrite(data, size, (size_t)count, file) == count;
}

static int dsv4_session_write(const char *path,
                              const dsv4_session_identity *identity,
                              const dsv4_session_state *state) {
    uint64_t payload = 0;
    if (!path || !dsv4_session_identity_valid(identity) ||
        !dsv4_session_layout_valid(state, &payload)) return 0;
    dsv4_session_header header;
    dsv4_session_fill_header(&header, identity, state);
    header.checksum = dsv4_session_checksum(header, state);
    char temporary[2304];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp.%ld", path,
                 (long)getpid()) >= (int)sizeof(temporary)) return 0;
    FILE *file = fopen(temporary, "wb");
    if (!file) return 0;
    int ok = fwrite(&header, 1, sizeof(header), file) == sizeof(header) &&
        dsv4_session_write_array(file, state->history, sizeof(int32_t), state->history_n) &&
        dsv4_session_write_array(file, state->mhc, sizeof(uint16_t), state->mhc_n) &&
        dsv4_session_write_array(file, state->window_kv, sizeof(uint16_t), state->window_kv_n) &&
        dsv4_session_write_array(file, state->compressed_kv, sizeof(uint16_t), state->compressed_kv_n) &&
        dsv4_session_write_array(file, state->compressor, sizeof(float), state->compressor_n) &&
        fflush(file) == 0 && fsync(fileno(file)) == 0;
    if (fclose(file)) ok = 0;
    if (ok && rename(temporary, path) == 0) {
        char directory[2304];
        snprintf(directory, sizeof(directory), "%s", path);
        char *slash = strrchr(directory, '/');
        if (slash) { if (slash == directory) slash[1] = 0; else *slash = 0; }
        else snprintf(directory, sizeof(directory), ".");
        int descriptor = open(directory, O_RDONLY | O_DIRECTORY);
        if (descriptor < 0) return 0;
        int synced = fsync(descriptor) == 0;
        close(descriptor);
        return synced;
    }
    unlink(temporary);
    return 0;
}

static void dsv4_session_free(dsv4_session_state *state) {
    if (!state) return;
    free(state->history); free(state->mhc); free(state->window_kv);
    free(state->compressed_kv); free(state->compressor);
    memset(state, 0, sizeof(*state));
}

static int dsv4_session_read_array(FILE *file, void **output,
                                    size_t size, uint64_t count) {
    *output = NULL;
    if (!count) return 1;
    if (count > SIZE_MAX / size) return 0;
    void *data = calloc((size_t)count, size);
    if (!data) return 0;
    if (fread(data, size, (size_t)count, file) != count) {
        free(data); return 0;
    }
    *output = data;
    return 1;
}

static int dsv4_session_read(const char *path,
                             const dsv4_session_identity *expected,
                             uint32_t expected_context,
                             dsv4_session_state *state) {
    if (!path || !state || !dsv4_session_identity_valid(expected) ||
        !expected_context || expected_context > DSV4_SESSION_MAX_CONTEXT)
        return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    dsv4_session_header header;
    int ok = fread(&header, 1, sizeof(header), file) == sizeof(header) &&
        !memcmp(header.magic, "COLIDSV4", 8) &&
        header.version == DSV4_SESSION_VERSION &&
        header.header_bytes == sizeof(header) && header.endian == DSV4_SESSION_ENDIAN &&
        !memcmp(&header.identity, expected, sizeof(*expected)) &&
        header.context == expected_context && header.position <= header.context &&
        header.position > 0 && header.layers == 43 && header.hidden == 4096 &&
        header.history_n == header.position;
    uint64_t payload = 0;
    if (ok)
        ok = dsv4_session_add_bytes(&payload, header.history_n, sizeof(int32_t)) &&
             dsv4_session_add_bytes(&payload, header.mhc_n, sizeof(uint16_t)) &&
             dsv4_session_add_bytes(&payload, header.window_kv_n, sizeof(uint16_t)) &&
             dsv4_session_add_bytes(&payload, header.compressed_kv_n, sizeof(uint16_t)) &&
             dsv4_session_add_bytes(&payload, header.compressor_n, sizeof(float));
    if (ok) {
        struct stat status;
        ok = !fstat(fileno(file), &status) && status.st_size >= 0 &&
             (uint64_t)status.st_size == sizeof(header) + payload;
    }
    if (!ok) { fclose(file); return 0; }
    dsv4_session_state loaded = {0};
    loaded.position = header.position; loaded.context = header.context;
    loaded.layers = header.layers; loaded.hidden = header.hidden;
    loaded.sampler_profile = header.sampler_profile;
    loaded.temperature = header.temperature; loaded.top_p = header.top_p;
    memcpy(loaded.sampler_rng, header.sampler_rng, sizeof(loaded.sampler_rng));
    loaded.history_n = header.history_n; loaded.mhc_n = header.mhc_n;
    loaded.window_kv_n = header.window_kv_n;
    loaded.compressed_kv_n = header.compressed_kv_n;
    loaded.compressor_n = header.compressor_n;
    ok = dsv4_session_read_array(file, (void **)&loaded.history, sizeof(int32_t), loaded.history_n) &&
         dsv4_session_read_array(file, (void **)&loaded.mhc, sizeof(uint16_t), loaded.mhc_n) &&
         dsv4_session_read_array(file, (void **)&loaded.window_kv, sizeof(uint16_t), loaded.window_kv_n) &&
         dsv4_session_read_array(file, (void **)&loaded.compressed_kv, sizeof(uint16_t), loaded.compressed_kv_n) &&
         dsv4_session_read_array(file, (void **)&loaded.compressor, sizeof(float), loaded.compressor_n) &&
         fgetc(file) == EOF;
    fclose(file);
    if (!ok || !dsv4_session_layout_valid(&loaded, NULL) ||
        dsv4_session_checksum(header, &loaded) != header.checksum) {
        dsv4_session_free(&loaded); return 0;
    }
    dsv4_session_free(state);
    *state = loaded;
    return 1;
}

#endif
