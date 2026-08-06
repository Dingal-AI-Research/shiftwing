#ifndef COLIB_SERVE_MUX_H
#define COLIB_SERVE_MUX_H

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MUX_HEADER_MAX 512
#define MUX_DEFAULT_PAYLOAD_MAX (16u * 1024u * 1024u)

typedef enum {
    MUX_FRAME_EOF = 0,
    MUX_FRAME_SUBMIT = 1,
    MUX_FRAME_CANCEL = 2,
    MUX_FRAME_ERROR = -1
} mux_frame_kind;

typedef struct {
    mux_frame_kind kind;
    uint64_t id;
    int slot;
    size_t nbytes;
    int max_tokens;
    float temperature;
    float top_p;
    unsigned char *payload;
} mux_frame;

typedef struct {
    const char *code;
    int fatal;
} mux_parse_error;

static void mux_frame_clear(mux_frame *frame) {
    if (!frame) return;
    free(frame->payload);
    memset(frame, 0, sizeof(*frame));
}

static int mux_only_space(const char *text, int used) {
    for (const unsigned char *p = (const unsigned char *)text + used; *p; p++)
        if (*p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') return 0;
    return 1;
}

/* Read one byte-counted request. A malformed header is recoverable because
 * fgets has already consumed its line. A malformed/truncated payload is fatal:
 * the next frame boundary can no longer be established safely. */
static mux_frame_kind mux_read_frame(FILE *input, mux_frame *frame,
                                     size_t payload_limit,
                                     mux_parse_error *error) {
    char header[MUX_HEADER_MAX];
    if (!payload_limit) payload_limit = MUX_DEFAULT_PAYLOAD_MAX;
    mux_frame_clear(frame);
    if (error) {
        error->code = NULL;
        error->fatal = 0;
    }
    if (!fgets(header, sizeof(header), input)) {
        if (feof(input)) return MUX_FRAME_EOF;
        if (error) {
            error->code = "BAD_FRAME";
            error->fatal = 1;
        }
        return MUX_FRAME_ERROR;
    }
    if (!strchr(header, '\n')) {
        int ch;
        while ((ch = fgetc(input)) != '\n' && ch != EOF) {}
        if (error) error->code = "BAD_FRAME";
        return MUX_FRAME_ERROR;
    }

    uint64_t id = 0;
    int used = 0;
    if (sscanf(header, "CANCEL %" SCNu64 " %n", &id, &used) == 1 &&
        id != 0 && mux_only_space(header, used)) {
        frame->kind = MUX_FRAME_CANCEL;
        frame->id = id;
        return frame->kind;
    }

    int slot = 0, max_tokens = 0;
    unsigned long long nbytes = 0;
    float temperature = 0.0f, top_p = 0.0f;
    used = 0;
    if (sscanf(header, "SUBMIT %" SCNu64 " %d %llu %d %f %f %n",
               &id, &slot, &nbytes, &max_tokens, &temperature, &top_p,
               &used) != 6 ||
        !mux_only_space(header, used)) {
        if (error) error->code = "BAD_FRAME";
        return MUX_FRAME_ERROR;
    }
    if (id == 0 || slot < 0 || max_tokens <= 0 || temperature < 0.0f ||
        top_p <= 0.0f || top_p > 1.0f ||
        nbytes == 0 || nbytes > payload_limit || nbytes > SIZE_MAX - 1) {
        if (error) {
            error->code = "BAD_REQUEST";
            /* An announced payload is still waiting in the stream. Rejecting
             * without consuming an untrusted length cannot recover framing. */
            error->fatal = 1;
        }
        return MUX_FRAME_ERROR;
    }

    frame->payload = (unsigned char *)calloc((size_t)nbytes + 1, 1);
    if (!frame->payload) {
        if (error) {
            error->code = "INTERNAL";
            error->fatal = 1;
        }
        return MUX_FRAME_ERROR;
    }
    if (fread(frame->payload, 1, (size_t)nbytes, input) != (size_t)nbytes ||
        fgetc(input) != '\n') {
        mux_frame_clear(frame);
        if (error) {
            error->code = "BAD_FRAME";
            error->fatal = 1;
        }
        return MUX_FRAME_ERROR;
    }
    frame->kind = MUX_FRAME_SUBMIT;
    frame->id = id;
    frame->slot = slot;
    frame->nbytes = (size_t)nbytes;
    frame->max_tokens = max_tokens;
    frame->temperature = temperature;
    frame->top_p = top_p;
    return frame->kind;
}

static int mux_write_ready(FILE *output) {
    return fputs("\x01\x01READY\x01\x01\n", output) >= 0 &&
           fflush(output) == 0;
}

static int mux_write_data(FILE *output, uint64_t id,
                          const void *data, size_t nbytes) {
    if (fprintf(output, "DATA %" PRIu64 " %zu\n", id, nbytes) < 0) return 0;
    if (nbytes && fwrite(data, 1, nbytes, output) != nbytes) return 0;
    return fputc('\n', output) != EOF && fflush(output) == 0;
}

static int mux_write_error(FILE *output, uint64_t id, const char *code) {
    return fprintf(output, "ERROR %" PRIu64 " %s\n", id, code) >= 0 &&
           fflush(output) == 0;
}

static int mux_write_done(FILE *output, uint64_t id, int emitted,
                          double tok_s, double hit_pct, double rss_gb,
                          int prompt_tokens, int length_limited) {
    return fprintf(output,
                   "DONE %" PRIu64
                   " STAT %d %.6f %.4f %.4f %d %d\n",
                   id, emitted, tok_s, hit_pct, rss_gb, prompt_tokens,
                   length_limited) >= 0 &&
           fflush(output) == 0;
}

#endif
