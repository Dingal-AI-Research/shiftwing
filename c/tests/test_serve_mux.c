#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../serve_mux.h"

static void fail(const char *message) {
    fprintf(stderr, "test_serve_mux: %s\n", message);
    exit(1);
}

static FILE *input_from(const void *data, size_t size) {
    FILE *file = tmpfile();
    if (!file) fail("tmpfile input");
    if (fwrite(data, 1, size, file) != size) fail("write input");
    rewind(file);
    return file;
}

static void test_submit_with_newlines(void) {
    const char wire[] =
        "SUBMIT 42 3 12 64 0.25 0.95\n"
        "hello\nworld!\n";
    FILE *input = input_from(wire, sizeof(wire) - 1);
    mux_frame frame = {0};
    mux_parse_error error = {0};
    if (mux_read_frame(input, &frame, 0, &error) != MUX_FRAME_SUBMIT)
        fail("valid SUBMIT rejected");
    if (frame.id != 42 || frame.slot != 3 || frame.nbytes != 12 ||
        frame.max_tokens != 64 || memcmp(frame.payload, "hello\nworld!", 12))
        fail("SUBMIT fields");
    mux_frame_clear(&frame);
    if (mux_read_frame(input, &frame, 0, &error) != MUX_FRAME_EOF)
        fail("EOF after SUBMIT");
    fclose(input);
}

static void test_cancel_and_bad_header_recovery(void) {
    const char wire[] = "WHAT 1\nCANCEL 91\n";
    FILE *input = input_from(wire, sizeof(wire) - 1);
    mux_frame frame = {0};
    mux_parse_error error = {0};
    if (mux_read_frame(input, &frame, 0, &error) != MUX_FRAME_ERROR ||
        strcmp(error.code, "BAD_FRAME") || error.fatal)
        fail("bad header classification");
    if (mux_read_frame(input, &frame, 0, &error) != MUX_FRAME_CANCEL ||
        frame.id != 91)
        fail("CANCEL recovery");
    fclose(input);
}

static void test_payload_boundary_failures_are_fatal(void) {
    const char truncated[] = "SUBMIT 7 0 5 4 0 1\nabc";
    FILE *input = input_from(truncated, sizeof(truncated) - 1);
    mux_frame frame = {0};
    mux_parse_error error = {0};
    if (mux_read_frame(input, &frame, 0, &error) != MUX_FRAME_ERROR ||
        strcmp(error.code, "BAD_FRAME") || !error.fatal)
        fail("truncated payload classification");
    fclose(input);

    const char oversized[] = "SUBMIT 8 0 99 4 0 1\n";
    input = input_from(oversized, sizeof(oversized) - 1);
    if (mux_read_frame(input, &frame, 16, &error) != MUX_FRAME_ERROR ||
        strcmp(error.code, "BAD_REQUEST") || !error.fatal)
        fail("oversized payload classification");
    fclose(input);
}

static void test_response_framing(void) {
    FILE *output = tmpfile();
    if (!output) fail("tmpfile output");
    if (!mux_write_ready(output) ||
        !mux_write_data(output, 5, "a\nb", 3) ||
        !mux_write_error(output, 6, "CANCELLED") ||
        !mux_write_done(output, 5, 2, 3.5, 91.25, 4.0, 7, 1))
        fail("write response");
    rewind(output);
    char got[512] = {0};
    size_t n = fread(got, 1, sizeof(got) - 1, output);
    const char expected[] =
        "\x01\x01READY\x01\x01\n"
        "DATA 5 3\n"
        "a\nb\n"
        "ERROR 6 CANCELLED\n"
        "DONE 5 STAT 2 3.500000 91.2500 4.0000 7 1\n";
    if (n != sizeof(expected) - 1 || memcmp(got, expected, n))
        fail("response bytes");
    fclose(output);
}

int main(void) {
    test_submit_with_newlines();
    test_cancel_and_bad_header_recovery();
    test_payload_boundary_failures_are_fatal();
    test_response_framing();
    puts("serve mux framing tests: ok");
    return 0;
}
