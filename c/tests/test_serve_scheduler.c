#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../serve_scheduler.h"

static void fail(const char *message) {
    fprintf(stderr, "test_serve_scheduler: %s\n", message);
    exit(1);
}

static void expect_code(const char *got, const char *want) {
    if ((!got) != (!want) || (got && strcmp(got, want))) fail("error code");
}

int main(void) {
    mux_scheduler scheduler;
    if (!mux_scheduler_init(&scheduler, 3)) fail("init");
    if (mux_scheduler_init(&scheduler, 0) ||
        mux_scheduler_init(&scheduler, MUX_MAX_SLOTS + 1))
        fail("invalid slot count");
    if (!mux_scheduler_init(&scheduler, 3)) fail("re-init");

    expect_code(mux_scheduler_submit(&scheduler, 10, 0, 2, 0.0f, 1.0f),
                NULL);
    expect_code(mux_scheduler_submit(&scheduler, 10, 1, 2, 0.0f, 1.0f),
                "DUPLICATE_ID");
    expect_code(mux_scheduler_submit(&scheduler, 11, 0, 2, 0.0f, 1.0f),
                "SLOT_BUSY");
    expect_code(mux_scheduler_submit(&scheduler, 11, 1, 3, 0.5f, 0.9f),
                NULL);
    expect_code(mux_scheduler_submit(&scheduler, 12, 3, 1, 0.0f, 1.0f),
                "BAD_REQUEST");

    if (!mux_scheduler_prefill_done(&scheduler, 1, 7) ||
        !mux_scheduler_prefill_done(&scheduler, 0, 5))
        fail("prefill transition");
    int rows[MUX_MAX_SLOTS] = {0};
    if (mux_scheduler_decode_rows(&scheduler, rows, MUX_MAX_SLOTS) != 2 ||
        rows[0] != 0 || rows[1] != 1)
        fail("stable decode rows");

    if (mux_scheduler_emitted(&scheduler, 0) ||
        !mux_scheduler_emitted(&scheduler, 0))
        fail("length limit");
    expect_code(mux_scheduler_cancel(&scheduler, 11), NULL);
    if (mux_scheduler_decode_rows(&scheduler, rows, MUX_MAX_SLOTS) != 0)
        fail("finished/cancelled rows excluded");
    expect_code(mux_scheduler_cancel(&scheduler, 999), "NOT_FOUND");

    mux_slot finished;
    if (!mux_scheduler_release(&scheduler, 0, &finished) ||
        finished.id != 10 || finished.emitted != 2 ||
        finished.prompt_tokens != 5)
        fail("finished snapshot");
    if (!mux_scheduler_release(&scheduler, 1, &finished) ||
        finished.state != MUX_SLOT_CANCEL_PENDING || finished.id != 11)
        fail("cancel snapshot");
    if (mux_scheduler_release(&scheduler, 1, NULL)) fail("double release");

    expect_code(mux_scheduler_submit(&scheduler, 12, 0, 1, 0.0f, 1.0f),
                NULL);
    if (mux_scheduler_find(&scheduler, 12) != 0) fail("slot reuse");

    puts("serve scheduler state tests: ok");
    return 0;
}
