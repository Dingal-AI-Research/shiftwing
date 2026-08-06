#ifndef COLIB_SERVE_SCHEDULER_H
#define COLIB_SERVE_SCHEDULER_H

#include <stdint.h>
#include <string.h>

#define MUX_MAX_SLOTS 16

typedef enum {
    MUX_SLOT_FREE = 0,
    MUX_SLOT_PREFILL = 1,
    MUX_SLOT_DECODE = 2,
    MUX_SLOT_CANCEL_PENDING = 3,
    MUX_SLOT_DONE_PENDING = 4
} mux_slot_state;

typedef struct {
    mux_slot_state state;
    uint64_t id;
    int max_tokens;
    int emitted;
    int prompt_tokens;
    float temperature;
    float top_p;
} mux_slot;

typedef struct {
    int nslots;
    mux_slot slot[MUX_MAX_SLOTS];
} mux_scheduler;

static int mux_scheduler_init(mux_scheduler *scheduler, int nslots) {
    if (!scheduler || nslots < 1 || nslots > MUX_MAX_SLOTS) return 0;
    memset(scheduler, 0, sizeof(*scheduler));
    scheduler->nslots = nslots;
    return 1;
}

static int mux_scheduler_find(const mux_scheduler *scheduler, uint64_t id) {
    if (!scheduler || !id) return -1;
    for (int slot = 0; slot < scheduler->nslots; slot++)
        if (scheduler->slot[slot].state != MUX_SLOT_FREE &&
            scheduler->slot[slot].id == id)
            return slot;
    return -1;
}

/* Return NULL on acceptance or the mux ERROR code on rejection. */
static const char *mux_scheduler_submit(mux_scheduler *scheduler,
                                        uint64_t id, int slot,
                                        int max_tokens, float temperature,
                                        float top_p) {
    if (!scheduler || !id || slot < 0 || slot >= scheduler->nslots ||
        max_tokens <= 0 || temperature < 0.0f || top_p <= 0.0f ||
        top_p > 1.0f)
        return "BAD_REQUEST";
    if (mux_scheduler_find(scheduler, id) >= 0) return "DUPLICATE_ID";
    if (scheduler->slot[slot].state != MUX_SLOT_FREE) return "SLOT_BUSY";
    mux_slot *request = &scheduler->slot[slot];
    memset(request, 0, sizeof(*request));
    request->state = MUX_SLOT_PREFILL;
    request->id = id;
    request->max_tokens = max_tokens;
    request->temperature = temperature;
    request->top_p = top_p;
    return NULL;
}

static const char *mux_scheduler_cancel(mux_scheduler *scheduler,
                                        uint64_t id) {
    int slot = mux_scheduler_find(scheduler, id);
    if (slot < 0) return "NOT_FOUND";
    scheduler->slot[slot].state = MUX_SLOT_CANCEL_PENDING;
    return NULL;
}

static int mux_scheduler_prefill_done(mux_scheduler *scheduler, int slot,
                                      int prompt_tokens) {
    if (!scheduler || slot < 0 || slot >= scheduler->nslots ||
        scheduler->slot[slot].state != MUX_SLOT_PREFILL ||
        prompt_tokens <= 0)
        return 0;
    scheduler->slot[slot].prompt_tokens = prompt_tokens;
    scheduler->slot[slot].state = MUX_SLOT_DECODE;
    return 1;
}

/* Return active decode slot indices in stable slot order. */
static int mux_scheduler_decode_rows(const mux_scheduler *scheduler,
                                     int *rows, int capacity) {
    if (!scheduler || !rows || capacity < 0) return 0;
    int count = 0;
    for (int slot = 0; slot < scheduler->nslots; slot++)
        if (scheduler->slot[slot].state == MUX_SLOT_DECODE) {
            if (count < capacity) rows[count] = slot;
            count++;
        }
    return count;
}

/* Account one emitted token. Return 1 when max_tokens closes the request. */
static int mux_scheduler_emitted(mux_scheduler *scheduler, int slot) {
    if (!scheduler || slot < 0 || slot >= scheduler->nslots ||
        scheduler->slot[slot].state != MUX_SLOT_DECODE)
        return 0;
    mux_slot *request = &scheduler->slot[slot];
    request->emitted++;
    if (request->emitted >= request->max_tokens) {
        request->state = MUX_SLOT_DONE_PENDING;
        return 1;
    }
    return 0;
}

static int mux_scheduler_done(mux_scheduler *scheduler, int slot) {
    if (!scheduler || slot < 0 || slot >= scheduler->nslots ||
        scheduler->slot[slot].state != MUX_SLOT_DECODE)
        return 0;
    scheduler->slot[slot].state = MUX_SLOT_DONE_PENDING;
    return 1;
}

/* The engine calls release only after DONE, or after recurrent/KV state has
 * been persisted for CANCELLED. */
static int mux_scheduler_release(mux_scheduler *scheduler, int slot,
                                 mux_slot *finished) {
    if (!scheduler || slot < 0 || slot >= scheduler->nslots ||
        scheduler->slot[slot].state == MUX_SLOT_FREE)
        return 0;
    if (finished) *finished = scheduler->slot[slot];
    memset(&scheduler->slot[slot], 0, sizeof(scheduler->slot[slot]));
    return 1;
}

#endif
