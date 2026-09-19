/* Pinned DeepSeek split sequence, with shared byte BPE and added-token handling.
 * Keep this separate from tok_encode: Qwen/GLM keep their own splitting rules. */
#ifndef DEEPSEEK_V4_TOKENIZER_H
#define DEEPSEEK_V4_TOKENIZER_H
#include "tok.h"
#include "deepseek_v4_unicode.h"

static int dsv4_tok_string(jval *obj, const char *key, const char *want) {
    jval *v = json_get(obj, key);
    return v && v->t == J_STR && !strcmp(v->str, want);
}

static int dsv4_tok_bool(jval *obj, const char *key, int want) {
    jval *v = json_get(obj, key);
    return v && v->t == J_BOOL && v->boolean == want;
}

static void dsv4_tok_free_json(jval *node) {
    if (!node) return;
    for (int i = 0; i < node->len; i++) {
        dsv4_tok_free_json(node->kids[i]);
        if (node->keys) free(node->keys[i]);
    }
    free(node->str); free(node->kids); free(node->keys); free(node);
}

/* Unsupported recipe drift fails visibly instead of falling back to Qwen. */
static int dsv4_tok_load(Tok *tok, const char *path) {
    long size;
    char *text = tk_read_file(path, &size);
    jval *root = json_parse(text, NULL);
    jval *pre = json_get(root, "pre_tokenizer");
    jval *parts = json_get(pre, "pretokenizers");
    jval *norm = json_get(root, "normalizer");
    jval *norms = json_get(norm, "normalizers");
    int ok = dsv4_tok_string(pre, "type", "Sequence") && parts &&
        parts->t == J_ARR && parts->len == 4 &&
        dsv4_tok_string(norm, "type", "Sequence") && norms &&
        norms->t == J_ARR && norms->len == 0;
    const char *patterns[] = {
        "\\p{N}{1,3}",
        "[一-龥぀-ゟ゠-ヿ]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
    };
    for (int i = 0; ok && i < 3; i++) {
        jval *part = parts->kids[i];
        ok = dsv4_tok_string(part, "type", "Split") &&
            dsv4_tok_string(part, "behavior", "Isolated") &&
            dsv4_tok_bool(part, "invert", 0) &&
            dsv4_tok_string(json_get(part, "pattern"), "Regex", patterns[i]);
    }
    if (ok) {
        jval *bytes = parts->kids[3];
        ok = dsv4_tok_string(bytes, "type", "ByteLevel") &&
            dsv4_tok_bool(bytes, "add_prefix_space", 0) &&
            dsv4_tok_bool(bytes, "use_regex", 0);
    }
    dsv4_tok_free_json(root); free(text);
    if (!ok) return 0;
    tok_load(tok, path);
    return 1;
}

static int dsv4_is_cjk(uint32_t c) {
    return (c >= 0x4e00 && c <= 0x9fa5) || (c >= 0x3040 && c <= 0x30ff);
}
static int dsv4_is_ascii_letter(uint32_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static int dsv4_is_mark(uint32_t c) {
    return is_M(c) || (c >= 0xfe00 && c <= 0xfe0f) ||
        (c >= 0xe0100 && c <= 0xe01ef);
}

/* Return the end of the first matching alternative in the third Split. */
static int dsv4_word_match(const uint32_t *cp, int i, int end) {
    uint32_t c = cp[i];
    int j = i;
    if (c >= '!' && c <= '~' && !dsv4_is_ascii_letter(c) &&
            !(c >= '0' && c <= '9') && i + 1 < end &&
            dsv4_is_ascii_letter(cp[i+1])) {
        j = i + 2;
        while (j < end && dsv4_is_ascii_letter(cp[j])) j++;
        return j;
    }
    if (c != '\r' && c != '\n' && !is_L(c) &&
            !dsv4_is_punctuation_symbol(c) && i + 1 < end &&
            (is_L(cp[i+1]) || dsv4_is_mark(cp[i+1]))) j++;
    if (is_L(cp[j]) || dsv4_is_mark(cp[j])) {
        do { j++; } while (j < end && (is_L(cp[j]) || dsv4_is_mark(cp[j])));
        return j;
    }
    j = i;
    if (c == ' ' && j + 1 < end && dsv4_is_punctuation_symbol(cp[j+1])) j++;
    if (dsv4_is_punctuation_symbol(cp[j])) {
        do { j++; } while (j < end && dsv4_is_punctuation_symbol(cp[j]));
        while (j < end && (cp[j] == '\r' || cp[j] == '\n')) j++;
        return j;
    }
    j = i;
    int last_newline = -1;
    while (j < end && is_S(cp[j])) {
        if (cp[j] == '\r' || cp[j] == '\n') last_newline = j;
        j++;
    }
    if (last_newline >= 0) return last_newline + 1;
    if (j > i) return j == end || j == i + 1 ? j : j - 1;
    return i;
}

static void dsv4_split_words(Tok *tok, const unsigned char *text,
        const uint32_t *cp, const int *offsets, int start, int end,
        int *ids, int *count, int maximum) {
    for (int i = start; i < end;) {
        int next = dsv4_word_match(cp, i, end);
        if (next == i) {
            next++;
            /* Split(Isolated) preserves a whole unmatched gap as one piece. */
            while (next < end && dsv4_word_match(cp, next, end) == next) next++;
        }
        bpe_piece(tok, text, offsets[i], offsets[next], ids, count, maximum);
        i = next;
    }
}

static int dsv4_pretok(Tok *tok, const unsigned char *text, int start, int end,
        int *ids, int *count, int maximum) {
    int bytes = end - start;
    if (!bytes) return 1;
    uint32_t *cp = malloc(((size_t)bytes + 1) * sizeof(*cp));
    int *offsets = malloc(((size_t)bytes + 1) * sizeof(*offsets));
    if (!cp || !offsets) { free(cp); free(offsets); return 0; }
    int n = 0;
    for (int i = start; i < end;) {
        offsets[n] = i;
        i += u8_next(text, end, i, &cp[n]);
        n++;
    }
    offsets[n] = end;
    for (int i = 0; i < n;) {
        int next = i + 1;
        if (is_N(cp[i])) {
            while (next < n && next - i < 3 && is_N(cp[next])) next++;
            bpe_piece(tok, text, offsets[i], offsets[next], ids, count, maximum);
        } else {
            while (next < n && !is_N(cp[next])) next++;
            /* Digits are isolated first, then CJK runs. Regex lookahead must
             * observe those boundaries, including whitespace before numbers. */
            for (int j = i; j < next;) {
                int last = j + 1, cjk = dsv4_is_cjk(cp[j]);
                while (last < next && dsv4_is_cjk(cp[last]) == cjk) last++;
                dsv4_split_words(tok, text, cp, offsets, j, last, ids, count, maximum);
                j = last;
            }
        }
        i = next;
    }
    free(cp); free(offsets);
    return 1;
}

/* One output id per input byte is a safe bound for this non-normalizing BPE.
 * Reject a smaller output buffer; the generic tokenizer silently truncates. */
static int dsv4_tok_encode(Tok *tok, const char *text, int bytes, int *ids, int maximum) {
    if (bytes < 0 || maximum < bytes || tok->use_nfc) return -1;
    int count = 0;
    for (int i = 0; i < bytes;) {
        int hit = -1, length = 0, id = -1;
        for (int j = i; j < bytes && hit < 0; j++) {
            for (int k = 0; k < tok->nsp; k++) {
                Special *s = &tok->sp[k];
                if (s->len > 0 && s->len <= bytes - j && text[j] == s->str[0] &&
                        !memcmp(text + j, s->str, s->len)) {
                    hit = j; length = s->len; id = s->id; break;
                }
            }
        }
        int end = hit < 0 ? bytes : hit;
        if (!dsv4_pretok(tok, (const unsigned char *)text, i, end, ids, &count, maximum)) return -1;
        if (hit < 0) break;
        ids[count++] = id;
        i = hit + length;
    }
    return count;
}
#endif
