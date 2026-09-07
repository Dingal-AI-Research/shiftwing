/* Il pre-tokenizer legge \p{N}{1,K} dal tokenizer, non lo assume.
 *
 * Qwen scrive \p{N} (una cifra per pezzo), GLM-5.3 scrive \p{N}{1,3}. Il codice
 * aveva la regola di Qwen cablata, quindi "19" usciva come "1"+"9" invece del
 * token unico: 6412 casi su 10000 divergevano dall'oracolo HF sul tokenizer di
 * GLM, e nessun tokenizer nel repo esercitava la regola multi-cifra, quindi
 * niente lo segnalava. Questo test tiene entrambe le regole.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../tok.h"

static int failures = 0;

static void check(const char *what, int got, int want) {
    if (got != want) { printf("FAIL %s: got %d want %d\n", what, got, want); failures++; }
}

/* Un tokenizer minimo: le cifre come token singoli, piu' il token "19", e una
 * sola merge che lo produce. Basta a distinguere le due regole. */
static void write_tokenizer(const char *path, const char *digit_rule) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "{\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
               "{\"type\":\"Split\",\"pattern\":{\"Regex\":"
               "\"[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|%s| ?[^\\\\s\\\\p{L}\\\\p{N}]+\"},"
               "\"behavior\":\"Isolated\"},"
               "{\"type\":\"ByteLevel\",\"add_prefix_space\":false}]},"
               "\"model\":{\"type\":\"BPE\",\"vocab\":{", digit_rule);
    for (int d = 0; d <= 9; d++) fprintf(f, "%s\"%d\":%d", d ? "," : "", d, d);
    fprintf(f, ",\"19\":10},\"merges\":[[\"1\",\"9\"]]},\"added_tokens\":[]}\n");
    fclose(f);
}

static int encode_one(const char *rule, const char *text, int *out, int max) {
    char path[] = "/tmp/shiftwing_tok_digit_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); exit(1); }
    close(fd);
    write_tokenizer(path, rule);
    Tok T;
    tok_load(&T, path);
    int n = tok_encode(&T, text, (int)strlen(text), out, max);
    check("max_digit_run", T.max_digit_run, strstr(rule, "{1,3}") ? 3 : 1);
    remove(path);
    return n;
}

int main(void) {
    int ids[16];

    /* GLM: "219" e' un solo pezzo, la merge (1,9) lo riduce a "2" + "19". */
    int n = encode_one("\\\\p{N}{1,3}", "219", ids, 16);
    check("glm count", n, 2);
    if (n == 2) { check("glm[0] '2'", ids[0], 2); check("glm[1] '19'", ids[1], 10); }

    /* Qwen: una cifra per pezzo, quindi nessuna merge puo' attraversarli. */
    n = encode_one("\\\\p{N}", "219", ids, 16);
    check("qwen count", n, 3);
    if (n == 3) {
        check("qwen[0] '2'", ids[0], 2);
        check("qwen[1] '1'", ids[1], 1);
        check("qwen[2] '9'", ids[2], 9);
    }

    /* Il pezzo si ferma a K cifre: "1919" diventa "191" + "9", quindi la merge
     * (1,9) scatta una volta sola. Senza il limite sarebbe un pezzo solo e
     * scatterebbe due volte, dando due token invece di tre. */
    n = encode_one("\\\\p{N}{1,3}", "1919", ids, 16);
    check("glm run stops at 3", n, 3);

    printf("tok digit run: %s (%d failures)\n", failures ? "FAIL" : "ok", failures);
    return failures ? 1 : 0;
}
