/* CPU-only, length-framed tokenizer helper. No weights, CUDA, or engine startup.
 * stdin: COUNT byte_length\n<exact UTF-8 bytes>\n (ENCODE also returns ids/roundtrip).
 * stdout: one JSON object per request. Invalid frames fail closed. */
#include "deepseek_v4_tokenizer.h"
#define DSV4_TOKENIZER_MAX_BYTES (16 * 1024 * 1024)

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: deepseek_v4_tokenize tokenizer.json\n"); return 2; }
    Tok tok;
    if (!dsv4_tok_load(&tok, argv[1])) {
        fprintf(stderr, "unsupported DeepSeek tokenizer recipe\n"); return 2;
    }
    char frame[128];
    while (fgets(frame, sizeof(frame), stdin)) {
        char command[16], extra;
        long bytes;
        if (!strchr(frame, '\n') || sscanf(frame, "%15s %ld %c", command, &bytes, &extra) != 2 ||
                (strcmp(command, "COUNT") && strcmp(command, "ENCODE")) ||
                bytes < 0 || bytes > DSV4_TOKENIZER_MAX_BYTES) {
            puts("{\"error\":\"invalid tokenizer frame\"}"); return 2;
        }
        char *text = malloc((size_t)bytes + 1);
        int *ids = malloc(((size_t)bytes + 1) * sizeof(*ids));
        if (!text || !ids) { free(text); free(ids); return 2; }
        if (fread(text, 1, bytes, stdin) != (size_t)bytes || fgetc(stdin) != '\n') {
            free(text); free(ids); puts("{\"error\":\"truncated tokenizer frame\"}"); return 2;
        }
        int count = dsv4_tok_encode(&tok, text, (int)bytes, ids, (int)bytes);
        free(text);
        if (count < 0) { free(ids); puts("{\"error\":\"tokenization failed\"}"); return 2; }
        printf("{\"tokens\":%d", count);
        if (!strcmp(command, "ENCODE")) {
            printf(",\"ids\":[");
            for (int i = 0; i < count; i++) printf("%s%d", i ? "," : "", ids[i]);
            char *decoded = malloc((size_t)bytes + 1);
            if (!decoded) { free(ids); return 2; }
            int n = tok_decode(&tok, ids, count, decoded, (int)bytes + 1);
            printf("],\"decoded_hex\":\"");
            for (int i = 0; i < n; i++) printf("%02x", (unsigned char)decoded[i]);
            printf("\""); free(decoded);
        }
        puts("}"); fflush(stdout); free(ids);
    }
    return ferror(stdin) ? 2 : 0;
}
