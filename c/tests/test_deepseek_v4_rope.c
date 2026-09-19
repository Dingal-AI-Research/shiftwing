#include "../deepseek_v4.h"
#include "../json.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "rope:%d: %s\n", __LINE__, #c); return 1; \
} } while (0)

int main(void) {
    FILE *file = fopen("tests/fixtures/deepseek_v4_rope.json", "rb");
    if (!file) file = fopen("c/tests/fixtures/deepseek_v4_rope.json", "rb");
    CHECK(file);
    CHECK(!fseek(file, 0, SEEK_END));
    long size = ftell(file);
    CHECK(size > 0 && size < 1024 * 1024 && !fseek(file, 0, SEEK_SET));
    char *text = malloc((size_t)size + 1);
    CHECK(text && fread(text, 1, (size_t)size, file) == (size_t)size);
    CHECK(!fclose(file)); text[size] = 0;
    jval *fixture = json_parse(text, NULL);
    CHECK(fixture);
    CHECK(!strcmp(json_get(fixture, "reference_model_sha256")->str,
        "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"));
    jval *positions = json_get(fixture, "positions");
    jval *cases = json_get(fixture, "cases");
    CHECK(positions && positions->len == 16 && cases && cases->len == 4);
    for (int c = 0; c < cases->len; c++) {
        jval *item = cases->kids[c], *expected = json_get(item, "output");
        CHECK(expected && expected->len == positions->len * 64);
        double error = 0, norm = 0;
        for (int t = 0; t < positions->len; t++) {
            float x[64];
            for (int i = 0; i < 64; i++)
                x[i] = (((t * 64 + i) * 37) % 97 - 48) / 16.0f;
            dsv4_rope(x, 64, (int)positions->kids[t]->num,
                (int)json_get(item, "original")->num,
                (float)json_get(item, "base")->num,
                (float)json_get(item, "factor")->num, 32, 1,
                (int)json_get(item, "inverse")->num);
            dsv4_round_bf16_array(x, 64);
            for (int i = 0; i < 64; i++) {
                float reference = (float)expected->kids[t * 64 + i]->num;
                double delta = (double)x[i] - reference;
                error += delta * delta; norm += (double)reference * reference;
            }
        }
        double relative = sqrt(error / norm);
        printf("DeepSeek RoPE case=%d relative=%.9g\n", c, relative);
        CHECK(relative < 1e-6);
    }
    puts("DeepSeek RoPE matches pinned reference through position 65535");
    return 0;
}
