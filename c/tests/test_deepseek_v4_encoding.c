#include "../deepseek_v4.h"
#include <stdio.h>
#include <stdlib.h>

static inline float reference_fp8_decode(uint8_t code) {
    int negative = (code & 0x80u) != 0;
    int exponent = (code >> 3) & 0x0f;
    int mantissa = code & 0x07;
    if (exponent == 15 && mantissa == 7) return NAN;
    float value = exponent
        ? ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7)
        : ldexpf((float)mantissa, -9);
    return negative ? -value : value;
}

static inline float reference_ue8m0_decode(uint8_t code) {
    return code == 255u ? NAN : ldexpf(1.0f, (int)code - 127);
}

static inline int equal_float_bits(float actual, float expected) {
    if (isnan(expected)) return isnan(actual);
    uint32_t actual_bits, expected_bits;
    memcpy(&actual_bits, &actual, sizeof(actual_bits));
    memcpy(&expected_bits, &expected, sizeof(expected_bits));
    return actual_bits == expected_bits;
}

static inline uint8_t reference_fp8_encode(float x) {
    if (isnan(x)) return 0x7f;
    int negative = signbit(x) != 0;
    float target = fminf(fabsf(x), 448.0f);
    uint8_t best = 0;
    float best_error = FLT_MAX;
    for (uint8_t code = 0; code <= 0x7e; code++) {
        float error = fabsf(reference_fp8_decode(code) - target);
        if (error < best_error ||
            (error == best_error && !(code & 1u) && (best & 1u))) {
            best = code;
            best_error = error;
        }
    }
    return (uint8_t)(best | (negative ? 0x80u : 0u));
}

int main(void) {
    for (int code = 0; code < 256; code++) {
        float actual_fp8 = dsv4_fp8_e4m3fn((uint8_t)code);
        float expected_fp8 = reference_fp8_decode((uint8_t)code);
        if (!equal_float_bits(actual_fp8, expected_fp8)) {
            fprintf(stderr, "FP8 decode mismatch at code %d\n", code);
            return 1;
        }
        float actual_scale = dsv4_ue8m0((uint8_t)code);
        float expected_scale = reference_ue8m0_decode((uint8_t)code);
        if (!equal_float_bits(actual_scale, expected_scale)) {
            fprintf(stderr, "UE8M0 decode mismatch at code %d\n", code);
            return 1;
        }
    }
    uint32_t state=173;
    for (int i=0;i<1000000;i++) {
        state=state*1664525u+1013904223u;float value;memcpy(&value,&state,4);
        if (dsv4_fp8_encode(value)!=reference_fp8_encode(value)) {
            fprintf(stderr,"FP8 mismatch %.9g\n",value);return 1;
        }
    }
    for (int i=0;i<126;i++) {
        float a=reference_fp8_decode((uint8_t)i);
        float b=reference_fp8_decode((uint8_t)(i+1)),middle=(a+b)/2;
        float values[]={a,b,middle,nextafterf(middle,a),nextafterf(middle,b)};
        for (int k=0;k<5;k++) for (int sign=-1;sign<=1;sign+=2)
            if (dsv4_fp8_encode(sign*values[k])!=reference_fp8_encode(sign*values[k])) return 1;
    }
    const float fp4[]={0,.5f,1,1.5f,2,3,4,6};
    for (int i=0;i<7;i++) {
        float midpoint=(fp4[i]+fp4[i+1])/2;
        for (int negative=0;negative<2;negative++) {
            unsigned sign=negative?8:0;float factor=negative?-1:1;
            if (dsv4_fp4_encode(factor*midpoint)!=((i%2?i+1:i)|sign) ||
                dsv4_fp4_encode(factor*nextafterf(midpoint,fp4[i]))!=(i|sign) ||
                dsv4_fp4_encode(factor*nextafterf(midpoint,fp4[i+1]))!=((i+1)|sign)) return 1;
        }
    }
    if (dsv4_fp4_encode(-0.0f)!=8) return 1;
    puts("FP8/UE8M0 decoders and FP8 encoder match independent exhaustive oracles");return 0;
}
