#ifndef COLIB_DEEPSEEK_V4_LIMITS_H
#define COLIB_DEEPSEEK_V4_LIMITS_H

/* Serving policy: 90 Ki input tokens plus a separate 8 Ki response reserve.
 * Keep in sync with tools/deepseek_v4_spec.py and LocalForge's review profile.
 * These limits do not change the checkpoint's original RoPE context. */
#define DSV4_REVIEW_INPUT_TOKENS 92160
#define DSV4_REVIEW_OUTPUT_TOKENS 8192
#define DSV4_MAX_CONTEXT (DSV4_REVIEW_INPUT_TOKENS + DSV4_REVIEW_OUTPUT_TOKENS)

#endif
