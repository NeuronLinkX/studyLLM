/* include/llm_ffn.h */
#pragma once
#include <stddef.h>
#include "llm_config.h"

typedef struct {
    /* W1: [D_MODEL x D_FF], b1: [D_FF]
       W2: [D_FF x D_MODEL], b2: [D_MODEL] */
    float W1[D_MODEL][D_FF], b1[D_FF];
    float W2[D_FF][D_MODEL], b2[D_MODEL];
} LlmFfnWeights;

/* X[T x D_MODEL] → Y[T x D_MODEL] */
void llm_ffn_forward(const float *X, float *Y, size_t T,
                     const LlmFfnWeights *w);
