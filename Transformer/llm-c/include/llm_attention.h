/* include/llm_attention.h */
#pragma once
#include <stddef.h>
#include "llm_config.h"

typedef struct {
    /* Wq/Wk/Wv: [D_MODEL x D_K], head별로 다름. Wo: [D_MODEL x D_MODEL] */
    float Wq[N_HEAD][D_MODEL][D_K];
    float Wk[N_HEAD][D_MODEL][D_K];
    float Wv[N_HEAD][D_MODEL][D_K];
    float Wo[D_MODEL][D_MODEL];
} LlmMhaWeights;

/* X: [T x D_MODEL], Y: [T x D_MODEL] */
void llm_mha_forward(const float *X, float *Y, size_t T,
                     const LlmMhaWeights *w);
