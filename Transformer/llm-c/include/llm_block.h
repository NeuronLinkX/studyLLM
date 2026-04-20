/* include/llm_block.h */
#pragma once
#include <stddef.h>
#include "llm_attention.h"
#include "llm_ffn.h"
#include "llm_config.h"

/* Pre-LN: y1 = x + MHA(LN(x)), y2 = y1 + FFN(LN(y1)) */
typedef struct {
    LlmMhaWeights mha;
    LlmFfnWeights ffn;
    float ln1_gamma[D_MODEL], ln1_beta[D_MODEL];
    float ln2_gamma[D_MODEL], ln2_beta[D_MODEL];
} LlmDecoderBlock;

void llm_decoder_block_forward(const float *x /*[T*D]*/, float *y /*[T*D]*/, size_t T,
                               const LlmDecoderBlock *blk);
