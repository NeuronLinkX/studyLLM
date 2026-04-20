/* src/llm_block.c */
#include "llm_block.h"
#include "llm_math.h"
#include <string.h>

void llm_decoder_block_forward(const float *x, float *y, size_t T,
                               const LlmDecoderBlock *blk)
{
    float x_ln[SEQ_LEN][D_MODEL];
    float mha_out[SEQ_LEN][D_MODEL];
    float y1[SEQ_LEN][D_MODEL];
    float y1_ln[SEQ_LEN][D_MODEL];
    float ffn_out[SEQ_LEN][D_MODEL];

    /* Pre-LN: LN(x) */
    for(size_t t=0;t<T;++t)
        llm_layernorm(&x[t*D_MODEL], x_ln[t], D_MODEL, 1e-5f,
                      blk->ln1_gamma, blk->ln1_beta);

    /* MHA */
    llm_mha_forward(&x_ln[0][0], &mha_out[0][0], T, &blk->mha);

    /* Residual: y1 = x + mha_out */
    for(size_t t=0;t<T;++t)
        for(size_t d=0; d<D_MODEL; ++d)
            y1[t][d] = x[t*D_MODEL + d] + mha_out[t][d];

    /* LN(y1) */
    for(size_t t=0;t<T;++t)
        llm_layernorm(&y1[t][0], y1_ln[t], D_MODEL, 1e-5f,
                      blk->ln2_gamma, blk->ln2_beta);

    /* FFN */
    llm_ffn_forward(&y1_ln[0][0], &ffn_out[0][0], T, &blk->ffn);

    /* Residual: y = y1 + ffn_out */
    for(size_t t=0;t<T;++t)
        for(size_t d=0; d<D_MODEL; ++d)
            y[t*D_MODEL + d] = y1[t][d] + ffn_out[t][d];
}
