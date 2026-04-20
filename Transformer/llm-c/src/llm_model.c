/* src/llm_model.c */
#include "llm_config.h"
#include "llm_model.h"
#include "llm_weights_io.h"
#include <string.h>
#include <stddef.h>
#include <stdio.h>

/* 내부 패턴 초기화(데모용) */
static void init_block(LlmDecoderBlock *blk){
    /* Wq/Wk/Wv */
    for (size_t h = 0; h < N_HEAD; ++h){
        float fq = 0.1f*(h+1), fk = 0.05f*(h+1), fv = -0.03f*(h+1);
        for (size_t i = 0; i < D_MODEL; ++i){
            for (size_t j = 0; j < D_K; ++j){
                blk->mha.Wq[h][i][j] = fq * (i+1);
                blk->mha.Wk[h][i][j] = fk * (i+1);
                blk->mha.Wv[h][i][j] = fv * (i+1);
            }
        }
    }
    /* Wo = I */
    for (size_t i = 0; i < D_MODEL; ++i)
        for (size_t j = 0; j < D_MODEL; ++j)
            blk->mha.Wo[i][j] = (i == j) ? 1.f : 0.f;

    /* FFN */
    for (size_t i = 0; i < D_MODEL; ++i)
        for (size_t j = 0; j < D_FF; ++j)
            blk->ffn.W1[i][j] = 0.02f * (float)((i+1)*(j+1));
    for (size_t j = 0; j < D_FF; ++j) blk->ffn.b1[j] = 0.f;

    for (size_t i = 0; i < D_FF; ++i)
        for (size_t j = 0; j < D_MODEL; ++j)
            blk->ffn.W2[i][j] = 0.01f * (float)((int)(i+1) - (int)(j+1));
    for (size_t j = 0; j < D_MODEL; ++j) blk->ffn.b2[j] = 0.f;

    /* LN gammas/betas */
    for (size_t d = 0; d < D_MODEL; ++d){
        blk->ln1_gamma[d] = 1.f; blk->ln1_beta[d] = 0.f;
        blk->ln2_gamma[d] = 1.f; blk->ln2_beta[d] = 0.f;
    }
}

/* 외부 weights.bin 로딩 → 블록에 주입 */
static void assign_from_loaded(LlmDecoderBlock* blk, const LlmWeights* w){
    /* Wq/Wk/Wv: H*D_MODEL*D_K 연속 → memcpy */
    size_t n_q = (size_t)N_HEAD * (size_t)D_MODEL * (size_t)D_K;
    memcpy(&blk->mha.Wq[0][0][0], w->Wq, n_q * sizeof(float));
    memcpy(&blk->mha.Wk[0][0][0], w->Wk, n_q * sizeof(float));
    memcpy(&blk->mha.Wv[0][0][0], w->Wv, n_q * sizeof(float));

    /* Wo */
    memcpy(&blk->mha.Wo[0][0], w->Wo, (size_t)D_MODEL * (size_t)D_MODEL * sizeof(float));

    /* FFN */
    memcpy(&blk->ffn.W1[0][0], w->W1, (size_t)D_MODEL * (size_t)D_FF * sizeof(float));
    memcpy(&blk->ffn.b1[0],    w->b1, (size_t)D_FF * sizeof(float));
    memcpy(&blk->ffn.W2[0][0], w->W2, (size_t)D_FF * (size_t)D_MODEL * sizeof(float));
    memcpy(&blk->ffn.b2[0],    w->b2, (size_t)D_MODEL * sizeof(float));

    /* LN */
    memcpy(&blk->ln1_gamma[0], w->ln1_gamma, (size_t)D_MODEL * sizeof(float));
    memcpy(&blk->ln1_beta[0],  w->ln1_beta,  (size_t)D_MODEL * sizeof(float));
    memcpy(&blk->ln2_gamma[0], w->ln2_gamma, (size_t)D_MODEL * sizeof(float));
    memcpy(&blk->ln2_beta[0],  w->ln2_beta,  (size_t)D_MODEL * sizeof(float));
}

void llm_model_init(LlmModel *m){
    LlmWeights w;
    if (llm_weights_load("build/weights.bin", &w)){
        if (w.d_model == D_MODEL && w.n_head == N_HEAD && w.d_k == D_K && w.d_ff == D_FF){
            assign_from_loaded(&m->block, &w);
            fprintf(stderr, "[llm] loaded external weights.bin\n");
            llm_weights_free(&w);
            return;
        }
        llm_weights_free(&w);
        fprintf(stderr, "[llm] weights.bin shape mismatch. fallback to internal init\n");
    }
    /* 내부 초기화 */
    init_block(&m->block);
}

void llm_model_forward(LlmModel *m, const float *xin, float *yout, size_t T){
    llm_decoder_block_forward(xin, yout, T, &m->block);
}
