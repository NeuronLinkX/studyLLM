// include/llm_weights_io.h
#pragma once
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    size_t d_model, n_head, d_k, d_ff;
    // 포인터들(파일 내 연속 float 메모리로 읽어들여 heap에 소유)
    float *Wq;  // H*D_MODEL*D_K
    float *Wk;
    float *Wv;
    float *Wo;  // D_MODEL*D_MODEL
    float *W1;  // D_MODEL*D_FF
    float *b1;  // D_FF
    float *W2;  // D_FF*D_MODEL
    float *b2;  // D_MODEL
    float *ln1_gamma, *ln1_beta, *ln2_gamma, *ln2_beta;
} LlmWeights;

int llm_weights_load(const char *path, LlmWeights *out);
void llm_weights_free(LlmWeights *w);
