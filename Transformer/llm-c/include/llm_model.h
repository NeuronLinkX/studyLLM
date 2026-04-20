/* include/llm_model.h */
#pragma once
#include "llm_block.h"
#include "llm_config.h"
#include <stddef.h>

typedef struct {
    LlmDecoderBlock block; /* 데모: 1블록 */
    /* (옵션) 토큰 임베딩/출력 헤드 등 확장 */
} LlmModel;

void llm_model_init(LlmModel *m);
void llm_model_forward(LlmModel *m, const float *xin /*[T*D]*/, float *yout /*[T*D]*/, size_t T);
