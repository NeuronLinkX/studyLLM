/* include/llm_math.h */
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void llm_softmax_row(float *row, size_t n);
void llm_layernorm(const float *x, float *y, size_t d, float eps,
                   const float *gamma, const float *beta);
/* y = A[MxK] * B[KxN], row-major, no-trans */
void llm_matmul(const float *A, const float *B, float *Y,
                size_t M, size_t K, size_t N);

#ifdef __cplusplus
}
#endif
