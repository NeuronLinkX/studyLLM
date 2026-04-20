/* include/llm_tensor.h */
#pragma once
#include <stddef.h>
#include <stdint.h>

void *llm_aligned_malloc(size_t bytes, size_t alignment);
void  llm_aligned_free(void *p);

/* 단순 범용 유틸 */
void llm_zero_f(float *p, size_t n);
void llm_copy_f(float *dst, const float *src, size_t n);
