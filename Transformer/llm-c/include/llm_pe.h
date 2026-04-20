/* include/llm_pe.h */
#pragma once
#include <stddef.h>
void llm_posenc_sincos(float *x /*[T*D]*/, size_t T, size_t D);
