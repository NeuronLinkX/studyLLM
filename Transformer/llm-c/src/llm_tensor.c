/* src/llm_tensor.c */
#include "llm_tensor.h"
#include <stdlib.h>
#include <string.h>

void *llm_aligned_malloc(size_t bytes, size_t alignment) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, alignment);
#else
    void *p = NULL;
    if (posix_memalign(&p, alignment, bytes) != 0) return NULL;
    return p;
#endif
}
void llm_aligned_free(void *p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}
void llm_zero_f(float *p, size_t n){ memset(p, 0, n * sizeof(float)); }
void llm_copy_f(float *dst, const float *src, size_t n){ memcpy(dst, src, n * sizeof(float)); }
