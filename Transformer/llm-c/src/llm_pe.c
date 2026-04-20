/* src/llm_pe.c */
#include "llm_pe.h"
#include <math.h>

void llm_posenc_sincos(float *x, size_t T, size_t D){
    /* x[t*D + d] += pe(t,d) */
    for(size_t t=0;t<T;++t){
        for(size_t d=0; d<D; d+=2){
            float pos = (float)t;
            float div = powf(10000.f, (float)d/(float)D);
            float pe_sin = sinf(pos/div);
            float pe_cos = (d+1<D) ? cosf(pos/div) : 0.f;
            x[t*D + d]     += pe_sin;
            if(d+1<D) x[t*D + d+1] += pe_cos;
        }
    }
}
