/* src/llm_math.c */
#include "llm_math.h"
#include <math.h>
#include <stddef.h>

void llm_softmax_row(float *row, size_t n){
    float maxv = row[0];
    for(size_t i=1;i<n;++i) if(row[i]>maxv) maxv=row[i];
    float sum=0.f;
    for(size_t i=0;i<n;++i){ row[i] = expf(row[i]-maxv); sum+=row[i]; }
    float inv = (sum>0.f)? (1.f/sum):1.f;
    for(size_t i=0;i<n;++i) row[i]*=inv;
}

void llm_layernorm(const float *x, float *y, size_t d, float eps,
                   const float *gamma, const float *beta){
    float mean=0.f, var=0.f;
    for(size_t i=0;i<d;++i) mean += x[i];
    mean /= (float)d;
    for(size_t i=0;i<d;++i){ float t=x[i]-mean; var += t*t; }
    var /= (float)d;
    float inv_std = 1.f / sqrtf(var + eps);
    for(size_t i=0;i<d;++i){
        float nrm = (x[i]-mean)*inv_std;
        float g = gamma ? gamma[i] : 1.f;
        float b = beta  ? beta[i]  : 0.f;
        y[i] = g*nrm + b;
    }
}

void llm_matmul(const float *A, const float *B, float *Y,
                size_t M, size_t K, size_t N){
    for(size_t m=0;m<M;++m){
        for(size_t n=0;n<N;++n){
            float acc=0.f;
            const float *a=&A[m*K];
            for(size_t k=0;k<K;++k) acc += a[k]*B[k*N+n];
            Y[m*N+n]=acc;
        }
    }
}
