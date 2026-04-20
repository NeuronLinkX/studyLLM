/* src/llm_ffn.c */
#include "llm_ffn.h"
#include "llm_math.h"
#include <math.h>
#include <string.h>

#if defined(ACT_RELU)
static inline float relu(float x){ return x>0.f? x:0.f; }
#endif

static inline float relu(float x){ return x>0.f? x:0.f; }
static inline float gelu(float x){ /* tanh approximation */
    const float c = 0.7978845608f; /* sqrt(2/pi) */
    return 0.5f * x * (1.f + tanhf(c*(x + 0.044715f*x*x*x)));
}

void llm_ffn_forward(const float *X, float *Y, size_t T,
                     const LlmFfnWeights *w)
{
    float hidden[SEQ_LEN][D_FF];

    /* Hidden = X * W1 + b1, 활성함수 */
    for(size_t t=0;t<T;++t){
        const float *x=&X[t*D_MODEL];
        for(size_t j=0;j<D_FF;++j){
            float s=w->b1[j];
            for(size_t i=0;i<D_MODEL;++i) s += x[i]*w->W1[i][j];
#if USE_GELU
            hidden[t][j]=gelu(s);
#else
            hidden[t][j]=relu(s);
#endif
        }
    }

    /* Y = Hidden * W2 + b2 */
    for(size_t t=0;t<T;++t){
        float *y=&Y[t*D_MODEL];
        for(size_t j=0;j<D_MODEL;++j){
            float s=w->b2[j];
            for(size_t i=0;i<D_FF;++i) s += hidden[t][i]*w->W2[i][j];
            y[j]=s;
        }
    }
}
