/* src/llm_attention.c */
#include "llm_attention.h"
#include "llm_math.h"
#include <math.h>
#include <string.h>

static inline float dotf(const float *a, const float *b, size_t n){
    float s=0.f; for(size_t i=0;i<n;++i) s+=a[i]*b[i]; return s;
}

void llm_mha_forward(const float *X, float *Y, size_t T,
                     const LlmMhaWeights *w)
{
    /* Q/K/V: [H x T x D_K] */
    float Q[N_HEAD][SEQ_LEN][D_K];
    float K[N_HEAD][SEQ_LEN][D_K];
    float V[N_HEAD][SEQ_LEN][D_K];

    /* 1) 프로젝션: Q/K/V */
    for(size_t h=0; h<N_HEAD; ++h){
        for(size_t t=0; t<T; ++t){
            const float *xrow=&X[t*D_MODEL];
            for(size_t j=0;j<D_K;++j){
                float q=0.f, k=0.f, v=0.f;
                for(size_t i=0;i<D_MODEL;++i){
                    float xi = xrow[i];
                    q += xi * w->Wq[h][i][j];
                    k += xi * w->Wk[h][i][j];
                    v += xi * w->Wv[h][i][j];
                }
                Q[h][t][j]=q; K[h][t][j]=k; V[h][t][j]=v;
            }
        }
    }

    /* 2) 각 헤드 스코어/소프트맥스/가중합 */
    float heads[N_HEAD][SEQ_LEN][D_K];
    const float scale = 1.f / sqrtf((float)D_K);
    for(size_t h=0; h<N_HEAD; ++h){
        float score[SEQ_LEN][SEQ_LEN];
        for(size_t t=0;t<T;++t){
            for(size_t s=0;s<T;++s){
                float val = dotf(Q[h][t], K[h][s], D_K) * scale;
#if USE_CAUSAL_MASK
                if(s>t) val = -1e30f;
#endif
                score[t][s]=val;
            }
        }
        for(size_t t=0;t<T;++t) llm_softmax_row(score[t], T);

        for(size_t t=0;t<T;++t){
            for(size_t r=0;r<D_K;++r){
                float sum=0.f;
                for(size_t s=0;s<T;++s) sum += score[t][s]*V[h][s][r];
                heads[h][t][r]=sum;
            }
        }
    }

    /* 3) concat(H) → Wo */
    for(size_t t=0;t<T;++t){
        float concat[D_MODEL];
        for(size_t h=0;h<N_HEAD;++h)
            for(size_t r=0;r<D_K;++r)
                concat[h*D_K + r] = heads[h][t][r];

        /* Y[t,:] = concat * Wo */
        for(size_t j=0;j<D_MODEL;++j){
            float s=0.f;
            for(size_t i=0;i<D_MODEL;++i) s += concat[i]*w->Wo[i][j];
            Y[t*D_MODEL + j] = s;
        }
    }
}
