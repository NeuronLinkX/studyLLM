#include <stdio.h>
#include "llm_model.h"
#include "llm_pe.h"
#include "llm_config.h"

int main(void){
    /* 입력 임베딩(데모) [T x D] */
    float X[SEQ_LEN][D_MODEL] = {
        { 1.0f, 0.5f, -1.0f, 2.0f, 0.0f, -0.5f},
        { 0.3f,-0.7f,  0.8f,-0.1f, 1.0f,  0.2f},
        { 0.0f, 0.4f,  0.5f,-0.5f,-0.2f,  0.3f}
    };

    /* 1) 위치 인코딩 추가(논문 본문과 정합) */
    llm_posenc_sincos(&X[0][0], SEQ_LEN, D_MODEL);

    /* 2) 모델 init + 1블록 forward */
    LlmModel M; llm_model_init(&M);
    float Y[SEQ_LEN][D_MODEL];
    llm_model_forward(&M, &X[0][0], &Y[0][0], SEQ_LEN);

    /* 3) 결과 출력 */
    for(int t=0;t<SEQ_LEN;++t){
        printf("Token %d output: ", t);
        for(int j=0;j<D_MODEL;++j) printf("% .6f ", Y[t][j]);
        printf("\n");
    }
    return 0;
}