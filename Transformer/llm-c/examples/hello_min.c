/**
 * @file hello_min.c
 * @brief 최소 디코더 블록 데모 (Positional Encoding + Decoder Block → "Hello, world!" 출력)
 *
 * - vocab 정의: {"<bos>","Hello",",","world","!","<eos>"} 
 *   · <bos> (Beginning of Sequence): 시퀀스 시작 토큰
 *   · <eos> (End of Sequence): 시퀀스 종료 토큰
 *   · 실제 모델에서는 디코딩 루프에서 <eos>를 만나면 생성 중단
 *
 * - 동작 개요:
 *   1) 입력 X를 0으로 초기화하고, 위치 인코딩(PE)을 추가
 *   2) 디코더 블록 1스텝 수행 (Pre-LN → MHA → FFN → Residual)
 *   3) 데모에서는 로짓을 계산하지 않고, 고정 규칙으로 "Hello, world!" 출력
 *
 * - 목적: LLM 디코더 파이프라인의 최소 전파 경로를 확인하기 위한 학습용 예제
 *   (실제 구현에서는 임베딩 lookup, out-proj, 토큰 샘플링 루프, KV-cache가 필요함)
 *
 * @author
 *  Azabell1993 (https://github.com/Azabell1993)
 * @date
 *  2025
 * @copyright
 *  Copyright 2025. All rights reserved.
 */

#include <stdio.h>
#include <string.h>
#include "llm_model.h"
#include "llm_pe.h"

static const char* vocab[] = {"<bos>","Hello",",","world","!","<eos>"};
enum { BOS=0, HELLO=1, COMMA=2, WORLD=3, EXC=4, EOS=5 };


/**
 * main()
 * -----------------------------------------------------------------------------
 * 설계 개요
 *  - 학습용 “미니 디코더” 파이프라인의 최소 동작을 보여주는 데모 엔트리포인트.
 *  - 실제 LM처럼 토크나이저/임베딩/출력프로젝션을 모두 구현하진 않고,
 *    1-스텝 디코더 블록을 통과시킨 뒤, 의도된 규칙으로 "Hello, world!"를 인쇄한다.
 *
 * 입력/내부 데이터
 *  - X[1][D_MODEL]: 1토큰(batch=1, T=1)의 임베딩 버퍼. 여기서는 0으로 채우고,
 *    위치인코딩(PE)을 더해 최소한의 위치신호만 제공한다.
 *  - LlmModel M: 내부적으로 Pre-LN 디코더 블록(= LN→MHA→Res→LN→FFN→Res)을 1회 수행.
 *    가중치는 외부 weights.bin이 있으면 로드하고, 없으면 데모용 초기화 사용.
 *  - Y[1][D_MODEL]: 블록 출력(히든 상태). 실제 모델이라면 여기에 vocab out-proj를 곱해 로짓 산출.
 *
 * 처리 순서
 *  1) 입력 버퍼 X를 0으로 초기화 → 사인/코사인 기반 PE를 더함.
 *  2) 모델 초기화(llm_model_init) → 블록 전파(llm_model_forward).
 *  3) (학습 편의상) 출력 Y를 로짓으로 해석하지 않고, 고정 규칙으로 "Hello, world!" 인쇄.
 *
 * 제약/의도
 *  - 본 데모는 “토큰 생성”이 아닌 “블록 전파 경로 확인”이 목적.
 *  - 토크나이저, 출력 프로젝션(Weight tying), KV-cache, RoPE 등은 구현 간소화를 위해 생략.
 *  - GGUF/실제 모델을 붙이려면 이름/차원 매핑, out-proj, 디코딩 루프 등을 추가해야 함.
 *
 * 기대 결과
 *  - 외부 가중치가 있을 경우 “[llm] loaded external weights.bin” 로그 출력 후,
 *    "Hello, world!"가 정확히 출력되면 데모 성공.
 */
int main(){
    // 1) 입력 임베딩 준비 (T=1 토큰, 배치=1). 여기선 학습용이라 0으로 시작.
    float X[1][D_MODEL]={0};

    // 2) 위치 인코딩(사인/코사인) 부여: RNN이 아닌 디코더에서 순서감각을 주는 핵심 요소.
    //    실제 모델에선 토큰 임베딩(embedding lookup) + PE가 합쳐져 입력으로 들어감.
    llm_posenc_sincos(&X[0][0], 1, D_MODEL);

    // 3) 모델 생성/초기화: 외부 weights.bin → 규격 OK면 로드, 아니면 내부 패턴 초기화.
    LlmModel M; llm_model_init(&M);

    // 4) 디코더 블록 1-스텝 전파. 결과는 Y로 수집(실전이면 out-proj 과정을 이어감).
    float Y[1][D_MODEL];
    llm_model_forward(&M, &X[0][0], &Y[0][0], 1);
    
    // 5) 실제 생성 단계는 아직 구현하지 않았다.
    //    즉, vocab 로짓 계산 -> softmax -> sampling/argmax 대신
    //    데모 목적의 하드코딩 규칙으로 "Hello, world!"를 직접 출력한다.
    printf("규칙 기반 출력(softmax/sampling 미구현): %s%s %s%s\n",
           vocab[HELLO], vocab[COMMA], vocab[WORLD], vocab[EXC]);

    return 0;
}
