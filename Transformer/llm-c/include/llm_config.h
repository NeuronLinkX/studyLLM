/* include/llm_config.h */
#pragma once

#include <stdalign.h>

/* 모델 하이퍼파라미터(데모용 소형) */
#define SEQ_LEN   3
#define D_MODEL   6
#define N_HEAD    2
#define D_K       (D_MODEL / N_HEAD)
#define D_FF      (4 * D_MODEL)   /* FFN 확장비 4x (논문 디폴트) */

/* 기능 토글 */
#define USE_CAUSAL_MASK 1         /* 디코더용 마스크 */
#define USE_GELU        1         /* FFN 활성함수: 1=GELU, 0=ReLU */

/* 정렬(캐시/벡터화 유리) */
#define LLM_ALIGN alignas(32)
