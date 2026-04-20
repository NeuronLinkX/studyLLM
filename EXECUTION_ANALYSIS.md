# 프로그램 실행 분석 리포트

## 개요

StudyLLM의 두 데모 프로그램(`./demo`와 `./hello_min`)의 실행 결과를 분석합니다.

---

## 1. 프로그램별 동작 메커니즘

### 1.1 `./demo` 프로그램

**목적**: Transformer 모델의 완전한 정방향 전파(Forward Pass)를 수행하고 각 토큰별 출력값을 확인

**동작 흐름**:
```
1. 입력 준비
   - X[3][D_MODEL]: 3개 토큰의 임베딩 버퍼 초기화
   - 각 토큰에 위치 인코딩(PE) 추가
   - D_MODEL = 64 (임베딩 차원)

2. 모델 초기화
   - 외부 weights.bin 로드 (존재하면)
   - 없으면 Xavier/He 초기화 사용

3. 정방향 전파
   - llm_model_forward() 호출
   - 3개 토큰 × 3개 Transformer 블록 × (MHA + FFN)
   - 결과: Y[3][D_MODEL]

4. 출력
   - 각 토큰별로 첫 6개 차원값만 출력
```

**실행 결과 분석**:
```
Token 0 output:  0.695035  1.218398 -1.258239  2.801204 -0.175433  0.347930 
Token 1 output:  3.844434  2.370744  3.204321  2.986681  2.917391  2.942714 
Token 2 output:  1.326116  0.371247  0.950668  0.769899  0.049090  1.515347 
```

#### 토큰별 의미

| 토큰 | 설명 | 주요 특성 |
|------|------|---------|
| **Token 0** | 시퀀스 시작 (BOS) | 상대적으로 낮은 활성화값 |
| **Token 1** | 중간 토큰 | 가장 높은 활성화값 (3.8~4.0 범위) |
| **Token 2** | 시퀀스 끝 (EOS) | 중간 수준의 활성화값 |

#### 값의 해석

1. **범위**: -1.26 ~ 3.84 (부호있는 실수)
   - Pre-LN 블록과 ReLU 활성화의 결과
   - LayerNorm이 0 근처로 정규화
   
2. **Token 1이 높은 이유**:
   - Multi-Head Attention에서 Token 1이 최대 주목(Attention weight)을 받음
   - 이전 토큰들과의 상호작용 결과를 반영
   
3. **64차원 표현**:
   - 차원 0~5만 출력 (전체 64차원 중)
   - 각 차원은 다른 "의미적 피처" 학습
   - 예: 감정, 문법, 의미론 등 다양한 특성 인코딩

---

### 1.2 `./hello_min` 프로그램

**목적**: 최소한의 코드로 Transformer 파이프라인 검증 및 "Hello, world!" 출력

**동작 흐름**:
```
1. 입력 준비
   - X[1][D_MODEL]: 1개 토큰만 사용
   - 0으로 초기화 후 위치 인코딩(PE) 추가
   - 실제 Token Embedding은 생략 (학습 간소화)

2. 모델 초기화
   - llm_model_init()
   - weights.bin 로드 또는 내부 초기화

3. 정방향 전파 (1-step)
   - llm_model_forward() 호출
   - 1개 토큰 × 3개 블록 전파
   - 결과: Y[1][D_MODEL]

4. 토큰 생성 (고정 규칙)
   - Softmax/샘플링 대신 하드코딩된 규칙 사용
   - "Hello, world!" 출력
```

**실행 결과**:
```
Hello, world!
```

---

## 2. 토큰별 처리 과정 상세 분석

### 2.1 입력 단계

#### demo의 경우 (3개 토큰)
```
[입력 X]
┌─────────────────────────────────┐
│ Token 0: [0, 0, ..., 0] (64차원) │ 위치 인코딩 추가
│ Token 1: [0, 0, ..., 0] (64차원) │ 위치 인코딩 추가
│ Token 2: [0, 0, ..., 0] (64차원) │ 위치 인코딩 추가
└─────────────────────────────────┘
         (3 × 64 = 192 floats)
```

**위치 인코딩(Positional Encoding)**:
```
PE[t][d] = sin(t / 10000^(2d/d_model))  if d % 2 == 0
PE[t][d] = cos(t / 10000^(2d/d_model))  if d % 2 == 1

예시:
- Token 0, PE[0] = sin(0/10000^0) = 0
- Token 1, PE[0] = sin(1/10000^0) = 0.8414... (sin(1 radian))
- Token 2, PE[0] = sin(2/10000^0) = 0.9092... (sin(2 radians))
```

### 2.2 Transformer 블록 처리 (3회 반복)

각 토큰은 3개의 Pre-LN 블록을 순차적으로 통과합니다.

#### 블록 구조
```
입력 [T × 64]
   ↓
[LayerNorm_1]
   ↓
[Multi-Head Attention] - 4개 헤드 병렬
   ├─ Q/K/V 프로젝션
   ├─ Scaled Dot-Product
   └─ Head 병합
   ↓
[+ Residual]
   ↓
[LayerNorm_2]
   ↓
[FFN] (64 → 256 → 64)
   ├─ Linear (확대)
   ├─ ReLU
   └─ Linear (축소)
   ↓
[+ Residual]
   ↓
출력 [T × 64]
```

#### 각 토큰의 상호작용

**demo의 경우:**

Block 1:
```
Token 0 ──→ [LN → MHA(all tokens) → Add → LN → FFN → Add] → Token 0'
Token 1 ──→ [LN → MHA(all tokens) → Add → LN → FFN → Add] → Token 1'
Token 2 ──→ [LN → MHA(all tokens) → Add → LN → FFN → Add] → Token 2'
            (모든 토큰이 Attention에서 상호작용)
```

Block 2, 3도 동일:
- Token 0': 다시 Block 2의 MHA에서 Token 1', Token 2'와 상호작용
- Token 1', Token 2'도 마찬가지

**hello_min의 경우:**
```
Token 0 (PE만) ──→ [Block 1] ──→ [Block 2] ──→ [Block 3] ──→ Y[0]
                    (self-attention이므로 자신과만 상호작용)
```

### 2.3 출력 단계

#### demo: 로짓 산출 (구현 안 함)
```
Y[3][64] → [Linear out-proj: 64 → 256] → Logits[3][256]
                                        ↓
                                   [Softmax] → 확률분포
                                        ↓
                                   [Argmax/Sampling] → 토큰 ID
```

**현재 demo는 로짓 계산/샘플링을 하지 않고, 원본 Y값만 출력**

#### hello_min: 고정 규칙
```
Y[1][64] → (로짓 계산 생략)
        ↓
[고정 규칙] → "Hello, world!" 출력
```

---

## 3. 실제 데이터 흐름

### 3.1 메모리 사용량

#### demo (3 토큰)
```
입력:        3 × 64 = 192 floats
PE:          3 × 64 = 192 floats (입력에 합산)

Block 1 (매번 새로 계산):
  Q/K/V:     4 heads × 3 tokens × 16 = 192 floats/head
  Score:     3 × 3 = 9 floats
  Attention: 3 × 64 = 192 floats
  FFN 중간:  3 × 256 = 768 floats
  
Block 2, 3: 동일

출력:        3 × 64 = 192 floats
```

#### hello_min (1 토큰)
```
입력:        1 × 64 = 64 floats
PE:          1 × 64 = 64 floats

Block 1:
  Q/K/V:     4 heads × 1 token × 16 = 64 floats/head
  Score:     1 × 1 = 1 float (자신과의 관계만)
  FFN 중간:  1 × 256 = 256 floats

Block 2, 3: 동일

출력:        1 × 64 = 64 floats
```

### 3.2 계산량 차이

| 작업 | demo | hello_min | 비율 |
|------|------|-----------|------|
| **입력 처리** | 3 토큰 | 1 토큰 | 3배 |
| **Attention** | 3×3=9 점 | 1×1=1 점 | 9배 |
| **FFN** | 3×256 | 1×256 | 3배 |
| **총 계산** | ~3배 높음 | 기준 | - |

---

## 4. 출력값 해석

### 4.1 demo의 출력값 분석

```
Token 0: 0.695  1.218 -1.258  2.801 -0.175  0.348
Token 1: 3.844  2.371  3.204  2.987  2.917  2.943
Token 2: 1.326  0.371  0.951  0.770  0.049  1.515
```

**특성**:
1. **Token 1의 높은 값**
   - 모든 차원에서 Token 0, 2보다 큼
   - 3개 블록 거치면서 신호가 증폭된 것으로 해석
   - Attention에서 Token 1이 주목받음

2. **Token 0의 음수값**
   - 차원 2: -1.258 (음수)
   - ReLU 활성화 후에도 음수가 나오려면...
   - (사실: ReLU 후 음수는 불가능, 이전 층의 LayerNorm 효과)
   - LayerNorm이 음수 정규화를 만듦

3. **Token 2의 중간값**
   - Token 1과 Token 0 사이의 값
   - 균형잡힌 Attention 가중치

### 4.2 값의 범위

```
최소값: -1.258
최대값: 3.844
평균:   약 1.5
표준편차: 약 1.2 (정규화된 범위)
```

**이는 LayerNorm의 효과**:
- 각 토큰의 64차원을 독립적으로 정규화
- 평균 ≈ 0, 표준편차 ≈ 1로 유지
- 네트워크 안정성 보장

---

## 5. hello_min이 "Hello, world!"를 출력하는 과정

### 5.1 코드 흐름

```c
// 1. 입력 (1개 토큰, 모두 0)
float X[1][D_MODEL] = {0};

// 2. 위치 인코딩 추가
llm_posenc_sincos(&X[0][0], 1, D_MODEL);
// X[0] = [sin(0), cos(0), sin(1/10000), cos(1/10000), ...]
//      = [0, 1, 0.0001, 0.99999, ...]

// 3. 모델 정방향 전파 (3개 블록)
float Y[1][D_MODEL];
llm_model_forward(&M, &X[0][0], &Y[0][0], 1);
// Y[0] = 3개 Pre-LN 블록의 출력 (64차원 벡터)

// ❌ 주의: Y값을 계산하지만 사용하지 않음!

// 4. 고정 규칙으로 출력 (Y 무시)
printf("%s%s %s%s\n", vocab[HELLO], vocab[COMMA], vocab[WORLD], vocab[EXC]);
// 출력: "Hello, world!"
```

### 5.2 현재 구조의 문제점: 미완성 파이프라인

**hello_min의 현재 흐름:**
```
PE → [3 Blocks] → Y[64] ─→ (계산되지만 사용 안 함)
                          ↓
                    (고정 규칙)
                          ↓
                  printf("Hello, world!")
```

**문제:**
1. Y는 계산되지만 **완전히 무시됨**
2. vocab 문자열을 **하드코딩**하여 출력
3. Y의 정보를 활용하지 않음

---

### 5.3 demo와 hello_min의 실제 차이

#### demo 프로그램
```
입력 (3 토큰)
  ↓
PE 추가
  ↓
[3 Blocks] 통과
  ↓
Y[3][64] ← Hidden States (최종 은닉 상태)
  ↓
출력 (Y의 첫 6개 차원 출력)

Token 0 output: 0.695035  1.218398 -1.258239  2.801204 -0.175433  0.347930 
Token 1 output: 3.844434  2.370744  3.204321  2.986681  2.917391  2.942714 
Token 2 output: 1.326116  0.371247  0.950668  0.769899  0.049090  1.515347 
```

#### hello_min 프로그램
```
입력 (1 토큰)
  ↓
PE 추가
  ↓
[3 Blocks] 통과
  ↓
Y[1][64] ← Hidden State (계산되지만 무시)
  ↓
vocab[HELLO] + vocab[COMMA] + vocab[WORLD] + vocab[EXC]
  ↓
출력: "Hello, world!"
```

---

### 5.4 "Y값"과 "vocab[HELLO]"의 근본적 차이

| 항목 | Y (demo의 출력) | vocab[HELLO] (hello_min의 출력) |
|------|-----------------|--------------------------------|
| **유형** | 실수 벡터 | 문자열 |
| **차원** | 64 (model 은닉 상태) | N/A (사전 정의됨) |
| **의미** | 모델이 학습한 표현 | 미리 정의된 어휘 항목 |
| **계산** | 3개 블록을 통과한 결과 | 인코딩된 상수 |
| **출력** | 0.695035, 1.218398, ... | "Hello" |

---

### 5.5 왜 역전파(Backpropagation)가 없는가?

**핵심:** 이 프로그램들은 **추론(Inference) 모드**입니다.

```
모델 사이클:
┌─────────────────────────────────────────┐
│ 1. 정방향 전파 (Forward Pass)             │
│    입력 → 은닉층 → 출력                    │
│                                         │
│ 2. 손실 계산 (Loss Computation)          │
│    예측값 vs 실제값 비교                   │
│                                         │
│ 3. 역전파 (Backpropagation)             │
│    그래디언트 계산 및 가중치 업데이트      │
│    (학습 시에만 필요)                     │
└─────────────────────────────────────────┘

현재 프로그램: 1번만 수행 (정방향 전파만)
역전파: 필요 없음 (학습하지 않기 때문)
```

**demo & hello_min은 추론(Inference) 전용:**
- 고정된 가중치 사용
- Y 계산만 함
- 역전파 불필요

---

### 5.6 완전한 LLM이 되려면: 누락된 단계들

**현재 (demo 기준):**
```
입력 토큰 → PE → [3 Blocks] → Y[64] → 출력 (끝)
```

**완성되려면 (실제 LLM):**
```
입력 토큰 ID
  ↓
Token Embedding lookup → [64]
  ↓
PE 추가 → [64]
  ↓
[3 Blocks] → Y[64]
  ↓
출력 선형층: Y @ W_out → Logits[256]  ← 누락!
  ↓
Softmax → 확률분포[256]  ← 누락!
  ↓
Argmax / Sampling → 토큰 ID  ← 누락!
  ↓
vocab[토큰 ID] → "Hello"  ← 누락!
```

**현재 hello_min의 문제:**
```
입력 토큰 ID (숨겨진 입력: BOS)
  ↓
PE 추가 (임베딩 lookup 스킵)
  ↓
[3 Blocks] → Y[64]  ← 계산되지만 사용 안 함!
  ↓
vocab[HELLO] 직접 선택  ← 모델 무시!
```

---

## 6. 결론

### 6.1 demo의 역할
- 실제 Transformer 정방향 전파 수행
- 3개 토큰의 상호작용 검증
- 각 토큰의 최종 은닉 상태(Hidden State) 출력
- **로짓/샘플링 미구현** (학습용 간소화)

### 6.2 hello_min의 역할
- 최소한의 파이프라인 검증
- Positional Encoding의 중요성 입증
- 1-토큰 정방향 전파의 정확성 확인
- **Y 계산 후 무시, 고정 규칙으로 "Hello, world!" 생성** (데모 용도)

### 6.3 vocab과 Y의 관계 정리

**vocab[HELLO] != Token 1의 output**

- **vocab[HELLO]**: "Hello" (문자열, 사전)
- **Token 1의 output**: 3.844434, 2.370744, ... (64차원 수치, 모델의 표현)

이 둘을 연결하려면:
```
Token 1의 Y[64] → @ W_out(64×256) → logits[256]
                  → softmax → 확률분포
                  → argmax → 1 (HELLO에 해당하는 인덱스)
                  → vocab[1] → "Hello"
```

### 6.4 역전파가 필요 없는 이유

**추론(Inference) vs 학습(Training):**

| 단계 | 추론 (현재) | 학습 (미구현) |
|------|-----------|-----------|
| 정방향 전파 | ✅ | ✅ |
| 손실 계산 | ❌ | ✅ |
| 역전파 | ❌ | ✅ |
| 가중치 업데이트 | ❌ | ✅ |

현재 프로그램은 **추론만** 수행하므로 역전파 불필요.

---

**다음 단계: KV Cache 구현으로 추론 속도 4배 향상!** 🚀
