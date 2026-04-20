# StudyLLM: Transformer 학습 프로젝트

> 순수 C로 구현한 LLM 추론 엔진 - 교육용 최적화 구현

<div align="center">

![LLM Inference Pipeline Architecture Flow](./LLM%20Inference%20Pipeline%20Architecture%20Flow.png)

**Transformer 디코더 기반 Self-Attention 다중헤드 어텐션 구현**

**[개요](#개요) • [핵심 개념](#핵심-개념) • [아키텍처](#아키텍처-계층) • [빌드](#빌드-및-실행) • [보강 영역](#보강-가능-영역)**

</div>

---

## 개요

`StudyLLM`은 **Transformer 디코더 아키텍처**를 순수 C로 구현한 **LLM 추론 엔진**입니다.

- 교육용 최적화: 명확한 알고리즘으로 논문의 수식과 1:1 대응
- 단계별 학습: 텐서 연산 → Attention → FFN → 전체 모델
- 수치 안정성: LayerNorm, Softmax 스케일링으로 안정적 학습
- 메모리 효율: Pre-LN + 버퍼 재사용으로 임베딩 메모리 최소화
- 외부 가중치: Python 스크립트로 생성한 가중치 바이너리 지원

### 아키텍처 특징

| 특성 | 설명 |
|------|------|
| 기본 구조 | Pre-LN Decoder-Only Transformer |
| Attention | Scaled Dot-Product Attention + Multi-Head |
| 정규화 | Pre-Layer Normalization (안정성 향상) |
| 활성화 | GELU 근사 또는 ReLU |
| 초기화 | Xavier (선형) / He (ReLU) 초기화 |
| 선택적 | Causal Mask (Auto-regressive generation) |

---

## 프로젝트 구조

```
StudyLLM/
├── README.md
├── QUICK_START.md
├── ARCHITECTURE.md
├── IMPLEMENTATION_DETAILS.md
├── DEVELOPMENT_GUIDE.md
├── EXECUTION_ANALYSIS.md
├── LLM Inference Pipeline Architecture Flow.png
│
└── Transformer/llm-c/
    ├── include/
    │   ├── llm_config.h
    │   ├── llm_tensor.h
    │   ├── llm_math.h
    │   ├── llm_pe.h
    │   ├── llm_attention.h
    │   ├── llm_ffn.h
    │   ├── llm_block.h
    │   ├── llm_model.h
    │   └── llm_weights_io.h
    │
    ├── src/
    │   ├── llm_attention.c
    │   ├── llm_block.c
    │   ├── llm_model.c
    │   └── ...
    │
    ├── examples/
    │   ├── demo_forward.c
    │   └── hello_min.c
    │
    └── scripts/
        └── gen_weights.py
```

---

## 핵심 개념

### (1) Scaled Dot-Product Self-Attention (완전 구현)

**수식**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Transformer/llm-c 구현의 특징**:
- Self-Attention: 모든 토큰이 모든 토큰과 상호작용
- Multi-Head (4헤드): 병렬로 4개 표현 공간에서 학습
- Causal Mask 선택 지원: Auto-regressive generation
- 수치 안정성: Softmax 전 행별 max 감산으로 overflow 방지
- 메모리 효율: 헤드별로 Q/K/V 독립 계산

**코드 위치**: [llm_attention.c](Transformer/llm-c/src/llm_attention.c)

**Self-Attention 메커니즘의 4단계 구현 프로세스**:
```c
// (1) Q/K/V 프로젝션: X @ W_q, W_k, W_v (헤드별 계산)
// (2) 스코어 계산: QK^T / sqrt(d_k) + Causal Mask (선택)
// (3) Softmax + 가중합: softmax(score) @ V (Attention 계산)
// (4) 헤드 결합: concat(heads) @ W_o (최종 출력)
```

**메모리**: O(T^2 * d_k) (시퀀스 길이 제곱에 비례)

---

### (2) Pre-LN Transformer 블록 (2024년 최신)

**구조** (Residual + LayerNorm 순서):
```
Input
  ↓
[LayerNorm] → [Multi-Head Attention] → [+ Residual]
  ↓
[LayerNorm] → [Feed Forward Network] → [+ Residual]
  ↓
Output
```

**vs Post-LN (이전 방식)**:
```
Post-LN (불안정):
Input → [MHA] → [+ Residual] → [LayerNorm]

Pre-LN (안정적):
Input → [LayerNorm] → [MHA] → [+ Residual]
```

**장점**:
- 학습 안정성: LayerNorm이 먼저 정규화
- 깊은 네트워크: Residual 스케일 폭주 방지
- 수렴 빠름: 초기화 이후 더 빠른 수렴

**코드 위치**: [llm_block.c](Transformer/llm-c/src/llm_block.c)

---

### (3) Feed Forward Network

**수식**:
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1) W_2 + b_2$$

**구조**:
```
x: [T × d_model]
   ↓ W_1: [d_model × 4·d_model] (확대)
hidden: [T × 4·d_model]
   ↓ W_2: [4·d_model × d_model] (축소)
out: [T × d_model]
```

**메모리 효율**:
- 중간 활성화를 1개 버퍼로 순차 재사용
- O(T·d_model·d_ff) 메모리로 제한

**코드 위치**: [llm_ffn.c](Transformer/llm-c/src/llm_ffn.c)

---

## 아키텍처 계층

### 계층 구조

```
계층 1: 입력 처리 (Input Layer)
├─ 토큰 ID (256개)
├─ Token Embedding: [T × 64]
├─ Positional Encoding: [T × 64]
└─ 결과: [T × d_model]

계층 2: Transformer 블록 (×3)
├─ Pre-LN Block
│  ├─ LayerNorm
│  ├─ MHA (4 heads)
│  │  ├─ Q/K/V 프로젝션
│  │  ├─ Scaled attention
│  │  └─ Head concat
│  ├─ Add & Norm
│  │
│  ├─ LayerNorm
│  ├─ FFN (4x expand)
│  │  ├─ Linear + ReLU
│  │  └─ Linear (축소)
│  └─ Add & Norm
└─ Layer 1 → Layer 2 → Layer 3

계층 3: 출력 처리 (Output Layer)
├─ Final LayerNorm
├─ Linear (d_model → vocab_size)
├─ Softmax → 확률분포
└─ Argmax/Sampling → 토큰
```

### 하이퍼파라미터 (기본값)

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| D_MODEL | 64 | 임베딩 차원 |
| N_HEAD | 4 | 어텐션 헤드 수 |
| D_K | 16 | 헤드당 차원 (64÷4) |
| D_FF | 256 | FFN 중간층 차원 (4×64) |
| N_LAYERS | 3 | Transformer 블록 반복 |
| SEQ_LEN | 256 | 최대 시퀀스 길이 |
| VOCAB_SIZE | 256 | 어휘집 크기 (ASCII) |

---

## 현재 구현 상태 분석

### 완성된 부분

| 모듈 | 상태 | 설명 |
|------|------|------|
| Tensor 기초 | 완료 | 행렬 연산, 활성화 함수 |
| Self-Attention | 완료 | Scaled Dot-Product + Multi-Head |
| Feed Forward | 완료 | 2-층 선형 + ReLU |
| LayerNorm | 완료 | Pre-normalization |
| Positional Encoding | 완료 | Learnable PE + scale |
| Pre-LN 블록 | 완료 | 블록 조립 + Residual |
| 모델 추론 | 완료 | Forward pass + 외부 가중치 |
| Causal Mask | 완료 | Auto-regressive 생성 지원 |

---

## 보강 가능 영역

### (1) Group Query Attention (GQA)

- K/V를 여러 헤드에서 공유하여 메모리 효율 증대
- 기존: 각 헤드별 독립 W_k, W_v → GQA: 1개만 사용

```c
typedef struct {
    float Wq[N_HEAD][D_MODEL][D_K];
    float Wk[GROUP_SIZE][D_MODEL][D_K];    // K/V 공유
    float Wv[GROUP_SIZE][D_MODEL][D_K];
    float Wo[D_MODEL][D_MODEL];
} LlmGqaWeights;
```

### (2) KV Cache

- 추론 시 과거 토큰의 K, V 저장하여 재계산 방지
- 초회: 모든 토큰 계산 → 이후: 1개 토큰만 처리

```c
typedef struct {
    float k_cache[SEQ_LEN][D_MODEL];
    float v_cache[SEQ_LEN][D_MODEL];
    int cached_len;
} LlmKvCache;

void llm_attention_with_cache(
    const float *Q_new,
    float *attn_out,
    LlmKvCache *cache
);
```

### (3) Flash Attention

- 블록 단위 Tiling으로 메모리 효율 극대화
- Score 행렬 [T×T]를 블록으로 나누어 순차 처리

```c
// 메모리 O(T²) → O(T)로 감소
void llm_flash_attention_forward(...);
```

### (4) Quantization (INT8/INT4)

- Float32 → INT8/INT4로 가중치 압축
- 메모리: Float32(4B) → INT8(1B) 75% 감소

```c
typedef struct {
    int8_t data[SIZE];
    float scale;
    int8_t zero_point;
} QuantTensor;
```

### (5) Batch 처리

- 여러 시퀀스를 동시에 처리하여 처리량 증가

```c
typedef struct {
    const float *inputs[BATCH_SIZE];
    float *outputs[BATCH_SIZE];
    int lengths[BATCH_SIZE];
} LlmBatch;
```

### (6) SIMD 최적화

- ARM NEON (모바일), x86 AVX2 (CPU) 명령어 활용
- 행렬곱, softmax 등 핵심 연산 가속

```c
#ifdef __ARM_NEON__
void matmul_neon(const float *a, const float *b, float *c, int M, int N, int K);
#endif
```

### (7) GPU 지원

- CUDA (NVIDIA), Metal (Apple Silicon) 백엔드 추가
- 대규모 행렬 연산 병렬화

```c
#ifdef ENABLE_CUDA
void matmul_cuda(const float *d_a, const float *d_b, float *d_c, int M, int N, int K);
#endif
```

### (8) Advanced Sampling

- Greedy(Argmax) 대신 Top-K, Top-P(nucleus) 샘플링
- 더 자연스러운 생성 품질

```c
void sample_top_k(const float *logits, int vocab_size, int k, float temperature, int *out);
void sample_top_p(const float *logits, int vocab_size, float p, float temperature, int *out);
```

---

## 빌드 및 실행

### macOS / Linux

```bash
cd Transformer/llm-c
mkdir build && cd build
cmake ..
make -j$(nproc)

# 실행
./demo              # 내부 초기화 데모
# 또는
./hello_min         # 고정 "Hello, world!" 예제
```

### 외부 가중치 생성 및 로드

```bash
# Python 환경 설정
cd Transformer/llm-c
python3 -m venv .venv
source .venv/bin/activate
pip install numpy

# 가중치 생성 (Xavier/He 초기화)
python3 scripts/gen_weights.py \
    --d_model 64 \
    --n_head 4 \
    --d_ff 256 \
    --n_layers 3 \
    --seed 42 \
    --out build/weights.bin

# demo에서 weights.bin 자동 로드 및 추론
cd build
./demo
```

---

## 학습 경로

### 추천 단계

```
1단계: 기초 이해 (2-3시간)
  - llm_config.h: 하이퍼파라미터 이해
  - llm_tensor.h: 행렬 연산 구조
  - llm_math.c: Softmax, LayerNorm 수식 분석

2단계: Attention 분석 (3-4시간)
  - llm_attention.h: 구조체 이해
  - llm_attention.c: 4단계 구현 추적
  - 종이에 그려보기: Q, K, V, Score, Attention

3단계: FFN & 블록 분석 (2-3시간)
  - llm_ffn.c: 확대-축소 구조
  - llm_block.c: Pre-LN + Residual 조립
  - 메모리 레이아웃 추적

4단계: 전체 모델 (2-3시간)
  - llm_model.c: 모델 초기화 + forward
  - demo_forward.c: 완전한 추론 흐름
  - 가중치 로드 파이프라인

5단계: 개선 구현 (시간은 개선사항마다 다름)
  - KV Cache (3-4시간)
  - GQA (5-6시간)
  - 양자화 (6-8시간)
```

---

## 참고 자료

### 논문
- Attention is All You Need (https://arxiv.org/abs/1706.03762)
- RMSNorm vs LayerNorm (https://arxiv.org/abs/1910.07468)
- llama.cpp (https://github.com/ggerganov/llama.cpp)

### 이미지
- LLM.png - 블록 다이어그램
- LLM Inference Pipeline Architecture Flow.png - 전체 파이프라인

---

**Happy Learning!**
