# StudyLLM: Transformer 학습 프로젝트


<div align="center">

![LLM Inference Pipeline Architecture Flow](./LLM%20Inference%20Pipeline%20Architecture%20Flow.png)

**Transformer 디코더 기반 Self-Attention 다중헤드 어텐션 구현**

**[개요](#개요) • [핵심 개념](#핵심-개념) • [아키텍처](#아키텍처-계층) • [빌드](#빌드-및-실행) • [개발 가이드](#🚧-보강-가능-영역-개발-가이드)**

</div>


<div style="border: 2px solid #333; border-radius: 12px; padding: 20px; margin: 20px 0; background-color: #f9f9f9;">

## 학습 목적: Strategies for Handling Hallucination in Transformer 2.0

### (1) Forced Confidence of Softmax & Hallucination

**Core Mechanism & Issue:**
- **Softmax Layer**는 최종 Output Vector의 총합을 항상 **1.0(100%)**으로 정규화함.
- 특정 Domain 지식이 부재한 상태에서도 내부 Logit Value를 강제로 배분하여, 통계적으로 가장 높은 확률을 가진 **"Plausible Wrong Answer(그럴싸한 오답)"**을 선택하게 유도함.

**Expert Insight & Tactics:**
- **Expert Workflow** 내에서 **Softmax Threshold(임계값)**을 엄격히 거버넌스(Governance) 해야 함.
- **Probability Distribution**이 지나치게 분산된 **Uncertainty(불확실성)** 상태이거나, Max Probability가 기준치 미달일 경우 **"I don't know"** 혹은 **"Tool Execution Refusal"**을 Trigger하는 로직이 **Transformer 2.0 Agent** 설계의 Core임.

### (2) Attention Bias & Data Skewness Management

**Core Mechanism & Issue:**
- **Attention Mechanism**은 Query와 Key의 **Similarity(유사도)**를 산출하여 Value에 **Weight(가중치)**를 할당함.
- High-quality 산업 코드는 대부분 **Private Status**인 반면, 학습 데이터는 **Open-source**가 지배적임.
- 이로 인해 **Attention Score**가 수적 우위가 있는 일반 데이터로 쏠리는 **Attention Bias**가 발생하며, 모델은 정교한 **Exception Handling(예외 처리)**보다 전시용 코드 패턴을 더 'Important'하다고 오판함.

**Expert Insight & Tactics:**
- **RAG(Retrieval-Augmented Generation)** 운용 시, 단순 연상 검색을 넘어 전문가의 **Core Logic**에 고정된 Attention을 부여하는 **In-context Weighting** 설계가 필요함.
- Generic Data보다 사용자가 주입한 **Narrow & Deep Context**에 모델이 압도적인 가중치를 두도록 유도하는 **Advanced Prompt Engineering**이 필수적임.

### (3) Multi-dimensional Evaluation Function: Beyond Loss to Domain Logic

**Core Mechanism & Issue:**
- Conventional LLM은 예측값과 실제값의 차이인 **Loss Function**을 최소화하는 방향으로만 **Optimization**됨.
- **Verification Agent**의 평가함수가 단순 **Syntax Check**나 **Unit Test** 통과에만 의존할 경우, 모델은 논리적 결함이 있는 **"Functionally working but technically garbage code"**를 생성하는 방향으로 수렴함.

**Expert Insight & Tactics:**
- 평가 메트릭에 **Softmax Entropy(정보 엔트로피)** 측정 지표를 통합하여 모델의 확신도를 정량화해야 함.
- 모델이 낮은 확신도로 결과를 도출했는지, 혹은 특정 **Open-source Pattern**에 **Over-attention**되었는지 감시(Monitoring)함으로써 **Hallucination Symptom**을 사전에 감지하는 다중 방어 체계를 구축해야 함.

</div>

## 개요

`StudyLLM`은 **Transformer 디코더 아키텍처**를 순수 C로 구현한 **LLM 추론 엔진**입니다.

- **교육용 최적화**: 명확한 알고리즘으로 논문의 수식과 1:1 대응
- **단계별 학습**: 텐서 연산 → Attention → FFN → 전체 모델
- **수치 안정성**: LayerNorm, Softmax 스케일링으로 안정적 학습
- **메모리 효율**: Pre-LN + 버퍼 재사용으로 임베딩 메모리 최소화
- **외부 가중치**: Python 스크립트로 생성한 가중치 바이너리 지원

## 핵심

**Transformer/llm-c는 Self-Attention 기반 디코더입니다.**

- Scaled Dot-Product Attention (간소화 없음)
- 4 Multi-Head 병렬 처리
- Pre-LN 블록 (최신, 안정적)
- FFN + Causal Mask 지원
- 메모리 병목: O(T²) 복잡도

### 아키텍처 특징

| 특성 | 설명 |
|------|------|
| **기본 구조** | Pre-LN Decoder-Only Transformer |
| **Attention** | **Scaled Dot-Product Attention** + Multi-Head (간소화) |
| **정규화** | **Pre-Layer Normalization** (안정성 향상) |
| **활성화** | GELU 근사 또는 ReLU |
| **초기화** | Xavier (선형) / He (ReLU) 초기화 |
| **선택적** | Causal Mask (Auto-regressive generation) |

---

## 프로젝트 구조

```
StudyLLM/
├── README.md                       # 이 파일
├── QUICK_START.md                  # 빠른 시작
├── ARCHITECTURE.md                 # 상세 아키텍처
├── LLM Inference Pipeline Architecture Flow.png
│
└── Transformer/llm-c/              # C 기반 LLM 엔진
    ├── include/
    │   ├── llm_config.h            # 하이퍼파라미터
    │   ├── llm_tensor.h            # 텐서 + 기본 연산
    │   ├── llm_math.h              # 행렬곱, softmax, layernorm
    │   ├── llm_pe.h                # 위치 임베딩
    │   ├── llm_attention.h         # ⭐ Multi-Head Self-Attention
    │   ├── llm_ffn.h               # Feed Forward
    │   ├── llm_block.h             # ⭐ Pre-LN 블록
    │   ├── llm_model.h             # ⭐ 모델 + 정방향
    │   └── llm_weights_io.h        # 외부 가중치 로드
    │
    ├── src/                        # 구현
    │   ├── llm_attention.c         # Scaled Dot-Product + MHA
    │   ├── llm_block.c             # 블록 조립
    │   ├── llm_model.c             # 모델 초기화 + forward
    │   └── ... (기타)
    │
    ├── examples/
    │   ├── demo_forward.c          # 완전한 추론 데모
    │   └── hello_min.c             # 최소 예제
    │
    └── scripts/
        └── gen_weights.py          # 가중치 생성 (Xavier/He)
```

---

## 핵심 개념

### Scaled Dot-Product Self-Attention

**수식**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Transformer/llm-c 구현의 특징**:
- **Self-Attention**: 모든 토큰이 모든 토큰과 상호작용
- **Multi-Head (4헤드)**: 병렬로 4개 표현 공간에서 학습
- **Causal Mask 선택 지원**: Auto-regressive generation (USE_CAUSAL_MASK 매크로)
- **수치 안정성**: Softmax 전 행별 max 감산으로 overflow 방지
- **메모리 효율**: 헤드별로 Q/K/V 독립 계산

**코드 위치**: [llm_attention.c](Transformer/llm-c/src/llm_attention.c)

**Self-Attention 메커니즘의 4단계 구현 프로세스**
```c
// Q/K/V 프로젝션: X @ W_q, W_k, W_v (헤드별 계산)
// 스코어 계산: QK^T / sqrt(d_k) + Causal Mask (선택)
// Softmax + 가중합: softmax(score) @ V (Attention 계산)
// 헤드 결합: concat(heads) @ W_o (최종 출력)
```

**메모리**: $O(T^2 \cdot d_k)$ (시퀀스 길이 제곱에 비례)

---

### Pre-LN Transformer 블록

**구조** (Residual + LayerNorm 순서)
```
Input
  ↓
[LayerNorm] → [Multi-Head Attention] → [+ Residual]
  ↓
[LayerNorm] → [Feed Forward Network] → [+ Residual]
  ↓
Output
```

**vs Post-LN (이전 방식)**
```
Post-LN (불안정):
Input → [MHA] → [+ Residual] → [LayerNorm]

Pre-LN (안정적) :
Input → [LayerNorm] → [MHA] → [+ Residual]
```

**장점**
- **학습 안정성**: LayerNorm이 먼저 정규화
- **깊은 네트워크**: Residual 스케일 폭주 방지
- **수렴 빠름**: 초기화 이후 더 빠른 수렴

**코드 위치**: [llm_block.c](Transformer/llm-c/src/llm_block.c)

---

### Feed Forward Network

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
- 중간 활성화 `[T × 4d_model]`를 1개 버퍼로 순차 재사용
- O(T·d_model·d_ff) 메모리로 제한

**코드 위치**: [llm_ffn.c](Transformer/llm-c/src/llm_ffn.c)

---

## 아키텍처 계층

### 계층 구조

```
┌────────────────────────────────────┐
│   입력 처리 (Input Layer)            │
│    - 토큰 ID (256개)                  │
│    - Token Embedding: [T × 64]      │
│    - Positional Encoding: [T × 64]  │
│    - 결과: [T × d_model]             │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│       Transformer 블록 (×3)         │
│    ┌──────────────────────┐        │
│    │ Pre-LN Block         │        │
│    │ ├─ LayerNorm         │        │
│    │ ├─ MHA (4 heads)     │        │
│    │ │  • Q/K/V 프로젝션    │        │
│    │ │  • Scaled attention│        │
│    │ │  • Head concat     │        │
│    │ ├─ Add & Norm       │         │
│    │ │                   │         │
│    │ ├─ LayerNorm        │         │
│    │ ├─ FFN (4x expand)  │         │
│    │ │  • Linear + ReLU  │         │
│    │ │  • Linear (축소)   │         │
│    │ └─ Add & Norm       │         │
│    └──────────────────────┘        │
│    Layer 1 → Layer 2 → Layer 3     │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│       출력 처리 (Output Layer)       │
│    - Final LayerNorm               │
│    - Linear (d_model → vocab_size) │
│    - Softmax → 확률분포              │
│    - Argmax/Sampling → 토큰         │
└────────────────────────────────────┘
```

### 하이퍼파라미터 (기본값)

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| `D_MODEL` | 64 | 임베딩 차원 |
| `N_HEAD` | 4 | 어텐션 헤드 수 |
| `D_K` | 16 | 헤드당 차원 (64÷4) |
| `D_FF` | 256 | FFN 중간층 차원 (4×64) |
| `N_LAYERS` | 3 | Transformer 블록 반복 |
| `SEQ_LEN` | 256 | 최대 시퀀스 길이 |
| `VOCAB_SIZE` | 256 | 어휘집 크기 (ASCII) |

---

## 현재 구현 상태 분석

### 완성된 부분

| 모듈 | 상태 | 설명 |
|------|------|------|
| **Tensor 기초** | ✅ | 행렬 연산, 활성화 함수 |
| **Self-Attention** | ✅ | Scaled Dot-Product + Multi-Head |
| **Feed Forward** | ✅ | 2-층 선형 + ReLU |
| **LayerNorm** | ✅ | Pre-normalization |
| **Positional Encoding** | ✅ | Learnable PE + scale |
| **Pre-LN 블록** | ✅ | 블록 조립 + Residual |
| **모델 추론** | ✅ | Forward pass + 외부 가중치 |
| **Causal Mask** | ✅ | Auto-regressive 생성 지원 |

---

## 보강 가능 영역

### Group Query Attention (GQA)

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

### KV Cache

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

### Flash Attention

- 블록 단위 Tiling으로 메모리 효율 극대화
- Score 행렬 [T×T]를 블록으로 나누어 순차 처리

```c
// 메모리 O(T²) → O(T)로 감소
void llm_flash_attention_forward(...);
```

### Quantization (INT8/INT4)

- Float32 → INT8/INT4로 가중치 압축
- 메모리: Float32(4B) → INT8(1B) 75% 감소

```c
typedef struct {
    int8_t data[SIZE];
    float scale;
    int8_t zero_point;
} QuantTensor;
```

### Batch 처리

- 여러 시퀀스를 동시에 처리하여 처리량 증가

```c
typedef struct {
    const float *inputs[BATCH_SIZE];
    float *outputs[BATCH_SIZE];
    int lengths[BATCH_SIZE];
} LlmBatch;
```

### SIMD 최적화

- ARM NEON (모바일), x86 AVX2 (CPU) 명령어 활용
- 행렬곱, softmax 등 핵심 연산 가속

```c
#ifdef __ARM_NEON__
void matmul_neon(const float *a, const float *b, float *c, int M, int N, int K);
#endif
```

### GPU 지원

- CUDA (NVIDIA), Metal (Apple Silicon) 백엔드 추가
- 대규모 행렬 연산 병렬화

```c
#ifdef ENABLE_CUDA
void matmul_cuda(const float *d_a, const float *d_b, float *d_c, int M, int N, int K);
#endif
```

### Advanced Sampling

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


## 참고 자료

### 논문
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- [RMSNorm vs LayerNorm](https://arxiv.org/abs/1910.07468)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ LLM 참고 구현

### 이미지
- [LLM.png](Transformer/LLM.png) - 블록 다이어그램
- [LLM Inference Pipeline Architecture Flow.png](LLM%20Inference%20Pipeline%20Architecture%20Flow.png) - 전체 파이프라인

