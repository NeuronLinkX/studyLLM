# Transformer/llm-c   

> StudyLLM   : Self-Attention, Pre-LN ,  

---

##  

1. [Attention  ](#1-attention--)
2. [Pre-LN  ](#2-pre-ln--)
3. [  ](#3---)
4. [ ](#4--)
5. [ ](#5--)

---

## 1. Attention  

### 1.1 Scaled Dot-Product Attention 

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 1.2   (llm_attention.c)

####  1:  (Q, K, V )

```c
// : X [T × d_model]
// : Q, K, V [N_HEAD × T × d_k]

for(size_t h=0; h<N_HEAD; ++h){
    for(size_t t=0; t<T; ++t){
        const float *xrow = &X[t*D_MODEL];
        
        //   
        for(size_t j=0; j<D_K; ++j){
            float q=0.f, k=0.f, v=0.f;
            
            // Q = X @ W_q, K = X @ W_k, V = X @ W_v
            for(size_t i=0; i<D_MODEL; ++i){
                q += xrow[i] * w->Wq[h][i][j];
                k += xrow[i] * w->Wk[h][i][j];
                v += xrow[i] * w->Wv[h][i][j];
            }
            
            Q[h][t][j] = q;
            K[h][t][j] = k;
            V[h][t][j] = v;
        }
    }
}
```

** **:
```
W_q: [N_HEAD × D_MODEL × D_K]
    = [4 × 64 × 16] = 4,096 floats

  Q/K/V:
    [T × D_K] = [256 × 16] = 4,096 floats
```

####  2:   

```c
// score[t][s] = Q[h][t] · K[h][s] / sqrt(d_k)
float score[SEQ_LEN][SEQ_LEN];

for(size_t t=0; t<T; ++t){
    for(size_t s=0; s<T; ++s){
        // : Q[t] · K[s]
        float val = dotf(Q[h][t], K[h][s], D_K) * scale;
        
        // Causal Mask:    (Auto-regressive)
#if USE_CAUSAL_MASK
        if(s > t) val = -1e30f;  //  
#endif
        
        score[t][s] = val;
    }
}
```

****: $T^2 = 256^2 = 65,536$ (  !)  
****:     → **Flash Attention **

####  3: Softmax + 

```c
//  softmax (   max )
for(size_t t=0; t<T; ++t) 
    llm_softmax_row(score[t], T);

// : output[t] = sum_s( score[t][s] * V[s] )
for(size_t t=0; t<T; ++t){
    for(size_t r=0; r<D_K; ++r){
        float sum = 0.f;
        for(size_t s=0; s<T; ++s)
            sum += score[t][s] * V[h][s][r];
        heads[h][t][r] = sum;
    }
}
```

####  4: Head Concat +  

```c
//    : [d_model] = concat(head0, head1, head2, head3)
for(size_t t=0; t<T; ++t){
    float concat[D_MODEL];
    for(size_t h=0; h<N_HEAD; ++h)
        for(size_t r=0; r<D_K; ++r)
            concat[h*D_K + r] = heads[h][t][r];
    
    //  : Y[t] = concat @ W_o
    for(size_t j=0; j<D_MODEL; ++j){
        float s=0.f;
        for(size_t i=0; i<D_MODEL; ++i)
            s += concat[i] * w->Wo[i][j];
        Y[t*D_MODEL + j] = s;
    }
}
```

### 1.3 Attention 

|  |  |
|------|------|
| **** |  Self-Attention (  ) |
| **** | $O(T^2 \cdot d_k)$ - ** **  |
| **** | $O(T^2)$ -    |
| **** |   /  |
| **** | KV Cache (), Flash Attention () |

---

## 2. Pre-LN  

### 2.1 Pre-LN vs Post-LN

#### Post-LN (  - )
```
Input → MHA → Add → LayerNorm → FFN → Add → LayerNorm → Output
```

****:
- LayerNorm Residual  , Residual  Norm  
-    (vanishing/exploding gradient)

#### Pre-LN (  - ) 
```
Input → LayerNorm → MHA → Add 
              ↓                       
              → LayerNorm → FFN → Add → Output
```

****:
- LayerNorm   → MHA/FFN  
-    
-     

### 2.2   (llm_block.c)

```c
typedef struct {
    // LayerNorm 
    float ln1_weight[D_MODEL];  // γ (gamma)
    float ln1_bias[D_MODEL];    // β (beta)
    float ln2_weight[D_MODEL];
    float ln2_bias[D_MODEL];
    
    // MHA 
    float Wq[N_HEAD][D_MODEL][D_K];
    float Wk[N_HEAD][D_MODEL][D_K];
    float Wv[N_HEAD][D_MODEL][D_K];
    float Wo[D_MODEL][D_MODEL];
    
    // FFN 
    float W_ff1[D_MODEL][D_FF];
    float W_ff2[D_FF][D_MODEL];
} TransformerBlock;
```

### 2.3   (Forward Pass)

```c
void llm_block_forward(
    const float *X,                //  [T × d_model]
    float *Y,                      //  [T × d_model]
    size_t T,
    const TransformerBlock *block
)
{
    float ln1_out[SEQ_LEN][D_MODEL];  // LayerNorm1 
    float attn_out[SEQ_LEN][D_MODEL]; // Attention 
    float residual[SEQ_LEN][D_MODEL]; // X + Attention
    
    float ln2_out[SEQ_LEN][D_MODEL];  // LayerNorm2 
    float ffn_out[SEQ_LEN][D_MODEL];  // FFN 
    
    // 
    // 1⃣ Pre-LN + MHA
    // 
    llm_layernorm(
        X, ln1_out, T,
        block->ln1_weight, block->ln1_bias
    );
    
    // Multi-Head Attention
    llm_mha_forward(ln1_out, attn_out, T, &block->mha_w);
    
    // Residual: Y = X + Attention(LN(X))
    for(size_t i=0; i<T*D_MODEL; ++i)
        residual[i/D_MODEL][i%D_MODEL] = 
            X[i/D_MODEL][i%D_MODEL] + attn_out[i/D_MODEL][i%D_MODEL];
    
    // 
    // 2⃣ Pre-LN + FFN
    // 
    llm_layernorm(
        residual, ln2_out, T,
        block->ln2_weight, block->ln2_bias
    );
    
    // Feed Forward Network
    llm_ffn_forward(ln2_out, ffn_out, T, &block->ffn_w);
    
    // Final Residual: Output = Residual + FFN(LN(Residual))
    for(size_t i=0; i<T*D_MODEL; ++i)
        Y[i/D_MODEL][i%D_MODEL] = 
            residual[i/D_MODEL][i%D_MODEL] + ffn_out[i/D_MODEL][i%D_MODEL];
}
```

---

## 3.   

### 3.1   

```
 X:                [256 × 64] = 16,384 floats
Token Embedding:       [256 × 64] = 16,384
Positional Encoding:   [256 × 64] = 16,384

Transformer  (×3):
  - Q/K/V ():    [256 × 16] × 3  = 12,288
  - Score :        [256 × 256] = 65,536 (! )
  - Head concat:       [256 × 64] = 16,384
  - FFN :          [256 × 256] = 65,536
  
:                  [256 × 64] = 16,384
```

** **: ~200KB ( Score  FFN )

### 3.2   

```c
//   :    
float score[SEQ_LEN][SEQ_LEN];  // 256×256 
float attn_out[SEQ_LEN][D_MODEL]; // 256×64 
float ffn_hidden[SEQ_LEN][D_FF]; // 256×256 

//   :  
float *workspace = malloc(MAX_BUFFER_SIZE);

// Score  (workspace  )
compute_attention_scores(Q, K, workspace);

// Softmax (  in-place)
softmax_inplace(workspace, T*T);

//   ( workspace )
apply_values(workspace, V, output);
```

---

## 4.  

### 4.1 Softmax 

```c
//  : Exp overflow
float softmax_bad(float x) {
    return exp(x) / sum(exp(all));  // x  overflow!
}

//  : Max 
void softmax_stable(float *x, int n) {
    // 1⃣  
    float max_val = x[0];
    for(int i=1; i<n; ++i)
        if(x[i] > max_val) max_val = x[i];
    
    // 2⃣ Max   exp ( !)
    float sum = 0.f;
    for(int i=0; i<n; ++i) {
        x[i] = exp(x[i] - max_val);  // ←   
        sum += x[i];
    }
    
    // 3⃣ 
    for(int i=0; i<n; ++i)
        x[i] /= sum;
}
```

****:
```
: [1000.1, 1000.0, 999.9]

 : exp(1000.1) = ∞ (overflow)
 : 
  max = 1000.1
  after: [exp(0), exp(-0.1), exp(-0.2)]
       = [1.0, 0.905, 0.819]
  normalized: [0.373, 0.337, 0.290]
```

### 4.2 LayerNorm 

```c
void layernorm_stable(
    const float *x, float *y,
    const float *weight, const float *bias,
    int n
)
{
    // 1⃣  
    float mean = 0.f;
    for(int i=0; i<n; ++i) mean += x[i];
    mean /= n;
    
    // 2⃣  
    float var = 0.f;
    for(int i=0; i<n; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= n;
    
    // 3⃣  (epsilon  0 )
    const float eps = 1e-5f;
    for(int i=0; i<n; ++i) {
        y[i] = (x[i] - mean) / sqrtf(var + eps);
        
        // 4⃣  & 
        y[i] = weight[i] * y[i] + bias[i];
    }
}
```

**epsilon **:
```
var = 0.0001, eps = 1e-5
√(0.0001 + 1e-5) = √0.00011 ≈ 0.0105 ()
vs
√(0.0001) = 0.01 ( -  )
```

---

## 5.  

### Phase 1:   ( )

- [x] Scaled Dot-Product Attention
- [x] Multi-Head Attention (4 )
- [x] Pre-LN Transformer 
- [x] FFN (2- )
- [x] LayerNorm + Softmax 
- [x] Causal Mask 

### Phase 2:    

```c
// Phase 2-1: KV Cache (→ 4  )
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

// Phase 2-2: Batch 
typedef struct {
    const float *inputs[BATCH_SIZE];
    int lengths[BATCH_SIZE];
    float *outputs[BATCH_SIZE];
} LlmBatch;

void llm_model_forward_batch(const LlmBatch *batch, LlmModel *model);
```

** **:
- KV Cache:   **4** 
- Batch:   

### Phase 3:    

```c
// Phase 3-1: Group Query Attention ( )
// K/V 4   1-2 
void llm_gqa_forward(...);  //  50% 

// Phase 3-2:  (INT8/INT4)
// Float32 → Int8:  75% 
void matmul_int8(QuantTensor *a, QuantTensor *b, float *out);

// Phase 3-3: Flash Attention ( )
// Tiling +    → O(T) 
void llm_flash_attention_forward(...);
```

### Phase 4:     

```c
// Phase 4-1: SIMD (ARM NEON, x86 AVX2)
void matmul_neon(...);  // 2-4 
void matmul_avx2(...);

// Phase 4-2: GPU (CUDA/Metal)
void matmul_cuda(...);   // 10+ 
void matmul_metal(...);  // Apple Silicon 
```

---

##    ()

|  |   |   |  |
|---------|-----------|---------|--------|
| **** | - | 1x | - |
| **KV Cache** | 20% | 4x |  |
| **GQA** | 50% | 1.2x |  |
| **INT8 ** | 75% | 0.8x* |  |
| **Flash Attention** | 90% | 2x |  |
| **SIMD** | - | 3x |  |
| **GPU** | - | 10x+ |  |

*   , -    .

---

**: [README.md](README.md)   !** 
