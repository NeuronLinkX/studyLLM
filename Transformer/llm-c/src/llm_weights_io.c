// src/llm_weights_io.c
#include "llm_weights_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "LLMW"
#define VERSION_OK 1

static int read_u16(FILE *fp, unsigned short *v){ return fread(v, sizeof(*v), 1, fp) == 1; }
static int read_u32(FILE *fp, unsigned int *v){ return fread(v, sizeof(*v), 1, fp) == 1; }

static int map_tensor_slot(LlmWeights *w, const char *name, float ***out_slot){
    // name → 해당 필드의 "float* 를 가리키는 포인터" 반환(out_slot)
    if      (strcmp(name, "Wq") == 0)        *out_slot = &w->Wq;
    else if (strcmp(name, "Wk") == 0)        *out_slot = &w->Wk;
    else if (strcmp(name, "Wv") == 0)        *out_slot = &w->Wv;
    else if (strcmp(name, "Wo") == 0)        *out_slot = &w->Wo;
    else if (strcmp(name, "W1") == 0)        *out_slot = &w->W1;
    else if (strcmp(name, "b1") == 0)        *out_slot = &w->b1;
    else if (strcmp(name, "W2") == 0)        *out_slot = &w->W2;
    else if (strcmp(name, "b2") == 0)        *out_slot = &w->b2;
    else if (strcmp(name, "ln1_gamma") == 0) *out_slot = &w->ln1_gamma;
    else if (strcmp(name, "ln1_beta")  == 0) *out_slot = &w->ln1_beta;
    else if (strcmp(name, "ln2_gamma") == 0) *out_slot = &w->ln2_gamma;
    else if (strcmp(name, "ln2_beta")  == 0) *out_slot = &w->ln2_beta;
    else return 0;
    return 1;
}

int llm_weights_load(const char *path, LlmWeights *w){
    memset(w, 0, sizeof(*w));

    FILE *fp = fopen(path, "rb");
    if(!fp) return 0;

    // ---- 헤더 ----
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, MAGIC, 4) != 0){ fclose(fp); return 0; }

    unsigned int ver;
    if (!read_u32(fp, &ver) || ver != VERSION_OK){ fclose(fp); return 0; }

    unsigned int d_model, n_head, d_k, d_ff;
    if (!read_u32(fp, &d_model) || !read_u32(fp, &n_head) ||
        !read_u32(fp, &d_k)     || !read_u32(fp, &d_ff)) { fclose(fp); return 0; }

    w->d_model = (int)d_model;
    w->n_head  = (int)n_head;
    w->d_k     = (int)d_k;
    w->d_ff    = (int)d_ff;

    unsigned int n_tensors;
    if (!read_u32(fp, &n_tensors)){ fclose(fp); return 0; }

    // ---- 텐서들 ----
    for (unsigned int t = 0; t < n_tensors; ++t){
        // name
        unsigned short name_len;
        if (!read_u16(fp, &name_len)){ fclose(fp); return 0; }
        char nbuf[128];
        if (name_len >= sizeof(nbuf)) { fclose(fp); return 0; }
        if (fread(nbuf, 1, name_len, fp) != name_len){ fclose(fp); return 0; }
        nbuf[name_len] = '\0';

        // rank + shape
        unsigned short rank;
        if (!read_u16(fp, &rank) || rank > 8){ fclose(fp); return 0; }
        unsigned int shape[8];
        unsigned long long elems = 1;
        for (unsigned short r = 0; r < rank; ++r){
            if (!read_u32(fp, &shape[r])) { fclose(fp); return 0; }
            elems *= (unsigned long long)shape[r];
        }

        // 바이트 크기
        unsigned int byte_size;
        if (!read_u32(fp, &byte_size)){ fclose(fp); return 0; }
        if (byte_size % sizeof(float) != 0){ fclose(fp); return 0; }
        if (elems != (unsigned long long)(byte_size / sizeof(float))){ fclose(fp); return 0; }

        // 목적지 슬롯 찾기
        float **dst = NULL;
        if (!map_tensor_slot(w, nbuf, &dst)){
            // 알 수 없는 텐서: 건너뛰기
            fseek(fp, (long)byte_size, SEEK_CUR);
            continue;
        }

        // 메모리 확보 + 로드
        *dst = (float*)malloc(byte_size);
        if (!*dst){ fclose(fp); return 0; }
        if (fread(*dst, 1, byte_size, fp) != byte_size){ fclose(fp); return 0; }
    }

    fclose(fp);
    return 1;
}

void llm_weights_free(LlmWeights *w){
    if (!w) return;
    #define FREEM(p) do{ if((p)){ free((p)); (p)=NULL; } }while(0)
    FREEM(w->Wq); FREEM(w->Wk); FREEM(w->Wv); FREEM(w->Wo);
    FREEM(w->W1); FREEM(w->b1); FREEM(w->W2); FREEM(w->b2);
    FREEM(w->ln1_gamma); FREEM(w->ln1_beta);
    FREEM(w->ln2_gamma); FREEM(w->ln2_beta);
    #undef FREEM
}
