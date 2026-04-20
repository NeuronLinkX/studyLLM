# scripts/gen_weights.py
import struct, math, argparse, random
import numpy as np

MAGIC = b"LLMW"   # LLM Weights
VERSION = 1

# 순서/이름은 llm_weights_io.c와 일치해야 함
TENSORS = [
    # MHA (per-head Wq/Wk/Wv) : layout [H, D_MODEL, D_K]
    ("Wq", "H*D_MODEL*D_K"),
    ("Wk", "H*D_MODEL*D_K"),
    ("Wv", "H*D_MODEL*D_K"),
    # Wo : [D_MODEL, D_MODEL]
    ("Wo", "D_MODEL*D_MODEL"),
    # FFN
    ("W1", "D_MODEL*D_FF"),
    ("b1", "D_FF"),
    ("W2", "D_FF*D_MODEL"),
    ("b2", "D_MODEL"),
    # LN gammas/betas
    ("ln1_gamma", "D_MODEL"),
    ("ln1_beta",  "D_MODEL"),
    ("ln2_gamma", "D_MODEL"),
    ("ln2_beta",  "D_MODEL"),
]

def parse_shape(expr, H, D_MODEL, D_K, D_FF):
    # "H*D_MODEL*D_K" → [H, D_MODEL, D_K]
    dims = []
    for tok in expr.split("*"):
        tok = tok.strip()
        if tok == "H": dims.append(H)
        elif tok == "D_MODEL": dims.append(D_MODEL)
        elif tok == "D_K": dims.append(D_K)
        elif tok == "D_FF": dims.append(D_FF)
        else: dims.append(int(tok))
    return dims

def he_limit(fan_in):  # He-uniform
    return math.sqrt(6.0 / float(fan_in))

def xavier_limit(fan_in, fan_out):
    return math.sqrt(6.0 / float(fan_in + fan_out))

def init_weight(shape, kind, rng):
    # kind: "he" (ReLU/GeLU 계열) 또는 "xavier"
    size = 1
    for d in shape: size *= d
    if len(shape) >= 2:
        fan_in  = shape[-2]
        fan_out = shape[-1]
    else:
        fan_in = fan_out = shape[0]
    if kind == "he":
        a = he_limit(fan_in)
    else:
        a = xavier_limit(fan_in, fan_out)
    arr = rng.uniform(low=-a, high=a, size=size).astype(np.float32)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=6)
    ap.add_argument("--n_head",  type=int, default=2)
    ap.add_argument("--d_ff",    type=int, default=24)
    ap.add_argument("--seed",    type=int, default=1234)
    ap.add_argument("--act",     type=str, default="gelu",
        help="ffn act (relu/gelu) → 초기화 힌트")
    ap.add_argument("--out",     type=str, default="weights.bin")
    args = ap.parse_args()

    D_MODEL = args.d_model
    H       = args.n_head
    D_K     = D_MODEL // H
    D_FF    = args.d_ff

    rng = np.random.default_rng(args.seed)

    # 헤더: MAGIC(4), VERSION(u32), dims(u32 x4)
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<IIII", D_MODEL, H, D_K, D_FF))

        # 텐서 개수
        f.write(struct.pack("<I", len(TENSORS)))

        for name, expr in TENSORS:
            shape = parse_shape(expr, H, D_MODEL, D_K, D_FF)
            # 초기화 종류
            kind = "he" if args.act.lower() in ("relu","gelu") and name in ("W1","W2") else "xavier"
            if name.startswith("b") or name.endswith("beta"):
                arr = np.zeros((int(np.prod(shape)),), dtype=np.float32)
            elif name.endswith("gamma"):
                arr = np.ones((int(np.prod(shape)),), dtype=np.float32)
            else:
                arr = init_weight(shape, kind, rng)

            # 메타: 이름 길이(u16), 이름, rank(u16), shape(u32*rank), byte_size(u32)
            name_b = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<H", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<I", arr.nbytes))
            f.write(arr.tobytes())

    print(f"[OK] wrote {args.out}: d_model={D_MODEL}, n_head={H}, d_k={D_K}, d_ff={D_FF}")

if __name__ == "__main__":
    main()
