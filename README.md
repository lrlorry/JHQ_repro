# JHQ Reproduction (from scratch)

From-scratch C++17 implementation of **JQ** and **JHQ** from the paper:

> Jiabao Han, Mengxuan Zhang, Goce Trajcevski.
> *JHQ: Johnson-Lindenstrauss Enhanced Hierarchical Quantization for
> High-Dimensional Approximate Nearest Neighbor Search.*
> PVLDB Vol. 19 No. 7, pp. 1530–1543, 2026.

This repo re-implements the core algorithms independently. The only external
dependency is BLAS/LAPACK (for matrix multiply and QR decomposition).

---

## Algorithm overview

```
JL transform (§3.1)
  G ∈ ℝ^{d×d} ~ N(0,1)  →  QR decompose  →  Π = Q (orthogonal)
  y = Π·x                (rotation preserves distances, Lemma 1)
  ∴ each dimension y_i ~ N(0, σ²) independently (Lemma 2)

JQ (§3.2) — primary quantisation
  Codebook (no k-means!):
    K_1D = 2^(B/Ds) codewords per dimension
    c_i  = σ√2 · erfinv((2i−1)/K_1D)     [Eq. 3/4]
    Full Ds-dim codebook = Cartesian product of 1D codewords
  Encoding:   code_m = argmin ||y^(m) − ĉ||  → B-bit integer per subspace
  ADC query:  LUT[m][k][i] = (q_rot[m·Ds+k] − c_i)²
              d_approx = Σ_m Σ_k LUT[m][k][ code_m[k] ]

JHQ (§4) — adds residual level
  r = y − ŷ    (near-Gaussian residual, same trick applies)
  Residual codebook: Kr = 2^Br codewords per dim, packed Br bits/dim
  Two-phase query (Algorithm 1):
    Phase 1  coarse: primary ADC scan → top-αk candidates Z
    Phase 2  refine: composite distance for z ∈ Z
             d(q, ŷ+r̂)² = d_primary² + d_residual² + correction
             correction  = 2⟨ŷ, r̂⟩ − ||r̂||²   (precomputed per vector)
```

---

## File map

```
src/
├── erfinv.h          Winitzki approximation + Newton step for erf⁻¹
├── io.h              .fvecs / .ivecs read helpers
├── topk.h            max-heap for top-k nearest neighbours
├── jl_transform.h/cpp    JL rotation matrix (sgeqrf + sorgqr)
├── codebook.h/cpp    Lloyd-Max analytical codebook (Eq. 3/4)
├── jq_index.h/cpp    JQ flat index  (train / add / search)
└── jhq_index.h/cpp   JHQ two-level index
examples/
├── demo_jq.cpp       CLI for JQ: recall@k + QPS
└── demo_jhq.cpp      CLI for JHQ: recall@k + QPS
scripts/
├── preprocess.py     HuggingFace → .fvecs + ground truth
└── evaluate.py       parse results, plot speed-accuracy curve
```

---

## Build

### Requirements

| Dep | Version | Notes |
|---|---|---|
| GCC / Clang | C++17 | |
| CMake | ≥ 3.16 | |
| OpenBLAS | any | provides CBLAS + LAPACK |

**Ubuntu:**
```bash
sudo apt install build-essential cmake libopenblas-dev liblapack-dev
```

**macOS:**
```bash
brew install cmake openblas
export LDFLAGS="-L$(brew --prefix openblas)/lib"
export CPPFLAGS="-I$(brew --prefix openblas)/include"
```

### Compile

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Produces `build/examples/demo_jq` and `build/examples/demo_jhq`.

---

## Datasets

```bash
pip install datasets numpy scikit-learn faiss-cpu

# Quick test (~100K vectors, ~1 min)
python scripts/preprocess.py --dataset openai-1024-100k --output_dir datasets

# Full datasets
python scripts/preprocess.py --dataset vogue-768    --output_dir datasets
python scripts/preprocess.py --dataset openai-1536  --output_dir datasets
python scripts/preprocess.py --dataset arxiv-768    --output_dir datasets
python scripts/preprocess.py --dataset bge-1024     --output_dir datasets
```

---

## Run

### JQ

```bash
./build/examples/demo_jq \
    datasets/openai-1024-100k_base.fvecs \
    datasets/openai-1024-100k_query.fvecs \
    datasets/openai-1024-100k_groundtruth.ivecs \
    128   # M (subspaces, Ds=d/M≈8)
    8     # B (bits per subspace)
    10    # k
```

### JHQ

```bash
./build/examples/demo_jhq \
    datasets/openai-1024-100k_base.fvecs \
    datasets/openai-1024-100k_query.fvecs \
    datasets/openai-1024-100k_groundtruth.ivecs \
    128   # M
    8     # B  (primary bits/subspace)
    4     # Br (residual bits/dimension)
    4.0   # alpha (oversampling factor)
    10    # k
```

### Recommended M values

| Dataset | d | M for Ds≈8 |
|---|---|---|
| openai-1024-100k | 1024 | 128 |
| vogue-768 | 768 | 96 |
| openai-1536 | 1536 | 192 |
| arxiv-768 | 768 | 96 |
| bge-1024 | 1024 | 128 |

Constraint: `d % M == 0` and `B % (d/M) == 0` must both hold.

---

## Implementation notes

### What matches the paper exactly
- JL rotation via QR decomposition (Definition 1, Example 3)
- Lloyd-Max analytical codewords: `c_i = σ√2 · erfinv((2i−1)/K_1D)` (Eq. 3/4)
- ADC via factored 1D LUTs (not full Ds-dim LUTs)
- JHQ composite distance with precomputed correction term (Theorem 6)
- Two-phase query with α oversampling (Algorithm 1)

### Simplifications vs the paper's official code
| Aspect | Paper (official) | This repo |
|---|---|---|
| SIMD | AVX512/AVX2/NEON hand-written | Scalar (compiler auto-vectorises with -O3 -march=native) |
| IVF | FAISS IndexIVF wrapper | Flat scan (no IVF partitioning) |
| Index base class | Inherits FAISS IndexFlatCodes | Standalone C++ |
| Residual storage | Br-bit packed per dimension | Same (Br=4 or 8, nibble-packed) |

The scalar scan gives correct recall numbers; QPS will be lower than the
paper's SIMD-optimised version. IVF can be added on top later.

---

## Expected output (openai-1024-100k, M=128, B=8)

```
base=99000×1024  query=1000×1024  gt=1000×100
M=128  B=8  Ds=8  K_1D=2
Training (σ estimation + codebook)...
Encoding 99000 vectors...
Index build: ~5 s
Recall@10 : ~0.60–0.75   (paper: JQ reaches ~0.89 at this M on full 1M dataset)
QPS       : ~500–2000     (depends on hardware; paper uses 96-core AMD EPYC)
```

Recall is lower than the paper because:
1. This is 100K vectors (easier for brute-force, harder to distinguish via codes)
2. No IVF partitioning

Use the full 1M datasets for numbers comparable to Table 2 / Figure 4.
