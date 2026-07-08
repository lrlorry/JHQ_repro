#pragma once
#include "codebook.h"
#include "jl_transform.h"
#include <memory>
#include <vector>

// JQ (JL-Enhanced Quantization) flat index — paper §3.
//
// Build pipeline:
//   train()  → estimate σ from sample vectors → build analytical codebook
//   add()    → rotate vectors (y = Π·x) → encode to M-byte primary codes
//
// Search pipeline:
//   rotate query → build LUT (M × Ds × K_1D floats) →
//   scan all codes with ADC → top-k heap
//
// Complexity:
//   Build: O(nd²) for rotation  +  O(nM) for encoding
//   Query: O(M·Ds·K_1D) for LUT + O(nM·Ds) for scan (each scan step = Ds table lookups)
class JQIndex {
public:
    // d      : vector dimension
    // M      : number of subspaces (Ds = d/M, keep Ds ≈ 8)
    // B      : bits per subspace (8 recommended, must satisfy B % Ds == 0)
    // seed   : RNG seed for the JL rotation matrix
    JQIndex(int d, int M, int B = 8, int seed = 42);

    // Estimate σ and finalize the Lloyd-Max codebook.
    // x: (n_train × d) row-major float array.
    void train(const float* x, int n_train);

    // Rotate and encode n vectors (n×d). Appends to the index.
    void add(const float* x, int n);

    // For each of nq queries (nq×d), find the k approximate nearest neighbours.
    // dists : (nq×k) output squared distances  (caller-allocated)
    // labels: (nq×k) output integer indices     (caller-allocated)
    void search(const float* q, int nq, int k, float* dists, int* labels) const;

    int ntotal() const { return ntotal_; }
    int dim()    const { return d_; }

private:
    int d_, M_, B_, ntotal_;
    JLTransform jl_;
    std::unique_ptr<LloydMaxCodebook> cb_;
    std::vector<uint8_t> codes_;  // (ntotal_ × M_) bytes

    void search_one(const float* q_rot, int k, float* dists, int* labels) const;
};
