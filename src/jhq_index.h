#pragma once
#include "codebook.h"
#include "jl_transform.h"
#include <memory>
#include <vector>

// JHQ (JL-Enhanced Hierarchical Quantization) index — paper §4.
//
// Two-level architecture on top of JQ:
//
//   Primary level  (fast, coarse):
//     Same as JQ — M-byte codes, ADC scan, O(nM) per query.
//
//   Residual level  (slow, accurate):
//     After primary quantisation, residual r = y − ŷ is still near-Gaussian.
//     Each dimension of r is quantised with Kr = 2^Br 1D Lloyd-Max codewords.
//     Stored as Br bits per dimension → d·Br bits per vector total.
//
//   Query (Algorithm 1 of the paper):
//     1. Primary ADC scan → top-αk candidates Z   (coarse, fast)
//     2. For each z ∈ Z: compute composite distance using residual codes   (fine)
//     3. Return top-k from Z
//
// Composite distance (Theorem 6, L2 case):
//   d(q, ŷ+r̂)² = d(q,ŷ)² + d(q,r̂)² − ||q||² + 2⟨ŷ, r̂⟩
// Since ||q||² is constant per query, ranking by
//   d_primary + d_residual + correction   with correction = 2⟨ŷ, r̂⟩
// is equivalent to ranking by true distance.
class JHQIndex {
public:
    // d     : vector dimension
    // M     : primary subspaces  (Ds = d/M, keep ≈ 8)
    // B     : bits per primary subspace (must satisfy B % Ds == 0)
    // Br    : bits per residual dimension (4 recommended)
    // alpha : candidate oversampling factor; candidate set size = ceil(alpha·k)
    // seed  : RNG seed
    JHQIndex(int d, int M, int B = 8, int Br = 4, float alpha = 4.0f, int seed = 42);

    void train(const float* x, int n_train);
    void add(const float* x, int n);
    void search(const float* q, int nq, int k, float* dists, int* labels) const;

    int ntotal() const { return ntotal_; }

private:
    int   d_, M_, B_, Br_, ntotal_;
    float alpha_;
    int   Kr_;            // residual codewords per dimension = 2^Br

    JLTransform jl_;
    std::unique_ptr<LloydMaxCodebook> cb_;   // primary codebook

    // Residual scalar codebook (1D, shared across all dimensions)
    float              res_sigma_;
    std::vector<float> res_c1d_;   // (Kr_,) ascending codewords

    // Primary codes:  (ntotal_ × M_) uint8_t
    std::vector<uint8_t> primary_codes_;

    // Residual codes: (ntotal_ × d_) uint8_t, each stores a Br-bit index
    // Packing: two 4-bit indices per byte when Br=4.
    // For Br=8: one byte per dimension.
    std::vector<uint8_t> residual_codes_;

    // Per-vector precomputed correction: 2⟨ŷ, r̂⟩  (Theorem 6, query-constant dropped)
    std::vector<float> corrections_;

    // --- helpers ---
    void train_residual(const float* y_rot, int n);
    int  nearest_res_1d(float v) const;

    // Build residual LUT: lut_r[j][i] = (q_rot[j] − res_c1d[i])²
    // lut_r must be d_ × Kr_ floats.
    void build_res_lut(const float* q_rot, float* lut_r) const;

    float composite_distance(int idx, float primary_dist,
                              const float* lut_r) const;

    void search_one(const float* q_rot, int k, float* dists, int* labels) const;
};
