#include "jhq_index.h"
#include "erfinv.h"
#include "topk.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

JHQIndex::JHQIndex(int d, int M, int B, int Br, float alpha, int seed)
    : d_(d), M_(M), B_(B), Br_(Br), alpha_(alpha), ntotal_(0),
      Kr_(1 << Br), jl_(d, seed)
{
    if (d % M != 0)       throw std::invalid_argument("d must be divisible by M");
    if (B % (d / M) != 0) throw std::invalid_argument("B must be divisible by Ds=d/M");
    if (Br != 4 && Br != 8)
        throw std::invalid_argument("Br must be 4 or 8");
}

// ---------------------------------------------------------------------------
// Train
// ---------------------------------------------------------------------------

void JHQIndex::train(const float* x, int n_train) {
    // Primary codebook: same as JQ
    jl_.estimate_sigma(x, n_train);
    cb_ = std::make_unique<LloydMaxCodebook>(d_, M_, B_, jl_.sigma());

    // Rotate training vectors and compute residuals to estimate res_sigma_
    std::vector<float> y((size_t)n_train * d_);
    jl_.apply(x, y.data(), n_train);
    train_residual(y.data(), n_train);
}

void JHQIndex::train_residual(const float* y_rot, int n) {
    // Primary encode training vectors and collect per-dimension residuals
    std::vector<uint8_t> tmp_codes((size_t)n * M_);
    cb_->encode(y_rot, tmp_codes.data(), n);

    std::vector<float> residuals;
    residuals.reserve((size_t)n * d_);
    std::vector<float> yhat(d_);
    for (int i = 0; i < n; i++) {
        cb_->reconstruct(tmp_codes.data() + (size_t)i * M_, yhat.data());
        const float* yi = y_rot + (size_t)i * d_;
        for (int j = 0; j < d_; j++) residuals.push_back(yi[j] - yhat[j]);
    }
    // 1D k-means on actual residuals (paper §4.2).
    // The residual distribution is a mixture of truncated Gaussians — not a
    // pure Gaussian — so the analytical Lloyd-Max formula would be suboptimal.
    res_c1d_ = train_1d_kmeans(residuals.data(), (int)residuals.size(), Kr_);
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

void JHQIndex::add(const float* x, int n) {
    if (!cb_) throw std::runtime_error("call train() before add()");
    const int bpv   = (d_ * Br_ + 7) / 8;
    const int batch = 32768;  // cap per-batch memory at ~32K × d floats

    for (int start = 0; start < n; start += batch) {
        int nb = std::min(batch, n - start);

        std::vector<float>   y  ((size_t)nb * d_);
        std::vector<uint8_t> pc ((size_t)nb * M_);
        std::vector<uint8_t> rc ((size_t)nb * bpv, 0);
        std::vector<float>   cor((size_t)nb);

        jl_.apply(x + (size_t)start * d_, y.data(), nb);
        cb_->encode(y.data(), pc.data(), nb);

        std::vector<float> yhat(d_);
        for (int i = 0; i < nb; i++) {
            const float* yi = y.data()  + (size_t)i * d_;
            uint8_t*     ri = rc.data() + (size_t)i * bpv;
            cb_->reconstruct(pc.data() + (size_t)i * M_, yhat.data());

            // Encode residual dims + accumulate correction 2⟨ŷ, r̂⟩ in one pass.
            // Theorem 6 (fixed): correction = 2⟨ŷ,r̂⟩; the ||q||² query-constant
            // is dropped because it does not affect ranking.
            float dot = 0.0f;
            for (int j = 0; j < d_; j++) {
                int k = nearest_res_1d(yi[j] - yhat[j]);
                float rhat_j = res_c1d_[k];
                if (Br_ == 4) {
                    if (j % 2 == 0) ri[j/2]  =  (uint8_t)(k & 0x0F);
                    else             ri[j/2] |= (uint8_t)((k & 0x0F) << 4);
                } else {
                    ri[j] = (uint8_t)k;
                }
                dot += yhat[j] * rhat_j;
            }
            cor[i] = 2.0f * dot;
        }

        primary_codes_.insert(primary_codes_.end(), pc.begin(), pc.end());
        residual_codes_.insert(residual_codes_.end(), rc.begin(), rc.end());
        corrections_.insert(corrections_.end(), cor.begin(), cor.end());
        ntotal_ += nb;
    }
}

// ---------------------------------------------------------------------------
// Residual helpers
// ---------------------------------------------------------------------------
int JHQIndex::nearest_res_1d(float v) const {
    if (Kr_ == 1) return 0;
    int lo = 0, hi = Kr_ - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (v < 0.5f * (res_c1d_[mid] + res_c1d_[mid + 1]))
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

void JHQIndex::build_res_lut(const float* q_rot, float* lut_r) const {
    // lut_r[j * Kr_ + i] = (q_rot[j] − res_c1d_[i])²
    for (int j = 0; j < d_; j++) {
        float* row = lut_r + j * Kr_;
        float  qj  = q_rot[j];
        for (int i = 0; i < Kr_; i++) {
            float diff = qj - res_c1d_[i];
            row[i] = diff * diff;
        }
    }
}

float JHQIndex::composite_distance(int idx, float primary_dist,
                                    const float* lut_r) const {
    // d(q, ŷ+r̂)² = d_primary(q,ŷ)² + d_residual(q,r̂)² − ||q||² + correction
    // Since ||q||² is constant per query we skip it (doesn't affect ranking).
    // composite ≈ d_primary² + d_residual² + corrections_[idx]

    const int bytes_per_vec = (d_ * Br_ + 7) / 8;
    const uint8_t* rc = residual_codes_.data() + (size_t)idx * bytes_per_vec;

    float d_res = 0.0f;
    for (int j = 0; j < d_; j++) {
        int k;
        if (Br_ == 4) {
            k = (j % 2 == 0) ? (rc[j / 2] & 0x0F) : (rc[j / 2] >> 4);
        } else {
            k = rc[j];
        }
        d_res += lut_r[j * Kr_ + k];
    }
    return primary_dist + d_res + corrections_[idx];
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

void JHQIndex::search(const float* q, int nq, int k,
                       float* dists, int* labels) const {
    if (!cb_) throw std::runtime_error("index is empty");

    std::vector<float> q_rot((size_t)nq * d_);
    jl_.apply(q, q_rot.data(), nq);

    for (int i = 0; i < nq; i++) {
        search_one(q_rot.data() + (size_t)i * d_, k,
                   dists  + i * k,
                   labels + i * k);
    }
}

void JHQIndex::search_one(const float* q_rot, int k,
                           float* dists, int* labels) const {
    // --- Phase 1: coarse primary-code scan, collect top-αk candidates ---
    int ck = std::min((int)std::ceil(alpha_ * k), ntotal_);

    std::vector<float> lut((size_t)M_ * cb_->Ds() * cb_->K1D());
    cb_->build_lut(q_rot, lut.data());

    TopKHeap coarse(ck);
    for (int i = 0; i < ntotal_; i++) {
        float d = cb_->adc_distance(primary_codes_.data() + (size_t)i * M_, lut.data());
        coarse.push(d, i);
    }

    std::vector<float> cdists(ck);
    std::vector<int>   cids(ck);
    coarse.results(cdists.data(), cids.data());

    // --- Phase 2: residual refinement for αk candidates ---
    std::vector<float> lut_r((size_t)d_ * Kr_);
    build_res_lut(q_rot, lut_r.data());

    TopKHeap fine(k);
    for (int ci = 0; ci < ck; ci++) {
        int   idx = cids[ci];
        float d   = composite_distance(idx, cdists[ci], lut_r.data());
        fine.push(d, idx);
    }
    fine.results(dists, labels);
}
