#include "jq_index.h"
#include "topk.h"
#include <stdexcept>
#include <vector>

JQIndex::JQIndex(int d, int M, int B, int seed)
    : d_(d), M_(M), B_(B), ntotal_(0), jl_(d, seed)
{
    if (d % M != 0)
        throw std::invalid_argument("d must be divisible by M");
    if (B % (d / M) != 0)
        throw std::invalid_argument("B must be divisible by Ds = d/M");
}

void JQIndex::train(const float* x, int n_train) {
    jl_.estimate_sigma(x, n_train);
    cb_ = std::make_unique<LloydMaxCodebook>(d_, M_, B_, jl_.sigma());
}

void JQIndex::add(const float* x, int n) {
    if (!cb_) throw std::runtime_error("call train() before add()");

    const int batch = 65536;
    for (int start = 0; start < n; start += batch) {
        int nb = std::min(batch, n - start);
        std::vector<float> y((size_t)nb * d_);
        jl_.apply(x + (size_t)start * d_, y.data(), nb);

        const size_t off = (size_t)ntotal_ * M_;
        codes_.resize(off + (size_t)nb * M_);
        cb_->encode(y.data(), codes_.data() + off, nb);
        ntotal_ += nb;
    }
}

void JQIndex::search(const float* q, int nq, int k,
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

void JQIndex::search_one(const float* q_rot, int k,
                          float* dists, int* labels) const {
    // Build LUT: M × Ds × K_1D floats
    const int lut_size = M_ * cb_->Ds() * cb_->K1D();
    std::vector<float> lut(lut_size);
    cb_->build_lut(q_rot, lut.data());

    TopKHeap heap(k);
    const uint8_t* codes = codes_.data();
    for (int i = 0; i < ntotal_; i++) {
        float d = cb_->adc_distance(codes + (size_t)i * M_, lut.data());
        heap.push(d, i);
    }
    heap.results(dists, labels);
}
