#include "jq_ivf_index.h"
#include "topk.h"
#include <stdexcept>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>

// AVX2 gather-based ADC: 8 subspaces per iteration via _mm256_i32gather_ps.
// Requires M divisible by 8.
static inline float adc_avx2(const uint8_t* __restrict__ code,
                               const float*   __restrict__ flat_lut,
                               int M) {
    __m256 acc = _mm256_setzero_ps();
    for (int m = 0; m < M; m += 8) {
        __m256i idx = _mm256_set_epi32(
            (m+7)*256 + code[m+7], (m+6)*256 + code[m+6],
            (m+5)*256 + code[m+5], (m+4)*256 + code[m+4],
            (m+3)*256 + code[m+3], (m+2)*256 + code[m+2],
            (m+1)*256 + code[m+1], (m+0)*256 + code[m+0]
        );
        acc = _mm256_add_ps(acc, _mm256_i32gather_ps(flat_lut, idx, 4));
    }
    // Horizontal sum
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 s   = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

JQIVFIndex::JQIVFIndex(int d, int M, int B, int nlist, int nprobe, int seed)
    : d_(d), M_(M), B_(B), nlist_(nlist), nprobe_(nprobe), ntotal_(0),
      jl_(d, seed),
      list_ids_(nlist), list_codes_(nlist)
{
    if (d % M != 0)       throw std::invalid_argument("d must be divisible by M");
    if (B % (d / M) != 0) throw std::invalid_argument("B must be divisible by Ds=d/M");
    if (nprobe > nlist)    nprobe_ = nlist;
}

void JQIVFIndex::train(const float* x, int n_train) {
    jl_.estimate_sigma(x, n_train);
    cb_ = std::make_unique<LloydMaxCodebook>(d_, M_, B_, jl_.sigma());

    std::vector<float> y((size_t)n_train * d_);
    jl_.apply(x, y.data(), n_train);

    ivf_ = std::make_unique<IVF>(d_, nlist_);
    ivf_->train(y.data(), n_train);
}

void JQIVFIndex::add(const float* x, int n) {
    if (!cb_ || !ivf_) throw std::runtime_error("先调用 train()");

    const int batch = 65536;
    for (int start = 0; start < n; start += batch) {
        int nb = std::min(batch, n - start);
        std::vector<float>   y      ((size_t)nb * d_);
        std::vector<int>     assigns((size_t)nb);
        std::vector<uint8_t> codes  ((size_t)nb * M_);

        jl_.apply(x + (size_t)start * d_, y.data(), nb);
        ivf_->assign(y.data(), nb, assigns.data());
        cb_->encode(y.data(), codes.data(), nb);

        for (int i = 0; i < nb; i++) {
            int c = assigns[i];
            list_ids_[c].push_back(ntotal_ + i);
            const uint8_t* ci = codes.data() + (size_t)i * M_;
            list_codes_[c].insert(list_codes_[c].end(), ci, ci + M_);
        }
        ntotal_ += nb;
    }
}

void JQIVFIndex::search(const float* q, int nq, int k,
                         float* dists, int* labels) const {
    if (!cb_ || !ivf_) throw std::runtime_error("索引为空");

    // 批量旋转所有查询向量（BLAS SGEMM，一次搞定）
    std::vector<float> q_rot((size_t)nq * d_);
    jl_.apply(q, q_rot.data(), nq);

    // OpenMP 并行化查询循环
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < nq; i++) {
        search_one(q_rot.data() + (size_t)i * d_, k,
                   dists  + i * k,
                   labels + i * k);
    }
}

void JQIVFIndex::search_one(const float* q_rot, int k,
                              float* dists, int* labels) const {
    std::vector<int> near_clusters(nprobe_);
    ivf_->find_nearest_centroids(q_rot, nprobe_, near_clusters.data());

    std::vector<float> flat_lut((size_t)M_ * 256);
    cb_->build_flat_lut(q_rot, flat_lut.data());

    TopKHeap heap(k);
    for (int ci = 0; ci < nprobe_; ci++) {
        int c = near_clusters[ci];
        const auto& ids   = list_ids_[c];
        const auto& codes = list_codes_[c];
        int sz = (int)ids.size();

#ifdef __AVX2__
        // M 是 8 的倍数时用 AVX2 gather（M=96,192,384 全满足）
        if (M_ % 8 == 0) {
            for (int j = 0; j < sz; j++) {
                float d = adc_avx2(codes.data() + (size_t)j * M_,
                                    flat_lut.data(), M_);
                heap.push(d, ids[j]);
            }
        } else
#endif
        {
            for (int j = 0; j < sz; j++) {
                const uint8_t* cj = codes.data() + (size_t)j * M_;
                float d = 0.0f;
                for (int m = 0; m < M_; m++)
                    d += flat_lut[(size_t)m * 256 + cj[m]];
                heap.push(d, ids[j]);
            }
        }
    }
    heap.results(dists, labels);
}
