#include "jhq_ivf_index.h"
#include "codebook.h"
#include "topk.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
static inline float adc_avx2(const uint8_t* __restrict__ code,
                               const float*   __restrict__ flat_lut,
                               int M) {
    __m256 acc = _mm256_setzero_ps();
    for (int m = 0; m < M; m += 8) {
        __m256i idx = _mm256_set_epi32(
            (m+7)*256+code[m+7], (m+6)*256+code[m+6],
            (m+5)*256+code[m+5], (m+4)*256+code[m+4],
            (m+3)*256+code[m+3], (m+2)*256+code[m+2],
            (m+1)*256+code[m+1], (m+0)*256+code[m+0]
        );
        acc = _mm256_add_ps(acc, _mm256_i32gather_ps(flat_lut, idx, 4));
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_hadd_ps(_mm_add_ps(lo, hi), _mm_setzero_ps());
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

JHQIVFIndex::JHQIVFIndex(int d, int M, int B, int Br,
                           int nlist, int nprobe, float alpha, int seed)
    : d_(d), M_(M), B_(B), Br_(Br), nlist_(nlist), nprobe_(nprobe),
      ntotal_(0), Kr_(1 << Br), alpha_(alpha),
      jl_(d, seed),
      list_entries_(nlist), list_primary_(nlist), list_residual_(nlist)
{
    if (d % M != 0)        throw std::invalid_argument("d must be divisible by M");
    if (B % (d / M) != 0)  throw std::invalid_argument("B must be divisible by Ds=d/M");
    if (Br != 4 && Br != 8) throw std::invalid_argument("Br must be 4 or 8");
    if (nprobe > nlist) nprobe_ = nlist;
}

// ---------------------------------------------------------------------------
// Train
// ---------------------------------------------------------------------------

void JHQIVFIndex::train(const float* x, int n_train) {
    jl_.estimate_sigma(x, n_train);
    cb_ = std::make_unique<LloydMaxCodebook>(d_, M_, B_, jl_.sigma());

    // 随机抽样 max_train_n 条，避免取连续前 N 条带来的分布偏差
    const int max_train_n = 65536;
    int n_use = std::min(n_train, max_train_n);

    std::vector<float> y((size_t)n_use * d_);
    if (n_use < n_train) {
        // 随机洗牌后取前 n_use 条
        std::vector<int> idx(n_train);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(42);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<float> x_sub((size_t)n_use * d_);
        for (int i = 0; i < n_use; i++)
            std::memcpy(x_sub.data() + (size_t)i * d_,
                        x + (size_t)idx[i] * d_, d_ * sizeof(float));
        jl_.apply(x_sub.data(), y.data(), n_use);
    } else {
        jl_.apply(x, y.data(), n_use);
    }

    ivf_ = std::make_unique<IVF>(d_, nlist_);
    ivf_->train(y.data(), n_use);

    train_residual(y.data(), n_use);
}

void JHQIVFIndex::train_residual(const float* y_rot, int n) {
    std::vector<uint8_t> codes((size_t)n * M_);
    cb_->encode(y_rot, codes.data(), n);

    // 每个训练向量的残差标量：n × d_ 个值
    std::vector<float> residuals((size_t)n * d_);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        std::vector<float> yhat(d_);
        cb_->reconstruct(codes.data() + (size_t)i * M_, yhat.data());
        const float* yi = y_rot + (size_t)i * d_;
        float* ri = residuals.data() + (size_t)i * d_;
        for (int j = 0; j < d_; j++) ri[j] = yi[j] - yhat[j];
    }
    res_c1d_ = train_1d_kmeans(residuals.data(), (int)residuals.size(), Kr_);
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

void JHQIVFIndex::add(const float* x, int n) {
    if (!cb_ || !ivf_) throw std::runtime_error("先调用 train()");
    const int bpv   = bytes_per_res_vec();
    const int batch = 32768;

    for (int start = 0; start < n; start += batch) {
        int nb = std::min(batch, n - start);

        std::vector<float>   y      ((size_t)nb * d_);
        std::vector<int>     assigns((size_t)nb);
        std::vector<uint8_t> primary((size_t)nb * M_);
        std::vector<uint8_t> residual((size_t)nb * bpv, 0);
        std::vector<float>   corr   ((size_t)nb);

        jl_.apply(x + (size_t)start * d_, y.data(), nb);
        ivf_->assign(y.data(), nb, assigns.data());
        cb_->encode(y.data(), primary.data(), nb);

        // 并行计算每向量的残差编码和修正项
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nb; i++) {
            const float* yi = y.data()       + (size_t)i * d_;
            uint8_t*     ri = residual.data() + (size_t)i * bpv;
            std::vector<float> yhat(d_);
            cb_->reconstruct(primary.data() + (size_t)i * M_, yhat.data());

            float dot = 0.0f;
            for (int j = 0; j < d_; j++) {
                int   k      = nearest_res_1d(yi[j] - yhat[j]);
                float rhat_j = res_c1d_[k];
                if (Br_ == 4) {
                    if (j % 2 == 0) ri[j/2]  =  (uint8_t)(k & 0x0F);
                    else             ri[j/2] |= (uint8_t)((k & 0x0F) << 4);
                } else {
                    ri[j] = (uint8_t)k;
                }
                dot += yhat[j] * rhat_j;
            }
            corr[i] = 2.0f * dot;
        }

        for (int i = 0; i < nb; i++) {
            int c = assigns[i];
            list_entries_[c].push_back({ntotal_ + i, corr[i]});

            const uint8_t* pi = primary.data()  + (size_t)i * M_;
            list_primary_[c].insert(list_primary_[c].end(), pi, pi + M_);

            const uint8_t* ri = residual.data() + (size_t)i * bpv;
            list_residual_[c].insert(list_residual_[c].end(), ri, ri + bpv);
        }
        ntotal_ += nb;
    }
}

// ---------------------------------------------------------------------------
// 残差辅助
// ---------------------------------------------------------------------------

int JHQIVFIndex::nearest_res_1d(float v) const {
    if (Kr_ == 1) return 0;
    int lo = 0, hi = Kr_ - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (v < 0.5f * (res_c1d_[mid] + res_c1d_[mid + 1])) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

void JHQIVFIndex::build_res_lut(const float* q_rot, float* lut_r) const {
    for (int j = 0; j < d_; j++) {
        float* row = lut_r + j * Kr_;
        float  qj  = q_rot[j];
        for (int i = 0; i < Kr_; i++) {
            float diff = qj - res_c1d_[i];
            row[i] = diff * diff;
        }
    }
}

float JHQIVFIndex::composite_dist(int cluster, int pos,
                                    float primary_dist,
                                    const float* lut_r) const {
    const int bpv = bytes_per_res_vec();
    const uint8_t* rc = list_residual_[cluster].data() + (size_t)pos * bpv;

    float d_res = 0.0f;
    for (int j = 0; j < d_; j++) {
        int k = (Br_ == 4)
                ? ((j % 2 == 0) ? (rc[j/2] & 0x0F) : (rc[j/2] >> 4))
                : rc[j];
        d_res += lut_r[j * Kr_ + k];
    }
    return primary_dist + d_res + list_entries_[cluster][pos].correction;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

void JHQIVFIndex::search(const float* q, int nq, int k,
                          float* dists, int* labels) const {
    if (!cb_ || !ivf_) throw std::runtime_error("索引为空");

    std::vector<float> q_rot((size_t)nq * d_);
    jl_.apply(q, q_rot.data(), nq);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < nq; i++)
        search_one(q_rot.data() + (size_t)i * d_, k,
                   dists + i*k, labels + i*k);
}

void JHQIVFIndex::search_one(const float* q_rot, int k,
                               float* dists, int* labels) const {
    // 找最近 nprobe 个质心
    std::vector<int> clusters(nprobe_);
    ivf_->find_nearest_centroids(q_rot, nprobe_, clusters.data());

    // 建 flat LUT：M × 256，每个 code byte 直接 1 次 lookup
    std::vector<float> flat_lut((size_t)M_ * 256);
    cb_->build_flat_lut(q_rot, flat_lut.data());

    // 统计这 nprobe 个簇里总向量数，确定候选集大小
    // primary_only_ 模式（JQ ablation）没有 phase 2 精排，候选池就是最终
    // top-k 本身，不需要像 JHQ 那样多取 alpha*k 条给精排阶段留余量
    int total_cands = 0;
    for (int c : clusters) total_cands += (int)list_entries_[c].size();
    int ck = primary_only_
                 ? std::min(k, total_cands)
                 : std::min((int)std::ceil(alpha_ * k), total_cands);

    // Phase 1: 主码 ADC 粗筛 → top-αk 候选
    // 候选记录 (primary_dist, cluster, pos_in_cluster)
    using Cand = std::tuple<float, int, int>;
    std::vector<Cand> all_cands;
    all_cands.reserve(total_cands);

    for (int ci = 0; ci < nprobe_; ci++) {
        int c = clusters[ci];
        const auto& prim = list_primary_[c];
        int sz = (int)list_entries_[c].size();
        for (int j = 0; j < sz; j++) {
            const uint8_t* pj = prim.data() + (size_t)j * M_;
            float d;
#ifdef __AVX2__
            d = (M_ % 8 == 0) ? adc_avx2(pj, flat_lut.data(), M_) : [&](){
                float s = 0.f;
                for (int m = 0; m < M_; m++) s += flat_lut[(size_t)m*256+pj[m]];
                return s;
            }();
#else
            d = 0.0f;
            for (int m = 0; m < M_; m++) d += flat_lut[(size_t)m*256+pj[m]];
#endif
            all_cands.emplace_back(d, c, j);
        }
    }
    // 取 top-αk（partial_sort）
    if ((int)all_cands.size() > ck) {
        std::partial_sort(all_cands.begin(), all_cands.begin() + ck, all_cands.end(),
                          [](const Cand& a, const Cand& b){
                              return std::get<0>(a) < std::get<0>(b);
                          });
        all_cands.resize(ck);
    }

    if (primary_only_) {
        // all_cands[0, take) 已经是按 primary_dist 升序排好的最终结果
        // (partial_sort 分支排过；未触发该分支时 all_cands.size()==ck<=k，
        // 全部返回，顺序不影响 recall@k 的集合命中判定)
        int take = std::min(k, (int)all_cands.size());
        for (int i = 0; i < take; i++) {
            dists[i]  = std::get<0>(all_cands[i]);
            labels[i] = list_entries_[std::get<1>(all_cands[i])]
                            [std::get<2>(all_cands[i])].orig_idx;
        }
        for (int i = take; i < k; i++) { dists[i] = 1e30f; labels[i] = -1; }
        return;
    }

    // Phase 2: 残差精排
    std::vector<float> lut_r((size_t)d_ * Kr_);
    build_res_lut(q_rot, lut_r.data());

    TopKHeap heap(k);
    for (auto& [pd, c, pos] : all_cands) {
        float d = composite_dist(c, pos, pd, lut_r.data());
        heap.push(d, list_entries_[c][pos].orig_idx);
    }
    heap.results(dists, labels);
}
