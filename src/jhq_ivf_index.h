#pragma once
#include "codebook.h"
#include "ivf.h"
#include "jl_transform.h"
#include <memory>
#include <vector>

// JHQ + IVF 索引 — 对应论文 Algorithm 1 的完整工业版本
//
// 三层结构：
//   IVF（最外层）：   k-means 分桶，查询时只进 nprobe 个簇
//   Primary（第一层）：JQ 主码，簇内 ADC 粗筛，取 top-αk 候选
//   Residual（第二层）：标量残差码，候选集精排，取最终 top-k
class JHQIVFIndex {
public:
    JHQIVFIndex(int d, int M, int B = 8, int Br = 4,
                int nlist = 256, int nprobe = 32,
                float alpha = 4.0f, int seed = 42);

    void train(const float* x, int n_train);
    void add(const float* x, int n);
    void search(const float* q, int nq, int k,
                float* dists, int* labels) const;

    void set_nprobe(int nprobe) { nprobe_ = nprobe; }
    int  ntotal() const { return ntotal_; }

    // true: 只用 Primary(JQ)码 ADC 距离出 top-k，跳过 Residual 精排阶段
    // (ablation baseline -- 对应论文里单独报告的 "JQ" 方法，而非完整 JHQ)
    void set_primary_only(bool v) { primary_only_ = v; }

private:
    int   d_, M_, B_, Br_, nlist_, nprobe_, ntotal_, Kr_;
    float alpha_;
    bool  primary_only_ = false;

    JLTransform jl_;
    std::unique_ptr<LloydMaxCodebook> cb_;
    std::unique_ptr<IVF> ivf_;

    float              res_sigma_;
    std::vector<float> res_c1d_;   // (Kr_,) 残差 1D 码字

    // 倒排列表：每簇存原始 idx、主码（M bytes）、残差码（ceil(d*Br/8) bytes）、correction
    struct Entry {
        int     orig_idx;
        float   correction;  // 2<ŷ,r̂> − ||r̂||²，Theorem 6 修正项
    };
    std::vector<std::vector<Entry>>   list_entries_;
    std::vector<std::vector<uint8_t>> list_primary_;   // M bytes/entry
    std::vector<std::vector<uint8_t>> list_residual_;  // bytes_per_vec bytes/entry

    int bytes_per_res_vec() const { return (d_ * Br_ + 7) / 8; }

    void train_residual(const float* y_rot, int n);
    int  nearest_res_1d(float v) const;
    void build_res_lut(const float* q_rot, float* lut_r) const;
    float composite_dist(int cluster, int pos_in_cluster,
                         float primary_dist, const float* lut_r) const;
    void search_one(const float* q_rot, int k,
                    float* dists, int* labels) const;
};
