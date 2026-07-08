#pragma once
#include "codebook.h"
#include "ivf.h"
#include "jl_transform.h"
#include <memory>
#include <vector>

// JQ + IVF 索引
//
// 对应官方代码的 IndexIVFJHQ（单层）。
//
// 建索引：
//   train()  → 估计 σ → 建 JQ 码本 → k-means 训练 IVF
//   add()    → 旋转 → 分配到簇 → JQ 编码 → 写入倒排列表
//
// 查询：
//   旋转查询向量 → 找最近 nprobe 个质心 →
//   对这些簇里的向量做 ADC 扫描 → top-k
//
// nprobe 越大 recall 越高、QPS 越低，论文里 nprobe ∈ {1,2,4,8,16,32,64,128}
class JQIVFIndex {
public:
    JQIVFIndex(int d, int M, int B = 8,
               int nlist = 256, int nprobe = 32,
               int seed = 42);

    void train(const float* x, int n_train);
    void add(const float* x, int n);
    void search(const float* q, int nq, int k,
                float* dists, int* labels) const;

    void set_nprobe(int nprobe) { nprobe_ = nprobe; }
    int  ntotal() const { return ntotal_; }

private:
    int d_, M_, B_, nlist_, nprobe_, ntotal_;
    JLTransform jl_;
    std::unique_ptr<LloydMaxCodebook> cb_;
    std::unique_ptr<IVF> ivf_;

    // 倒排列表：per-cluster 存原始编号 + JQ 编码
    std::vector<std::vector<int>>     list_ids_;    // list_ids_[c]   → 原始 idx
    std::vector<std::vector<uint8_t>> list_codes_;  // list_codes_[c] → M 字节 codes

    void search_one(const float* q_rot, int k,
                    float* dists, int* labels) const;
};
