#include "ivf.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

IVF::IVF(int d, int nlist) : d_(d), nlist_(nlist), centroids_(nlist * d) {}

void IVF::train(const float* y, int n, int max_iter, int max_train_n, int /*seed*/) {
    if (n < nlist_)
        throw std::invalid_argument("训练向量数不能少于 nlist");

    faiss::ClusteringParameters cp;
    cp.niter = max_iter;
    cp.max_points_per_centroid = (max_train_n + nlist_ - 1) / nlist_;
    cp.verbose = false;

    faiss::Clustering clus(d_, nlist_, cp);
    faiss::IndexFlatL2 index(d_);
    clus.train(n, y, index);

    std::memcpy(centroids_.data(), clus.centroids.data(),
                (size_t)nlist_ * d_ * sizeof(float));

    // 用训练好的质心建快速搜索索引
    flat_idx_ = std::make_unique<faiss::IndexFlatL2>(d_);
    flat_idx_->add(nlist_, centroids_.data());
}

void IVF::assign(const float* y, int n, int* assigns) const {
    std::vector<float>         dists((size_t)n);
    std::vector<faiss::idx_t>  ids((size_t)n);
    flat_idx_->search(n, y, 1, dists.data(), ids.data());
    for (int i = 0; i < n; i++) assigns[i] = (int)ids[i];
}

void IVF::find_nearest_centroids(const float* q, int nprobe, int* out) const {
    std::vector<float>        dists((size_t)nprobe);
    std::vector<faiss::idx_t> ids((size_t)nprobe);
    flat_idx_->search(1, q, nprobe, dists.data(), ids.data());
    for (int i = 0; i < nprobe; i++) out[i] = (int)ids[i];
}
