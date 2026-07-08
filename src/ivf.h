#pragma once
#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>

class IVF {
public:
    IVF(int d, int nlist);

    void train(const float* y, int n,
               int max_iter = 20, int max_train_n = 65536, int seed = 0);

    void assign(const float* y, int n, int* assigns) const;

    void find_nearest_centroids(const float* q, int nprobe, int* out) const;

    int nlist() const { return nlist_; }
    int d()     const { return d_; }

private:
    int d_, nlist_;
    std::vector<float> centroids_;
    std::unique_ptr<faiss::IndexFlatL2> flat_idx_;
};
