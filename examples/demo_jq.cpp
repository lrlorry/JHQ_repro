#include "io.h"
#include "jq_index.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float recall_at_k(const int* pred, const int* gt,
                          int nq, int k, int gt_k) {
    int hits = 0;
    for (int i = 0; i < nq; i++) {
        for (int p = 0; p < k; p++) {
            for (int g = 0; g < gt_k; g++) {
                if (pred[i * k + p] == gt[i * gt_k + g]) { hits++; break; }
            }
        }
    }
    return (float)hits / ((float)nq * k);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s base.fvecs query.fvecs gt.ivecs [M] [B] [k]\n",
                argv[0]);
        return 1;
    }
    const char* base_path  = argv[1];
    const char* query_path = argv[2];
    const char* gt_path    = argv[3];
    int M = (argc > 4) ? atoi(argv[4]) : 96;
    int B = (argc > 5) ? atoi(argv[5]) : 8;
    int k = (argc > 6) ? atoi(argv[6]) : 10;

    int nb, d, nq, dq, ngt, dgt;
    auto base  = read_fvecs(base_path,  &nb,  &d);
    auto query = read_fvecs(query_path, &nq,  &dq);
    auto gt    = read_ivecs(gt_path,    &ngt, &dgt);

    printf("base=%d×%d  query=%d×%d  gt=%d×%d\n", nb, d, nq, dq, ngt, dgt);
    printf("M=%d  B=%d  Ds=%d  K_1D=%d\n", M, B, d/M, 1 << (B / (d/M)));

    JQIndex idx(d, M, B);

    // Build index
    auto t0 = std::chrono::high_resolution_clock::now();
    printf("Training (σ estimation + codebook)...\n");
    idx.train(base.data(), nb);
    printf("Encoding %d vectors...\n", nb);
    idx.add(base.data(), nb);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();
    printf("Index build: %.2f s\n", build_s);

    // Search
    std::vector<float> dists((size_t)nq * k);
    std::vector<int>   ids((size_t)nq * k);

    auto t2 = std::chrono::high_resolution_clock::now();
    idx.search(query.data(), nq, k, dists.data(), ids.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    double search_s = std::chrono::duration<double>(t3 - t2).count();

    float recall = recall_at_k(ids.data(), gt.data(), nq, k, dgt);
    printf("Recall@%d : %.4f\n", k, recall);
    printf("QPS      : %.1f\n",  nq / search_s);
    return 0;
}
