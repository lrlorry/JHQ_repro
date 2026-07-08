#include "io.h"
#include "jq_ivf_index.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float recall_at_k(const int* pred, const int* gt,
                          int nq, int k, int gt_k) {
    int hits = 0;
    for (int i = 0; i < nq; i++)
        for (int p = 0; p < k; p++)
            for (int g = 0; g < gt_k; g++)
                if (pred[i*k+p] == gt[i*gt_k+g]) { hits++; break; }
    return (float)hits / ((float)nq * k);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s base.fvecs query.fvecs gt.ivecs [M] [B] [nlist] [k]\n"
            "  sweeps nprobe ∈ {1,2,4,8,16,32,64,128} and prints recall@k + QPS\n",
            argv[0]);
        return 1;
    }
    int nb, d, nq, dq, ngt, dgt;
    auto base  = read_fvecs(argv[1], &nb,  &d);
    auto query = read_fvecs(argv[2], &nq,  &dq);
    auto gt    = read_ivecs(argv[3], &ngt, &dgt);

    // B 决定 bits_per_dim = B/Ds，必须满足 B >= Ds = d/M.
    // 默认 M = d/B（最小合法值），给出 Ds=B, bits_per_dim=1, K1D=2.
    int B     = (argc > 5) ? atoi(argv[5]) : 8;
    int M     = (argc > 4) ? atoi(argv[4]) : std::max(1, d / B);
    int nlist = (argc > 6) ? atoi(argv[6]) : 256;
    int k     = (argc > 7) ? atoi(argv[7]) : 10;

    if (d % M != 0 || B % (d / M) != 0) {
        fprintf(stderr,
            "Invalid M=%d for d=%d B=%d: need M>=d/B=%d and d%%M==0\n",
            M, d, B, d / B);
        return 1;
    }
    printf("base=%d×%d  query=%d×%d  M=%d  B=%d  nlist=%d  k=%d\n",
           nb, d, nq, dq, M, B, nlist, k);

    // Build once
    JQIVFIndex idx(d, M, B, nlist, nlist /*nprobe=nlist for build*/);
    printf("Training (σ + codebook + IVF k-means)...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    idx.train(base.data(), nb);
    printf("Encoding %d vectors...\n", nb);
    idx.add(base.data(), nb);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Index build: %.2f s\n",
           std::chrono::duration<double>(t1 - t0).count());

    // Sweep nprobe
    printf("\n%-10s %-12s %-10s\n", "nprobe", "Recall@10", "QPS");
    printf("%-10s %-12s %-10s\n", "------", "---------", "---");

    std::vector<float> dists((size_t)nq * k);
    std::vector<int>   ids((size_t)nq * k);

    for (int nprobe : {1, 2, 4, 8, 16, 32, 64, 128}) {
        if (nprobe > nlist) break;
        idx.set_nprobe(nprobe);

        auto t2 = std::chrono::high_resolution_clock::now();
        idx.search(query.data(), nq, k, dists.data(), ids.data());
        auto t3 = std::chrono::high_resolution_clock::now();

        float recall = recall_at_k(ids.data(), gt.data(), nq, k, dgt);
        double qps   = nq / std::chrono::duration<double>(t3 - t2).count();
        printf("%-10d %-12.4f %-10.1f\n", nprobe, recall, qps);
    }
    return 0;
}
