#include "io.h"
#include "io_mmap.h"
#include "jhq_ivf_index.h"
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

// Prints one "=== label ===" + nprobe/recall/qps table. sweep_cpu_ivf.py
// (JHQ_GPU/scripts) matches the "=== ... ===" marker to tag subsequent
// rows with this method name, so both tables land in one CSV.
static void run_sweep(JHQIVFIndex& idx, const char* label, int nlist,
                       const float* query, int nq, int k,
                       const int* gt, int dgt) {
    printf("\n=== %s ===\n", label);
    printf("%-10s %-12s %-10s\n", "nprobe", "Recall@10", "QPS");
    printf("%-10s %-12s %-10s\n", "------", "---------", "---");

    std::vector<float> dists((size_t)nq * k);
    std::vector<int>   ids((size_t)nq * k);

    for (int nprobe : {1, 2, 4, 8, 16, 32, 64, 128}) {
        if (nprobe > nlist) break;
        idx.set_nprobe(nprobe);

        auto t2 = std::chrono::high_resolution_clock::now();
        idx.search(query, nq, k, dists.data(), ids.data());
        auto t3 = std::chrono::high_resolution_clock::now();

        float recall = recall_at_k(ids.data(), gt, nq, k, dgt);
        double qps   = nq / std::chrono::duration<double>(t3 - t2).count();
        printf("%-10d %-12.4f %-10.1f\n", nprobe, recall, qps);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s base.fvecs query.fvecs gt.ivecs "
            "[M] [B] [Br] [nlist] [alpha] [k]\n"
            "  Builds one index, then sweeps nprobe in {1,2,4,8,16,32,64,128}\n"
            "  for two methods: JHQ-CPU-IVF (primary + residual refine) and\n"
            "  JQ-CPU-IVF (primary-only ablation, no residual stage)\n",
            argv[0]);
        return 1;
    }
    int nq, dq, ngt, dgt;
    // Base is by far the largest of the three (query/gt are a few thousand
    // rows at most) -- loaded via mmap, not read_fvecs(), so it isn't fully
    // materialized in host RAM. See io_mmap.h for why.
    MmapFloatMatrix base = load_fvecs_mmap(argv[1]);
    int nb = base.n;
    int d  = base.d;
    auto query = read_fvecs(argv[2], &nq,  &dq);
    auto gt    = read_ivecs(argv[3], &ngt, &dgt);

    int   B     = (argc > 5) ? atoi(argv[5])  : 8;
    int   M     = (argc > 4) ? atoi(argv[4])  : std::max(1, d / B);
    int   Br    = (argc > 6) ? atoi(argv[6])  : 4;
    int   nlist = (argc > 7) ? atoi(argv[7])  : 256;
    float alpha = (argc > 8) ? atof(argv[8])  : 4.0f;
    int   k     = (argc > 9) ? atoi(argv[9])  : 10;

    if (d % M != 0 || B % (d / M) != 0) {
        fprintf(stderr,
            "Invalid M=%d for d=%d B=%d: need M>=d/B=%d and d%%M==0\n",
            M, d, B, d / B);
        return 1;
    }
    printf("base=%d×%d  query=%d×%d  M=%d  B=%d  Br=%d  nlist=%d  alpha=%.1f  k=%d\n",
           nb, d, nq, dq, M, B, Br, nlist, alpha, k);

    JHQIVFIndex idx(d, M, B, Br, nlist, nlist, alpha);
    printf("Training...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    idx.train(base.data, nb);
    printf("Encoding %d vectors...\n", nb);
    idx.add(base.data, nb);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Index build: %.2f s\n",
           std::chrono::duration<double>(t1 - t0).count());

    run_sweep(idx, "JHQ-CPU-IVF", nlist, query.data(), nq, k, gt.data(), dgt);

    idx.set_primary_only(true);
    run_sweep(idx, "JQ-CPU-IVF", nlist, query.data(), nq, k, gt.data(), dgt);

    return 0;
}
