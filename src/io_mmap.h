#pragma once
// mmap-backed .fvecs loader for the JHQ CPU demos (JHQ_repro).
//
// read_fvecs() (io.h) loads the whole base set into a heap
// std::vector<float> -- fine for small datasets, but this project hit a
// container with a 2GB cgroup memory.max, and the biggest JHQ paper
// datasets (bge-m3 ~41GB, stella-trec24 ~73GB) exceed that before any
// index code even runs. The equivalent problem was already found and
// fixed on the GPU side (JHQ_GPU/common/fvecs_mmap_io.cuh); this is the
// same fix for JHQ_repro.
//
// Unlike the GPU port's add() (which needed a full n*d contiguous host
// buffer to hand to a single monolithic rotate+encode pass), the CPU
// IndexIVFJHQ::add_core() (JHQ_official/jhq/jhqlib/impl/IndexIVFJHQTrain.cpp)
// already self-batches internally in fixed 65536-row chunks and inserts
// into growable per-list storage (InvertedLists::add_entry) -- it does
// NOT need global visibility of the whole dataset up front the way the
// GPU port's single static transposed-array layout does. So on the CPU
// side, fixing this one loading step (host RAM) should be enough on its
// own -- there's no equivalent GPU-VRAM-sized blocker left after this.
//
// load_fvecs_mmap() streams the source .fvecs into a header-free packed
// float32 sibling file (<path>.raw_f32, bounded host buffer, a few
// thousand rows at a time, cached across runs), then mmap()s it PROT_READ
// and returns a pointer to n*d contiguous floats backed by the OS page
// cache (demand-paged from disk, reclaimable under memory pressure)
// instead of a committed allocation sized to the whole dataset.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct MmapFloatMatrix {
    const float* data = nullptr;
    int n = 0, d = 0;
    void* map_base = nullptr;
    size_t map_bytes = 0;
    int fd = -1;

    MmapFloatMatrix() = default;
    MmapFloatMatrix(const MmapFloatMatrix&) = delete;
    MmapFloatMatrix& operator=(const MmapFloatMatrix&) = delete;
    MmapFloatMatrix(MmapFloatMatrix&& o) noexcept { *this = std::move(o); }
    MmapFloatMatrix& operator=(MmapFloatMatrix&& o) noexcept {
        if (this != &o) {
            reset();
            data = o.data; n = o.n; d = o.d;
            map_base = o.map_base; map_bytes = o.map_bytes; fd = o.fd;
            o.data = nullptr; o.map_base = nullptr; o.map_bytes = 0; o.fd = -1;
        }
        return *this;
    }
    ~MmapFloatMatrix() { reset(); }

    void reset() {
        if (map_base) { munmap(map_base, map_bytes); map_base = nullptr; }
        if (fd >= 0) { close(fd); fd = -1; }
    }
};

inline MmapFloatMatrix load_fvecs_mmap(const char* fvecs_path,
                                        size_t convert_chunk_rows = 20000) {
    FILE* f = fopen(fvecs_path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + fvecs_path);

    int32_t d = 0;
    if (fread(&d, sizeof(int32_t), 1, f) != 1 || d <= 0)
        throw std::runtime_error(std::string("empty/corrupt fvecs: ") + fvecs_path);
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    long rec_bytes = (long)sizeof(int32_t) + (long)d * (long)sizeof(float);
    long n = fsz / rec_bytes;

    std::string raw_path = std::string(fvecs_path) + ".raw_f32";
    long want_bytes = n * (long)d * (long)sizeof(float);

    struct stat st;
    bool need_convert = !(stat(raw_path.c_str(), &st) == 0 &&
                           (long)st.st_size == want_bytes);

    if (need_convert) {
        printf("  [mmap loader] converting %s -> %s (%ld x %d, streamed, "
               "%zu rows/chunk, no full-array buffering)\n",
               fvecs_path, raw_path.c_str(), n, (int)d, convert_chunk_rows);
        FILE* out = fopen(raw_path.c_str(), "wb");
        if (!out) throw std::runtime_error("cannot create " + raw_path);
        std::vector<float> buf((size_t)convert_chunk_rows * d);
        long done = 0;
        while (done < n) {
            long take = std::min((long)convert_chunk_rows, n - done);
            for (long i = 0; i < take; i++) {
                int32_t dd;
                if (fread(&dd, sizeof(int32_t), 1, f) != 1 || dd != d)
                    throw std::runtime_error("corrupt/mismatched fvecs record in " +
                                              std::string(fvecs_path));
                if (fread(buf.data() + (size_t)i * d, sizeof(float), (size_t)d, f)
                        != (size_t)d)
                    throw std::runtime_error("truncated fvecs record in " +
                                              std::string(fvecs_path));
            }
            fwrite(buf.data(), sizeof(float), (size_t)take * d, out);
            done += take;
        }
        fclose(out);
    } else {
        printf("  [mmap loader] reusing cached %s\n", raw_path.c_str());
    }
    fclose(f);

    MmapFloatMatrix m;
    m.n = (int)n; m.d = d;
    m.fd = open(raw_path.c_str(), O_RDONLY);
    if (m.fd < 0) throw std::runtime_error("cannot open " + raw_path);
    m.map_bytes = (size_t)want_bytes;
    void* p = mmap(nullptr, m.map_bytes, PROT_READ, MAP_SHARED, m.fd, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + raw_path);
#ifdef MADV_SEQUENTIAL
    // add_core() reads this sequentially in 65536-row chunks -- hint the
    // kernel so it read-aheads and evicts-behind instead of caching the
    // whole mapping resident, which is the whole point.
    madvise(p, m.map_bytes, MADV_SEQUENTIAL);
#endif
    m.map_base = p;
    m.data = (const float*)p;
    return m;
}
