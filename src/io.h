#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Read float vectors from .fvecs file. Returns flat (n×d) array.
inline std::vector<float> read_fvecs(const char* path, int* n_out, int* d_out) {
    FILE* f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);

    int d;
    if (fread(&d, 4, 1, f) != 1) {
        fclose(f); throw std::runtime_error(std::string("Read error (dim): ") + path);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(sz / (4 + (long)d * 4));

    std::vector<float> data((size_t)n * d);
    for (int i = 0; i < n; i++) {
        int d2;
        if (fread(&d2, 4, 1, f) != 1 ||
            fread(data.data() + (size_t)i * d, 4, d, f) != (size_t)d) {
            fclose(f);
            throw std::runtime_error(std::string("Read error at row ") +
                                     std::to_string(i) + " in " + path);
        }
    }
    fclose(f);
    *n_out = n;
    *d_out = d;
    return data;
}

// Read integer vectors from .ivecs file.
inline std::vector<int32_t> read_ivecs(const char* path, int* n_out, int* d_out) {
    FILE* f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);

    int d;
    if (fread(&d, 4, 1, f) != 1) {
        fclose(f); throw std::runtime_error(std::string("Read error (dim): ") + path);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(sz / (4 + (long)d * 4));

    std::vector<int32_t> data((size_t)n * d);
    for (int i = 0; i < n; i++) {
        int d2;
        if (fread(&d2, 4, 1, f) != 1 ||
            fread(data.data() + (size_t)i * d, 4, d, f) != (size_t)d) {
            fclose(f);
            throw std::runtime_error(std::string("Read error at row ") +
                                     std::to_string(i) + " in " + path);
        }
    }
    fclose(f);
    *n_out = n;
    *d_out = d;
    return data;
}
