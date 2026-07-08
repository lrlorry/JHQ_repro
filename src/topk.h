#pragma once
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

// Max-heap that retains the k smallest (distance, id) pairs seen so far.
struct TopKHeap {
    int k;
    std::vector<std::pair<float, int>> heap;  // invariant: max-heap of size ≤ k

    explicit TopKHeap(int k) : k(k) { heap.reserve(k + 1); }

    void push(float dist, int id) {
        if ((int)heap.size() < k) {
            heap.emplace_back(dist, id);
            std::push_heap(heap.begin(), heap.end());
        } else if (dist < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {dist, id};
            std::push_heap(heap.begin(), heap.end());
        }
    }

    // Write results sorted by ascending distance into caller-provided arrays.
    // Destroys heap property — do not call push() after this.
    void results(float* dists_out, int* ids_out) {
        std::sort_heap(heap.begin(), heap.end());
        std::reverse(heap.begin(), heap.end());
        int sz = (int)heap.size();
        for (int i = 0; i < sz; i++) {
            dists_out[i] = heap[i].first;
            ids_out[i]   = heap[i].second;
        }
        // Pad remaining with sentinels if heap has fewer than k entries
        for (int i = sz; i < k; i++) {
            dists_out[i] = std::numeric_limits<float>::infinity();
            ids_out[i]   = -1;
        }
    }
};
