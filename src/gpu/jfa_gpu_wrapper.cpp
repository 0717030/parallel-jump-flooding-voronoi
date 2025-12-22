#include <jfa/gpu.hpp>
#include <vector>
#include <cstring> // for memcpy if needed
#include <stdexcept>
#include <cuda_runtime.h> // For cudaMallocHost

namespace jfa {

// Helper to allocate pinned memory
int* allocate_pinned_memory(size_t count) {
    int* ptr = nullptr;
    if (cudaMallocHost(&ptr, count * sizeof(int)) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

// Helper to free pinned memory
void free_pinned_memory(int* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

void cuda_warmup() {
    cudaFree(0);
}

// Define GPUSeed here to match .cu (layout compatible with Seed)
struct GPUSeed {
    int x, y;
};

// Declare the internal implementation
using InternalCallback = void (*)(int pass_idx, int step, const int* data, int size, void* user_data);

int jfa_gpu_cuda_impl(const Config& cfg,
                       const GPUSeed* seeds, 
                       const int* seeds_x, const int* seeds_y,
                       int num_seeds,
                       int* out_buffer,
                       InternalCallback cb,
                       void* user_data);

// Static wrapper function to bridge C-style callback to std::function
static void callback_wrapper(int pass_idx, int step, const int* data, int size, void* user_data) {
    auto* func = static_cast<PassCallback*>(user_data);
    if (func && *func) {
        // Convert raw buffer to SeedIndexBuffer (std::vector<int>)
        // This involves a copy, but callback is for visualization/debug anyway.
        SeedIndexBuffer buf(data, data + size);
        (*func)(pass_idx, step, buf);
    }
}

void jfa_gpu_cuda(const Config& cfg,
                  const std::vector<Seed>& seeds,
                  SeedIndexBuffer& out_buffer,
                  PassCallback pass_cb)
{
    out_buffer.resize(cfg.width * cfg.height);
    jfa_gpu_cuda(cfg, seeds, out_buffer.data(), pass_cb);
}

void jfa_gpu_cuda(const Config& cfg,
                  const std::vector<Seed>& seeds,
                  int* out_buffer_ptr,
                  PassCallback pass_cb)
{
    std::vector<GPUSeed> gpu_seeds;
    std::vector<int> seeds_x;
    std::vector<int> seeds_y;

    // If using Coordinate Propagation, we prefer AoS seeds for initialization simplicity
    bool use_soa_layout = cfg.use_soa && !cfg.use_coord_prop;

    if (use_soa_layout) {
        seeds_x.resize(seeds.size());
        seeds_y.resize(seeds.size());
        for(size_t i=0; i<seeds.size(); ++i) {
            seeds_x[i] = seeds[i].x;
            seeds_y[i] = seeds[i].y;
        }
    } else {
        gpu_seeds.resize(seeds.size());
        for(size_t i=0; i<seeds.size(); ++i) {
            gpu_seeds[i] = {seeds[i].x, seeds[i].y};
        }
    }

    int ret = jfa_gpu_cuda_impl(cfg,
                                use_soa_layout ? nullptr : gpu_seeds.data(),
                                use_soa_layout ? seeds_x.data() : nullptr,
                                use_soa_layout ? seeds_y.data() : nullptr,
                                (int)seeds.size(),
                                out_buffer_ptr,
                                pass_cb ? callback_wrapper : nullptr, 
                                pass_cb ? &pass_cb : nullptr);

    if (ret != 0) {
        throw std::runtime_error("CUDA execution failed (see stdout for details)");
    }
}

} // namespace jfa
