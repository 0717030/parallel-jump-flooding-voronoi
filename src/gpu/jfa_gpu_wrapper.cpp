#include <jfa/gpu.hpp>
#include <vector>
#include <cstring> // for memcpy if needed
#include <stdexcept>

namespace jfa {

// Define GPUSeed here to match .cu (layout compatible with Seed)
struct GPUSeed {
    int x, y;
};

// Declare the internal implementation
using InternalCallback = void (*)(int pass_idx, int step, const int* data, int size, void* user_data);

int jfa_gpu_cuda_impl(const Config& cfg,
                       const GPUSeed* seeds, int num_seeds,
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

    // Convert seeds to GPUSeed array
    std::vector<GPUSeed> gpu_seeds(seeds.size());
    for(size_t i=0; i<seeds.size(); ++i) {
        gpu_seeds[i] = {seeds[i].x, seeds[i].y};
    }

    int ret = 0;
    if (pass_cb) {
        ret = jfa_gpu_cuda_impl(cfg,
                                gpu_seeds.data(), (int)gpu_seeds.size(),
                                out_buffer.data(),
                                callback_wrapper, &pass_cb);
    } else {
        ret = jfa_gpu_cuda_impl(cfg,
                                gpu_seeds.data(), (int)gpu_seeds.size(),
                                out_buffer.data(),
                                nullptr, nullptr);
    }    if (ret != 0) {
        throw std::runtime_error("CUDA execution failed (see stdout for details)");
    }
}

void jfa_gpu_cuda(const Config& cfg,
                  const std::vector<Seed>& seeds,
                  int* out_buffer_ptr,
                  PassCallback pass_cb)
{
    // Convert seeds to GPUSeed array
    std::vector<GPUSeed> gpu_seeds(seeds.size());
    for(size_t i=0; i<seeds.size(); ++i) {
        gpu_seeds[i] = {seeds[i].x, seeds[i].y};
    }

    int ret = 0;
    if (pass_cb) {
        ret = jfa_gpu_cuda_impl(cfg,
                                gpu_seeds.data(), (int)gpu_seeds.size(),
                                out_buffer_ptr,
                                callback_wrapper, &pass_cb);
    } else {
        ret = jfa_gpu_cuda_impl(cfg,
                                gpu_seeds.data(), (int)gpu_seeds.size(),
                                out_buffer_ptr,
                                nullptr, nullptr);
    }

    if (ret != 0) {
        throw std::runtime_error("CUDA execution failed (see stdout for details)");
    }
}

} // namespace jfa
