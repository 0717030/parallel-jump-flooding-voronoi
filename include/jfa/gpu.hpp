#pragma once
#include <jfa/types.hpp>
#include <vector>
#include <functional>

namespace jfa {

// Callback for visualization/debugging, similar to CPU version
using PassCallback = std::function<void(int passIndex,
                                        int step,
                                        const SeedIndexBuffer& buffer)>;

void jfa_gpu_cuda(const Config& cfg,
                  const std::vector<Seed>& seeds,
                  SeedIndexBuffer& out_buffer,
                  PassCallback pass_cb = nullptr);

// Overload for raw pointer output (useful for custom host memory allocation)
void jfa_gpu_cuda(const Config& cfg,
                  const std::vector<Seed>& seeds,
                  int* out_buffer_ptr,
                  PassCallback pass_cb = nullptr);

} // namespace jfa
