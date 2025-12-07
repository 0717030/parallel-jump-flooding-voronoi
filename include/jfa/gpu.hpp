// include/jfa/gpu.hpp
#pragma once
#include "jfa/types.hpp"

namespace jfa {

void jfa_cuda_basic(const Config& cfg,
                    const std::vector<Seed>& seeds,
                    SeedIndexBuffer& out,
                    PassCallback cb = nullptr);

// 之後隊友可以再加進階版本：
// void jfa_cuda_inplace(...);
// void jfa_cuda_jfa_plus1(...);

} // namespace jfa
