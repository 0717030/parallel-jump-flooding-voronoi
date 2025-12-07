// include/jfa/cpu.hpp
#pragma once
#include <vector>
#include <functional>
#include <jfa/types.hpp>

namespace jfa {

using PassCallback = std::function<void(int passIndex,
                                        int step,
                                        const SeedIndexBuffer& buffer)>;

void jfa_cpu_serial(const Config& cfg,
                    const std::vector<Seed>& seeds,
                    SeedIndexBuffer& out_buffer,
                    PassCallback pass_cb = nullptr);

void jfa_cpu_omp(const Config& cfg,
                 const std::vector<Seed>& seeds,
                 SeedIndexBuffer& out_buffer,
                 int num_threads = 0,
                 PassCallback pass_cb = nullptr);

} // namespace jfa
