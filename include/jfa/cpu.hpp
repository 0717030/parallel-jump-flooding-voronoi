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

// CPU SIMD (AVX2) backend.
// Notes:
// - Intended for x86_64 CPUs with AVX2 (e.g. i7-12700).
// - When cfg.use_coord_prop is true, uses Coordinate Propagation (stores seed coordinates per pixel).
// - When cfg.use_soa is true, uses SoA layout for internal per-pixel coord buffers; otherwise AoS.
void jfa_cpu_simd(const Config& cfg,
                  const std::vector<Seed>& seeds,
                  SeedIndexBuffer& out_buffer,
                  PassCallback pass_cb = nullptr);

// OpenMP + SIMD variant (threads + AVX2).
void jfa_cpu_omp_simd(const Config& cfg,
                      const std::vector<Seed>& seeds,
                      SeedIndexBuffer& out_buffer,
                      int num_threads = 0,
                      PassCallback pass_cb = nullptr);

} // namespace jfa
