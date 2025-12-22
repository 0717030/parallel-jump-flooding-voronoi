// src/cpu/jfa_openmp.cpp
#include <jfa/cpu.hpp>
#include "jfa_common_impl.hpp"
#include "cpu_affinity.hpp"

#include <omp.h>

namespace jfa {

void jfa_cpu_omp(const Config& cfg,
                 const std::vector<Seed>& seeds,
                 SeedIndexBuffer& out_buffer,
                 int num_threads,
                 PassCallback pass_cb)
{
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Set CPU affinity:
    // - For small thread counts (<= P-core count), pin to P-cores only.
    // - For larger thread counts, allow all cores so extra threads can use E-cores too.
    detail::set_affinity_for_threads(num_threads);

    detail::jfa_cpu_common<true>(cfg, seeds, out_buffer, pass_cb);
}

} // namespace jfa
