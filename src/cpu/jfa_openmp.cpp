// src/cpu/jfa_openmp.cpp
#include <jfa/cpu.hpp>
#include "jfa_common_impl.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace jfa {

void jfa_cpu_omp(const Config& cfg,
                 const std::vector<Seed>& seeds,
                 SeedIndexBuffer& out_buffer,
                 int num_threads,
                 PassCallback pass_cb)
{
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif
    detail::jfa_cpu_common<true>(cfg, seeds, out_buffer, pass_cb);
}

} // namespace jfa
