// src/cpu/jfa_serial.cpp
#include <jfa/cpu.hpp>
#include "jfa_common_impl.hpp"

namespace jfa {

void jfa_cpu_serial(const Config& cfg,
                    const std::vector<Seed>& seeds,
                    SeedIndexBuffer& out_buffer,
                    PassCallback pass_cb)
{
    detail::jfa_cpu_common<false>(cfg, seeds, out_buffer, pass_cb);
}

} // namespace jfa
