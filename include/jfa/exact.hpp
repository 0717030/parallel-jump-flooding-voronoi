// include/jfa/exact.hpp
#pragma once
#include <vector>
#include <jfa/types.hpp>

namespace jfa {

// O(W * H * #seeds) 的精確 Voronoi baseline
void voronoi_exact_cpu(const Config& cfg,
                       const std::vector<Seed>& seeds,
                       SeedIndexBuffer& out_buffer);

} // namespace jfa
