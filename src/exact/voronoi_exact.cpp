// src/exact/voronoi_exact.cpp
#include <jfa/exact.hpp>
#include <limits>
#include <cmath>

namespace jfa {

void voronoi_exact_cpu(const Config& cfg,
                       const std::vector<Seed>& seeds,
                       SeedIndexBuffer& out_buffer)
{
    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;

    out_buffer.assign(N, -1);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int best_seed = -1;
            float best_dist = std::numeric_limits<float>::infinity();

            for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
                const auto& s = seeds[i];
                float dx = static_cast<float>(s.x - x);
                float dy = static_cast<float>(s.y - y);
                float d2 = dx*dx + dy*dy;

                if (d2 < best_dist) {
                    best_dist = d2;
                    best_seed = i;
                }
            }

            out_buffer[y * W + x] = best_seed;
        }
    }
}

} // namespace jfa
