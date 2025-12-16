// include/jfa/types.hpp
#pragma once
#include <vector>
#include <cstdint>
#include <string>

namespace jfa {

struct Seed {
    int x;
    int y;
};

struct Color {
    std::uint8_t r, g, b;
};

using SeedIndexBuffer = std::vector<int>;
using RGBImage        = std::vector<Color>;

struct Config {
    int width;
    int height;
    int block_dim_x = 16; // CUDA block dimension X
    int block_dim_y = 16; // CUDA block dimension Y
    int pixels_per_thread = 1; // Number of pixels processed by one thread
    bool use_pitch = false; // Use cudaMallocPitch for memory alignment
    bool use_shared_mem = false; // Use shared memory for seeds
    bool use_constant_mem = false; // Use constant memory for seeds
    bool use_soa = false; // Use Structure of Arrays for seeds
    bool use_coord_prop = false; // Use Coordinate Propagation
    bool use_ldg = false; // Use __ldg() intrinsic
    bool use_restrict = false; // Use __restrict__ keyword
    bool use_fma = false; // Use fmaf() intrinsic
    bool use_ultimate = false; // Use combined intrinsics (__restrict__, __ldg, fmaf)
    bool use_int_math = false; // Use integer math instead of float
    bool use_manhattan = false; // Use Manhattan distance (__sad)
};

} // namespace jfa
