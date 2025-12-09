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
};

} // namespace jfa
