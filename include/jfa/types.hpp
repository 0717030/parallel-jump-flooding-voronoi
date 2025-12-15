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

enum class CpuSeedsLayout {
    Packed, // one int per seed: (y << 16) | (x & 0xFFFF) (CPU SIMD gather optimization)
    SoA,    // two arrays: seeds_x[], seeds_y[]
    AoS,    // interleaved int array: [x0,y0,x1,y1,...] (matches struct Seed layout)
};

enum class CpuCoordBufLayout {
    SoA, // per-pixel buffers: sx[], sy[]
    AoS, // per-pixel buffer: xy[] interleaved [sx,sy,sx,sy,...]
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
    bool use_coord_prop = false; // Use Coordinate Propagation (avoids seed lookup)
    bool cpu_use_pitch = false; // CPU SIMD: pad internal row stride to a multiple of 16 pixels, improves alignment for vector loads/stores
    CpuSeedsLayout cpu_seeds_layout = CpuSeedsLayout::Packed; // CPU SIMD only (index-based mode)
    CpuCoordBufLayout cpu_coordbuf_layout = CpuCoordBufLayout::SoA; // CPU SIMD only (coord-prop mode)
    int benchmark_frames = 1; // Number of frames to simulate (reuse allocation)
};

} // namespace jfa
