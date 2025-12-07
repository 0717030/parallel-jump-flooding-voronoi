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
};

} // namespace jfa
