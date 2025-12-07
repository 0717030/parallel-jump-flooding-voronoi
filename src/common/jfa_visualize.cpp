// src/common/jfa_visualize.cpp
#include <jfa/visualize.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>

namespace jfa {

std::vector<Color> make_seed_palette(std::size_t num_seeds)
{
    std::vector<Color> palette(num_seeds);

    for (std::size_t i = 0; i < num_seeds; ++i) {
        // Knuth hash，讓顏色在 index 上分散一點
        std::uint32_t h = static_cast<std::uint32_t>(i * 2654435761u);
        std::uint8_t r = static_cast<std::uint8_t>((h >> 16) & 0xFF);
        std::uint8_t g = static_cast<std::uint8_t>((h >>  8) & 0xFF);
        std::uint8_t b = static_cast<std::uint8_t>((h      ) & 0xFF);

        // 稍微避免太暗
        if (r < 64 && g < 64 && b < 64) {
            r = static_cast<std::uint8_t>(r + 64);
            g = static_cast<std::uint8_t>(g + 64);
            b = static_cast<std::uint8_t>(b + 64);
        }

        palette[i] = Color{r, g, b};
    }

    return palette;
}

void seed_index_to_rgb(const SeedIndexBuffer& seed_indices,
                       const std::vector<Seed>& /*seeds*/,
                       const std::vector<Color>& palette,
                       int width,
                       int height,
                       RGBImage& out_img)
{
    
    assert(static_cast<int>(seed_indices.size()) == width * height);
    
    const int N = width * height;
    out_img.resize(N);

    for (int i = 0; i < N; ++i) {
        int idx = seed_indices[i];
        if (idx < 0 || idx >= static_cast<int>(palette.size())) {
            // 沒有 seed → 畫黑色背景
            out_img[i] = Color{0, 0, 0};
        } else {
            out_img[i] = palette[idx];
        }
    }
}

bool write_ppm(const std::string& filename,
               int width,
               int height,
               const RGBImage& img)
{
    // 在 write_ppm 開頭
    const std::size_t expected = static_cast<std::size_t>(width) * height;
    if (img.size() != expected) {
        return false;
    }

    // 確保目錄存在
    std::filesystem::path path(filename);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) return false;

    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (const auto& c : img) {
        ofs.put(static_cast<char>(c.r));
        ofs.put(static_cast<char>(c.g));
        ofs.put(static_cast<char>(c.b));
    }
    return true;
}

} // namespace jfa
