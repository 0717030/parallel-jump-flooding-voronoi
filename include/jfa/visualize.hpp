// include/jfa/visualize.hpp
#pragma once
#include <vector>
#include <string>
#include <jfa/types.hpp>

namespace jfa {

// 根據 seed 數量產生一組顏色表
std::vector<Color> make_seed_palette(std::size_t num_seeds);

// 把每個 pixel 的 seed index 映成 RGB
void seed_index_to_rgb(const SeedIndexBuffer& seed_indices,
                       const std::vector<Seed>& seeds,
                       const std::vector<Color>& palette,
                       int width,
                       int height,
                       RGBImage& out_img);

// 輸出二進位 PPM (P6)
bool write_ppm(const std::string& filename,
               int width,
               int height,
               const RGBImage& img);

} // namespace jfa
