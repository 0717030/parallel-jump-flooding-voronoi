// src/cpu/jfa_common_impl.hpp
#pragma once

#include <jfa/cpu.hpp>
#include <algorithm>
#include <limits>
#include <cmath>

namespace jfa::detail {

constexpr int INVALID_SEED = -1;
constexpr int INVALID_COORD = -1;

inline float sq_distance_to_seed(int px, int py,
                                 int seed_idx,
                                 const std::vector<Seed>& seeds)
{
    if (seed_idx == INVALID_SEED) {
        return std::numeric_limits<float>::infinity();
    }
    const Seed& s = seeds[seed_idx];
    float dx = static_cast<float>(s.x - px);
    float dy = static_cast<float>(s.y - py);
    return dx * dx + dy * dy;
}

inline float sq_distance_to_coord(int px, int py, int sx, int sy)
{
    if (sx == INVALID_COORD) {
        return std::numeric_limits<float>::infinity();
    }
    float dx = static_cast<float>(sx - px);
    float dy = static_cast<float>(sy - py);
    return dx * dx + dy * dy;
}

// 單一 JFA step（stride = step），in_buf -> out_buf
template <bool UseOpenMP>
inline void jfa_step(const Config& cfg,
                     const std::vector<Seed>& seeds,
                     const SeedIndexBuffer& in_buf,
                     SeedIndexBuffer& out_buf,
                     int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    // 先直接複製，避免沒有更好的 seed 時資料被洗掉
    out_buf = in_buf;

    if constexpr (UseOpenMP) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;
                int best_seed = in_buf[idx];
                float best_dist = sq_distance_to_seed(x, y, best_seed, seeds);

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx * step;
                        int ny = y + dy * step;
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

                        int nidx = ny * W + nx;
                        int n_seed = in_buf[nidx];
                        if (n_seed == INVALID_SEED) continue;

                        float d = sq_distance_to_seed(x, y, n_seed, seeds);
                        if (d < best_dist) {
                            best_dist = d;
                            best_seed = n_seed;
                        }
                    }
                }

                out_buf[idx] = best_seed;
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;
                int best_seed = in_buf[idx];
                float best_dist = sq_distance_to_seed(x, y, best_seed, seeds);

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = x + dx * step;
                        int ny = y + dy * step;
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

                        int nidx = ny * W + nx;
                        int n_seed = in_buf[nidx];
                        if (n_seed == INVALID_SEED) continue;

                        float d = sq_distance_to_seed(x, y, n_seed, seeds);
                        if (d < best_dist) {
                            best_dist = d;
                            best_seed = n_seed;
                        }
                    }
                }

                out_buf[idx] = best_seed;
            }
        }
    }
}

// 單一 JFA step（stride = step），coordinate propagation path.
// Input/Output are per-pixel seed coordinates (sx,sy). INVALID_COORD means unset.
template <bool UseOpenMP>
inline void jfa_step_coord_prop(const Config& cfg,
                                const int* sx_in,
                                const int* sy_in,
                                int* sx_out,
                                int* sy_out,
                                int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    if constexpr (UseOpenMP) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int idx = y * W + x;
                int best_sx = sx_in[idx];
                int best_sy = sy_in[idx];
                float best_dist = sq_distance_to_coord(x, y, best_sx, best_sy);

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx * step;
                        const int ny = y + dy * step;
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

                        const int nidx = ny * W + nx;
                        const int csx = sx_in[nidx];
                        if (csx == INVALID_COORD) continue;
                        const int csy = sy_in[nidx];

                        const float d = sq_distance_to_coord(x, y, csx, csy);
                        if (d < best_dist) {
                            best_dist = d;
                            best_sx = csx;
                            best_sy = csy;
                        }
                    }
                }

                sx_out[idx] = best_sx;
                sy_out[idx] = best_sy;
            }
        }
    } else {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int idx = y * W + x;
                int best_sx = sx_in[idx];
                int best_sy = sy_in[idx];
                float best_dist = sq_distance_to_coord(x, y, best_sx, best_sy);

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx * step;
                        const int ny = y + dy * step;
                        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

                        const int nidx = ny * W + nx;
                        const int csx = sx_in[nidx];
                        if (csx == INVALID_COORD) continue;
                        const int csy = sy_in[nidx];

                        const float d = sq_distance_to_coord(x, y, csx, csy);
                        if (d < best_dist) {
                            best_dist = d;
                            best_sx = csx;
                            best_sy = csy;
                        }
                    }
                }

                sx_out[idx] = best_sx;
                sy_out[idx] = best_sy;
            }
        }
    }
}

inline void coords_to_indices(const Config& cfg,
                              const std::vector<int>& id_map,
                              const int* sx,
                              const int* sy,
                              SeedIndexBuffer& out_indices)
{
    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;
    out_indices.resize(N);

    for (int i = 0; i < N; ++i) {
        const int x = sx[i];
        if (x == INVALID_COORD) {
            out_indices[i] = INVALID_SEED;
            continue;
        }
        const int y = sy[i];
        // Defensive: coordinates should always be in-bounds.
        if (x < 0 || x >= W || y < 0 || y >= H) {
            out_indices[i] = INVALID_SEED;
            continue;
        }
        out_indices[i] = id_map[y * W + x];
    }
}

// 多輪 JFA，step = maxDim/2, maxDim/4, ..., 1
template <bool UseOpenMP>
inline void jfa_cpu_common(const Config& cfg,
                           const std::vector<Seed>& seeds,
                           SeedIndexBuffer& out_buffer,
                           PassCallback pass_cb)
{
    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;

    // Coordinate propagation mode (store per-pixel seed coordinates, then map back to seed indices).
    if (cfg.use_coord_prop) {
        std::vector<int> id_map(N, INVALID_SEED);
        for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
            const auto& s = seeds[i];
            if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
            id_map[s.y * W + s.x] = i;
        }

        std::vector<int> sxA(N, INVALID_COORD), syA(N, INVALID_COORD);
        std::vector<int> sxB(N, INVALID_COORD), syB(N, INVALID_COORD);

        for (const auto& s : seeds) {
            if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
            const int idx = s.y * W + s.x;
            sxA[idx] = s.x;
            syA[idx] = s.y;
        }

        int max_dim = std::max(W, H);
        int step = max_dim / 2;
        if (step <= 0) step = 1;

        bool fromA = true;
        int pass_idx = 0;

        SeedIndexBuffer tmp_indices;

        while (step >= 1) {
            const int* sx_in = fromA ? sxA.data() : sxB.data();
            const int* sy_in = fromA ? syA.data() : syB.data();
            int* sx_out = fromA ? sxB.data() : sxA.data();
            int* sy_out = fromA ? syB.data() : syA.data();

            jfa_step_coord_prop<UseOpenMP>(cfg, sx_in, sy_in, sx_out, sy_out, step);

            if (pass_cb) {
                coords_to_indices(cfg, id_map, sx_out, sy_out, tmp_indices);
                pass_cb(pass_idx, step, tmp_indices);
            }

            fromA = !fromA;
            step /= 2;
            ++pass_idx;
        }

        const int* sx_final = fromA ? sxA.data() : sxB.data();
        const int* sy_final = fromA ? syA.data() : syB.data();
        coords_to_indices(cfg, id_map, sx_final, sy_final, out_buffer);
        return;
    }

    SeedIndexBuffer bufA(N, INVALID_SEED);
    SeedIndexBuffer bufB(N, INVALID_SEED);

    // 初始化：每個 seed 格子存自己的 index
    for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
        const auto& s = seeds[i];
        if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
        bufA[s.y * W + s.x] = i;
    }

    int max_dim = std::max(W, H);
    int step = max_dim / 2;
    if (step <= 0) step = 1;

    bool fromA = true;
    int pass_idx = 0;

    while (step >= 1) {
        const SeedIndexBuffer& in  = fromA ? bufA : bufB;
        SeedIndexBuffer&       out = fromA ? bufB : bufA;

        jfa_step<UseOpenMP>(cfg, seeds, in, out, step);

        if (pass_cb) {
            pass_cb(pass_idx, step, out);
        }

        fromA = !fromA;
        step /= 2;
        ++pass_idx;
    }

    const SeedIndexBuffer& final_buf = fromA ? bufA : bufB;
    out_buffer = final_buf;
}

} // namespace jfa::detail
