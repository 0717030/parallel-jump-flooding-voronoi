// src/cpu/jfa_simd_avx2.cpp
//
// CPU SIMD backend (AVX2) for Jump Flooding Algorithm (JFA).
//
// Primary fast path: Coordinate Propagation (store per-pixel seed coordinates),
// avoiding random seed lookups and enabling contiguous loads in X direction.
//
// CPU coord-prop uses a fixed internal SoA buffer layout (sx[], sy[]) for best performance.
//
// Parallel option:
// - jfa_cpu_omp_simd(): OpenMP over rows + AVX2 inside row.

#include <jfa/cpu.hpp>
#include "cpu_affinity.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <immintrin.h>

#include <omp.h>

namespace jfa {

namespace {

constexpr int INVALID_COORD = -1;
constexpr int DIST_INF = 0x3fffffff;

inline int sq_dist_i32(int x, int y, int sx, int sy)
{
    if (sx == INVALID_COORD) return DIST_INF;
    int dx = sx - x;
    int dy = sy - y;
    return dx * dx + dy * dy;
}

// --------------------------
// Index-based (seed-index) helpers
// --------------------------

inline int sq_dist_seed_scalar(int x, int y, int seed_idx, const std::vector<Seed>& seeds)
{
    if (seed_idx < 0) return DIST_INF;
    const auto& s = seeds[seed_idx];
    return sq_dist_i32(x, y, s.x, s.y);
}

// Seed coordinate gatherer for AVX2.
// - If seeds_xy != nullptr: gather from interleaved AoS layout [x0,y0,x1,y1,...]
// - Else: gather from SoA arrays seeds_x / seeds_y.
struct SeedGather {
    const int* seeds_x = nullptr;
    const int* seeds_y = nullptr;
    const int* seeds_xy = nullptr;
    const int* seeds_packed = nullptr; // packed: (y << 16) | (x & 0xFFFF)

    inline void gather_xy(__m256i seed_idx, __m256i& sx, __m256i& sy, __m256i& invalid_mask) const
    {
        const __m256i neg1 = _mm256_set1_epi32(-1);
        invalid_mask = _mm256_cmpeq_epi32(seed_idx, neg1);
        // safe_idx maps -1 -> 0 to avoid OOB in gathers; distances will be masked to INF.
        __m256i safe_idx = _mm256_andnot_si256(invalid_mask, seed_idx);

        if (seeds_packed) {
            const __m256i mask_lo = _mm256_set1_epi32(0xFFFF);
            __m256i packed = _mm256_i32gather_epi32(seeds_packed, safe_idx, 4);
            sx = _mm256_and_si256(packed, mask_lo);
            sy = _mm256_srli_epi32(packed, 16);
        } else if (seeds_xy) {
            __m256i idx2 = _mm256_slli_epi32(safe_idx, 1);
            sx = _mm256_i32gather_epi32(seeds_xy, idx2, 4);
            __m256i idx2p1 = _mm256_add_epi32(idx2, _mm256_set1_epi32(1));
            sy = _mm256_i32gather_epi32(seeds_xy, idx2p1, 4);
        } else {
            sx = _mm256_i32gather_epi32(seeds_x, safe_idx, 4);
            sy = _mm256_i32gather_epi32(seeds_y, safe_idx, 4);
        }
    }

    // Variant when caller already computed invalid_mask = (seed_idx == -1).
    inline void gather_xy_known_mask(__m256i seed_idx, __m256i invalid_mask, __m256i& sx, __m256i& sy) const
    {
        __m256i safe_idx = _mm256_andnot_si256(invalid_mask, seed_idx);
        if (seeds_packed) {
            const __m256i mask_lo = _mm256_set1_epi32(0xFFFF);
            __m256i packed = _mm256_i32gather_epi32(seeds_packed, safe_idx, 4);
            sx = _mm256_and_si256(packed, mask_lo);
            sy = _mm256_srli_epi32(packed, 16);
        } else if (seeds_xy) {
            __m256i idx2 = _mm256_slli_epi32(safe_idx, 1);
            sx = _mm256_i32gather_epi32(seeds_xy, idx2, 4);
            __m256i idx2p1 = _mm256_add_epi32(idx2, _mm256_set1_epi32(1));
            sy = _mm256_i32gather_epi32(seeds_xy, idx2p1, 4);
        } else {
            sx = _mm256_i32gather_epi32(seeds_x, safe_idx, 4);
            sy = _mm256_i32gather_epi32(seeds_y, safe_idx, 4);
        }
    }
};

inline __m256i dist_sq_vec_seed(__m256i xv, __m256i yv, __m256i sx, __m256i sy, __m256i invalid_mask)
{
    const __m256i inf  = _mm256_set1_epi32(DIST_INF);
    __m256i dx = _mm256_sub_epi32(sx, xv);
    __m256i dy = _mm256_sub_epi32(sy, yv);
    __m256i dx2 = _mm256_mullo_epi32(dx, dx);
    __m256i dy2 = _mm256_mullo_epi32(dy, dy);
    __m256i d = _mm256_add_epi32(dx2, dy2);
    return _mm256_blendv_epi8(d, inf, invalid_mask);
}

inline void update_best_idx(__m256i& best_d, __m256i& best_idx, __m256i cand_d, __m256i cand_idx)
{
    __m256i better = _mm256_cmpgt_epi32(best_d, cand_d);
    best_d   = _mm256_blendv_epi8(best_d,   cand_d,   better);
    best_idx = _mm256_blendv_epi8(best_idx, cand_idx, better);
}

inline void idx_step_scalar(const Config& cfg,
                            const std::vector<Seed>& seeds,
                            const SeedIndexBuffer& in_buf,
                            SeedIndexBuffer& out_buf,
                            int step)
{
    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;
    out_buf.resize(N);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            int best_seed = in_buf[idx];
            int best_d = sq_dist_seed_scalar(x, y, best_seed, seeds);

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx * step;
                    const int ny = y + dy * step;
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                    const int nidx = ny * W + nx;
                    const int n_seed = in_buf[nidx];
                    if (n_seed < 0) continue;

                    const int d = sq_dist_seed_scalar(x, y, n_seed, seeds);
                    if (d < best_d) {
                        best_d = d;
                        best_seed = n_seed;
                    }
                }
            }

            out_buf[idx] = best_seed;
        }
    }
}

// --------------------------
// Scalar coord-prop fallback
// --------------------------

inline void coord_step_scalar_soa(const Config& cfg,
                                  const int* sx_in,
                                  const int* sy_in,
                                  int* sx_out,
                                  int* sy_out,
                                  int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            int best_sx = sx_in[idx];
            int best_sy = sy_in[idx];
            int best_d  = sq_dist_i32(x, y, best_sx, best_sy);

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx * step;
                    const int ny = y + dy * step;
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                    const int nidx = ny * W + nx;
                    const int csx = sx_in[nidx];
                    if (csx == INVALID_COORD) continue;
                    const int csy = sy_in[nidx];
                    const int d = sq_dist_i32(x, y, csx, csy);
                    if (d < best_d) {
                        best_d = d;
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

inline void coord_step_scalar_aos(const Config& cfg,
                                  const int* xy_in,   // length 2*N
                                  int* xy_out,        // length 2*N
                                  int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int idx = y * W + x;
            const int base = 2 * idx;
            int best_sx = xy_in[base + 0];
            int best_sy = xy_in[base + 1];
            int best_d  = sq_dist_i32(x, y, best_sx, best_sy);

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx * step;
                    const int ny = y + dy * step;
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                    const int nidx = ny * W + nx;
                    const int nbase = 2 * nidx;
                    const int csx = xy_in[nbase + 0];
                    if (csx == INVALID_COORD) continue;
                    const int csy = xy_in[nbase + 1];
                    const int d = sq_dist_i32(x, y, csx, csy);
                    if (d < best_d) {
                        best_d = d;
                        best_sx = csx;
                        best_sy = csy;
                    }
                }
            }

            xy_out[base + 0] = best_sx;
            xy_out[base + 1] = best_sy;
        }
    }
}

// --------------------------
// AVX2 coord-prop kernels
// --------------------------

inline __m256i dist_sq_vec(__m256i x, __m256i y, __m256i sx, __m256i sy)
{
    const __m256i neg1 = _mm256_set1_epi32(INVALID_COORD);
    const __m256i inf  = _mm256_set1_epi32(DIST_INF);

    __m256i dx = _mm256_sub_epi32(sx, x);
    __m256i dy = _mm256_sub_epi32(sy, y);
    __m256i dx2 = _mm256_mullo_epi32(dx, dx);
    __m256i dy2 = _mm256_mullo_epi32(dy, dy);
    __m256i d = _mm256_add_epi32(dx2, dy2);

    __m256i invalid_mask = _mm256_cmpeq_epi32(sx, neg1);
    d = _mm256_blendv_epi8(d, inf, invalid_mask);
    return d;
}

inline void update_best(__m256i& best_d,
                        __m256i& best_sx,
                        __m256i& best_sy,
                        __m256i cand_d,
                        __m256i cand_sx,
                        __m256i cand_sy)
{
    // better = cand_d < best_d  <=> best_d > cand_d
    __m256i better = _mm256_cmpgt_epi32(best_d, cand_d);
    best_d  = _mm256_blendv_epi8(best_d,  cand_d,  better);
    best_sx = _mm256_blendv_epi8(best_sx, cand_sx, better);
    best_sy = _mm256_blendv_epi8(best_sy, cand_sy, better);
}

template <bool UseOpenMP>
inline void coord_step_soa_avx2(const Config& cfg,
                                const int* sx_in,
                                const int* sy_in,
                                int* sx_out,
                                int* sy_out,
                                int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    const __m256i inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    auto scalar_pixel = [&](int x, int y) {
        const int idx = y * W + x;
        int best_sx = sx_in[idx];
        int best_sy = sy_in[idx];
        int best_d  = sq_dist_i32(x, y, best_sx, best_sy);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx * step;
                const int ny = y + dy * step;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                const int nidx = ny * W + nx;
                const int csx = sx_in[nidx];
                if (csx == INVALID_COORD) continue;
                const int csy = sy_in[nidx];
                const int d = sq_dist_i32(x, y, csx, csy);
                if (d < best_d) {
                    best_d = d;
                    best_sx = csx;
                    best_sy = csy;
                }
            }
        }
        sx_out[idx] = best_sx;
        sy_out[idx] = best_sy;
    };

    if constexpr (UseOpenMP) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const bool y_interior = (y >= step) && (y < H - step);
            const int row = y * W;

            if (!y_interior || W < 8 || step <= 0) {
                for (int x = 0; x < W; ++x) scalar_pixel(x, y);
                continue;
            }

            for (int x = 0; x < step; ++x) scalar_pixel(x, y);

            int x = step;
            const int x_vec_end = (W - step) - 8;
            const __m256i yv = _mm256_set1_epi32(y);

            for (; x <= x_vec_end; x += 8) {
                const int idx0 = row + x;
                const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

                __m256i best_sx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + idx0));
                __m256i best_sy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + idx0));
                __m256i best_d  = dist_sq_vec(xv, yv, best_sx, best_sy);

                // dy=-1
                {
                    const int ny = y - step;
                    const int base = ny * W;
                    // dx=-1,0,1
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx * step;
                        const int nidx0 = base + nx;
                        __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                        __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                        __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                        update_best(best_d, best_sx, best_sy, cd, csx, csy);
                    }
                }

                // dy=0
                {
                    const int ny = y;
                    const int base = ny * W;
                    // dx=-1, +1 (skip center dx=0)
                    for (int dx : {-1, 1}) {
                        const int nx = x + dx * step;
                        const int nidx0 = base + nx;
                        __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                        __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                        __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                        update_best(best_d, best_sx, best_sy, cd, csx, csy);
                    }
                }

                // dy=+1
                {
                    const int ny = y + step;
                    const int base = ny * W;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx * step;
                        const int nidx0 = base + nx;
                        __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                        __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                        __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                        update_best(best_d, best_sx, best_sy, cd, csx, csy);
                    }
                }

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(sx_out + idx0), best_sx);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(sy_out + idx0), best_sy);
            }

            for (; x < W - step; ++x) scalar_pixel(x, y);
            for (int xb = W - step; xb < W; ++xb) scalar_pixel(xb, y);
        }
        return;
    }

    // Non-OpenMP path
    for (int y = 0; y < H; ++y) {
        const bool y_interior = (y >= step) && (y < H - step);
        const int row = y * W;

        if (!y_interior || W < 8 || step <= 0) {
            for (int x = 0; x < W; ++x) scalar_pixel(x, y);
            continue;
        }

        for (int x = 0; x < step; ++x) scalar_pixel(x, y);

        int x = step;
        const int x_vec_end = (W - step) - 8;
        const __m256i yv = _mm256_set1_epi32(y);

        for (; x <= x_vec_end; x += 8) {
            const int idx0 = row + x;
            const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

            __m256i best_sx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + idx0));
            __m256i best_sy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + idx0));
            __m256i best_d  = dist_sq_vec(xv, yv, best_sx, best_sy);

            // dy=-1 (dx=-1,0,1)
            {
                const int base = (y - step) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                    __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                    __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            // dy=0 (dx=-1,+1)
            {
                const int base = y * W;
                for (int dx : {-1, 1}) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                    __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                    __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            // dy=+1 (dx=-1,0,1)
            {
                const int base = (y + step) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                    __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                    __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sx_out + idx0), best_sx);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sy_out + idx0), best_sy);
        }

        for (; x < W - step; ++x) scalar_pixel(x, y);
        for (int xb = W - step; xb < W; ++xb) scalar_pixel(xb, y);
    }
}

// OpenMP-friendly variant: expects to be called *inside* an existing `#pragma omp parallel` region.
// Uses `#pragma omp for` to avoid re-creating a parallel team per JFA pass (important for small images).
inline void coord_step_soa_avx2_omp_for(const Config& cfg,
                                       const int* sx_in,
                                       const int* sy_in,
                                       int* sx_out,
                                       int* sy_out,
                                       int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    const __m256i inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    auto scalar_pixel = [&](int x, int y) {
        const int idx = y * W + x;
        int best_sx = sx_in[idx];
        int best_sy = sy_in[idx];
        int best_d  = sq_dist_i32(x, y, best_sx, best_sy);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx * step;
                const int ny = y + dy * step;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                const int nidx = ny * W + nx;
                const int csx = sx_in[nidx];
                if (csx == INVALID_COORD) continue;
                const int csy = sy_in[nidx];
                const int d = sq_dist_i32(x, y, csx, csy);
                if (d < best_d) {
                    best_d = d;
                    best_sx = csx;
                    best_sy = csy;
                }
            }
        }
        sx_out[idx] = best_sx;
        sy_out[idx] = best_sy;
    };

    #pragma omp for schedule(static)
    for (int y = 0; y < H; ++y) {
        const bool y_interior = (y >= step) && (y < H - step);
        const int row = y * W;

        if (!y_interior || W < 8 || step <= 0) {
            for (int x = 0; x < W; ++x) scalar_pixel(x, y);
            continue;
        }

        for (int x = 0; x < step; ++x) scalar_pixel(x, y);

        int x = step;
        const int x_vec_end = (W - step) - 8;
        const __m256i yv = _mm256_set1_epi32(y);

        for (; x <= x_vec_end; x += 8) {
            const int idx0 = row + x;
            const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

            __m256i best_sx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + idx0));
            __m256i best_sy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + idx0));
            __m256i best_d  = dist_sq_vec(xv, yv, best_sx, best_sy);

            // dy=-1: dx=-1,0,1
            {
                const int base = (y - step) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                    __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                    __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            // dy=0: dx=-1,+1
            {
                const int base = y * W;
                for (int dx : {-1, 1}) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                    __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                    __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            // dy=+1: dx=-1,0,1
            {
                const int base = (y + step) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i csx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sx_in + nidx0));
                    __m256i csy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sy_in + nidx0));
                    __m256i cd  = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sx_out + idx0), best_sx);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sy_out + idx0), best_sy);
        }

        for (; x < W - step; ++x) scalar_pixel(x, y);
        for (int xb = W - step; xb < W; ++xb) scalar_pixel(xb, y);
    }
}

template <bool UseOpenMP>
inline void coord_step_aos_avx2(const Config& cfg,
                                const int* xy_in, // length 2*N
                                int* xy_out,      // length 2*N
                                int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    const __m256i inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i even_idx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i odd_idx  = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);

    auto load_xy8 = [&](const int* base, __m256i& sx, __m256i& sy) {
        // base points to xy for the first pixel of this 8-wide chunk: [sx0,sy0,sx1,sy1,...]
        sx = _mm256_i32gather_epi32(base, even_idx, 4);
        sy = _mm256_i32gather_epi32(base, odd_idx, 4);
    };

    auto store_xy8 = [&](int* base, __m256i sx, __m256i sy) {
        // Interleave and store as two 256-bit chunks to match AoS order:
        // [sx0 sy0 sx1 sy1 sx2 sy2 sx3 sy3] then [sx4 sy4 sx5 sy5 sx6 sy6 sx7 sy7]
        __m256i a = _mm256_unpacklo_epi32(sx, sy);
        __m256i b = _mm256_unpackhi_epi32(sx, sy);
        __m256i lo = _mm256_permute2x128_si256(a, b, 0x20);
        __m256i hi = _mm256_permute2x128_si256(a, b, 0x31);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(base + 0), lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(base + 8), hi);
    };

    auto scalar_pixel = [&](int x, int y) {
        const int idx = y * W + x;
        const int base = 2 * idx;
        int best_sx = xy_in[base + 0];
        int best_sy = xy_in[base + 1];
        int best_d  = sq_dist_i32(x, y, best_sx, best_sy);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx * step;
                const int ny = y + dy * step;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                const int nidx = ny * W + nx;
                const int nbase = 2 * nidx;
                const int csx = xy_in[nbase + 0];
                if (csx == INVALID_COORD) continue;
                const int csy = xy_in[nbase + 1];
                const int d = sq_dist_i32(x, y, csx, csy);
                if (d < best_d) {
                    best_d = d;
                    best_sx = csx;
                    best_sy = csy;
                }
            }
        }

        xy_out[base + 0] = best_sx;
        xy_out[base + 1] = best_sy;
    };

    if constexpr (UseOpenMP) {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; ++y) {
            const bool y_interior = (y >= step) && (y < H - step);
            if (!y_interior || W < 8 || step <= 0) {
                for (int x = 0; x < W; ++x) scalar_pixel(x, y);
                continue;
            }

            for (int x = 0; x < step; ++x) scalar_pixel(x, y);

            int x = step;
            const int x_vec_end = (W - step) - 8;
            const __m256i yv = _mm256_set1_epi32(y);

            for (; x <= x_vec_end; x += 8) {
                const int idx0 = y * W + x;
                const int base0 = 2 * idx0;
                const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

                __m256i best_sx, best_sy;
                load_xy8(xy_in + base0, best_sx, best_sy);
                __m256i best_d = dist_sq_vec(xv, yv, best_sx, best_sy);

                // dy=-1 (dx=-1,0,1)
                {
                    const int ny = y - step;
                    const int row_base = 2 * (ny * W);
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx * step;
                        const int nbase0 = row_base + 2 * nx;
                        __m256i csx, csy;
                        load_xy8(xy_in + nbase0, csx, csy);
                        __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                        update_best(best_d, best_sx, best_sy, cd, csx, csy);
                    }
                }

                // dy=0 (dx=-1,+1)
                {
                    const int ny = y;
                    const int row_base = 2 * (ny * W);
                    for (int dx : {-1, 1}) {
                        const int nx = x + dx * step;
                        const int nbase0 = row_base + 2 * nx;
                        __m256i csx, csy;
                        load_xy8(xy_in + nbase0, csx, csy);
                        __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                        update_best(best_d, best_sx, best_sy, cd, csx, csy);
                    }
                }

                // dy=+1 (dx=-1,0,1)
                {
                    const int ny = y + step;
                    const int row_base = 2 * (ny * W);
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx * step;
                        const int nbase0 = row_base + 2 * nx;
                        __m256i csx, csy;
                        load_xy8(xy_in + nbase0, csx, csy);
                        __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                        update_best(best_d, best_sx, best_sy, cd, csx, csy);
                    }
                }

                store_xy8(xy_out + base0, best_sx, best_sy);
            }

            for (; x < W - step; ++x) scalar_pixel(x, y);
            for (int xb = W - step; xb < W; ++xb) scalar_pixel(xb, y);
        }
        return;
    }

    for (int y = 0; y < H; ++y) {
        const bool y_interior = (y >= step) && (y < H - step);
        if (!y_interior || W < 8 || step <= 0) {
            for (int x = 0; x < W; ++x) scalar_pixel(x, y);
            continue;
        }

        for (int x = 0; x < step; ++x) scalar_pixel(x, y);

        int x = step;
        const int x_vec_end = (W - step) - 8;
        const __m256i yv = _mm256_set1_epi32(y);

        for (; x <= x_vec_end; x += 8) {
            const int idx0 = y * W + x;
            const int base0 = 2 * idx0;
            const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

            __m256i best_sx, best_sy;
            load_xy8(xy_in + base0, best_sx, best_sy);
            __m256i best_d = dist_sq_vec(xv, yv, best_sx, best_sy);

            // dy=-1 (dx=-1,0,1)
            {
                const int row_base = 2 * ((y - step) * W);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nbase0 = row_base + 2 * (x + dx * step);
                    __m256i csx, csy;
                    load_xy8(xy_in + nbase0, csx, csy);
                    __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            // dy=0 (dx=-1,+1)
            {
                const int row_base = 2 * (y * W);
                for (int dx : {-1, 1}) {
                    const int nbase0 = row_base + 2 * (x + dx * step);
                    __m256i csx, csy;
                    load_xy8(xy_in + nbase0, csx, csy);
                    __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            // dy=+1 (dx=-1,0,1)
            {
                const int row_base = 2 * ((y + step) * W);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nbase0 = row_base + 2 * (x + dx * step);
                    __m256i csx, csy;
                    load_xy8(xy_in + nbase0, csx, csy);
                    __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            store_xy8(xy_out + base0, best_sx, best_sy);
        }

        for (; x < W - step; ++x) scalar_pixel(x, y);
        for (int xb = W - step; xb < W; ++xb) scalar_pixel(xb, y);
    }
}

inline void coord_step_aos_avx2_omp_for(const Config& cfg,
                                       const int* xy_in, // length 2*N
                                       int* xy_out,      // length 2*N
                                       int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    const __m256i inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i even_idx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i odd_idx  = _mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);

    auto load_xy8 = [&](const int* base, __m256i& sx, __m256i& sy) {
        sx = _mm256_i32gather_epi32(base, even_idx, 4);
        sy = _mm256_i32gather_epi32(base, odd_idx, 4);
    };

    auto store_xy8 = [&](int* base, __m256i sx, __m256i sy) {
        __m256i a = _mm256_unpacklo_epi32(sx, sy);
        __m256i b = _mm256_unpackhi_epi32(sx, sy);
        __m256i lo = _mm256_permute2x128_si256(a, b, 0x20);
        __m256i hi = _mm256_permute2x128_si256(a, b, 0x31);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(base + 0), lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(base + 8), hi);
    };

    auto scalar_pixel = [&](int x, int y) {
        const int idx = y * W + x;
        const int base = 2 * idx;
        int best_sx = xy_in[base + 0];
        int best_sy = xy_in[base + 1];
        int best_d  = sq_dist_i32(x, y, best_sx, best_sy);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx * step;
                const int ny = y + dy * step;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                const int nidx = ny * W + nx;
                const int nbase = 2 * nidx;
                const int csx = xy_in[nbase + 0];
                if (csx == INVALID_COORD) continue;
                const int csy = xy_in[nbase + 1];
                const int d = sq_dist_i32(x, y, csx, csy);
                if (d < best_d) {
                    best_d = d;
                    best_sx = csx;
                    best_sy = csy;
                }
            }
        }

        xy_out[base + 0] = best_sx;
        xy_out[base + 1] = best_sy;
    };

    #pragma omp for schedule(static)
    for (int y = 0; y < H; ++y) {
        const bool y_interior = (y >= step) && (y < H - step);
        if (!y_interior || W < 8 || step <= 0) {
            for (int x = 0; x < W; ++x) scalar_pixel(x, y);
            continue;
        }

        for (int x = 0; x < step; ++x) scalar_pixel(x, y);

        int x = step;
        const int x_vec_end = (W - step) - 8;
        const __m256i yv = _mm256_set1_epi32(y);

        for (; x <= x_vec_end; x += 8) {
            const int idx0 = y * W + x;
            const int base0 = 2 * idx0;
            const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

            __m256i best_sx, best_sy;
            load_xy8(xy_in + base0, best_sx, best_sy);
            __m256i best_d = dist_sq_vec(xv, yv, best_sx, best_sy);

            // dy=-1: dx=-1,0,1
            {
                const int row_base = 2 * ((y - step) * W);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nbase0 = row_base + 2 * (x + dx * step);
                    __m256i csx, csy;
                    load_xy8(xy_in + nbase0, csx, csy);
                    __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            // dy=0: dx=-1,+1
            {
                const int row_base = 2 * (y * W);
                for (int dx : {-1, 1}) {
                    const int nbase0 = row_base + 2 * (x + dx * step);
                    __m256i csx, csy;
                    load_xy8(xy_in + nbase0, csx, csy);
                    __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            // dy=+1: dx=-1,0,1
            {
                const int row_base = 2 * ((y + step) * W);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nbase0 = row_base + 2 * (x + dx * step);
                    __m256i csx, csy;
                    load_xy8(xy_in + nbase0, csx, csy);
                    __m256i cd = dist_sq_vec(xv, yv, csx, csy);
                    update_best(best_d, best_sx, best_sy, cd, csx, csy);
                }
            }

            store_xy8(xy_out + base0, best_sx, best_sy);
        }

        for (; x < W - step; ++x) scalar_pixel(x, y);
        for (int xb = W - step; xb < W; ++xb) scalar_pixel(xb, y);
    }
}

// --------------------------
// AVX2 index-based kernels
// --------------------------

inline void idx_step_avx2(const Config& cfg,
                          const std::vector<Seed>& seeds,
                          const SeedGather& gather,
                          const SeedIndexBuffer& in_buf,
                          SeedIndexBuffer& out_buf,
                          int step)
{
    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;
    out_buf.resize(N);

    const __m256i inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i neg1 = _mm256_set1_epi32(-1);
    const __m256i inf  = _mm256_set1_epi32(DIST_INF);

    auto scalar_pixel = [&](int x, int y) {
        const int idx = y * W + x;
        int best_seed = in_buf[idx];
        int best_d = sq_dist_seed_scalar(x, y, best_seed, seeds);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx * step;
                const int ny = y + dy * step;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                const int nidx = ny * W + nx;
                const int cand_seed = in_buf[nidx];
                if (cand_seed < 0) continue;
                const int d = sq_dist_seed_scalar(x, y, cand_seed, seeds);
                if (d < best_d) {
                    best_d = d;
                    best_seed = cand_seed;
                }
            }
        }
        out_buf[idx] = best_seed;
    };

    auto vector_pixel_safe = [&](int& x, int x_end, int y) {
        const int row = y * W;
        const __m256i yv = _mm256_set1_epi32(y);
        const __m256i W_vec = _mm256_set1_epi32(W);

        for (; x <= x_end; x += 8) {
            const int idx0 = row + x;
            const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

            __m256i best_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + idx0));
            __m256i best_invalid = _mm256_cmpeq_epi32(best_idx, neg1);
            __m256i best_d;
            if (_mm256_movemask_epi8(best_invalid) == -1) {
                best_d = inf;
            } else {
                __m256i best_sx, best_sy;
                gather.gather_xy_known_mask(best_idx, best_invalid, best_sx, best_sy);
                best_d = dist_sq_vec_seed(xv, yv, best_sx, best_sy, best_invalid);
            }

            for (int dy = -1; dy <= 1; ++dy) {
                int ny_s = y + dy * step;
                __m256i mask_y = (ny_s >= 0 && ny_s < H) ? _mm256_set1_epi32(-1) : _mm256_setzero_si256();
                if (_mm256_movemask_epi8(mask_y) == 0) continue;

                const int base = ny_s * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dy == 0 && dx == 0) continue;

                    __m256i nx = _mm256_add_epi32(xv, _mm256_set1_epi32(dx * step));
                    __m256i mask_x = _mm256_and_si256(_mm256_cmpgt_epi32(nx, neg1), _mm256_cmpgt_epi32(W_vec, nx));
                    __m256i mask = _mm256_and_si256(mask_x, mask_y);

                    if (_mm256_movemask_epi8(mask) == 0) continue;

                    const int offset = base + x + dx * step;
                    const int* addr = in_buf.data() + offset;
                    
                    __m256i cand_idx = _mm256_maskload_epi32(addr, mask);
                    cand_idx = _mm256_blendv_epi8(neg1, cand_idx, mask);

                    __m256i inv = _mm256_cmpeq_epi32(cand_idx, neg1);
                    if (_mm256_movemask_epi8(inv) != -1) {
                        __m256i csx, csy;
                        gather.gather_xy_known_mask(cand_idx, inv, csx, csy);
                        __m256i cd = dist_sq_vec_seed(xv, yv, csx, csy, inv);
                        update_best_idx(best_d, best_idx, cd, cand_idx);
                    }
                }
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_buf.data() + idx0), best_idx);
        }
    };

    auto vector_pixel_fast = [&](int& x, int x_end, int y) {
        const int row = y * W;
        const __m256i yv = _mm256_set1_epi32(y);

        for (; x <= x_end; x += 8) {
            const int idx0 = row + x;
            const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

            __m256i best_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + idx0));
            __m256i best_invalid = _mm256_cmpeq_epi32(best_idx, neg1);
            __m256i best_d;
            if (_mm256_movemask_epi8(best_invalid) == -1) {
                best_d = inf;
            } else {
                __m256i best_sx, best_sy;
                gather.gather_xy_known_mask(best_idx, best_invalid, best_sx, best_sy);
                best_d = dist_sq_vec_seed(xv, yv, best_sx, best_sy, best_invalid);
            }

            // dy=-1
            {
                const int base = (y - step) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i cand_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + nidx0));
                    __m256i inv = _mm256_cmpeq_epi32(cand_idx, neg1);
                    if (_mm256_movemask_epi8(inv) != -1) {
                        __m256i csx, csy;
                        gather.gather_xy_known_mask(cand_idx, inv, csx, csy);
                        __m256i cd = dist_sq_vec_seed(xv, yv, csx, csy, inv);
                        update_best_idx(best_d, best_idx, cd, cand_idx);
                    }
                }
            }
            // dy=0
            {
                const int base = y * W;
                for (int dx : {-1, 1}) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i cand_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + nidx0));
                    __m256i inv = _mm256_cmpeq_epi32(cand_idx, neg1);
                    if (_mm256_movemask_epi8(inv) != -1) {
                        __m256i csx, csy;
                        gather.gather_xy_known_mask(cand_idx, inv, csx, csy);
                        __m256i cd = dist_sq_vec_seed(xv, yv, csx, csy, inv);
                        update_best_idx(best_d, best_idx, cd, cand_idx);
                    }
                }
            }
            // dy=+1
            {
                const int base = (y + step) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i cand_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + nidx0));
                    __m256i inv = _mm256_cmpeq_epi32(cand_idx, neg1);
                    if (_mm256_movemask_epi8(inv) != -1) {
                        __m256i csx, csy;
                        gather.gather_xy_known_mask(cand_idx, inv, csx, csy);
                        __m256i cd = dist_sq_vec_seed(xv, yv, csx, csy, inv);
                        update_best_idx(best_d, best_idx, cd, cand_idx);
                    }
                }
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_buf.data() + idx0), best_idx);
        }
    };

    for (int y = 0; y < H; ++y) {
        int x = 0;
        bool y_safe = (y >= step && y < H - step);
        int safe_x_end = (W - step) - 8;
        
        // 1. Left / Generic Unsafe
        int end_left = std::min(W - 8, step);
        if (end_left >= x) {
             vector_pixel_safe(x, end_left, y);
        }
        
        // 2. Center Safe
        if (y_safe) {
             if (safe_x_end >= x) {
                 vector_pixel_fast(x, safe_x_end, y);
             }
        }
        
        // 3. Right Unsafe
        int end_right = W - 8;
        if (end_right >= x) {
             vector_pixel_safe(x, end_right, y);
        }
        
        // 4. Tail Scalar
        for (; x < W; ++x) scalar_pixel(x, y);
    }
}

// Variant intended to run inside an existing OpenMP parallel region.
// Uses `#pragma omp for` to avoid per-pass team creation overhead.
inline void idx_step_avx2_omp_for(const Config& cfg,
                                 const std::vector<Seed>& seeds,
                                 const SeedGather& gather,
                                 const SeedIndexBuffer& in_buf,
                                 SeedIndexBuffer& out_buf,
                                 int step)
{
    const int W = cfg.width;
    const int H = cfg.height;

    const __m256i inc = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i neg1 = _mm256_set1_epi32(-1);
    const __m256i inf  = _mm256_set1_epi32(DIST_INF);

    auto scalar_pixel = [&](int x, int y) {
        const int idx = y * W + x;
        int best_seed = in_buf[idx];
        int best_d = sq_dist_seed_scalar(x, y, best_seed, seeds);

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx * step;
                const int ny = y + dy * step;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                const int nidx = ny * W + nx;
                const int cand_seed = in_buf[nidx];
                if (cand_seed < 0) continue;
                const int d = sq_dist_seed_scalar(x, y, cand_seed, seeds);
                if (d < best_d) {
                    best_d = d;
                    best_seed = cand_seed;
                }
            }
        }
        out_buf[idx] = best_seed;
    };

    #pragma omp for schedule(static)
    for (int y = 0; y < H; ++y) {
        const bool y_interior = (y >= step) && (y < H - step);
        const int row = y * W;

        if (!y_interior || W < 8 || step <= 0) {
            for (int x = 0; x < W; ++x) scalar_pixel(x, y);
            continue;
        }

        for (int x = 0; x < step; ++x) scalar_pixel(x, y);

        int x = step;
        const int x_vec_end = (W - step) - 8;
        const __m256i yv = _mm256_set1_epi32(y);

        for (; x <= x_vec_end; x += 8) {
            const int idx0 = row + x;
            const __m256i xv = _mm256_add_epi32(_mm256_set1_epi32(x), inc);

            __m256i best_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + idx0));
            __m256i best_invalid = _mm256_cmpeq_epi32(best_idx, neg1);
            __m256i best_d;
            if (_mm256_movemask_epi8(best_invalid) == -1) {
                best_d = inf;
            } else {
                __m256i best_sx, best_sy;
                gather.gather_xy_known_mask(best_idx, best_invalid, best_sx, best_sy);
                best_d = dist_sq_vec_seed(xv, yv, best_sx, best_sy, best_invalid);
            }

            // dy=-1: dx=-1,0,1
            {
                const int base = (y - step) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i cand_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + nidx0));
                    __m256i inv = _mm256_cmpeq_epi32(cand_idx, neg1);
                    if (_mm256_movemask_epi8(inv) != -1) {
                        __m256i csx, csy;
                        gather.gather_xy_known_mask(cand_idx, inv, csx, csy);
                        __m256i cd = dist_sq_vec_seed(xv, yv, csx, csy, inv);
                        update_best_idx(best_d, best_idx, cd, cand_idx);
                    }
                }
            }

            // dy=0: dx=-1,+1
            {
                const int base = y * W;
                for (int dx : {-1, 1}) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i cand_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + nidx0));
                    __m256i inv = _mm256_cmpeq_epi32(cand_idx, neg1);
                    if (_mm256_movemask_epi8(inv) != -1) {
                        __m256i csx, csy;
                        gather.gather_xy_known_mask(cand_idx, inv, csx, csy);
                        __m256i cd = dist_sq_vec_seed(xv, yv, csx, csy, inv);
                        update_best_idx(best_d, best_idx, cd, cand_idx);
                    }
                }
            }

            // dy=+1: dx=-1,0,1
            {
                const int base = (y + step) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx0 = base + (x + dx * step);
                    __m256i cand_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_buf.data() + nidx0));
                    __m256i inv = _mm256_cmpeq_epi32(cand_idx, neg1);
                    if (_mm256_movemask_epi8(inv) != -1) {
                        __m256i csx, csy;
                        gather.gather_xy_known_mask(cand_idx, inv, csx, csy);
                        __m256i cd = dist_sq_vec_seed(xv, yv, csx, csy, inv);
                        update_best_idx(best_d, best_idx, cd, cand_idx);
                    }
                }
            }

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_buf.data() + idx0), best_idx);
        }

        for (; x < W - step; ++x) scalar_pixel(x, y);
        for (int xb = W - step; xb < W; ++xb) scalar_pixel(xb, y);
    }
}

inline void coordbuf_to_seed_indices_soa(const Config& cfg,
                                        const std::vector<int>& sx,
                                        const std::vector<int>& sy,
                                        const std::vector<int>& id_map,
                                        SeedIndexBuffer& out)
{
    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;
    out.resize(N);
    for (int i = 0; i < N; ++i) {
        const int x = sx[i];
        const int y = sy[i];
        if (x < 0 || y < 0) {
            out[i] = -1;
        } else {
            const int idx = y * W + x;
            out[i] = (idx >= 0 && idx < N) ? id_map[idx] : -1;
        }
    }
}

inline void coordbuf_to_seed_indices_aos(const Config& cfg,
                                        const std::vector<int>& xy,
                                        const std::vector<int>& id_map,
                                        SeedIndexBuffer& out)
{
    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;
    out.resize(N);
    for (int i = 0; i < N; ++i) {
        const int x = xy[2 * i + 0];
        const int y = xy[2 * i + 1];
        if (x < 0 || y < 0) {
            out[i] = -1;
        } else {
            const int idx = y * W + x;
            out[i] = (idx >= 0 && idx < N) ? id_map[idx] : -1;
        }
    }
}

} // namespace

void jfa_cpu_simd(const Config& cfg,
                  const std::vector<Seed>& seeds,
                  SeedIndexBuffer& out_buffer,
                  PassCallback pass_cb)
{
        // Mode selection:
    // - cfg.use_coord_prop == false (default): index-based JFA (per-pixel seed index)
    // - cfg.use_coord_prop == true           : coordinate propagation (per-pixel seed coordinates)
    
    if (!cfg.use_coord_prop) {
        const int W = cfg.width;
        const int H = cfg.height;
        const int N = W * H;

        SeedIndexBuffer bufA(N, -1);
        SeedIndexBuffer bufB(N, -1);

        // Init: seed pixels contain their seed index (later seeds overwrite earlier ones).
        for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
            const auto& s = seeds[i];
            if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
            bufA[s.y * W + s.x] = i;
        }

        // Prepare gatherer based on cfg.cpu_seeds_layout (CPU SIMD only; for experiments).
        // - Packed: one gather -> unpack x/y in registers
        // - SoA: two gathers
        // - AoS: two gathers (interleaved [x0,y0,x1,y1,...])
        std::vector<int> seeds_packed;
        std::vector<int> seeds_x;
        std::vector<int> seeds_y;
        SeedGather gather;
        if (cfg.cpu_seeds_layout == CpuSeedsLayout::Packed) {
            seeds_packed.resize(seeds.size());
            for (size_t i = 0; i < seeds.size(); ++i) {
                const int x = seeds[i].x;
                const int y = seeds[i].y;
                seeds_packed[i] = (y << 16) | (x & 0xFFFF);
            }
            gather.seeds_packed = seeds_packed.data();
        } else if (cfg.cpu_seeds_layout == CpuSeedsLayout::SoA) {
            seeds_x.resize(seeds.size());
            seeds_y.resize(seeds.size());
            for (size_t i = 0; i < seeds.size(); ++i) {
                seeds_x[i] = seeds[i].x;
                seeds_y[i] = seeds[i].y;
            }
            gather.seeds_x = seeds_x.data();
            gather.seeds_y = seeds_y.data();
        } else { // AoS
            gather.seeds_xy = reinterpret_cast<const int*>(seeds.data());
        }

        int max_dim = std::max(W, H);
        int step = max_dim / 2;
        if (step <= 0) step = 1;

        bool fromA = true;
        int pass_idx = 0;

        while (step >= 1) {
            const SeedIndexBuffer& in = fromA ? bufA : bufB;
            SeedIndexBuffer& out = fromA ? bufB : bufA;

            idx_step_avx2(cfg, seeds, gather, in, out, step);

            if (pass_cb) {
                pass_cb(pass_idx, step, out);
            }

            fromA = !fromA;
            step /= 2;
            ++pass_idx;
        }

        const SeedIndexBuffer& final_buf = fromA ? bufA : bufB;
        out_buffer = final_buf;
        return;
    }

    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;

    // Build an ID map so we can convert propagated coordinates back to seed indices.
    // Matches CPU serial init behavior: later seeds overwrite earlier ones at the same pixel.
    std::vector<int> id_map(N, -1);
    for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
        const auto& s = seeds[i];
        if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
        id_map[s.y * W + s.x] = i;
    }

    // Step schedule
    int max_dim = std::max(W, H);
    int step = max_dim / 2;
    if (step <= 0) step = 1;

    int pass_idx = 0;

    if (cfg.cpu_coordbuf_layout == CpuCoordBufLayout::SoA) {
        std::vector<int> sxA(N, INVALID_COORD), syA(N, INVALID_COORD);
        std::vector<int> sxB(N, INVALID_COORD), syB(N, INVALID_COORD);

        for (const auto& s : seeds) {
            if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
            const int idx = s.y * W + s.x;
            sxA[idx] = s.x;
            syA[idx] = s.y;
        }

        bool fromA = true;
        SeedIndexBuffer tmp_indices;

        while (step >= 1) {
            const int* sx_in = fromA ? sxA.data() : sxB.data();
            const int* sy_in = fromA ? syA.data() : syB.data();
            int* sx_out = fromA ? sxB.data() : sxA.data();
            int* sy_out = fromA ? syB.data() : syA.data();

            coord_step_soa_avx2<false>(cfg, sx_in, sy_in, sx_out, sy_out, step);

            if (pass_cb) {
                const auto& sx_cur = fromA ? sxB : sxA;
                const auto& sy_cur = fromA ? syB : syA;
                coordbuf_to_seed_indices_soa(cfg, sx_cur, sy_cur, id_map, tmp_indices);
                pass_cb(pass_idx, step, tmp_indices);
            }

            fromA = !fromA;
            step /= 2;
            ++pass_idx;
        }

        const auto& sx_final = fromA ? sxA : sxB;
        const auto& sy_final = fromA ? syA : syB;
        coordbuf_to_seed_indices_soa(cfg, sx_final, sy_final, id_map, out_buffer);
        return;
    }

    // AoS coord buffer: xy buffers interleaved [sx,sy,...] length 2*N
    std::vector<int> xyA(2 * N, INVALID_COORD);
    std::vector<int> xyB(2 * N, INVALID_COORD);

    for (const auto& s : seeds) {
        if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
        const int idx = s.y * W + s.x;
        xyA[2 * idx + 0] = s.x;
        xyA[2 * idx + 1] = s.y;
    }

    bool fromA = true;
    SeedIndexBuffer tmp_indices;

    while (step >= 1) {
        const int* in = fromA ? xyA.data() : xyB.data();
        int* out = fromA ? xyB.data() : xyA.data();

        coord_step_aos_avx2<false>(cfg, in, out, step);

        if (pass_cb) {
            const auto& cur = fromA ? xyB : xyA;
            coordbuf_to_seed_indices_aos(cfg, cur, id_map, tmp_indices);
            pass_cb(pass_idx, step, tmp_indices);
        }

        fromA = !fromA;
        step /= 2;
        ++pass_idx;
    }

    const auto& final_xy = fromA ? xyA : xyB;
    coordbuf_to_seed_indices_aos(cfg, final_xy, id_map, out_buffer);
}

void jfa_cpu_omp_simd(const Config& cfg,
                      const std::vector<Seed>& seeds,
                      SeedIndexBuffer& out_buffer,
                      int num_threads,
                      PassCallback pass_cb)
{
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // Same implementation as jfa_cpu_simd, but the step kernels run with OpenMP over rows.
    const int W = cfg.width;
    const int H = cfg.height;
    const int N = W * H;

    if (!cfg.use_coord_prop) {
        // Index-based JFA (default): store per-pixel seed index, gather seed coordinates via AVX2.
        SeedIndexBuffer bufA(N, -1);
        SeedIndexBuffer bufB(N, -1);

        for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
            const auto& s = seeds[i];
            if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
            bufA[s.y * W + s.x] = i;
        }

        std::vector<int> steps;
        {
            int max_dim = std::max(W, H);
            int step = max_dim / 2;
            if (step <= 0) step = 1;
            while (step >= 1) {
                steps.push_back(step);
                step /= 2;
            }
        }

        // Prepare gatherer based on cfg.cpu_seeds_layout (CPU SIMD only; for experiments).
        std::vector<int> seeds_packed;
        std::vector<int> seeds_x;
        std::vector<int> seeds_y;
        SeedGather gather;
        if (cfg.cpu_seeds_layout == CpuSeedsLayout::Packed) {
            seeds_packed.resize(seeds.size());
            for (size_t i = 0; i < seeds.size(); ++i) {
                const int x = seeds[i].x;
                const int y = seeds[i].y;
                seeds_packed[i] = (y << 16) | (x & 0xFFFF);
            }
            gather.seeds_packed = seeds_packed.data();
        } else if (cfg.cpu_seeds_layout == CpuSeedsLayout::SoA) {
            seeds_x.resize(seeds.size());
            seeds_y.resize(seeds.size());
            for (size_t i = 0; i < seeds.size(); ++i) {
                seeds_x[i] = seeds[i].x;
                seeds_y[i] = seeds[i].y;
            }
            gather.seeds_x = seeds_x.data();
            gather.seeds_y = seeds_y.data();
        } else { // AoS
            gather.seeds_xy = reinterpret_cast<const int*>(seeds.data());
        }

        #pragma omp parallel
        {
            for (int pass_idx = 0; pass_idx < static_cast<int>(steps.size()); ++pass_idx) {
                const int step = steps[pass_idx];
                const bool fromA = (pass_idx % 2 == 0);
                const SeedIndexBuffer& in = fromA ? bufA : bufB;
                SeedIndexBuffer& out = fromA ? bufB : bufA;

                idx_step_avx2_omp_for(cfg, seeds, gather, in, out, step);

                #pragma omp single
                {
                    if (pass_cb) {
                        pass_cb(pass_idx, step, out);
                    }
                }
                #pragma omp barrier
            }
        }

        const bool final_fromA = (steps.size() % 2 == 0);
        const SeedIndexBuffer& final_buf = final_fromA ? bufA : bufB;
        out_buffer = final_buf;
        return;
    }

    std::vector<int> id_map(N, -1);
    for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
        const auto& s = seeds[i];
        if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
        id_map[s.y * W + s.x] = i;
    }

    // Precompute step schedule so we can run with a single OpenMP parallel region.
    std::vector<int> steps;
    {
        int max_dim = std::max(W, H);
        int step = max_dim / 2;
        if (step <= 0) step = 1;
        while (step >= 1) {
            steps.push_back(step);
            step /= 2;
        }
    }

    if (cfg.cpu_coordbuf_layout == CpuCoordBufLayout::SoA) {
        std::vector<int> sxA(N, INVALID_COORD), syA(N, INVALID_COORD);
        std::vector<int> sxB(N, INVALID_COORD), syB(N, INVALID_COORD);

        for (const auto& s : seeds) {
            if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
            const int idx = s.y * W + s.x;
            sxA[idx] = s.x;
            syA[idx] = s.y;
        }

        SeedIndexBuffer tmp_indices;

        #pragma omp parallel
        {
            for (int pass_idx = 0; pass_idx < static_cast<int>(steps.size()); ++pass_idx) {
                const int step = steps[pass_idx];
                const bool fromA = (pass_idx % 2 == 0);

                const int* sx_in = fromA ? sxA.data() : sxB.data();
                const int* sy_in = fromA ? syA.data() : syB.data();
                int* sx_out = fromA ? sxB.data() : sxA.data();
                int* sy_out = fromA ? syB.data() : syA.data();

                coord_step_soa_avx2_omp_for(cfg, sx_in, sy_in, sx_out, sy_out, step);

                #pragma omp single
                {
                    if (pass_cb) {
                        const auto& sx_cur = fromA ? sxB : sxA;
                        const auto& sy_cur = fromA ? syB : syA;
                        coordbuf_to_seed_indices_soa(cfg, sx_cur, sy_cur, id_map, tmp_indices);
                        pass_cb(pass_idx, step, tmp_indices);
                    }
                }
                #pragma omp barrier
            }
        }

        const bool final_fromA = (steps.size() % 2 == 0); // even passes => final written to A
        const auto& sx_final = final_fromA ? sxA : sxB;
        const auto& sy_final = final_fromA ? syA : syB;
        coordbuf_to_seed_indices_soa(cfg, sx_final, sy_final, id_map, out_buffer);
        return;
    }

    // AoS coord buffer: xy buffers interleaved [sx,sy,...] length 2*N
    std::vector<int> xyA(2 * N, INVALID_COORD);
    std::vector<int> xyB(2 * N, INVALID_COORD);

    for (const auto& s : seeds) {
        if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H) continue;
        const int idx = s.y * W + s.x;
        xyA[2 * idx + 0] = s.x;
        xyA[2 * idx + 1] = s.y;
    }

    SeedIndexBuffer tmp_indices;

    #pragma omp parallel
    {
        for (int pass_idx = 0; pass_idx < static_cast<int>(steps.size()); ++pass_idx) {
            const int step = steps[pass_idx];
            const bool fromA = (pass_idx % 2 == 0);

            const int* in = fromA ? xyA.data() : xyB.data();
            int* out = fromA ? xyB.data() : xyA.data();

            coord_step_aos_avx2_omp_for(cfg, in, out, step);

            #pragma omp single
            {
                if (pass_cb) {
                    const auto& cur = fromA ? xyB : xyA;
                    coordbuf_to_seed_indices_aos(cfg, cur, id_map, tmp_indices);
                    pass_cb(pass_idx, step, tmp_indices);
                }
            }
            #pragma omp barrier
        }
    }

    const bool final_fromA = (steps.size() % 2 == 0);
    const auto& final_xy = final_fromA ? xyA : xyB;
    coordbuf_to_seed_indices_aos(cfg, final_xy, id_map, out_buffer);
}

} // namespace jfa


