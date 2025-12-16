#include <cuda_runtime.h>
#include <stdio.h>
#include <jfa/types.hpp>
#include <chrono>
#include <iostream>

namespace jfa {

// Simple struct to replace std::vector<Seed> usage in kernel
struct GPUSeed {
    int x, y;
};

// Constant memory for seeds (Max 4096 seeds to fit in 64KB total)
// We need 3 arrays: c_seeds (AoS), c_seeds_x (SoA), c_seeds_y (SoA)
// Total size must be < 64KB (0x10000 bytes)
// Each seed is 8 bytes (int2) or 2 ints.
// If we want to support both modes, we have to split the space.
// Let's reduce MAX_CONST_SEEDS to 2048 to be safe (2048 * 8 * 3 = 48KB < 64KB)
#define MAX_CONST_SEEDS 2048
__constant__ GPUSeed c_seeds[MAX_CONST_SEEDS];
// Constant memory for SoA
__constant__ int c_seeds_x[MAX_CONST_SEEDS];
__constant__ int c_seeds_y[MAX_CONST_SEEDS];

// C-style callback for internal use
// data is host pointer
using InternalCallback = void (*)(int pass_idx, int step, const int* data, int size, void* user_data);

int jfa_gpu_cuda_impl(const Config& cfg,
                       const GPUSeed* seeds, int num_seeds,
                       int* out_buffer, // pre-allocated on host
                       InternalCallback cb,
                       void* user_data);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            return -1; \
        } \
    } while (0)

// Helper device function to process one pixel with pitch
__device__ inline void process_pixel_pitch(int x, int y, int width, int height, int step, 
                                           const int* in_buf, int* out_buf, 
                                           const GPUSeed* seeds, size_t pitch_ints) 
{
    if (x >= width || y >= height) return;

    // Use pitch for row indexing
    int idx = y * pitch_ints + x;
    
    // Current best seed from previous step
    int best_seed_idx = in_buf[idx];
    float best_dist = 1e30f; // Infinity

    if (best_seed_idx != -1) {
        GPUSeed s = seeds[best_seed_idx];
        float dx = (float)(x - s.x);
        float dy = (float)(y - s.y);
        best_dist = dx * dx + dy * dy;
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * pitch_ints + nx;
                int neighbor_seed_idx = in_buf[n_idx];

                if (neighbor_seed_idx != -1) {
                    GPUSeed s = seeds[neighbor_seed_idx];
                    float dist_x = (float)(x - s.x);
                    float dist_y = (float)(y - s.y);
                    float dist_sq = dist_x * dist_x + dist_y * dist_y;

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Kernel to initialize the buffer with -1 (Pitch version)
__global__ void init_buffer_kernel_pitch(int* buffer, int width, int height, int value, size_t pitch_ints) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        buffer[y * pitch_ints + x] = value;
    }
}

// Kernel to place initial seeds into the buffer (Pitch version)
__global__ void place_seeds_kernel_pitch(int* buffer, const GPUSeed* seeds, int num_seeds, int width, int height, size_t pitch_ints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    GPUSeed s = seeds[idx];
    if (s.x >= 0 && s.x < width && s.y >= 0 && s.y < height) {
        atomicMax(&buffer[s.y * pitch_ints + s.x], idx);
    }
}

// Kernel to initialize the buffer with -1
__global__ void init_buffer_kernel(int* buffer, int size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] = value;
    }
}

// Kernel to place initial seeds into the buffer
__global__ void place_seeds_kernel(int* buffer, const GPUSeed* seeds, int num_seeds, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    GPUSeed s = seeds[idx];
    if (s.x >= 0 && s.x < width && s.y >= 0 && s.y < height) {
        // Use atomicMax to ensure deterministic behavior (largest index wins, matching CPU sequential overwrite)
        atomicMax(&buffer[s.y * width + s.x], idx);
    }
}

// Helper device function to process one pixel
__device__ inline void process_pixel(int x, int y, int width, int height, int step, 
                                     const int* in_buf, int* out_buf, 
                                     const GPUSeed* seeds) 
{
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Current best seed from previous step
    int best_seed_idx = in_buf[idx];
    float best_dist = 1e30f; // Infinity

    if (best_seed_idx != -1) {
        GPUSeed s = seeds[best_seed_idx];
        float dx = (float)(x - s.x);
        float dy = (float)(y - s.y);
        best_dist = dx * dx + dy * dy;
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int neighbor_seed_idx = in_buf[n_idx];

                if (neighbor_seed_idx != -1) {
                    GPUSeed s = seeds[neighbor_seed_idx];
                    float dist_x = (float)(x - s.x);
                    float dist_y = (float)(y - s.y);
                    float dist_sq = dist_x * dist_x + dist_y * dist_y;

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Helper device function to process one pixel with intrinsics (No Coord Prop)
__device__ inline void process_pixel_ultimate(int x, int y, int width, int height, int step, 
                                     const int* __restrict__ in_buf, int* __restrict__ out_buf, 
                                     const GPUSeed* __restrict__ seeds) 
{
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Current best seed from previous step
    int best_seed_idx = __ldg(&in_buf[idx]);
    float best_dist = 1e30f; // Infinity

    if (best_seed_idx != -1) {
        // __ldg only supports built-in types. We need to cast or read members individually.
        // Since GPUSeed is just {int x, y}, we can read it as int2 if aligned, or just read members.
        // But seeds array is GPUSeed*.
        // Let's read members manually using __ldg on int* cast.
        const int* seeds_ptr = (const int*)seeds;
        int sx = __ldg(&seeds_ptr[best_seed_idx * 2]);
        int sy = __ldg(&seeds_ptr[best_seed_idx * 2 + 1]);
        float dx = (float)(x - sx);
        float dy = (float)(y - sy);
        best_dist = fmaf(dx, dx, dy * dy);
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int neighbor_seed_idx = __ldg(&in_buf[n_idx]);

                if (neighbor_seed_idx != -1) {
                    const int* seeds_ptr = (const int*)seeds;
                    int sx = __ldg(&seeds_ptr[neighbor_seed_idx * 2]);
                    int sy = __ldg(&seeds_ptr[neighbor_seed_idx * 2 + 1]);
                    float dist_x = (float)(x - sx);
                    float dist_y = (float)(y - sy);
                    float dist_sq = fmaf(dist_x, dist_x, dist_y * dist_y);

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Helper device function to process one pixel with __restrict__
__device__ inline void process_pixel_restrict(int x, int y, int width, int height, int step, 
                                     const int* __restrict__ in_buf, int* __restrict__ out_buf, 
                                     const GPUSeed* __restrict__ seeds) 
{
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Current best seed from previous step
    int best_seed_idx = in_buf[idx];
    float best_dist = 1e30f; // Infinity

    if (best_seed_idx != -1) {
        GPUSeed s = seeds[best_seed_idx];
        float dx = (float)(x - s.x);
        float dy = (float)(y - s.y);
        best_dist = dx * dx + dy * dy;
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int neighbor_seed_idx = in_buf[n_idx];

                if (neighbor_seed_idx != -1) {
                    GPUSeed s = seeds[neighbor_seed_idx];
                    float dist_x = (float)(x - s.x);
                    float dist_y = (float)(y - s.y);
                    float dist_sq = dist_x * dist_x + dist_y * dist_y;

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Helper device function to process one pixel with __ldg
__device__ inline void process_pixel_ldg(int x, int y, int width, int height, int step, 
                                     const int* in_buf, int* out_buf, 
                                     const GPUSeed* seeds) 
{
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Current best seed from previous step
    int best_seed_idx = __ldg(&in_buf[idx]);
    float best_dist = 1e30f; // Infinity

    if (best_seed_idx != -1) {
        const int* seeds_ptr = (const int*)seeds;
        int sx = __ldg(&seeds_ptr[best_seed_idx * 2]);
        int sy = __ldg(&seeds_ptr[best_seed_idx * 2 + 1]);
        float dx = (float)(x - sx);
        float dy = (float)(y - sy);
        best_dist = dx * dx + dy * dy;
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int neighbor_seed_idx = __ldg(&in_buf[n_idx]);

                if (neighbor_seed_idx != -1) {
                    const int* seeds_ptr = (const int*)seeds;
                    int sx = __ldg(&seeds_ptr[neighbor_seed_idx * 2]);
                    int sy = __ldg(&seeds_ptr[neighbor_seed_idx * 2 + 1]);
                    float dist_x = (float)(x - sx);
                    float dist_y = (float)(y - sy);
                    float dist_sq = dist_x * dist_x + dist_y * dist_y;

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Helper device function to process one pixel with fmaf
__device__ inline void process_pixel_fma(int x, int y, int width, int height, int step, 
                                     const int* in_buf, int* out_buf, 
                                     const GPUSeed* seeds) 
{
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Current best seed from previous step
    int best_seed_idx = in_buf[idx];
    float best_dist = 1e30f; // Infinity

    if (best_seed_idx != -1) {
        GPUSeed s = seeds[best_seed_idx];
        float dx = (float)(x - s.x);
        float dy = (float)(y - s.y);
        best_dist = fmaf(dx, dx, dy * dy);
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int neighbor_seed_idx = in_buf[n_idx];

                if (neighbor_seed_idx != -1) {
                    GPUSeed s = seeds[neighbor_seed_idx];
                    float dist_x = (float)(x - s.x);
                    float dist_y = (float)(y - s.y);
                    float dist_sq = fmaf(dist_x, dist_x, dist_y * dist_y);

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Helper device function to process one pixel (SoA version)
__device__ inline void process_pixel_soa(int x, int y, int width, int height, int step, 
                                     const int* in_buf, int* out_buf, 
                                     const int* seeds_x, const int* seeds_y) 
{
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Current best seed from previous step
    int best_seed_idx = in_buf[idx];
    float best_dist = 1e30f; // Infinity

    if (best_seed_idx != -1) {
        float dx = (float)(x - seeds_x[best_seed_idx]);
        float dy = (float)(y - seeds_y[best_seed_idx]);
        best_dist = dx * dx + dy * dy;
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int neighbor_seed_idx = in_buf[n_idx];

                if (neighbor_seed_idx != -1) {
                    float dist_x = (float)(x - seeds_x[neighbor_seed_idx]);
                    float dist_y = (float)(y - seeds_y[neighbor_seed_idx]);
                    float dist_sq = dist_x * dist_x + dist_y * dist_y;

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Helper device function to process one pixel with pitch (SoA version)
__device__ inline void process_pixel_pitch_soa(int x, int y, int width, int height, int step, 
                                           const int* in_buf, int* out_buf, 
                                           const int* seeds_x, const int* seeds_y, size_t pitch_ints) 
{
    if (x >= width || y >= height) return;

    // Use pitch for row indexing
    int idx = y * pitch_ints + x;
    
    // Current best seed from previous step
    int best_seed_idx = in_buf[idx];
    float best_dist = 1e30f; // Infinity

    if (best_seed_idx != -1) {
        float dx = (float)(x - seeds_x[best_seed_idx]);
        float dy = (float)(y - seeds_y[best_seed_idx]);
        best_dist = dx * dx + dy * dy;
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * pitch_ints + nx;
                int neighbor_seed_idx = in_buf[n_idx];

                if (neighbor_seed_idx != -1) {
                    float dist_x = (float)(x - seeds_x[neighbor_seed_idx]);
                    float dist_y = (float)(y - seeds_y[neighbor_seed_idx]);
                    float dist_sq = dist_x * dist_x + dist_y * dist_y;

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Kernel to place initial seeds into the buffer (SoA version)
__global__ void place_seeds_kernel_soa(int* buffer, const int* seeds_x, const int* seeds_y, int num_seeds, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int sx = seeds_x[idx];
    int sy = seeds_y[idx];
    if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
        atomicMax(&buffer[sy * width + sx], idx);
    }
}

// Kernel to place initial seeds into the buffer (Pitch + SoA version)
__global__ void place_seeds_kernel_pitch_soa(int* buffer, const int* seeds_x, const int* seeds_y, int num_seeds, int width, int height, size_t pitch_ints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int sx = seeds_x[idx];
    int sy = seeds_y[idx];
    if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
        atomicMax(&buffer[sy * pitch_ints + sx], idx);
    }
}

// --- Coordinate Propagation Kernels ---

__global__ void init_buffer_kernel_coord(short2* buffer, int size, short2 value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] = value;
    }
}

__global__ void place_seeds_kernel_coord(short2* buffer, const GPUSeed* seeds, int num_seeds, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    GPUSeed s = seeds[idx];
    if (s.x >= 0 && s.x < width && s.y >= 0 && s.y < height) {
        // No race condition here if seeds are unique locations.
        // If multiple seeds at same location, last one wins (arbitrary).
        buffer[s.y * width + s.x] = make_short2((short)s.x, (short)s.y);
    }
}

__global__ void jfa_step_kernel_int_math(const short2* __restrict__ in_buf, 
                                         short2* __restrict__ out_buf,
                                         int width, int height, int step, int ppt)
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * ppt;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < ppt; ++i) {
        int x = start_x + i;
        if (x >= width || y >= height) continue;

        int idx = y * width + x;
        short2 best_seed = __ldg(&in_buf[idx]);
        int best_dist = 2000000000; // Max int approx

        if (best_seed.x != -1) {
            int dx = x - best_seed.x;
            int dy = y - best_seed.y;
            best_dist = dx * dx + dy * dy;
        }

        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;

                int nx = x + dx * step;
                int ny = y + dy * step;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int n_idx = ny * width + nx;
                    short2 neighbor_seed = __ldg(&in_buf[n_idx]);

                    if (neighbor_seed.x != -1) {
                        int dist_x = x - neighbor_seed.x;
                        int dist_y = y - neighbor_seed.y;
                        int dist_sq = dist_x * dist_x + dist_y * dist_y;

                        if (dist_sq < best_dist) {
                            best_dist = dist_sq;
                            best_seed = neighbor_seed;
                        }
                    }
                }
            }
        }
        out_buf[idx] = best_seed;
    }
}

__global__ void jfa_step_kernel_manhattan(const short2* __restrict__ in_buf, 
                                          short2* __restrict__ out_buf,
                                          int width, int height, int step, int ppt)
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * ppt;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < ppt; ++i) {
        int x = start_x + i;
        if (x >= width || y >= height) continue;

        int idx = y * width + x;
        short2 best_seed = __ldg(&in_buf[idx]);
        int best_dist = 2000000000;

        if (best_seed.x != -1) {
            // __sad(a, b, c) = |a - b| + c
            best_dist = __sad(x, best_seed.x, __sad(y, best_seed.y, 0));
        }

        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;

                int nx = x + dx * step;
                int ny = y + dy * step;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int n_idx = ny * width + nx;
                    short2 neighbor_seed = __ldg(&in_buf[n_idx]);

                    if (neighbor_seed.x != -1) {
                        int dist = __sad(x, neighbor_seed.x, __sad(y, neighbor_seed.y, 0));

                        if (dist < best_dist) {
                            best_dist = dist;
                            best_seed = neighbor_seed;
                        }
                    }
                }
            }
        }
        out_buf[idx] = best_seed;
    }
}

__global__ void jfa_step_kernel_ultimate(const short2* __restrict__ in_buf, 
                                         short2* __restrict__ out_buf,
                                         int width, int height, int step, int ppt)
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * ppt;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < ppt; ++i) {
        int x = start_x + i;
        if (x >= width || y >= height) continue;

        int idx = y * width + x;
        short2 best_seed = __ldg(&in_buf[idx]);
        float best_dist = 1e30f;

        if (best_seed.x != -1) {
            float dx = (float)(x - best_seed.x);
            float dy = (float)(y - best_seed.y);
            best_dist = fmaf(dx, dx, dy * dy);
        }

        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;

                int nx = x + dx * step;
                int ny = y + dy * step;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int n_idx = ny * width + nx;
                    short2 neighbor_seed = __ldg(&in_buf[n_idx]);

                    if (neighbor_seed.x != -1) {
                        float dist_x = (float)(x - neighbor_seed.x);
                        float dist_y = (float)(y - neighbor_seed.y);
                        float dist_sq = fmaf(dist_x, dist_x, dist_y * dist_y);

                        if (dist_sq < best_dist) {
                            best_dist = dist_sq;
                            best_seed = neighbor_seed;
                        }
                    }
                }
            }
        }
        out_buf[idx] = best_seed;
    }
}

__global__ void jfa_step_kernel_coord_ldg(const short2* in_buf, short2* out_buf,
                                      int width, int height, int step, int ppt)
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * ppt;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < ppt; ++i) {
        int x = start_x + i;
        if (x >= width || y >= height) continue;

        int idx = y * width + x;
        short2 best_seed = __ldg(&in_buf[idx]);
        float best_dist = 1e30f;

        if (best_seed.x != -1) {
            float dx = (float)(x - best_seed.x);
            float dy = (float)(y - best_seed.y);
            best_dist = dx * dx + dy * dy;
        }

        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;

                int nx = x + dx * step;
                int ny = y + dy * step;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int n_idx = ny * width + nx;
                    short2 neighbor_seed = __ldg(&in_buf[n_idx]);

                    if (neighbor_seed.x != -1) {
                        float dist_x = (float)(x - neighbor_seed.x);
                        float dist_y = (float)(y - neighbor_seed.y);
                        float dist_sq = dist_x * dist_x + dist_y * dist_y;

                        if (dist_sq < best_dist) {
                            best_dist = dist_sq;
                            best_seed = neighbor_seed;
                        }
                    }
                }
            }
        }
        out_buf[idx] = best_seed;
    }
}

__global__ void jfa_step_kernel_coord(const short2* in_buf, short2* out_buf,
                                      int width, int height, int step, int ppt)
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * ppt;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < ppt; ++i) {
        int x = start_x + i;
        if (x >= width || y >= height) continue;

        int idx = y * width + x;
        short2 best_seed = in_buf[idx];
        float best_dist = 1e30f;

        if (best_seed.x != -1) {
            float dx = (float)(x - best_seed.x);
            float dy = (float)(y - best_seed.y);
            best_dist = dx * dx + dy * dy;
        }

        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue; // Optimization: skip center (already checked)

                int nx = x + dx * step;
                int ny = y + dy * step;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int n_idx = ny * width + nx;
                    short2 neighbor_seed = in_buf[n_idx];

                    if (neighbor_seed.x != -1) {
                        float dist_x = (float)(x - neighbor_seed.x);
                        float dist_y = (float)(y - neighbor_seed.y);
                        float dist_sq = dist_x * dist_x + dist_y * dist_y;

                        if (dist_sq < best_dist) {
                            best_dist = dist_sq;
                            best_seed = neighbor_seed;
                        }
                    }
                }
            }
        }
        out_buf[idx] = best_seed;
    }
}

__global__ void place_seeds_kernel_coord_pitch(short2* buffer, const GPUSeed* seeds, int num_seeds, int width, int height, size_t pitch_elms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    GPUSeed s = seeds[idx];
    if (s.x >= 0 && s.x < width && s.y >= 0 && s.y < height) {
        buffer[s.y * pitch_elms + s.x] = make_short2(s.x, s.y);
    }
}

__global__ void init_id_map_kernel(int* map, int size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        map[idx] = value;
    }
}

__global__ void fill_id_map_kernel(int* map, const GPUSeed* seeds, int num_seeds, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;
    GPUSeed s = seeds[idx];
    if (s.x >= 0 && s.x < width && s.y >= 0 && s.y < height) {
        map[s.y * width + s.x] = idx;
    }
}

__global__ void convert_coord_to_id_kernel_pitch(const short2* coord_buf, int* id_buf, const int* id_map, int width, int height, size_t pitch_elms) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x; // Output is linear
    int p_idx = y * pitch_elms + x; // Input is pitched
    
    short2 s = coord_buf[p_idx];
    if (s.x != -1) {
        id_buf[idx] = id_map[s.y * width + s.x];
    } else {
        id_buf[idx] = -1;
    }
}

__global__ void convert_coord_to_id_kernel(const short2* coord_buf, int* id_buf, const int* id_map, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    short2 s = coord_buf[idx];
    if (s.x != -1) {
        // Lookup ID from the map at the seed's location
        id_buf[idx] = id_map[s.y * width + s.x];
    } else {
        id_buf[idx] = -1;
    }
}

// JFA Step Kernel for Coord Prop with Pitch
__global__ void jfa_step_kernel_coord_pitch(const short2* in_buf, short2* out_buf, 
                                      int width, int height, int step,
                                      int pixels_per_thread, size_t pitch_elms) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        if (x >= width || y >= height) continue;

        int idx = y * pitch_elms + x;
        short2 best_coord = in_buf[idx];
        float best_dist = 1e30f;

        if (best_coord.x != -1) {
            float dx = (float)(x - best_coord.x);
            float dy = (float)(y - best_coord.y);
            best_dist = dx * dx + dy * dy;
        }

        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx * step;
                int ny = y + dy * step;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int n_idx = ny * pitch_elms + nx;
                    short2 neighbor_seed = in_buf[n_idx];

                    if (neighbor_seed.x != -1) {
                        float dist_x = (float)(x - neighbor_seed.x);
                        float dist_y = (float)(y - neighbor_seed.y);
                        float dist_sq = dist_x * dist_x + dist_y * dist_y;

                        if (dist_sq < best_dist) {
                            best_dist = dist_sq;
                            best_coord = neighbor_seed;
                        }
                    }
                }
            }
        }
        out_buf[idx] = best_coord;
    }
}

// JFA Step Kernel with Thread Coarsening
__global__ void jfa_step_kernel(const int* in_buf, int* out_buf, 
                                const GPUSeed* seeds, 
                                int width, int height, int step,
                                int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel(x, y, width, height, step, in_buf, out_buf, seeds);
    }
}

// JFA Step Kernel (SoA version)
__global__ void jfa_step_kernel_soa(const int* in_buf, int* out_buf, 
                                const int* seeds_x, const int* seeds_y, 
                                int width, int height, int step,
                                int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_soa(x, y, width, height, step, in_buf, out_buf, seeds_x, seeds_y);
    }
}

// JFA Step Kernel (SoA + Shared Memory)
__global__ void jfa_step_kernel_soa_shared(const int* in_buf, int* out_buf, 
                                       const int* global_seeds_x,
                                       const int* global_seeds_y,
                                       int width, int height, int step,
                                       int num_seeds,
                                       int pixels_per_thread) 
{
    // Shared memory for seeds: First half is X, second half is Y
    extern __shared__ int s_seeds_soa[];
    int* s_x = s_seeds_soa;
    int* s_y = &s_seeds_soa[num_seeds];

    // Cooperative loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    
    for (int i = tid; i < num_seeds; i += blockSize) {
        s_x[i] = global_seeds_x[i];
        s_y[i] = global_seeds_y[i];
    }
    __syncthreads();

    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        // Reuse the device function, passing shared memory pointers
        process_pixel_soa(x, y, width, height, step, in_buf, out_buf, s_x, s_y);
    }
}

// JFA Step Kernel (SoA + Constant Memory)
__global__ void jfa_step_kernel_soa_constant(const int* in_buf, int* out_buf, 
                                         int width, int height, int step,
                                         int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        // Reuse the device function, passing constant memory pointers
        process_pixel_soa(x, y, width, height, step, in_buf, out_buf, c_seeds_x, c_seeds_y);
    }
}

// JFA Step Kernel using Shared Memory for Seeds
__global__ void jfa_step_kernel_shared(const int* in_buf, int* out_buf, 
                                       const GPUSeed* global_seeds, 
                                       int width, int height, int step,
                                       int num_seeds,
                                       int pixels_per_thread) 
{
    // Shared memory for seeds
    extern __shared__ GPUSeed s_seeds[];

    // Cooperative loading of seeds into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    
    for (int i = tid; i < num_seeds; i += blockSize) {
        s_seeds[i] = global_seeds[i];
    }
    __syncthreads();

    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel(x, y, width, height, step, in_buf, out_buf, s_seeds);
    }
}

// JFA Step Kernel using Constant Memory for Seeds
__global__ void jfa_step_kernel_constant(const int* in_buf, int* out_buf, 
                                         int width, int height, int step,
                                         int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel(x, y, width, height, step, in_buf, out_buf, c_seeds);
    }
}

// JFA Step Kernel with Ultimate Intrinsics (No Coord Prop)
__global__ void jfa_step_kernel_ultimate_no_coord(const int* __restrict__ in_buf, int* __restrict__ out_buf, 
                                const GPUSeed* __restrict__ seeds, 
                                int width, int height, int step,
                                int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_ultimate(x, y, width, height, step, in_buf, out_buf, seeds);
    }
}

__global__ void jfa_step_kernel_restrict(const int* __restrict__ in_buf, int* __restrict__ out_buf, 
                                const GPUSeed* __restrict__ seeds, 
                                int width, int height, int step,
                                int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_restrict(x, y, width, height, step, in_buf, out_buf, seeds);
    }
}

__global__ void jfa_step_kernel_ldg_no_coord(const int* in_buf, int* out_buf, 
                                const GPUSeed* seeds, 
                                int width, int height, int step,
                                int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_ldg(x, y, width, height, step, in_buf, out_buf, seeds);
    }
}

__global__ void jfa_step_kernel_fma_no_coord(const int* in_buf, int* out_buf, 
                                const GPUSeed* seeds, 
                                int width, int height, int step,
                                int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_fma(x, y, width, height, step, in_buf, out_buf, seeds);
    }
}

// Helper device function to process one pixel with Integer Math (No Coord Prop)
__device__ inline void process_pixel_int_math(int x, int y, int width, int height, int step, 
                                     const int* __restrict__ in_buf, int* __restrict__ out_buf, 
                                     const GPUSeed* __restrict__ seeds) 
{
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Current best seed from previous step
    int best_seed_idx = __ldg(&in_buf[idx]);
    int best_dist = 2000000000; // Max int approx

    if (best_seed_idx != -1) {
        const int* seeds_ptr = (const int*)seeds;
        int sx = __ldg(&seeds_ptr[best_seed_idx * 2]);
        int sy = __ldg(&seeds_ptr[best_seed_idx * 2 + 1]);
        int dx = x - sx;
        int dy = y - sy;
        best_dist = dx * dx + dy * dy;
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int neighbor_seed_idx = __ldg(&in_buf[n_idx]);

                if (neighbor_seed_idx != -1) {
                    const int* seeds_ptr = (const int*)seeds;
                    int sx = __ldg(&seeds_ptr[neighbor_seed_idx * 2]);
                    int sy = __ldg(&seeds_ptr[neighbor_seed_idx * 2 + 1]);
                    int dist_x = x - sx;
                    int dist_y = y - sy;
                    int dist_sq = dist_x * dist_x + dist_y * dist_y;

                    if (dist_sq < best_dist) {
                        best_dist = dist_sq;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

// Helper device function to process one pixel with Manhattan Distance (No Coord Prop)
__device__ inline void process_pixel_manhattan(int x, int y, int width, int height, int step, 
                                     const int* __restrict__ in_buf, int* __restrict__ out_buf, 
                                     const GPUSeed* __restrict__ seeds) 
{
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Current best seed from previous step
    int best_seed_idx = __ldg(&in_buf[idx]);
    int best_dist = 2000000000;

    if (best_seed_idx != -1) {
        const int* seeds_ptr = (const int*)seeds;
        int sx = __ldg(&seeds_ptr[best_seed_idx * 2]);
        int sy = __ldg(&seeds_ptr[best_seed_idx * 2 + 1]);
        // __sad(a, b, c) = |a - b| + c
        best_dist = __sad(x, sx, __sad(y, sy, 0));
    }

    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int neighbor_seed_idx = __ldg(&in_buf[n_idx]);

                if (neighbor_seed_idx != -1) {
                    const int* seeds_ptr = (const int*)seeds;
                    int sx = __ldg(&seeds_ptr[neighbor_seed_idx * 2]);
                    int sy = __ldg(&seeds_ptr[neighbor_seed_idx * 2 + 1]);
                    int dist = __sad(x, sx, __sad(y, sy, 0));

                    if (dist < best_dist) {
                        best_dist = dist;
                        best_seed_idx = neighbor_seed_idx;
                    }
                }
            }
        }
    }

    out_buf[idx] = best_seed_idx;
}

__global__ void jfa_step_kernel_int_math_no_coord(const int* __restrict__ in_buf, int* __restrict__ out_buf, 
                                const GPUSeed* __restrict__ seeds, 
                                int width, int height, int step,
                                int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_int_math(x, y, width, height, step, in_buf, out_buf, seeds);
    }
}

__global__ void jfa_step_kernel_manhattan_no_coord(const int* __restrict__ in_buf, int* __restrict__ out_buf, 
                                const GPUSeed* __restrict__ seeds, 
                                int width, int height, int step,
                                int pixels_per_thread) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_manhattan(x, y, width, height, step, in_buf, out_buf, seeds);
    }
}

// JFA Step Kernel with Pitch
__global__ void jfa_step_kernel_pitch(const int* in_buf, int* out_buf, 
                                      const GPUSeed* seeds, 
                                      int width, int height, int step,
                                      int pixels_per_thread, size_t pitch_ints) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_pitch(x, y, width, height, step, in_buf, out_buf, seeds, pitch_ints);
    }
}

// JFA Step Kernel with Pitch (SoA version)
__global__ void jfa_step_kernel_pitch_soa(const int* in_buf, int* out_buf, 
                                      const int* seeds_x, const int* seeds_y, 
                                      int width, int height, int step,
                                      int pixels_per_thread, size_t pitch_ints) 
{
    int start_x = (blockIdx.x * blockDim.x + threadIdx.x) * pixels_per_thread;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < pixels_per_thread; ++i) {
        int x = start_x + i;
        process_pixel_pitch_soa(x, y, width, height, step, in_buf, out_buf, seeds_x, seeds_y, pitch_ints);
    }
}

int jfa_gpu_cuda_impl(const Config& cfg,
                       const GPUSeed* seeds, 
                       const int* seeds_x, const int* seeds_y,
                       int num_seeds,
                       int* out_buffer,
                       InternalCallback cb,
                       void* user_data)
{
    int width = cfg.width;
    int height = cfg.height;
    int block_dim_x = cfg.block_dim_x;
    int block_dim_y = cfg.block_dim_y;
    bool use_shared = cfg.use_shared_mem;
    bool use_constant = cfg.use_constant_mem; // Need to add this to Config

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("CUDA Error: No CUDA-capable device found (error: %s)\n", cudaGetErrorString(err));
        return -1;
    }

    int num_pixels = width * height;
    
    auto t_start = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    int *d_bufA = nullptr;
    int *d_bufB = nullptr;
    short2 *d_coordA = nullptr; // For Coordinate Propagation
    short2 *d_coordB = nullptr; // For Coordinate Propagation
    int *d_id_map = nullptr;    // For Coordinate Propagation ID recovery
    
    GPUSeed* d_seeds = nullptr;
    int* d_seeds_x = nullptr;
    int* d_seeds_y = nullptr;
    size_t pitch_bytes = 0;
    size_t pitch_ints = 0;

    size_t pitch_bytes_coord = 0;
    size_t pitch_elms_coord = 0;

    if (cfg.use_coord_prop) {
        if (cfg.use_pitch) {
             CUDA_CHECK(cudaMallocPitch(&d_coordA, &pitch_bytes_coord, width * sizeof(short2), height));
             CUDA_CHECK(cudaMallocPitch(&d_coordB, &pitch_bytes_coord, width * sizeof(short2), height));
             pitch_elms_coord = pitch_bytes_coord / sizeof(short2);
        } else {
             CUDA_CHECK(cudaMalloc(&d_coordA, num_pixels * sizeof(short2)));
             CUDA_CHECK(cudaMalloc(&d_coordB, num_pixels * sizeof(short2)));
             pitch_elms_coord = width;
        }
        // We also need d_seeds for initialization
        CUDA_CHECK(cudaMalloc(&d_seeds, num_seeds * sizeof(GPUSeed)));
    } else if (cfg.use_pitch) {
        // Use cudaMallocPitch
        CUDA_CHECK(cudaMallocPitch(&d_bufA, &pitch_bytes, width * sizeof(int), height));
        CUDA_CHECK(cudaMallocPitch(&d_bufB, &pitch_bytes, width * sizeof(int), height));
        pitch_ints = pitch_bytes / sizeof(int);
    } else {
        // Use cudaMalloc
        CUDA_CHECK(cudaMalloc(&d_bufA, num_pixels * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bufB, num_pixels * sizeof(int)));
        pitch_ints = width; // Pitch is just width
    }

    if (cfg.use_soa && !cfg.use_coord_prop) {
        CUDA_CHECK(cudaMalloc(&d_seeds_x, num_seeds * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seeds_y, num_seeds * sizeof(int)));
    } else {
        CUDA_CHECK(cudaMalloc(&d_seeds, num_seeds * sizeof(GPUSeed)));
    }

    auto t_alloc = std::chrono::high_resolution_clock::now();

    // Copy seeds to device
    if (cfg.use_soa && !cfg.use_coord_prop) {
        CUDA_CHECK(cudaMemcpy(d_seeds_x, seeds_x, num_seeds * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds_y, seeds_y, num_seeds * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_seeds, seeds, num_seeds * sizeof(GPUSeed), cudaMemcpyHostToDevice));
    }

    // Copy seeds to constant memory if requested
    if (use_constant) {
        if (num_seeds > MAX_CONST_SEEDS) {
            printf("Warning: Too many seeds for constant memory (%d > %d). Falling back to global memory.\n", 
                   num_seeds, MAX_CONST_SEEDS);
            use_constant = false;
        } else {
            if (cfg.use_soa) {
                CUDA_CHECK(cudaMemcpyToSymbol(c_seeds_x, seeds_x, num_seeds * sizeof(int)));
                CUDA_CHECK(cudaMemcpyToSymbol(c_seeds_y, seeds_y, num_seeds * sizeof(int)));
            } else {
                CUDA_CHECK(cudaMemcpyToSymbol(c_seeds, seeds, num_seeds * sizeof(GPUSeed)));
            }
        }
    }

    auto t_copy_seeds = std::chrono::high_resolution_clock::now();

    // Initialize bufA with -1
    if (cfg.use_coord_prop) {
        if (cfg.use_pitch) {
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            // We need a pitch version of init_buffer_kernel_coord, but since it just fills -1, 
            // we can reuse init_buffer_kernel_pitch if we cast short2* to int*? No, size differs.
            // Let's just use a simple 2D kernel for now or reuse init_buffer_kernel_coord with pitch logic?
            // Actually, init_buffer_kernel_coord is 1D linear. With pitch, we have gaps.
            // We need a new kernel: init_buffer_kernel_coord_pitch
            // For simplicity, let's just use cudaMemset2D which is faster anyway!
            CUDA_CHECK(cudaMemset2D(d_coordA, pitch_bytes_coord, -1, width * sizeof(short2), height));
        } else {
            int blockSize = 256;
            int numBlocks = (num_pixels + blockSize - 1) / blockSize;
            init_buffer_kernel_coord<<<numBlocks, blockSize>>>(d_coordA, num_pixels, make_short2(-1, -1));
        }
        
        int blockSize = 256;
        int seedBlocks = (num_seeds + blockSize - 1) / blockSize;
        if (cfg.use_pitch) {
             place_seeds_kernel_coord_pitch<<<seedBlocks, blockSize>>>(d_coordA, d_seeds, num_seeds, width, height, pitch_elms_coord);
        } else {
             place_seeds_kernel_coord<<<seedBlocks, blockSize>>>(d_coordA, d_seeds, num_seeds, width, height);
        }
    } else if (cfg.use_pitch) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        init_buffer_kernel_pitch<<<grid, block>>>(d_bufA, width, height, -1, pitch_ints);
    } else {
        int blockSize = 256;
        int numBlocks = (num_pixels + blockSize - 1) / blockSize;
        init_buffer_kernel<<<numBlocks, blockSize>>>(d_bufA, num_pixels, -1);
    }
    CUDA_CHECK(cudaGetLastError());

    // Place seeds into bufA (if not coord prop)
    if (!cfg.use_coord_prop) {
        int blockSize = 256;
        int seedBlocks = (num_seeds + blockSize - 1) / blockSize;
        if (cfg.use_soa) {
            if (cfg.use_pitch) {
                place_seeds_kernel_pitch_soa<<<seedBlocks, blockSize>>>(d_bufA, d_seeds_x, d_seeds_y, num_seeds, width, height, pitch_ints);
            } else {
                place_seeds_kernel_soa<<<seedBlocks, blockSize>>>(d_bufA, d_seeds_x, d_seeds_y, num_seeds, width, height);
            }
        } else {
            if (cfg.use_pitch) {
                place_seeds_kernel_pitch<<<seedBlocks, blockSize>>>(d_bufA, d_seeds, num_seeds, width, height, pitch_ints);
            } else {
                place_seeds_kernel<<<seedBlocks, blockSize>>>(d_bufA, d_seeds, num_seeds, width, height);
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Sync for timing

    auto t_init = std::chrono::high_resolution_clock::now();

    // Create events for timing the kernel loop
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // JFA Loop
    int max_dim = (width > height) ? width : height;
    int step = max_dim / 2;
    if (step < 1) step = 1;

    int* d_in = d_bufA;
    int* d_out = d_bufB;
    short2* d_in_coord = d_coordA;
    short2* d_out_coord = d_coordB;

    dim3 dimBlock(block_dim_x, block_dim_y);
    
    int ppt = cfg.pixels_per_thread;
    if (ppt < 1) ppt = 1;

    // Adjust grid width based on pixels_per_thread
    // We assume thread coarsening happens in X direction
    dim3 dimGrid((width + (dimBlock.x * ppt) - 1) / (dimBlock.x * ppt), 
                 (height + dimBlock.y - 1) / dimBlock.y);

    CUDA_CHECK(cudaEventRecord(start));

    int pass_idx = 0;
    while (step >= 1) {
        if (cfg.use_coord_prop) {
            if (cfg.use_pitch) {
                jfa_step_kernel_coord_pitch<<<dimGrid, dimBlock>>>(d_in_coord, d_out_coord, width, height, step, ppt, pitch_elms_coord);
            } else {
                if (cfg.use_ultimate) {
                    jfa_step_kernel_ultimate<<<dimGrid, dimBlock>>>(d_in_coord, d_out_coord, width, height, step, ppt);
                } else if (cfg.use_int_math) {
                    jfa_step_kernel_int_math<<<dimGrid, dimBlock>>>(d_in_coord, d_out_coord, width, height, step, ppt);
                } else if (cfg.use_manhattan) {
                    jfa_step_kernel_manhattan<<<dimGrid, dimBlock>>>(d_in_coord, d_out_coord, width, height, step, ppt);
                } else if (cfg.use_ldg) {
                    jfa_step_kernel_coord_ldg<<<dimGrid, dimBlock>>>(d_in_coord, d_out_coord, width, height, step, ppt);
                } else {
                    jfa_step_kernel_coord<<<dimGrid, dimBlock>>>(d_in_coord, d_out_coord, width, height, step, ppt);
                }
            }
        } else if (cfg.use_soa) {
            if (cfg.use_pitch) {
                jfa_step_kernel_pitch_soa<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds_x, d_seeds_y, width, height, step, ppt, pitch_ints);
            } else {
                if (use_shared) {
                    int shared_mem_size = 2 * num_seeds * sizeof(int);
                    jfa_step_kernel_soa_shared<<<dimGrid, dimBlock, shared_mem_size>>>(d_in, d_out, d_seeds_x, d_seeds_y, width, height, step, num_seeds, ppt);
                } else if (use_constant) {
                    jfa_step_kernel_soa_constant<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, step, ppt);
                } else {
                    jfa_step_kernel_soa<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds_x, d_seeds_y, width, height, step, ppt);
                }
            }
        } else {
            if (cfg.use_pitch) {
                // Method 2 & 3: Use Pitch Kernel
                jfa_step_kernel_pitch<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt, pitch_ints);
            } else {
                // Method 1
                if (use_shared) {
                    int shared_mem_size = num_seeds * sizeof(GPUSeed);
                    jfa_step_kernel_shared<<<dimGrid, dimBlock, shared_mem_size>>>(d_in, d_out, d_seeds, width, height, step, num_seeds, ppt);
                } else if (use_constant) {
                    jfa_step_kernel_constant<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, step, ppt);
                } else {
                    if (cfg.use_ultimate) {
                        jfa_step_kernel_ultimate_no_coord<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt);
                    } else if (cfg.use_int_math) {
                        jfa_step_kernel_int_math_no_coord<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt);
                    } else if (cfg.use_manhattan) {
                        jfa_step_kernel_manhattan_no_coord<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt);
                    } else if (cfg.use_restrict) {
                        jfa_step_kernel_restrict<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt);
                    } else if (cfg.use_ldg) {
                        jfa_step_kernel_ldg_no_coord<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt);
                    } else if (cfg.use_fma) {
                        jfa_step_kernel_fma_no_coord<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt);
                    } else {
                        jfa_step_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt);
                    }
                }
            }
        }
        CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize()); // Not strictly needed if we use events, but good for debugging

        // Optional: Callback for visualization
        if (cb && !cfg.use_coord_prop) { // Skip callback for coord prop for now
            // If we have a callback, we must sync to copy data
            CUDA_CHECK(cudaDeviceSynchronize());
            // Copy d_out to host
            // We need a temporary buffer. 
            // Since we don't use std::vector, we use malloc/free or just reuse out_buffer if allowed?
            // But out_buffer is for final result.
            // Let's allocate a temp buffer.
            int* temp_host = (int*)malloc(num_pixels * sizeof(int));
            CUDA_CHECK(cudaMemcpy(temp_host, d_out, num_pixels * sizeof(int), cudaMemcpyDeviceToHost));
            cb(pass_idx, step, temp_host, num_pixels, user_data);
            free(temp_host);
        }

        // Swap buffers
        if (cfg.use_coord_prop) {
            short2* temp = d_in_coord;
            d_in_coord = d_out_coord;
            d_out_coord = temp;
        } else {
            int* temp = d_in;
            d_in = d_out;
            d_out = temp;
        }
        
        step /= 2;
        pass_idx++;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("  [CUDA Kernel Loop] Time: %.3f ms (block_dim=%dx%d)\n", milliseconds, block_dim_x, block_dim_y);

    auto t_loop_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Final result is in d_in (or d_in_coord)
    if (cfg.use_coord_prop) {
        // Convert back to IDs
        CUDA_CHECK(cudaMalloc(&d_id_map, width * height * sizeof(int)));
        int blockSize = 256;
        int numBlocks = (num_pixels + blockSize - 1) / blockSize;
        init_id_map_kernel<<<numBlocks, blockSize>>>(d_id_map, num_pixels, -1);
        
        int seedBlocks = (num_seeds + blockSize - 1) / blockSize;
        fill_id_map_kernel<<<seedBlocks, blockSize>>>(d_id_map, d_seeds, num_seeds, width, height);
        
        // Allocate output buffer (we can reuse d_bufA pointer variable, but need to malloc)
        CUDA_CHECK(cudaMalloc(&d_bufA, num_pixels * sizeof(int)));
        
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        if (cfg.use_pitch) {
            convert_coord_to_id_kernel_pitch<<<grid, block>>>(d_in_coord, d_bufA, d_id_map, width, height, pitch_elms_coord);
        } else {
            convert_coord_to_id_kernel<<<grid, block>>>(d_in_coord, d_bufA, d_id_map, width, height);
        }
        
        CUDA_CHECK(cudaMemcpy(out_buffer, d_bufA, num_pixels * sizeof(int), cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(d_id_map));
        CUDA_CHECK(cudaFree(d_bufA)); 
        d_bufA = nullptr; // Prevent double free
    } else if (cfg.use_pitch) {
        // For pitch memory, we need to copy row by row or use cudaMemcpy2D
        CUDA_CHECK(cudaMemcpy2D(out_buffer, width * sizeof(int), 
                                d_in, pitch_bytes, 
                                width * sizeof(int), height, 
                                cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(out_buffer, d_in, num_pixels * sizeof(int), cudaMemcpyDeviceToHost));
    }

    auto t_copy_back = std::chrono::high_resolution_clock::now();

    // Cleanup
    if (d_bufA) CUDA_CHECK(cudaFree(d_bufA));
    if (d_bufB) CUDA_CHECK(cudaFree(d_bufB));
    if (d_coordA) CUDA_CHECK(cudaFree(d_coordA));
    if (d_coordB) CUDA_CHECK(cudaFree(d_coordB));

    if (cfg.use_soa && !cfg.use_coord_prop) {
        CUDA_CHECK(cudaFree(d_seeds_x));
        CUDA_CHECK(cudaFree(d_seeds_y));
    } else {
        CUDA_CHECK(cudaFree(d_seeds));
    }

    auto t_free = std::chrono::high_resolution_clock::now();

    // Print breakdown
    auto d_alloc = std::chrono::duration<double, std::milli>(t_alloc - t_start).count();
    auto d_copy_seeds = std::chrono::duration<double, std::milli>(t_copy_seeds - t_alloc).count();
    auto d_init = std::chrono::duration<double, std::milli>(t_init - t_copy_seeds).count();
    auto d_loop = std::chrono::duration<double, std::milli>(t_loop_end - t_init).count();
    auto d_copy_back = std::chrono::duration<double, std::milli>(t_copy_back - t_loop_end).count();
    auto d_free = std::chrono::duration<double, std::milli>(t_free - t_copy_back).count();
    auto d_total = std::chrono::duration<double, std::milli>(t_free - t_start).count();

    std::cout << "  [Breakdown] Alloc: " << d_alloc << " ms, "
              << "H2D: " << d_copy_seeds << " ms, "
              << "Init: " << d_init << " ms, "
              << "Loop(CPU): " << d_loop << " ms, "
              << "D2H: " << d_copy_back << " ms, "
              << "Free: " << d_free << " ms\n"
              << "  [Total Internal] " << d_total << " ms\n";
    
    return 0;
}

} // namespace jfa
