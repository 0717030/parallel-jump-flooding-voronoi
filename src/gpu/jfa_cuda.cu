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

// Constant memory for seeds (Max 8192 seeds to fit in 64KB)
#define MAX_CONST_SEEDS 8192
__constant__ GPUSeed c_seeds[MAX_CONST_SEEDS];

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
    GPUSeed* d_seeds = nullptr;
    int* d_seeds_x = nullptr;
    int* d_seeds_y = nullptr;
    size_t pitch_bytes = 0;
    size_t pitch_ints = 0;

    if (cfg.use_pitch) {
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

    if (cfg.use_soa) {
        CUDA_CHECK(cudaMalloc(&d_seeds_x, num_seeds * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seeds_y, num_seeds * sizeof(int)));
    } else {
        CUDA_CHECK(cudaMalloc(&d_seeds, num_seeds * sizeof(GPUSeed)));
    }

    auto t_alloc = std::chrono::high_resolution_clock::now();

    // Copy seeds to device
    if (cfg.use_soa) {
        CUDA_CHECK(cudaMemcpy(d_seeds_x, seeds_x, num_seeds * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds_y, seeds_y, num_seeds * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_seeds, seeds, num_seeds * sizeof(GPUSeed), cudaMemcpyHostToDevice));
    }

    // Copy seeds to constant memory if requested
    if (use_constant && !cfg.use_soa) { // Constant memory only implemented for AoS for now
        if (num_seeds > MAX_CONST_SEEDS) {
            printf("Warning: Too many seeds for constant memory (%d > %d). Falling back to global memory.\n", 
                   num_seeds, MAX_CONST_SEEDS);
            use_constant = false;
        } else {
            CUDA_CHECK(cudaMemcpyToSymbol(c_seeds, seeds, num_seeds * sizeof(GPUSeed)));
        }
    }

    auto t_copy_seeds = std::chrono::high_resolution_clock::now();

    // Initialize bufA with -1
    if (cfg.use_pitch) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        init_buffer_kernel_pitch<<<grid, block>>>(d_bufA, width, height, -1, pitch_ints);
    } else {
        int blockSize = 256;
        int numBlocks = (num_pixels + blockSize - 1) / blockSize;
        init_buffer_kernel<<<numBlocks, blockSize>>>(d_bufA, num_pixels, -1);
    }
    CUDA_CHECK(cudaGetLastError());

    // Place seeds into bufA
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
        if (cfg.use_soa) {
            if (cfg.use_pitch) {
                jfa_step_kernel_pitch_soa<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds_x, d_seeds_y, width, height, step, ppt, pitch_ints);
            } else {
                jfa_step_kernel_soa<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds_x, d_seeds_y, width, height, step, ppt);
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
                    jfa_step_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, d_seeds, width, height, step, ppt);
                }
            }
        }
        CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaDeviceSynchronize()); // Not strictly needed if we use events, but good for debugging

        // Optional: Callback for visualization
        if (cb) {
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
        int* temp = d_in;
        d_in = d_out;
        d_out = temp;
        
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

    // Final result is in d_in
    if (cfg.use_pitch) {
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
    CUDA_CHECK(cudaFree(d_bufA));
    CUDA_CHECK(cudaFree(d_bufB));
    if (cfg.use_soa) {
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
