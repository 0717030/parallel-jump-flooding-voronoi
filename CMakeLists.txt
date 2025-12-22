cmake_minimum_required(VERSION 3.18)
project(pp_final_voronoi CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Make CUDA Toolkit headers/libs (e.g. cuda_runtime.h, cudart) available to C++ targets.
find_package(CUDAToolkit REQUIRED)

# CPU JFA library
add_library(jfa_cpu
    src/cpu/jfa_serial.cpp
    src/cpu/jfa_openmp.cpp
    src/cpu/jfa_common_impl.hpp
    src/cpu/jfa_simd_avx2.cpp
)

# Compile SIMD translation unit with AVX2 enabled (scoped to this file).
set_source_files_properties(src/cpu/jfa_simd_avx2.cpp
    PROPERTIES
        COMPILE_OPTIONS "-mavx2"
)

# GPU JFA library
add_library(jfa_gpu
    src/gpu/jfa_cuda.cu
    src/gpu/jfa_gpu_wrapper.cpp
)

# Ensure C++ compilation units in this target can find CUDA headers and link cudart.
target_link_libraries(jfa_gpu PUBLIC CUDA::cudart)

# If user didn't specify architectures, include RTX 4090 (sm_89) plus a couple of common ones.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET jfa_gpu PROPERTY CUDA_ARCHITECTURES 60 75 89)
endif()

# Exact Voronoi baseline（你也可以跟 jfa_cpu 合在一起）
add_library(jfa_exact
    src/exact/voronoi_exact.cpp
)

# visualize helper
add_library(jfa_visualize
    src/common/jfa_visualize.cpp
)

# demo app
add_executable(jfa_demo
    src/apps/jfa_demo.cpp
)

target_link_libraries(jfa_demo
    PRIVATE jfa_cpu jfa_exact jfa_visualize jfa_gpu
)

find_package(OpenMP REQUIRED)
target_link_libraries(jfa_cpu PUBLIC OpenMP::OpenMP_CXX)
