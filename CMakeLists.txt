cmake_minimum_required(VERSION 3.18)
project(pp_final_voronoi CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CPU JFA library
add_library(jfa_cpu
    src/cpu/jfa_serial.cpp
    src/cpu/jfa_openmp.cpp
    src/cpu/jfa_common_impl.hpp
)

# GPU JFA library
add_library(jfa_gpu
    src/gpu/jfa_cuda.cu
    src/gpu/jfa_gpu_wrapper.cpp
)
# set_property(TARGET jfa_gpu PROPERTY CUDA_ARCHITECTURES native)
# Fallback to a common architecture if native detection fails (e.g. no GPU during build)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET jfa_gpu PROPERTY CUDA_ARCHITECTURES 60 75)
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

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(jfa_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()
