cmake_minimum_required(VERSION 3.18)
project(pp_final_voronoi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CPU JFA library
add_library(jfa_cpu
    src/cpu/jfa_serial.cpp
    src/cpu/jfa_openmp.cpp
    src/cpu/jfa_common_impl.hpp
)

# # ✅ 在這裡加 profiling 友善的編譯選項
# if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(jfa_cpu PRIVATE -g -fno-omit-frame-pointer)
# else()
#     target_compile_options(jfa_cpu PRIVATE -fno-omit-frame-pointer)
# endif()

# ✅ 在這裡加 profiling 友善的編譯選項（只影響 RelWithDebInfo）
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(jfa_cpu PRIVATE -g -fno-omit-frame-pointer)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(jfa_cpu PRIVATE -fno-omit-frame-pointer)
else()
    target_compile_options(jfa_cpu PRIVATE)
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
    PRIVATE jfa_cpu jfa_exact jfa_visualize
)

# NEW: benchmark app（只做 timing，不做 exact / visualize）
add_executable(jfa_bench
    src/apps/jfa_bench.cpp
)

target_link_libraries(jfa_bench
    PRIVATE jfa_cpu
)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(jfa_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()
