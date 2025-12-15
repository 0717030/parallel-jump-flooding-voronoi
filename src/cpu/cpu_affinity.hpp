// src/cpu/cpu_affinity.hpp
#pragma once

// Utility to set CPU affinity to P-cores only (for hybrid CPUs like i7-12700).
// This prevents threads from running on E-cores, which can improve performance.

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <cstring>
#endif

namespace jfa {
namespace detail {

// Set CPU affinity to P-cores only.
// For i7-12700: 8 P-cores (typically cores 0-7), 4 E-cores (typically cores 8-11).
// This function pins the current thread and all future OpenMP threads to P-cores.
inline void set_pcore_affinity()
{
#ifdef __linux__
    // For i7-12700: P-cores are typically cores 0-7 (8 cores)
    // We'll use cores 0-7 as P-cores
    const int num_pcores = 8;
    const int first_pcore = 0;
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    // Set affinity to P-cores only (cores 0-7)
    for (int i = 0; i < num_pcores; ++i) {
        CPU_SET(first_pcore + i, &cpuset);
    }
    
    // Set affinity for the current thread
    pthread_t thread = pthread_self();
    int ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    
    // Also set process affinity (affects all threads in the process)
    // This ensures OpenMP threads also respect the affinity
    pid_t pid = getpid();
    ret = sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset);
    
    (void)ret; // Suppress unused variable warning
#endif
    // On non-Linux systems, this is a no-op
}

} // namespace detail
} // namespace jfa





