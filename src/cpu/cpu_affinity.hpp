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
// For i7-12700: 8 physical P-cores, each with 2 hyperthreads.
// This function pins the current thread and all future OpenMP threads to P-cores.
// Uses 8 different physical cores (one thread per physical core) for optimal performance.
inline void set_pcore_affinity()
{
#ifdef __linux__
    // For i7-12700: Use 8 different physical P-cores
    // Physical cores: 0->CPU 0,1 | 1->CPU 2,3 | 2->CPU 4,5 | 3->CPU 6,7
    //                 4->CPU 8,9 | 5->CPU 10,11 | 6->CPU 12,13 | 7->CPU 14,15
    // We select one logical CPU from each physical core: 0, 2, 4, 6, 8, 10, 12, 14
    const int pcore_cpus[] = {0, 2, 4, 6, 8, 10, 12, 14};
    const int num_pcores = sizeof(pcore_cpus) / sizeof(pcore_cpus[0]);
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    // Set affinity to 8 different physical P-cores
    for (int i = 0; i < num_pcores; ++i) {
        CPU_SET(pcore_cpus[i], &cpuset);
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





