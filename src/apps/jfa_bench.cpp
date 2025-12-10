// src/apps/jfa_bench.cpp
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>

#include <jfa/types.hpp>
#include <jfa/cpu.hpp>

struct Options {
    std::string backend = "serial"; // serial / omp / simd / cuda / mpi
    bool run_serial_baseline = true; // 對非 serial backend，是否順便跑 serial baseline

    int threads = 1;                // CPU threads (for omp / simd)
    int ranks   = 1;                // MPI ranks (之後用；目前沒用到)
    int device  = 0;                // GPU id (之後用；目前沒用到)

    int width   = 512;
    int height  = 512;
    int num_seeds = 500;
    unsigned int rng_seed = 42;

    bool csv = false;
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --backend {serial|omp|simd|cuda|mpi}\n"
              << "  --width N\n"
              << "  --height N\n"
              << "  --seeds N\n"
              << "  --threads N           # for omp/simd\n"
              << "  --ranks N             # for mpi (future)\n"
              << "  --device N            # for cuda (future)\n"
              << "  --rng-seed N\n"
              << "  --csv                 # print machine-readable CSV line\n"
              << "  --no-serial           # do NOT run serial baseline (for profiling)\n"
              << "  --no-dump             # ignored; kept for compatibility\n";
}

int main(int argc, char** argv) {
    Options opt;

    // --- CLI parsing ---
    auto get_value = [&](int& i, std::string& out) -> bool {
        if (i + 1 >= argc) return false;
        out = argv[++i];
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--backend") {
            std::string v;
            if (!get_value(i, v)) {
                std::cerr << "Missing value for --backend\n";
                print_usage(argv[0]);
                return 1;
            }
            opt.backend = v;
        } else if (arg == "--width") {
            std::string v;
            if (!get_value(i, v)) return 1;
            opt.width = std::stoi(v);
        } else if (arg == "--height") {
            std::string v;
            if (!get_value(i, v)) return 1;
            opt.height = std::stoi(v);
        } else if (arg == "--seeds") {
            std::string v;
            if (!get_value(i, v)) return 1;
            opt.num_seeds = std::stoi(v);
        } else if (arg == "--threads") {
            std::string v;
            if (!get_value(i, v)) return 1;
            opt.threads = std::stoi(v);
        } else if (arg == "--ranks") {
            std::string v;
            if (!get_value(i, v)) return 1;
            opt.ranks = std::stoi(v);
        } else if (arg == "--device") {
            std::string v;
            if (!get_value(i, v)) return 1;
            opt.device = std::stoi(v);
        } else if (arg == "--rng-seed") {
            std::string v;
            if (!get_value(i, v)) return 1;
            opt.rng_seed = static_cast<unsigned int>(std::stoul(v));
        } else if (arg == "--csv") {
            opt.csv = true;
        } else if (arg == "--no-serial") {
            // 對非 serial backend，不跑 baseline
            opt.run_serial_baseline = false;
        } else if (arg == "--no-dump") {
            // benchmark 模式本來就不 dump，這個 flag 只是為了跟 script 相容
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // --- 檢查 backend 合法性 ---
    if (opt.backend != "serial" &&
        opt.backend != "omp" &&
        opt.backend != "simd" &&
        opt.backend != "cuda" &&
        opt.backend != "mpi") {
        std::cerr << "Unsupported backend: " << opt.backend << "\n";
        print_usage(argv[0]);
        return 1;
    }

    // --- 準備 config / seeds / buffer ---
    jfa::Config cfg;
    cfg.width  = opt.width;
    cfg.height = opt.height;

    std::vector<jfa::Seed> seeds;
    seeds.reserve(opt.num_seeds);

    std::mt19937 rng(opt.rng_seed);
    std::uniform_int_distribution<int> dist_x(0, cfg.width  - 1);
    std::uniform_int_distribution<int> dist_y(0, cfg.height - 1);

    for (int i = 0; i < opt.num_seeds; ++i) {
        seeds.push_back(jfa::Seed{dist_x(rng), dist_y(rng)});
    }

    jfa::SeedIndexBuffer buffer(cfg.width * cfg.height);

    auto now_ms = [] {
        return std::chrono::steady_clock::now();
    };

    auto elapsed_ms = [](auto t0, auto t1) {
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    double serial_ms   = 0.0;
    double parallel_ms = 0.0;
    int    used_threads = 1;

    // --- Serial baseline (如果需要) ---
    bool did_serial = false;
    if (opt.backend == "serial" || opt.run_serial_baseline) {
        auto t0 = now_ms();
        jfa::jfa_cpu_serial(cfg, seeds, buffer, nullptr);
        auto t1 = now_ms();
        serial_ms = elapsed_ms(t0, t1);
        did_serial = true;
        used_threads = 1;
    }

    // --- 根據 backend 跑對應實作 ---
    if (opt.backend == "serial") {
        // 只有 serial，parallel_ms 就跟 serial 一樣
        parallel_ms = serial_ms;
    } else if (opt.backend == "omp") {
        auto t0 = now_ms();
        jfa::jfa_cpu_omp(cfg, seeds, buffer, opt.threads, nullptr);
        auto t1 = now_ms();
        parallel_ms = elapsed_ms(t0, t1);
        used_threads = opt.threads;
    } else if (opt.backend == "simd" ||
               opt.backend == "cuda" ||
               opt.backend == "mpi") {
        std::cerr << "[ERROR] backend '" << opt.backend
                  << "' is not implemented yet in jfa_bench.\n";
        std::cerr << "        (stub only; please implement jfa_cpu_simd / jfa_gpu_cuda / jfa_mpi_xxx later.)\n";
        return 1;
    }

    // --- 人類看的輸出 ---
    std::cout << "Config:\n";
    std::cout << "  backend   = " << opt.backend << "\n";
    std::cout << "  threads   = " << used_threads << "\n";
    std::cout << "  size      = " << cfg.width << " x " << cfg.height << "\n";
    std::cout << "  #seeds    = " << opt.num_seeds << "\n";
    std::cout << "  rng_seed  = " << opt.rng_seed << "\n\n";

    if (did_serial) {
        std::cout << "[Serial JFA] time = " << serial_ms << " ms\n";
    } else {
        std::cout << "[Serial JFA] (not run: --no-serial)\n";
    }

    if (opt.backend == "omp") {
        std::cout << "[OpenMP JFA] threads = " << used_threads
                  << ", time = " << parallel_ms << " ms\n";
        if (did_serial) {
            std::cout << "  speedup vs serial = " << (serial_ms / parallel_ms) << "x\n";
        }
    } else if (opt.backend == "serial") {
        // 就只是 serial，前面已經印過
    } else {
        // 之後 simd / cuda / mpi 完成後，可以在這裡加對應的 summary
    }

    // --- CSV 給 run_benchmarks.py 用 ---
    if (opt.csv) {
        double exact_ms = 0.0;       // benchmark 不跑 exact
        int diff_exact_parallel  = 0;
        int diff_serial_parallel = 0;

        // 對於沒跑 serial baseline 的平行 backend，serial_ms 沒意義，填 0
        double csv_serial_ms = did_serial ? serial_ms : 0.0;

        // backend_name: 給你 scripts 用的 label，例如 cpu-serial / cpu-omp
        std::string backend_name = "cpu-" + opt.backend;

        std::cout << "CSV," 
                  << opt.backend << ","
                  << used_threads << ","
                  << cfg.width << ","
                  << cfg.height << ","
                  << opt.num_seeds << ","
                  << exact_ms << ","
                  << csv_serial_ms << ","
                  << parallel_ms << ","
                  << diff_exact_parallel << ","
                  << diff_serial_parallel << "\n";
    }

    return 0;
}
