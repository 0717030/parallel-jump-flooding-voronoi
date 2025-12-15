// src/apps/jfa_demo.cpp
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <cctype>

#include <jfa/types.hpp>
#include <jfa/cpu.hpp>
#include <jfa/gpu.hpp>
#include <jfa/exact.hpp>
#include <jfa/visualize.hpp>

struct Options {
    std::string backend = "omp"; // "serial", "omp", "simd", "omp_simd", or "cuda"
    int threads = 8;             // used for omp / omp_simd
    int width = 512;
    int height = 512;
    int num_seeds = 50;
    bool dump_frames = true;
    unsigned int rng_seed = 42;
    bool csv = false;            // NEW: enable machine-readable CSV output

    std::string tag;                 // NEW: for naming
    std::string output_dir = "output";   // 新增：輸出資料夾（相對於執行時工作目錄）
    bool has_output_dir = false;     // NEW: user explicitly gave --output-dir
    int block_dim_x = 16;            // NEW: CUDA block dimension X
    int block_dim_y = 16;            // NEW: CUDA block dimension Y
    bool use_shared_mem = false;     // NEW: Use shared memory
    bool use_constant_mem = false;   // NEW: Use constant memory
    int pixels_per_thread = 1;       // NEW: Pixels per thread
    bool use_pitch = false;          // NEW: Use pitched memory
    bool use_pinned = false;         // NEW: Use pinned memory
    bool use_soa = false;            // NEW: Use Structure of Arrays
    bool use_coord_prop = false;     // NEW: Use Coordinate Propagation
    bool cpu_use_pitch = false;      // NEW: CPU SIMD: pad internal row stride (pitch)
    std::string cpu_seeds_layout = "packed";   // CPU SIMD only (index-based): packed|soa|aos
    std::string cpu_coordbuf_layout = "soa";   // CPU SIMD only (coord-prop): soa|aos
    bool skip_exact = false;         // NEW: Skip exact check
    bool skip_serial = false;        // NEW: Skip serial check
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --backend {serial|omp|simd|omp_simd|cuda} Select JFA implementation (default: omp)\n"
              << "  --threads N              Number of threads for OpenMP backends (omp/omp_simd) (default: 8)\n"
              << "  --block-dim N            CUDA block dimension (square, default: 16)\n"
              << "  --block-dim-x N          CUDA block dimension X (default: 16)\n"
              << "  --block-dim-y N          CUDA block dimension Y (default: 16)\n"
              << "  --ppt N                  Pixels per thread (default: 1)\n"
              << "  --use-pitch              Use cudaMallocPitch for memory alignment (default: off)\n"
              << "  --pinned                 Use cudaMallocHost for pinned memory (default: off)\n"
              << "  --soa                    Use Structure of Arrays (CUDA only: seeds layout) (default: off)\n"
              << "  --use-coord-prop         Use Coordinate Propagation (CPU and CUDA) (default: off)\n"
              << "  --cpu-pitch              CPU SIMD only: pad internal row stride to align vector loads/stores (default: off)\n"
              << "  --cpu-seeds-layout {packed|soa|aos}  CPU SIMD only (index-based mode): choose seeds gather layout (default: packed)\n"
              << "  --cpu-coordbuf-layout {soa|aos}      CPU SIMD only (coord-prop mode): choose pixel coord buffer layout (default: soa)\n"
              << "  --use-shared             Use shared memory for seeds (CUDA only)\n"
              << "  --use-constant           Use constant memory for seeds (CUDA only)\n"
              << "  --width W                Image width  (default: 512)\n"
              << "  --height H               Image height (default: 512)\n"
              << "  --seeds N                Number of random seeds (default: 50)\n"
              << "  --seed N                 RNG seed for reproducibility (default: 42)\n"
              << "  --no-dump                Do not dump per-pass PPM frames (profiling mode)\n"
              << "  --skip-check             Skip both exact and serial verification (for large scale bench)\n"
              << "  --skip-exact             Skip exact verification (for large scale bench)\n"
              << "  --skip-serial            Skip serial verification (for large scale bench)\n"
              << "  --output-dir DIR        Directory to store PPM frames (default: output)\n" // NEW
              << "  --csv                    Also print a machine-readable CSV summary line\n" // NEW
              << "  --tag NAME               Logical name of this run; used in auto output folder\n" // NEW
              << "  -h, --help               Show this help message\n";
}

bool parse_int(const std::string& s, int& out) {
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto get_value = [&](std::string& val) -> bool {
            auto eq_pos = arg.find('=');
            if (eq_pos != std::string::npos) {
                val = arg.substr(eq_pos + 1);
                return true;
            }
            if (i + 1 >= argc) return false;
            val = argv[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg.rfind("--backend", 0) == 0) {
            std::string v;
            if (!get_value(v)) {
                std::cerr << "Missing value for --backend\n";
                print_usage(argv[0]);
                std::exit(1);
            }
            opt.backend = v;
        } else if (arg.rfind("--threads", 0) == 0) {
            std::string v;
            if (!get_value(v) || !parse_int(v, opt.threads) || opt.threads < 1) {
                std::cerr << "Invalid value for --threads\n";
                std::exit(1);
            }
        } else if (arg.rfind("--block-dim-x", 0) == 0) {
            std::string v;
            if (!get_value(v) || !parse_int(v, opt.block_dim_x) || opt.block_dim_x < 1) {
                std::cerr << "Invalid value for --block-dim-x\n";
                std::exit(1);
            }
        } else if (arg.rfind("--block-dim-y", 0) == 0) {
            std::string v;
            if (!get_value(v) || !parse_int(v, opt.block_dim_y) || opt.block_dim_y < 1) {
                std::cerr << "Invalid value for --block-dim-y\n";
                std::exit(1);
            }
        } else if (arg.rfind("--block-dim", 0) == 0) {
            std::string v;
            int dim;
            if (!get_value(v) || !parse_int(v, dim) || dim < 1) {
                std::cerr << "Invalid value for --block-dim\n";
                std::exit(1);
            }
            opt.block_dim_x = dim;
            opt.block_dim_y = dim;
        } else if (arg == "--use-shared") {
            opt.use_shared_mem = true;
        } else if (arg == "--use-constant") {
            opt.use_constant_mem = true;
        } else if (arg.rfind("--ppt", 0) == 0) {
            std::string v;
            if (!get_value(v) || !parse_int(v, opt.pixels_per_thread) || opt.pixels_per_thread < 1) {
                std::cerr << "Invalid value for --ppt\n";
                std::exit(1);
            }
        } else if (arg == "--use-pitch") {
            opt.use_pitch = true;
        } else if (arg == "--pinned") {
            opt.use_pinned = true;
        } else if (arg == "--soa") {
            opt.use_soa = true;
        } else if (arg == "--use-coord-prop") {
            opt.use_coord_prop = true;
        } else if (arg == "--cpu-pitch") {
            opt.cpu_use_pitch = true;
        } else if (arg.rfind("--cpu-seeds-layout", 0) == 0) {
            std::string v;
            if (!get_value(v)) {
                std::cerr << "Missing value for --cpu-seeds-layout\n";
                std::exit(1);
            }
            opt.cpu_seeds_layout = v;
        } else if (arg.rfind("--cpu-coordbuf-layout", 0) == 0) {
            std::string v;
            if (!get_value(v)) {
                std::cerr << "Missing value for --cpu-coordbuf-layout\n";
                std::exit(1);
            }
            opt.cpu_coordbuf_layout = v;
        } else if (arg == "--skip-check") {
            opt.skip_exact = true;
            opt.skip_serial = true;
        } else if (arg == "--skip-exact") {
            opt.skip_exact = true;
        } else if (arg == "--skip-serial") {
            opt.skip_serial = true;
        } else if (arg.rfind("--width", 0) == 0) {
            std::string v;
            if (!get_value(v) || !parse_int(v, opt.width) || opt.width <= 0) {
                std::cerr << "Invalid value for --width\n";
                std::exit(1);
            }
        } else if (arg.rfind("--height", 0) == 0) {
            std::string v;
            if (!get_value(v) || !parse_int(v, opt.height) || opt.height <= 0) {
                std::cerr << "Invalid value for --height\n";
                std::exit(1);
            }
        } else if (arg.rfind("--seeds", 0) == 0) {
            std::string v;
            if (!get_value(v) || !parse_int(v, opt.num_seeds) || opt.num_seeds <= 0) {
                std::cerr << "Invalid value for --seeds\n";
                std::exit(1);
            }
        } else if (arg.rfind("--seed", 0) == 0) {
            std::string v;
            int tmp;
            if (!get_value(v) || !parse_int(v, tmp)) {
                std::cerr << "Invalid value for --seed\n";
                std::exit(1);
            }
            opt.rng_seed = static_cast<unsigned int>(tmp);
        } else if (arg == "--no-dump") {
            opt.dump_frames = false;
        } else if (arg.rfind("--output-dir", 0) == 0) {
            std::string v;
            if (!get_value(v)) {
                std::cerr << "Missing value for --output-dir\n";
                std::exit(1);
            }
            opt.output_dir = v;
            opt.has_output_dir = true;   // NEW
        } else if (arg.rfind("--tag", 0) == 0) {   // NEW
            std::string v;
            if (!get_value(v)) {
                std::cerr << "Missing value for --tag\n";
                std::exit(1);
            }
            opt.tag = v;
        } else if (arg == "--csv") {        // NEW
            opt.csv = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (opt.backend != "serial" && opt.backend != "omp" &&
        opt.backend != "simd" && opt.backend != "omp_simd" &&
        opt.backend != "cuda") {
        std::cerr << "Unsupported backend '" << opt.backend
                  << "', falling back to 'omp'\n";
        opt.backend = "omp";
    }

    return opt;
}

int main(int argc, char** argv)
{
    using Clock = std::chrono::high_resolution_clock;

    Options opt = parse_args(argc, argv);

    // 決定實際 output 目錄
    std::string auto_name;
    if (!opt.tag.empty()) {
        std::ostringstream oss;
        oss << opt.tag << "_" << opt.backend
            << "_t" << ((opt.backend == "omp" || opt.backend == "omp_simd") ? opt.threads : 1)
            << "_" << opt.width << "x" << opt.height
            << "_s" << opt.num_seeds;
        auto_name = oss.str();
    }

    // root_output: 使用者有指定就用指定的，否則預設 "output"
    std::string root_output = opt.has_output_dir ? opt.output_dir : "output";

    // final_output_dir: 有 tag 的話 → root_output/auto_name，否則就是 root_output
    std::string final_output_dir = root_output;
    if (!auto_name.empty()) {
        final_output_dir += "/" + auto_name;
    }

    // 記錄回 opt（讓 make_callback / 其他地方用）
    opt.output_dir = final_output_dir;

    // 建目錄
    //std::filesystem::create_directories(opt.output_dir);

    jfa::Config cfg{opt.width, opt.height};
    cfg.block_dim_x = opt.block_dim_x;
    cfg.block_dim_y = opt.block_dim_y;
    cfg.use_shared_mem = opt.use_shared_mem;
    cfg.use_constant_mem = opt.use_constant_mem;
    cfg.pixels_per_thread = opt.pixels_per_thread;
    cfg.use_pitch = opt.use_pitch;
    cfg.use_soa = opt.use_soa;
    cfg.use_coord_prop = opt.use_coord_prop;
    cfg.cpu_use_pitch = opt.cpu_use_pitch;

    // CPU SIMD-only layouts (do not reuse cfg.use_soa to avoid semantic confusion)
    auto lower = [](std::string s) {
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    {
        const std::string seeds_layout = lower(opt.cpu_seeds_layout);
        if (seeds_layout == "packed") cfg.cpu_seeds_layout = jfa::CpuSeedsLayout::Packed;
        else if (seeds_layout == "soa") cfg.cpu_seeds_layout = jfa::CpuSeedsLayout::SoA;
        else if (seeds_layout == "aos") cfg.cpu_seeds_layout = jfa::CpuSeedsLayout::AoS;
        else {
            std::cerr << "Invalid --cpu-seeds-layout: " << opt.cpu_seeds_layout << " (use packed|soa|aos)\n";
            std::exit(1);
        }

        const std::string coordbuf_layout = lower(opt.cpu_coordbuf_layout);
        if (coordbuf_layout == "soa") cfg.cpu_coordbuf_layout = jfa::CpuCoordBufLayout::SoA;
        else if (coordbuf_layout == "aos") cfg.cpu_coordbuf_layout = jfa::CpuCoordBufLayout::AoS;
        else {
            std::cerr << "Invalid --cpu-coordbuf-layout: " << opt.cpu_coordbuf_layout << " (use soa|aos)\n";
            std::exit(1);
        }
    }

    std::cout << "Config:\n"
              << "  backend   = " << opt.backend << "\n";

    if (opt.backend == "serial") {
        std::cout << "  coord_prop= " << (opt.use_coord_prop ? "yes" : "no") << "\n";
    } else if (opt.backend == "omp") {
        std::cout << "  threads   = " << opt.threads << "\n"
                  << "  coord_prop= " << (opt.use_coord_prop ? "yes" : "no") << "\n";
    } else if (opt.backend == "simd") {
        std::cout << "  coord_prop= " << (opt.use_coord_prop ? "yes" : "no") << "\n"
                  << "  cpu_pitch = " << (opt.cpu_use_pitch ? "yes" : "no") << "\n"
                  << "  cpu_seeds_layout   = " << opt.cpu_seeds_layout << "\n"
                  << "  cpu_coordbuf_layout= " << opt.cpu_coordbuf_layout << "\n";
    } else if (opt.backend == "omp_simd") {
        std::cout << "  threads   = " << opt.threads << "\n"
                  << "  coord_prop= " << (opt.use_coord_prop ? "yes" : "no") << "\n"
                  << "  cpu_pitch = " << (opt.cpu_use_pitch ? "yes" : "no") << "\n"
                  << "  cpu_seeds_layout   = " << opt.cpu_seeds_layout << "\n"
                  << "  cpu_coordbuf_layout= " << opt.cpu_coordbuf_layout << "\n";
    } else if (opt.backend == "cuda") {
        std::cout << "  block_dim = " << opt.block_dim_x << "x" << opt.block_dim_y << "\n"
                  << "  ppt       = " << opt.pixels_per_thread << "\n"
                  << "  use_pitch = " << (opt.use_pitch ? "yes" : "no") << "\n"
                  << "  pinned    = " << (opt.use_pinned ? "yes" : "no") << "\n"
                  << "  use_soa   = " << (opt.use_soa ? "yes" : "no") << "\n"
                  << "  coord_prop= " << (opt.use_coord_prop ? "yes" : "no") << "\n"
                  << "  shared_mem= " << (opt.use_shared_mem ? "yes" : "no") << "\n"
                  << "  const_mem = " << (opt.use_constant_mem ? "yes" : "no") << "\n";
    }

    std::cout << "  size      = " << cfg.width << " x " << cfg.height << "\n"
              << "  #seeds    = " << opt.num_seeds << "\n"
              << "  dump      = " << (opt.dump_frames ? "yes" : "no") << "\n"
              << "  rng_seed  = " << opt.rng_seed << "\n";

    // 產生隨機 seeds（可重現）
    std::vector<jfa::Seed> seeds;
    seeds.reserve(opt.num_seeds);
    std::mt19937 rng(opt.rng_seed);
    std::uniform_int_distribution<int> dist_x(0, cfg.width  - 1);
    std::uniform_int_distribution<int> dist_y(0, cfg.height - 1);
    for (int i = 0; i < opt.num_seeds; ++i) {
        seeds.push_back(jfa::Seed{dist_x(rng), dist_y(rng)});
    }

    std::filesystem::create_directories(opt.output_dir);

    jfa::SeedIndexBuffer exact_buf;
    jfa::SeedIndexBuffer serial_buf;
    jfa::SeedIndexBuffer omp_buf;
    jfa::SeedIndexBuffer simd_buf;
    jfa::SeedIndexBuffer omp_simd_buf;

    double exact_ms = 0;
    double serial_ms = 0;

    // 1) exact baseline
    if (!opt.skip_exact) {
        auto t_exact_0 = Clock::now();
        jfa::voronoi_exact_cpu(cfg, seeds, exact_buf);
        auto t_exact_1 = Clock::now();
        exact_ms = std::chrono::duration<double, std::milli>(t_exact_1 - t_exact_0).count();
        std::cout << "\n[Exact] time = " << exact_ms << " ms\n";
    } else {
        std::cout << "\n[Exact] Skipped\n";
    }

    // 2) serial JFA
    if (!opt.skip_serial) {
        auto t_serial_0 = Clock::now();
        jfa::jfa_cpu_serial(cfg, seeds, serial_buf, nullptr);
        auto t_serial_1 = Clock::now();
        serial_ms = std::chrono::duration<double, std::milli>(t_serial_1 - t_serial_0).count();
    } else {
        std::cout << "\n[Serial] Skipped\n";
        serial_ms = 1.0; // avoid div by zero
    }

    auto diff_count = [](const jfa::SeedIndexBuffer& a,
                         const jfa::SeedIndexBuffer& b) {
        if (a.size() != b.size()) return -1;
        int diff = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) ++diff;
        return diff;
    };

    int diff_exact_serial = (!opt.skip_exact && !opt.skip_serial) ? diff_count(exact_buf, serial_buf) : -1;

    if (!opt.skip_serial) {
        std::cout << "[Serial JFA] time = " << serial_ms << " ms";
        if (!opt.skip_exact) {
            std::cout << ", diff vs exact = " << diff_exact_serial << " pixels";
        }
        std::cout << "\n";
    }

    // NEW: 預設「目前 backend 的結果」先等於 serial baseline
    double parallel_ms = serial_ms;
    int diff_exact_parallel = diff_exact_serial;
    int diff_serial_parallel = 0;

    // dump 用到的顏色 & buffer
    std::vector<jfa::Color> palette;
    jfa::RGBImage rgb;

    auto make_callback = [&](const std::string& prefix)
        -> jfa::PassCallback {
        if (!opt.dump_frames) return nullptr;

        palette = jfa::make_seed_palette(seeds.size());
        return [&](int pass_idx, int step, const jfa::SeedIndexBuffer& buf) {
            jfa::seed_index_to_rgb(buf, seeds, palette,
                                   cfg.width, cfg.height, rgb);
            std::ostringstream oss;

            oss << opt.output_dir << "/" << prefix << "_pass_"
                << std::setw(3) << std::setfill('0') << pass_idx
                << "_k" << step << ".ppm";
            jfa::write_ppm(oss.str(), cfg.width, cfg.height, rgb);
        };
    };

    if (opt.backend == "serial") {
        // backend=serial 時，再跑一次 serial JFA + callback（給動畫）
        jfa::SeedIndexBuffer tmp_buf;
        auto cb = make_callback("cpu_jfa_serial");
        auto t0 = Clock::now();
        jfa::jfa_cpu_serial(cfg, seeds, tmp_buf, cb);
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int diff_exact_tmp = diff_count(exact_buf, tmp_buf);
        std::cout << "[Serial JFA (with dump)] time = " << ms << " ms"
                  << ", diff vs exact = " << diff_exact_tmp << " pixels\n";

        // NEW: 對 CSV 來說，serial backend 的「parallel」結果就是 serial baseline
        parallel_ms = serial_ms;
        diff_exact_parallel = diff_exact_serial;
        diff_serial_parallel = 0;
    } else if (opt.backend == "omp") {
        auto cb = make_callback("cpu_jfa_omp");
        auto t0 = Clock::now();
        jfa::jfa_cpu_omp(cfg, seeds, omp_buf, opt.threads, cb);
        auto t1 = Clock::now();
        double omp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int diff_exact_omp = opt.skip_exact ? -1 : diff_count(exact_buf, omp_buf);
        int diff_serial_omp = opt.skip_serial ? -1 : diff_count(serial_buf, omp_buf);

        std::cout << "[OpenMP JFA] threads = " << opt.threads
                  << ", time = " << omp_ms << " ms";
        if (!opt.skip_exact) std::cout << ", diff vs exact = " << diff_exact_omp;
        if (!opt.skip_serial) std::cout << ", diff vs serial = " << diff_serial_omp;
        std::cout << " pixels\n";

        if (!opt.skip_serial) {
            double speedup_vs_serial = serial_ms / omp_ms;
            std::cout << "  speedup vs serial JFA = "
                      << speedup_vs_serial << "x\n";
        }

        // NEW: 給 CSV 用的「parallel」結果改成 OMP 版
        parallel_ms = omp_ms;
        diff_exact_parallel = diff_exact_omp;
        diff_serial_parallel = diff_serial_omp;
    } else if (opt.backend == "simd") {
        auto cb = make_callback("cpu_jfa_simd");
        auto t0 = Clock::now();
        // SIMD backend supports two modes:
        // - default (no --use-coord-prop): index-based SIMD (lighter memory, often better on CPU)
        // - with --use-coord-prop: coord-prop SIMD (useful for comparison / GPU-like behavior)
        jfa::jfa_cpu_simd(cfg, seeds, simd_buf, cb);
        auto t1 = Clock::now();
        double simd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int diff_exact_simd = opt.skip_exact ? -1 : diff_count(exact_buf, simd_buf);
        int diff_serial_simd = opt.skip_serial ? -1 : diff_count(serial_buf, simd_buf);

        std::cout << "[SIMD JFA] time = " << simd_ms << " ms";
        if (!opt.skip_exact) std::cout << ", diff vs exact = " << diff_exact_simd;
        if (!opt.skip_serial) std::cout << ", diff vs serial = " << diff_serial_simd;
        std::cout << " pixels\n";

        if (!opt.skip_serial) {
            double speedup_vs_serial = serial_ms / simd_ms;
            std::cout << "  speedup vs serial JFA = " << speedup_vs_serial << "x\n";
        }

        parallel_ms = simd_ms;
        diff_exact_parallel = diff_exact_simd;
        diff_serial_parallel = diff_serial_simd;
    } else if (opt.backend == "omp_simd") {
        auto cb = make_callback("cpu_jfa_omp_simd");
        auto t0 = Clock::now();
        jfa::jfa_cpu_omp_simd(cfg, seeds, omp_simd_buf, opt.threads, cb);
        auto t1 = Clock::now();
        double omp_simd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int diff_exact_tmp = opt.skip_exact ? -1 : diff_count(exact_buf, omp_simd_buf);
        int diff_serial_tmp = opt.skip_serial ? -1 : diff_count(serial_buf, omp_simd_buf);

        std::cout << "[OpenMP+SIMD JFA] threads = " << opt.threads
                  << ", time = " << omp_simd_ms << " ms";
        if (!opt.skip_exact) std::cout << ", diff vs exact = " << diff_exact_tmp;
        if (!opt.skip_serial) std::cout << ", diff vs serial = " << diff_serial_tmp;
        std::cout << " pixels\n";

        if (!opt.skip_serial) {
            double speedup_vs_serial = serial_ms / omp_simd_ms;
            std::cout << "  speedup vs serial JFA = " << speedup_vs_serial << "x\n";
        }

        parallel_ms = omp_simd_ms;
        diff_exact_parallel = diff_exact_tmp;
        diff_serial_parallel = diff_serial_tmp;
    } else if (opt.backend == "cuda") {
        jfa::SeedIndexBuffer cuda_buf;
        auto cb = make_callback("gpu_jfa_cuda");
        auto t0 = Clock::now();
        try {
            if (opt.use_pinned) {
                // Use Pinned Memory (Host)
                int* pinned_buf = jfa::allocate_pinned_memory(cfg.width * cfg.height);
                if (!pinned_buf) throw std::runtime_error("Failed to allocate pinned memory");
                
                jfa::jfa_gpu_cuda(cfg, seeds, pinned_buf, cb);
                
                // Copy back to vector for validation/dump
                cuda_buf.assign(pinned_buf, pinned_buf + cfg.width * cfg.height);
                
                jfa::free_pinned_memory(pinned_buf);
            } else {
                // Use Pageable Memory (Host)
                jfa::jfa_gpu_cuda(cfg, seeds, cuda_buf, cb);
            }
            
            auto t1 = Clock::now();
            double cuda_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int diff_exact_cuda = opt.skip_exact ? -1 : diff_count(exact_buf, cuda_buf);
            int diff_serial_cuda = opt.skip_serial ? -1 : diff_count(serial_buf, cuda_buf);

            std::cout << "[CUDA JFA] time = " << cuda_ms << " ms";
            if (!opt.skip_exact) std::cout << ", diff vs exact = " << diff_exact_cuda;
            if (!opt.skip_serial) std::cout << ", diff vs serial = " << diff_serial_cuda;
            std::cout << " pixels\n";

            if (!opt.skip_serial) {
                double speedup_vs_serial = serial_ms / cuda_ms;
                std::cout << "  speedup vs serial JFA = "
                          << speedup_vs_serial << "x\n";
            }

            parallel_ms = cuda_ms;
            diff_exact_parallel = diff_exact_cuda;
            diff_serial_parallel = diff_serial_cuda;
        } catch (const std::exception& e) {
            std::cerr << "CUDA Backend Error: " << e.what() << "\n";
            // Fallback or just exit gracefully?
            // Since user asked for CUDA, we should probably just report it failed.
        }
    }

    // NEW: 如果要給 script 用，就印一行 machine-readable CSV
    if (opt.csv) {
        std::cout << "CSV,"
                  << opt.backend << ","
                  << ((opt.backend == "omp" || opt.backend == "omp_simd") ? opt.threads : 1) << ","
                  << cfg.width << ","
                  << cfg.height << ","
                  << opt.num_seeds << ","
                  << exact_ms << ","
                  << serial_ms << ","
                  << parallel_ms << ","
                  << diff_exact_parallel << ","
                  << diff_serial_parallel << "\n";
    }

    std::cout << "\nDone. Frames (if enabled) are in ./" 
          << opt.output_dir << "/*.ppm\n";

    return 0;
}
