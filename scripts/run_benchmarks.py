#!/usr/bin/env python3
import csv
import subprocess
import statistics
from pathlib import Path

# 假設你的 build 在 ../build
JFA_BIN = Path(__file__).resolve().parent.parent / "build" / "jfa_demo"

# 想測哪些 backend（之後加 CUDA 只要在這裡加）
BACKENDS = [
    ("cpu-serial", {"backend": "serial"}),  # 目前 serial 是 baseline
    ("cpu-omp",    {"backend": "omp"}),
    # SIMD (single-thread AVX2) and OpenMP+SIMD
    # - default (no coord-prop): index-based
    ("cpu-simd-index-aos", {"backend": "simd", "soa": False, "coord_prop": False}),
    ("cpu-simd-index-soa", {"backend": "simd", "soa": True,  "coord_prop": False}),
    ("cpu-omp-simd-index-aos", {"backend": "omp_simd", "soa": False, "coord_prop": False}),
    ("cpu-omp-simd-index-soa", {"backend": "omp_simd", "soa": True,  "coord_prop": False}),
    # - coord-prop: opt-in via --use-coord-prop
    ("cpu-simd-coord-aos", {"backend": "simd", "soa": False, "coord_prop": True}),
    ("cpu-simd-coord-soa", {"backend": "simd", "soa": True,  "coord_prop": True}),
    ("cpu-omp-simd-coord-aos", {"backend": "omp_simd", "soa": False, "coord_prop": True}),
    ("cpu-omp-simd-coord-soa", {"backend": "omp_simd", "soa": True,  "coord_prop": True}),
    # 之後可以加:
    # ("cuda-basic", {"backend": "cuda-basic"}),
    # ("cuda-pingpong", {"backend": "cuda-pingpong"}),
]

# 問題大小（strong scaling / weak scaling 可以改這裡）
SIZES = [
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]

SEEDS = 500  # 每個 case 用同一個 seeds 數

THREADS_FOR_OMP = [1, 2, 4, 8, 16]

# Speed-oriented benchmarks:
# - We skip exact by default because it dominates runtime and is not needed for performance plots.
SKIP_EXACT = True
# Repeat each case and take median to reduce noise.
REPEATS = 10


def run_case_once(backend_name, backend_opts, width, height, threads):
    args = [str(JFA_BIN)]
    args += ["--backend", backend_opts["backend"]]
    args += ["--width", str(width), "--height", str(height)]
    args += ["--seeds", str(SEEDS)]
    args += ["--no-dump", "--csv"]  # profiling 模式，不要 I/O 噪音

    if backend_opts.get("soa", False):
        args += ["--soa"]

    if backend_opts.get("coord_prop", False):
        args += ["--use-coord-prop"]

    if SKIP_EXACT:
        args += ["--skip-exact"]

    if backend_opts["backend"] in ("omp", "omp_simd"):
        args += ["--threads", str(threads)]
    else:
        # 對 serial / CUDA 這種 model，我們統一把 threads 當成 1
        threads = 1

    print("Running:", " ".join(args))
    proc = subprocess.run(args, capture_output=True, text=True, check=True)

    csv_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("CSV,"):
            csv_line = line.strip()
            break

    if csv_line is None:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError("No CSV line found in output")

    # CSV,backend,threads,W,H,seeds,exact_ms,serial_ms,parallel_ms,diff_exact_parallel,diff_serial_parallel
    parts = csv_line.split(",")
    if len(parts) != 11:
        raise RuntimeError(f"Unexpected CSV format: {csv_line}")

    _, backend_str, threads_str, W_str, H_str, seeds_str, \
        exact_ms, serial_ms, parallel_ms, diff_exact, diff_serial = parts

    return {
        "backend_name": backend_name,        # 自己取的 label，畫圖用
        "backend_str": backend_str,          # jfa_demo 的 backend 字串
        "threads": int(threads_str),
        "width": int(W_str),
        "height": int(H_str),
        "seeds": int(seeds_str),
        "exact_ms": float(exact_ms),
        "serial_ms": float(serial_ms),
        "parallel_ms": float(parallel_ms),
        "diff_exact_parallel": int(diff_exact),
        "diff_serial_parallel": int(diff_serial),
    }

def run_case(backend_name, backend_opts, width, height, threads):
    """
    Run the same case multiple times and return the median of numeric fields.
    """
    runs = []
    for i in range(REPEATS):
        print(f"  repeat {i+1}/{REPEATS}")
        runs.append(run_case_once(backend_name, backend_opts, width, height, threads))

    # Non-numeric identity fields should be identical across repeats.
    base = runs[0].copy()
    for k in ("backend_name", "backend_str", "threads", "width", "height", "seeds"):
        for r in runs[1:]:
            if r[k] != base[k]:
                raise RuntimeError(f"Inconsistent '{k}' across repeats: {base[k]} vs {r[k]}")

    # Median numeric fields.
    for k in ("exact_ms", "serial_ms", "parallel_ms"):
        base[k] = float(statistics.median([r[k] for r in runs]))

    for k in ("diff_exact_parallel", "diff_serial_parallel"):
        base[k] = int(statistics.median([r[k] for r in runs]))

    return base


def main():
    results = []

    for W, H in SIZES:
        for backend_name, backend_opts in BACKENDS:
            if backend_opts["backend"] in ("omp", "omp_simd"):
                thread_list = THREADS_FOR_OMP
            else:
                thread_list = [1]

            for t in thread_list:
                res = run_case(backend_name, backend_opts, W, H, t)
                results.append(res)

    ROOT = Path(__file__).resolve().parent.parent
    OUT_DIR = ROOT / "results"
    OUT_DIR.mkdir(exist_ok=True)

    out_csv = OUT_DIR / "results_jfa.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "backend_name", "backend_str", "threads",
            "width", "height", "seeds",
            "exact_ms", "serial_ms", "parallel_ms",
            "diff_exact_parallel", "diff_serial_parallel",
        ])
        writer.writeheader()
        writer.writerows(results)

    print("Saved", out_csv)


if __name__ == "__main__":
    main()
