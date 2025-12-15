#!/usr/bin/env python3
"""
Run stable JFA performance experiments and report median timings.

Design goals:
- Repeat each measurement N times (default 15) and report median (plus quartiles).
- Sweep over image sizes (WxH) and seed counts.
- Compare multiple CPU backends/modes (serial / simd / omp / omp_simd) + selected flags.
- Pin CPU cores via taskset and pin OpenMP threads via env vars for stability.

Example:
  cd /workspaces/parallel-jump-flooding-voronoi
  python3 scripts/run_experiments_median.py \\
    --exe /workspaces/parallel-jump-flooding-voronoi/build/jfa_demo \\
    --sizes 1024x1024 2048x2048 4096x4096 8192x8192 \\
    --seeds 50 2048 10000 500000 \\
    --threads 8 \\
    --repeats 15 --warmup 1 \\
    --out /workspaces/parallel-jump-flooding-voronoi/results/bench_median.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SERIAL_RE = re.compile(r"\[Serial JFA\]\s+time\s*=\s*([0-9.]+)\s*ms")
SIMD_RE = re.compile(r"\[SIMD JFA\]\s+time\s*=\s*([0-9.]+)\s*ms")
OMP_RE = re.compile(r"\[OpenMP JFA\]\s+threads\s*=\s*(\d+),\s*time\s*=\s*([0-9.]+)\s*ms")
OMP_SIMD_RE = re.compile(r"\[OpenMP\+SIMD JFA\]\s+threads\s*=\s*(\d+),\s*time\s*=\s*([0-9.]+)\s*ms")


def parse_size(s: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid size '{s}'. Use WxH like 2048x2048.")
    w = int(m.group(1))
    h = int(m.group(2))
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Width/height must be positive.")
    return w, h


def median_ms(xs: List[float]) -> float:
    return statistics.median(xs)


def percentile(xs: List[float], p: float) -> float:
    """Nearest-rank percentile on sorted data, p in [0,1]."""
    if not xs:
        raise ValueError("Empty sample list")
    ys = sorted(xs)
    k = max(1, min(len(ys), int(round(p * len(ys)))))
    return ys[k - 1]


def run_cmd(cmd: List[str], env: Dict[str, str]) -> str:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"exit: {p.returncode}\n"
            f"output:\n{p.stdout}"
        )
    return p.stdout


def parse_time_ms(output: str, backend: str) -> float:
    if backend == "serial":
        m = SERIAL_RE.search(output)
        if not m:
            raise ValueError("Failed to parse [Serial JFA] time from output")
        return float(m.group(1))
    if backend == "simd":
        m = SIMD_RE.search(output)
        if not m:
            raise ValueError("Failed to parse [SIMD JFA] time from output")
        return float(m.group(1))
    if backend == "omp":
        m = OMP_RE.search(output)
        if not m:
            raise ValueError("Failed to parse [OpenMP JFA] time from output")
        return float(m.group(2))
    if backend == "omp_simd":
        m = OMP_SIMD_RE.search(output)
        if not m:
            raise ValueError("Failed to parse [OpenMP+SIMD JFA] time from output")
        return float(m.group(2))
    raise ValueError(f"Unknown backend '{backend}'")

def fmt_eta(seconds: float) -> str:
    if seconds != seconds or seconds < 0:  # NaN/negative
        return "ETA ?"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"ETA {h:d}:{m:02d}:{sec:02d}"
    return f"ETA {m:d}:{sec:02d}"


def progress_line(done: int, total: int, desc: str, start_t: float) -> str:
    # Simple ASCII bar.
    width = 28
    frac = 0.0 if total <= 0 else min(1.0, max(0.0, done / total))
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - start_t
    eta = (elapsed / done) * (total - done) if done > 0 else float("nan")
    return f"[{bar}] {done}/{total}  {desc}  {fmt_eta(eta)}"


def print_progress(done: int, total: int, desc: str, start_t: float) -> None:
    line = progress_line(done, total, desc, start_t)
    # Write to stderr to keep stdout clean for piping/CSV.
    sys.stderr.write("\r" + line[:200] + " " * 10)  # trim + clear tail
    sys.stderr.flush()


@dataclass(frozen=True)
class Method:
    name: str
    backend: str
    threads: int
    extra_args: Tuple[str, ...]

    def args(self, w: int, h: int, seeds: int, rng_seed: int) -> List[str]:
        args: List[str] = [
            "--backend", self.backend,
            "--width", str(w),
            "--height", str(h),
            "--seeds", str(seeds),
            "--seed", str(rng_seed),
            "--no-dump",
            "--skip-exact",
        ]
        # Only measure this backend (avoid baseline runs) unless it's serial baseline.
        if self.backend != "serial":
            args.append("--skip-serial")
        if self.backend in ("omp", "omp_simd"):
            args += ["--threads", str(self.threads)]
        args += list(self.extra_args)
        return args

    def pin_cores(self) -> str:
        # For stability:
        # - serial/simd: pin to core 0
        # - omp/omp_simd: pin to first N cores
        if self.backend in ("serial", "simd"):
            return "0"
        if self.backend in ("omp", "omp_simd"):
            return f"0-{self.threads-1}"
        return "0"


def measure_method(
    exe: str,
    method: Method,
    w: int,
    h: int,
    seeds: int,
    rng_seed: int,
    warmup: int,
    repeats: int,
    base_env: Dict[str, str],
) -> List[float]:
    cmd_base = [exe] + method.args(w, h, seeds, rng_seed)
    pin = method.pin_cores()

    # Wrap with taskset for CPU pinning.
    cmd = ["taskset", "-c", pin] + cmd_base

    samples: List[float] = []
    total_runs = warmup + repeats
    for r in range(total_runs):
        out = run_cmd(cmd, env=base_env)
        ms = parse_time_ms(out, backend=method.backend)
        if r >= warmup:
            samples.append(ms)
    return samples


def default_methods(threads_list: List[int], include_simd_layouts: bool) -> List[Method]:
    methods: List[Method] = []

    # Serial baseline (note: jfa_demo's serial backend prints [Serial JFA] time; it may do extra work,
    # but we parse the baseline serial time line for consistency.)
    methods.append(Method("serial_baseline", "serial", 1, ()))

    # SIMD index-based (fastest general path)
    # Explicitly pin the layout to make CSV easier to interpret.
    methods.append(Method("simd_index_seeds_packed", "simd", 1, ("--cpu-seeds-layout", "packed")))
    methods.append(Method("simd_index_seeds_packed_cpu_pitch", "simd", 1, ("--cpu-seeds-layout", "packed", "--cpu-pitch")))

    # SIMD coord-prop (CPU internal coord buffer layout; --soa does not affect CPU SIMD)
    methods.append(Method("simd_coordprop_coordbuf_soa", "simd", 1, ("--use-coord-prop", "--cpu-coordbuf-layout", "soa")))

    if include_simd_layouts:
        # Seeds-layout sweep (index-based mode)
        methods.append(Method("simd_index_seeds_soa", "simd", 1, ("--cpu-seeds-layout", "soa")))
        methods.append(Method("simd_index_seeds_aos", "simd", 1, ("--cpu-seeds-layout", "aos")))

        # Coord-buffer-layout sweep (coord-prop mode)
        methods.append(Method("simd_coordprop_coordbuf_aos", "simd", 1, ("--use-coord-prop", "--cpu-coordbuf-layout", "aos")))

    # OpenMP-only and OpenMP+SIMD
    for t in threads_list:
        methods.append(Method(f"omp_t{t}", "omp", t, ()))
        methods.append(Method(f"omp_simd_t{t}", "omp_simd", t, ()))

    return methods


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="Path to jfa_demo executable")
    ap.add_argument("--sizes", nargs="+", type=parse_size, required=True, help="List of WxH sizes, e.g. 2048x2048")
    ap.add_argument("--seeds", nargs="+", type=int, required=True, help="List of seed counts")
    ap.add_argument("--threads", nargs="+", type=int, default=[8], help="OpenMP thread counts to test (default: 8)")
    ap.add_argument("--rng-seed", type=int, default=42, help="RNG seed for seed generation (default: 42)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs per method (not recorded) (default: 1)")
    ap.add_argument("--repeats", type=int, default=15, help="Recorded runs per method (default: 15)")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between runs (default: 0)")
    ap.add_argument(
        "--allow-oversubscribe-seeds",
        action="store_true",
        help="Allow seeds > (width*height). By default such cases are skipped as they are not meaningful for random unique seed placement.",
    )
    ap.add_argument("--no-progress", action="store_true", help="Disable progress output")
    ap.add_argument(
        "--include-simd-layouts",
        action="store_true",
        help="Include CPU SIMD SoA/AoS layout comparison methods (more runs).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned experiments and exit")
    args = ap.parse_args()

    exe = str(Path(args.exe))
    if not Path(exe).exists():
        print(f"ERROR: executable not found: {exe}", file=sys.stderr)
        return 2

    for s in args.seeds:
        if s <= 0:
            print("ERROR: seeds must be > 0", file=sys.stderr)
            return 2

    if args.repeats < 1:
        print("ERROR: repeats must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("ERROR: warmup must be >= 0", file=sys.stderr)
        return 2

    # Stable OpenMP settings
    base_env = os.environ.copy()
    base_env["OMP_PROC_BIND"] = "true"
    base_env["OMP_PLACES"] = "cores"
    base_env["OMP_DYNAMIC"] = "false"

    methods = default_methods(args.threads, include_simd_layouts=args.include_simd_layouts)

    plan = [(w, h, seeds, m) for (w, h) in args.sizes for seeds in args.seeds for m in methods]

    if args.dry_run:
        print(f"Planned experiments: {len(plan)}")
        for (w, h, seeds, m) in plan[:30]:
            print(f"- {w}x{h} seeds={seeds} method={m.name} backend={m.backend} threads={m.threads} args={list(m.extra_args)}")
        if len(plan) > 30:
            print("... (truncated)")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "width", "height", "seeds", "rng_seed",
        "method", "backend", "threads", "extra_args",
        "warmup", "repeats",
        "status", "note",
        "median_ms", "p25_ms", "p75_ms", "min_ms", "max_ms",
        "serial_median_ms", "speedup_vs_serial",
        "samples_ms",
    ]

    t0 = time.time()
    t0m = time.monotonic()
    with out_path.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()

        # Precompute total tasks (method-level), excluding skipped size/seed combos.
        total_tasks = 0
        for (w, h) in args.sizes:
            for seeds in args.seeds:
                if (seeds > w * h) and (not args.allow_oversubscribe_seeds):
                    continue
                total_tasks += len(methods)
        done_tasks = 0
        if not args.no_progress:
            print_progress(done_tasks, total_tasks, "starting…", t0m)

        for (w, h) in args.sizes:
            for seeds in args.seeds:
                max_unique = w * h
                if (seeds > max_unique) and (not args.allow_oversubscribe_seeds):
                    note = f"skipped: seeds ({seeds}) > width*height ({max_unique})"
                    # Write a single marker row (serial_baseline) to make the skip explicit in CSV.
                    serial_method = next(m for m in methods if m.name == "serial_baseline")
                    wcsv.writerow(
                        {
                            "width": w,
                            "height": h,
                            "seeds": seeds,
                            "rng_seed": args.rng_seed,
                            "method": serial_method.name,
                            "backend": serial_method.backend,
                            "threads": serial_method.threads,
                            "extra_args": " ".join(serial_method.extra_args),
                            "warmup": args.warmup,
                            "repeats": args.repeats,
                            "status": "skipped",
                            "note": note,
                            "median_ms": "",
                            "p25_ms": "",
                            "p75_ms": "",
                            "min_ms": "",
                            "max_ms": "",
                            "serial_median_ms": "",
                            "speedup_vs_serial": "",
                            "samples_ms": "",
                        }
                    )
                    f.flush()
                    continue

                # 1) measure serial baseline first (for speedup computation)
                serial_method = next(m for m in methods if m.name == "serial_baseline")
                if not args.no_progress:
                    print_progress(
                        done_tasks, total_tasks,
                        f"{w}x{h} seeds={seeds}  {serial_method.name}",
                        t0m,
                    )
                serial_samples = measure_method(
                    exe=exe,
                    method=serial_method,
                    w=w, h=h, seeds=seeds,
                    rng_seed=args.rng_seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    base_env=base_env,
                )
                serial_med = median_ms(serial_samples)
                done_tasks += 1
                if not args.no_progress:
                    print_progress(
                        done_tasks, total_tasks,
                        f"{w}x{h} seeds={seeds}  {serial_method.name}",
                        t0m,
                    )

                row = {
                    "width": w,
                    "height": h,
                    "seeds": seeds,
                    "rng_seed": args.rng_seed,
                    "method": serial_method.name,
                    "backend": serial_method.backend,
                    "threads": serial_method.threads,
                    "extra_args": " ".join(serial_method.extra_args),
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "status": "ok",
                    "note": "",
                    "median_ms": f"{serial_med:.6f}",
                    "p25_ms": f"{percentile(serial_samples, 0.25):.6f}",
                    "p75_ms": f"{percentile(serial_samples, 0.75):.6f}",
                    "min_ms": f"{min(serial_samples):.6f}",
                    "max_ms": f"{max(serial_samples):.6f}",
                    "serial_median_ms": f"{serial_med:.6f}",
                    "speedup_vs_serial": f"{1.0:.6f}",
                    "samples_ms": ";".join(f"{x:.6f}" for x in serial_samples),
                }
                wcsv.writerow(row)
                f.flush()

                # 2) measure other methods and compute speedup vs serial median
                for m in methods:
                    if m.name == "serial_baseline":
                        continue
                    if not args.no_progress:
                        print_progress(
                            done_tasks, total_tasks,
                            f"{w}x{h} seeds={seeds}  {m.name}",
                            t0m,
                        )
                    samples = measure_method(
                        exe=exe,
                        method=m,
                        w=w, h=h, seeds=seeds,
                        rng_seed=args.rng_seed,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        base_env=base_env,
                    )
                    med = median_ms(samples)
                    speedup = serial_med / med if med > 0 else float("nan")
                    done_tasks += 1
                    if not args.no_progress:
                        print_progress(
                            done_tasks, total_tasks,
                            f"{w}x{h} seeds={seeds}  {m.name}",
                            t0m,
                        )

                    row = {
                        "width": w,
                        "height": h,
                        "seeds": seeds,
                        "rng_seed": args.rng_seed,
                        "method": m.name,
                        "backend": m.backend,
                        "threads": m.threads,
                        "extra_args": " ".join(m.extra_args),
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "status": "ok",
                        "note": "",
                        "median_ms": f"{med:.6f}",
                        "p25_ms": f"{percentile(samples, 0.25):.6f}",
                        "p75_ms": f"{percentile(samples, 0.75):.6f}",
                        "min_ms": f"{min(samples):.6f}",
                        "max_ms": f"{max(samples):.6f}",
                        "serial_median_ms": f"{serial_med:.6f}",
                        "speedup_vs_serial": f"{speedup:.6f}",
                        "samples_ms": ";".join(f"{x:.6f}" for x in samples),
                    }
                    wcsv.writerow(row)
                    f.flush()

                    if args.sleep > 0:
                        time.sleep(args.sleep)

    dt = time.time() - t0
    if not args.no_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()
    print(f"Done. Wrote: {out_path}  (elapsed {dt/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


