#!/usr/bin/env python3
"""
Run speedup-vs-serial experiments for:
  1) 1024x1024, seeds=1000
  2) 16384x16384, seeds=1000
  3) 16384x16384, seeds=10000000

Backends:
  - serial (baseline)
  - simd (single-thread)
  - omp (threads=1..8)
  - omp_simd (threads=1..8)

For each (case, backend, threads), run N times and take the median.
Write 3 CSVs and 3 separate PNG plots (speedup vs threads, relative to serial median).

Notes / realism:
  - 16384x16384 is ~2.68e8 pixels. CPU RAM usage is still very large; runs may OOM or take a long time depending on buffers and RAM.
  - We run all backends in the SAME algorithmic mode: Coordinate Propagation, to keep speedups comparable.
    (serial/omp use SoA coord-prop; simd/omp_simd use coord-prop + packed coord buffer.)
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CSV_RE = re.compile(r"^CSV,.*$", re.MULTILINE)

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class Case:
    name: str
    width: int
    height: int
    seeds: int


@dataclass
class RunResult:
    backend: str
    threads: int
    median_ms: float
    samples_ms: List[float]
    speedup_vs_serial: float


def _pin_for(backend: str, threads: int) -> str:
    # Similar to scripts/run_experiments_median.py pinning.
    if backend in ("serial", "simd"):
        return "0"
    if backend in ("omp", "omp_simd"):
        return f"0-{threads-1}"
    return "0"


def _run_cmd(cmd: List[str], env: Dict[str, str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}\n\nOutput:\n{p.stdout}")
    return p.stdout


def _parse_csv_line(stdout: str) -> Tuple[str, int, int, int, int, float, float, float]:
    """
    Parse the CSV line emitted by jfa_demo:
      CSV,backend,threads,width,height,seeds,exact_ms,serial_ms,parallel_ms,diff_exact_parallel,diff_serial_parallel
    Returns: (backend, threads, w, h, seeds, exact_ms, serial_ms, parallel_ms)
    """
    m = CSV_RE.findall(stdout)
    if not m:
        raise RuntimeError(f"Could not find CSV line in output.\n\nOutput:\n{stdout}")
    line = m[-1].strip()
    parts = line.split(",")
    if len(parts) < 11 or parts[0] != "CSV":
        raise RuntimeError(f"Malformed CSV line: {line}")
    backend = parts[1]
    threads = int(parts[2])
    w = int(parts[3])
    h = int(parts[4])
    seeds = int(parts[5])
    exact_ms = float(parts[6])
    serial_ms = float(parts[7])
    parallel_ms = float(parts[8])
    return backend, threads, w, h, seeds, exact_ms, serial_ms, parallel_ms


def _jfa_args_common(case: Case, rng_seed: int, coord_prop: bool) -> List[str]:
    args = [
        "--width", str(case.width),
        "--height", str(case.height),
        "--seeds", str(case.seeds),
        "--seed", str(rng_seed),
        "--no-dump",
        "--skip-exact",
        "--csv",
    ]
    if coord_prop:
        args.append("--use-coord-prop")
    return args


def _measure_median_ms(
    exe: str,
    backend: str,
    threads: int,
    case: Case,
    repeats: int,
    rng_seed: int,
    pin: bool,
    coord_prop: bool,
    extra_args: Optional[List[str]] = None,
    skip_serial: bool = True,
    progress=None,
    progress_label: str = "",
) -> Tuple[float, List[float]]:
    extra_args = extra_args or []

    cmd_base = [exe, "--backend", backend] + _jfa_args_common(case, rng_seed=rng_seed, coord_prop=coord_prop)
    if backend in ("omp", "omp_simd"):
        cmd_base += ["--threads", str(threads)]

    if skip_serial:
        cmd_base += ["--skip-serial"]

    cmd_base += extra_args

    if pin:
        cmd = ["taskset", "-c", _pin_for(backend, threads)] + cmd_base
    else:
        cmd = cmd_base

    env = dict(os.environ)
    # Encourage stable OpenMP placement.
    env.setdefault("OMP_PROC_BIND", "true")
    env.setdefault("OMP_PLACES", "cores")

    samples: List[float] = []
    for _ in range(repeats):
        out = _run_cmd(cmd, env=env)
        _, _, _, _, _, _, _, parallel_ms = _parse_csv_line(out)
        samples.append(float(parallel_ms))
        if progress is not None:
            try:
                if progress_label:
                    progress.set_postfix_str(progress_label, refresh=False)
                progress.update(1)
            except Exception:
                # best-effort only
                pass

    return statistics.median(samples), samples


def _write_case_csv(out_csv: Path, case: Case, serial_median_ms: float, results: List[RunResult]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("case,width,height,seeds,serial_median_ms,backend,threads,median_ms,speedup_vs_serial,samples_ms\n")
        for r in sorted(results, key=lambda x: (x.backend, x.threads)):
            samples = ";".join(f"{v:.6f}" for v in r.samples_ms)
            f.write(
                f"{case.name},{case.width},{case.height},{case.seeds},"
                f"{serial_median_ms:.6f},{r.backend},{r.threads},"
                f"{r.median_ms:.6f},{r.speedup_vs_serial:.6f},{samples}\n"
            )


def _plot_case(out_png: Path, case: Case, serial_median_ms: float, results: List[RunResult]) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Extract series.
    omp = {r.threads: r.speedup_vs_serial for r in results if r.backend == "omp"}
    omp_simd = {r.threads: r.speedup_vs_serial for r in results if r.backend == "omp_simd"}
    simd = next((r for r in results if r.backend == "simd"), None)

    xs = list(range(1, 9))
    omp_y = [omp.get(t, float("nan")) for t in xs]
    omp_simd_y = [omp_simd.get(t, float("nan")) for t in xs]

    plt.figure(figsize=(8.5, 5.0), dpi=140)
    plt.plot(xs, omp_y, marker="o", label="omp")
    plt.plot(xs, omp_simd_y, marker="o", label="omp_simd")
    if simd is not None:
        plt.axhline(simd.speedup_vs_serial, linestyle="--", label=f"simd (1T) = {simd.speedup_vs_serial:.2f}x")

    plt.xticks(xs)
    plt.xlabel("threads")
    plt.ylabel("speedup vs serial (median)")
    plt.title(f"Speedup vs Serial — {case.width}x{case.height}, seeds={case.seeds} (serial median={serial_median_ms:.1f} ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(Path("build") / "jfa_demo"), help="Path to jfa_demo executable")
    ap.add_argument("--out-dir", default="results", help="Output directory for CSV/PNG")
    ap.add_argument("--repeats", type=int, default=9, help="Runs per measurement (median over repeats)")
    ap.add_argument("--rng-seed", type=int, default=42, help="RNG seed for seed generation (keep fixed)")
    ap.add_argument("--pin", action="store_true", help="Pin CPU cores using taskset (recommended)")
    ap.add_argument("--no-pin", dest="pin", action="store_false", help="Do not pin cores")
    ap.set_defaults(pin=True)
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    ap.add_argument("--dry-run", action="store_true", help="Print planned cases and exit")
    args = ap.parse_args()

    exe = str(Path(args.exe))
    if not Path(exe).exists():
        print(f"ERROR: executable not found: {exe}", file=sys.stderr)
        return 2

    if args.repeats < 1:
        print("ERROR: repeats must be >= 1", file=sys.stderr)
        return 2

    cases = [
        Case("small_1024_s1000", 1024, 1024, 1000),
        Case("big_16384_s1000", 16384, 16384, 1000),
        Case("big_16384_s10000000", 16384, 16384, 10_000_000),
    ]

    # We keep mode consistent across backends for comparable speedups.
    coord_prop = True
    simd_extra = ["--cpu-coordbuf-layout", "packed"]  # make explicit

    if args.dry_run:
        for c in cases:
            print(f"- {c.name}: {c.width}x{c.height}, seeds={c.seeds}, repeats={args.repeats}, pin={args.pin}")
        print("Mode: coord-prop enabled for all; simd/omp_simd use packed coord buffer.")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        print(f"\n=== Case: {case.name} ({case.width}x{case.height}, seeds={case.seeds}) ===")

        # 18 measurement points per case:
        # - serial (1)
        # - simd (1)
        # - omp t=1..8 (8)
        # - omp_simd t=1..8 (8)
        total_runs = 18 * args.repeats
        if (not args.no_progress) and tqdm is not None:
            pbar = tqdm(total=total_runs, desc=case.name, unit="run")
        else:
            pbar = None
            done = 0

            class _SimpleProgress:
                def update(self, n: int) -> None:
                    nonlocal done
                    done += n
                    # print occasionally to avoid spam
                    if done == total_runs or done % max(1, total_runs // 50) == 0:
                        print(f"[{case.name}] progress {done}/{total_runs} runs", flush=True)

                def set_postfix_str(self, s: str, refresh: bool = False) -> None:
                    return

            if not args.no_progress:
                pbar = _SimpleProgress()

        # 1) serial baseline (do NOT skip serial; measure its parallel_ms which equals serial_ms)
        serial_med, serial_samples = _measure_median_ms(
            exe=exe,
            backend="serial",
            threads=1,
            case=case,
            repeats=args.repeats,
            rng_seed=args.rng_seed,
            pin=args.pin,
            coord_prop=coord_prop,
            extra_args=[],
            skip_serial=False,
            progress=pbar,
            progress_label="serial",
        )
        print(f"serial median = {serial_med:.3f} ms")

        results: List[RunResult] = []
        results.append(
            RunResult(
                backend="serial",
                threads=1,
                median_ms=serial_med,
                samples_ms=serial_samples,
                speedup_vs_serial=1.0,
            )
        )

        # 2) simd (single-thread)
        simd_med, simd_samples = _measure_median_ms(
            exe=exe,
            backend="simd",
            threads=1,
            case=case,
            repeats=args.repeats,
            rng_seed=args.rng_seed,
            pin=args.pin,
            coord_prop=coord_prop,
            extra_args=simd_extra,
            skip_serial=True,
            progress=pbar,
            progress_label="simd t=1",
        )
        results.append(
            RunResult(
                backend="simd",
                threads=1,
                median_ms=simd_med,
                samples_ms=simd_samples,
                speedup_vs_serial=serial_med / simd_med,
            )
        )
        print(f"simd (1T) median = {simd_med:.3f} ms, speedup={serial_med/simd_med:.2f}x")

        # 3) omp and omp_simd for threads 1..8
        for t in range(1, 9):
            omp_med, omp_samples = _measure_median_ms(
                exe=exe,
                backend="omp",
                threads=t,
                case=case,
                repeats=args.repeats,
                rng_seed=args.rng_seed,
                pin=args.pin,
                coord_prop=coord_prop,
                extra_args=[],
                skip_serial=True,
                progress=pbar,
                progress_label=f"omp t={t}",
            )
            results.append(
                RunResult(
                    backend="omp",
                    threads=t,
                    median_ms=omp_med,
                    samples_ms=omp_samples,
                    speedup_vs_serial=serial_med / omp_med,
                )
            )
            print(f"omp t={t}: {omp_med:.3f} ms, speedup={serial_med/omp_med:.2f}x")

            omp_simd_med, omp_simd_samples = _measure_median_ms(
                exe=exe,
                backend="omp_simd",
                threads=t,
                case=case,
                repeats=args.repeats,
                rng_seed=args.rng_seed,
                pin=args.pin,
                coord_prop=coord_prop,
                extra_args=simd_extra,
                skip_serial=True,
                progress=pbar,
                progress_label=f"omp_simd t={t}",
            )
            results.append(
                RunResult(
                    backend="omp_simd",
                    threads=t,
                    median_ms=omp_simd_med,
                    samples_ms=omp_simd_samples,
                    speedup_vs_serial=serial_med / omp_simd_med,
                )
            )
            print(f"omp_simd t={t}: {omp_simd_med:.3f} ms, speedup={serial_med/omp_simd_med:.2f}x")

        out_csv = out_dir / f"speedup_{case.name}.csv"
        out_png = out_dir / f"speedup_{case.name}.png"
        _write_case_csv(out_csv, case, serial_med, results)
        _plot_case(out_png, case, serial_med, results)

        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_png}")
        try:
            close_fn = getattr(pbar, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


