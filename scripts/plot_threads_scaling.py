#!/usr/bin/env python3
"""
Plot CPU backend speedup-vs-threads curves from run_experiments_median.py output.

Expected CSV columns:
  width,height,seeds,method,threads,median_ms,serial_median_ms,speedup_vs_serial

We plot:
- OpenMP (omp_tN) speedup vs threads
- OpenMP+SIMD (omp_simd_tN) speedup vs threads
- SIMD (single-thread) as a horizontal line (optional; pulled from a separate SIMD CSV if provided)

Theory lines:
- Ideal OpenMP scaling: speedup = N
- Ideal OpenMP+SIMD scaling: speedup = N * (SIMD_1T_speedup)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _to_num(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df[col], errors="coerce")


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("width", "height", "seeds", "threads"):
        _to_num(df, c)
    for c in ("median_ms", "serial_median_ms", "speedup_vs_serial"):
        if c in df.columns:
            _to_num(df, c)
    return df


def _simd_1t_speedup_map(simd_csv: Path) -> Dict[Tuple[int, int, int], float]:
    """
    Extract SIMD(1 thread) speedup_vs_serial per (W,H,seeds) from a 'simd layouts' CSV.
    We pick method=simd_index_seeds_soa if present; otherwise, fall back to best SIMD method.
    """
    df = _load_csv(simd_csv)
    df = df[df["status"] == "ok"].copy()
    df = df[df["backend"] == "simd"].copy()
    out: Dict[Tuple[int, int, int], float] = {}
    if df.empty:
        return out

    # Prefer simd_index_seeds_soa; else pick min median_ms among SIMD methods.
    for (w, h, seeds), g in df.groupby(["width", "height", "seeds"]):
        g2 = g[g["method"] == "simd_index_seeds_soa"]
        if not g2.empty:
            row = g2.sort_values("median_ms").iloc[0]
        else:
            row = g.sort_values("median_ms").iloc[0]
        sp = float(row["speedup_vs_serial"])
        out[(int(w), int(h), int(seeds))] = sp
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to bench_threads_scaling.csv")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument(
        "--simd-csv",
        default="",
        help="Optional: CSV containing simd_index_seeds_soa rows (e.g. results/bench_median_simd_layouts.csv)",
    )
    ap.add_argument("--title", default="CPU Speedup vs Threads", help="Figure title")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    simd_csv = Path(args.simd_csv) if args.simd_csv else None

    df = _load_csv(csv_path)
    df = df[df["status"] == "ok"].copy()

    # Scenarios present in scaling CSV.
    scenarios = sorted(df[["width", "height", "seeds"]].drop_duplicates().itertuples(index=False, name=None))
    if not scenarios:
        raise SystemExit("No ok rows found in scaling CSV.")

    simd_map: Dict[Tuple[int, int, int], float] = {}
    if simd_csv and simd_csv.exists():
        simd_map = _simd_1t_speedup_map(simd_csv)

    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(args.title)

    for ax, (w, h, seeds) in zip(axes, scenarios):
        sub = df[(df["width"] == w) & (df["height"] == h) & (df["seeds"] == seeds)]
        if sub.empty:
            continue

        # Use speedup_vs_serial from CSV (already computed from medians).
        omp = sub[sub["method"].str.startswith("omp_t")].sort_values("threads")
        omp_simd = sub[sub["method"].str.startswith("omp_simd_t")].sort_values("threads")

        ax.plot(omp["threads"], omp["speedup_vs_serial"], marker="o", label="omp")
        ax.plot(omp_simd["threads"], omp_simd["speedup_vs_serial"], marker="o", label="omp_simd")

        # SIMD(1T) horizontal line: prefer from simd-csv; else approximate with omp_simd_t1.
        key = (int(w), int(h), int(seeds))
        simd_1t = simd_map.get(key, None)
        if simd_1t is None:
            t1 = omp_simd[omp_simd["threads"] == 1]
            if not t1.empty:
                simd_1t = float(t1.iloc[0]["speedup_vs_serial"])

        if simd_1t is not None:
            ax.hlines(simd_1t, xmin=1, xmax=8, linestyles=":", color="C2", label="simd (1T)")

        # Theory lines
        xs = list(range(1, 9))
        ax.plot(xs, xs, linestyle="--", color="gray", linewidth=1.5, label="ideal omp: y=N")
        if simd_1t is not None:
            ax.plot(xs, [x * simd_1t for x in xs], linestyle="--", color="black", linewidth=1.5, label="ideal omp_simd: y=N*SIMD(1T)")

        ax.set_title(f"{int(w)}x{int(h)}, seeds={int(seeds)}")
        ax.set_xlabel("threads")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    axes[0].set_ylabel("speedup vs serial (median)")
    fig.tight_layout(rect=(0, 0.0, 1, 0.92))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


