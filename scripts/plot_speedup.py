#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "results_jfa.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    # CPU strong scaling, with serial baseline
    sizes = df[["width", "height"]].drop_duplicates().sort_values(["width", "height"])

    # 做一張「每個 size 一條線」的 speedup vs threads 圖
    plt.figure()

    for _, row in sizes.iterrows():
        W = int(row["width"])
        H = int(row["height"])

        df_size = df[(df["width"] == W) & (df["height"] == H)]

        # baseline: serial JFA 的 serial_ms
        df_serial = df_size[df_size["backend_str"] == "serial"]
        if df_serial.empty:
            print(f"[WARN] No serial baseline for size {W}x{H}, skip")
            continue

        serial_ms = df_serial["serial_ms"].iloc[0]

        label = f"{W}x{H}"

        # Plot by backend_name to avoid mixing multiple variants (AoS/SoA, coord/index).
        for backend_name, marker in [
            ("cpu-omp", "o"),
            ("cpu-omp-simd-index-soa", "s"),
            ("cpu-omp-simd-coord-soa", "^"),
        ]:
            df_b = df_size[df_size["backend_name"] == backend_name].copy()
            if df_b.empty:
                continue
            df_b.sort_values("threads", inplace=True)
            df_b["speedup"] = serial_ms / df_b["parallel_ms"]
            plt.plot(df_b["threads"], df_b["speedup"], marker=marker, label=f"{label} ({backend_name})")

    plt.xlabel("Threads")
    plt.ylabel("Speedup vs serial JFA")
    plt.title("CPU JFA strong scaling (selected variants)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "speedup_omp_strong_scaling.png", dpi=200)

    # 再畫一張 efficiency vs threads（只示範最大 size）
    max_area = (df["width"] * df["height"]).max()
    df_max = df[(df["width"] * df["height"]) == max_area]
    Wm = int(df_max["width"].iloc[0])
    Hm = int(df_max["height"].iloc[0])

    df_serial = df_max[df_max["backend_str"] == "serial"]
    serial_ms = df_serial["serial_ms"].iloc[0]
    plt.figure()

    for backend_name, marker in [
        ("cpu-omp", "o"),
        ("cpu-omp-simd-index-soa", "s"),
        ("cpu-omp-simd-coord-soa", "^"),
    ]:
        df_b = df_max[df_max["backend_name"] == backend_name].copy()
        if df_b.empty:
            continue
        df_b.sort_values("threads", inplace=True)
        df_b["speedup"] = serial_ms / df_b["parallel_ms"]
        df_b["efficiency"] = df_b["speedup"] / df_b["threads"]
        plt.plot(df_b["threads"], df_b["efficiency"], marker=marker, label=backend_name)

    plt.xlabel("Threads")
    plt.ylabel("Parallel efficiency")
    plt.title(f"CPU JFA efficiency ({Wm}x{Hm})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "efficiency_omp_maxsize.png", dpi=200)

    print("Saved plots: speedup_omp_strong_scaling.png, efficiency_omp_maxsize.png")

if __name__ == "__main__":
    main()
