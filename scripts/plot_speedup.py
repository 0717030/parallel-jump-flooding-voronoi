#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "results_jfa.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    # 只看 CPU OMP strong scaling
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

        # OMP rows
        df_omp = df_size[df_size["backend_str"] == "omp"].copy()
        if df_omp.empty:
            print(f"[WARN] No OMP rows for size {W}x{H}, skip")
            continue

        df_omp.sort_values("threads", inplace=True)
        df_omp["speedup"] = serial_ms / df_omp["parallel_ms"]
        df_omp["efficiency"] = df_omp["speedup"] / df_omp["threads"]

        # 畫 speedup 線
        label = f"{W}x{H}"
        plt.plot(df_omp["threads"], df_omp["speedup"], marker="o", label=label)

    plt.xlabel("Threads")
    plt.ylabel("Speedup vs serial JFA")
    plt.title("OpenMP JFA strong scaling")
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
    df_omp = df_max[df_max["backend_str"] == "omp"].copy()
    df_omp.sort_values("threads", inplace=True)
    df_omp["speedup"] = serial_ms / df_omp["parallel_ms"]
    df_omp["efficiency"] = df_omp["speedup"] / df_omp["threads"]

    plt.figure()
    plt.plot(df_omp["threads"], df_omp["efficiency"], marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Parallel efficiency")
    plt.title(f"OpenMP JFA efficiency ({Wm}x{Hm})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "efficiency_omp_maxsize.png", dpi=200)

    print("Saved plots: speedup_omp_strong_scaling.png, efficiency_omp_maxsize.png")

if __name__ == "__main__":
    main()
