# parallel-jump-flooding-voronoi

```
parallel-jump-flooding-voronoi/
│
├── CMakeLists.txt          # 主 CMake 設定，定義 jfa_cpu, jfa_exact, jfa_demo 等 target
├── README.md               # 專案簡介＋如何 build / run 的說明
├── LICENSE                 # 授權條款
├── .gitignore              # 忽略 build/、output/ 等
│
├── build/                  # CMake out-of-source build (完全可以 rm -rf)
│   └── ...                 # jfa_demo, libjfa_cpu.a, etc.
│
├── output/                 # 視覺化輸出（PPM / MP4）─ 依實驗 case 分資料夾
│   ├── case1_serial_t1_512x512_s100/
│   │   ├── cpu_jfa_serial_pass_000_k256.ppm
│   │   └── ...
│   ├── case1_omp_t4_512x512_s100/
│   │   ├── cpu_jfa_omp_pass_000_k256.ppm
│   │   └── ...
│   └── case1_omp_t16_512x512_s100/
│       └── ...
│
├── results/                # 實驗數據與圖表（report 直接用這裡的東西）
│   ├── results_jfa.csv
│   ├── speedup_omp_strong_scaling.png
│   └── efficiency_omp_maxsize.png
│
├── include/
│   └── jfa/                # 所有 public header（給 src/* 共用）
│       ├── types.hpp       # Config, Seed, SeedIndexBuffer, RGBImage 等 core 型別
│       ├── cpu.hpp         # jfa_cpu_serial, jfa_cpu_omp 的宣告（CPU JFA API）
│       ├── exact.hpp       # voronoi_exact_cpu 的宣告（exact baseline API）
│       ├── gpu.hpp         # CUDA JFA 的宣告
│       └── visualize.hpp   # make_seed_palette, seed_index_to_rgb, write_ppm 等
│
├── src/
│   ├── apps/
│   │   └── jfa_demo.cpp    # CLI 主程式：parse args → run exact/serial baseline → run selected backend → dump PPM/CSV
│   │                       # （建議刪掉 jfa_demo copy.cpp）
│   │
│   ├── common/
│   │   └── jfa_visualize.cpp   # 實作 visualize.hpp 的函式（palette, PPM dump）
│   │
│   ├── cpu/
│   │   ├── jfa_common_impl.hpp # JFA cpu 共同邏輯（step sequence、distance 函式等）
│   │   ├── jfa_serial.cpp      # 單執行緒 JFA 實作（jfa_cpu_serial）
│   │   └── jfa_openmp.cpp      # OpenMP JFA 實作（jfa_cpu_omp）
│   │   └── jfa_simd_avx2.cpp   # AVX2 SIMD / OpenMP+SIMD backend（jfa_cpu_simd / jfa_cpu_omp_simd）
│   │
│   ├── exact/
│   │   └── voronoi_exact.cpp   # O(N² * #seeds) 的精確 Voronoi baseline
│   │
│   └── gpu/
│       ├── jfa_cuda.cu         # CUDA kernel + device-side pipeline
│       └── jfa_gpu_wrapper.cpp # C++ wrapper（host allocations / seeds layout / invoke kernels）
│
├── scripts/
│   ├── run_benchmarks.py       # 跑多組 jfa_demo（serial/omp/simd/omp_simd）→ 寫 results/results_jfa.csv
│   ├── plot_speedup.py         # 讀 results_jfa.csv → 產生 speedup / efficiency 圖
│   └── render_videos.py        # 掃 output/*/ *_pass_*.ppm → 用 ffmpeg 合成 mp4 demo
│
└── tests/
    ├── CMakeLists.txt (可選)   # 之後要加單元測試再用；現在可以先放 TODO 註解
    └── data/                   # 測試用小 input（例如固定 seed set, small size）
```

## Getting started
### Clone 專案
```
git clone https://github.com/<YOUR_ORG>/parallel-jump-flooding-voronoi.git
cd parallel-jump-flooding-voronoi
```


### 依賴 (Dependencies)
* C++17 編譯器（g++ / clang++）
* CMake ≥ 3.x
* OpenMP（通常隨 gcc/clang 一起提供）
* CUDA Toolkit
*（可選）Python 3 + matplotlib：跑 benchmark / 畫圖用
*（可選）ffmpeg：把 PPM frame 合成 mp4 動畫用

### Build

專案使用 out-of-source CMake build：
```
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

成功後，會在 build/ 底下看到：
* jfa_demo – main CLI 可執行檔
* libjfa_cpu.a, libjfa_exact.a, libjfa_visualize.a, libjfa_gpu.a – 靜態 library（被 jfa_demo link）

### jfa_demo 使用方式

回到 build/ 目錄下執行：
```
cd build
./jfa_demo [options...]
```

**支援的參數**

（這對應你現在的 Options ＋ parse_args）
* `--backend {serial|omp|simd|omp_simd|cuda}`
    * 選擇要展示/測的 JFA 實作（預設：`omp`）
    * `serial`：單執行緒 JFA（參考/動畫 demo 用）
    * `omp`：OpenMP 多執行緒 JFA
    * `simd`：AVX2 SIMD（單執行緒，CPU 需支援 AVX2）
    * `omp_simd`：OpenMP + AVX2 SIMD（CPU 需支援 AVX2）
    * `cuda`：CUDA GPU backend（需要可用的 CUDA runtime / NVIDIA driver；某些環境可能能跑但效能受限）

* `--threads N`
    * OpenMP backend 使用的 threads 數
    * 僅在 `--backend omp` / `--backend omp_simd` 時有效（其他 backend 會被忽略）
    * 預設：8

* `--width W` / `--height H`
    * 圖像解析度（像素）
    * 預設：512×512

*  `--seeds N`
    * 隨機產生的 seed 數量
    * 預設：50

* `--seed N`
    * RNG seed（讓實驗可重現）
    * 預設：42

* `--no-dump`
    * 不輸出 PPM frame（只做效能測試／跑得比較快）

* `--output-dir DIR`
    * 輸出根資料夾（預設：`output`；相對於執行時的 working directory）
    * 若有 `--tag`：frame 會寫到 `DIR/<auto_name>/...`
    * 若沒有 `--tag`：frame 會直接寫到 `DIR/*.ppm`

* `--tag NAME`
    * 給這次實驗一個邏輯名字，用來組成輸出資料夾名稱
    * 實際的輸出資料夾會是：
    ```
    <output_root>/<tag>_<backend>_t<threads>_<width>x<height>_s<seeds>/
    ```
    * `threads` 只有在 `omp/omp_simd` 時才會反映實際 threads；其他 backend 會固定寫 `t1`。

* `--csv`
    * 除了人類可讀的 log 之外，再印出一行 machine-readable 的 CSV summary：
    ```
    CSV,backend,threads,width,height,seeds,exact_ms,serial_ms,parallel_ms,diff_exact_parallel,diff_serial_parallel
    ```
    * `diff_*` 欄位：如果你用 `--skip-exact` 或 `--skip-serial`，對應的 diff 會是 `-1`（表示沒有做比對）。
    * `serial_ms`：如果你用 `--skip-serial`，目前會被設成 `1.0` 來避免除以 0（因此不代表真實 serial 時間；建議 benchmark 保留 serial baseline）。

* 正確性/驗證開關（建議大圖 benchmark 用）
    * `--skip-check`：同時跳過 exact 與 serial baseline（= `--skip-exact --skip-serial`）
    * `--skip-exact`：跳過 exact baseline
    * `--skip-serial`：跳過 serial baseline（注意 `serial_ms` 會是 dummy 值）

* SIMD/CUDA 相關 flags（只有特定 backend 會真的用到；其他 backend 只是「可解析但不生效」）
    * `--soa`
        * `cuda`：seeds layout 使用 SoA（某些 kernel path）
        * `simd/omp_simd`：內部 coord buffer 使用 SoA layout（否則 AoS）
    * `--use-coord-prop`
        * `serial/omp/simd/omp_simd/cuda`：啟用 Coordinate Propagation（以 per-pixel seed coordinates 傳播，再映射回 seed index）
        * `cuda`：目前 coord-prop 模式下 callback/PPM dump 有部分行為會被限制（程式內有註解）
    * `--block-dim N` / `--block-dim-x N` / `--block-dim-y N`（`cuda`）
    * `--ppt N`（`cuda`）：pixels per thread
    * `--use-pitch`（`cuda`）：使用 pitched allocations（`cudaMallocPitch`）
    * `--pinned`（`cuda`）：host buffer 使用 pinned memory（`cudaMallocHost`）
    * `--use-shared`（`cuda`）：seeds 使用 shared memory
    * `--use-constant`（`cuda`）：seeds 使用 constant memory

* `-h, --help`
    * 顯示說明並結束

## 常用範例
1. 跑一個 serial baseline + 動畫 frames
    ```
    cd build

    ./jfa_demo \
    --backend serial \
    --width 512 --height 512 \
    --seeds 100 \
    --tag case1
    ```
    這會：
    * 用單執行緒 exact Voronoi + serial JFA 跑 512×512、100 個 seeds
    * 在 `output/case1_serial_t1_512x512_s100/` 底下產生：
        ```
        cpu_jfa_serial_pass_000_k256.ppm
        cpu_jfa_serial_pass_001_k128.ppm
        ...
        ```
    你可以用 `ffmpeg` 或 `scripts/render_videos.py` 把這些 PPM 合成一支 demo 影片。

2. 同一個 case，跑 OpenMP 4 執行緒版本
    ```
    cd build

    ./jfa_demo \
    --backend omp \
    --threads 4 \
    --width 512 --height 512 \
    --seeds 100 \
    --tag case1
    ```

    這會寫到：
    ```
    output/case1_omp_t4_512x512_s100/*.ppm
    ```

    你就可以直接比對 case1_serial... 跟 case1_omp_t4... 的 PPM 或動畫。

3. 純效能測試（不輸出 PPM；跳過 exact，保留 serial baseline，給 benchmark script 用）
    ```
    cd build

    ./jfa_demo \
    --backend omp \
    --threads 8 \
    --width 2048 --height 2048 \
    --seeds 500 \
    --no-dump \
    --skip-exact \
    --csv
    ```

    這會：

    * 不產生任何 PPM frame（只建一個空的 output 資料夾）
    * 在 stdout 印出像這樣的摘要行：
        ```
        CSV,omp,8,2048,2048,500,0,1929.82,411.547,-1,0
        ```

`scripts/run_benchmarks.py` 會自動呼叫 `jfa_demo`（加上 `--no-dump --csv`）並收集這些 CSV 行，寫進 `results/results_jfa.csv`。

4. 跑 SIMD backend（AVX2）做快速對照（不輸出 PPM）
    ```
    cd build

    ./jfa_demo \
    --backend simd \
    --width 512 --height 512 \
    --seeds 100 \
    --no-dump \
    --skip-exact \
    --csv
    ```

5. 跑 CUDA backend（不輸出 PPM；可用 `--block-dim/--ppt/--use-pitch/--pinned` 調參）
    ```
    cd build

    ./jfa_demo \
    --backend cuda \
    --width 512 --height 512 \
    --seeds 100 \
    --no-dump \
    --skip-exact \
    --csv
    ```

