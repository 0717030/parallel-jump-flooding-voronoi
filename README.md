# parallel-jump-flooding-voronoi
## 0. 檔案結構
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
├── docs/
│   ├── images/             # readme 裡用到的圖片 
│   │   └── ...
│   └── report.pdf          # 之後完成後加上 
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
│       ├── gpu.hpp         # 之後 CUDA JFA 的宣告（現在可能先是 stub）
│       └── visualize.hpp   # make_seed_palette, seed_index_to_rgb, write_ppm 等
│
├── src/
│   ├── apps/
│   │   ├── jfa_bench.cpp   # parse args → call serial / omp → 比較速度差距
│   │   └── jfa_demo.cpp    # CLI 主程式：parse args → call exact / serial / omp → dump PPM/CSV
│   │                       
│   │
│   ├── common/
│   │   └── jfa_visualize.cpp   # 實作 visualize.hpp 的函式（palette, PPM dump）
│   │
│   ├── cpu/
│   │   ├── jfa_common_impl.hpp # JFA cpu 共同邏輯（step sequence、distance 函式等）
│   │   ├── jfa_serial.cpp      # 單執行緒 JFA 實作（jfa_cpu_serial）
│   │   └── jfa_openmp.cpp      # OpenMP JFA 實作（jfa_cpu_omp）
│   │
│   ├── exact/
│   │   └── voronoi_exact.cpp   # O(N² * #seeds) 的精確 Voronoi baseline
│   │
│   └── gpu/
│       ├── jfa_cuda.cu          # CUDA kernel：在 GPU 上做 Jump Flooding
│       └── jfa_gpu_wrapper.cpp  # C++ 包裝，提供 jfa_gpu_run(...) 等介面給 jfa_demo/jfa_bench 使用
│
├── scripts/
│   ├── run_benchmarks.py       # 跑多組 jfa_demo（serial/omp）→ 寫 results/results_jfa.csv
│   ├── plot_speedup.py         # 讀 results_jfa.csv → 產生 speedup / efficiency 圖
│   └── render_videos.py        # 掃 output/*/ *_pass_*.ppm → 用 ffmpeg 合成 mp4 demo
│
└── tests/
    ├── CMakeLists.txt (可選)   # 之後要加單元測試再用；現在可以先放 TODO 註解
    └── data/                   # 測試用小 input（例如固定 seed set, small size）
```

## 1. Getting started
### 1.1 Clone 專案
```
git clone https://github.com/<YOUR_ORG>/parallel-jump-flooding-voronoi.git
cd parallel-jump-flooding-voronoi
```


### 1.2 依賴 (Dependencies)
* C++17 編譯器（g++ / clang++）
* CMake ≥ 3.x
* OpenMP（通常隨 gcc/clang 一起提供）
* NVIDIA GPU + 驅動
* CUDA Toolkit (nvcc)
*（可選）Python 3 + matplotlib：跑 benchmark / 畫圖用
*（可選）ffmpeg：把 PPM frame 合成 mp4 動畫用

### 1.3 Build

專案使用 out-of-source CMake build：
```
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. # DCMAKE_BUILD_TYPE=Release/Debug/RelWithDebInfo 用不同的 type 請開不同的資料夾
make -j
```
**使用不同的 CMAKE_BUILD_TYPE 參數**
* `Release`
    * 使用
        ```
        mkdir -p build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j
        ```
    * 特性：`-O3 -DNDEBUG`
    * 用途：
        * 用 run_benchmarks.py 或手動跑 jfa_bench，量正式的 serial vs OpenMP 時間，寫報告
        * 當 baseline + speedup

* `Debug`
    * 使用
        ```
        mkdir -p build-debug
        cd build-debug
        cmake -DCMAKE_BUILD_TYPE=Debug ..
        make -j
        ```
    * 特性：`-g`、沒優化
    * 用途：
        * 開發、除錯、看 assert、gdb
        * 不拿來畫 speedup / 報告

* `RelWithDebInfo`
    * 使用
        ```
        mkdir -p build-prof
        cd build-prof
        cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
        make -j
        ```
    * 特性：`C-O2 -g -DNDEBUG`
    * 用途：
        * 拿來跑 `perf record -g`，看 serial / OpenMP 的 bottleneck
        * 行為接近 Release，但又有 debug symbol


成功後，會在 build/ 底下看到：
* jfa_demo, jfa_bench – main CLI 可執行檔
* libjfa_cpu.a, libjfa_exact.a, libjfa_visualize.a – 靜態 library（被 jfa_demo link）

## 2. jfa_demo 使用方式

離開 build 目錄下執行：
```
./build/jfa_demo [options...]
```

**支援的參數**
* `--backend {serial|omp}`
    * 選擇 CPU JFA 實作：
        * serial：單執行緒 JFA
        * omp：OpenMP 多執行緒 JFA
        * cuda：GPU JFA（使用 jfa_gpu）
    * 預設：omp

* `--threads N`
    * OpenMP backend 使用的 threads 數
    * 僅在 --backend omp 時有效
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
    * 根輸出資料夾，預設為 output
    * 實際 frame 會寫到 DIR/<auto_name>/...（見下面 --tag）

* `--tag NAME`
    *  給這次實驗一個邏輯名字，用來組成輸出資料夾名稱
    * 實際的輸出資料夾會是：
        ```
        <output_root>/<tag>_<backend>_t<threads>_<width>x<height>_s<seeds>/
        ```

* `--csv`
    * 除了人類可讀的 log 之外，再印出一行 machine-readable 的 CSV summary：
        ```
        CSV,backend,threads,width,height,seeds,exact_ms,serial_ms,parallel_ms,diff_exact_parallel,diff_serial_parallel
        ```
* `-h, --help`
    * 顯示說明並結束

![jfa-demo](docs/images/jfa_demo.png)

## 3. jfa_bench 使用方式
離開 build 目錄下執行：
```
./build/jfa_bench [options...]
```
支援參數大致同 jfa_demo，但是沒有  `--tag`
![jfa_bench](docs/images/jfa_bench.png)

## 4. 常用範例
### 4.1 使用 jfa_demo 測試演算法正確性、生成 voronoi 圖
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

3. 純效能測試（不輸出 PPM，給 benchmark script 用）
    ```
    cd build

    ./jfa_demo \
    --backend omp \
    --threads 8 \
    --width 2048 --height 2048 \
    --seeds 500 \
    --no-dump \
    --csv
    ```

    這會：

    * 不產生任何 PPM frame（只建一個空的 output 資料夾）
    * 在 stdout 印出像這樣的摘要行：
        ```
        CSV,omp,8,2048,2048,500,6043.98,1929.82,411.547,3145744,0
        ```
### 4.2測試不同圖片大小與平行化方法的加速
`scripts/run_benchmarks.py` 會自動呼叫 `jfa_bench`（加上 `--no-dump --csv`）並收集這些 CSV 行，寫進 `results/results_jfa.csv`。
![run_benchmarks](docs/images/run_benchmarks.png)
