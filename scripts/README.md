# Scripts

- `run_benchmarks.py`  
  Runs `jfa_demo` for multiple configurations (serial vs OpenMP, different sizes / seeds) and writes a CSV file to `results/results_jfa.csv`.

- `plot_speedup.py`  
  Reads `results/results_jfa.csv` and generates speedup / efficiency plots into `results/*.png`.

- `render_videos.py`  
  Looks for `*_pass_*.ppm` frames under `output/*/` and uses `ffmpeg` to generate MP4 animations for demo purposes.
