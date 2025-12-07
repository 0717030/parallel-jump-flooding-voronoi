#!/usr/bin/env python3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "output"

def render_dir(dir_path: Path, fps=10):
    # 假設你的檔名格式為 cpu_jfa_xxx_pass_000_k256.ppm
    pattern = str(dir_path / "*_pass_*.ppm")
    mp4_path = dir_path / "anim.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-vf", "scale=512:-1",
        "-pix_fmt", "yuv420p",
        str(mp4_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    for sub in sorted(OUT_ROOT.iterdir()):
        if not sub.is_dir():
            continue
        # 確認裡面有 ppm 檔才做
        if not any(sub.glob("*_pass_*.ppm")):
            continue
        render_dir(sub)

if __name__ == "__main__":
    main()
