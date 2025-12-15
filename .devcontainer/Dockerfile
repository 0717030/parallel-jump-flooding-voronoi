FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-22.04

# Core tools for CMake + OpenMP builds, Python plotting scripts, and video rendering
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-pip python3-venv python3-dev \
        ffmpeg \
        ca-certificates curl gnupg \
        linux-tools-common \
    && rm -rf /var/lib/apt/lists/*

# CUDA Toolkit (nvcc) for building CMake CUDA language targets inside the container.
# Host GPU/driver is provided by the machine (e.g. RTX4090); the container needs the toolkit.
RUN curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm -f /tmp/cuda-keyring.deb && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/*

# Python deps for scripts/plot_speedup.py
RUN pip3 install --no-cache-dir pandas matplotlib

ENV CC=gcc CXX=g++

# Convenience: make sure nvcc is on PATH for shells and tools.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

