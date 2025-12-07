# GPU backends (TODO)

This folder is reserved for CUDA JFA implementations.

Planned files:
- `jfa_cuda_basic.cu`      – baseline CUDA JFA (single-kernel multi-pass).
- `jfa_cuda_pingpong.cu`   – multi-kernel ping-pong buffer JFA.
- Future variants (JFA+1, JFA², etc.).

The public API for these implementations will be declared in `include/jfa/gpu.hpp`.
