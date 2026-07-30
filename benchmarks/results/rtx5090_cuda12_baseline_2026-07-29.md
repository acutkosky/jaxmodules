# RTX 5090 CUDA 12 baseline after the Ubuntu 26.04 upgrade

This is the control measurement for later CUDA-runtime and driver A/B tests.
It measures the current Mosaic implementation against cuDNN at repository
commit `2369507` (`codex/memory-efficient-attention`). The attention code is
unchanged from its parent, `bf2123b`.

The six raw JSON files contain a versioned environment record for every
isolated worker:

- [FP16 unmasked](rtx5090_cuda12_fp16_unmasked_2026-07-29.json)
- [FP16 causal](rtx5090_cuda12_fp16_causal_2026-07-29.json)
- [FP16 general mask](rtx5090_cuda12_fp16_general_2026-07-29.json)
- [BF16 unmasked](rtx5090_cuda12_bf16_unmasked_2026-07-29.json)
- [BF16 causal](rtx5090_cuda12_bf16_causal_2026-07-29.json)
- [BF16 general mask](rtx5090_cuda12_bf16_general_2026-07-29.json)

## Environment and method

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-28-generic |
| Driver | 580.173.02; CUDA driver API 13.0 |
| JAX | JAX/JAXlib 0.11.0; `jax-cuda12-plugin` and PJRT 0.11.0 |
| CUDA runtime | JAX build 12.9; loaded runtime 12.8 |
| cuDNN | JAX build 9.8.0; loaded runtime 9.10.2 |
| Inputs | `B=1`, `Hq=Hkv=4`, `D=64`, FP16 or BF16 |
| Sequence lengths | 1,024 through 32,768; nothing below 1,024 |
| Timing | Explicit lower/compile, 5 warmups, median of 30 iterations |
| Isolation | One fresh process per case, run sequentially |
| Allocation | `XLA_PYTHON_CLIENT_PREALLOCATE=false` |

A backward time is one compiled gradient call, including the forward needed by
the gradient. Each runtime cell is `median milliseconds / device peak above
the live-input baseline`. Ratio is Mosaic divided by cuDNN, so values below
one favor Mosaic.

## FP16

### Unmasked

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.074 / 33 MiB | 0.081 / 1 MiB | 0.92x | 0.202 / 33 MiB | 0.177 / 6 MiB | 1.14x |
| 2,048 | 0.111 / 34 MiB | 0.100 / 3 MiB | 1.11x | 0.350 / 34 MiB | 0.279 / 12.1 MiB | 1.26x |
| 4,096 | 0.155 / 36 MiB | 0.141 / 4 MiB | 1.10x | 0.703 / 42 MiB | 0.466 / 24.1 MiB | 1.51x |
| 8,192 | 0.453 / 40 MiB | 0.450 / 8 MiB | 1.01x | 2.670 / 56 MiB | 1.687 / 48.3 MiB | 1.58x |
| 16,384 | 1.678 / 48 MiB | 1.649 / 16 MiB | 1.02x | 11.032 / 104 MiB | 6.583 / 104 MiB | 1.68x |
| 32,768 | 5.969 / 64 MiB | 5.927 / 32 MiB | 1.01x | 44.678 / 208 MiB | 24.782 / 208 MiB | 1.80x |

### Causal

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.075 / 33 MiB | 0.086 / 1 MiB | 0.88x | 0.200 / 33 MiB | 0.184 / 6 MiB | 1.09x |
| 2,048 | 0.116 / 34 MiB | 0.102 / 2 MiB | 1.13x | 0.339 / 34 MiB | 0.290 / 12.1 MiB | 1.17x |
| 4,096 | 0.231 / 36 MiB | 0.173 / 4 MiB | 1.33x | 0.797 / 42 MiB | 0.505 / 24.1 MiB | 1.58x |
| 8,192 | 0.498 / 40 MiB | 0.409 / 8 MiB | 1.22x | 1.914 / 56 MiB | 1.172 / 48.3 MiB | 1.63x |
| 16,384 | 1.503 / 48 MiB | 1.092 / 16 MiB | 1.38x | 6.418 / 104 MiB | 3.352 / 104 MiB | 1.91x |
| 32,768 | 5.362 / 64 MiB | 3.573 / 32 MiB | 1.50x | 24.933 / 208 MiB | 11.408 / 208 MiB | 2.19x |

### General callable mask

Mosaic evaluates the callable inside each tile and skips empty tiles. JAX
materializes the equivalent boolean mask for cuDNN, so the cuDNN memory here
does not describe the linear-memory unmasked flash-attention core.

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.065 / 33 MiB | 0.109 / 17 MiB | 0.60x | 0.183 / 33 MiB | 0.218 / 35 MiB | 0.84x |
| 2,048 | 0.073 / 34 MiB | 0.194 / 66 MiB | 0.38x | 0.206 / 34 MiB | 0.363 / 134 MiB | 0.57x |
| 4,096 | 0.111 / 36 MiB | 0.511 / 260 MiB | 0.22x | 0.320 / 42 MiB | 1.272 / 520 MiB | 0.25x |
| 8,192 | 0.209 / 40 MiB | 1.325 / 520 MiB | 0.16x | 0.693 / 56 MiB | 4.718 / 1.5 GiB | 0.15x |
| 16,384 | 0.568 / 48 MiB | 5.484 / 2.0 GiB | 0.10x | 1.840 / 104 MiB | 19.910 / 6.1 GiB | 0.09x |
| 32,768 | 1.883 / 64 MiB | 24.352 / 8.0 GiB | 0.08x | 6.015 / 208 MiB | OOM | - |

## BF16

### Unmasked

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.083 / 33 MiB | 0.076 / 1 MiB | 1.09x | 0.180 / 33 MiB | 0.174 / 6 MiB | 1.03x |
| 2,048 | 0.121 / 34 MiB | 0.108 / 2 MiB | 1.12x | 0.328 / 34 MiB | 0.274 / 12.1 MiB | 1.20x |
| 4,096 | 0.139 / 36 MiB | 0.149 / 4 MiB | 0.93x | 0.704 / 42 MiB | 0.484 / 24.1 MiB | 1.45x |
| 8,192 | 0.458 / 40 MiB | 0.440 / 8 MiB | 1.04x | 2.656 / 56 MiB | 1.667 / 48.3 MiB | 1.59x |
| 16,384 | 1.649 / 48 MiB | 1.650 / 16 MiB | 1.00x | 10.760 / 104 MiB | 6.575 / 104 MiB | 1.64x |
| 32,768 | 5.944 / 64 MiB | 5.821 / 32 MiB | 1.02x | 41.761 / 208 MiB | 24.537 / 208 MiB | 1.70x |

### Causal

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.080 / 33 MiB | 0.074 / 1 MiB | 1.08x | 0.200 / 33 MiB | 0.181 / 6 MiB | 1.11x |
| 2,048 | 0.111 / 35 MiB | 0.095 / 2 MiB | 1.18x | 0.326 / 34 MiB | 0.289 / 12.1 MiB | 1.13x |
| 4,096 | 0.223 / 36 MiB | 0.161 / 4 MiB | 1.39x | 0.783 / 42 MiB | 0.493 / 24.1 MiB | 1.59x |
| 8,192 | 0.512 / 40 MiB | 0.423 / 8 MiB | 1.21x | 1.907 / 56 MiB | 1.202 / 48.3 MiB | 1.59x |
| 16,384 | 1.511 / 48 MiB | 1.093 / 16 MiB | 1.38x | 6.430 / 104 MiB | 3.343 / 104 MiB | 1.92x |
| 32,768 | 5.579 / 64 MiB | 3.409 / 32 MiB | 1.64x | 25.220 / 208 MiB | 11.602 / 208 MiB | 2.17x |

### General callable mask

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.063 / 33 MiB | 0.111 / 17 MiB | 0.56x | 0.189 / 33 MiB | 0.213 / 35 MiB | 0.89x |
| 2,048 | 0.077 / 34 MiB | 0.192 / 66 MiB | 0.40x | 0.205 / 34 MiB | 0.362 / 134 MiB | 0.57x |
| 4,096 | 0.104 / 36 MiB | 0.513 / 260 MiB | 0.20x | 0.308 / 42 MiB | 1.297 / 520 MiB | 0.24x |
| 8,192 | 0.226 / 40 MiB | 1.407 / 520 MiB | 0.16x | 0.676 / 56 MiB | 4.881 / 1.5 GiB | 0.14x |
| 16,384 | 0.567 / 48 MiB | 5.565 / 2.0 GiB | 0.10x | 1.843 / 104 MiB | 20.561 / 6.1 GiB | 0.09x |
| 32,768 | 1.868 / 64 MiB | 25.564 / 8.0 GiB | 0.07x | 6.091 / 208 MiB | OOM | - |

## Conclusions

- The current CUDA 12 stack is healthy: 142 of 144 cases completed. The only
  failures were the expected 32K general-mask cuDNN gradients, each requesting
  a 24.03 GiB dense allocation.
- The current unmasked Mosaic forward is effectively at cuDNN parity at large
  scale: 1.01x at 32K FP16 and 1.02x at 32K BF16.
- Dense backward remains the main gap. At 32K, Mosaic is 1.80x slower in FP16
  and 1.70x slower in BF16; both implementations peak at 208 MiB.
- At 32K causal, Mosaic is 1.50x/1.64x slower forward and 2.19x/2.17x slower
  backward for FP16/BF16.
- The callable-mask specialization remains the strong case. At 32K, Mosaic
  uses 64 MiB forward and 208 MiB backward; cuDNN uses 8 GiB forward and OOMs
  backward because JAX's general-mask adapter is dense.

The older July 27 measurements are not a clean OS or CUDA comparison because
the attention kernels changed afterward. Future CUDA 13 and driver tests
should compare directly with these JSON files while keeping this commit,
shapes, warmups, iteration count, and process isolation fixed.
