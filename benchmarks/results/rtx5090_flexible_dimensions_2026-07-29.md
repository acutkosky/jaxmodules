# RTX 5090 flexible-head Mosaic attention

This report covers the dimension-generic Mosaic attention work after the
Ubuntu 26.04, NVIDIA 580.173.02, and JAX 0.11.0 upgrades.

The implementation is not tied to one model shape:

- FP16 and BF16 head dimensions can be any multiple of 16 from 64 through
  2,048.
- D=64 retains the established 64x64 fast paths.
- Every supported width from D=80 through D=2048 uses the same general 32x32
  synchronous-MMA forward and split backward kernel family.
- Callable masks, maximal-causal attention, unmasked attention, MHA, and GQA
  retain the same public interface.
- Softmax and matrix-product accumulation remain FP32. Inputs and outputs
  remain in the input dtype.

The 32x32 retile reduces register pressure and enables wide heads. The default
wide-head backward uses separate query-major dQ and key-major dK/dV passes, so
complete dK/dV tiles are accumulated locally without global atomics. An
explicit `backward_strategy="one_pass"` remains available for unusually sparse
callable masks.

## Raw results

- [D=128 FP16 causal, Mosaic and cuDNN](rtx5090_flexible_d128_fp16_causal_2026-07-29.json)
- [D=128 BF16 causal, Mosaic and cuDNN](rtx5090_flexible_d128_bf16_causal_2026-07-29.json)
- [D=128 FP16 causal, XLA](rtx5090_flexible_d128_fp16_causal_xla_2026-07-29.json)
- [D=128 FP16 unmasked](rtx5090_flexible_d128_fp16_unmasked_2026-07-29.json)
- [D=128 FP16 sparse general mask, 1K-8K](rtx5090_flexible_d128_fp16_general_2026-07-29.json)
- [D=128 FP16 sparse general mask, 16K-32K](rtx5090_flexible_d128_fp16_general_large_2026-07-29.json)
- [FP16 head-dimension sweep](rtx5090_flexible_head_dimensions_fp16_2026-07-29.json)
- [Previous 64x64 D=128 causal kernel](rtx5090_flexible_d128_fp16_causal_previous_tile64_2026-07-29.json)
- [Previous 64x64 D=128 unmasked kernel](rtx5090_flexible_d128_fp16_unmasked_previous_tile64_2026-07-29.json)
- [Previous 64x64 D=128 sparse-mask kernel](rtx5090_flexible_d128_fp16_general_previous_tile64_2026-07-29.json)
- [Sparse-mask one-pass backward](rtx5090_flexible_d128_fp16_general_one_pass_2026-07-29.json)
- [Dense-mask split backward](rtx5090_flexible_d128_fp16_general_dense_split_2026-07-29.json)
- [Dense-mask one-pass backward](rtx5090_flexible_d128_fp16_general_dense_one_pass_2026-07-29.json)

## Environment and method

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-28-generic |
| Driver | 580.173.02; CUDA driver API 13.0 |
| JAX | JAX/JAXlib and CUDA 12 PJRT/plugin 0.11.0 |
| CUDA runtime | JAX build 12.9; loaded runtime 12.8 |
| cuDNN | JAX build 9.8.0; loaded runtime 9.10.2 |
| Main scale sweep | `B=1`, `Hq=Hkv=8`, `D=128` (1,024 values/token) |
| Width sweep | `B=1`, `Hq=Hkv=8`, `N=1024`, `D=64..2048` |
| Main timing | Explicit lower/compile, 5 warmups, median of 30 iterations |
| Width timing | Explicit lower/compile, 5 warmups, median of 30 iterations |
| Isolation | One fresh process per case, run sequentially on the single GPU |
| Allocation | `XLA_PYTHON_CLIENT_PREALLOCATE=false` |

A “gradient” time is one compiled `jax.grad` call and includes the forward
needed by that call. Device peak is the process peak allocation above the
live-input baseline. The block-size fields in the older JSON files are API
inputs; the effective Mosaic tile is 32x32 for D=128.

## Eight heads, D=128: maximal causal

The previous Mosaic column is the exact 64x64 implementation immediately
before the dimension-generic retile. XLA was stopped after 8K because its
quadratic score materialization is already 4.1 GiB there.

### FP16 forward

| N | Previous Mosaic | New Mosaic | cuDNN | XLA |
|---:|---:|---:|---:|---:|
| 1,024 | 0.113 ms | 0.090 ms | 0.076 ms | 0.086 ms |
| 2,048 | 0.274 ms | 0.163 ms | 0.148 ms | 0.327 ms |
| 4,096 | 0.587 ms | 0.475 ms | 0.284 ms | 1.153 ms |
| 8,192 | 1.813 ms | 1.568 ms | 0.862 ms | 4.262 ms |
| 16,384 | 6.343 ms | 5.379 ms | 2.831 ms | - |
| 32,768 | 25.520 ms | 23.450 ms | 10.670 ms | - |

### FP16 forward plus gradient

| N | Previous Mosaic | New Mosaic | cuDNN | XLA |
|---:|---:|---:|---:|---:|
| 1,024 | 0.351 ms | 0.277 ms | 0.179 ms | 0.186 ms |
| 2,048 | 0.991 ms | 0.599 ms | 0.398 ms | 0.751 ms |
| 4,096 | 2.480 ms | 1.968 ms | 0.950 ms | 2.642 ms |
| 8,192 | 8.374 ms | 6.752 ms | 2.938 ms | 10.555 ms |
| 16,384 | 32.072 ms | 25.831 ms | 10.381 ms | - |
| 32,768 | 119.812 ms | 97.099 ms | 40.925 ms | - |

The retile improves causal forward by 8-41% and the complete gradient call by
19-40% over the previous D=128 kernel. At 32K, Mosaic remains 2.20x slower
forward and 2.37x slower for the gradient than cuDNN.

### FP16 gradient memory

| N | Previous Mosaic | New Mosaic | cuDNN |
|---:|---:|---:|---:|
| 1,024 | 36 MiB | 36 MiB | 24.1 MiB |
| 2,048 | 56 MiB | 48 MiB | 48.1 MiB |
| 4,096 | 104 MiB | 72.3 MiB | 104 MiB |
| 8,192 | 208 MiB | 144.5 MiB | 208 MiB |
| 16,384 | 416 MiB | 289 MiB | 416 MiB |
| 32,768 | 832 MiB | 578 MiB | 832 MiB |

The non-atomic split lowers the 32K measured peak by 30.5% relative to both
the previous one-pass kernel and cuDNN.

### BF16

| N | Mosaic fwd | cuDNN fwd | Mosaic grad | cuDNN grad |
|---:|---:|---:|---:|---:|
| 1,024 | 0.092 ms | 0.072 ms | 0.285 ms | 0.189 ms |
| 2,048 | 0.161 ms | 0.142 ms | 0.611 ms | 0.387 ms |
| 4,096 | 0.489 ms | 0.284 ms | 1.964 ms | 0.940 ms |
| 8,192 | 1.545 ms | 0.859 ms | 6.633 ms | 2.932 ms |
| 16,384 | 5.472 ms | 2.826 ms | 25.986 ms | 10.345 ms |
| 32,768 | 23.208 ms | 10.635 ms | 97.244 ms | 41.126 ms |

FP16 and BF16 behave similarly, as expected: both use input-precision tensor
core operands and FP32 accumulation.

## Eight heads, D=128: unmasked

| N | Mosaic fwd | cuDNN fwd | Mosaic grad | cuDNN grad |
|---:|---:|---:|---:|---:|
| 1,024 | 0.090 ms | 0.078 ms | 0.295 ms | 0.191 ms |
| 2,048 | 0.284 ms | 0.141 ms | 1.121 ms | 0.484 ms |
| 4,096 | 0.918 ms | 0.459 ms | 3.780 ms | 1.599 ms |
| 8,192 | 3.109 ms | 1.452 ms | 14.864 ms | 5.622 ms |
| 16,384 | 11.897 ms | 5.438 ms | 58.553 ms | 22.445 ms |
| 32,768 | 46.420 ms | 22.901 ms | 222.586 ms | 81.978 ms |

At 32K, the new split backward is 11.1% faster than the previous 250.445 ms
one-pass result and reduces peak memory from 832 MiB to 578 MiB. Forward is
within 2.2% of the previous kernel. cuDNN remains substantially faster for
dense unmasked attention.

## Eight heads, D=128: general callable mask

The representative callable is noncausal and has a local radius plus a
coordinate-dependent modulo predicate. Mosaic evaluates it inside each score
tile and skips empty tiles. XLA and cuDNN receive the equivalent materialized
boolean mask, so their general-mask memory is quadratic.

| N | Mosaic fwd | XLA fwd | cuDNN fwd | Mosaic grad | XLA grad | cuDNN grad |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.077 ms | 0.074 ms | 0.100 ms | 0.227 ms | 0.168 ms | 0.230 ms |
| 2,048 | 0.117 ms | 0.321 ms | 0.205 ms | 0.401 ms | 0.688 ms | 0.658 ms |
| 4,096 | 0.202 ms | 1.134 ms | 0.671 ms | 0.768 ms | 2.629 ms | 2.632 ms |
| 8,192 | 0.490 ms | 4.363 ms | 2.333 ms | 1.904 ms | 10.453 ms | 9.712 ms |
| 16,384 | 1.441 ms | - | 11.746 ms | 6.004 ms | - | 38.430 ms |
| 32,768 | 4.654 ms | - | 36.398 ms | 21.590 ms | - | OOM |

At 32K, Mosaic peaks at 208 MiB forward and 578 MiB for the gradient. cuDNN
peaks at 16.1 GiB forward; its gradient attempts a 48.13 GiB allocation and
OOMs.

### Sparse versus dense backward strategy

There is no safe general way to infer the density of an arbitrary callable
mask. Auto therefore chooses the non-atomic split for wide heads: it is
consistently faster for a dense callable and has the lower memory bound.
`one_pass` is an explicit alternative when the user knows that very few tiles
survive.

| N | Sparse split | Sparse one-pass | Dense split | Dense one-pass |
|---:|---:|---:|---:|---:|
| 1,024 | 0.227 ms | 0.238 ms | 0.319 ms | 0.388 ms |
| 4,096 | 0.768 ms | 0.796 ms | 3.929 ms | 5.167 ms |
| 8,192 | 1.904 ms | 1.933 ms | 14.760 ms | 21.418 ms |
| 16,384 | 6.004 ms | 4.991 ms | 60.557 ms | 78.526 ms |
| 32,768 | 21.590 ms | 16.385 ms | 229.666 ms | 317.652 ms |

At 32K, split uses 578 MiB while one-pass uses 832 MiB. The explicit strategy
exposes the speed/memory tradeoff without inspecting or restricting the mask
callable.

## Per-head dimension sweep

This is a compiled `B=1, H=8, N=1024`, FP16 maximal-causal sweep. It is a
support and short-context control, not the intended large-context performance
regime.

| D | Mosaic fwd | cuDNN fwd | XLA fwd | Mosaic grad | cuDNN grad | XLA grad | Mosaic grad peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.065 ms | 0.083 ms | 0.072 ms | 0.186 ms | 0.188 ms | 0.143 ms | 34.0 MiB |
| 80 | 0.078 ms | 0.081 ms | 0.091 ms | 0.223 ms | 0.190 ms | 0.169 ms | 34.8 MiB |
| 96 | 0.074 ms | 0.079 ms | 0.083 ms | 0.238 ms | 0.200 ms | 0.179 ms | 35.0 MiB |
| 128 | 0.093 ms | 0.072 ms | 0.083 ms | 0.292 ms | 0.188 ms | 0.206 ms | 36.0 MiB |
| 192 | 0.123 ms | 0.130 ms | 0.094 ms | 0.432 ms | unsupported | 0.215 ms | 38.0 MiB |
| 256 | 0.146 ms | 0.151 ms | 0.104 ms | 0.660 ms | unsupported | 0.239 ms | 40.0 MiB |
| 512 | 0.390 ms | unsupported | 0.135 ms | 1.954 ms | unsupported | 0.362 ms | 72.1 MiB |
| 1,024 | 0.823 ms | unsupported | 0.226 ms | 6.522 ms | unsupported | 0.603 ms | 144.1 MiB |
| 2,048 | 3.897 ms | unsupported | 0.378 ms | 24.073 ms | unsupported | 1.114 ms | 288.1 MiB |

On this JAX/cuDNN stack, cuDNN forward supports D up to 256, but backward
rejects D greater than 128. XLA is the fastest option for very wide heads at
N=1024. The Mosaic value proposition in that corner is compatibility with
large-context, linear-memory execution and callable masks, not short-context
GEMM speed.

## Validation and conclusions

- Forward and dQ/dK/dV comparisons against XLA pass for D=80, 96, 128, 192,
  256, 512, 1,024, and 2,048, including causal, unmasked, and noncausal
  callable masks.
- The complete Mosaic GPU test module passes: 47 tests.
- The D=128 retile is a real generalization, not a shape-specific kernel. It
  improves causal runtime and reduces large-scale backward memory while the
  same implementation compiles across the full supported width range.
- cuDNN remains the dense-attention speed target. General callable masks are
  the current strongest case: Mosaic remains linear-memory and completes the
  32K backward where the dense-mask cuDNN adapter OOMs.
