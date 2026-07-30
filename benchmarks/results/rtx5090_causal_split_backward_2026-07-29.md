# RTX 5090 non-atomic maximal-causal backward

Commit `e4937db` replaces the large maximal-causal MHA backward with two
warp-specialized passes:

- a query-major pass computes dQ while sharing each staged K/V tile between
  two adjacent query tiles;
- a key-major pass assigns two adjacent K/V tiles to two compute warpgroups,
  streams their common causal Q/dO suffix with a producer warpgroup, accumulates
  complete FP32 dK/dV tiles locally, and writes each result once.

The key-major pass removes the quadratic FP32 global atomics that dominated the
previous backward. It is selected for maximal-causal FP16/BF16 MHA with
`D=64`, square sequence lengths of at least 1,024, and no additional user mask.
Causal GQA retains the prior atomic path because multiple query heads contribute
to each K/V head. General callable masks and other unsupported shapes retain
their existing fallbacks.

Raw results:

- [FP16, 1K-32K](rtx5090_causal_split_backward_fp16_2026-07-29.json)
- [BF16, 1K-32K](rtx5090_causal_split_backward_bf16_2026-07-29.json)
- [FP16, 64K-128K](rtx5090_causal_split_backward_fp16_xlarge_2026-07-29.json)
- [BF16, 64K-128K](rtx5090_causal_split_backward_bf16_xlarge_2026-07-29.json)

## Environment and method

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-28-generic |
| Driver | 580.173.02; CUDA driver API 13.0 |
| JAX | JAX/JAXlib 0.11.0; CUDA 12 PJRT/plugin 0.11.0 |
| CUDA runtime | JAX build 12.9; loaded runtime 12.8 |
| cuDNN | JAX build 9.8.0; loaded runtime 9.10.2 |
| Inputs | `B=1`, `Hq=Hkv=4`, `D=64`, FP16 or BF16 |
| Sequence lengths | 1,024 through 131,072 |
| Timing | Explicit lower/compile, 5 warmups, median of 30 iterations |
| Isolation | One fresh process per case, run sequentially on the single GPU |
| Allocation | `XLA_PYTHON_CLIENT_PREALLOCATE=false` |

A gradient time is one compiled `jax.grad` call, so it includes the forward
pass. “Previous” is the causal-warp VJP at commit `a9e47b8`, which already used
the optimized causal forward but retained the atomic backward. “vs cuDNN” is
new Mosaic divided by cuDNN; values below one favor Mosaic.

Short sub-millisecond cases show more clock and launch noise than the large
cases. The dedicated 1K/2K crossover control measured the split kernel at
0.168/0.306 ms in FP16 and 0.168/0.295 ms in BF16.

## FP16 forward plus gradient

| N | Previous Mosaic | Split Mosaic | Reduction | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.179 ms | 0.194 ms | -8.5% | 0.173 ms | 1.12x |
| 2,048 | 0.298 ms | 0.294 ms | 1.3% | 0.296 ms | 0.99x |
| 4,096 | 0.690 ms | 0.541 ms | 21.6% | 0.497 ms | 1.09x |
| 8,192 | 1.662 ms | 1.211 ms | 27.1% | 1.166 ms | 1.04x |
| 16,384 | 5.667 ms | 3.718 ms | 34.4% | 3.329 ms | 1.12x |
| 32,768 | 22.278 ms | 12.469 ms | 44.0% | 11.496 ms | 1.08x |
| 65,536 | - | 53.382 ms | - | 43.503 ms | 1.23x |
| 131,072 | - | 198.034 ms | - | 170.025 ms | 1.16x |

## BF16 forward plus gradient

| N | Previous Mosaic | Split Mosaic | Reduction | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.188 ms | 0.185 ms | 1.8% | 0.172 ms | 1.07x |
| 2,048 | 0.298 ms | 0.301 ms | -1.2% | 0.281 ms | 1.07x |
| 4,096 | 0.729 ms | 0.534 ms | 26.8% | 0.493 ms | 1.08x |
| 8,192 | 1.676 ms | 1.189 ms | 29.1% | 1.181 ms | 1.01x |
| 16,384 | 5.781 ms | 3.540 ms | 38.8% | 3.388 ms | 1.04x |
| 32,768 | 22.094 ms | 12.866 ms | 41.8% | 11.440 ms | 1.12x |
| 65,536 | - | 52.723 ms | - | 43.952 ms | 1.20x |
| 131,072 | - | 202.692 ms | - | 174.265 ms | 1.16x |

## Memory

FP16 and BF16 produced the same rounded memory values. Device peak is process
peak allocation above the live-input baseline. Compiler temporary is the
compiled executable's estimate.

| N | Mosaic device peak | cuDNN device peak | Mosaic compiler temp | cuDNN compiler temp |
|---:|---:|---:|---:|---:|
| 1,024 | 33 MiB | 6 MiB | 2 MiB | 3 MiB |
| 2,048 | 34 MiB | 12.1 MiB | 4.1 MiB | 6.1 MiB |
| 4,096 | 42 MiB | 24.1 MiB | 8.1 MiB | 12.1 MiB |
| 8,192 | 56 MiB | 48.3 MiB | 16.3 MiB | 24.3 MiB |
| 16,384 | 104 MiB | 104 MiB | 32.5 MiB | 48.5 MiB |
| 32,768 | 208 MiB | 208 MiB | 65 MiB | 97 MiB |
| 65,536 | 416 MiB | 416 MiB | 130 MiB | 194 MiB |
| 131,072 | 644 MiB | 832 MiB | 260 MiB | 388 MiB |

Both implementations remain linear-memory and completed through 128K on this
workload. Mosaic's smaller compiler temporary becomes visible in the measured
device peak at 128K.

## Profiling and tuning

A 32K FP16 JAX/CUPTI trace attributes approximately 2.47 ms to forward,
3.72 ms to dQ, and 5.44 ms to non-atomic dK/dV. Surrounding compiler-generated
transpose, reduction, and cast kernels total less than 0.1 ms.

- A diagnostic dQ-only backward showed that the old dK/dV atomics were the
  entire large-scale bottleneck. The existing generic two-pass strategy was
  17% slower than the atomic one-pass kernel at 32K, so merely selecting it was
  not sufficient.
- Keeping complete dK/dV tiles local makes a 256-register compute / 24-register
  producer split best. The neighboring allocations cross a spill-pressure
  cliff. Two TMA stages are best; one stage was about 25% slower and three
  stages about 2% slower at 32K.
- Removing unused dK/dV scratch from the dQ pass reduces its dynamic shared
  memory from about 74 KiB to 28 KiB. Its runtime is already close to the
  expected three-contraction versus two-contraction forward ratio.
- One dK/dV compute warpgroup was about 40% slower than two because it loses
  Q/dO staging reuse. Reversing the query traversal to align cache accesses
  increased the dK/dV kernel from about 5.4 ms to 8.5 ms and was reverted.
- JAX 0.11 includes a Tensor Core Gen 5 API, but compiling its bundled
  Blackwell matmul on this machine fails because `tcgen05.alloc` is explicitly
  unsupported for `sm_120a`. The RTX 5090 path therefore continues to use
  native `mma.sync` tensor-core instructions.

## Precision and validation

No precision policy or public attention interface changed. Inputs and outputs
remain FP16/BF16, the tensor-core operands remain in the input dtype, and
softmax plus matrix-product accumulation remain FP32.

Direct FP16 and BF16 dQ/dK/dV comparisons against XLA pass. The public 4K
custom VJP, compiled split-backward `vmap`, and 1K causal GQA fallback all
match XLA. Unsupported masks, shapes, dtypes, and GQA continue through their
previous paths.

## Conclusions

- The non-atomic key-major pass removes the structural backward bottleneck:
  the 32K total gradient call improves by 44% in FP16 and 42% in BF16.
- From 4K through 32K, Mosaic is generally within about 1-12% of cuDNN. At
  64K-128K the measured gap is 16-23%, while Mosaic uses less compiler
  temporary and a lower measured device peak at 128K.
- cuDNN does not OOM on this maximal-causal `B=1, H=4, D=64` workload through
  128K with the current JAX/driver stack. Both implementations scale linearly
  in memory here.
