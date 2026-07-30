# RTX 5090 warp-specialized maximal-causal attention

Commit `a9e47b8` extends the shared-K/V warp-specialized forward kernel to
maximal causal self-attention. The public API and precision policy are
unchanged.

The fast path uses two compute warpgroups for adjacent 64-query tiles and one
producer warpgroup. The producer loads each K/V tile once for both consumers,
stops at their common causal prefix, and applies a triangular mask only on
each query tile's diagonal. Query supertiles are launched longest-prefix
first so the scheduler starts the most expensive programs early.

It is selected for square FP16/BF16 self-attention with `D=64`, sequence
length at least 1,024, `is_causal=True`, and no additional user mask
(`mask_fn=None`, canonically the literal-true mask). A supplied general mask,
rectangular causal attention, unsupported shapes, and other dtypes continue
to use the existing callable-mask or mapped fallback. The custom VJP uses the
new forward but retains the established causal backward kernel.

Raw results:

- [FP16 causal](rtx5090_causal_warp_fp16_2026-07-29.json)
- [BF16 causal](rtx5090_causal_warp_bf16_2026-07-29.json)

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

“Previous” is the causal Mosaic result at commit `2369507` from the
[same-stack control run](rtx5090_cuda12_baseline_2026-07-29.md). cuDNN was
remeasured alongside the new kernel. A gradient time is one compiled
`jax.grad` call and therefore includes its forward pass. “vs cuDNN” is new
Mosaic divided by cuDNN, so values below one favor Mosaic.

## FP16 runtime

### Forward

| N | Previous Mosaic | Causal-warp Mosaic | Reduction | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.075 ms | 0.068 ms | 9.9% | 0.076 ms | 0.90x |
| 2,048 | 0.116 ms | 0.096 ms | 17.0% | 0.100 ms | 0.96x |
| 4,096 | 0.231 ms | 0.138 ms | 40.3% | 0.168 ms | 0.82x |
| 8,192 | 0.498 ms | 0.273 ms | 45.0% | 0.406 ms | 0.67x |
| 16,384 | 1.503 ms | 0.775 ms | 48.4% | 1.099 ms | 0.71x |
| 32,768 | 5.362 ms | 2.626 ms | 51.0% | 3.463 ms | 0.76x |

### Forward plus gradient

| N | Previous Mosaic | Causal-warp VJP | Reduction | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.200 ms | 0.179 ms | 10.7% | 0.195 ms | 0.92x |
| 2,048 | 0.339 ms | 0.298 ms | 12.1% | 0.294 ms | 1.01x |
| 4,096 | 0.797 ms | 0.690 ms | 13.4% | 0.506 ms | 1.36x |
| 8,192 | 1.914 ms | 1.662 ms | 13.2% | 1.178 ms | 1.41x |
| 16,384 | 6.418 ms | 5.667 ms | 11.7% | 3.361 ms | 1.69x |
| 32,768 | 24.933 ms | 22.278 ms | 10.7% | 11.645 ms | 1.91x |

## BF16 runtime

### Forward

| N | Previous Mosaic | Causal-warp Mosaic | Reduction | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.080 ms | 0.063 ms | 21.5% | 0.074 ms | 0.85x |
| 2,048 | 0.111 ms | 0.089 ms | 20.4% | 0.105 ms | 0.84x |
| 4,096 | 0.223 ms | 0.134 ms | 39.8% | 0.171 ms | 0.78x |
| 8,192 | 0.512 ms | 0.286 ms | 44.2% | 0.421 ms | 0.68x |
| 16,384 | 1.511 ms | 0.789 ms | 47.8% | 1.092 ms | 0.72x |
| 32,768 | 5.579 ms | 2.553 ms | 54.2% | 3.609 ms | 0.71x |

### Forward plus gradient

| N | Previous Mosaic | Causal-warp VJP | Reduction | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.200 ms | 0.188 ms | 5.8% | 0.188 ms | 1.00x |
| 2,048 | 0.326 ms | 0.298 ms | 8.7% | 0.301 ms | 0.99x |
| 4,096 | 0.783 ms | 0.729 ms | 6.8% | 0.505 ms | 1.45x |
| 8,192 | 1.907 ms | 1.676 ms | 12.1% | 1.189 ms | 1.41x |
| 16,384 | 6.430 ms | 5.781 ms | 10.1% | 3.356 ms | 1.72x |
| 32,768 | 25.220 ms | 22.094 ms | 12.4% | 11.513 ms | 1.92x |

## Memory

FP16 and BF16 produced the same rounded device peaks. Values are peak
allocation above the live-input baseline.

| N | Mosaic forward | cuDNN forward | Mosaic gradient | cuDNN gradient |
|---:|---:|---:|---:|---:|
| 1,024 | 33 MiB | 1 MiB | 33 MiB | 6 MiB |
| 2,048 | 34 MiB | 2 MiB | 34 MiB | 12.1 MiB |
| 4,096 | 36 MiB | 4 MiB | 42 MiB | 24.1 MiB |
| 8,192 | 40 MiB | 8 MiB | 56 MiB | 48.3 MiB |
| 16,384 | 48 MiB | 16 MiB | 104 MiB | 104 MiB |
| 32,768 | 64 MiB | 32 MiB | 208 MiB | 208 MiB |

## Tuning and validation

- Keeping the query-supertiles in natural order was 12–17% slower at
  8K–32K than launching the longest causal prefixes first.
- Reducing the K/V pipeline from two stages to one was 2–4% slower at the
  large sizes. Both alternatives were reverted.
- Softmax work is scheduled before waiting for the corresponding value tile,
  preserving overlap with the producer.
- No input, contraction, accumulation, or output dtype was relaxed. Outputs
  remain in the input dtype and softmax/MMA accumulation remains FP32.
- Direct FP16 and BF16 forward comparisons against XLA passed, as did compiled
  `jit`/`vmap` coverage and a 4K public custom-VJP value/gradient comparison.
  The full tracked test suite passed: 151 tests, with one unrelated
  disabled-FP64 warning.

## Conclusions

- The causal forward is now faster than cuDNN at every tested scale. At 32K,
  it cuts the previous Mosaic latency by 51% in FP16 and 54% in BF16, and is
  24% and 29% faster than cuDNN respectively.
- The end-to-end gradient call improves by roughly 11–12% at 32K solely from
  the faster forward. The unchanged causal backward is now the clear
  bottleneck and remains about 1.9x cuDNN at 32K.
- Memory scaling is still linear. This forward optimization does not change
  the measured peak; both gradient implementations use 208 MiB at 32K.
