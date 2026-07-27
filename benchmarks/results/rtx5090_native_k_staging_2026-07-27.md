# RTX 5090 native single-copy K staging

Environment: RTX 5090, JAX 0.11.0, batch 1, 4 query/KV heads,
head dimension 64, unmasked self-attention. All timings are end-to-end
`jax.grad` latency in milliseconds, including the forward pass. Every function
was explicitly lowered and compiled before timing. Cases ran sequentially in
fresh worker processes with GPU preallocation disabled.

The warp-specialized backward now stages each row-major K tile once. Native
`ldmatrix.trans` loads derive the column-major `mma.sync` operand needed by dQ,
while the existing native load derives the transposed score operand. A two-slot
K pipeline preserves producer overlap and occupies the same 16 KiB of shared
memory as the former K plus explicitly transposed-K buffers.

## Direct FP16 before/after checkpoint

Five warmups and 20 measured iterations:

| Context | Previous warp backward | Single-copy K | Speedup | Compiler temp before | Compiler temp after |
|---:|---:|---:|---:|---:|---:|
| 4K | 0.715 | 0.710 | 1.01x | 12.1 MiB | 10.1 MiB |
| 8K | 2.705 | 2.661 | 1.02x | 24.3 MiB | 20.3 MiB |
| 16K | 10.767 | 10.291 | 1.05x | 48.5 MiB | 40.5 MiB |
| 32K | 41.561 | 38.827 | 1.07x | 97.0 MiB | 81.0 MiB |

The compiler temporary falls by approximately 16.5%. Process peak allocation
is unchanged: 42, 56, 104, and 208 MiB above the post-input baseline.

## Fresh Mosaic and cuDNN comparison

Five warmups and 20 measured iterations:

| Dtype | Context | Mosaic | cuDNN | Mosaic / cuDNN | Mosaic peak | cuDNN peak |
|:---|---:|---:|---:|---:|---:|---:|
| FP16 | 4K | 0.695 | 0.469 | 1.48x | 42 MiB | 24 MiB |
| FP16 | 8K | 2.654 | 1.631 | 1.63x | 56 MiB | 48 MiB |
| FP16 | 16K | 10.950 | 6.244 | 1.75x | 104 MiB | 104 MiB |
| FP16 | 32K | 40.917 | 22.430 | 1.82x | 208 MiB | 208 MiB |
| BF16 | 4K | 0.687 | 0.487 | 1.41x | 42 MiB | 24 MiB |
| BF16 | 8K | 2.650 | 1.644 | 1.61x | 56 MiB | 48 MiB |
| BF16 | 16K | 11.068 | 6.459 | 1.71x | 104 MiB | 104 MiB |
| BF16 | 32K | 40.897 | 22.730 | 1.80x | 208 MiB | 208 MiB |

BF16 runtime is approximately neutral relative to the preceding checkpoint
within observed run-to-run variance, while retaining the temporary-memory
reduction.

At 64K FP16, using two warmups and five measured iterations, Mosaic took
164.200 ms and cuDNN took 81.020 ms. Both peaked 416 MiB above the input
baseline. Compiled temporary memory was 162 MiB for Mosaic and 194 MiB for
cuDNN.

## Validation

The complete uv-managed suite passed: 151 tests. This includes FP16 and BF16
comparison against XLA for the warp-specialized backward and public compiled
composition coverage.
