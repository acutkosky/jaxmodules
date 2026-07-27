# RTX 5090 warp-specialized unmasked attention

Environment: RTX 5090, JAX 0.11.0, batch 1, 4 query/KV heads,
head dimension 64, unmasked self-attention. Every function was lowered and
compiled before timing. Cases ran sequentially in fresh worker processes with
GPU preallocation disabled. The 1K–32K tables use 5 warmups and 20 measured
iterations; 64K–128K use 2 warmups and 5 measured iterations.

The new path uses two compute warpgroups for adjacent query tiles and one
producer warpgroup. The producer loads each K/V tile once through TMA and both
compute warpgroups consume it from shared memory. It is selected only for
noncausal unmasked FP16/BF16 attention with D=64 and Q/K lengths at least 4096.

## FP16 forward

Times are milliseconds. “Speedup” compares the previous input-precision Mosaic
kernel with the integrated warp-specialized path. “vs cuDNN” is
Mosaic time / cuDNN time.

| Context | Previous Mosaic | Warp Mosaic | Speedup | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1K | 0.077 | 0.074 | 1.03x | 0.073 | 1.02x |
| 2K | 0.108 | 0.117 | 0.93x | 0.092 | 1.27x |
| 4K | 0.260 | 0.211 | 1.23x | 0.146 | 1.45x |
| 8K | 0.897 | 0.715 | 1.26x | 0.459 | 1.56x |
| 16K | 3.065 | 2.750 | 1.11x | 1.643 | 1.67x |
| 32K | 11.533 | 9.933 | 1.16x | 5.753 | 1.73x |

The 1K and 2K rows continue to use the old kernel; differences there are
run-to-run noise.

## FP16 gradient

These are end-to-end `jax.grad` times, including the forward call. The backward
kernel itself is unchanged, so its gain is smaller than the forward gain.

| Context | Previous Mosaic | Warp-forward VJP | Speedup | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1K | 0.192 | 0.191 | 1.00x | 0.178 | 1.07x |
| 2K | 0.327 | 0.334 | 0.98x | 0.279 | 1.20x |
| 4K | 0.916 | 0.854 | 1.07x | 0.476 | 1.79x |
| 8K | 3.833 | 3.583 | 1.07x | 1.677 | 2.14x |
| 16K | 12.774 | 12.241 | 1.04x | 6.855 | 1.79x |
| 32K | 50.015 | 47.242 | 1.06x | 22.486 | 2.10x |

## BF16 forward

| Context | Previous Mosaic | Warp Mosaic | Speedup | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1K | 0.078 | 0.073 | 1.08x | 0.064 | 1.13x |
| 2K | 0.112 | 0.115 | 0.97x | 0.098 | 1.18x |
| 4K | 0.250 | 0.210 | 1.19x | 0.145 | 1.45x |
| 8K | 0.882 | 0.709 | 1.24x | 0.447 | 1.59x |
| 16K | 3.074 | 2.718 | 1.13x | 1.664 | 1.63x |
| 32K | 11.605 | 9.857 | 1.18x | 5.906 | 1.67x |

## BF16 gradient

| Context | Previous Mosaic | Warp-forward VJP | Speedup | cuDNN | vs cuDNN |
|---:|---:|---:|---:|---:|---:|
| 1K | 0.190 | 0.197 | 0.97x | 0.180 | 1.09x |
| 2K | 0.341 | 0.338 | 1.01x | 0.278 | 1.22x |
| 4K | 0.912 | 0.865 | 1.05x | 0.466 | 1.86x |
| 8K | 3.467 | 3.283 | 1.06x | 1.646 | 1.99x |
| 16K | 12.622 | 12.272 | 1.03x | 6.737 | 1.82x |
| 32K | 49.015 | 48.380 | 1.01x | 22.279 | 2.17x |

## Large-context FP16 and memory

Both implementations completed unmasked forward and backward through 128K.
The cuDNN flash path is linear-memory here; the earlier quadratic/OOM behavior
applies to the dense adapter used for an arbitrary materialized mask, not
cuDNN’s unmasked flash core.

| Context | Mosaic forward | cuDNN forward | Mosaic gradient | cuDNN gradient |
|---:|---:|---:|---:|---:|
| 64K | 37.701 ms | 21.402 ms | 186.787 ms | 79.797 ms |
| 128K | 142.598 ms | 83.899 ms | 772.896 ms | 319.929 ms |

Process peak allocation above the post-input baseline:

| Context | Mosaic forward | cuDNN forward | Mosaic gradient | cuDNN gradient |
|---:|---:|---:|---:|---:|
| 1K | 33 MiB | 1 MiB | 33 MiB | 6 MiB |
| 2K | 34 MiB | 2 MiB | 34 MiB | 12 MiB |
| 4K | 36 MiB | 4 MiB | 42 MiB | 24 MiB |
| 8K | 40 MiB | 8 MiB | 56 MiB | 48 MiB |
| 16K | 48 MiB | 16 MiB | 104 MiB | 104 MiB |
| 32K | 64 MiB | 32 MiB | 208 MiB | 208 MiB |
| 64K | 112 MiB | 64 MiB | 416 MiB | 416 MiB |
| 128K | 208 MiB | 128 MiB | 832 MiB | 832 MiB |

## Tuning decisions

- Two compute warpgroups plus one producer was retained.
- Four compute warpgroups were 23–47% slower because each compute warpgroup
  had to drop from 232 to 112 registers.
- Pipeline depth 2 was best at long context. Depths 1 and 3 were neutral or
  slower at 8K–32K.
- Direct Q loads and output stores beat staging Q/output through shared memory.
- The TMA-compatible shared layout does not expose contiguous two-element
  `mma.sync` RHS vectors. A current `inline_mgpu` helper therefore performs
  scalar shared loads with the same lane ownership and locally packs each
  two-element MMA register. No deprecated Pallas API is used.

Raw data:

- `rtx5090_warp_specialized_fp16_unmasked_2026-07-27.json`
- `rtx5090_warp_specialized_bf16_unmasked_2026-07-27.json`
- `rtx5090_warp_specialized_fp16_xlarge_2026-07-27.json`
