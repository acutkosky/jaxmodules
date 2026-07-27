# RTX 5090 input-precision attention results

These measurements establish the baseline before attempting a multi-warpgroup
kernel. They use JAX 0.11.0 on an RTX 5090 with FP16 or BF16 inputs,
`B=1`, `Hq=Hkv=4`, and `D=64`. Every case was explicitly lowered and
compiled, warmed up five times, and timed for 30 iterations. Implementations
ran sequentially in fresh processes with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

The numerical contract now matches conventional FlashAttention:

- the result has the same dtype as Q/K/V;
- score scaling, softmax reductions, normalizers, and matrix-product
  accumulators use FP32;
- tensor-core operands use the input dtype.

The previous Mosaic implementation returned FP32 and represented FP32
probabilities and cotangents with two FP16 or three BF16 components. Removing
that emulation changes the numerical contract, eliminates repeated tensor-core
products, and makes the cuDNN comparison substantially closer to
like-for-like.

A backward measurement is one compiled gradient call, including its required
forward. Times below are median milliseconds. `ratio` is Mosaic divided by
cuDNN, so a value above one means cuDNN is faster.

## FP16

### Unmasked

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.077 | 0.072 | 1.07x | 0.192 | 0.171 | 1.12x |
| 2,048 | 0.108 | 0.091 | 1.19x | 0.327 | 0.289 | 1.13x |
| 4,096 | 0.260 | 0.150 | 1.73x | 0.916 | 0.491 | 1.87x |
| 8,192 | 0.897 | 0.452 | 1.98x | 3.833 | 1.676 | 2.29x |
| 16,384 | 3.065 | 1.651 | 1.86x | 12.774 | 6.674 | 1.91x |
| 32,768 | 11.533 | 6.102 | 1.89x | 50.015 | 22.406 | 2.23x |

### Causal

`is_causal=True` is now a structural hint that intersects an optional user
mask. In this maximal-causal benchmark there is no additional user mask:
tiles above the diagonal are skipped, tiles below it are unmasked, and
diagonal tiles use the triangular predicate.

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.079 | 0.074 | 1.07x | 0.187 | 0.185 | 1.01x |
| 2,048 | 0.111 | 0.102 | 1.09x | 0.325 | 0.295 | 1.10x |
| 4,096 | 0.226 | 0.161 | 1.40x | 0.788 | 0.502 | 1.57x |
| 8,192 | 0.510 | 0.402 | 1.27x | 1.904 | 1.169 | 1.63x |
| 16,384 | 1.528 | 1.095 | 1.40x | 6.705 | 3.609 | 1.86x |
| 32,768 | 6.101 | 3.549 | 1.72x | 23.342 | 11.689 | 2.00x |

### General callable mask

Mosaic evaluates the coordinate callable inside each 64x64 tile and skips
empty tiles. cuDNN receives the equivalent materialized dense boolean mask.
Consequently, ratios below one favor Mosaic.

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.068 | 0.113 | 0.60x | 0.181 | 0.210 | 0.86x |
| 2,048 | 0.081 | 0.194 | 0.42x | 0.191 | 0.354 | 0.54x |
| 4,096 | 0.108 | 0.498 | 0.22x | 0.303 | 1.278 | 0.24x |
| 8,192 | 0.208 | 1.342 | 0.15x | 0.678 | 4.549 | 0.15x |
| 16,384 | 0.565 | 5.522 | 0.10x | 1.817 | 17.540 | 0.10x |
| 32,768 | 1.871 | 22.147 | 0.08x | 5.678 | OOM | - |

The 32K cuDNN gradient requested 24.03 GiB and OOMed. This is caused by the
dense adaptation of the arbitrary mask, not by cuDNN's flash-attention core.

## BF16

### Unmasked

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.078 | 0.063 | 1.24x | 0.190 | 0.178 | 1.07x |
| 2,048 | 0.112 | 0.096 | 1.17x | 0.341 | 0.273 | 1.25x |
| 4,096 | 0.250 | 0.141 | 1.77x | 0.912 | 0.470 | 1.94x |
| 8,192 | 0.882 | 0.442 | 2.00x | 3.467 | 1.665 | 2.08x |
| 16,384 | 3.074 | 1.644 | 1.87x | 12.622 | 6.654 | 1.90x |
| 32,768 | 11.605 | 6.264 | 1.85x | 49.015 | 22.635 | 2.17x |

### Causal

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.076 | 0.067 | 1.13x | 0.176 | 0.173 | 1.02x |
| 2,048 | 0.111 | 0.094 | 1.18x | 0.321 | 0.278 | 1.15x |
| 4,096 | 0.226 | 0.158 | 1.43x | 0.794 | 0.482 | 1.65x |
| 8,192 | 0.504 | 0.401 | 1.26x | 1.884 | 1.163 | 1.62x |
| 16,384 | 1.495 | 1.076 | 1.39x | 6.406 | 3.424 | 1.87x |
| 32,768 | 5.693 | 3.414 | 1.67x | 22.851 | 11.799 | 1.94x |

### General callable mask

| N | Mosaic fwd | cuDNN fwd | ratio | Mosaic grad | cuDNN grad | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.068 | 0.118 | 0.58x | 0.176 | 0.212 | 0.83x |
| 2,048 | 0.067 | 0.196 | 0.34x | 0.200 | 0.346 | 0.58x |
| 4,096 | 0.102 | 0.510 | 0.20x | 0.295 | 1.310 | 0.23x |
| 8,192 | 0.213 | 1.447 | 0.15x | 0.680 | 4.589 | 0.15x |
| 16,384 | 0.571 | 5.703 | 0.10x | 1.837 | 17.899 | 0.10x |
| 32,768 | 1.857 | 23.008 | 0.08x | 5.859 | OOM | - |

## Memory

FP16 and BF16 landed in the same allocator buckets. General-mask cuDNN memory
is shown separately because it includes the dense mask adaptation.

| N | Mosaic fwd | cuDNN dense fwd | cuDNN general fwd | Mosaic grad | cuDNN dense grad | cuDNN general grad |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 33 MiB | 1 MiB | 17 MiB | 33 MiB | 6 MiB | 35 MiB |
| 2,048 | 34 MiB | 2 MiB | 66 MiB | 34-35 MiB | 12.1 MiB | 134 MiB |
| 4,096 | 36 MiB | 4 MiB | 260 MiB | 42 MiB | 24.1 MiB | 520 MiB |
| 8,192 | 40 MiB | 8 MiB | 520 MiB | 56 MiB | 48.3 MiB | 1.5 GiB |
| 16,384 | 48 MiB | 16 MiB | 2.0 GiB | 104 MiB | 104 MiB | 6.1 GiB |
| 32,768 | 64 MiB | 32 MiB | 8.0 GiB | 208 MiB | 208 MiB | OOM |

At 32K, the old higher-precision FP16 implementation used 80.5 MiB forward
and 352 MiB backward. Returning the input dtype and removing the component
axis reduces those peaks to 64 and 208 MiB. For unmasked FP16, runtime fell
from 13.043 to 11.533 ms forward and from 67.067 to 50.015 ms backward. For
causal FP16 it fell from 6.362 to 6.101 ms and from 32.034 to 23.342 ms.

The remaining dense gap is therefore approximately 1.7-1.9x forward and
1.9-2.2x backward at 32K. This is the baseline for the subsequent
multi-warpgroup K/V-reuse experiment.
