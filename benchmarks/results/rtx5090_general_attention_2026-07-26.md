# RTX 5090 general-mask attention results

These results were collected on an NVIDIA RTX 5090 with JAX 0.11.0. Every
case was lowered and compiled explicitly, then run in a fresh worker process
with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

The workload is float16, `B=1`, `Hq=Hkv=4`, and `D=64`. The general callable
mask is batch/head dependent and combines a radius with a modulo pattern. The
Mosaic and mapped implementations evaluate the callable inside each score
tile. XLA and cuDNN receive the equivalent dense boolean mask.

Times are median milliseconds. Memory is the process peak allocation above
the input-buffer baseline. "Backward" measures the compiled gradient call,
including its forward pass. The mapped column uses the fastest tested equal
query/KV tile at each sequence length.

## Forward

| N | Mosaic | Mapped best (tile) | XLA | cuDNN |
|---:|---:|---:|---:|---:|
| 1,024 | 0.087 ms / 33 MiB | 0.119 ms / 129 MiB (1,024) | 0.066 ms / 129 MiB | 0.123 ms / 17 MiB |
| 2,048 | 0.124 ms / 34 MiB | 0.212 ms / 280 MiB (2,048) | 0.114 ms / 256 MiB | 0.215 ms / 66 MiB |
| 4,096 | 0.301 ms / 36 MiB | 0.928 ms / 304 MiB (2,048) | 0.545 ms / 612 MiB | 0.492 ms / 260 MiB |
| 8,192 | 1.061 ms / 40 MiB | 3.508 ms / 304 MiB (2,048) | 2.066 ms / 2.1 GiB | 1.349 ms / 520 MiB |
| 16,384 | 3.808 ms / 48 MiB | 14.232 ms / 280 MiB (2,048) | 8.480 ms / 8.1 GiB | 5.579 ms / 2.0 GiB |
| 32,768 | 13.754 ms / 80.5 MiB | 57.987 ms / 320 MiB (2,048) | 30-second timeout | 21.780 ms / 8.0 GiB |

## Compiled gradient (forward and backward)

| N | Mosaic | Mapped best (tile) | XLA | cuDNN |
|---:|---:|---:|---:|---:|
| 1,024 | 0.310 ms / 34 MiB | 0.219 ms / 129 MiB (1,024) | 0.122 ms / 129 MiB | 0.233 ms / 35 MiB |
| 2,048 | 0.575 ms / 36 MiB | 0.605 ms / 288 MiB (2,048) | 0.294 ms / 256 MiB | 0.380 ms / 134 MiB |
| 4,096 | 1.736 ms / 42 MiB | 2.701 ms / 620 MiB (4,096) | 1.203 ms / 584 MiB | 1.266 ms / 520 MiB |
| 8,192 | 6.893 ms / 56 MiB | 11.309 ms / 2.3 GiB (8,192) | 5.471 ms / 2.1 GiB | 4.544 ms / 1.5 GiB |
| 16,384 | 24.603 ms / 104 MiB | 45.677 ms / 304 MiB (2,048) | 19.663 ms / 8.1 GiB | 17.923 ms / 6.1 GiB |
| 32,768 | 87.013 ms / 208 MiB | 177.569 ms / 352 MiB (2,048) | 30-second timeout | OOM requesting 24.03 GiB |

At `N=16,384`, the old full-tile mapped backward also OOMed while requesting
9.03 GiB; its 2,048 tile is both faster and much smaller. At `N=32,768`,
mapped tiles of 4,096 and 8,192 were also tested and were slower than 2,048.

## Causal specialization spot check

The causal path uses the same kernel and callable-mask interface, with fully
masked blocks pruned from the loop.

| N | Mode | Mosaic | Mapped best | cuDNN |
|---:|---|---:|---:|---:|
| 4,096 | forward | 0.258 ms / 36 MiB | 0.462 ms / 280 MiB | 0.165 ms / 4 MiB |
| 4,096 | gradient | 1.476 ms / 42 MiB | 1.922 ms / 134 MiB | 0.517 ms / 24 MiB |
| 8,192 | forward | 0.659 ms / 40 MiB | 1.396 ms / 280 MiB | 0.407 ms / 8 MiB |
| 8,192 | gradient | 3.766 ms / 56 MiB | 6.156 ms / 280 MiB | 1.244 ms / 48 MiB |

cuDNN is still substantially faster for these moderate causal workloads. The
main benefit of the new implementation is general callable masking and stable
linear memory at scales where dense-mask vendor backward runs out of memory.
