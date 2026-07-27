# RTX 5090 attention results

These results were collected on an NVIDIA RTX 5090 with JAX 0.11.0. Every
case was explicitly lowered and compiled, warmed up five times, and timed for
20 iterations. Cases ran sequentially in fresh worker processes with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; benchmark workers never shared the GPU.

The primary workload is float16, `B=1`, `Hq=Hkv=4`, and `D=64`. Times are
median milliseconds. Memory is the process peak allocation above the live
input-buffer baseline. A backward measurement is one compiled gradient call,
including the forward pass needed by that gradient.

Mosaic uses its internally tuned 64x64 tile. Separate 128-wide query and K/V
tile experiments were slower because of register pressure and were discarded.
The original mapped implementation was retuned at every sequence length. Its
reported result is the fastest tested equal query/KV tile; candidates ranged
from 256 through 8,192, subject to the sequence length. The winning tile is
shown in parentheses.

## What changed

The previous Mosaic backward traversed every active score tile twice: once in
query-major order for dQ, and once in K/V-major order for dK and dV. The new
default traverses it once in query-major order, writes dQ directly, and uses
FP32 atomics to accumulate dK and dV.

This does not relax precision. FP16 still uses two low-precision components,
BF16 uses three, matrix products accumulate in FP32, the atomics are FP32, and
Mosaic approximate math remains disabled. The old two-pass algorithm remains
available as `backward_strategy="minimal"` when its smaller temporary is more
important than runtime.

## General callable mask

The representative callable mask is batch/head dependent and combines a
radius with a modulo pattern. Mosaic and mapped attention evaluate it inside
each 64x64 score tile and Mosaic skips fully masked tiles. XLA and cuDNN receive
the equivalent dense boolean mask.

### Forward

The one-pass change is backward-only, so current and two-pass Mosaic have the
same forward result.

| N | Mosaic | Original mapped best | cuDNN | XLA |
|---:|---:|---:|---:|---:|
| 1,024 | 0.072 ms / 33 MiB | 0.120 ms / 129 MiB (1,024) | 0.121 ms / 17 MiB | 0.058 ms / 129 MiB |
| 2,048 | 0.084 ms / 34 MiB | 0.218 ms / 280 MiB (2,048) | 0.202 ms / 66 MiB | 0.104 ms / 256 MiB |
| 4,096 | 0.118 ms / 36 MiB | 0.901 ms / 280 MiB (2,048) | 0.499 ms / 260 MiB | 0.527 ms / 612 MiB |
| 8,192 | 0.226 ms / 40 MiB | 3.464 ms / 304 MiB (2,048) | 1.368 ms / 520 MiB | 2.097 ms / 2.1 GiB |
| 16,384 | 0.580 ms / 48 MiB | 13.138 ms / 280 MiB (2,048) | 5.627 ms / 2.0 GiB | 8.533 ms / 8.1 GiB |
| 32,768 | 1.858 ms / 80.5 MiB | 52.383 ms / 320 MiB (2,048) | 21.824 ms / 8.0 GiB | 45-second timeout |

### Compiled gradient

| N | Mosaic one-pass | Mosaic two-pass | Original mapped best | cuDNN | XLA |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.238 ms / 34 MiB | 0.267 ms / 34 MiB | 0.209 ms / 129 MiB (1,024) | 0.218 ms / 35 MiB | 0.099 ms / 129 MiB |
| 2,048 | 0.246 ms / 36 MiB | 0.300 ms / 36 MiB | 0.615 ms / 288 MiB (2,048) | 0.353 ms / 134 MiB | 0.266 ms / 256 MiB |
| 4,096 | 0.399 ms / 42 MiB | 0.469 ms / 42 MiB | 2.713 ms / 620 MiB (4,096) | 1.280 ms / 520 MiB | 1.194 ms / 598 MiB |
| 8,192 | 0.839 ms / 88 MiB | 0.991 ms / 56 MiB | 11.062 ms / 2.3 GiB (8,192) | 5.043 ms / 1.5 GiB | 4.856 ms / 2.1 GiB |
| 16,384 | 2.053 ms / 176 MiB | 2.341 ms / 104 MiB | 44.175 ms / 304 MiB (2,048) | 18.337 ms / 6.1 GiB | 19.894 ms / 8.1 GiB |
| 32,768 | 5.935 ms / 352 MiB | 7.125 ms / 208 MiB | 167.779 ms / 352 MiB (2,048) | OOM: 24.03 GiB request | 45-second timeout |

The one-pass latency reduction over two-pass is 10.7%, 18.2%, 15.0%, 15.3%,
12.3%, and 16.7% across the table. Its compiler-estimated temporary grows
linearly from 4.0 MiB at `N=1,024` to 129.0 MiB at `N=32,768`; two-pass grows
from 3.0 MiB to 97.0 MiB. Allocator granularity makes the observed peak
difference larger at some sizes, but both paths remain linear-memory.

The cuDNN OOM here is not evidence that its flash-attention core is quadratic.
JAX materializes this arbitrary mask as a dense bias, and backward also
produces the corresponding dense bias gradient. Those dense `N x N` buffers
dominate the result.

## Unmasked dense work

Without a user mask, cuDNN exercises its linear-memory flash path. This is the
cleanest measurement of the remaining raw-kernel performance gap.

### Forward

| N | Mosaic | cuDNN | XLA |
|---:|---:|---:|---:|
| 1,024 | 0.085 ms / 33 MiB | 0.077 ms / 1 MiB | 0.056 ms / 129 MiB |
| 2,048 | 0.135 ms / 34 MiB | 0.102 ms / 2 MiB | 0.108 ms / 256 MiB |
| 4,096 | 0.292 ms / 36 MiB | 0.152 ms / 4 MiB | 0.526 ms / 612 MiB |
| 8,192 | 0.982 ms / 40 MiB | 0.445 ms / 8 MiB | 2.115 ms / 2.1 GiB |
| 16,384 | 3.482 ms / 48 MiB | 1.616 ms / 16 MiB | 8.766 ms / 8.1 GiB |
| 32,768 | 13.073 ms / 80.5 MiB | 5.545 ms / 32 MiB | 45-second timeout |

### Compiled gradient

| N | Mosaic one-pass | Mosaic two-pass | cuDNN | XLA |
|---:|---:|---:|---:|---:|
| 1,024 | 0.251 ms / 34 MiB | 0.309 ms / 34 MiB | 0.178 ms / 6 MiB | 0.108 ms / 129 MiB |
| 2,048 | 0.461 ms / 36 MiB | 0.561 ms / 36 MiB | 0.278 ms / 12.1 MiB | 0.255 ms / 256 MiB |
| 4,096 | 1.241 ms / 42 MiB | 1.759 ms / 42 MiB | 0.471 ms / 24.1 MiB | 1.216 ms / 598 MiB |
| 8,192 | 5.148 ms / 88 MiB | 7.032 ms / 56 MiB | 1.670 ms / 48.3 MiB | 4.806 ms / 2.1 GiB |
| 16,384 | 18.043 ms / 176 MiB | 25.180 ms / 104 MiB | 7.016 ms / 104 MiB | 19.575 ms / 8.1 GiB |
| 32,768 | 66.740 ms / 352 MiB | 91.264 ms / 208 MiB | 22.394 ms / 208 MiB | 45-second timeout |

One-pass reduces unmasked FP16 gradient latency by 17.8% to 29.4%. cuDNN does
not OOM at `N=32,768`: its compiled gradient is about 3.0x faster and uses
208 MiB peak versus Mosaic's 352 MiB. The Mosaic compiler temporary is 129 MiB
there; selecting two-pass returns the device peak to 208 MiB at the cost of
36.7% more runtime.

## Causal work

### Forward

| N | Mosaic | cuDNN | XLA |
|---:|---:|---:|---:|
| 1,024 | 0.085 ms / 33 MiB | 0.077 ms / 1 MiB | 0.067 ms / 129 MiB |
| 2,048 | 0.130 ms / 34 MiB | 0.104 ms / 2 MiB | 0.099 ms / 256 MiB |
| 4,096 | 0.256 ms / 36 MiB | 0.160 ms / 4 MiB | 0.545 ms / 612 MiB |
| 8,192 | 0.655 ms / 40 MiB | 0.407 ms / 8 MiB | 2.137 ms / 2.1 GiB |
| 16,384 | 2.043 ms / 48 MiB | 1.071 ms / 16 MiB | 8.789 ms / 8.1 GiB |
| 32,768 | 7.112 ms / 80.5 MiB | 3.409 ms / 32 MiB | 45-second timeout |

### Compiled gradient

| N | Mosaic one-pass | Mosaic two-pass | cuDNN | XLA |
|---:|---:|---:|---:|---:|
| 1,024 | 0.264 ms / 34 MiB | 0.316 ms / 34 MiB | 0.181 ms / 6 MiB | 0.101 ms / 129 MiB |
| 2,048 | 0.473 ms / 36 MiB | 0.564 ms / 36 MiB | 0.289 ms / 12.1 MiB | 0.242 ms / 256 MiB |
| 4,096 | 1.106 ms / 42 MiB | 1.458 ms / 42 MiB | 0.499 ms / 24.1 MiB | 1.208 ms / 598 MiB |
| 8,192 | 3.138 ms / 88 MiB | 3.757 ms / 56 MiB | 1.184 ms / 48.3 MiB | 5.656 ms / 2.1 GiB |
| 16,384 | 10.949 ms / 176 MiB | 12.407 ms / 104 MiB | 3.415 ms / 104 MiB | 20.140 ms / 8.1 GiB |
| 32,768 | 40.287 ms / 352 MiB | 45.850 ms / 208 MiB | 11.711 ms / 208 MiB | 45-second timeout |

One-pass reduces causal FP16 gradient latency by 11.7% to 24.1%. cuDNN again
remains linear and does not OOM. At `N=32,768`, Mosaic is 2.1x slower forward
and 3.4x slower for the compiled gradient.

## BF16 same-precision check

BF16 requires three decomposition components rather than FP16's two, so
eliminating the duplicate score traversal has a larger payoff.

| N | Mosaic one-pass | Mosaic two-pass | Latency reduction | Device peak | Compiler temp, one/two |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.340 ms | 0.439 ms | 22.4% | 35 MiB | 5.0 / 4.0 MiB |
| 2,048 | 0.647 ms | 0.835 ms | 22.5% | 38 MiB | 10.1 / 8.1 MiB |
| 4,096 | 1.892 ms | 2.850 ms | 33.6% | 44 MiB | 20.1 / 16.1 MiB |
| 8,192 | 7.730 ms | 13.784 ms | 43.9% | 88 MiB | 40.3 / 32.3 MiB |
| 16,384 | 27.801 ms | 47.506 ms | 41.5% | 176 MiB | 80.5 / 64.5 MiB |
| 32,768 | 106.246 ms | 205.164 ms | 48.2% | 352 MiB | 161.0 / 129.0 MiB |

The observed device peaks happen to fall in the same allocator buckets for
both BF16 strategies; the compiler temporary exposes the actual difference.

## Assessment

For large sparse callable masks, Mosaic now has both the strongest runtime and
the only practical backward at the largest tested scale: at `N=32,768` it
takes 5.935 ms, while cuDNN's dense-mask adaptation OOMs and XLA times out.
The original mapped custom VJP remains useful as the universal fallback, but
is 28x slower on that workload even with its best tested tile.

For unmasked and causal attention, cuDNN remains the target. Closing that gap
without changing precision likely requires a separately tuned dense/causal
kernel with asynchronous global-to-shared pipelining and larger effective
tiles. Simply changing the current tile to 128 was counterproductive on SM120;
it increased register pressure without adding the load/MMA overlap present in
the vendor kernel.

Correctness was checked by eight Mosaic GPU tests, including FP16 and BF16
gradient agreement with the mapped implementation and composition under JIT
and VMAP. The broader mapped-attention suite adds 63 passing regression tests.
