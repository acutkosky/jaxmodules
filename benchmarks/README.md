# Attention benchmarks

`benchmark_mapped_attention.py` compares the integrated Mosaic fast path, its
large-unmasked warp kernel in isolation (`mosaic-warp`), the previous mapped
implementation, and JAX's XLA and cuDNN scaled-dot-product attention
implementations.

The checked-in [RTX 5090 general-mask results](results/rtx5090_general_attention_2026-07-26.md)
show the benchmark format and the large-context crossover.
The [warp-specialized results](results/rtx5090_warp_specialized_2026-07-27.md)
compare its input-precision FP16/BF16 performance and memory with cuDNN through
128K context.
The [causal warp-specialized results](results/rtx5090_causal_warp_2026-07-29.md)
compare the maximal-causal fast path with the previous Mosaic kernel and cuDNN
from 1K through 32K context.
The [non-atomic causal-backward results](results/rtx5090_causal_split_backward_2026-07-29.md)
compare the split query-major/key-major VJP with the previous atomic backward
and cuDNN from 1K through 128K context.
The [flexible-head results](results/rtx5090_flexible_dimensions_2026-07-29.md)
cover 8-head, 1,024-dimensional-per-token attention at scale and validate
per-head dimensions from 64 through 2,048.

Run the smoke suite while iterating:

```bash
uv run python benchmarks/benchmark_mapped_attention.py \
  --seq-len 1024 \
  --block-size 64 \
  --warmup 2 \
  --iterations 5
```

Run a larger causal sweep:

```bash
uv run python benchmarks/benchmark_mapped_attention.py \
  --seq-len 1024 2048 4096 \
  --block-size 128 256 512 \
  --batch-size 2 \
  --query-heads 8 \
  --kv-heads 8 \
  --head-dim 64 \
  --dtype bfloat16 \
  --mask causal \
  --warmup 5 \
  --iterations 20 \
  --json-output benchmark-results.json
```

Each implementation and mode runs in a fresh process. This is important because
JAX's device allocator keeps allocations alive and exposes process-wide peak
statistics. The benchmark reports:

- compilation time;
- median, minimum, and mean execution latency;
- tokens processed per second;
- peak device allocation reported by the backend;
- input-buffer baseline and peak allocation above that baseline;
- compiler-estimated argument, output, temporary, and aliased bytes.

JSON results also include a versioned `environment` record for each worker:
Python, operating system and kernel; JAX, JAXlib, CUDA-plugin, CUDA-library,
cuDNN, and Triton package versions; the CUDA and cuDNN build/runtime versions
reported by JAX; NVIDIA driver, GPU, compute capability, PCI bus, and physical
memory reported by `nvidia-smi`; and relevant JAX/XLA/CUDA environment
variables. Metadata collection is best-effort and does not turn an otherwise
valid benchmark into a failure.

Each worker is terminated after five minutes by default so a pathological
compile or allocation does not prevent later implementations from running.
Use `--case-timeout-seconds` to change that limit.

Both forward and forward-plus-backward modes are measured by default. Pass
`--mode forward` or `--mode backward` to select one.
`--implementation mosaic-warp` is an unmasked/maximal-causal forward-only
diagnostic for the private warp kernel; the normal `mosaic` implementation
dispatches to it automatically at supported context sizes.

`--block-size` accepts multiple values so that mapped attention is tuned at
each workload rather than compared against XLA and cuDNN at an arbitrary tile
size. Mosaic tunes independently: the D=64 kernels use internal 64x64 score
tiles and the dimension-generic D=80..2048 kernels use 32x32 tiles. The
requested block sizes therefore do not change Mosaic. Every JSON record and
printed row identifies both the requested and effective tile. XLA and cuDNN
report no effective tile because their libraries choose internally.

By default, each `--block-size` is used for both query and K/V tiles. Pass one
or more `--kv-block-size` values to benchmark the Cartesian product of query
and K/V tile sizes:

```bash
uv run python benchmarks/benchmark_mapped_attention.py \
  --implementation mapped \
  --seq-len 8192 \
  --block-size 256 512 1024 \
  --kv-block-size 128 256 512 \
  --warmup 3 \
  --iterations 10
```

Mapped and Mosaic backward choose a strategy by default. To benchmark the
generic two-pass variant that eliminates sequence-sized FP32 gradient carries,
pass `--mapped-backward-strategy minimal`. To force a single score-tile
traversal, pass `--mapped-backward-strategy one_pass`; Mosaic then uses FP32
atomic dK/dV accumulation. The latter can help sufficiently sparse masks but
uses larger compiler temporaries. Inputs, outputs, forward residuals, or
score-tile temporaries may dominate the measured peak on other shapes.

The mapped and Mosaic standard-attention paths return the input dtype. They
compute softmax reductions and low-precision matrix-product accumulations in
FP32, matching the conventional FlashAttention precision policy. Tile and
backward-strategy sweeps do not relax that policy.

Use `--mask unmasked`, `--mask causal`, `--mask general`, or
`--mask general-dense`. The general case uses a sparse noncausal
batch/head-dependent radius-plus-modulo callable; general-dense uses a
coordinate-dependent mask with no fully empty tiles. XLA and cuDNN receive
equivalent dense boolean masks, so their memory results include the cost of
representing either general mask.

cuDNN results are reported as unavailable when the current backend or input
configuration does not support cuDNN attention. The XLA and mapped
implementations remain usable on CPU; Mosaic requires a supported GPU and
float16 or bfloat16 inputs with a head dimension that is a multiple of 16 from
64 through 2048.
