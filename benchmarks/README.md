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
`--implementation mosaic-warp` is an unmasked-forward-only diagnostic for the
private warp kernel; the normal `mosaic` implementation dispatches to it
automatically at supported context sizes.

`--block-size` accepts multiple values so that mapped attention is tuned at
each workload rather than compared against XLA and cuDNN at an arbitrary tile
size. Mosaic forward and backward currently use internal 64x64 tiles, so the
requested block sizes do not change Mosaic. The block-size field is also
ignored by XLA and cuDNN, but those cases are still repeated to keep each
result independently isolated.

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

Mapped and Mosaic backward use their faster one-pass strategies by default. To
benchmark the two-pass variants that eliminate sequence-sized FP32 gradient
carries, pass `--mapped-backward-strategy minimal`. Mosaic's one-pass path
combines query-major programs with FP32 atomic dK/dV accumulation. The
two-pass variant can reduce the total peak when that carry is material;
inputs, outputs, forward residuals, or score-tile temporaries may dominate
other shapes.

The mapped and Mosaic standard-attention paths return the input dtype. They
compute softmax reductions and low-precision matrix-product accumulations in
FP32, matching the conventional FlashAttention precision policy. Tile and
backward-strategy sweeps do not relax that policy.

Use `--mask unmasked`, `--mask causal`, or `--mask general`. The general case
uses a noncausal batch/head-dependent radius-plus-modulo callable for Mosaic
and mapped attention. XLA and cuDNN receive the equivalent dense boolean mask,
so their memory results include the cost of representing that general mask.

cuDNN results are reported as unavailable when the current backend or input
configuration does not support cuDNN attention. The XLA and mapped
implementations remain usable on CPU; Mosaic requires a supported GPU and
float16 or bfloat16 inputs with 64-wide heads.
