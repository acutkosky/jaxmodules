# Attention benchmarks

`benchmark_mapped_attention.py` compares `masked_attention_via_map` with JAX's
XLA and cuDNN scaled-dot-product attention implementations.

Run the smoke suite while iterating:

```bash
uv run python benchmarks/benchmark_mapped_attention.py \
  --seq-len 512 \
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
  --causal \
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

Both forward and forward-plus-backward modes are measured by default. Pass
`--mode forward` or `--mode backward` to select one.

`--block-size` accepts multiple values so that mapped attention is tuned at
each workload rather than compared against XLA and cuDNN at an arbitrary tile
size. The block-size field is ignored by XLA and cuDNN, but those cases are
still repeated to keep each result independently isolated.

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

cuDNN results are reported as unavailable when the current backend or input
configuration does not support cuDNN attention. The XLA and mapped
implementations remain usable on CPU.
