"""Time generated Mosaic backward passes independently for kernel tuning."""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from jaxmodules._mosaic_attention import (
    _mosaic_attention_backward_warp_specialized_dkv,
    _mosaic_attention_backward_warp_specialized_dkv_split,
    _mosaic_attention_backward_warp_specialized_dq,
    _mosaic_attention_forward_warp_specialized,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--unmasked", action="store_true")
    parser.add_argument("--split-dkv", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    return parser


def _time(compiled, arguments, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(compiled(*arguments))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*arguments))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    args = _parser().parse_args()
    dtype = {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }[args.dtype]
    shape = (1, args.seq_len, args.heads, args.head_dim)
    keys = jax.random.split(jax.random.key(0), 4)
    query = jax.random.normal(keys[0], shape, dtype=dtype)
    key = jax.random.normal(keys[1], shape, dtype=dtype)
    value = jax.random.normal(keys[2], shape, dtype=dtype)
    upstream = jax.random.normal(keys[3], shape, dtype=jnp.float32)
    is_causal = not args.unmasked

    forward = jax.jit(
        lambda q, k, v: _mosaic_attention_forward_warp_specialized(
            q,
            k,
            v,
            is_causal=is_causal,
        )
    ).lower(query, key, value).compile()
    output, log_normalizer = forward(query, key, value)
    jax.block_until_ready((output, log_normalizer))

    dq = jax.jit(
        lambda q, k, v, o, lse, do: (
            _mosaic_attention_backward_warp_specialized_dq(
                q,
                k,
                v,
                o,
                lse,
                do,
                is_causal=is_causal,
            )
        )
    ).lower(
        query,
        key,
        value,
        output,
        log_normalizer,
        upstream,
    ).compile()
    if args.split_dkv:
        dkv_function = _mosaic_attention_backward_warp_specialized_dkv_split
    else:
        dkv_function = _mosaic_attention_backward_warp_specialized_dkv

    dkv = jax.jit(
        lambda q, k, v, o, lse, do: (
            dkv_function(
                q,
                k,
                v,
                o,
                lse,
                do,
                is_causal=is_causal,
            )
        )
    ).lower(
        query,
        key,
        value,
        output,
        log_normalizer,
        upstream,
    ).compile()
    arguments = (
        query,
        key,
        value,
        output,
        log_normalizer,
        upstream,
    )
    dq_seconds = _time(
        dq,
        arguments,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    dkv_seconds = _time(
        dkv,
        arguments,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(
        f"N={args.seq_len} H={args.heads} D={args.head_dim} "
        f"dtype={args.dtype} causal={is_causal}: "
        f"dQ={dq_seconds * 1e3:.3f} ms, "
        f"dK/dV={dkv_seconds * 1e3:.3f} ms "
        f"(split={args.split_dkv})"
    )


if __name__ == "__main__":
    main()
