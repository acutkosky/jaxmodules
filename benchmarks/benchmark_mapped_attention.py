"""Benchmark mapped attention against JAX XLA and cuDNN attention.

The public command is a small driver. Every benchmark case is executed in a
fresh worker process so that process-wide device allocator peaks are comparable
between implementations.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

IMPLEMENTATIONS = ("mapped", "xla", "cudnn")
MODES = ("forward", "backward")
DTYPES = ("float32", "bfloat16", "float16")


@dataclass(frozen=True)
class Case:
    implementation: str
    mode: str
    seq_len: int
    block_size: int
    kv_block_size: int
    batch_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    causal: bool
    warmup: int
    iterations: int
    seed: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, action="append")
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--seq-len", type=_positive_int, nargs="+", default=[512])
    parser.add_argument("--block-size", type=_positive_int, nargs="+", default=[64])
    parser.add_argument("--kv-block-size", type=_positive_int, nargs="+")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--query-heads", type=_positive_int, default=4)
    parser.add_argument("--kv-heads", type=_positive_int, default=4)
    parser.add_argument("--head-dim", type=_positive_int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    return parser


def _memory_stats(device: Any) -> dict[str, int]:
    stats = device.memory_stats()
    if stats is None:
        return {}
    return {
        key: int(value)
        for key, value in stats.items()
        if isinstance(value, (int, float))
    }


def _compiled_memory_stats(compiled: Any) -> dict[str, int]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {}
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {field: int(getattr(stats, field)) for field in fields}


def _make_inputs(case: Case) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    if case.query_heads % case.kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")

    dtype = getattr(jnp, case.dtype)
    keys = jax.random.split(jax.random.key(case.seed), 3)
    query = jax.random.normal(
        keys[0],
        (case.batch_size, case.seq_len, case.query_heads, case.head_dim),
        dtype=dtype,
    )
    key = jax.random.normal(
        keys[1],
        (case.batch_size, case.seq_len, case.kv_heads, case.head_dim),
        dtype=dtype,
    )
    value = jax.random.normal(
        keys[2],
        (case.batch_size, case.seq_len, case.kv_heads, case.head_dim),
        dtype=dtype,
    )
    jax.block_until_ready((query, key, value))
    return query, key, value


def _make_function(case: Case) -> Any:
    import jax
    import jax.numpy as jnp

    if case.implementation == "mapped":
        from jaxmodules.attention import masked_attention_via_map

        def attention(query: Any, key: Any, value: Any) -> Any:
            return masked_attention_via_map(
                query,
                key,
                value,
                block_size=case.block_size,
                kv_block_size=case.kv_block_size,
                is_causal=case.causal,
            )

    else:
        implementation = cast(
            "Literal['xla', 'cudnn']",
            case.implementation,
        )

        def attention(query: Any, key: Any, value: Any) -> Any:
            return jax.nn.dot_product_attention(
                query,
                key,
                value,
                is_causal=case.causal,
                implementation=implementation,
            )

    if case.mode == "forward":
        return jax.jit(attention)

    def loss(query: Any, key: Any, value: Any) -> Any:
        output = attention(query, key, value)
        # Accumulate the benchmark loss in FP32 so the reduction itself does not
        # overflow for low-precision inputs.
        return jnp.mean(output.astype(jnp.float32) ** 2)

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))


def _run_worker(case: Case) -> dict[str, Any]:
    # Disable the default large GPU preallocation. Otherwise allocator peaks
    # describe the preallocated pool rather than the operation being measured.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax

    result: dict[str, Any] = {"case": asdict(case), "status": "ok"}
    result["jax_version"] = jax.__version__
    result["backend"] = jax.default_backend()
    result["device"] = str(jax.devices()[0])

    try:
        query, key, value = _make_inputs(case)
        device = query.device
        input_stats = _memory_stats(device)

        function = _make_function(case)
        compile_start = time.perf_counter()
        compiled = function.lower(query, key, value).compile()
        result["compile_seconds"] = time.perf_counter() - compile_start
        result["compiled_memory"] = _compiled_memory_stats(compiled)
        result["memory_after_compile"] = _memory_stats(device)

        for _ in range(case.warmup):
            jax.block_until_ready(compiled(query, key, value))

        timings = []
        final_output = None
        for _ in range(case.iterations):
            start = time.perf_counter()
            final_output = compiled(query, key, value)
            jax.block_until_ready(final_output)
            timings.append(time.perf_counter() - start)

        final_stats = _memory_stats(device)
        median_seconds = statistics.median(timings)
        tokens = case.batch_size * case.seq_len

        result.update(
            {
                "latency_median_seconds": median_seconds,
                "latency_min_seconds": min(timings),
                "latency_mean_seconds": statistics.mean(timings),
                "tokens_per_second": tokens / median_seconds,
                "input_memory": input_stats,
                "final_memory": final_stats,
            }
        )
        if input_stats and final_stats:
            baseline = input_stats.get("bytes_in_use", 0)
            peak = final_stats.get("peak_bytes_in_use", 0)
            result["peak_bytes_above_input_baseline"] = max(0, peak - baseline)
    except Exception as error:  # noqa: BLE001 - benchmark records unsupported cases.
        result["status"] = "error"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)

    return result


def _worker_command(case: Case) -> list[str]:
    payload = json.dumps(asdict(case), separators=(",", ":"))
    return [sys.executable, str(Path(__file__).resolve()), "--worker-case", payload]


def _run_isolated(case: Case) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S603 - command is constructed locally.
        _worker_command(case),
        check=False,
        capture_output=True,
        text=True,
        env=os.environ | {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
    if completed.returncode:
        return {
            "case": asdict(case),
            "status": "error",
            "error_type": "WorkerProcessError",
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "case": asdict(case),
            "status": "error",
            "error_type": "WorkerOutputError",
            "error": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f}{unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def _print_results(results: list[dict[str, Any]]) -> None:
    heading = (
        "seq | qblk | kvblk | mode | implementation | median | tokens/s | "
        "device peak* | compiler temp | status"
    )
    print(heading)
    print("-" * len(heading))
    for result in results:
        case = result["case"]
        if result["status"] == "ok":
            latency = f"{result['latency_median_seconds'] * 1_000:.3f}ms"
            throughput = f"{result['tokens_per_second']:,.0f}"
            peak = _format_bytes(result.get("peak_bytes_above_input_baseline"))
            temporary = _format_bytes(
                result.get("compiled_memory", {}).get("temp_size_in_bytes")
            )
            status = "ok"
        else:
            latency = throughput = peak = temporary = "-"
            status = f"{result.get('error_type')}: {result.get('error')}"
        print(
            f"{case['seq_len']:>3} | {case['block_size']:>4} | "
            f"{case['kv_block_size']:>5} | {case['mode']:<8} | "
            f"{case['implementation']:<14} | {latency:>9} | "
            f"{throughput:>9} | {peak:>12} | {temporary:>13} | {status}"
        )
    print("\n* Process peak allocation above bytes in use after creating inputs.")


def main() -> int:
    args = _parser().parse_args()
    if args.worker_case is not None:
        case = Case(**json.loads(args.worker_case))
        print(json.dumps(_run_worker(case), separators=(",", ":")))
        return 0

    implementations = args.implementation or list(IMPLEMENTATIONS)
    modes = args.mode or list(MODES)
    tile_sizes = (
        [(block_size, block_size) for block_size in args.block_size]
        if args.kv_block_size is None
        else [
            (block_size, kv_block_size)
            for block_size in args.block_size
            for kv_block_size in args.kv_block_size
        ]
    )
    cases = [
        Case(
            implementation=implementation,
            mode=mode,
            seq_len=seq_len,
            block_size=block_size,
            kv_block_size=kv_block_size,
            batch_size=args.batch_size,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=args.dtype,
            causal=args.causal,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        )
        for seq_len in args.seq_len
        for block_size, kv_block_size in tile_sizes
        for mode in modes
        for implementation in implementations
    ]

    results = []
    for case in cases:
        print(
            f"Running {case.implementation} {case.mode} "
            f"(sequence length {case.seq_len})...",
            flush=True,
        )
        results.append(_run_isolated(case))

    print()
    _print_results(results)
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {args.json_output}")
    return int(not all(result["status"] == "ok" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
