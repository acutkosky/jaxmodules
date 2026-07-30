"""Benchmark Mosaic and mapped attention against JAX XLA and cuDNN attention.

The public command is a small driver. Every benchmark case is executed in a
fresh worker process so that process-wide device allocator peaks are comparable
between implementations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

IMPLEMENTATIONS = ("mosaic", "mosaic-warp", "mapped", "xla", "cudnn")
MODES = ("forward", "backward")
DTYPES = ("float32", "bfloat16", "float16")
MASKS = ("causal", "unmasked", "general")
CUDA_VERSION_FUNCTIONS = (
    "cuda_driver_get_version",
    "cuda_runtime_build_version",
    "cuda_runtime_get_version",
    "cudnn_build_version",
    "cudnn_get_version",
    "cublas_build_version",
    "cublas_get_version",
    "cufft_build_version",
    "cufft_get_version",
    "cupti_build_version",
    "cupti_get_version",
    "cusolver_build_version",
    "cusolver_get_version",
    "cusparse_build_version",
    "cusparse_get_version",
)
REPRODUCIBILITY_ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_PLATFORMS",
    "LD_LIBRARY_PATH",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


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
    mask: str
    mapped_backward_strategy: str
    warmup: int
    iterations: int
    seed: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, action="append")
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--seq-len", type=_positive_int, nargs="+", default=[1024])
    parser.add_argument("--block-size", type=_positive_int, nargs="+", default=[64])
    parser.add_argument("--kv-block-size", type=_positive_int, nargs="+")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--query-heads", type=_positive_int, default=4)
    parser.add_argument("--kv-heads", type=_positive_int, default=4)
    parser.add_argument("--head-dim", type=_positive_int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--mask", choices=MASKS, default="causal")
    parser.add_argument(
        "--mapped-backward-strategy",
        choices=("auto", "minimal"),
        default="auto",
    )
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--case-timeout-seconds", type=_positive_float, default=300)
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


def _error_metadata(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _relevant_distribution_versions() -> dict[str, str]:
    """Return package versions that can affect JAX GPU code generation."""
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name is None:
            continue
        normalized_name = name.lower()
        if normalized_name in {"jax", "jaxlib", "triton"} or normalized_name.startswith(
            ("jax-cuda", "nvidia-")
        ):
            versions[name] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def _jax_cuda_versions() -> dict[str, Any]:
    """Read both compile-time and runtime CUDA library versions from JAX."""
    try:
        from jax._src.lib import cuda_versions  # noqa: PLC0415
    except (AttributeError, ImportError) as error:
        return {"available": False, **_error_metadata(error)}

    values = {}
    errors = {}
    for function_name in CUDA_VERSION_FUNCTIONS:
        function = getattr(cuda_versions, function_name, None)
        if function is None:
            continue
        try:
            values[function_name] = int(function())
        except Exception as error:  # noqa: BLE001 - metadata must be best effort.
            errors[function_name] = _error_metadata(error)

    result: dict[str, Any] = {"available": bool(values), "versions": values}
    if errors:
        result["errors"] = errors
    return result


def _nvidia_smi_metadata() -> dict[str, Any]:
    """Query the host driver and physical GPUs without initializing CUDA."""
    fields = (
        "driver_version",
        "name",
        "compute_capability",
        "pci_bus_id",
        "memory_total_mib",
    )
    query_fields = (
        "driver_version",
        "name",
        "compute_cap",
        "pci.bus_id",
        "memory.total",
    )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed diagnostic command.
            [  # noqa: S607 - nvidia-smi is intentionally resolved through PATH.
                "nvidia-smi",
                f"--query-gpu={','.join(query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"available": False, **_error_metadata(error)}
    rows = [
        [value.strip() for value in row]
        for row in csv.reader(completed.stdout.splitlines())
        if row
    ]
    gpus = [
        dict(zip(fields, row, strict=True)) for row in rows if len(row) == len(fields)
    ]
    return {"available": True, "gpus": gpus}


def _environment_metadata(jax: ModuleType, device: object) -> dict[str, Any]:
    """Capture enough of the software and hardware stack to reproduce a run."""
    try:
        os_release = platform.freedesktop_os_release()
    except OSError as error:
        os_metadata: dict[str, Any] = _error_metadata(error)
    else:
        os_metadata = {
            key.lower(): os_release[key]
            for key in ("ID", "VERSION_ID", "PRETTY_NAME")
            if key in os_release
        }

    client = getattr(device, "client", None)
    return {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "operating_system": os_metadata,
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "packages": _relevant_distribution_versions(),
        "jax": {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "device": {
                "description": str(device),
                "id": getattr(device, "id", None),
                "kind": getattr(device, "device_kind", None),
                "platform": getattr(device, "platform", None),
                "local_hardware_id": getattr(device, "local_hardware_id", None),
            },
            "platform_version": getattr(client, "platform_version", None),
            "cuda_versions": _jax_cuda_versions(),
        },
        "nvidia_smi": _nvidia_smi_metadata(),
        "environment_variables": {
            name: os.environ.get(name) for name in REPRODUCIBILITY_ENVIRONMENT_VARIABLES
        },
    }


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


def _general_mask(batch: Any, head: Any, query: Any, key: Any) -> Any:
    """Representative noncausal, batch/head-dependent coordinate mask."""
    radius = 256 + 16 * (head % 2) + 8 * batch
    return (abs(query - key) <= radius) & ((query + 2 * key + head) % 5 != 1)


def _make_function(case: Case) -> Any:
    import jax
    import jax.numpy as jnp

    from jaxmodules.attention import (
        _masked_attention_via_map,
        _masked_attention_via_mosaic,
        _unmasked,
        default_kernel,
    )
    from jaxmodules._mosaic_attention import (
        _mosaic_attention_forward_warp_specialized_unmasked,
    )

    is_causal = case.mask == "causal"
    mask_fn = {
        "causal": _unmasked,
        "unmasked": _unmasked,
        "general": _general_mask,
    }[case.mask]

    if case.implementation == "mosaic-warp":
        if case.mask != "unmasked":
            raise ValueError("mosaic-warp currently supports only unmasked attention")
        if case.mode != "forward":
            raise ValueError("mosaic-warp currently implements only the forward pass")

        def attention(query: Any, key: Any, value: Any) -> Any:
            output, _ = _mosaic_attention_forward_warp_specialized_unmasked(
                query,
                key,
                value,
            )
            return output

    elif case.implementation in ("mosaic", "mapped"):
        implementation = case.implementation

        def attention(query: Any, key: Any, value: Any) -> Any:
            function = (
                _masked_attention_via_mosaic
                if implementation == "mosaic"
                else _masked_attention_via_map
            )
            kwargs = {
                "mask_fn": mask_fn,
                "block_size": case.block_size,
                "kv_block_size": case.kv_block_size,
                "window_size": None,
                "is_causal": is_causal,
                "backward_strategy": case.mapped_backward_strategy,
            }
            if implementation == "mapped":
                kwargs["kernel_fn"] = default_kernel
            return function(
                query,
                key,
                value,
                **kwargs,
            )

    else:
        implementation = cast(
            "Literal['xla', 'cudnn']",
            case.implementation,
        )

        def attention(query: Any, key: Any, value: Any) -> Any:
            dense_mask = None
            if case.mask == "general":
                dense_mask = _general_mask(
                    jnp.arange(case.batch_size)[:, None, None, None],
                    jnp.arange(case.query_heads)[None, :, None, None],
                    jnp.arange(case.seq_len)[None, None, :, None],
                    jnp.arange(case.seq_len)[None, None, None, :],
                )
            return jax.nn.dot_product_attention(
                query,
                key,
                value,
                mask=dense_mask,
                is_causal=is_causal,
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
    device = jax.devices()[0]
    result["device"] = str(device)
    try:
        result["environment"] = _environment_metadata(jax, device)
    except Exception as error:  # noqa: BLE001 - never fail a run on metadata.
        result["environment"] = {
            "schema_version": 1,
            "error": _error_metadata(error),
        }

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


def _run_isolated(case: Case, timeout_seconds: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(  # noqa: S603 - command is constructed locally.
            _worker_command(case),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ | {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        )
    except subprocess.TimeoutExpired:
        return {
            "case": asdict(case),
            "status": "error",
            "error_type": "WorkerTimeout",
            "error": f"case exceeded {timeout_seconds:g} seconds",
        }
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
        "seq | mask     | qblk | kvblk | strategy | mode | implementation | median | "
        "tokens/s | device peak* | compiler temp | status"
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
            f"{case['seq_len']:>3} | {case['mask']:<8} | "
            f"{case['block_size']:>4} | "
            f"{case['kv_block_size']:>5} | "
            f"{case['mapped_backward_strategy']:<8} | {case['mode']:<8} | "
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
            mask=args.mask,
            mapped_backward_strategy=args.mapped_backward_strategy,
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
        results.append(_run_isolated(case, args.case_timeout_seconds))

    print()
    _print_results(results)
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {args.json_output}")
    return int(not all(result["status"] == "ok" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
