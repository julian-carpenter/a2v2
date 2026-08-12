#!/usr/bin/env python3
"""Fail fast when the paper-reproduction host or checkpoint is not launchable."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

_RUNTIME_IMPORT_ERROR: Exception | None = None
try:
    import h5py
    import numpy as np
    import soundfile
    import tensorboard
    import torch
    import yaml

    from a2v2.training import load_checkpoint
except Exception as error:  # pragma: no cover - exercised by a broken deployment
    _RUNTIME_IMPORT_ERROR = error


GIB = 1024**3


def check_cuda_devices(
    expected_count: int,
    min_total_bytes: int,
    min_free_bytes: int,
) -> list[dict[str, object]]:
    """Validate visible A100 count and per-device total and free memory."""

    visible_count = torch.cuda.device_count()
    if visible_count != expected_count:
        raise RuntimeError(
            f"expected {expected_count} visible CUDA devices, found {visible_count}"
        )
    devices: list[dict[str, object]] = []
    for index in range(visible_count):
        properties = torch.cuda.get_device_properties(index)
        name = str(properties.name)
        total_bytes = int(properties.total_memory)
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_bytes = int(free_bytes)
        if "A100" not in name:
            raise RuntimeError(
                f"CUDA device {index} is {name!r}; expected an NVIDIA A100"
            )
        if total_bytes < min_total_bytes:
            raise RuntimeError(
                f"CUDA device {index} has {total_bytes / GIB:.2f} GiB total memory; "
                f"requires at least {min_total_bytes / GIB:.2f} GiB"
            )
        if free_bytes < min_free_bytes:
            raise RuntimeError(
                f"CUDA device {index} has {free_bytes / GIB:.2f} GiB free memory; "
                f"requires at least {min_free_bytes / GIB:.2f} GiB"
            )
        devices.append({
            "index": index,
            "name": name,
            "total_bytes": total_bytes,
            "free_bytes": free_bytes,
        })
    return devices


def check_output_space(path: Path, min_free_bytes: int) -> dict[str, object]:
    """Validate free space for large checkpoints and their atomic temporary files."""

    resolved = path.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"output directory does not exist: {resolved}")
    usage = shutil.disk_usage(resolved)
    if usage.free < min_free_bytes:
        raise RuntimeError(
            f"output filesystem for {resolved} has {usage.free / GIB:.2f} GiB free; "
            f"requires at least {min_free_bytes / GIB:.2f} GiB"
        )
    return {
        "path": str(resolved),
        "total_bytes": int(usage.total),
        "free_bytes": int(usage.free),
    }


def check_training_checkpoint(
    path: Path,
    expected_world_size: int,
    expected_stage: str | None = None,
) -> dict[str, object]:
    """Validate that a native checkpoint can resume eight-rank optimization."""

    checkpoint = load_checkpoint(path, map_location="cpu")
    stage = str(checkpoint["stage"])
    if expected_stage is not None and stage != expected_stage:
        raise RuntimeError(
            f"checkpoint stage is {stage}; expected {expected_stage}: {path.resolve()}"
        )
    if checkpoint["optimizer"] is None:
        raise RuntimeError("checkpoint optimizer state is missing; cannot resume training")
    if checkpoint["scheduler"] is None:
        raise RuntimeError("checkpoint scheduler state is missing; cannot resume training")
    rng_state = checkpoint["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise RuntimeError("checkpoint RNG state is not a mapping")
    saved_world_size = int(rng_state.get("world_size", 0))
    ranked = rng_state.get("by_rank")
    if saved_world_size != expected_world_size:
        raise RuntimeError(
            f"checkpoint RNG world size is {saved_world_size}; expected {expected_world_size}"
        )
    if not isinstance(ranked, (list, tuple)) or len(ranked) != expected_world_size:
        raise RuntimeError(
            "checkpoint does not contain one RNG state for every expected rank"
        )
    return {
        "path": str(path.resolve()),
        "stage": stage,
        "update": int(checkpoint["update"]),
        "epoch": int(checkpoint["epoch"]),
        "rng_world_size": saved_world_size,
        "size_bytes": path.stat().st_size,
    }


def _runtime_report() -> dict[str, object]:
    """Return versions for every native runtime dependency imported above."""

    if _RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"could not import reproduction runtime dependencies: {_RUNTIME_IMPORT_ERROR}"
        ) from _RUNTIME_IMPORT_ERROR
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "soundfile": soundfile.__version__,
        "h5py": h5py.__version__,
        "pyyaml": yaml.__version__,
        "tensorboard": tensorboard.__version__,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the deployable preflight command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--min-total-gib", type=float, default=39.0)
    parser.add_argument("--min-free-gib", type=float, default=38.0)
    parser.add_argument("--min-disk-gib", type=float, default=64.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-stage", choices=("pretrain", "finetune"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run all requested checks and emit exactly one JSON report."""

    arguments = build_parser().parse_args(argv)
    try:
        report: dict[str, object] = {
            "pass": True,
            "runtime": _runtime_report(),
            "cuda_devices": check_cuda_devices(
                arguments.expected_gpus,
                int(arguments.min_total_gib * GIB),
                int(arguments.min_free_gib * GIB),
            ),
            "output_space": check_output_space(
                arguments.output_dir,
                int(arguments.min_disk_gib * GIB),
            ),
        }
        if arguments.checkpoint is not None:
            report["checkpoint"] = check_training_checkpoint(
                arguments.checkpoint,
                arguments.expected_gpus,
                arguments.expected_stage,
            )
    except Exception as error:
        print(json.dumps({"pass": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
