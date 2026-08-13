#!/usr/bin/env python3
"""Fail fast when the paper-reproduction host or checkpoint is not launchable."""

from __future__ import annotations

import argparse
import json
import platform
import random
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

    resolved = path.resolve()

    def fail(reason: str) -> None:
        """Raise one actionable failure that always identifies the selected file."""

        raise RuntimeError(f"checkpoint {resolved}: {reason}")

    try:
        checkpoint = load_checkpoint(resolved, map_location="cpu")
    except Exception as error:
        fail(f"cannot load or validate native checkpoint: {error}")
    stage = str(checkpoint["stage"])
    if expected_stage is not None and stage != expected_stage:
        fail(f"checkpoint stage is {stage}; expected {expected_stage}")
    if not isinstance(checkpoint["optimizer"], Mapping):
        fail("optimizer state is not a mapping; cannot resume training")
    if not isinstance(checkpoint["scheduler"], Mapping):
        fail("scheduler state is not a mapping; cannot resume training")

    stored_config = checkpoint["config"]
    active_config = (
        stored_config.get("active") if isinstance(stored_config, Mapping) else None
    )
    common_config = (
        active_config.get("common") if isinstance(active_config, Mapping) else None
    )
    amp_enabled = (
        isinstance(common_config, Mapping) and common_config.get("fp16") is True
    )
    if amp_enabled and not isinstance(checkpoint["scaler"], Mapping):
        fail("AMP scaler state is not a mapping; cannot resume fp16 training")

    rng_state = checkpoint["rng_state"]
    if not isinstance(rng_state, Mapping):
        fail("RNG state is not a mapping")
    try:
        saved_world_size = int(rng_state.get("world_size", 0))
    except (TypeError, ValueError, OverflowError) as error:
        fail(f"RNG world size is invalid: {error}")
    ranked = rng_state.get("by_rank")
    if saved_world_size != expected_world_size:
        fail(
            f"RNG world size is {saved_world_size}; expected {expected_world_size}"
        )
    if not isinstance(ranked, (list, tuple)) or len(ranked) != expected_world_size:
        fail("does not contain exactly one RNG state for every expected rank")
    for rank, rank_state in enumerate(ranked):
        if not isinstance(rank_state, Mapping):
            fail(f"RNG state for rank {rank} is not a mapping")
        if "python" not in rank_state:
            fail(f"RNG state for rank {rank} is missing Python state")
        try:
            random.Random().setstate(rank_state["python"])
        except Exception as error:
            fail(
                f"Python RNG state for rank {rank} cannot be restored: {error}"
            )
        numpy_state = rank_state.get("numpy")
        if not isinstance(numpy_state, Mapping):
            fail(f"NumPy RNG state for rank {rank} is not a mapping")
        missing_numpy = sorted({
            "algorithm",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        } - set(numpy_state))
        if missing_numpy:
            fail(
                f"NumPy RNG state for rank {rank} is missing fields: "
                f"{missing_numpy}"
            )
        if not isinstance(numpy_state["keys"], torch.Tensor):
            fail(f"NumPy RNG keys for rank {rank} are not a tensor")
        try:
            np.random.RandomState().set_state((
                str(numpy_state["algorithm"]),
                numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            ))
        except Exception as error:
            fail(
                f"NumPy RNG state for rank {rank} cannot be restored: {error}"
            )
        torch_state = rank_state.get("torch")
        if not isinstance(torch_state, torch.Tensor):
            fail(f"torch RNG state for rank {rank} is not a tensor")
        try:
            torch.Generator(device="cpu").set_state(torch_state.cpu())
        except Exception as error:
            fail(
                f"CPU torch RNG state for rank {rank} cannot be restored: {error}"
            )
        cuda_state = rank_state.get("cuda")
        if cuda_state is None:
            if amp_enabled:
                fail(f"RNG state for rank {rank} is missing CUDA state")
        elif (
            not isinstance(cuda_state, (list, tuple))
            or not cuda_state
            or not all(isinstance(item, torch.Tensor) for item in cuda_state)
        ):
            fail(f"CUDA RNG state for rank {rank} is invalid")
        elif amp_enabled and len(cuda_state) != expected_world_size:
            fail(
                f"CUDA RNG state for rank {rank} has {len(cuda_state)} entries; "
                f"expected {expected_world_size}"
            )

    sampler_state = checkpoint["sampler_state"]
    if not isinstance(sampler_state, Mapping):
        fail("sampler state is not a mapping")
    for cursor in ("epoch", "next_batch"):
        value = sampler_state.get(cursor)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail(f"sampler state {cursor} is not a non-negative integer")

    try:
        update = int(checkpoint["update"])
        epoch = int(checkpoint["epoch"])
        size_bytes = resolved.stat().st_size
    except (TypeError, ValueError, OverflowError, OSError) as error:
        fail(f"metadata is invalid: {error}")
    return {
        "path": str(resolved),
        "stage": stage,
        "update": update,
        "epoch": epoch,
        "rng_world_size": saved_world_size,
        "size_bytes": size_bytes,
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
