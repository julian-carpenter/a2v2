#!/usr/bin/env python3
"""Fail fast when the paper-reproduction host or checkpoint is not launchable."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
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


def check_bitsandbytes() -> dict[str, object]:
    """Validate the pinned 8-bit optimizer package and its CUDA backend."""

    try:
        import bitsandbytes
        from bitsandbytes.cextension import lib
    except Exception as error:
        raise RuntimeError(
            f"could not import bitsandbytes and its native backend: {error}"
        ) from error

    version = str(getattr(bitsandbytes, "__version__", "unknown"))
    if re.fullmatch(r"0\.50(?:\.\d+)?(?:[-+].*)?", version) is None:
        raise RuntimeError(
            "modern benchmark requires bitsandbytes >=0.50,<0.51; "
            f"found {version}"
        )
    compiled_with_cuda = bool(getattr(lib, "compiled_with_cuda", False))
    if not compiled_with_cuda:
        raise RuntimeError(
            "bitsandbytes loaded without CUDA support; the AdamW8bit benchmark "
            "cannot run"
        )
    return {
        "version": version,
        "compiled_with_cuda": compiled_with_cuda,
        "native_library": str(getattr(lib, "_name", "unknown")),
    }


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
    reference_path: Path | None = None,
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
    except (TypeError, ValueError, OverflowError) as error:
        fail(f"update metadata is invalid: {error}")

    active_config = (
        checkpoint["config"].get("active")
        if isinstance(checkpoint.get("config"), Mapping)
        else None
    )
    if isinstance(active_config, Mapping):
        optimization_config = active_config.get("optimization")
        optimizer_config = active_config.get("optimizer")
        checkpoint_config = active_config.get("checkpoint")
        if (
            isinstance(optimization_config, Mapping)
            and optimization_config.get("gradient_clip_method") == "adagc"
        ):
            clipper_state = checkpoint.get("gradient_clipper")
            expected_clipper_keys = {
                "algorithm_version",
                "update",
                "parameter_names",
                "norm_emas",
            }
            if not isinstance(clipper_state, Mapping):
                fail("AdaGC checkpoint is missing clipper state")
            if set(clipper_state) != expected_clipper_keys:
                fail("AdaGC clipper state keys are malformed")
            if clipper_state.get("algorithm_version") != 1:
                fail("AdaGC clipper algorithm version is unsupported")
            if clipper_state.get("update") != update:
                fail("AdaGC clipper state does not match checkpoint update")
            parameter_names = clipper_state.get("parameter_names")
            norm_emas = clipper_state.get("norm_emas")
            if (
                not isinstance(parameter_names, list)
                or not all(isinstance(name, str) for name in parameter_names)
                or not isinstance(norm_emas, Mapping)
                or set(norm_emas) != set(parameter_names)
            ):
                fail("AdaGC clipper parameter state is malformed")
            for name in parameter_names:
                norm = norm_emas[name]
                if (
                    not isinstance(norm, torch.Tensor)
                    or norm.shape != torch.Size([])
                    or norm.dtype != torch.float32
                    or not (bool(torch.isfinite(norm)) or bool(torch.isposinf(norm)))
                    or (bool(torch.isfinite(norm)) and float(norm) < 0.0)
                ):
                    fail(f"AdaGC norm for {name!r} is malformed")
        if (
            isinstance(optimizer_config, Mapping)
            and optimizer_config.get("weight_decay_schedule") == "cosine"
        ):
            decay_state = checkpoint.get("weight_decay_scheduler")
            if not isinstance(decay_state, Mapping):
                fail("weight-decay scheduler is missing state")
            if set(decay_state) != {"last_update"}:
                fail("weight-decay scheduler state is malformed")
            if decay_state.get("last_update") != update:
                fail("weight-decay scheduler state does not match checkpoint update")
        if (
            isinstance(checkpoint_config, Mapping)
            and checkpoint_config.get("resume_policy") == "strict"
        ):
            fingerprint = checkpoint.get("resume_compatibility")
            if (
                not isinstance(fingerprint, Mapping)
                or fingerprint.get("training_data.schema")
                != "a2v2.training-data.v2"
            ):
                fail("strict resume checkpoint lacks training-data provenance")

    try:
        epoch = int(checkpoint["epoch"])
        size_bytes = resolved.stat().st_size
    except (TypeError, ValueError, OverflowError, OSError) as error:
        fail(f"metadata is invalid: {error}")
    report = {
        "path": str(resolved),
        "stage": stage,
        "update": update,
        "epoch": epoch,
        "rng_world_size": saved_world_size,
        "size_bytes": size_bytes,
    }
    if reference_path is not None:
        reference_report = check_training_checkpoint(
            reference_path,
            expected_world_size,
            expected_stage,
        )
        try:
            reference_checkpoint = load_checkpoint(
                reference_path.resolve(),
                map_location="cpu",
            )
        except Exception as error:
            fail(f"cannot load reference checkpoint {reference_path}: {error}")
        selected_fingerprint = checkpoint.get("resume_compatibility")
        reference_fingerprint = reference_checkpoint.get("resume_compatibility")
        if (
            not isinstance(selected_fingerprint, Mapping)
            or not isinstance(reference_fingerprint, Mapping)
            or dict(selected_fingerprint) != dict(reference_fingerprint)
        ):
            fail(
                "resume fingerprint differs from reference checkpoint "
                f"{reference_path.resolve()}"
            )
        report["reference_checkpoint"] = reference_report
    return report


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
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--expected-stage", choices=("pretrain", "finetune"))
    parser.add_argument(
        "--require-bitsandbytes",
        action="store_true",
        help="require the pinned bitsandbytes release and a loaded CUDA backend",
    )
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
                arguments.reference_checkpoint,
            )
        elif arguments.reference_checkpoint is not None:
            raise RuntimeError("--reference-checkpoint requires --checkpoint")
        if arguments.require_bitsandbytes:
            report["bitsandbytes"] = check_bitsandbytes()
    except Exception as error:
        print(json.dumps({"pass": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
