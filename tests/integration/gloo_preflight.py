"""Torchrun entry point for coordinated pre-model training preflight tests."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch.distributed as dist

from a2v2.config import load_config
import a2v2.workflows as workflows


def main() -> int:
    """Require every rank to receive one canonical pre-model failure."""

    first_data = Path(sys.argv[1])
    second_data = Path(sys.argv[2])
    output_directory = Path(sys.argv[3])
    mode = sys.argv[4]
    marker_directory = Path(sys.argv[5])
    first_pretrained = None if sys.argv[6] == "-" else Path(sys.argv[6])
    second_pretrained = None if sys.argv[7] == "-" else Path(sys.argv[7])
    rank = int(os.environ["RANK"])
    data = first_data if rank == 0 else second_data
    fine_tuning = mode.startswith("pretrained-")
    requested_world_size = 1 if mode == "requested-world-size" and rank == 1 else 2
    config = load_config(
        Path(__file__).parents[2]
        / (
            "configs/cpu_smoke_finetuning.yaml"
            if fine_tuning
            else "configs/cpu_smoke_pretraining.yaml"
        ),
        overrides=(
            f"task.data={data}",
            f"checkpoint.save_dir={output_directory}",
            "checkpoint.resume_policy=strict",
            "dataset.crop_strategy=stateless",
            f"distributed_training.distributed_world_size={requested_world_size}",
            "distributed_training.ddp_backend=gloo",
            "optimization.max_update=1",
        ),
    )

    def forbidden_model(*args: object, **kwargs: object) -> object:
        """Leave durable evidence if any rank crosses the preflight boundary."""

        (marker_directory / f"model-rank-{rank}").write_text(
            "constructed\n",
            encoding="utf-8",
        )
        raise AssertionError("model construction crossed distributed preflight")

    if fine_tuning:
        workflows.Animal2VecFineTuningModel.from_config = forbidden_model  # type: ignore[method-assign]
    else:
        workflows._make_model = forbidden_model  # type: ignore[assignment]
    pretrained = first_pretrained if rank == 0 else second_pretrained
    error = ""
    try:
        workflows._run_training(
            config,
            device_name="cpu",
            resume_path=None,
            pretrained_checkpoint=pretrained,
        )
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
    if not error:
        raise AssertionError("distributed preflight unexpectedly succeeded")

    gathered: list[object] = [None, None]
    dist.all_gather_object(gathered, error)
    if len(set(gathered)) != 1:
        raise AssertionError(f"ranks received different errors: {gathered!r}")
    if mode == "data-failure":
        expected = "distributed training preflight failed on rank 1: ManifestError"
    elif mode == "fingerprint-mismatch":
        expected = "distributed training preflight fingerprint mismatch"
    elif mode in {
        "pretrained-missing",
        "pretrained-corrupt",
        "pretrained-config",
        "pretrained-state",
    }:
        expected = "distributed training preflight failed on rank 1"
    elif mode == "pretrained-identity":
        expected = "distributed training preflight fingerprint mismatch"
    elif mode == "requested-world-size":
        expected = "distributed training preflight failed on rank 1: ValueError"
    else:
        raise AssertionError(f"unknown preflight test mode {mode!r}")
    if expected not in error:
        raise AssertionError(f"missing {expected!r} in {error!r}")

    dist.barrier()
    crossed = sorted(path.name for path in marker_directory.glob("model-rank-*"))
    if crossed:
        raise AssertionError(f"model construction markers exist: {crossed!r}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
