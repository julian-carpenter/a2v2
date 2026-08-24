"""Exercise real workflow startup, resume, and coordinated preflight over NCCL.

Launch with ``torchrun --standalone --nproc-per-node=2
tests/gpu/nccl_workflow_resume.py OUTPUT``. The fixture is intentionally tiny:
one update is checkpointed, one update is resumed, and one rank-divergent
preflight is rejected before model construction.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist

from a2v2.config import load_config
from a2v2.training import CheckpointError, load_checkpoint
import a2v2.workflows as workflows


ROOT = Path(__file__).parents[2]


def _prepare_manifest(root: Path, rank: int) -> Path:
    """Create four equal tiny recordings once, then synchronize their visibility."""

    manifests = root / "manifests"
    if rank == 0:
        audio = root / "wav"
        manifests.mkdir(parents=True, exist_ok=True)
        audio.mkdir(parents=True, exist_ok=True)
        rows: list[str] = []
        for index in range(4):
            name = f"sample_{index}.wav"
            sf.write(
                audio / name,
                np.linspace(-0.5, 0.5, 64, dtype=np.float32),
                8000,
                subtype="FLOAT",
            )
            rows.append(f"{name}\t64")
        (manifests / "pretrain.tsv").write_text(
            f"{audio}\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
    dist.barrier()
    return manifests


def main() -> int:
    """Run two bounded workflow updates and one pre-model failure on two GPUs."""

    if len(sys.argv) != 2:
        raise SystemExit("usage: nccl_workflow_resume.py OUTPUT")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise AssertionError(f"expected exactly two ranks, received {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    output_root = Path(sys.argv[1]).resolve()
    manifests = _prepare_manifest(output_root, rank)
    checkpoint_dir = output_root / "checkpoints"
    config = load_config(
        ROOT / "configs/cpu_smoke_pretraining.yaml",
        overrides=(
            f"task.data={manifests}",
            f"checkpoint.save_dir={checkpoint_dir}",
            "checkpoint.resume_policy=strict",
            "dataset.crop_strategy=stateless",
            "dataset.max_tokens=128",
            "distributed_training.distributed_world_size=2",
            "distributed_training.ddp_backend=nccl",
            "optimization.max_update=2",
        ),
    )

    first = workflows.run_training(
        config,
        device_name="cuda",
        resume_path=None,
        pretrained_checkpoint=None,
        stop_at_update=1,
    )
    first_checkpoint = load_checkpoint(first)
    if first_checkpoint["update"] != 1:
        raise AssertionError(
            f"first workflow segment stopped at {first_checkpoint['update']!r}"
        )
    resumed = workflows.run_training(
        config,
        device_name="cuda",
        resume_path=first,
        pretrained_checkpoint=None,
        stop_at_update=2,
    )
    resumed_checkpoint = load_checkpoint(resumed)
    if resumed_checkpoint["update"] != 2:
        raise AssertionError(
            f"resumed workflow stopped at {resumed_checkpoint['update']!r}"
        )

    requested_world_size = 1 if rank == 1 else 2
    divergent = replace(
        config,
        distributed=replace(
            config.distributed,
            requested_world_size=requested_world_size,
        ),
    )
    marker = output_root / f"model-rank-{rank}"
    original_make_model = workflows._make_model

    def forbidden_model(*args: object, **kwargs: object) -> object:
        """Record any rank that incorrectly crosses coordinated preflight."""

        marker.write_text("constructed\n", encoding="utf-8")
        raise AssertionError("model construction crossed NCCL preflight")

    workflows._make_model = forbidden_model  # type: ignore[assignment]
    error = ""
    try:
        workflows.run_training(
            divergent,
            device_name="cuda",
            resume_path=None,
            pretrained_checkpoint=None,
        )
    except CheckpointError as caught:
        error = str(caught)
    finally:
        workflows._make_model = original_make_model
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, error)
    if len(set(gathered)) != 1 or "failed on rank 1: ValueError" not in error:
        raise AssertionError(f"NCCL ranks did not receive one canonical error: {gathered!r}")
    dist.barrier()
    crossed = sorted(path.name for path in output_root.glob("model-rank-*"))
    if crossed:
        raise AssertionError(f"model construction markers exist: {crossed!r}")

    if rank == 0:
        print(json.dumps({
            "nccl_workflow_resume": {
                "world_size": world_size,
                "first_update": first_checkpoint["update"],
                "resumed_update": resumed_checkpoint["update"],
                "preflight_error": error,
                "model_construction_crossed": False,
            }
        }), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
