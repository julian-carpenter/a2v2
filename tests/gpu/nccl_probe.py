"""Verify NCCL collectives, CUDA checkpoint placement, and rank RNG restore.

Launch with ``torchrun --standalone --nproc-per-node=N tests/gpu/nccl_probe.py``.
Each process binds to its ``LOCAL_RANK`` GPU. The probe checks all-reduce,
broadcast, all-gather, rank-specific Python/NumPy/PyTorch random replay, and
loading a checkpoint tensor onto the local CUDA device.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from a2v2.training import (
    capture_rng_state,
    deserialize_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    serialize_rng_state,
)


def _draw_random(device: torch.device) -> dict[str, object]:
    """Draw one comparable sample from each rank-local random generator."""

    return {
        "python": random.random(),
        "numpy": np.random.random(4).tolist(),
        "torch_cpu": torch.rand(4).tolist(),
        "torch_cuda": torch.rand(4, device=device).cpu().tolist(),
    }


def main() -> int:
    """Run the collective and checkpoint probe and write one report per rank."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("NCCL probe requires CUDA")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    random.seed(10_000 + rank)
    np.random.seed(20_000 + rank)
    torch.manual_seed(30_000 + rank)
    torch.cuda.manual_seed_all(40_000 + rank)

    reduced = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    expected_sum = world_size * (world_size + 1) / 2

    broadcast = torch.tensor(97.0 if rank == 0 else -1.0, device=device)
    dist.broadcast(broadcast, src=0)

    gathered_tensors = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, torch.tensor([rank], dtype=torch.int64, device=device))
    gathered_ranks = [int(value.item()) for value in gathered_tensors]

    captured = capture_rng_state()
    expected_random = _draw_random(device)
    encoded = serialize_rng_state(captured)
    gathered_rng: list[object] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(encoded, gathered_rng, dst=0)

    checkpoint_path = arguments.output_dir / "nccl-rng-checkpoint.pt"
    if rank == 0:
        assert gathered_rng is not None
        ranked_rng = [
            deserialize_rng_state(item)
            for item in gathered_rng
            if isinstance(item, bytes)
        ]
        if len(ranked_rng) != world_size:
            raise RuntimeError("did not gather one RNG payload per rank")
        save_checkpoint(checkpoint_path, {
            "format_version": 1,
            "stage": "pretrain",
            "config": {"probe": "nccl"},
            "model": {"collective_sum": reduced.detach().clone()},
            "teacher": None,
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "update": 0,
            "epoch": 1,
            "batch_in_epoch": 0,
            "rng_state": {"world_size": world_size, "by_rank": ranked_rng},
            "sampler_state": None,
            "best_metric": None,
        })
    dist.barrier()

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    restore_rng_state(checkpoint["rng_state"])
    actual_random = _draw_random(device)

    properties = torch.cuda.get_device_properties(device)
    local_ok = (
        float(reduced.item()) == expected_sum
        and float(broadcast.item()) == 97.0
        and gathered_ranks == list(range(world_size))
        and expected_random == actual_random
        and checkpoint["model"]["collective_sum"].device == device
    )
    all_ok = torch.tensor(int(local_ok), dtype=torch.int32, device=device)
    dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)

    report = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "backend": str(dist.get_backend()),
        "device": str(device),
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "unavailable")),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "all_reduce_sum": float(reduced.item()),
        "expected_sum": expected_sum,
        "broadcast": float(broadcast.item()),
        "all_gather_ranks": gathered_ranks,
        "rank_rng_restore_exact": expected_random == actual_random,
        "checkpoint_device": str(checkpoint["model"]["collective_sum"].device),
        "pass": bool(all_ok.item()),
    }
    (arguments.output_dir / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if not report["pass"]:
        raise RuntimeError(f"NCCL probe failed on rank {rank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
