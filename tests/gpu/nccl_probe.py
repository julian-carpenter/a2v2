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

from a2v2.slurm import (
    DistributedEnvironment,
    RankTopology,
    gather_rank_topologies,
)
from a2v2.training import (
    FORMAT_VERSION,
    RANK_LOCAL_RNG_SCHEMA,
    capture_rng_state,
    gather_rank_rng_states,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
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
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    checkpoint_group = dist.new_group(backend="gloo")
    distributed = DistributedEnvironment.from_mapping(os.environ)
    properties = torch.cuda.get_device_properties(device)
    topology = RankTopology(
        hostname=os.uname().nodename,
        global_rank=rank,
        local_rank=local_rank,
        local_world_size=distributed.local_world_size,
        visible_cuda_devices=torch.cuda.device_count(),
        selected_cuda_device=local_rank,
        cuda_device_name=properties.name,
        cuda_device_uuid=str(getattr(properties, "uuid", "unavailable")),
    )

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
    ranked_rng = gather_rank_rng_states(
        captured,
        world_size=world_size,
        rank=rank,
        group=checkpoint_group,
    )
    ranked_topology = gather_rank_topologies(
        topology,
        world_size=world_size,
        rank=rank,
        group=checkpoint_group,
    )

    checkpoint_path = arguments.output_dir / "nccl-rng-checkpoint.pt"
    if rank == 0:
        if ranked_rng is None:
            raise RuntimeError("rank zero received no distributed RNG state")
        if ranked_topology is None:
            raise RuntimeError("rank zero received no distributed topology state")
        save_checkpoint(checkpoint_path, {
            "format_version": FORMAT_VERSION,
            "stage": "pretrain",
            "config": {"probe": "nccl"},
            "model": {"collective_sum": reduced.detach().clone()},
            "teacher": None,
            "optimizer": None,
            "scheduler": None,
            "scaler": None,
            "gradient_clipper": None,
            "weight_decay_scheduler": None,
            "topology": ranked_topology,
            "resume_compatibility": None,
            "update": 0,
            "epoch": 1,
            "batch_in_epoch": 0,
            "rng_state": ranked_rng,
            "sampler_state": None,
            "best_metric": None,
        })
    dist.barrier()

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    restore_rng_state(checkpoint["rng_state"])
    actual_random = _draw_random(device)

    topology_ok = (
        checkpoint["topology"]["schema"] == "a2v2.topology.v1"
        and [
            record["global_rank"] for record in checkpoint["topology"]["by_rank"]
        ] == list(range(world_size))
        and checkpoint["rng_state"]["schema"] == RANK_LOCAL_RNG_SCHEMA
    )
    local_ok = (
        float(reduced.item()) == expected_sum
        and float(broadcast.item()) == 97.0
        and gathered_ranks == list(range(world_size))
        and expected_random == actual_random
        and checkpoint["model"]["collective_sum"].device == device
        and topology_ok
    )
    all_ok = torch.tensor(int(local_ok), dtype=torch.int32, device=device)
    dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)

    report = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "backend": str(dist.get_backend()),
        "rng_gather_backend": str(dist.get_backend(checkpoint_group)),
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
        "rng_schema": checkpoint["rng_state"]["schema"],
        "topology_schema": checkpoint["topology"]["schema"],
        "topology_ok": topology_ok,
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
