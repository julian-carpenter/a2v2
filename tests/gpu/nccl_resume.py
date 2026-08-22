"""Compare uninterrupted and resumed DDP pretraining state bit for bit.

Launch with ``torchrun --standalone --nproc-per-node=N tests/gpu/nccl_resume.py``.
The first branch runs a second update without interruption. The second branch
rebuilds models and engines, restores the shared checkpoint plus rank-local RNG
state, and runs the same update. Each rank compares model, teacher, optimizer,
scheduler, scaler, counters, sampler metadata, results, and RNG state.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from a2v2.slurm import (
    DistributedEnvironment,
    RankTopology,
    gather_rank_topologies,
)
from a2v2.config import config_to_dict, load_config
from a2v2.model import Animal2VecPretrainingModel
from a2v2.training import (
    RANK_LOCAL_RNG_SCHEMA,
    capture_rng_state,
    gather_rank_rng_states,
    load_checkpoint,
    save_checkpoint,
)
from a2v2.training import TrainingEngine
from a2v2.training import CosineUpdateScheduler, build_gradient_clipper, build_optimizer


ROOT = Path(__file__).parents[2]


def _clone_tree(value: Any) -> Any:
    """Detach tensors to CPU and recursively copy an experiment state tree."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return deepcopy(value)


def _first_difference(actual: Any, expected: Any, path: str = "root") -> dict[str, object] | None:
    """Locate the first exact mismatch and report useful tensor diagnostics."""

    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return {"path": path, "reason": "type", "actual": type(actual).__name__, "expected": "Tensor"}
        actual_cpu = actual.detach().cpu()
        if actual_cpu.shape != expected.shape or actual_cpu.dtype != expected.dtype:
            return {
                "path": path,
                "reason": "metadata",
                "actual_shape": list(actual_cpu.shape),
                "expected_shape": list(expected.shape),
                "actual_dtype": str(actual_cpu.dtype),
                "expected_dtype": str(expected.dtype),
            }
        if torch.equal(actual_cpu, expected):
            return None
        flat_actual = actual_cpu.reshape(-1)
        flat_expected = expected.reshape(-1)
        index = int(torch.nonzero(flat_actual != flat_expected, as_tuple=False)[0])
        report: dict[str, object] = {
            "path": path,
            "reason": "tensor_value",
            "first_flat_index": index,
            "actual": flat_actual[index].item(),
            "expected": flat_expected[index].item(),
        }
        if actual_cpu.is_floating_point() or actual_cpu.is_complex():
            difference = (actual_cpu - expected).abs().float()
            report["max_abs"] = float(difference.max())
            report["mean_abs"] = float(difference.mean())
        return report
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return {"path": path, "reason": "type", "actual": type(actual).__name__, "expected": "dict"}
        if set(actual) != set(expected):
            return {
                "path": path,
                "reason": "keys",
                "actual_only": sorted(set(actual) - set(expected)),
                "expected_only": sorted(set(expected) - set(actual)),
            }
        for key in expected:
            difference = _first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            return {"path": path, "reason": "sequence_metadata"}
        for index, item in enumerate(expected):
            difference = _first_difference(actual[index], item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if actual != expected:
        return {"path": path, "reason": "value", "actual": actual, "expected": expected}
    return None


def _digest(value: Any) -> str:
    """Hash types, tensor metadata, tensor bytes, keys, and sequence order."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        """Feed one recursively encoded state item into the digest."""

        digest.update(type(item).__name__.encode("utf-8"))
        digest.update(b"\0")
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _build(
    device: torch.device,
    world_size: int,
) -> tuple[object, Animal2VecPretrainingModel, DistributedDataParallel, TrainingEngine]:
    """Build the deterministic tiny AMP/DDP training stack for one rank."""

    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    config = replace(
        config,
        common=replace(
            config.common,
            fp16=True,
            fp16_init_scale=1.0,
            min_loss_scale=1e-6,
        ),
        distributed=replace(config.distributed, requested_world_size=world_size),
        optimization=replace(
            config.optimization,
            max_update=4,
            gradient_clip_method="adagc",
            adagc_warmup_updates=1,
        ),
    )
    model = Animal2VecPretrainingModel.from_config(config).to(device)
    optimizer = build_optimizer(
        model,
        name=config.optimizer.name,
        learning_rate=config.optimization.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=config.optimization.learning_rate,
        min_lr=config.scheduler.min_lr,
        warmup_updates=config.scheduler.warmup_updates,
        warmup_init_lr=config.scheduler.warmup_init_lr,
        max_updates=config.optimization.max_update,
    )
    wrapped = DistributedDataParallel(
        model,
        device_ids=[device.index],
        find_unused_parameters=True,
        bucket_cap_mb_list=[4096],
    )
    gradient_clipper = build_gradient_clipper(
        model,
        method=config.optimization.gradient_clip_method,
        clip_norm=config.optimization.clip_norm,
        adagc_beta=config.optimization.adagc_beta,
        adagc_relative_clip=config.optimization.adagc_relative_clip,
        adagc_warmup_updates=config.optimization.adagc_warmup_updates,
    )
    engine = TrainingEngine(
        wrapped,
        optimizer,
        scheduler,
        clip_norm=config.optimization.clip_norm,
        device=device,
        gradient_clipper=gradient_clipper,
        use_amp=True,
        amp_init_scale=config.common.fp16_init_scale,
        amp_min_scale=config.common.min_loss_scale,
    )
    return config, model, wrapped, engine


def _step(
    model: DistributedDataParallel,
    engine: TrainingEngine,
    device: torch.device,
    rank: int,
) -> object:
    """Generate a rank-specific synthetic batch and execute one update."""

    batch = {
        "source": torch.randn(2, 64, device=device),
        "id": torch.tensor([rank * 2 + 100, rank * 2 + 101], device=device),
    }
    return engine.step(
        [batch],
        lambda value: model(
            value["source"],
            sample_ids=value["id"],
            update=engine.update,
        ),
    )


def _snapshot(model: Animal2VecPretrainingModel, engine: TrainingEngine, result: object) -> dict[str, Any]:
    """Capture every state component that must match after resume."""

    return _clone_tree({
        "result": {
            "loss": result.loss,
            "sample_size": result.sample_size,
            "gradient_norm": result.gradient_norm,
            "learning_rate": result.learning_rate,
            "update": result.update,
            "pred_var": result.pred_var,
            "target_var": result.target_var,
        },
        "model": model.state_dict(),
        "teacher": model.teacher.model.state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "gradient_clipper": engine.gradient_clipper.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "scaler": engine.scaler.state_dict() if engine.scaler is not None else None,
        "update": engine.update,
        "epoch": engine.epoch,
        "batch_in_epoch": engine.batch_in_epoch,
        "sampler_state": engine.sampler_state,
        "best_metric": engine.best_metric,
        "rng_state": capture_rng_state(),
    })


def _post_step_clocks(
    model: nn.Module,
    engine: TrainingEngine,
    *,
    include_scaler: bool = True,
) -> dict[str, Any]:
    """Capture only state that must never commit after a failed update."""

    teacher = getattr(model, "teacher", None)
    clocks = {
        "gradient_clipper": engine.gradient_clipper.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "teacher": teacher.model.state_dict() if teacher is not None else None,
        "update": engine.update,
    }
    if include_scaler:
        clocks["scaler"] = engine.scaler.state_dict() if engine.scaler is not None else None
    return _clone_tree(clocks)


def _write_failure_report(
    output_dir: Path,
    rank: int,
    report: dict[str, Any],
) -> None:
    """Persist and print one rank's bounded failure-path evidence."""

    (output_dir / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


def _run_optimizer_failure(
    *,
    output_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    checkpoint_group: object,
    failure_rank: int,
) -> None:
    """Inject one rank-local optimizer exception and require terminal agreement."""

    config, model, wrapped, engine = _build(device, world_size)
    before = _post_step_clocks(model, engine)
    original_step = engine.optimizer.step
    if rank == failure_rank:
        def fail_step(*args: object, **kwargs: object) -> None:
            """Raise before mutating optimizer or parameters on the selected rank."""

            raise RuntimeError("injected rank-local optimizer failure")

        engine.optimizer.step = fail_step  # type: ignore[method-assign]

    caught: BaseException | None = None
    try:
        _step(wrapped, engine, device, rank)
    except BaseException as error:
        caught = error
    finally:
        engine.optimizer.step = original_step  # type: ignore[method-assign]
    after = _post_step_clocks(model, engine)

    checkpoint_blocked = False
    try:
        engine.checkpoint_payload(
            stage="pretrain",
            config={"active": config_to_dict(config)},
        )
    except BaseException as error:
        checkpoint_blocked = type(error).__name__ == "DistributedOptimizerStepError"
    future_step_blocked = False
    try:
        engine.step([], lambda _: (_ for _ in ()).throw(AssertionError("unused")))
    except BaseException as error:
        future_step_blocked = type(error).__name__ == "DistributedOptimizerStepError"

    local_report = {
        "rank": rank,
        "error_type": type(caught).__name__ if caught is not None else None,
        "error_message": str(caught) if caught is not None else None,
        "local_cause": str(caught.__cause__) if caught is not None and caught.__cause__ else None,
        "clocks_unchanged": _first_difference(after, before) is None,
        "checkpoint_blocked": checkpoint_blocked,
        "future_step_blocked": future_step_blocked,
    }
    rank_reports: list[object] = [None] * world_size
    dist.all_gather_object(rank_reports, local_report, group=checkpoint_group)
    reports = [item for item in rank_reports if isinstance(item, dict)]
    common_errors = {
        (item["error_type"], item["error_message"])
        for item in reports
    }
    all_ok = (
        len(reports) == world_size
        and len(common_errors) == 1
        and next(iter(common_errors))[0] == "DistributedOptimizerStepError"
        and all(item["clocks_unchanged"] for item in reports)
        and all(item["checkpoint_blocked"] for item in reports)
        and all(item["future_step_blocked"] for item in reports)
        and reports[failure_rank]["local_cause"] == "injected rank-local optimizer failure"
    )
    report = {
        **local_report,
        "failure_mode": "optimizer",
        "failure_rank": failure_rank,
        "all_ranks_terminal": all_ok,
    }
    _write_failure_report(output_dir, rank, report)
    if not all_ok:
        raise RuntimeError(f"distributed optimizer failure was not terminal: {rank_reports}")


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    """Keep a finite loss while injecting a rank-local nonfinite gradient."""

    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        """Return the scalar loss unchanged."""

        return value.clone()

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        """Replace the selected rank's loss derivative with positive infinity."""

        return (torch.full_like(gradient, float("inf")),)


def _run_amp_overflow(
    *,
    output_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    checkpoint_group: object,
    failure_rank: int,
) -> None:
    """Require a rank-local overflow to skip the optimizer on every rank."""

    model = nn.Linear(2, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-2,
        min_lr=1e-4,
        warmup_updates=1,
        max_updates=4,
    )
    clipper = build_gradient_clipper(
        model,
        method="adagc",
        clip_norm=1.0,
        adagc_warmup_updates=1,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=device,
        gradient_clipper=clipper,
        use_amp=True,
        amp_init_scale=8.0,
    )
    before = _clone_tree({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "clocks": _post_step_clocks(model, engine, include_scaler=False),
    })

    def forward(value: torch.Tensor) -> object:
        """Return one finite local loss, with an infinite backward only on one rank."""

        loss = model(value).float().square().sum()
        if rank == failure_rank:
            loss = _FiniteForwardInfiniteBackward.apply(loss)
        return type("Result", (), {"loss": loss, "sample_size": value.shape[0]})()

    result = engine.step([torch.ones(2, 2, device=device)], forward)
    after = _clone_tree({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "clocks": _post_step_clocks(model, engine, include_scaler=False),
    })
    local_report = {
        "rank": rank,
        "skipped": result.skipped,
        "result_update": result.update,
        "engine_update": engine.update,
        "scale": engine.scaler.get_scale() if engine.scaler is not None else None,
        "state_unchanged": _first_difference(after, before) is None,
    }
    rank_reports: list[object] = [None] * world_size
    dist.all_gather_object(rank_reports, local_report, group=checkpoint_group)
    reports = [item for item in rank_reports if isinstance(item, dict)]
    all_ok = (
        len(reports) == world_size
        and all(item["skipped"] for item in reports)
        and all(item["result_update"] == 0 for item in reports)
        and all(item["engine_update"] == 0 for item in reports)
        and all(item["scale"] == 4.0 for item in reports)
        and all(item["state_unchanged"] for item in reports)
    )
    report = {
        **local_report,
        "failure_mode": "amp-overflow",
        "failure_rank": failure_rank,
        "all_ranks_skipped": all_ok,
    }
    _write_failure_report(output_dir, rank, report)
    if not all_ok:
        raise RuntimeError(f"distributed AMP overflow did not skip every rank: {rank_reports}")


def _run_nonfinite_loss(
    *,
    output_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    checkpoint_group: object,
    failure_rank: int,
) -> None:
    """Require one rank-local nonfinite loss to fail identically before backward."""

    model = nn.Linear(2, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-2,
        min_lr=1e-4,
        warmup_updates=1,
        max_updates=4,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=device,
    )
    before = _clone_tree({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "clocks": _post_step_clocks(model, engine),
    })

    def forward(value: torch.Tensor) -> object:
        """Return a nonfinite forward loss on only the selected rank."""

        loss = model(value).float().square().sum()
        if rank == failure_rank:
            loss = loss * torch.tensor(float("inf"), device=device)
        return type("Result", (), {"loss": loss, "sample_size": value.shape[0]})()

    caught: BaseException | None = None
    try:
        engine.step([torch.ones(2, 2, device=device)], forward)
    except BaseException as error:
        caught = error
        # The pre-fix rank-local branch exits before its peers' totals and
        # decision collectives. Match those calls so RED reports divergence
        # instead of hanging. The coordinated implementation never enters here.
        if str(error) == "non-finite loss at update 0":
            dist.all_reduce(torch.zeros(2, dtype=torch.float64, device=device))
            dist.all_reduce(torch.zeros((), dtype=torch.int32, device=device))
    after = _clone_tree({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "clocks": _post_step_clocks(model, engine),
    })
    local_report = {
        "rank": rank,
        "error_type": type(caught).__name__ if caught is not None else None,
        "error_message": str(caught) if caught is not None else None,
        "state_unchanged": _first_difference(after, before) is None,
    }
    rank_reports: list[object] = [None] * world_size
    dist.all_gather_object(rank_reports, local_report, group=checkpoint_group)
    reports = [item for item in rank_reports if isinstance(item, dict)]
    common_errors = {
        (item["error_type"], item["error_message"])
        for item in reports
    }
    all_ok = (
        len(reports) == world_size
        and common_errors == {
            (
                "FloatingPointError",
                "non-finite loss on one or more ranks at update 0",
            )
        }
        and all(item["state_unchanged"] for item in reports)
    )
    report = {
        **local_report,
        "failure_mode": "nonfinite-loss",
        "failure_rank": failure_rank,
        "all_ranks_failed_identically": all_ok,
    }
    _write_failure_report(output_dir, rank, report)
    if not all_ok:
        raise RuntimeError(f"distributed nonfinite loss decision diverged: {rank_reports}")


def main() -> int:
    """Run continuous and resumed branches and emit exactness evidence per rank."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--failure-mode",
        choices=("none", "optimizer", "amp-overflow", "nonfinite-loss"),
        default="none",
    )
    parser.add_argument("--failure-rank", type=int, default=1)
    arguments = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if rank == 0:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if arguments.failure_mode != "none":
        if not 0 <= arguments.failure_rank < world_size:
            raise ValueError("--failure-rank must name an active distributed rank")
        runners = {
            "optimizer": _run_optimizer_failure,
            "amp-overflow": _run_amp_overflow,
            "nonfinite-loss": _run_nonfinite_loss,
        }
        runner = runners[arguments.failure_mode]
        runner(
            output_dir=arguments.output_dir,
            device=device,
            rank=rank,
            world_size=world_size,
            checkpoint_group=checkpoint_group,
            failure_rank=arguments.failure_rank,
        )
        dist.barrier()
        dist.destroy_process_group()
        return 0

    random.seed(4242)
    np.random.seed(4242)
    torch.manual_seed(4242)
    torch.cuda.manual_seed_all(4242)
    config, model, wrapped, engine = _build(device, world_size)

    random.seed(50_000 + rank)
    np.random.seed(60_000 + rank)
    torch.manual_seed(70_000 + rank)
    torch.cuda.manual_seed_all(80_000 + rank)
    first_result = _step(wrapped, engine, device, rank)
    engine.epoch = 3
    # The production checkpoint stores the global TokenBatchSampler state once;
    # only RNG state is rank-specific. Keep this state identical on all ranks.
    engine.sampler_state = {"epoch": 3, "next_batch": 11}
    engine.best_metric = 0.125

    payload = engine.checkpoint_payload(
        stage="pretrain",
        config={"active": config_to_dict(config)},
    )
    ranked_rng = gather_rank_rng_states(
        payload["rng_state"],
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
    checkpoint_path = arguments.output_dir / "resume-point.pt"
    if rank == 0:
        if ranked_rng is None:
            raise RuntimeError("rank zero received no distributed RNG state")
        if ranked_topology is None:
            raise RuntimeError("rank zero received no distributed topology state")
        payload["rng_state"] = ranked_rng
        payload["topology"] = ranked_topology
        save_checkpoint(checkpoint_path, payload)
    dist.barrier()

    expected_result = _step(wrapped, engine, device, rank)
    expected = _snapshot(model, engine, expected_result)
    expected_digest = _digest(expected)

    del engine, wrapped, model
    torch.cuda.empty_cache()
    random.seed(1 + rank)
    np.random.seed(2 + rank)
    torch.manual_seed(3 + rank)
    torch.cuda.manual_seed_all(4 + rank)
    _, resumed_model, resumed_wrapped, resumed_engine = _build(device, world_size)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    resumed_engine.restore(checkpoint)
    actual_result = _step(resumed_wrapped, resumed_engine, device, rank)
    actual = _snapshot(resumed_model, resumed_engine, actual_result)
    actual_digest = _digest(actual)
    difference = _first_difference(actual, expected)
    adagc_state_digest = _digest(
        _clone_tree(resumed_engine.gradient_clipper.state_dict())
    )
    rank_adagc_digests: list[object] = [None] * world_size
    dist.all_gather_object(
        rank_adagc_digests,
        adagc_state_digest,
        group=checkpoint_group,
    )
    all_ranks_adagc_state = len(set(rank_adagc_digests)) == 1

    topology_ok = (
        checkpoint["rng_state"]["schema"] == RANK_LOCAL_RNG_SCHEMA
        and checkpoint["topology"]["schema"] == "a2v2.topology.v1"
        and [
            record["global_rank"] for record in checkpoint["topology"]["by_rank"]
        ] == list(range(world_size))
    )
    local_ok = torch.tensor(
        int(difference is None and topology_ok),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    report = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "rng_gather_backend": str(dist.get_backend(checkpoint_group)),
        "device": str(device),
        "first_update": first_result.update,
        "compared_update": actual_result.update,
        "expected_digest": expected_digest,
        "actual_digest": actual_digest,
        "local_exact": difference is None,
        "all_ranks_exact": bool(local_ok.item()),
        "adagc_state_digest": adagc_state_digest,
        "all_ranks_adagc_state": all_ranks_adagc_state,
        "rng_schema": checkpoint["rng_state"]["schema"],
        "topology_schema": checkpoint["topology"]["schema"],
        "topology_ok": topology_ok,
        "first_difference": difference,
        "amp_scale": resumed_engine.scaler.get_scale() if resumed_engine.scaler is not None else None,
        "checkpoint": str(checkpoint_path),
    }
    (arguments.output_dir / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if (
        difference is not None
        or not report["topology_ok"]
        or not report["all_ranks_exact"]
        or not report["all_ranks_adagc_state"]
    ):
        raise RuntimeError(f"distributed resume diverged on rank {rank}: {difference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
