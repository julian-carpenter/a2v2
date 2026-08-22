"""Two-rank Gloo evidence for coordinated safe-point preemption checkpoints."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn

from a2v2.slurm import (
    OUTPUT_LOCK_NAME,
    PREEMPTION_EXIT_CODE,
    DistributedEnvironment,
    OutputLockError,
    PreemptionFlag,
    RankTopology,
    SlurmEnvironment,
    TrainingPreempted,
)
from a2v2.training import (
    CheckpointError,
    RANK_LOCAL_RNG_SCHEMA,
    CosineUpdateScheduler,
    TrainingEngine,
    load_checkpoint,
    restore_rng_state,
)


class _Result:
    """Minimal summed-loss contract consumed by TrainingEngine."""

    def __init__(self, loss: torch.Tensor, sample_size: int) -> None:
        self.loss = loss
        self.sample_size = sample_size


class _InjectedPayloadFailure:
    """Fail checkpoint payload preparation on exactly one selected rank."""

    def checkpoint_payload(self, **_: object) -> dict[str, object]:
        """Raise before any checkpoint state gather can begin."""

        raise RuntimeError("injected rank-local payload preparation failure")


def _engine() -> tuple[nn.Linear, TrainingEngine]:
    """Build a tiny CPU engine for a completed-update checkpoint boundary."""

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.05,
        min_lr=0.01,
        warmup_updates=0,
        max_updates=2,
    )
    return model, TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
    )


def _worker(output_directory: Path) -> int:
    """Run one torchrun worker and emit rank-local checkpoint evidence."""

    from a2v2.workflows import (
        _checkpoint_at_preemption_safe_point,
        _prepare_output_directory,
        _write_training_checkpoint,
    )

    dist.init_process_group("gloo")
    distributed = DistributedEnvironment.from_mapping(os.environ)
    rank = distributed.rank
    allocation = SlurmEnvironment(
        job_id="gloo-job-4815",
        node_count=1,
        node_id=0,
        processes_per_node=2,
        node_list="localhost",
    )
    owned_locks: list[object] = []
    _prepare_output_directory(
        output_directory,
        distributed=distributed,
        slurm=allocation,
        run_id="gloo-preemption",
        group=None,
        owned_locks=owned_locks,
    )
    lock_visible = (output_directory / OUTPUT_LOCK_NAME).is_file()

    random.seed(10_000 + rank)
    np.random.seed(20_000 + rank)
    torch.manual_seed(30_000 + rank)
    model, engine = _engine()
    batch = torch.tensor([[rank + 1.0, 2.0]])
    engine.step(
        [batch],
        lambda value: _Result(model(value).square().sum(), value.shape[0]),
    )
    assert engine.update == 1

    flag = PreemptionFlag()
    if rank == 1:
        flag.request()
    checkpoint_path = output_directory / "checkpoint_last.pt"
    topology = RankTopology(
        hostname=f"gloo-host-{rank}",
        global_rank=rank,
        local_rank=distributed.local_rank,
        local_world_size=distributed.local_world_size,
        visible_cuda_devices=0,
        selected_cuda_device=None,
        cuda_device_name=None,
        cuda_device_uuid=None,
    )
    wrote = False

    def write_checkpoint() -> None:
        """Record whether this rank was the sole atomic checkpoint writer."""

        nonlocal wrote
        wrote = _write_training_checkpoint(
            checkpoint_path,
            engine=engine,
            stage="pretrain",
            config={"active": {}},
            topology=topology,
            world_size=distributed.world_size,
            rank=rank,
            group=None,
        )

    flushed: list[bool] = []
    preemption_error: TrainingPreempted | None = None
    try:
        _checkpoint_at_preemption_safe_point(
            flag,
            update=engine.update,
            checkpoint_path=checkpoint_path,
            collective_device=torch.device("cpu"),
            group=None,
            write_checkpoint=write_checkpoint,
            flush_logs=lambda: flushed.append(True),
        )
    except TrainingPreempted as error:
        preemption_error = error
    else:
        raise AssertionError("all ranks must take the coordinated preemption exit")
    expected_random = torch.rand(4)
    assert preemption_error is not None

    torch.manual_seed(1 + rank)
    checkpoint = load_checkpoint(checkpoint_path)
    restore_rng_state(checkpoint["rng_state"])
    actual_random = torch.rand(4)
    local = {
        "rank": rank,
        "requested": True,
        "writer": wrote,
        "flushed": flushed == [True],
        "lock_visible": lock_visible,
        "checkpoint_update": checkpoint["update"],
        "rng_schema": checkpoint["rng_state"]["schema"],
        "rng_exact": torch.equal(expected_random, actual_random),
        "topology_schema": checkpoint["topology"]["schema"],
        "topology_ranks": [
            record["global_rank"] for record in checkpoint["topology"]["by_rank"]
        ],
        "exit_code": preemption_error.exit_code,
    }
    reports: list[object] = [None] * distributed.world_size
    dist.all_gather_object(reports, local)
    synchronized = (
        all(isinstance(report, dict) for report in reports)
        and all(report["requested"] for report in reports)
        and sum(int(report["writer"]) for report in reports) == 1
        and all(report["flushed"] for report in reports)
        and all(report["lock_visible"] for report in reports)
        and all(report["checkpoint_update"] == 1 for report in reports)
        and all(report["rng_schema"] == RANK_LOCAL_RNG_SCHEMA for report in reports)
        and all(report["rng_exact"] for report in reports)
        and all(report["topology_schema"] == "a2v2.topology.v1" for report in reports)
        and all(report["topology_ranks"] == [0, 1] for report in reports)
        and all(report["exit_code"] == PREEMPTION_EXIT_CODE for report in reports)
    )
    local["synchronized"] = synchronized
    (output_directory / f"rank-{rank}.json").write_text(
        json.dumps(local, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    if rank == 0:
        assert len(owned_locks) == 1
        owned_locks[0].release()
    dist.barrier()
    dist.destroy_process_group()
    if not synchronized:
        raise RuntimeError(f"Gloo preemption contract diverged: {reports}")
    return 0


def test_two_rank_gloo_preemption_is_coordinated(tmp_path: Path) -> None:
    """Catch lone-rank checkpointing, mismatched exits, or rank-zero RNG replay."""

    output_directory = tmp_path / "gloo-preemption"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--worker-output",
            str(output_directory),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    reports = [
        json.loads((output_directory / f"rank-{rank}.json").read_text())
        for rank in range(2)
    ]
    assert all(report["synchronized"] for report in reports)
    assert sum(int(report["writer"]) for report in reports) == 1
    assert {report["exit_code"] for report in reports} == {75}


def _failure_worker(output_directory: Path, failure_mode: str) -> int:
    """Inject one checkpoint or visibility failure and require a common outcome."""

    import a2v2.training as training
    import a2v2.workflows as workflows

    dist.init_process_group("gloo", timeout=timedelta(seconds=10))
    distributed = DistributedEnvironment.from_mapping(os.environ)
    rank = distributed.rank
    reports: list[object] = [None] * distributed.world_size

    if failure_mode == "visibility":
        original_is_dir = Path.is_dir

        def asymmetric_is_dir(path: Path) -> bool:
            """Hide only the shared output directory from global rank one."""

            if path == output_directory and rank == 1:
                return False
            return original_is_dir(path)

        Path.is_dir = asymmetric_is_dir
        error: str | None = None
        try:
            workflows._prepare_output_directory(
                output_directory,
                distributed=distributed,
                slurm=None,
                run_id="visibility-failure",
                group=None,
                owned_locks=[],
            )
        except OutputLockError as caught:
            error = str(caught)
        finally:
            Path.is_dir = original_is_dir
        local = {"rank": rank, "error": error}
        dist.all_gather_object(reports, local)
        synchronized = (
            all(isinstance(report, dict) for report in reports)
            and all(report["error"] is not None for report in reports)
            and len({report["error"] for report in reports}) == 1
            and "rank 1" in str(reports[0]["error"])
        )
    else:
        workflows._prepare_output_directory(
            output_directory,
            distributed=distributed,
            slurm=None,
            run_id=failure_mode,
            group=None,
            owned_locks=[],
        )
        random.seed(40_000 + rank)
        np.random.seed(50_000 + rank)
        torch.manual_seed(60_000 + rank)
        _, engine = _engine()
        checkpoint_engine: object = engine
        if failure_mode == "prepare" and rank == 1:
            checkpoint_engine = _InjectedPayloadFailure()
        if failure_mode == "merge" and rank == 1:
            def corrupt_rng_state(_: object) -> bytes:
                """Pass local preparation but fail rank-zero RNG decoding."""

                return b"injected-corrupt-rng-state"

            training.serialize_rng_state = corrupt_rng_state
            workflows.serialize_rng_state = corrupt_rng_state
        if failure_mode in {"write", "ordinary-write"} and rank == 0:
            def fail_save(*_: object, **__: object) -> None:
                """Fail only the sole checkpoint writer after every gather."""

                raise OSError("injected rank-zero checkpoint write failure")

            workflows.save_checkpoint = fail_save
        topology = RankTopology(
            hostname=f"gloo-host-{rank}",
            global_rank=rank,
            local_rank=distributed.local_rank,
            local_world_size=distributed.local_world_size,
            visible_cuda_devices=0,
            selected_cuda_device=None,
            cuda_device_name=None,
            cuda_device_uuid=None,
        )
        checkpoint_path = output_directory / f"{failure_mode}.pt"

        def write_checkpoint() -> None:
            """Run the production distributed writer with the injected stage."""

            workflows._write_training_checkpoint(
                checkpoint_path,
                engine=checkpoint_engine,  # type: ignore[arg-type]
                stage="pretrain",
                config={"active": {}},
                topology=topology,
                world_size=distributed.world_size,
                rank=rank,
                group=None,
            )

        error = None
        preempted = False
        flushed: list[bool] = []
        try:
            if failure_mode == "ordinary-write":
                write_checkpoint()
            else:
                flag = PreemptionFlag()
                flag.request()
                workflows._checkpoint_at_preemption_safe_point(
                    flag,
                    update=1,
                    checkpoint_path=checkpoint_path,
                    collective_device=torch.device("cpu"),
                    group=None,
                    write_checkpoint=write_checkpoint,
                    flush_logs=lambda: flushed.append(True),
                )
        except CheckpointError as caught:
            error = str(caught)
        except TrainingPreempted:
            preempted = True
        local = {
            "rank": rank,
            "error": error,
            "preempted": preempted,
            "flushed": bool(flushed),
            "checkpoint_exists": checkpoint_path.exists(),
        }
        dist.all_gather_object(reports, local)
        synchronized = (
            all(isinstance(report, dict) for report in reports)
            and all(report["error"] is not None for report in reports)
            and len({report["error"] for report in reports}) == 1
            and all(not report["preempted"] for report in reports)
            and all(not report["flushed"] for report in reports)
            and all(not report["checkpoint_exists"] for report in reports)
        )

    result = {"rank": rank, "mode": failure_mode, "synchronized": synchronized}
    (output_directory / f"{failure_mode}-rank-{rank}.json").write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    dist.destroy_process_group()
    if not synchronized:
        raise RuntimeError(f"Gloo {failure_mode} failure diverged: {reports}")
    return 0


@pytest.mark.parametrize("failure_mode", ["prepare", "merge", "write"])
def test_two_rank_gloo_preemption_checkpoint_failures_are_common(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    """Catch stage failures that hang peers or permit a requeue exit."""

    _run_failure_subprocess(tmp_path, failure_mode)


def test_two_rank_gloo_ordinary_writer_failure_is_common(tmp_path: Path) -> None:
    """Catch rank zero failing a periodic/final save while peers continue."""

    _run_failure_subprocess(tmp_path, "ordinary-write")


def test_two_rank_gloo_output_visibility_failure_is_common(tmp_path: Path) -> None:
    """Catch one rank entering training when another cannot see shared output."""

    _run_failure_subprocess(tmp_path, "visibility")


def _run_failure_subprocess(tmp_path: Path, failure_mode: str) -> None:
    """Launch one bounded Gloo failure case and require both worker reports."""

    output_directory = tmp_path / failure_mode
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--worker-output",
            str(output_directory),
            "--failure-mode",
            failure_mode,
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    reports = [
        json.loads(
            (output_directory / f"{failure_mode}-rank-{rank}.json").read_text()
        )
        for rank in range(2)
    ]
    assert all(report["synchronized"] for report in reports)


def main() -> int:
    """Dispatch torchrun worker mode without asking pytest to recurse."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-output", required=True, type=Path)
    parser.add_argument("--failure-mode")
    arguments = parser.parse_args()
    if arguments.failure_mode is not None:
        return _failure_worker(arguments.worker_output, arguments.failure_mode)
    return _worker(arguments.worker_output)


if __name__ == "__main__":
    raise SystemExit(main())
