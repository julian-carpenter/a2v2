"""Two-rank Gloo evidence for coordinated safe-point preemption checkpoints."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import sys
import time

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


ROOT = Path(__file__).resolve().parents[2]
NODE_LAUNCHER = ROOT / "scripts/a2v2_slurm_node.sh"
SLURM_DRIVER = ROOT / "scripts/reproduce_meerkat_slurm.sh"
EVALUATOR = ROOT / "scripts/evaluate_finetuning_checkpoint.py"


def _write_executable(path: Path, source: str) -> None:
    """Write one executable test double at an external command boundary."""

    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _launcher_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create shared-looking manifest, output, and config paths for real mode."""

    manifest_directory = tmp_path / "shared manifests"
    output_directory = tmp_path / "shared output"
    config_path = tmp_path / "config with spaces.yaml"
    manifest_directory.mkdir()
    for name in ("pretrain.tsv", "train_0.tsv", "valid_0.tsv"):
        (manifest_directory / name).touch()
    output_directory.mkdir()
    config_path.write_text(
        (ROOT / "configs/MeerKAT/a2v_large_pretrain_best.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    return manifest_directory, output_directory, config_path


def _slurm_environment(
    *,
    nodes: int = 2,
    gpus_per_node: int = 4,
    node_id: int = 0,
) -> dict[str, str]:
    """Return one complete homogeneous scheduler environment."""

    return {
        **os.environ,
        "SLURM_JOB_ID": "4815",
        "SLURM_JOB_NUM_NODES": str(nodes),
        "SLURM_NODEID": str(node_id),
        "SLURM_GPUS_ON_NODE": str(gpus_per_node),
        "SLURM_GPUS_PER_NODE": f"{gpus_per_node}(x{nodes})",
        "SLURM_JOB_NODELIST": "node[01-02]" if nodes == 2 else "node01",
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(index) for index in range(gpus_per_node)
        ),
        "A2V2_CONTRACT_PYTHON": sys.executable,
    }


def _single_node_stage_environment(tmp_path: Path) -> dict[str, str]:
    """Provide real contract construction with only external SLURM work mocked."""

    fake_bin = tmp_path / "fake-stage-bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "python", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "srun",
        """#!/usr/bin/env bash
output=''
while (($#)); do
    if [[ "$1" == '--output-dir' ]]; then output="$2"; shift 2; else shift; fi
done
if [[ -n "$output" ]]; then
    mkdir -p "$output"
    printf 'stage checkpoint\n' > "$output/checkpoint_last.pt"
fi
""",
    )
    return {
        **_slurm_environment(nodes=1, gpus_per_node=1),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_PYTHON": str(fake_bin / "python"),
    }


def _run_shell(
    script: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one launcher and retain its rendered command or validation error."""

    return subprocess.run(
        ["bash", str(script), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _rendered_commands(stdout: str) -> list[list[str]]:
    """Parse Bash percent-q audit records back into exact argument arrays."""

    return [
        shlex.split(line.removeprefix("Launching:"))
        for line in stdout.splitlines()
        if line.startswith("Launching:")
    ]


def _option_value(command: list[str], option: str) -> str:
    """Return one exact separate-token option value from a rendered command."""

    index = command.index(option)
    return command[index + 1]


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


def test_node_launcher_preserves_scheduler_visibility_and_maps_local_workers(
    tmp_path: Path,
) -> None:
    """Catch replacing SLURM's GPU mask or launching a worker count unlike it."""

    manifests, output, config = _launcher_inputs(tmp_path)
    fake_python = tmp_path / "python bin" / "python"
    fake_python.parent.mkdir()
    trace = tmp_path / "python-trace.txt"
    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
set -eu
{
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
    printf 'A2V2_RUN_ID=%s\n' "${A2V2_RUN_ID-<unset>}"
    printf 'A2V2_SLURM_CONTRACT_FINGERPRINT=%s\n' "${A2V2_SLURM_CONTRACT_FINGERPRINT-<unset>}"
    printf 'ARG=%s\n' "$@"
} > "${A2V2_TEST_TRACE}"
""",
    )
    environment = {
        **_slurm_environment(node_id=1),
        "CUDA_VISIBLE_DEVICES": "7,5,3,1",
        "A2V2_PYTHON": str(fake_python),
        "A2V2_TEST_TRACE": str(trace),
    }

    completed = _run_shell(
        NODE_LAUNCHER,
        "pretrain",
        "--job-id", "4815",
        "--nodes", "2",
        "--gpus-per-node", "4",
        "--master-addr", "node01",
        "--master-port", "23451",
        "--run-id", "research run/pretrain",
        "--manifest-dir", str(manifests),
        "--manifest-fingerprint", "sha256:manifest",
        "--output-dir", str(output),
        "--config", str(config),
        "--",
        "a2v2-train",
        "--config", str(config),
        "--override", f"checkpoint.save_dir={output}",
        environment=environment,
    )

    assert completed.returncode == 0, completed.stderr
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "CUDA_VISIBLE_DEVICES=7,5,3,1",
        "A2V2_RUN_ID=research run/pretrain",
    ]
    contract = lines[2].partition("=")[2]
    assert len(contract) == 64
    arguments = [line.removeprefix("ARG=") for line in lines[3:]]
    assert arguments == [
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--node-rank=1",
        "--nproc-per-node=4",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=node01:23451",
        "--rdzv-id=4815-pretrain",
        "a2v2-train",
        "--config",
        str(config),
        "--override",
        f"checkpoint.save_dir={output}",
    ]


def test_node_dry_run_preserves_an_explicit_endpoint_and_uses_unique_phase_ids(
    tmp_path: Path,
) -> None:
    """Catch endpoint rewriting or rendezvous collisions between training phases."""

    commands = []
    for phase in ("pretrain", "finetune"):
        completed = _run_shell(
            NODE_LAUNCHER,
            phase,
            "--dry-run",
            "--job-id", "4815",
            "--nodes", "2",
            "--node-rank", "1",
            "--gpus-per-node", "4",
            "--rdzv-endpoint", "rendezvous.example:23451",
            "--run-id", f"research/{phase}",
            "--manifest-dir", str(tmp_path / "absent manifests"),
            "--output-dir", str(tmp_path / phase),
            "--config", str(tmp_path / f"{phase}.yaml"),
            "--",
            "a2v2-train",
        )
        assert completed.returncode == 0, completed.stderr
        commands.extend(_rendered_commands(completed.stdout))

    assert len(commands) == 2
    assert all(
        "--rdzv-endpoint=rendezvous.example:23451" in command
        for command in commands
    )
    assert [
        next(argument for argument in command if argument.startswith("--rdzv-id="))
        for command in commands
    ] == ["--rdzv-id=4815-pretrain", "--rdzv-id=4815-finetune"]


def test_node_launcher_uses_python_canonical_contract_and_bracketed_ipv6(
    tmp_path: Path,
) -> None:
    """Catch shell-only hashing or rejection of Task 9 bracketed IPv6 syntax."""

    from a2v2.config import config_to_dict, load_config
    from a2v2.slurm import RunContract, config_fingerprint

    manifests, output, config = _launcher_inputs(tmp_path)
    overrides = [
        f"task.data={manifests}",
        f"checkpoint.save_dir={output}",
        "distributed_training.distributed_world_size=8",
    ]
    arguments = [
        "a2v2-train",
        "--config",
        str(config),
    ]
    for override in overrides:
        arguments.extend(("--override", override))
    completed = _run_shell(
        NODE_LAUNCHER,
        "pretrain",
        "--dry-run",
        "--job-id", "4815",
        "--nodes", "2",
        "--node-rank", "1",
        "--gpus-per-node", "4",
        "--rdzv-endpoint", "[2001:db8::1]:23451",
        "--run-id", "research/pretrain",
        "--manifest-dir", str(manifests),
        "--manifest-fingerprint", "sha256:manifest",
        "--output-dir", str(output),
        "--config", str(config),
        "--",
        *arguments,
    )

    assert completed.returncode == 0, completed.stderr
    command = _rendered_commands(completed.stdout)[0]
    exported = {
        item.partition("=")[0]: item.partition("=")[2]
        for item in command
        if item.startswith("A2V2_")
    }
    contract = RunContract.from_mapping(
        json.loads(exported["A2V2_SLURM_RUN_CONTRACT"])
    )
    assert contract.rendezvous_endpoint == "[2001:db8::1]:23451"
    assert exported["A2V2_SLURM_CONTRACT_FINGERPRINT"] == contract.fingerprint()
    assert exported["A2V2_SLURM_CONFIG_FINGERPRINT"] == config_fingerprint(
        config_to_dict(load_config(config, overrides))
    )


def test_all_and_standalone_launches_share_each_phase_contract_identity(
    tmp_path: Path,
) -> None:
    """Catch overall phase selection leaking into a stage manifest contract."""

    common = (
        str(tmp_path / "missing manifests"),
        str(tmp_path / "output"),
        "--dry-run",
        "--nodes", "2",
        "--gpus-per-node", "4",
        "--job-id", "4815",
        "--master-addr", "node01",
        "--run-id", "stable-run",
    )

    def contracts(phase: str) -> dict[str, tuple[str, str]]:
        """Return contract and manifest fingerprints rendered for one selection."""

        completed = _run_shell(
            SLURM_DRIVER,
            *common,
            "--phase", phase,
        )
        assert completed.returncode == 0, completed.stderr
        result: dict[str, tuple[str, str]] = {}
        for command in _rendered_commands(completed.stdout):
            if str(NODE_LAUNCHER) not in command:
                continue
            stage = command[command.index(str(NODE_LAUNCHER)) + 1]
            result[stage] = (
                _option_value(command, "--contract-fingerprint"),
                _option_value(command, "--manifest-fingerprint"),
            )
        return result

    all_contracts = contracts("all")
    assert all_contracts["pretrain"] == contracts("pretrain")["pretrain"]
    assert all_contracts["finetune"] == contracts("finetune")["finetune"]
    assert all_contracts["pretrain"] != all_contracts["finetune"]


def test_pretrain_contract_uses_configured_train_subset_manifest(
    tmp_path: Path,
) -> None:
    """Bind pretraining to dataset.train_subset, not a neighboring legacy TSV."""

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    legacy_manifest = manifests / "pretrain.tsv"
    selected_manifest = manifests / "alternate.tsv"
    legacy_manifest.write_text("legacy\n", encoding="utf-8")
    selected_manifest.write_text("selected-v1\n", encoding="utf-8")
    config = tmp_path / "alternate-pretrain.yaml"
    source = (
        ROOT / "configs/MeerKAT/a2v_large_pretrain_best.yaml"
    ).read_text(encoding="utf-8")
    assert "train_subset: pretrain" in source
    config.write_text(
        source.replace("train_subset: pretrain", "train_subset: alternate"),
        encoding="utf-8",
    )

    def identity() -> tuple[str, str]:
        """Return rendered contract and selected-manifest fingerprints."""

        completed = _run_shell(
            SLURM_DRIVER,
            str(manifests),
            str(tmp_path / "output"),
            "--phase", "pretrain",
            "--pretrain-config", str(config),
            "--dry-run",
            "--nodes", "1",
            "--gpus-per-node", "1",
            "--job-id", "4815",
            "--master-addr", "node01",
        )
        assert completed.returncode == 0, completed.stderr
        command = next(
            command
            for command in _rendered_commands(completed.stdout)
            if str(NODE_LAUNCHER) in command
        )
        return (
            _option_value(command, "--contract-fingerprint"),
            _option_value(command, "--manifest-fingerprint"),
        )

    baseline = identity()
    legacy_manifest.write_text("legacy-changed\n", encoding="utf-8")
    assert identity() == baseline
    selected_manifest.write_text("selected-v2\n", encoding="utf-8")
    assert identity() != baseline


@pytest.mark.parametrize(
    ("phase", "manifest_names", "unrelated_config_option", "checkpoint"),
    (
        (
            "pretrain",
            ("pretrain.tsv",),
            "--finetune-config",
            "pretrain/checkpoint_last.pt",
        ),
        (
            "finetune",
            ("train_0.tsv", "valid_0.tsv"),
            "--pretrain-config",
            "finetune/checkpoint_last.pt",
        ),
    ),
)
def test_real_selected_stage_ignores_unrelated_manifest_and_config_inputs(
    tmp_path: Path,
    phase: str,
    manifest_names: tuple[str, ...],
    unrelated_config_option: str,
    checkpoint: str,
) -> None:
    """Catch a standalone stage loading the other stage's inaccessible inputs."""

    manifests = tmp_path / "selected manifests"
    manifests.mkdir()
    for name in manifest_names:
        (manifests / name).write_text("selected\n", encoding="utf-8")
    output = tmp_path / "output"
    if phase == "finetune":
        pretrained = output / "pretrain/checkpoint_last.pt"
        pretrained.parent.mkdir(parents=True)
        pretrained.write_bytes(b"pretrained")

    completed = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--phase", phase,
        "--master-addr", "node01",
        unrelated_config_option, str(tmp_path / "inaccessible unrelated.yaml"),
        environment=_single_node_stage_environment(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / checkpoint).is_file()
    assert (output / f"{checkpoint}.stage-complete").is_file()


def test_real_evaluation_uses_only_finetune_contract_inputs(tmp_path: Path) -> None:
    """Catch final evaluation loading the unrelated pretraining config or manifest."""

    manifests = tmp_path / "finetune manifests"
    manifests.mkdir()
    for name in ("train_0.tsv", "valid_0.tsv"):
        (manifests / name).write_text("selected\n", encoding="utf-8")
    output = tmp_path / "output"
    pretrained = output / "pretrain/checkpoint_last.pt"
    pretrained.parent.mkdir(parents=True)
    pretrained.write_bytes(b"pretrained")
    environment = _single_node_stage_environment(tmp_path)
    missing_pretrain_config = tmp_path / "inaccessible pretrain.yaml"
    common = (
        str(manifests),
        str(output),
        "--master-addr", "node01",
        "--pretrain-config", str(missing_pretrain_config),
    )
    trained = _run_shell(
        SLURM_DRIVER,
        *common,
        "--phase", "finetune",
        environment=environment,
    )
    assert trained.returncode == 0, trained.stderr

    evaluated = _run_shell(
        SLURM_DRIVER,
        *common,
        "--phase", "evaluate",
        environment=environment,
    )

    assert evaluated.returncode == 0, evaluated.stderr


@pytest.mark.parametrize(
    ("phase", "present_manifests", "config_option", "expected_error"),
    (
        ("pretrain", (), None, "pretrain.tsv"),
        ("finetune", ("valid_0.tsv",), None, "train_0.tsv"),
        ("all", ("pretrain.tsv", "valid_0.tsv"), None, "train_0.tsv"),
        ("pretrain", ("pretrain.tsv",), "--pretrain-config", "pretraining config"),
        (
            "finetune",
            ("train_0.tsv", "valid_0.tsv"),
            "--finetune-config",
            "fine-tuning config",
        ),
        (
            "all",
            ("pretrain.tsv", "train_0.tsv", "valid_0.tsv"),
            "--finetune-config",
            "fine-tuning config",
        ),
    ),
)
def test_real_selected_stage_still_requires_its_own_inputs(
    tmp_path: Path,
    phase: str,
    present_manifests: tuple[str, ...],
    config_option: str | None,
    expected_error: str,
) -> None:
    """Catch phase isolation weakening actionable selected-input validation."""

    manifests = tmp_path / "selected manifests"
    manifests.mkdir()
    for name in present_manifests:
        (manifests / name).write_text("selected\n", encoding="utf-8")
    output = tmp_path / "output"
    arguments = [
        str(manifests),
        str(output),
        "--phase", phase,
        "--master-addr", "node01",
    ]
    if config_option is not None:
        arguments.extend((config_option, str(tmp_path / "missing selected.yaml")))

    completed = _run_shell(
        SLURM_DRIVER,
        *arguments,
        environment=_single_node_stage_environment(tmp_path),
    )

    assert completed.returncode == 2
    assert expected_error in completed.stderr


def test_node_launcher_fails_before_torchrun_on_output_lock_contention(
    tmp_path: Path,
) -> None:
    """Catch starting workers behind a stale lock or silently stealing ownership."""

    manifests, output, config = _launcher_inputs(tmp_path)
    (output / OUTPUT_LOCK_NAME).write_text("occupied\n", encoding="utf-8")
    trace = tmp_path / "python-ran"
    fake_python = tmp_path / "python"
    _write_executable(fake_python, "#!/usr/bin/env bash\ntouch \"${A2V2_TEST_TRACE}\"\n")
    environment = {
        **_slurm_environment(nodes=1, gpus_per_node=1),
        "A2V2_PYTHON": str(fake_python),
        "A2V2_TEST_TRACE": str(trace),
    }

    completed = _run_shell(
        NODE_LAUNCHER,
        "pretrain",
        "--run-id", "locked/pretrain",
        "--manifest-dir", str(manifests),
        "--output-dir", str(output),
        "--config", str(config),
        "--master-addr", "node01",
        "--master-port", "24500",
        "--",
        "a2v2-train",
        environment=environment,
    )

    assert completed.returncode == 2
    assert "output lock already exists" in completed.stderr
    assert "--recover-lock pretrain:" in completed.stderr
    assert not trace.exists()


def test_nonzero_node_accepts_the_lock_rank_zero_may_have_just_acquired(
    tmp_path: Path,
) -> None:
    """Catch a slow peer mistaking the current run's rank-zero lock for contention."""

    manifests, output, config = _launcher_inputs(tmp_path)
    (output / OUTPUT_LOCK_NAME).write_text("rank zero now owns it\n", encoding="utf-8")
    trace = tmp_path / "python-ran"
    fake_python = tmp_path / "python"
    _write_executable(fake_python, "#!/usr/bin/env bash\ntouch \"${A2V2_TEST_TRACE}\"\n")
    environment = {
        **_slurm_environment(node_id=1),
        "A2V2_PYTHON": str(fake_python),
        "A2V2_TEST_TRACE": str(trace),
    }

    completed = _run_shell(
        NODE_LAUNCHER,
        "pretrain",
        "--run-id", "current/pretrain",
        "--manifest-dir", str(manifests),
        "--output-dir", str(output),
        "--config", str(config),
        "--master-addr", "node01",
        "--master-port", "24500",
        "--",
        "a2v2-train",
        environment=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert trace.is_file()


@pytest.mark.parametrize(
    ("environment_update", "arguments", "expected_error"),
    [
        ({"SLURM_JOB_NUM_NODES": None}, (), "SLURM_JOB_NUM_NODES"),
        ({"SLURM_GPUS_ON_NODE": "2"}, ("--gpus-per-node", "4"), "expected 4 GPUs"),
        ({"CUDA_VISIBLE_DEVICES": "0,1,2"}, (), "CUDA_VISIBLE_DEVICES exposes 3"),
    ],
)
def test_node_launcher_rejects_missing_or_inconsistent_topology(
    tmp_path: Path,
    environment_update: dict[str, str | None],
    arguments: tuple[str, ...],
    expected_error: str,
) -> None:
    """Catch partial, heterogeneous, or scheduler-mask-inconsistent nodes."""

    manifests, output, config = _launcher_inputs(tmp_path)
    environment = _slurm_environment()
    for name, value in environment_update.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value

    completed = _run_shell(
        NODE_LAUNCHER,
        "pretrain",
        *arguments,
        "--run-id", "invalid/pretrain",
        "--manifest-dir", str(manifests),
        "--output-dir", str(output),
        "--config", str(config),
        "--master-addr", "node01",
        "--master-port", "24500",
        "--",
        "a2v2-train",
        environment=environment,
    )

    assert completed.returncode == 2
    assert expected_error in completed.stderr


def test_slurm_dry_run_renders_separate_homogeneous_training_and_evaluation(
    tmp_path: Path,
) -> None:
    """Catch shared phase ownership, wrong rank math, quoting loss, or DDP evaluation."""

    manifests = tmp_path / "not present manifests"
    output = tmp_path / "not present output"
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("SLURM_") and name != "CUDA_VISIBLE_DEVICES"
    }

    completed = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--dry-run",
        "--nodes", "2",
        "--gpus-per-node", "4",
        "--job-id", "4815",
        "--master-addr", "node01",
        environment=environment,
    )

    assert completed.returncode == 0, completed.stderr
    commands = _rendered_commands(completed.stdout)
    training = [command for command in commands if str(NODE_LAUNCHER) in command]
    assert len(training) == 2
    pretraining = next(command for command in training if "pretrain" in command)
    finetuning = next(command for command in training if "finetune" in command)
    for command in training:
        assert "--nodes=2" in command
        assert "--ntasks=2" in command
        assert "--ntasks-per-node=1" in command
        assert "--gpus-per-node=4" in command
        assert "distributed_training.distributed_world_size=8" in command
        port = int(_option_value(command, "--master-port"))
        assert 15000 <= port <= 34999
    assert _option_value(pretraining, "--run-id") == "4815-meerkat-f0-100/pretrain"
    assert _option_value(finetuning, "--run-id") == "4815-meerkat-f0-100/finetune"
    assert _option_value(pretraining, "--output-dir") == str(output / "pretrain")
    assert _option_value(finetuning, "--output-dir") == str(output / "finetune")
    assert str(output / "pretrain/checkpoint_last.pt") in finetuning
    assert str(manifests) in pretraining
    assert str(manifests) in finetuning
    assert "CUDA_VISIBLE_DEVICES" not in completed.stdout

    evaluation = [command for command in commands if str(EVALUATOR) in command]
    assert len(evaluation) == 1
    assert "--nodes=1" in evaluation[0]
    assert "--ntasks=1" in evaluation[0]
    assert "--ntasks-per-node=1" in evaluation[0]
    assert "--gpus-per-node=1" in evaluation[0]
    assert "torch.distributed.run" not in evaluation[0]


def test_slurm_dry_run_resumes_each_stage_without_crossing_checkpoint_roles(
    tmp_path: Path,
) -> None:
    """Catch fine-tuning from its pretrained input or pretraining from fine-tune state."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "output"
    pretrain_checkpoint = output / "pretrain/checkpoint_last.pt"
    finetune_checkpoint = output / "finetune/checkpoint_last.pt"
    pretrain_checkpoint.parent.mkdir(parents=True)
    finetune_checkpoint.parent.mkdir(parents=True)
    pretrain_checkpoint.touch()
    finetune_checkpoint.touch()

    completed = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--dry-run",
        "--nodes", "1",
        "--gpus-per-node", "2",
        "--job-id", "9001",
        "--master-addr", "node01",
    )

    assert completed.returncode == 0, completed.stderr
    commands = _rendered_commands(completed.stdout)
    training = [command for command in commands if str(NODE_LAUNCHER) in command]
    pretraining = next(command for command in training if "pretrain" in command)
    finetuning = next(command for command in training if "finetune" in command)
    assert _option_value(pretraining, "--resume-checkpoint") == str(pretrain_checkpoint)
    assert "--pretrained-checkpoint" not in pretraining
    assert _option_value(finetuning, "--pretrained-checkpoint") == str(pretrain_checkpoint)
    assert _option_value(finetuning, "--resume-checkpoint") == str(finetune_checkpoint)
    assert pretraining[pretraining.index("--resume") + 1] == str(pretrain_checkpoint)
    assert finetuning[finetuning.index("--pretrained-checkpoint") + 1] == str(
        pretrain_checkpoint
    )
    assert finetuning[finetuning.index("--resume") + 1] == str(finetune_checkpoint)


def test_explicit_pretraining_handoff_owns_the_directory_pretraining_writes(
    tmp_path: Path,
) -> None:
    """Catch fine-tuning consuming a custom path that pretraining never owns."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "output"
    pretrain_checkpoint = output / "alternate pretrain/checkpoint_last.pt"

    completed = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--dry-run",
        "--nodes", "1",
        "--gpus-per-node", "2",
        "--job-id", "9001",
        "--master-addr", "node01",
        "--pretrain-checkpoint", str(pretrain_checkpoint),
    )

    assert completed.returncode == 0, completed.stderr
    commands = _rendered_commands(completed.stdout)
    training = [command for command in commands if str(NODE_LAUNCHER) in command]
    pretraining = next(command for command in training if "pretrain" in command)
    finetuning = next(command for command in training if "finetune" in command)
    assert _option_value(pretraining, "--output-dir") == str(pretrain_checkpoint.parent)
    assert f"checkpoint.save_dir={pretrain_checkpoint.parent}" in pretraining
    assert _option_value(finetuning, "--pretrained-checkpoint") == str(
        pretrain_checkpoint
    )


def test_pretraining_and_finetuning_cannot_share_one_output_owner(
    tmp_path: Path,
) -> None:
    """Catch two mathematical stages claiming one lock and checkpoint_last path."""

    output = tmp_path / "output"
    completed = _run_shell(
        SLURM_DRIVER,
        str(tmp_path / "manifests"),
        str(output),
        "--dry-run",
        "--pretrain-checkpoint", str(output / "finetune/checkpoint_last.pt"),
    )

    assert completed.returncode == 2
    assert "pretraining and fine-tuning output directories must be distinct" in (
        completed.stderr
    )


def test_slurm_real_launch_resolves_first_host_and_invokes_one_srun_task_per_node(
    tmp_path: Path,
) -> None:
    """Catch bypassing scontrol resolution or starting one launcher per GPU."""

    manifests, output, _ = _launcher_inputs(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    scheduler_trace = tmp_path / "scheduler.trace"
    fake_scontrol = fake_bin / "scontrol"
    fake_srun = fake_bin / "srun"
    fake_python = fake_bin / "python"
    _write_executable(
        fake_scontrol,
        """#!/usr/bin/env bash
printf 'scontrol:%s\n' "$*" >> "${A2V2_SCHEDULER_TRACE}"
printf 'node01\nnode02\n'
""",
    )
    _write_executable(
        fake_srun,
        """#!/usr/bin/env bash
printf 'srun:' >> "${A2V2_SCHEDULER_TRACE}"
printf ' %q' "$@" >> "${A2V2_SCHEDULER_TRACE}"
printf '\n' >> "${A2V2_SCHEDULER_TRACE}"
output=''
while (($#)); do
    if [[ "$1" == '--output-dir' ]]; then output="$2"; shift 2; else shift; fi
done
mkdir -p "$output"
touch "$output/checkpoint_last.pt"
""",
    )
    _write_executable(
        fake_python,
        """#!/usr/bin/env bash
printf 'python:' >> "${A2V2_SCHEDULER_TRACE}"
printf ' %q' "$@" >> "${A2V2_SCHEDULER_TRACE}"
printf '\n' >> "${A2V2_SCHEDULER_TRACE}"
""",
    )
    environment = {
        **_slurm_environment(),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_PYTHON": str(fake_python),
        "A2V2_SCHEDULER_TRACE": str(scheduler_trace),
    }

    completed = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--phase", "pretrain",
        environment=environment,
    )

    assert completed.returncode == 0, completed.stderr
    trace = scheduler_trace.read_text(encoding="utf-8")
    assert "scontrol:show hostnames node[01-02]" in trace
    srun_line = next(line for line in trace.splitlines() if line.startswith("srun:"))
    srun_command = shlex.split(srun_line.removeprefix("srun:"))
    assert "--nodes=2" in srun_command
    assert "--ntasks=2" in srun_command
    assert "--ntasks-per-node=1" in srun_command
    assert "--gpus-per-node=4" in srun_command
    assert _option_value(srun_command, "--master-addr") == "node01"


def test_evaluation_reuses_validated_finetune_marker_and_ignores_stale_best(
    tmp_path: Path,
) -> None:
    """Catch arbitrary checkpoint_best selection or trust in a stale/partial marker."""

    manifests, output, _ = _launcher_inputs(tmp_path)
    (output / "pretrain").mkdir()
    (output / "pretrain/checkpoint_last.pt").write_bytes(b"pretrained")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    trace = tmp_path / "srun.trace"
    _write_executable(
        fake_bin / "scontrol",
        "#!/usr/bin/env bash\nprintf 'node01\n'\n",
    )
    _write_executable(
        fake_bin / "python",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    _write_executable(
        fake_bin / "srun",
        """#!/usr/bin/env bash
printf 'srun:' >> "${A2V2_SRUN_TRACE}"
printf ' %q' "$@" >> "${A2V2_SRUN_TRACE}"
printf '\n' >> "${A2V2_SRUN_TRACE}"
output=''
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    if [[ "${arguments[index]}" == '--output-dir' ]]; then
        output="${arguments[index + 1]}"
    fi
done
if [[ -n "${output}" ]]; then
    mkdir -p "${output}"
    printf 'fine checkpoint\n' > "${output}/checkpoint_last.pt"
fi
""",
    )
    environment = {
        **_slurm_environment(nodes=1, gpus_per_node=1),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_PYTHON": str(fake_bin / "python"),
        "A2V2_SRUN_TRACE": str(trace),
    }

    trained = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--phase", "finetune",
        environment=environment,
    )
    assert trained.returncode == 0, trained.stderr
    marker = output / "finetune/checkpoint_last.pt.stage-complete"
    assert json.loads(marker.read_text(encoding="utf-8"))["schema"] == (
        "a2v2.stage-completion.v1"
    )
    stale_best = output / "finetune/checkpoint_best.pt"
    stale_best.write_bytes(b"stale best from another run")

    trace.write_text("", encoding="utf-8")
    evaluated = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--phase", "evaluate",
        environment=environment,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    evaluation_command = shlex.split(
        trace.read_text(encoding="utf-8").strip().removeprefix("srun:")
    )
    assert _option_value(evaluation_command, "--checkpoint") == str(
        output / "finetune/checkpoint_last.pt"
    )
    assert str(stale_best) not in evaluation_command

    trace.write_text("", encoding="utf-8")
    mismatched_environment = {
        **environment,
        "A2V2_FINETUNE_MAX_TOKENS": "123",
    }
    mismatched = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--phase", "evaluate",
        environment=mismatched_environment,
    )
    assert mismatched.returncode == 2
    assert "malformed, mismatched, or stale" in mismatched.stderr
    assert trace.read_text(encoding="utf-8") == ""

    marker.write_text('{"schema":', encoding="utf-8")
    partial = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--phase", "evaluate",
        environment=environment,
    )
    assert partial.returncode == 2
    assert "malformed, mismatched, or stale" in partial.stderr
    assert trace.read_text(encoding="utf-8") == ""


def test_slurm_driver_rejects_a_heterogeneous_gpu_allocation_before_srun(
    tmp_path: Path,
) -> None:
    """Catch a world-size claim assembled from unequal local worker counts."""

    manifests, output, _ = _launcher_inputs(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    srun_trace = tmp_path / "srun.trace"
    _write_executable(
        fake_bin / "srun",
        "#!/usr/bin/env bash\ntouch \"${A2V2_SRUN_TRACE}\"\n",
    )
    environment = {
        **_slurm_environment(),
        "SLURM_GPUS_PER_NODE": "gpu:4,gpu:2",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_SRUN_TRACE": str(srun_trace),
    }

    completed = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--phase", "pretrain",
        environment=environment,
    )

    assert completed.returncode == 2
    assert "homogeneous" in completed.stderr
    assert "gpu:4,gpu:2" in completed.stderr
    assert not srun_trace.exists()


@pytest.mark.parametrize(
    ("step_exit", "validation_exit", "write_checkpoint", "expected_exit"),
    [
        (0, 0, True, 75),
        (75, 0, True, 75),
        (1, 0, True, 75),
        (75, 1, True, 2),
        (1, 0, False, 2),
    ],
)
def test_sigusr1_is_forwarded_and_exit_75_requires_checkpoint_validation(
    tmp_path: Path,
    step_exit: int,
    validation_exit: int,
    write_checkpoint: bool,
    expected_exit: int,
) -> None:
    """Catch lost preemption signals or a requeue-friendly exit without valid state."""

    manifests, output, _ = _launcher_inputs(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ready = tmp_path / "srun.ready"
    events = tmp_path / "events.trace"
    scheduler_trace = tmp_path / "scheduler.trace"
    _write_executable(
        fake_bin / "scontrol",
        """#!/usr/bin/env bash
printf 'scontrol:%s\n' "$*" >> "${A2V2_SCHEDULER_TRACE}"
printf 'node01\n'
""",
    )
    _write_executable(
        fake_bin / "srun",
        """#!/usr/bin/env bash
output=''
while (($#)); do
    if [[ "$1" == '--output-dir' ]]; then output="$2"; shift 2; else shift; fi
done
on_usr1() {
    printf 'signal\n' >> "${A2V2_EVENTS}"
    if [[ "${A2V2_WRITE_CHECKPOINT}" == true ]]; then
        mkdir -p "$output"
        touch "$output/checkpoint_last.pt"
    fi
    exit "${A2V2_STEP_EXIT}"
}
trap on_usr1 USR1
touch "${A2V2_READY}"
while true; do sleep 0.05; done
""",
    )
    _write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
printf 'validate\n' >> "${A2V2_EVENTS}"
exit "${A2V2_VALIDATION_EXIT}"
""",
    )
    environment = {
        **_slurm_environment(nodes=1, gpus_per_node=1),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_PYTHON": str(fake_bin / "python"),
        "A2V2_READY": str(ready),
        "A2V2_EVENTS": str(events),
        "A2V2_SCHEDULER_TRACE": str(scheduler_trace),
        "A2V2_STEP_EXIT": str(step_exit),
        "A2V2_VALIDATION_EXIT": str(validation_exit),
        "A2V2_WRITE_CHECKPOINT": str(write_checkpoint).lower(),
    }
    process = subprocess.Popen(
        [
            "bash",
            str(SLURM_DRIVER),
            str(manifests),
            str(output),
            "--phase", "pretrain",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), process.communicate(timeout=2)

    os.kill(process.pid, signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == expected_exit, stdout + stderr
    expected_events = ["signal", "validate"] if write_checkpoint else ["signal"]
    assert events.read_text(encoding="utf-8").splitlines() == expected_events
    assert "requeue" not in scheduler_trace.read_text(encoding="utf-8")
    if validation_exit:
        assert "refusing requeue-friendly exit 75" in stderr
    if not write_checkpoint:
        assert "did not publish a new checkpoint" in stderr


def test_sigusr1_before_launch_is_latched_until_safe_boundary(
    tmp_path: Path,
) -> None:
    """Catch a request received during hashing being cleared before srun."""

    manifests, output, _ = _launcher_inputs(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ready = tmp_path / "contract.ready"
    release = tmp_path / "contract.release"
    once = tmp_path / "contract.once"
    srun_trace = tmp_path / "srun.trace"
    real_python = shlex.quote(sys.executable)
    _write_executable(
        fake_bin / "contract-python",
        f"""#!/usr/bin/env bash
set -eu
if [[ ! -e "${{A2V2_ONCE}}" ]]; then
    touch "${{A2V2_ONCE}}"
    touch "${{A2V2_READY}}"
    while [[ ! -e "${{A2V2_RELEASE}}" ]]; do sleep 0.02; done
fi
exec {real_python} "$@"
""",
    )
    _write_executable(
        fake_bin / "srun",
        "#!/usr/bin/env bash\ntouch \"${A2V2_SRUN_TRACE}\"\n",
    )
    environment = {
        **_slurm_environment(nodes=1, gpus_per_node=1),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_CONTRACT_PYTHON": str(fake_bin / "contract-python"),
        "A2V2_ONCE": str(once),
        "A2V2_READY": str(ready),
        "A2V2_RELEASE": str(release),
        "A2V2_SRUN_TRACE": str(srun_trace),
    }
    process = subprocess.Popen(
        [
            "bash",
            str(SLURM_DRIVER),
            str(manifests),
            str(output),
            "--phase", "pretrain",
            "--master-addr", "node01",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), process.communicate(timeout=2)

    os.kill(process.pid, signal.SIGUSR1)
    release.touch()
    stdout, stderr = process.communicate(timeout=20)

    assert process.returncode == 75, stdout + stderr
    assert "safe phase boundary" in stderr
    assert not srun_trace.exists()


def test_sigusr1_during_phase_handoff_keeps_completed_marker_and_stops(
    tmp_path: Path,
) -> None:
    """Catch a handoff signal being discarded before the next training phase."""

    manifests, output, _ = _launcher_inputs(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ready = tmp_path / "marker.ready"
    release = tmp_path / "marker.release"
    srun_trace = tmp_path / "srun.trace"
    real_python = shlex.quote(sys.executable)
    _write_executable(
        fake_bin / "contract-python",
        f"""#!/usr/bin/env bash
set -eu
if [[ " $* " == *" write-completion "* ]]; then
    {real_python} "$@"
    touch "${{A2V2_READY}}"
    while [[ ! -e "${{A2V2_RELEASE}}" ]]; do sleep 0.02; done
    exit 0
fi
exec {real_python} "$@"
""",
    )
    _write_executable(
        fake_bin / "python",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    _write_executable(
        fake_bin / "srun",
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${A2V2_SRUN_TRACE}"
output=''
while (($#)); do
    if [[ "$1" == '--output-dir' ]]; then output="$2"; shift 2; else shift; fi
done
mkdir -p "$output"
printf 'checkpoint\n' > "$output/checkpoint_last.pt"
""",
    )
    environment = {
        **_slurm_environment(nodes=1, gpus_per_node=1),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_PYTHON": str(fake_bin / "python"),
        "A2V2_CONTRACT_PYTHON": str(fake_bin / "contract-python"),
        "A2V2_READY": str(ready),
        "A2V2_RELEASE": str(release),
        "A2V2_SRUN_TRACE": str(srun_trace),
    }
    process = subprocess.Popen(
        [
            "bash",
            str(SLURM_DRIVER),
            str(manifests),
            str(output),
            "--phase", "all",
            "--master-addr", "node01",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 15
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), process.communicate(timeout=2)

    os.kill(process.pid, signal.SIGUSR1)
    release.touch()
    stdout, stderr = process.communicate(timeout=20)

    assert process.returncode == 75, stdout + stderr
    assert (output / "pretrain/checkpoint_last.pt.stage-complete").is_file()
    launches = srun_trace.read_text(encoding="utf-8").splitlines()
    assert len(launches) == 1
    assert " pretrain " in f" {launches[0]} "
    assert " finetune " not in f" {launches[0]} "


def test_explicit_output_lock_recovery_is_fingerprinted_and_rendered_first(
    tmp_path: Path,
) -> None:
    """Catch automatic lock stealing or recovery after a new launch has begun."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "output"
    fingerprint = "a" * 64

    completed = _run_shell(
        SLURM_DRIVER,
        str(manifests),
        str(output),
        "--dry-run",
        "--phase", "pretrain",
        "--nodes", "1",
        "--gpus-per-node", "1",
        "--job-id", "4815",
        "--master-addr", "node01",
        "--recover-lock", f"pretrain:{fingerprint}",
    )

    assert completed.returncode == 0, completed.stderr
    commands = _rendered_commands(completed.stdout)
    recovery_index = next(
        index for index, command in enumerate(commands)
        if str(output / "pretrain") in command and fingerprint in command
    )
    launch_index = next(
        index for index, command in enumerate(commands)
        if str(NODE_LAUNCHER) in command
    )
    assert recovery_index < launch_index


def test_output_lock_recovery_rejects_an_unfingerprinted_request(tmp_path: Path) -> None:
    """Catch weakening explicit stale-owner recovery to a phase-only switch."""

    completed = _run_shell(
        SLURM_DRIVER,
        str(tmp_path / "manifests"),
        str(tmp_path / "output"),
        "--dry-run",
        "--recover-lock", "pretrain:not-a-fingerprint",
    )

    assert completed.returncode == 2
    assert "64 lowercase hexadecimal" in completed.stderr


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
