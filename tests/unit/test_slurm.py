"""Pure contracts for torchrun and SLURM-safe training runtime behavior."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import signal
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread, current_thread

import pytest
import numpy as np
import torch


def test_slurm_runtime_module_is_available() -> None:
    """Catch packaging that omits the runtime contract module."""

    assert importlib.util.find_spec("a2v2.slurm") is not None


def _slurm_environment() -> dict[str, str]:
    """Return one hand-checked homogeneous two-node allocation."""

    return {
        "SLURM_JOB_ID": "4815",
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_NODEID": "1",
        "SLURM_GPUS_ON_NODE": "4",
        "SLURM_JOB_NODELIST": "gpu[07-08]",
    }


def test_slurm_environment_parses_positive_allocation_fields() -> None:
    """Catch stringly typed or incomplete allocation parsing."""

    from a2v2.slurm import SlurmEnvironment

    allocation = SlurmEnvironment.from_mapping(_slurm_environment())

    assert allocation.job_id == "4815"
    assert allocation.node_count == 2
    assert allocation.node_id == 1
    assert allocation.processes_per_node == 4
    assert allocation.world_size == 8
    assert allocation.node_list == "gpu[07-08]"


@pytest.mark.parametrize(
    "missing",
    (
        "SLURM_JOB_ID",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NODEID",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_NODELIST",
    ),
)
def test_slurm_environment_names_each_missing_required_variable(missing: str) -> None:
    """Catch launch errors that omit the actionable scheduler variable name."""

    from a2v2.slurm import SlurmEnvironment, TopologyError

    environment = _slurm_environment()
    environment.pop(missing)

    with pytest.raises(TopologyError, match=missing):
        SlurmEnvironment.from_mapping(environment)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("SLURM_JOB_NUM_NODES", "0"),
        ("SLURM_NODEID", "2"),
        ("SLURM_GPUS_ON_NODE", "gpu:a100:4"),
    ),
)
def test_slurm_environment_rejects_invalid_allocation_arithmetic(
    field: str,
    value: str,
) -> None:
    """Catch invalid node bounds and ambiguous per-node process counts."""

    from a2v2.slurm import SlurmEnvironment, TopologyError

    environment = _slurm_environment()
    environment[field] = value

    with pytest.raises(TopologyError, match=field):
        SlurmEnvironment.from_mapping(environment)


def test_homogeneous_world_size_rejects_heterogeneous_node_counts() -> None:
    """Catch arithmetic that silently multiplies one node's GPU count."""

    from a2v2.slurm import TopologyError, homogeneous_world_size

    assert homogeneous_world_size((4, 4), node_count=2) == 8
    with pytest.raises(TopologyError, match="homogeneous"):
        homogeneous_world_size((4, 3), node_count=2)
    with pytest.raises(TopologyError, match="2 node"):
        homogeneous_world_size((4,), node_count=2)


def test_rendezvous_uses_resolved_first_host_and_phase_specific_id() -> None:
    """Catch shell-bound host discovery or cross-phase rendezvous collisions."""

    from a2v2.slurm import SlurmEnvironment, build_rendezvous

    seen: list[str] = []

    def resolve_hosts(expression: str) -> tuple[str, ...]:
        """Resolve the scheduler expression without executing a shell command."""

        seen.append(expression)
        return ("gpu07", "gpu08")

    rendezvous = build_rendezvous(
        SlurmEnvironment.from_mapping(_slurm_environment()),
        phase="pretrain",
        port=29400,
        resolve_hosts=resolve_hosts,
    )

    assert seen == ["gpu[07-08]"]
    assert rendezvous.endpoint == "gpu07:29400"
    assert rendezvous.rendezvous_id == "4815-pretrain"


def test_rendezvous_override_and_phase_validation_are_actionable() -> None:
    """Catch endpoint rewriting and unsafe phase identifiers."""

    from a2v2.slurm import SlurmEnvironment, TopologyError, build_rendezvous

    allocation = SlurmEnvironment.from_mapping(_slurm_environment())
    rendezvous = build_rendezvous(
        allocation,
        phase="finetune_025",
        port=29400,
        resolve_hosts=lambda _: (_ for _ in ()).throw(
            AssertionError("override must not resolve hosts")
        ),
        explicit_endpoint="rdzv.internal:30001",
    )
    assert rendezvous.endpoint == "rdzv.internal:30001"
    assert rendezvous.rendezvous_id == "4815-finetune_025"

    with pytest.raises(TopologyError, match="phase"):
        build_rendezvous(
            allocation,
            phase="../collision",
            port=29400,
            resolve_hosts=lambda _: ("gpu07",),
        )


def test_run_contract_fingerprint_is_canonical_and_diff_names_fields(
    tmp_path: Path,
) -> None:
    """Catch unstable fingerprints and opaque distributed contract mismatch errors."""

    from a2v2.slurm import RunContract, contract_differences

    expected = RunContract(
        job_id="4815",
        phase="pretrain",
        node_count=2,
        processes_per_node=4,
        world_size=8,
        rendezvous_endpoint="gpu07:29400",
        rendezvous_id="4815-pretrain",
        output_directory=str(tmp_path / "run"),
        manifest_fingerprint="sha256:abc",
    )
    same = RunContract.from_mapping(dict(reversed(tuple(expected.to_mapping().items()))))
    actual = RunContract.from_mapping({
        **expected.to_mapping(),
        "world_size": 4,
        "manifest_fingerprint": "sha256:def",
    })

    assert same.fingerprint() == expected.fingerprint()
    assert contract_differences(expected, actual) == {
        "manifest_fingerprint": ("sha256:abc", "sha256:def"),
        "world_size": (8, 4),
    }


def _completion_contract(tmp_path: Path, *, phase: str = "pretrain") -> object:
    """Build one stage-specific canonical contract for marker tests."""

    from a2v2.slurm import RunContract

    return RunContract(
        job_id="4815",
        phase=phase,
        node_count=2,
        processes_per_node=4,
        world_size=8,
        rendezvous_endpoint="gpu07:29400",
        rendezvous_id=f"4815-{phase}",
        output_directory=str(tmp_path / phase),
        manifest_fingerprint=f"sha256:{phase}-manifest",
    )


def test_stage_completion_round_trip_binds_exact_contract_and_checkpoint(
    tmp_path: Path,
) -> None:
    """Catch existence-only markers or checkpoint paths without byte identity."""

    from a2v2.slurm import (
        COMPLETION_MARKER_SCHEMA,
        checkpoint_identity,
        validate_stage_completion,
        write_stage_completion,
    )

    contract = _completion_contract(tmp_path)
    checkpoint = tmp_path / "pretrain" / "checkpoint_last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint-one")
    marker = checkpoint.with_suffix(checkpoint.suffix + ".stage-complete")

    write_stage_completion(
        marker,
        contract=contract,
        config_fingerprint="config-fingerprint",
        checkpoint_path=checkpoint,
    )

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["schema"] == COMPLETION_MARKER_SCHEMA
    assert payload["run_contract"] == contract.to_mapping()
    assert payload["run_contract_fingerprint"] == contract.fingerprint()
    assert payload["config_fingerprint"] == "config-fingerprint"
    assert payload["checkpoint"] == checkpoint_identity(checkpoint).to_mapping()
    assert validate_stage_completion(
        marker,
        expected_contract=contract,
        expected_config_fingerprint="config-fingerprint",
        expected_checkpoint_path=checkpoint,
    ) == checkpoint.resolve()


@pytest.mark.parametrize(
    "mutation",
    ("partial", "contract", "config", "checkpoint-path", "checkpoint-bytes"),
)
def test_stage_completion_rejects_partial_mismatched_or_stale_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Require explicit action for every untrusted completed-stage record."""

    from a2v2.slurm import (
        CompletionMarkerError,
        RunContract,
        validate_stage_completion,
        write_stage_completion,
    )

    contract = _completion_contract(tmp_path)
    checkpoint = tmp_path / "pretrain" / "checkpoint_last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint-one")
    marker = checkpoint.with_suffix(checkpoint.suffix + ".stage-complete")
    write_stage_completion(
        marker,
        contract=contract,
        config_fingerprint="config-fingerprint",
        checkpoint_path=checkpoint,
    )
    expected = contract
    config_fingerprint = "config-fingerprint"
    expected_checkpoint = checkpoint
    if mutation == "partial":
        marker.write_text('{"schema":', encoding="utf-8")
    elif mutation == "contract":
        expected = RunContract.from_mapping({
            **contract.to_mapping(),
            "manifest_fingerprint": "sha256:new-manifest",
        })
    elif mutation == "config":
        config_fingerprint = "new-config-fingerprint"
    elif mutation == "checkpoint-path":
        expected_checkpoint = tmp_path / "different.pt"
    else:
        checkpoint.write_bytes(b"checkpoint-two")

    with pytest.raises(CompletionMarkerError):
        validate_stage_completion(
            marker,
            expected_contract=expected,
            expected_config_fingerprint=config_fingerprint,
            expected_checkpoint_path=expected_checkpoint,
        )


def test_stage_completion_atomic_write_interruption_preserves_old_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch truncate-in-place updates that leave a trusted partial marker."""

    from a2v2.slurm import write_stage_completion

    contract = _completion_contract(tmp_path)
    checkpoint = tmp_path / "pretrain" / "checkpoint_last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint-one")
    marker = checkpoint.with_suffix(checkpoint.suffix + ".stage-complete")
    marker.write_bytes(b"old-complete-record\n")

    def interrupted_replace(_source: object, _destination: object) -> None:
        """Simulate power loss after a durable candidate but before replacement."""

        raise OSError("injected interruption before atomic replace")

    monkeypatch.setattr(os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="injected interruption"):
        write_stage_completion(
            marker,
            contract=contract,
            config_fingerprint="config-fingerprint",
            checkpoint_path=checkpoint,
        )

    assert marker.read_bytes() == b"old-complete-record\n"
    assert not tuple(marker.parent.glob("." + marker.name + ".*.candidate"))


@pytest.mark.parametrize("rank", (0, 1))
def test_launcher_contract_mismatch_fails_early_on_every_rank(
    tmp_path: Path,
    rank: int,
) -> None:
    """Catch rank-zero-only or advisory validation of exported launch state."""

    from a2v2.slurm import (
        DistributedEnvironment,
        SlurmEnvironment,
        TopologyError,
        config_fingerprint,
        validate_launcher_contract,
    )

    contract = _completion_contract(tmp_path)
    config = {
        "stage": "pretrain",
        "checkpoint": {"save_dir": str(tmp_path / "pretrain")},
    }
    environment = {
        **_slurm_environment(),
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NODEID": "0",
        "SLURM_GPUS_ON_NODE": "2",
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "A2V2_RUN_ID": "4815/pretrain",
        "A2V2_SLURM_PHASE": "pretrain",
        "A2V2_SLURM_WORLD_SIZE": "8",
        "A2V2_SLURM_RDZV_ENDPOINT": "gpu07:29400",
        "A2V2_SLURM_RDZV_ID": "4815-pretrain",
        "A2V2_SLURM_MANIFEST_FINGERPRINT": contract.manifest_fingerprint,
        "A2V2_SLURM_RUN_CONTRACT": json.dumps(contract.to_mapping()),
        "A2V2_SLURM_CONTRACT_FINGERPRINT": "0" * 64,
        "A2V2_SLURM_CONFIG_FINGERPRINT": config_fingerprint(config),
    }

    with pytest.raises(TopologyError, match="fingerprint"):
        validate_launcher_contract(
            environment,
            config=config,
            slurm=SlurmEnvironment.from_mapping(environment),
            distributed=DistributedEnvironment.from_mapping(environment),
        )


def test_training_consumes_launcher_contract_before_model_or_output_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a validated helper that the real training path never invokes."""

    from dataclasses import replace

    import a2v2.workflows as workflows
    from a2v2.config import config_to_dict, load_config
    from a2v2.slurm import RunContract, TopologyError, config_fingerprint

    config = load_config(
        Path(__file__).resolve().parents[2]
        / "configs/MeerKAT/a2v_large_pretrain_best.yaml"
    )
    config = replace(
        config,
        checkpoint=replace(config.checkpoint, save_dir=tmp_path / "pretrain"),
        distributed=replace(config.distributed, requested_world_size=1),
    )
    contract = RunContract(
        job_id="4815",
        phase="pretrain",
        node_count=1,
        processes_per_node=1,
        world_size=1,
        rendezvous_endpoint="gpu07:29400",
        rendezvous_id="4815-pretrain",
        output_directory=str(tmp_path / "pretrain"),
        manifest_fingerprint="sha256:manifest",
    )
    launcher_environment = {
        "A2V2_RUN_ID": "4815/pretrain",
        "A2V2_SLURM_PHASE": "pretrain",
        "A2V2_SLURM_WORLD_SIZE": "1",
        "A2V2_SLURM_RDZV_ENDPOINT": contract.rendezvous_endpoint,
        "A2V2_SLURM_RDZV_ID": contract.rendezvous_id,
        "A2V2_SLURM_MANIFEST_FINGERPRINT": contract.manifest_fingerprint,
        "A2V2_SLURM_RUN_CONTRACT": json.dumps(contract.to_mapping()),
        "A2V2_SLURM_CONTRACT_FINGERPRINT": "0" * 64,
        "A2V2_SLURM_CONFIG_FINGERPRINT": config_fingerprint(config_to_dict(config)),
    }
    for name, value in launcher_environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        workflows,
        "_distributed_device",
        lambda _: (torch.device("cpu"), 0, 1, False),
    )
    monkeypatch.setattr(
        workflows,
        "_make_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model construction must not run")
        ),
    )

    with pytest.raises(TopologyError, match="fingerprint"):
        workflows._run_training(
            config,
            device_name="cpu",
            resume_path=None,
            pretrained_checkpoint=None,
        )


def test_runtime_recomputes_active_phase_manifest_identity(tmp_path: Path) -> None:
    """Catch an exported self-consistent contract after shared data changed."""

    from a2v2.slurm import (
        DistributedEnvironment,
        RunContract,
        TopologyError,
        config_fingerprint,
        manifest_fingerprint,
        validate_launcher_contract,
    )

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    pretrain_manifest = manifests / "pretrain.tsv"
    pretrain_manifest.write_text("first\n", encoding="utf-8")
    output = tmp_path / "pretrain"
    config = {
        "stage": "pretrain",
        "task": {"data": str(manifests)},
        "dataset": {"train_subset": "train", "valid_subset": "valid"},
        "checkpoint": {"save_dir": str(output)},
    }
    contract = RunContract(
        job_id="4815",
        phase="pretrain",
        node_count=1,
        processes_per_node=1,
        world_size=1,
        rendezvous_endpoint="gpu07:29400",
        rendezvous_id="4815-pretrain",
        output_directory=str(output),
        manifest_fingerprint=manifest_fingerprint(
            (("pretrain.tsv", pretrain_manifest),)
        ),
    )
    environment = {
        "A2V2_RUN_ID": "4815/pretrain",
        "A2V2_SLURM_PHASE": "pretrain",
        "A2V2_SLURM_WORLD_SIZE": "1",
        "A2V2_SLURM_RDZV_ENDPOINT": contract.rendezvous_endpoint,
        "A2V2_SLURM_RDZV_ID": contract.rendezvous_id,
        "A2V2_SLURM_MANIFEST_FINGERPRINT": contract.manifest_fingerprint,
        "A2V2_SLURM_RUN_CONTRACT": json.dumps(contract.to_mapping()),
        "A2V2_SLURM_CONTRACT_FINGERPRINT": contract.fingerprint(),
        "A2V2_SLURM_CONFIG_FINGERPRINT": config_fingerprint(config),
    }
    assert validate_launcher_contract(
        environment,
        config=config,
        slurm=None,
        distributed=DistributedEnvironment.from_mapping({}),
    ) == contract

    pretrain_manifest.write_text("changed\n", encoding="utf-8")
    with pytest.raises(TopologyError, match="manifest fingerprint"):
        validate_launcher_contract(
            environment,
            config=config,
            slurm=None,
            distributed=DistributedEnvironment.from_mapping({}),
        )


def _torchrun_environment() -> dict[str, str]:
    """Return rank five on the second homogeneous four-worker node."""

    return {
        "RANK": "5",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "4",
    }


def test_distributed_environment_preserves_ordinary_single_process_defaults() -> None:
    """Catch strict launch parsing that breaks the established local commands."""

    from a2v2.slurm import DistributedEnvironment

    assert DistributedEnvironment.from_mapping({}) == DistributedEnvironment(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
    )


def test_distributed_environment_parses_complete_torchrun_topology() -> None:
    """Catch rank/local-rank swaps or missing local world-size tracking."""

    from a2v2.slurm import DistributedEnvironment

    topology = DistributedEnvironment.from_mapping(_torchrun_environment())

    assert topology.rank == 5
    assert topology.local_rank == 1
    assert topology.world_size == 8
    assert topology.local_world_size == 4
    assert topology.node_rank == 1


@pytest.mark.parametrize("missing", ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"))
def test_partial_torchrun_environment_names_missing_variable(missing: str) -> None:
    """Catch partial torchrun launches that previously fell through to defaults."""

    from a2v2.slurm import DistributedEnvironment, TopologyError

    environment = _torchrun_environment()
    environment.pop(missing)

    with pytest.raises(TopologyError, match=missing):
        DistributedEnvironment.from_mapping(environment)


def test_runtime_topology_matches_slurm_node_and_cuda_mapping() -> None:
    """Catch rank arithmetic that maps a worker onto the wrong SLURM node or GPU."""

    from a2v2.slurm import (
        DistributedEnvironment,
        SlurmEnvironment,
        validate_runtime_topology,
    )

    validate_runtime_topology(
        DistributedEnvironment.from_mapping(_torchrun_environment()),
        slurm=SlurmEnvironment.from_mapping(_slurm_environment()),
        cuda_requested=True,
        visible_cuda_devices=4,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"RANK": "0", "LOCAL_RANK": "0"}, "SLURM_NODEID"),
        ({"WORLD_SIZE": "4", "RANK": "1"}, "WORLD_SIZE"),
        ({"LOCAL_WORLD_SIZE": "2", "RANK": "3"}, "LOCAL_WORLD_SIZE"),
    ),
)
def test_runtime_topology_rejects_slurm_arithmetic_mismatch(
    mutation: dict[str, str],
    message: str,
) -> None:
    """Catch inconsistent global, local, and scheduler launch dimensions."""

    from a2v2.slurm import (
        DistributedEnvironment,
        SlurmEnvironment,
        TopologyError,
        validate_runtime_topology,
    )

    environment = {**_torchrun_environment(), **mutation}
    with pytest.raises(TopologyError, match=message):
        validate_runtime_topology(
            DistributedEnvironment.from_mapping(environment),
            slurm=SlurmEnvironment.from_mapping(_slurm_environment()),
            cuda_requested=False,
            visible_cuda_devices=0,
        )


def test_runtime_topology_rejects_invalid_cuda_cardinality_and_ordinal() -> None:
    """Catch oversubscribed local workers before torch.cuda.set_device runs."""

    from a2v2.slurm import (
        DistributedEnvironment,
        TopologyError,
        validate_runtime_topology,
    )

    topology = DistributedEnvironment.from_mapping(_torchrun_environment())
    with pytest.raises(TopologyError, match="visible CUDA"):
        validate_runtime_topology(
            topology,
            slurm=None,
            cuda_requested=True,
            visible_cuda_devices=3,
        )

    invalid_ordinal = DistributedEnvironment.from_mapping({
        **_torchrun_environment(),
        "RANK": "7",
        "LOCAL_RANK": "3",
        "LOCAL_WORLD_SIZE": "4",
    })
    with pytest.raises(TopologyError, match="LOCAL_RANK"):
        validate_runtime_topology(
            invalid_ordinal,
            slurm=None,
            cuda_requested=True,
            visible_cuda_devices=3,
        )


def test_local_single_gpu_launch_may_leave_other_devices_visible() -> None:
    """Catch SLURM-only equality rules leaking into ordinary local CUDA runs."""

    from a2v2.slurm import DistributedEnvironment, validate_runtime_topology

    validate_runtime_topology(
        DistributedEnvironment.from_mapping({}),
        slurm=None,
        cuda_requested=True,
        visible_cuda_devices=8,
    )


def _rank_topology(rank: int, *, hostname: str = "gpu07") -> object:
    """Build one hand-derived two-rank local topology record."""

    from a2v2.slurm import RankTopology

    return RankTopology(
        hostname=hostname,
        global_rank=rank,
        local_rank=rank,
        local_world_size=2,
        visible_cuda_devices=2,
        selected_cuda_device=rank,
        cuda_device_name="NVIDIA A100-SXM4-40GB",
        cuda_device_uuid=f"GPU-rank-{rank}",
    )


def test_topology_state_is_json_safe_and_indexed_by_global_rank() -> None:
    """Catch host-order inference or device objects leaking into checkpoints."""

    from a2v2.slurm import TOPOLOGY_SCHEMA, build_topology_state

    state = build_topology_state((_rank_topology(0), _rank_topology(1)))

    assert state["schema"] == TOPOLOGY_SCHEMA == "a2v2.topology.v1"
    assert state["world_size"] == 2
    assert [item["global_rank"] for item in state["by_rank"]] == [0, 1]
    assert json.loads(json.dumps(state)) == state


def test_topology_state_rejects_duplicate_or_noncontiguous_global_ranks() -> None:
    """Catch malformed topology arrays that could restore the wrong rank state."""

    from a2v2.slurm import TopologyError, build_topology_state

    with pytest.raises(TopologyError, match="global ranks"):
        build_topology_state((_rank_topology(0), _rank_topology(0)))


def test_resume_topology_fails_on_world_mismatch_and_warns_on_hardware_change() -> None:
    """Catch world-size drift while permitting explicit hardware equivalence warnings."""

    from a2v2.slurm import (
        TopologyError,
        build_topology_state,
        validate_resume_topology,
    )

    saved = build_topology_state((_rank_topology(0), _rank_topology(1)))
    with pytest.raises(TopologyError, match="world size 2"):
        validate_resume_topology(saved, current=_rank_topology(0), world_size=1)

    warnings = validate_resume_topology(
        saved,
        current=_rank_topology(0, hostname="replacement-host"),
        world_size=2,
    )
    assert warnings == (
        "rank 0 topology differs at hostname: checkpoint='gpu07', current='replacement-host'",
    )


def _lock_owner(run_id: str, *, pid: int = 1001) -> object:
    """Build one deterministic scheduler lock identity."""

    from a2v2.slurm import LockOwner

    return LockOwner(
        job_id="4815",
        run_id=run_id,
        hostname="login01",
        pid=pid,
    )


def test_output_lock_writes_complete_owner_and_releases_only_its_inode(
    tmp_path: Path,
) -> None:
    """Catch non-atomic lock content and ownership-blind cleanup."""

    from a2v2.slurm import OUTPUT_LOCK_NAME, OUTPUT_LOCK_SCHEMA, OutputLock

    output_directory = tmp_path / "run"
    output_directory.mkdir()
    owner = _lock_owner("pretrain")
    lock = OutputLock.acquire(output_directory, owner)
    payload = json.loads((output_directory / OUTPUT_LOCK_NAME).read_text())

    assert payload == {
        "schema": OUTPUT_LOCK_SCHEMA,
        "job_id": "4815",
        "run_id": "pretrain",
        "hostname": "login01",
        "pid": 1001,
    }

    lock.release()
    assert not (output_directory / OUTPUT_LOCK_NAME).exists()


def test_output_lock_contention_reports_existing_and_requested_owners(
    tmp_path: Path,
) -> None:
    """Catch a second job silently sharing the checkpoint temporary path."""

    from a2v2.slurm import OutputLock, OutputLockError

    output_directory = tmp_path / "run"
    output_directory.mkdir()
    first = OutputLock.acquire(output_directory, _lock_owner("pretrain"))
    try:
        with pytest.raises(OutputLockError, match="pretrain.*finetune"):
            OutputLock.acquire(
                output_directory,
                _lock_owner("finetune", pid=1002),
            )
    finally:
        first.release()


def test_output_lock_concurrent_acquisition_has_exactly_one_winner(
    tmp_path: Path,
) -> None:
    """Catch check-then-create races that grant two writers ownership."""

    from a2v2.slurm import OutputLock, OutputLockError

    output_directory = tmp_path / "run"
    output_directory.mkdir()

    def acquire(run_id: str, pid: int) -> object:
        """Return either the acquired real lock or its contention error."""

        try:
            return OutputLock.acquire(
                output_directory,
                _lock_owner(run_id, pid=pid),
            )
        except OutputLockError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(
            lambda arguments: acquire(*arguments),
            (("first", 2001), ("second", 2002)),
        ))

    winners = [item for item in outcomes if isinstance(item, OutputLock)]
    losers = [item for item in outcomes if isinstance(item, OutputLockError)]
    assert len(winners) == len(losers) == 1
    winners[0].release()


def test_stale_output_lock_requires_explicit_fingerprinted_recovery(
    tmp_path: Path,
) -> None:
    """Catch implicit stale takeover and recovery of a different live owner."""

    from a2v2.slurm import (
        OUTPUT_LOCK_NAME,
        OutputLock,
        OutputLockError,
        recover_output_lock,
    )

    output_directory = tmp_path / "run"
    output_directory.mkdir()
    stale_owner = _lock_owner("pretrain", pid=999_999)
    stale_lock = OutputLock.acquire(output_directory, stale_owner)
    requested_owner = _lock_owner("resume", pid=3002)

    with pytest.raises(OutputLockError, match="explicit recovery"):
        OutputLock.acquire(output_directory, requested_owner)
    with pytest.raises(OutputLockError, match="fingerprint"):
        recover_output_lock(
            output_directory,
            expected_fingerprint="wrong-owner",
            owner_is_alive=lambda _: False,
        )
    assert (output_directory / OUTPUT_LOCK_NAME).exists()
    with pytest.raises(OutputLockError, match="still alive"):
        recover_output_lock(
            output_directory,
            expected_fingerprint=stale_owner.fingerprint(),
            owner_is_alive=lambda _: True,
        )
    assert (output_directory / OUTPUT_LOCK_NAME).exists()

    recovered = recover_output_lock(
        output_directory,
        expected_fingerprint=stale_owner.fingerprint(),
        owner_is_alive=lambda _: False,
    )
    assert recovered == stale_owner
    resumed = OutputLock.acquire(output_directory, requested_owner)
    resumed.release()
    with pytest.raises(OutputLockError, match="no longer owns"):
        stale_lock.release()


def test_output_lock_release_cannot_unlink_a_recovered_new_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch release removing a replacement lock after validating the old record."""

    import a2v2.slurm as slurm

    output_directory = tmp_path / "run"
    output_directory.mkdir()
    old_owner = _lock_owner("old", pid=4001)
    new_owner = _lock_owner("new", pid=4002)
    old_lock = slurm.OutputLock.acquire(output_directory, old_owner)
    release_read = Event()
    continue_release = Event()
    replacement_done = Event()
    real_read = slurm._read_lock_owner
    replacement: list[slurm.OutputLock] = []

    def pause_release_read(path: Path) -> object:
        """Pause release after it has read the old owner but before unlink."""

        owner = real_read(path)
        if current_thread().name == "releasing-owner":
            release_read.set()
            assert continue_release.wait(timeout=5)
        return owner

    def release_old_owner() -> None:
        """Run the old owner's release at the controlled interleaving point."""

        old_lock.release()

    def recover_then_acquire() -> None:
        """Recover the stale record, then establish a distinct new owner."""

        try:
            slurm.recover_output_lock(
                output_directory,
                expected_fingerprint=old_owner.fingerprint(),
                owner_is_alive=lambda _: False,
            )
        except slurm.OutputLockError:
            pass
        replacement.append(slurm.OutputLock.acquire(output_directory, new_owner))
        replacement_done.set()

    monkeypatch.setattr(slurm, "_read_lock_owner", pause_release_read)
    releasing = Thread(target=release_old_owner, name="releasing-owner")
    replacing = Thread(target=recover_then_acquire, name="replacing-owner")
    releasing.start()
    assert release_read.wait(timeout=5)
    replacing.start()
    try:
        assert not replacement_done.wait(timeout=0.2)
    finally:
        continue_release.set()
        releasing.join(timeout=5)
        replacing.join(timeout=5)

    assert not releasing.is_alive()
    assert not replacing.is_alive()
    assert replacement_done.is_set()
    assert len(replacement) == 1
    replacement[0].release()


def test_preemption_signal_handler_only_sets_process_local_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch collectives or checkpoint I/O running from signal context."""

    import a2v2.training as training
    from a2v2.slurm import PreemptionFlag

    def unexpected(*_: object, **__: object) -> None:
        """Fail if signal receipt performs any collective or file write."""

        raise AssertionError("signal handler performed unsafe work")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected)
    monkeypatch.setattr(training, "save_checkpoint", unexpected)
    flag = PreemptionFlag()

    result = flag.signal_handler(signal.SIGUSR1, None)

    assert result is None
    assert flag.requested
    assert flag.signal_number == signal.SIGUSR1


def test_preemption_handlers_install_usr1_and_term_then_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch process-global signal handlers leaking after the training call."""

    import a2v2.slurm as slurm

    installed: list[tuple[int, object]] = []

    def fake_signal(number: int, handler: object) -> object:
        """Record installation and return one distinguishable prior handler."""

        installed.append((number, handler))
        return f"previous-{number}"

    monkeypatch.setattr(slurm.signal, "signal", fake_signal)
    flag = slurm.PreemptionFlag()
    registration = slurm.install_preemption_handlers(flag)

    assert installed == [
        (signal.SIGUSR1, flag.signal_handler),
        (signal.SIGTERM, flag.signal_handler),
    ]
    registration.restore()
    assert installed[-2:] == [
        (signal.SIGUSR1, f"previous-{signal.SIGUSR1}"),
        (signal.SIGTERM, f"previous-{signal.SIGTERM}"),
    ]


def test_local_preemption_safe_point_and_exit_contract() -> None:
    """Catch a local run ignoring a flag or returning a success exit code."""

    from a2v2.slurm import (
        PREEMPTION_EXIT_CODE,
        PreemptionFlag,
        TrainingPreempted,
        coordinated_preemption_requested,
    )

    flag = PreemptionFlag()
    assert not coordinated_preemption_requested(
        flag,
        collective_device=torch.device("cpu"),
    )
    flag.request(signal.SIGTERM)
    assert coordinated_preemption_requested(
        flag,
        collective_device=torch.device("cpu"),
    )
    error = TrainingPreempted(Path("checkpoint_last.pt"), update=17)
    assert PREEMPTION_EXIT_CODE == error.exit_code == 75
    assert error.checkpoint_path == Path("checkpoint_last.pt")
    assert error.update == 17


def test_rank_local_rng_schema_round_trips_the_selected_global_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch rank zero RNG broadcast or an untagged distributed RNG payload."""

    import a2v2.training as training

    torch.manual_seed(101)
    rank_zero = training.capture_rng_state()
    torch.manual_seed(202)
    rank_one = training.capture_rng_state()
    expected = torch.rand(4)
    encoded = [
        training.serialize_rng_state(rank_zero),
        training.serialize_rng_state(rank_one),
    ]

    def fake_gather(
        local: bytes,
        gathered: list[object] | None,
        *,
        dst: int,
        group: object,
    ) -> None:
        """Fill rank zero's gather destination with two serialized RNG states."""

        assert local == encoded[0]
        assert dst == 0
        assert gathered is not None
        gathered[:] = encoded

    monkeypatch.setattr(training.dist, "gather_object", fake_gather)
    gathered = training.gather_rank_rng_states(
        rank_zero,
        world_size=2,
        rank=0,
        group=None,
    )
    assert gathered is not None
    assert gathered["schema"] == training.RANK_LOCAL_RNG_SCHEMA
    monkeypatch.setattr(training.dist, "is_available", lambda: True)
    monkeypatch.setattr(training.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(training.dist, "get_rank", lambda: 1)

    training.restore_rng_state(gathered)

    assert torch.equal(torch.rand(4), expected)


def _global_rng_snapshot() -> tuple[object, tuple[object, ...], torch.Tensor]:
    """Capture Python, NumPy, and CPU Torch generators for mutation assertions."""

    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def _assert_global_rng_snapshot(
    expected: tuple[object, tuple[object, ...], torch.Tensor],
) -> None:
    """Assert every CPU-global generator still matches a prior snapshot."""

    expected_python, expected_numpy, expected_torch = expected
    assert random.getstate() == expected_python
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected_numpy[0]
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert torch.equal(torch.random.get_rng_state(), expected_torch)


@pytest.mark.parametrize("malformed_field", ["numpy", "torch", "cuda"])
def test_rng_restore_validates_every_late_state_before_mutating_any_generator(
    monkeypatch: pytest.MonkeyPatch,
    malformed_field: str,
) -> None:
    """Catch malformed late RNG fields leaving earlier global streams changed."""

    import a2v2.training as training

    random.seed(707)
    np.random.seed(808)
    torch.manual_seed(909)
    state = deepcopy(training.capture_rng_state())
    random.seed(1707)
    np.random.seed(1808)
    torch.manual_seed(1909)
    before = _global_rng_snapshot()

    if malformed_field == "numpy":
        state["numpy"]["position"] = "not-an-integer"
    elif malformed_field == "torch":
        state["torch"] = torch.tensor([1], dtype=torch.uint8)
    else:
        state["cuda"] = [torch.tensor([1], dtype=torch.uint8)]
        original_generator = torch.Generator

        class InvalidCudaProbe:
            """Model an isolated CUDA generator that rejects malformed bytes."""

            def set_state(self, _: torch.Tensor) -> None:
                """Reject the deliberately truncated CUDA generator state."""

                raise RuntimeError("CUDA RNG state is wrong size")

        def generator(*args: object, **kwargs: object) -> object:
            """Keep CPU probes real and isolate the unavailable CUDA dependency."""

            device = kwargs.get("device", args[0] if args else "cpu")
            if str(device).startswith("cuda"):
                return InvalidCudaProbe()
            return original_generator(*args, **kwargs)

        def unsafe_global_cuda_install(_: object) -> None:
            """Expose any attempt to install before isolated validation."""

            raise RuntimeError("global CUDA install received malformed state")

        monkeypatch.setattr(torch, "Generator", generator)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "set_rng_state_all", unsafe_global_cuda_install)

    with pytest.raises(training.CheckpointError, match=f"invalid {malformed_field}"):
        training.restore_rng_state(state)

    _assert_global_rng_snapshot(before)


def test_cuda_rng_preflight_compares_local_states_to_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch the historical error of comparing local CUDA states to global world size."""

    import a2v2.training as training

    torch.manual_seed(303)
    local_state = training.capture_rng_state()
    local_state["cuda"] = [
        torch.tensor([11], dtype=torch.uint8),
        torch.tensor([12], dtype=torch.uint8),
    ]
    ranked = {
        "schema": training.RANK_LOCAL_RNG_SCHEMA,
        "world_size": 8,
        "by_rank": [deepcopy(local_state) for _ in range(8)],
    }
    restored: list[list[torch.Tensor]] = []
    original_generator = torch.Generator

    class ValidCudaProbe:
        """Accept bytes after this test has established local cardinality."""

        def set_state(self, _: torch.Tensor) -> None:
            """Model successful isolated CUDA state validation."""

    def generator(*args: object, **kwargs: object) -> object:
        """Keep the CPU probe real while replacing unavailable CUDA probes."""

        device = kwargs.get("device", args[0] if args else "cpu")
        if str(device).startswith("cuda"):
            return ValidCudaProbe()
        return original_generator(*args, **kwargs)

    monkeypatch.setattr(training.dist, "is_available", lambda: True)
    monkeypatch.setattr(training.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(training.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(training.dist, "get_rank", lambda: 5)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)
    monkeypatch.setattr(torch, "Generator", generator)

    training.restore_rng_state(ranked)

    assert len(restored) == 1
    assert len(restored[0]) == 2


def test_cuda_rng_preflight_rejects_local_state_cardinality_before_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch partial CUDA generator restore on a differently visible device set."""

    import a2v2.training as training

    torch.manual_seed(404)
    state = training.capture_rng_state()
    state["cuda"] = [torch.tensor([1], dtype=torch.uint8)]
    restored: list[object] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)

    with pytest.raises(training.CheckpointError, match="1 CUDA.*2 visible"):
        training.restore_rng_state(state)

    assert restored == []
