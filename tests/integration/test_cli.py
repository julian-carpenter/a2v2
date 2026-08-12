"""Run the native training command through pretraining, resume, and fine-tuning workflows.
The suite checks checkpoint continuity, DataLoader prefetch accounting, scheduler
horizons, overrides, and distributed launch validation."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import soundfile as sf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import a2v2.workflows as workflows
from a2v2.workflows import train_main
from a2v2.training import load_checkpoint


ROOT = Path(__file__).parents[2]


def test_checkpoint_process_group_uses_timed_gloo_for_distributed_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check CUDA training separates checkpoint objects from NCCL collectives."""
    selected_group = object()

    def fake_new_group(*, backend: str, timeout: timedelta) -> object:
        """Return a sentinel for the requested Gloo group."""
        assert backend == "gloo"
        assert timeout == timedelta(hours=2)
        return selected_group

    monkeypatch.setattr(workflows.dist, "new_group", fake_new_group)

    assert workflows._checkpoint_process_group(torch.device("cuda", 0), 8) is selected_group


@pytest.mark.parametrize(
    ("device", "world_size"),
    [(torch.device("cpu"), 8), (torch.device("cuda", 0), 1)],
)
def test_checkpoint_process_group_skips_cpu_or_single_rank(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    world_size: int,
) -> None:
    """Check launches that do not need a second backend create no group."""

    def unexpected_new_group(*, backend: str, timeout: timedelta) -> object:
        """Fail if a test case attempts to create an unnecessary group."""
        raise AssertionError(f"unexpected {backend} process group")

    monkeypatch.setattr(workflows.dist, "new_group", unexpected_new_group)

    assert workflows._checkpoint_process_group(device, world_size) is None


def test_validation_decision_uses_cpu_control_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check validation decisions leave rank zero through the CPU control plane."""
    selected_group = object()

    def fake_broadcast(tensor: torch.Tensor, *, src: int, group: object) -> None:
        """Populate the received decision while checking the control-plane contract."""
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64
        assert src == 0
        assert group is selected_group
        tensor.copy_(torch.tensor([0.75, 1.0], dtype=torch.float64))

    monkeypatch.setattr(workflows.dist, "broadcast", fake_broadcast)

    assert workflows._synchronize_validation_decision(
        0.0,
        False,
        world_size=8,
        group=selected_group,  # type: ignore[arg-type]
    ) == (0.75, True)


def test_run_training_destroys_process_group_created_during_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check exceptions cannot leak a process group initialized by training."""
    distributed = {"initialized": False}
    destroyed: list[bool] = []

    monkeypatch.setattr(
        workflows.dist,
        "is_initialized",
        lambda: distributed["initialized"],
    )

    def fake_run_training(*args: object, **kwargs: object) -> Path:
        """Model a failure after this training call initialized distributed state."""
        distributed["initialized"] = True
        raise RuntimeError("forced training failure")

    def fake_destroy_process_group() -> None:
        """Record cleanup and clear the simulated distributed state."""
        destroyed.append(True)
        distributed["initialized"] = False

    monkeypatch.setattr(workflows, "_run_training", fake_run_training)
    monkeypatch.setattr(workflows.dist, "destroy_process_group", fake_destroy_process_group)

    with pytest.raises(RuntimeError, match="forced training failure"):
        workflows.run_training(
            object(),  # type: ignore[arg-type]
            device_name="cuda",
            resume_path=None,
            pretrained_checkpoint=None,
        )

    assert destroyed == [True]


def test_run_training_preserves_caller_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check training does not destroy a process group initialized by its caller."""
    monkeypatch.setattr(workflows.dist, "is_initialized", lambda: True)

    def fake_run_training(*args: object, **kwargs: object) -> Path:
        """Model a training failure beneath a caller-owned process group."""
        raise RuntimeError("forced training failure")

    def unexpected_destroy_process_group() -> None:
        """Fail if training destroys its caller's simulated group."""
        raise AssertionError("caller-owned process group was destroyed")

    monkeypatch.setattr(workflows, "_run_training", fake_run_training)
    monkeypatch.setattr(
        workflows.dist,
        "destroy_process_group",
        unexpected_destroy_process_group,
    )

    with pytest.raises(RuntimeError, match="forced training failure"):
        workflows.run_training(
            object(),  # type: ignore[arg-type]
            device_name="cuda",
            resume_path=None,
            pretrained_checkpoint=None,
        )


def test_run_training_destroys_subgroup_created_under_caller_owned_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check A2V2 cleans up its subgroup while preserving the caller's default."""
    checkpoint_group = object()
    destroyed: list[object | None] = []
    monkeypatch.setattr(workflows.dist, "is_initialized", lambda: True)

    def fake_run_training(*args: object, **kwargs: object) -> Path:
        """Register one simulated A2V2-owned subgroup before failing."""
        created_groups = kwargs.get("created_checkpoint_groups")
        assert isinstance(created_groups, list)
        created_groups.append(checkpoint_group)
        raise RuntimeError("forced training failure")

    def fake_destroy_process_group(group: object | None = None) -> None:
        """Record the exact group selected for cleanup."""
        destroyed.append(group)

    monkeypatch.setattr(workflows, "_run_training", fake_run_training)
    monkeypatch.setattr(workflows.dist, "destroy_process_group", fake_destroy_process_group)

    with pytest.raises(RuntimeError, match="forced training failure"):
        workflows.run_training(
            object(),  # type: ignore[arg-type]
            device_name="cuda",
            resume_path=None,
            pretrained_checkpoint=None,
        )

    assert destroyed == [checkpoint_group]


def _tensorboard_tags(directory: Path) -> dict[str, object]:
    """Load tags after the training workflow has closed its event writer."""

    events = EventAccumulator(str(directory))
    events.Reload()
    return events.Tags()


def _assert_tree_equal(actual: Any, expected: Any, path: str = "root") -> None:
    """Compare nested tensors and containers with exact value equality."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert torch.equal(actual.cpu(), expected.cpu()), path
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        for key in expected:
            _assert_tree_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected), path
        for index, item in enumerate(expected):
            _assert_tree_equal(actual[index], item, f"{path}[{index}]")
        return
    assert actual == expected, path


def _data(tmp_path: Path, lengths: tuple[int, ...] = (64, 72, 80)) -> Path:
    """Create tiny audio, labels, and manifests for the end-to-end CLI workflow."""
    manifests = tmp_path / "manifests"
    audio_root = tmp_path / "wav"
    label_root = tmp_path / "lbl"
    manifests.mkdir()
    audio_root.mkdir()
    label_root.mkdir()
    rows = []
    for index, length in enumerate(lengths):
        name = f"sample_{index}.wav"
        sf.write(audio_root / name, np.linspace(-0.5, 0.5, length), 8000, subtype="FLOAT")
        with h5py.File(label_root / f"sample_{index}.h5", "w") as handle:
            handle["start_frame_lbl"] = [8]
            handle["end_frame_lbl"] = [length - 8]
            handle["lbl_cat"] = [0]
            handle["foc"] = [1]
        rows.append(f"{name}\t{length}")
    body = f"{audio_root}\n" + "\n".join(rows) + "\n"
    (manifests / "pretrain.tsv").write_text(body, encoding="utf-8")
    (manifests / "train.tsv").write_text(body, encoding="utf-8")
    (manifests / "valid.tsv").write_text(body, encoding="utf-8")
    return manifests


def test_cli_pretrain_resume_and_finetune_flow(tmp_path: Path, capsys) -> None:
    """Check cli pretrain resume and finetune flow."""
    manifests = _data(tmp_path)
    pretrain_dir = tmp_path / "pretrain"
    common = [
        "--config", str(ROOT / "configs/cpu_smoke_pretraining.yaml"),
        "--override", f"task.data={manifests}",
        "--override", f"checkpoint.save_dir={pretrain_dir}",
        "--override", "optimization.max_update=1",
        "--device", "cpu",
    ]
    assert train_main(common) == 0
    pretrain_checkpoint = pretrain_dir / "checkpoint_last.pt"
    assert pretrain_checkpoint.is_file()
    assert load_checkpoint(pretrain_checkpoint)["update"] == 1

    assert train_main(common + ["--resume", str(pretrain_checkpoint), "--max-updates", "2"]) == 0
    assert load_checkpoint(pretrain_checkpoint)["update"] == 2

    finetune_dir = tmp_path / "finetune"
    assert train_main([
        "--config", str(ROOT / "configs/cpu_smoke_finetuning.yaml"),
        "--override", f"task.data={manifests}",
        "--override", f"checkpoint.save_dir={finetune_dir}",
        "--override", "dataset.disable_validation=false",
        "--override", "dataset.validate_after_updates=1",
        "--override", "dataset.validate_interval_updates=0",
        "--pretrained-checkpoint", str(pretrain_checkpoint),
        "--max-updates", "1",
        "--device", "cpu",
    ]) == 0
    finetune_checkpoint = finetune_dir / "checkpoint_last.pt"
    assert finetune_checkpoint.is_file()
    checkpoint = load_checkpoint(finetune_checkpoint)
    assert checkpoint["stage"] == "finetune"
    assert checkpoint["update"] == 1
    assert "pretrained" in checkpoint["config"]
    assert checkpoint["best_metric"] is not None
    assert (finetune_dir / "checkpoint_best.pt").is_file()
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    validation = next(record for record in records if "validation" in record)
    for name in (
        "segmented_precision",
        "segmented_recall",
        "segmented_f1",
        "segmented_accuracy",
        "segmented_average_precision",
        "segmented_micro_average_precision",
    ):
        assert name in validation
        assert np.isfinite(validation[name])
    finetuning_tags = _tensorboard_tags(finetune_dir / "tensorboard")
    assert {
        "train/loss",
        "train/gradient_norm",
        "train/learning_rate",
        "validation/valid/loss",
        "validation/valid/frame/f1",
        "validation/valid/segmented/f1",
        "validation/valid/segmented/average_precision/call",
    } <= set(finetuning_tags["scalars"])
    assert "validation/valid/segmented/pr_micro" in finetuning_tags["tensors"]


def test_cli_stop_at_update_preserves_configured_scheduler_horizon(tmp_path: Path, capsys) -> None:
    """Check cli stop at update preserves configured scheduler horizon."""
    manifests = _data(tmp_path)
    output_dir = tmp_path / "bounded_pretrain"
    arguments = [
        "--config", str(ROOT / "configs/cpu_smoke_pretraining.yaml"),
        "--override", f"task.data={manifests}",
        "--override", f"checkpoint.save_dir={output_dir}",
        "--stop-at-update", "1",
        "--device", "cpu",
    ]

    assert train_main(arguments) == 0
    checkpoint = load_checkpoint(output_dir / "checkpoint_last.pt")
    assert checkpoint["update"] == 1
    assert checkpoint["scheduler"]["last_update"] == 1
    assert checkpoint["config"]["active"]["optimization"]["max_update"] == 2
    assert checkpoint["epoch"] == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = next(record["training_summary"] for record in records if "training_summary" in record)
    assert summary["rank"] == 0
    assert summary["world_size"] == 1
    assert summary["terminal_update"] == 1
    assert summary["configured_max_update"] == 2


def test_cli_logs_pretraining_variance_diagnostics_only_for_pretraining(
    tmp_path: Path,
    capsys,
) -> None:
    """Expose collapse diagnostics in pretraining JSON and omit them in fine-tuning."""

    manifests = _data(tmp_path)
    pretrain_dir = tmp_path / "variance-pretrain"
    assert train_main([
        "--config", str(ROOT / "configs/cpu_smoke_pretraining.yaml"),
        "--override", f"task.data={manifests}",
        "--override", f"checkpoint.save_dir={pretrain_dir}",
        "--max-updates", "1",
        "--device", "cpu",
    ]) == 0

    pretraining_records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    update_record = next(
        record
        for record in pretraining_records
        if "update" in record and "loss" in record
    )
    summary = next(
        record["training_summary"]
        for record in pretraining_records
        if "training_summary" in record
    )
    for record in (update_record, summary):
        assert record["pred_var"] > 0
        assert record["target_var"] > 0
        assert np.isfinite(record["pred_var"])
        assert np.isfinite(record["target_var"])
    pretraining_tags = _tensorboard_tags(pretrain_dir / "tensorboard")
    assert {
        "train/loss",
        "train/sample_size",
        "train/gradient_norm",
        "train/learning_rate",
        "pretrain/pred_var",
        "pretrain/target_var",
        "run/parameters",
        "run/trainable_parameters",
    } <= set(pretraining_tags["scalars"])

    finetune_dir = tmp_path / "variance-finetune"
    assert train_main([
        "--config", str(ROOT / "configs/cpu_smoke_finetuning.yaml"),
        "--override", f"task.data={manifests}",
        "--override", f"checkpoint.save_dir={finetune_dir}",
        "--pretrained-checkpoint", str(pretrain_dir / "checkpoint_last.pt"),
        "--max-updates", "1",
        "--device", "cpu",
    ]) == 0
    finetuning_records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    finetuning_update = next(
        record
        for record in finetuning_records
        if "update" in record and "loss" in record
    )
    assert "pred_var" not in finetuning_update
    assert "target_var" not in finetuning_update
    finetuning_tags = _tensorboard_tags(finetune_dir / "tensorboard")
    assert "train/loss" in finetuning_tags["scalars"]
    assert "pretrain/pred_var" not in finetuning_tags["scalars"]


def test_cli_checkpoint_tracks_consumed_batches_not_worker_prefetch(tmp_path: Path) -> None:
    """Check cli checkpoint tracks consumed batches not worker prefetch."""
    manifests = _data(tmp_path, (64,) * 12)
    output_dir = tmp_path / "multiworker_pretrain"

    assert train_main([
        "--config", str(ROOT / "configs/cpu_smoke_pretraining.yaml"),
        "--override", f"task.data={manifests}",
        "--override", f"checkpoint.save_dir={output_dir}",
        "--override", "dataset.max_tokens=64",
        "--override", "dataset.num_workers=2",
        "--stop-at-update", "1",
        "--device", "cpu",
    ]) == 0

    checkpoint = load_checkpoint(output_dir / "checkpoint_last.pt")
    assert checkpoint["batch_in_epoch"] == 1
    assert checkpoint["sampler_state"] == {"epoch": 0, "next_batch": 1}


def test_cli_multiworker_resume_matches_uninterrupted_state(tmp_path: Path) -> None:
    """Check cli multiworker resume matches uninterrupted state."""
    manifests = _data(tmp_path, (64,) * 12)

    def arguments(output_dir: Path, stop: int) -> list[str]:
        """Return arguments for the child training process used by this test."""
        return [
            "--config", str(ROOT / "configs/cpu_smoke_pretraining.yaml"),
            "--override", f"task.data={manifests}",
            "--override", f"checkpoint.save_dir={output_dir}",
            "--override", "dataset.max_tokens=64",
            "--override", "dataset.num_workers=2",
            "--stop-at-update", str(stop),
            "--device", "cpu",
        ]

    continuous_dir = tmp_path / "continuous"
    assert train_main(arguments(continuous_dir, 2)) == 0

    resumed_dir = tmp_path / "resumed"
    assert train_main(arguments(resumed_dir, 1)) == 0
    resume_point = resumed_dir / "checkpoint_last.pt"
    assert train_main(arguments(resumed_dir, 2) + ["--resume", str(resume_point)]) == 0

    continuous = load_checkpoint(continuous_dir / "checkpoint_last.pt")
    resumed = load_checkpoint(resumed_dir / "checkpoint_last.pt")
    for field in (
        "model", "teacher", "optimizer", "scheduler", "scaler", "update",
        "epoch", "batch_in_epoch", "rng_state", "sampler_state", "best_metric",
    ):
        _assert_tree_equal(resumed[field], continuous[field], field)


def test_cli_returns_parser_error_for_unknown_override(tmp_path: Path) -> None:
    """Check cli returns parser error for unknown override."""
    try:
        train_main([
            "--config", str(ROOT / "configs/cpu_smoke_pretraining.yaml"),
            "--override", "model.unknown_field=1",
            "--device", "cpu",
        ])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("invalid CLI config returned success")


def test_cli_rejects_a_launch_world_size_that_differs_from_the_recipe(capsys) -> None:
    """Check cli rejects a launch world size that differs from the recipe."""
    try:
        train_main([
            "--config", str(ROOT / "configs/cpu_smoke_pretraining.yaml"),
            "--override", "distributed_training.distributed_world_size=2",
            "--device", "cpu",
        ])
    except SystemExit as exc:
        assert exc.code != 0
        assert "distributed_world_size" in capsys.readouterr().err
    else:
        raise AssertionError("world-size mismatch returned success")
