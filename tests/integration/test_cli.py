"""Run the native training command through pretraining, resume, and fine-tuning workflows.
The suite checks checkpoint continuity, DataLoader prefetch accounting, scheduler
horizons, overrides, and distributed launch validation."""

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import soundfile as sf
import torch

from a2v2.workflows import train_main
from a2v2.training import load_checkpoint


ROOT = Path(__file__).parents[2]


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


def test_cli_pretrain_resume_and_finetune_flow(tmp_path: Path) -> None:
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
