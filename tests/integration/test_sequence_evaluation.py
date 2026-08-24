"""Integration tests for checkpoint-backed sequence evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import soundfile as sf
import torch

from a2v2.config import config_to_dict, load_config
from a2v2.model import Animal2VecFineTuningModel
from a2v2.training import capture_rng_state, load_checkpoint, save_checkpoint
import a2v2.workflows as workflows


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_data(tmp_path: Path) -> Path:
    """Create a two-example validation manifest with one positive sequence."""

    wav_dir = tmp_path / "wav"
    label_dir = tmp_path / "lbl"
    data_dir = tmp_path / "manifests"
    wav_dir.mkdir()
    label_dir.mkdir()
    data_dir.mkdir()

    for stem in ("positive", "negative"):
        sf.write(
            wav_dir / f"{stem}.wav",
            np.zeros(64, dtype=np.float32),
            8_000,
            subtype="FLOAT",
        )

    with h5py.File(label_dir / "positive.h5", "w") as handle:
        handle.create_dataset("start_frame_lbl", data=np.asarray([8], dtype=np.int64))
        handle.create_dataset("end_frame_lbl", data=np.asarray([56], dtype=np.int64))
        handle.create_dataset("lbl_cat", data=np.asarray([0], dtype=np.int64))
        handle.create_dataset("foc", data=np.asarray([1], dtype=np.int64))
    with h5py.File(label_dir / "negative.h5", "w") as handle:
        handle.create_dataset("start_frame_lbl", data=np.asarray([], dtype=np.int64))
        handle.create_dataset("end_frame_lbl", data=np.asarray([], dtype=np.int64))
        handle.create_dataset("lbl_cat", data=np.asarray([], dtype=np.int64))
        handle.create_dataset("foc", data=np.asarray([], dtype=np.int64))

    (data_dir / "holdout.tsv").write_text(
        f"{wav_dir}\n"
        "positive.wav\t64\n"
        "negative.wav\t64\n",
        encoding="utf-8",
    )
    return data_dir


def _sequence_checkpoint(tmp_path: Path, *, head: str = "cls") -> Path:
    """Write one deterministic native fine-tuning checkpoint."""

    pretrained = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=["model.use_cls_token=true"],
    )
    active = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=[
            "model.use_cls_token=true",
            f"model.classification_head={head}",
        ],
    )
    model = Animal2VecFineTuningModel.from_config(
        active,
        pretrained_config=pretrained,
    )
    if head == "cls":
        torch.nn.init.zeros_(model.classifier.weight)
        torch.nn.init.zeros_(model.classifier.bias)

    checkpoint = tmp_path / f"{head}.pt"
    save_checkpoint(
        checkpoint,
        {
            "format_version": 1,
            "stage": "finetune",
            "config": {
                "active": config_to_dict(active),
                "pretrained": config_to_dict(pretrained),
            },
            "model": model.state_dict(),
            "teacher": None,
            "optimizer": {},
            "scheduler": {},
            "scaler": None,
            "early_stopper": None,
            "ema": None,
            "update": 2,
            "epoch": 1,
            "batch_in_epoch": 0,
            "best_metric": None,
            "rng_state": capture_rng_state(),
            "sampler_state": None,
            "data_order": {},
            "gradient_accumulation": {},
        },
    )
    return checkpoint


def _arguments(checkpoint: Path, data_dir: Path, *, threshold: float) -> list[str]:
    """Build evaluator arguments for one metric threshold."""

    return [
        str(checkpoint),
        "--trust-checkpoint",
        "--config",
        str(ROOT / "tests/fixtures/tiny_finetune.yaml"),
        "--override",
        f"task.data={data_dir}",
        "--override",
        "dataset.valid_subset=holdout",
        "--override",
        "dataset.num_workers=0",
        "--override",
        "dataset.required_batch_size_multiple=1",
        "--override",
        f"criterion.metric_threshold={threshold}",
        "--override",
        "model.use_cls_token=true",
        "--override",
        "model.classification_head=cls",
        "--override",
        "distributed_training.distributed_world_size=7",
        "--device",
        "cpu",
    ]


def test_sequence_evaluator_refuses_untrusted_checkpoint_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing trust acknowledgement blocks checkpoint deserialization."""

    loader_called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        """Record and reject any attempted checkpoint load."""

        nonlocal loader_called
        loader_called = True
        raise AssertionError("checkpoint loader must not run")

    monkeypatch.setattr(workflows, "load_checkpoint", forbidden_load)
    with pytest.raises(SystemExit, match="2"):
        workflows.evaluate_sequence_main(
            [
                str(tmp_path / "untrusted.pt"),
                "--config",
                str(ROOT / "tests/fixtures/tiny_finetune.yaml"),
                "--override",
                "model.use_cls_token=true",
                "--override",
                "model.classification_head=cls",
            ]
        )

    assert loader_called is False
    error = capsys.readouterr().err
    assert "--trust-checkpoint" in error
    assert "pickle" in error
    assert "does not make pickle safe" in error


def test_sequence_evaluator_reports_real_metrics_and_preserves_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI evaluates data, honors overrides, and never writes its checkpoint."""

    data_dir = _sequence_data(tmp_path)
    checkpoint = _sequence_checkpoint(tmp_path)
    digest_before = _sha256(checkpoint)

    assert workflows.evaluate_sequence_main(
        _arguments(checkpoint, data_dir, threshold=0.6)
    ) == 0
    high_threshold = json.loads(capsys.readouterr().out)
    assert workflows.evaluate_sequence_main(
        _arguments(checkpoint, data_dir, threshold=0.4)
    ) == 0
    low_threshold = json.loads(capsys.readouterr().out)

    assert high_threshold["sequence_precision"] == pytest.approx(0.0)
    assert high_threshold["sequence_recall"] == pytest.approx(0.0)
    assert high_threshold["sequence_f1"] == pytest.approx(0.0)
    assert high_threshold["sequence_accuracy"] == pytest.approx(0.5)
    assert high_threshold["sequence_average_precision"] == pytest.approx(0.5)
    assert high_threshold["loss"] > 0.0
    assert low_threshold["sequence_f1"] == pytest.approx(2.0 / 3.0)
    assert _sha256(checkpoint) == digest_before


@pytest.mark.parametrize(
    ("checkpoint_head", "config_head", "message"),
    [
        ("cls", "frame", "classification_head=cls"),
        ("frame", "cls", "CLS fine-tuning checkpoint"),
    ],
)
def test_sequence_evaluator_rejects_frame_heads_actionably(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    checkpoint_head: str,
    config_head: str,
    message: str,
) -> None:
    """The evaluator rejects frame-head checkpoints and frame-head configs."""

    data_dir = _sequence_data(tmp_path)
    checkpoint = _sequence_checkpoint(tmp_path, head=checkpoint_head)
    arguments = _arguments(checkpoint, data_dir, threshold=0.4)
    arguments.extend(["--override", f"model.classification_head={config_head}"])
    with pytest.raises(SystemExit, match="2"):
        workflows.evaluate_sequence_main(arguments)
    assert message in capsys.readouterr().err


def test_sequence_evaluator_rejects_non_finetune_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The evaluator requires a native fine-tuning checkpoint."""

    data_dir = _sequence_data(tmp_path)
    source = _sequence_checkpoint(tmp_path)
    payload = load_checkpoint(source)
    payload["stage"] = "pretrain"
    checkpoint = tmp_path / "pretrain.pt"
    save_checkpoint(checkpoint, payload)

    with pytest.raises(SystemExit, match="2"):
        workflows.evaluate_sequence_main(
            _arguments(checkpoint, data_dir, threshold=0.4)
        )
    assert "fine-tuning checkpoint" in capsys.readouterr().err


def test_sequence_evaluator_checks_semantic_config_and_model_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The evaluator rejects label-order drift and incomplete model state."""

    data_dir = _sequence_data(tmp_path)
    checkpoint = _sequence_checkpoint(tmp_path)
    common = _arguments(checkpoint, data_dir, threshold=0.4)

    with pytest.raises(SystemExit, match="2"):
        workflows.evaluate_sequence_main(
            [*common, "--override", "task.unique_labels=[focal,call]"]
        )
    assert "unique_labels" in capsys.readouterr().err

    payload = load_checkpoint(checkpoint)
    payload["model"].pop("classifier.bias")
    corrupt = tmp_path / "missing-state.pt"
    save_checkpoint(corrupt, payload)
    with pytest.raises(SystemExit, match="2"):
        workflows.evaluate_sequence_main([str(corrupt), *common[1:]])
    assert "strict model-state load failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure_type",
    (RuntimeError, TypeError, AttributeError, ValueError, KeyError),
)
def test_sequence_evaluator_normalizes_strict_load_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_type: type[Exception],
) -> None:
    """Strict-load validation failures become actionable CLI errors."""

    data_dir = _sequence_data(tmp_path)
    checkpoint = _sequence_checkpoint(tmp_path)

    def reject_state(*args: object, **kwargs: object) -> object:
        """Raise the selected malformed-state validation error."""

        raise failure_type("malformed model-state value")

    monkeypatch.setattr(
        Animal2VecFineTuningModel,
        "load_state_dict",
        reject_state,
    )
    with pytest.raises(SystemExit, match="2"):
        workflows.evaluate_sequence_main(
            _arguments(checkpoint, data_dir, threshold=0.4)
        )
    error = capsys.readouterr().err
    assert "strict model-state load failed for sequence evaluation" in error
    assert "malformed model-state value" in error


def test_sequence_evaluator_requires_stored_pretraining_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The checkpoint config pair must identify its pretraining authority."""

    data_dir = _sequence_data(tmp_path)
    source = _sequence_checkpoint(tmp_path)
    payload = load_checkpoint(source)
    payload["config"]["pretrained"] = payload["config"]["active"]
    checkpoint = tmp_path / "wrong-pretrained-config.pt"
    save_checkpoint(checkpoint, payload)

    with pytest.raises(SystemExit, match="2"):
        workflows.evaluate_sequence_main(
            _arguments(checkpoint, data_dir, threshold=0.4)
        )
    assert "stored pretraining config" in capsys.readouterr().err


def test_sequence_evaluator_rejects_unknown_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Strict config loading rejects unknown evaluator overrides."""

    checkpoint = _sequence_checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        workflows.evaluate_sequence_main(
            [
                str(checkpoint),
                "--trust-checkpoint",
                "--config",
                str(ROOT / "tests/fixtures/tiny_finetune.yaml"),
                "--override",
                "model.not_a_real_field=1",
            ]
        )
    assert "unknown configuration field" in capsys.readouterr().err
