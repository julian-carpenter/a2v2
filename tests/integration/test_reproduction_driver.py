"""Exercise the deployable one-fold, eight-A100 reproduction workflow.

The shell tests intentionally use ``--dry-run``: this CPU repository can
verify every resolved path, topology override, and checkpoint hand-off without
pretending to execute NCCL or measure A100 memory. A separate test below runs
the report helper against a genuinely loadable tiny native checkpoint.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
import torch

from a2v2.config import config_to_dict, load_config
from a2v2.model import Animal2VecFineTuningModel
from a2v2.training import capture_rng_state, save_checkpoint


ROOT = Path(__file__).parents[2]
DRIVER = ROOT / "scripts/reproduce_meerkat_paper.sh"
EVALUATOR = ROOT / "scripts/evaluate_finetuning_checkpoint.py"


def _placeholder_manifests(directory: Path, *, fold: int = 0, fraction: str = "100") -> None:
    """Create the manifest names that shell preflight must resolve.

    Dry-run never parses the contents, so an empty file is sufficient to prove
    that subset-to-filename mapping is correct.
    """

    directory.mkdir()
    (directory / "pretrain.tsv").touch()
    suffix = {"100": "", "025": "_few_2", "001": "_few_0"}[fraction]
    (directory / f"train_{fold}{suffix}.tsv").touch()
    (directory / f"valid_{fold}.tsv").touch()


def _run_driver(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the deployment shell while retaining output for exact assertions."""

    return subprocess.run(
        ["bash", str(DRIVER), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_is_one_fold_and_uses_all_eight_gpus(tmp_path: Path) -> None:
    """Resolve the requested 8-rank, memory-aware pretrain→finetune→report flow."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    _placeholder_manifests(manifests)

    completed = _run_driver(str(manifests), str(output), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout
    # Bash's ``printf %q`` escapes commas and brackets in its executable
    # dry-run rendering. Removing those escape markers lets assertions inspect
    # semantic command arguments rather than presentation syntax.
    semantic_stdout = stdout.replace("\\", "")
    # Exactly two distributed training stages are launched: one pretraining
    # job and one fine-tuning job. Validation is a normal single-GPU process.
    assert semantic_stdout.count("--nproc-per-node=8") == 2
    assert semantic_stdout.count("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7") >= 2
    assert "distributed_training.distributed_world_size=8" in semantic_stdout

    # Pretraining is already memory-bound at the published per-rank budget.
    assert "dataset.max_tokens=408000" in semantic_stdout
    assert "optimization.update_freq=[3]" in semantic_stdout
    # Fine-tuning increases the per-rank budget while preserving almost the
    # same global effective token batch as the four-rank paper recipe.
    assert "dataset.max_tokens=960000" in semantic_stdout
    assert "optimization.update_freq=[2]" in semantic_stdout

    pretrain_checkpoint = output / "pretrain/checkpoint_last.pt"
    assert f"--pretrained-checkpoint {pretrain_checkpoint}" in semantic_stdout
    assert "configs/MeerKAT/finetune_mixup_100.yaml" in semantic_stdout
    assert "dataset.train_subset=train_0" in semantic_stdout
    assert "dataset.valid_subset=valid_0" in semantic_stdout
    assert str(output / "final-evaluation/final-evaluation-report.json") in semantic_stdout


def test_dry_run_rejects_a_missing_selected_manifest(tmp_path: Path) -> None:
    """Fail before an expensive launch when the chosen fold is incomplete."""

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "pretrain.tsv").touch()
    (manifests / "train_0.tsv").touch()

    completed = _run_driver(str(manifests), str(tmp_path / "output"), "--dry-run")

    assert completed.returncode != 0
    assert "valid_0.tsv" in completed.stderr


def _tiny_validation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a strict native checkpoint and eight labeled validation files."""

    manifests = tmp_path / "manifests"
    audio_root = tmp_path / "wav"
    label_root = tmp_path / "lbl"
    manifests.mkdir()
    audio_root.mkdir()
    label_root.mkdir()

    rows: list[str] = []
    sample_count = 8  # satisfies the stored required-batch multiple
    sample_length = 256
    for index in range(sample_count):
        filename = f"recording_{index}.wav"
        waveform = np.linspace(-0.25, 0.25, sample_length, dtype=np.float32)
        sf.write(audio_root / filename, waveform, 8_000, subtype="FLOAT")
        with h5py.File(label_root / f"recording_{index}.h5", "w") as handle:
            handle["start_frame_lbl"] = [32]
            handle["end_frame_lbl"] = [192]
            handle["lbl_cat"] = [0]
            handle["foc"] = [1]
        rows.append(f"{filename}\t{sample_length}")
    (manifests / "valid_0.tsv").write_text(
        f"{audio_root}\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    pretrained = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    active = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(
        active,
        pretrained_config=pretrained,
    )
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([2.0, -2.0]))
    checkpoint = tmp_path / "tiny-finetuned.pt"
    save_checkpoint(checkpoint, {
        "format_version": 1,
        "stage": "finetune",
        "config": {
            "active": config_to_dict(active),
            "pretrained": config_to_dict(pretrained),
        },
        "model": model.state_dict(),
        "teacher": None,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "update": 2,
        "epoch": 1,
        "batch_in_epoch": 0,
        "rng_state": capture_rng_state(),
        "sampler_state": None,
        "best_metric": 0.5,
    })
    return manifests, checkpoint


def test_evaluator_loads_a_native_checkpoint_and_writes_frame_metrics(
    tmp_path: Path,
) -> None:
    """Run real CPU validation and retain enough provenance to audit the result."""

    manifests, checkpoint = _tiny_validation_fixture(tmp_path)
    output = tmp_path / "report.json"

    completed = subprocess.run(
        [
            "python",
            str(EVALUATOR),
            "--checkpoint", str(checkpoint),
            "--manifest-dir", str(manifests),
            "--subset", "valid_0",
            "--output", str(output),
            "--fold", "0",
            "--fraction", "100",
            "--device", "cpu",
            "--max-tokens", "4096",
            "--num-workers", "0",
            "--training-world-size", "8",
            "--pretrain-max-tokens", "408000",
            "--pretrain-update-freq", "3",
            "--finetune-max-tokens", "960000",
            "--finetune-update-freq", "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(completed.stdout) == report
    assert report["schema_version"] == 1
    assert report["scope"]["exact_paper_reproduction"] is False
    assert report["run"]["fold"] == 0
    assert report["run"]["label_fraction"] == "100"
    assert report["run"]["training_world_size"] == 8
    assert report["checkpoint"]["update"] == 2
    assert len(report["checkpoint"]["sha256"]) == 64
    assert report["validation_data"]["manifest_rows"] == 8
    assert len(report["validation_data"]["sha256"]) == 64
    assert set(report["metrics"]) == {
        "loss",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "average_precision",
    }
    assert all(np.isfinite(value) for value in report["metrics"].values())
