"""Exercise the deployable one-fold, eight-A100 reproduction workflow.

The shell tests intentionally use ``--dry-run``: this CPU repository can
verify every resolved path, topology override, and checkpoint hand-off without
pretending to execute NCCL or measure A100 memory. A separate test below runs
the report helper against a genuinely loadable tiny native checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
import soundfile as sf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from a2v2.config import config_to_dict, load_config
from a2v2.model import Animal2VecFineTuningModel
from a2v2.training import capture_rng_state, save_checkpoint


ROOT = Path(__file__).parents[2]
DRIVER = ROOT / "scripts/reproduce_meerkat_paper.sh"
EVALUATOR = ROOT / "scripts/evaluate_finetuning_checkpoint.py"
PREFLIGHT = ROOT / "scripts/check_reproduction_environment.py"


def _load_preflight() -> ModuleType:
    """Load the deployable preflight script as a testable Python module."""

    spec = importlib.util.spec_from_file_location("a2v2_reproduction_preflight", PREFLIGHT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PREFLIGHT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_cuda(*, low_free_index: int | None = None) -> object:
    """Return an eight-A100 CUDA facade with controllable free-memory reports."""

    gib = 1024**3
    total = 40 * gib
    free_by_device = [39 * gib] * 8
    if low_free_index is not None:
        free_by_device[low_free_index] = 37 * gib

    class FakeCuda:
        """Expose only the CUDA inspection methods used by preflight."""

        @staticmethod
        def device_count() -> int:
            """Return the required eight visible devices."""
            return 8

        @staticmethod
        def get_device_properties(index: int) -> object:
            """Return A100 identity and total memory for one device."""
            return SimpleNamespace(name="NVIDIA A100-SXM4-40GB", total_memory=total)

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            """Return the configured free and total bytes for one device."""
            return free_by_device[index], total

    return FakeCuda()


def test_environment_preflight_reports_all_eight_a100_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check CUDA preflight returns auditable per-device capacity metadata."""
    preflight = _load_preflight()
    monkeypatch.setattr(preflight.torch, "cuda", _fake_cuda())
    gib = 1024**3

    devices = preflight.check_cuda_devices(
        expected_count=8,
        min_total_bytes=39 * gib,
        min_free_bytes=38 * gib,
    )

    assert [device["index"] for device in devices] == list(range(8))
    assert all(device["name"] == "NVIDIA A100-SXM4-40GB" for device in devices)
    assert all(device["total_bytes"] == 40 * gib for device in devices)
    assert all(device["free_bytes"] == 39 * gib for device in devices)


def test_environment_preflight_names_a_low_memory_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check one occupied GPU stops the launch and identifies its index."""
    preflight = _load_preflight()
    monkeypatch.setattr(preflight.torch, "cuda", _fake_cuda(low_free_index=3))
    gib = 1024**3

    with pytest.raises(RuntimeError, match=r"CUDA device 3.*free memory"):
        preflight.check_cuda_devices(
            expected_count=8,
            min_total_bytes=39 * gib,
            min_free_bytes=38 * gib,
        )


def test_environment_preflight_rejects_low_output_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check atomic checkpoint headroom is enforced before training."""
    preflight = _load_preflight()
    gib = 1024**3
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100 * gib, used=40 * gib, free=60 * gib),
    )

    with pytest.raises(RuntimeError, match="output filesystem.*free"):
        preflight.check_output_space(tmp_path, min_free_bytes=64 * gib)


def test_checkpoint_preflight_requires_distributed_resume_state(tmp_path: Path) -> None:
    """Check a burn-in artifact is a complete eight-rank training checkpoint."""
    preflight = _load_preflight()
    checkpoint = tmp_path / "checkpoint_last.pt"
    rank_state = capture_rng_state()
    payload = {
        "format_version": 1,
        "stage": "pretrain",
        "config": {"active": {"name": "preflight-test"}},
        "model": {},
        "teacher": {},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"last_update": 1},
        "scaler": None,
        "update": 1,
        "epoch": 1,
        "batch_in_epoch": 0,
        "rng_state": {"world_size": 8, "by_rank": [rank_state] * 8},
        "sampler_state": {"epoch": 1, "next_batch": 0},
        "best_metric": None,
    }
    save_checkpoint(checkpoint, payload)

    report = preflight.check_training_checkpoint(
        checkpoint,
        expected_world_size=8,
        expected_stage="pretrain",
    )

    assert report["path"] == str(checkpoint.resolve())
    assert report["update"] == 1
    assert report["rng_world_size"] == 8

    with pytest.raises(RuntimeError, match=r"checkpoint stage is pretrain; expected finetune"):
        preflight.check_training_checkpoint(
            checkpoint,
            expected_world_size=8,
            expected_stage="finetune",
        )

    payload["optimizer"] = None
    save_checkpoint(checkpoint, payload)
    with pytest.raises(RuntimeError, match="optimizer.*resume"):
        preflight.check_training_checkpoint(checkpoint, expected_world_size=8)


def test_environment_preflight_cli_normalizes_unexpected_check_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check malformed checkpoint errors still produce one failed JSON report."""
    preflight = _load_preflight()
    checkpoint = tmp_path / "malformed.pt"
    checkpoint.touch()
    monkeypatch.setattr(preflight, "_runtime_report", lambda: {})
    monkeypatch.setattr(preflight, "check_cuda_devices", lambda *args: [])
    monkeypatch.setattr(preflight, "check_output_space", lambda *args: {})
    monkeypatch.setattr(
        preflight,
        "check_training_checkpoint",
        lambda *args: (_ for _ in ()).throw(TypeError("malformed checkpoint")),
    )

    exit_code = preflight.main([
        "--output-dir", str(tmp_path),
        "--checkpoint", str(checkpoint),
    ])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "pass": False,
        "error": "malformed checkpoint",
    }


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


def _run_driver(
    *arguments: str,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the deployment shell while retaining output for exact assertions."""

    return subprocess.run(
        ["bash", str(DRIVER), *arguments],
        cwd=ROOT,
        env=environment,
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
    # Fresh runs probe collectives, burn in one production update, resume the
    # full pretraining job, and finally fine-tune. Validation is single-GPU.
    assert semantic_stdout.count("--nproc-per-node=8") == 4
    assert semantic_stdout.count("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7") >= 4
    assert "python -m torch.distributed.run" in semantic_stdout
    assert "check_reproduction_environment.py" in semantic_stdout
    assert "tests/gpu/nccl_probe.py" in semantic_stdout
    assert "--stop-at-update 1" in semantic_stdout
    assert "distributed_training.distributed_world_size=8" in semantic_stdout

    # Pretraining is already memory-bound at the published per-rank budget.
    assert "dataset.max_tokens=408000" in semantic_stdout
    assert "optimization.update_freq=[3]" in semantic_stdout
    # Fine-tuning increases the per-rank budget while preserving almost the
    # same global effective token batch as the four-rank paper recipe.
    assert "dataset.max_tokens=960000" in semantic_stdout
    assert "optimization.update_freq=[2]" in semantic_stdout

    pretrain_checkpoint = output / "pretrain/checkpoint_last.pt"
    assert f"--resume {pretrain_checkpoint}" in semantic_stdout
    assert f"--pretrained-checkpoint {pretrain_checkpoint}" in semantic_stdout
    assert "configs/MeerKAT/finetune_mixup_100.yaml" in semantic_stdout
    assert "dataset.train_subset=train_0" in semantic_stdout
    assert "dataset.valid_subset=valid_0" in semantic_stdout
    assert str(output / "final-evaluation/final-evaluation-report.json") in semantic_stdout
    assert str(output / "final-evaluation/tensorboard") in semantic_stdout


def test_real_driver_burns_in_resumes_and_appends_phase_logs(tmp_path: Path) -> None:
    """Check real-mode orchestration with a deterministic fake runtime."""
    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _placeholder_manifests(manifests)
    invocations = tmp_path / "invocations.jsonl"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
record = Path(os.environ["A2V2_FAKE_INVOCATIONS"])
with record.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(arguments) + "\\n")

if arguments == ["--version"]:
    print("Python 3.12.3")
elif arguments[:3] == ["-m", "pip", "freeze"]:
    print("a2v2==0.1.0")
elif arguments and arguments[0].endswith("check_reproduction_environment.py"):
    print('{"pass": true}')
elif arguments[:2] == ["-m", "torch.distributed.run"]:
    if any(value.endswith("nccl_probe.py") for value in arguments):
        probe_output = Path(arguments[arguments.index("--output-dir") + 1])
        probe_output.mkdir(parents=True, exist_ok=True)
        for rank in range(8):
            (probe_output / f"rank-{rank}.json").write_text(
                json.dumps({"rank": rank, "pass": True}) + "\\n",
                encoding="utf-8",
            )
    else:
        overrides = [
            value.removeprefix("checkpoint.save_dir=")
            for value in arguments
            if value.startswith("checkpoint.save_dir=")
        ]
        training_output = Path(overrides[0])
        training_output.mkdir(parents=True, exist_ok=True)
        (training_output / "checkpoint_last.pt").touch()
        if any("finetune_mixup" in value for value in arguments):
            (training_output / "checkpoint_best.pt").touch()
    print("fake distributed command")
elif arguments and arguments[0].endswith("evaluate_finetuning_checkpoint.py"):
    report = Path(arguments[arguments.index("--output") + 1])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"pass": true}\\n', encoding="utf-8")
    print('{"pass": true}')
else:
    raise SystemExit(f"unexpected fake Python arguments: {arguments}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    # This container's Bash startup hook probes PATH's nvidia-smi. A shell
    # script stub would recursively trigger that hook, so use a no-op binary.
    fake_nvidia_smi.symlink_to("/bin/true")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_PYTHON": str(fake_python),
        "A2V2_TRAIN_ENTRY": "/bin/true",
        "A2V2_FAKE_INVOCATIONS": str(invocations),
    }

    first = _run_driver(str(manifests), str(output), environment=environment)

    assert first.returncode == 0, first.stderr
    training_log = output / "pretrain/train.log"
    first_log = training_log.read_text(encoding="utf-8")
    assert first_log.count("phase=pretraining-burn-in") == 1
    assert first_log.count("phase=pretraining-resume") == 1

    second = _run_driver(str(manifests), str(output), environment=environment)

    assert second.returncode == 0, second.stderr
    second_log = training_log.read_text(encoding="utf-8")
    assert second_log.startswith(first_log)
    assert second_log.count("phase=pretraining-burn-in") == 1
    assert second_log.count("phase=pretraining-resume") == 2


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


def test_evaluator_loads_a_native_checkpoint_and_writes_validation_metrics(
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
        "segmented_precision",
        "segmented_recall",
        "segmented_f1",
        "segmented_accuracy",
        "segmented_average_precision",
        "segmented_micro_average_precision",
        "segmented_focal_threshold",
        "segmented_focal_f1",
        "segmented_focal_precision",
        "segmented_focal_recall",
    }
    assert all(np.isfinite(value) for value in report["metrics"].values())
    tensorboard_directory = output.parent / "tensorboard"
    assert report["tensorboard"]["log_dir"] == str(tensorboard_directory.resolve())
    events = EventAccumulator(str(tensorboard_directory))
    events.Reload()
    assert {
        "validation/valid_0/loss",
        "validation/valid_0/frame/f1",
        "validation/valid_0/segmented/f1",
        "validation/valid_0/segmented/average_precision/call",
    } <= set(events.Tags()["scalars"])
    assert "validation/valid_0/segmented/pr_micro" in events.Tags()["tensors"]
