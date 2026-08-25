"""Exercise the full-label, eight-A100 modern Animal2Vec benchmark driver."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from a2v2.training import capture_rng_state

ROOT = Path(__file__).parents[2]
DRIVER = ROOT / "scripts/animal2vec2_benchmark.sh"
PREFLIGHT = ROOT / "scripts/check_reproduction_environment.py"


def _placeholder_manifests(directory: Path, *, fold: int = 0) -> None:
    """Create the full-label manifest names resolved by shell preflight."""

    directory.mkdir()
    (directory / "pretrain.tsv").touch()
    (directory / f"train_{fold}.tsv").touch()
    (directory / f"valid_{fold}.tsv").touch()


def _run_driver(
    *arguments: str,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the benchmark shell while retaining output for exact assertions."""

    return subprocess.run(
        ["bash", str(DRIVER), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _semantic_lines(output: str) -> list[str]:
    """Remove Bash's audit escaping from rendered launch commands."""

    return [line.replace("\\", "") for line in output.splitlines()]


def _load_preflight() -> ModuleType:
    """Load the deployable preflight script as a testable module."""

    spec = importlib.util.spec_from_file_location("a2v2_modern_preflight", PREFLIGHT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PREFLIGHT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_resolves_the_approved_modern_full_label_workflow(
    tmp_path: Path,
) -> None:
    """Resolve modern recipes, invariant batch exposure, and sequence evaluation."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    _placeholder_manifests(manifests)

    completed = _run_driver(str(manifests), str(output), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    semantic = "\n".join(_semantic_lines(completed.stdout))
    training = [
        line
        for line in _semantic_lines(completed.stdout)
        if line.startswith("Launching:") and "a2v2-train" in line
    ]
    assert len(training) == 3
    assert sum("rope_cls_geglu_pretrain.yaml" in line for line in training) == 2
    assert sum("rope_cls_geglu_finetune.yaml" in line for line in training) == 1
    assert all("--nproc-per-node=8" in line for line in training)
    assert all("OMP_NUM_THREADS=8" in line for line in training)
    assert all("env -u BNB_CUDA_VERSION" in line for line in training)
    assert all("TORCHINDUCTOR_CACHE_DIR=" in line for line in training)
    assert all(
        "--override model.checkpoint_activations=false" in line
        for line in training
    )

    training_text = "\n".join(training)
    assert training_text.count("dataset.max_tokens=408000") == 2
    assert training_text.count("optimization.update_freq=[3]") == 2
    assert "dataset.max_tokens=960000" in training_text
    assert "optimization.update_freq=[2]" in training_text
    assert "pretrain_max_update=384230" in semantic
    assert "finetune_max_update=30000" in semantic
    assert "--stop-at-update 1" in semantic
    assert f"--resume {output / 'pretrain/checkpoint_last.pt'}" in semantic
    assert f"--pretrained-checkpoint {output / 'pretrain/checkpoint_last.pt'}" in semantic
    assert "--require-bitsandbytes" in semantic
    assert "dataset.train_subset=train_0" in semantic
    assert "dataset.valid_subset=valid_0" in semantic

    evaluation = [
        line
        for line in _semantic_lines(completed.stdout)
        if line.startswith("Launching:") and "a2v2-evaluate-sequence" in line
    ]
    assert len(evaluation) == 1
    assert "--trust-checkpoint" in evaluation[0]
    assert f"{output / 'finetune/checkpoint_best.pt'}" in evaluation[0]
    assert "dataset.valid_subset=valid_0" in evaluation[0]
    assert "dataset.max_tokens=320000" in evaluation[0]
    assert "dataset.num_workers=20" in evaluation[0]
    assert "evaluate_finetuning_checkpoint.py" not in semantic
    assert str(output / "final-evaluation/final-evaluation-report.json") in semantic


def test_driver_supports_fold_selection_but_rejects_few_shot_modes(
    tmp_path: Path,
) -> None:
    """Keep fold selection while exposing only the approved 100-percent split."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    _placeholder_manifests(manifests, fold=3)

    completed = _run_driver(
        str(manifests), str(output), "--fold", "3", "--dry-run"
    )
    rejected = _run_driver(
        str(manifests), str(output), "--fraction", "025", "--dry-run"
    )

    semantic = completed.stdout.replace("\\", "")
    assert completed.returncode == 0, completed.stderr
    assert "dataset.train_subset=train_3" in semantic
    assert "dataset.valid_subset=valid_3" in semantic
    assert rejected.returncode != 0
    assert "unknown argument: --fraction" in rejected.stderr


def test_dry_run_validates_and_resumes_existing_native_checkpoints(
    tmp_path: Path,
) -> None:
    """Resume both stages without repeating the pretraining burn-in."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    _placeholder_manifests(manifests)
    pretrain = output / "pretrain/checkpoint_last.pt"
    finetune = output / "finetune/checkpoint_last.pt"
    pretrain.parent.mkdir(parents=True)
    finetune.parent.mkdir(parents=True)
    pretrain.touch()
    finetune.touch()

    completed = _run_driver(str(manifests), str(output), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    semantic = completed.stdout.replace("\\", "")
    assert "--stop-at-update 1" not in semantic
    assert f"--checkpoint {pretrain} --expected-stage pretrain" in semantic
    assert f"--resume {pretrain}" in semantic
    assert f"--checkpoint {finetune} --expected-stage finetune" in semantic
    assert f"--resume {finetune}" in semantic


def test_real_driver_writes_the_sequence_report_and_reuses_compile_cache(
    tmp_path: Path,
) -> None:
    """Execute orchestration through fake tools without launching CUDA work."""

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
    handle.write(json.dumps({"arguments": arguments, "environment": {
        "BNB_CUDA_VERSION": os.environ.get("BNB_CUDA_VERSION"),
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }}) + "\\n")

if arguments == ["--version"]:
    print("Python 3.12.3")
elif arguments[:3] == ["-m", "pip", "freeze"]:
    print("a2v2==0.1.0")
elif arguments[:2] == ["-m", "json.tool"]:
    json.loads(Path(arguments[2]).read_text(encoding="utf-8"))
elif arguments and arguments[0].endswith("check_reproduction_environment.py"):
    print('{"pass": true}')
elif arguments[:2] == ["-m", "torch.distributed.run"]:
    if any(value.endswith("nccl_probe.py") for value in arguments):
        probe_output = Path(arguments[arguments.index("--output-dir") + 1])
        probe_output.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = next(
            value.removeprefix("checkpoint.save_dir=")
            for value in arguments
            if value.startswith("checkpoint.save_dir=")
        )
        training_output = Path(save_dir)
        training_output.mkdir(parents=True, exist_ok=True)
        (training_output / "checkpoint_last.pt").touch()
        if any("rope_cls_geglu_finetune.yaml" in value for value in arguments):
            (training_output / "checkpoint_best.pt").touch()
    print("fake distributed command")
else:
    raise SystemExit(f"unexpected fake Python arguments: {arguments}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_evaluator = fake_bin / "a2v2-evaluate-sequence"
    fake_evaluator.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '{\"metrics/finetune/sequence_f1\": 0.75}'\n",
        encoding="utf-8",
    )
    fake_evaluator.chmod(0o755)
    (fake_bin / "nvidia-smi").symlink_to("/bin/true")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_PYTHON": str(fake_python),
        "A2V2_TRAIN_ENTRY": "/bin/true",
        "A2V2_SEQUENCE_EVAL_ENTRY": str(fake_evaluator),
        "A2V2_FAKE_INVOCATIONS": str(invocations),
    }

    completed = _run_driver(str(manifests), str(output), environment=environment)

    assert completed.returncode == 0, completed.stderr
    report = output / "final-evaluation/final-evaluation-report.json"
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "metrics/finetune/sequence_f1": 0.75,
    }
    checkpoint_identity = output / "final-evaluation/evaluation-checkpoint-sha256.txt"
    checkpoint_metadata = output / "final-evaluation/evaluation-checkpoint-preflight.json"
    assert str(output / "finetune/checkpoint_best.pt") in checkpoint_identity.read_text()
    assert json.loads(checkpoint_metadata.read_text(encoding="utf-8")) == {
        "pass": True,
    }
    profile = (output / "environment/run-profile.txt").read_text(encoding="utf-8")
    assert "variant=animal2vec2-modern\n" in profile
    assert "label_fraction=100\n" in profile
    assert "checkpoint_activations=false\n" in profile
    assert "pretrain_max_update=384230\n" in profile
    assert "finetune_max_update=30000\n" in profile
    pretrain_log = (output / "pretrain/train.log").read_text(encoding="utf-8")
    assert pretrain_log.count("phase=pretraining-burn-in") == 1
    assert pretrain_log.count("phase=pretraining-resume") == 1
    validation_log = (output / "final-evaluation/validation.log").read_text(
        encoding="utf-8"
    )
    assert "phase=evaluation" in validation_log
    assert '"metrics/finetune/sequence_f1": 0.75' in validation_log
    records = [json.loads(line) for line in invocations.read_text().splitlines()]
    training_records = [
        record
        for record in records
        if record["arguments"][:2] == ["-m", "torch.distributed.run"]
        and not any("nccl_probe.py" in value for value in record["arguments"])
    ]
    assert len(training_records) == 3
    assert all(
        record["environment"]["BNB_CUDA_VERSION"] is None
        for record in training_records
    )
    caches = {
        record["environment"]["TORCHINDUCTOR_CACHE_DIR"]
        for record in training_records
    }
    assert caches == {str(output / "environment/torchinductor-cache")}


def test_dry_run_can_request_bitsandbytes_cuda_auto_detection(tmp_path: Path) -> None:
    """Omit the CUDA override when the explicit auto selector is requested."""

    manifests = tmp_path / "manifests"
    _placeholder_manifests(manifests)
    environment = {**os.environ, "A2V2_BNB_CUDA_VERSION": "auto"}

    completed = _run_driver(
        str(manifests),
        str(tmp_path / "experiment"),
        "--dry-run",
        environment=environment,
    )

    assert completed.returncode == 0, completed.stderr
    launches = [
        line for line in completed.stdout.splitlines() if line.startswith("Launching:")
    ]
    assert launches
    assert all("BNB_CUDA_VERSION=" not in line for line in launches)
    assert "bitsandbytes CUDA selection=auto" in completed.stdout


def test_real_driver_falls_back_from_a_stale_best_checkpoint(tmp_path: Path) -> None:
    """Evaluate the current last checkpoint when best belongs to another run."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _placeholder_manifests(manifests)
    pretrain = output / "pretrain/checkpoint_last.pt"
    finetune_last = output / "finetune/checkpoint_last.pt"
    finetune_best = output / "finetune/checkpoint_best.pt"
    pretrain.parent.mkdir(parents=True)
    finetune_last.parent.mkdir(parents=True)
    pretrain.touch()
    finetune_last.touch()
    finetune_best.touch()
    evaluated = tmp_path / "evaluated.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
if os.environ.get("BNB_CUDA_VERSION") is not None:
    raise SystemExit("auto mode inherited BNB_CUDA_VERSION")
if arguments == ["--version"]:
    print("Python 3.12.3")
elif arguments[:3] == ["-m", "pip", "freeze"]:
    print("a2v2==0.1.0")
elif arguments[:2] == ["-m", "json.tool"]:
    json.loads(Path(arguments[2]).read_text(encoding="utf-8"))
elif arguments and arguments[0].endswith("check_reproduction_environment.py"):
    checkpoint = (
        Path(arguments[arguments.index("--checkpoint") + 1])
        if "--checkpoint" in arguments
        else None
    )
    if "--reference-checkpoint" in arguments and checkpoint is not None and checkpoint.name == "checkpoint_best.pt":
        print('{"pass": false, "error": "resume fingerprint differs"}')
        raise SystemExit(1)
    print('{"pass": true}')
elif arguments[:2] == ["-m", "torch.distributed.run"]:
    print("fake distributed command")
else:
    raise SystemExit(f"unexpected fake Python arguments: {arguments}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_evaluator = fake_bin / "a2v2-evaluate-sequence"
    fake_evaluator.write_text(
        "#!/usr/bin/env bash\n[[ -z \"${BNB_CUDA_VERSION+x}\" ]] || exit 9\nprintf '%s\\n' \"$1\" > \"$A2V2_FAKE_EVALUATED\"\nprintf '%s\\n' '{\"metrics/finetune/sequence_f1\": 0.5}'\n",
        encoding="utf-8",
    )
    fake_evaluator.chmod(0o755)
    (fake_bin / "nvidia-smi").symlink_to("/bin/true")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "A2V2_PYTHON": str(fake_python),
        "A2V2_TRAIN_ENTRY": "/bin/true",
        "A2V2_SEQUENCE_EVAL_ENTRY": str(fake_evaluator),
        "A2V2_FAKE_EVALUATED": str(evaluated),
        "A2V2_BNB_CUDA_VERSION": "auto",
        "BNB_CUDA_VERSION": "999",
    }

    completed = _run_driver(str(manifests), str(output), environment=environment)

    assert completed.returncode == 0, completed.stderr
    assert evaluated.read_text(encoding="utf-8").strip() == str(finetune_last)
    assert "checkpoint_best.pt does not match the current run" in completed.stderr


def _modern_checkpoint_payload() -> dict[str, object]:
    """Return structurally resumable modern state for preflight tests."""

    rank_state = capture_rng_state()
    return {
        "format_version": 2,
        "stage": "finetune",
        "config": {
            "active": {
                "common": {"fp16": False},
                "checkpoint": {"resume_policy": "strict"},
                "optimization": {"gradient_clip_method": "adagc"},
                "optimizer": {"weight_decay_schedule": "cosine"},
            }
        },
        "model": {},
        "teacher": None,
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"last_update": 4},
        "scaler": None,
        "gradient_clipper": {
            "algorithm_version": 1,
            "update": 4,
            "parameter_names": ["weight"],
            "norm_emas": {"weight": torch.tensor(1.0, dtype=torch.float32)},
        },
        "weight_decay_scheduler": {"last_update": 4},
        "topology": None,
        "resume_compatibility": {
            "training_data.schema": "a2v2.training-data.v2",
            "training_data.manifest_sha256": "same-run",
        },
        "update": 4,
        "epoch": 1,
        "batch_in_epoch": 0,
        "rng_state": {
            "world_size": 8,
            "by_rank": [dict(rank_state) for _ in range(8)],
        },
        "sampler_state": {"epoch": 1, "next_batch": 0},
        "best_metric": 0.5,
    }


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("gradient_clipper", "AdaGC.*clipper state"),
        ("weight_decay_scheduler", "weight-decay scheduler.*state"),
        ("resume_compatibility", "strict resume.*provenance"),
    ],
)
def test_modern_checkpoint_preflight_requires_transactional_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    """Reject modern checkpoints that would fail after distributed launch."""

    preflight = _load_preflight()
    checkpoint = tmp_path / "checkpoint_last.pt"
    checkpoint.touch()
    payload = _modern_checkpoint_payload()
    payload[field] = None
    monkeypatch.setattr(preflight, "load_checkpoint", lambda *args, **kwargs: payload)

    with pytest.raises(RuntimeError, match=message):
        preflight.check_training_checkpoint(
            checkpoint,
            expected_world_size=8,
            expected_stage="finetune",
        )


def test_checkpoint_preflight_rejects_a_different_run_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not evaluate a best checkpoint from another strict training run."""

    preflight = _load_preflight()
    selected = tmp_path / "checkpoint_best.pt"
    reference = tmp_path / "checkpoint_last.pt"
    selected.touch()
    reference.touch()
    selected_payload = _modern_checkpoint_payload()
    reference_payload = _modern_checkpoint_payload()
    reference_payload["resume_compatibility"] = {
        "training_data.schema": "a2v2.training-data.v2",
        "training_data.manifest_sha256": "current-run",
    }
    payloads = {selected.resolve(): selected_payload, reference.resolve(): reference_payload}
    monkeypatch.setattr(
        preflight,
        "load_checkpoint",
        lambda path, **kwargs: payloads[Path(path).resolve()],
    )

    with pytest.raises(RuntimeError, match="resume fingerprint differs"):
        preflight.check_training_checkpoint(
            selected,
            expected_world_size=8,
            expected_stage="finetune",
            reference_path=reference,
        )


def test_bitsandbytes_preflight_requires_the_pinned_cuda_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept the supported release only when its native CUDA library loaded."""

    preflight = _load_preflight()
    package = ModuleType("bitsandbytes")
    package.__version__ = "0.50.0"  # type: ignore[attr-defined]
    extension = ModuleType("bitsandbytes.cextension")
    extension.lib = SimpleNamespace(
        compiled_with_cuda=True,
        _name="libbitsandbytes_cuda132.so",
    )
    monkeypatch.setitem(sys.modules, "bitsandbytes", package)
    monkeypatch.setitem(sys.modules, "bitsandbytes.cextension", extension)

    report = preflight.check_bitsandbytes()

    assert report == {
        "version": "0.50.0",
        "compiled_with_cuda": True,
        "native_library": "libbitsandbytes_cuda132.so",
    }


@pytest.mark.parametrize(
    ("version", "compiled", "message"),
    [
        ("0.49.2", True, "requires bitsandbytes >=0.50,<0.51"),
        ("0.50.0", False, "without CUDA support"),
    ],
)
def test_bitsandbytes_preflight_rejects_an_unsupported_runtime(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    compiled: bool,
    message: str,
) -> None:
    """Fail before training for an incompatible release or CPU-only backend."""

    preflight = _load_preflight()
    package = ModuleType("bitsandbytes")
    package.__version__ = version  # type: ignore[attr-defined]
    extension = ModuleType("bitsandbytes.cextension")
    extension.lib = SimpleNamespace(compiled_with_cuda=compiled, _name="fake.so")
    monkeypatch.setitem(sys.modules, "bitsandbytes", package)
    monkeypatch.setitem(sys.modules, "bitsandbytes.cextension", extension)

    with pytest.raises(RuntimeError, match=message):
        preflight.check_bitsandbytes()
