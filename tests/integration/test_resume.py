"""Compare an uninterrupted optimizer update with one performed after native checkpoint
restoration. The test covers model, teacher, optimizer, scheduler, counters, and random
state together."""

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import a2v2.training as training
import a2v2.workflows as workflows
from a2v2.config import config_to_dict, load_config
from a2v2.model import EMATeacher
from a2v2.training import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from a2v2.training import TrainingEngine
from a2v2.training import CosineUpdateScheduler


class Result:
    """Store the summed loss and sample size expected by the training engine."""
    def __init__(self, loss: torch.Tensor, sample_size: int) -> None:
        self.loss = loss
        self.sample_size = sample_size


class TeacherFixture(nn.Module):
    """Keep a tiny registered EMA teacher on the real engine update boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.student = nn.Linear(2, 1)
        self.teacher = EMATeacher(self.student)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run only the trainable student."""

        return self.student(value)

    def update_teacher(self, update: int) -> None:
        """Move the teacher on the same successful-update clock as production."""

        self.teacher.update(self.student, decay=0.5)


def _components() -> tuple[nn.Linear, torch.optim.Optimizer, CosineUpdateScheduler]:
    """Construct the tiny model, optimizer, scheduler, and engine under test."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = CosineUpdateScheduler(
        optimizer, max_lr=1e-2, min_lr=1e-4, warmup_updates=1, max_updates=4
    )
    return model, optimizer, scheduler


def _step(engine: TrainingEngine, model: nn.Linear) -> None:
    """Perform one synthetic training update and return its measured result."""
    batches = [torch.randn(2, 2), torch.randn(1, 2)]

    def forward(value: torch.Tensor) -> Result:
        """Run the minimal model-specific forward path used by this test fixture."""
        prediction = model(value)
        return Result(prediction.square().sum(), value.shape[0])

    engine.step(batches, forward)


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    """Compare tensor-bearing optimizer, RNG, and runtime state exactly."""

    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _adagc_components() -> tuple[
    TeacherFixture,
    torch.optim.Optimizer,
    CosineUpdateScheduler,
    training.GradientClipper,
]:
    """Construct a tiny stateful AdaGC plus teacher resume stack."""

    model = TeacherFixture()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-2,
    )
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-2,
        min_lr=1e-4,
        warmup_updates=1,
        max_updates=4,
    )
    clipper = training.build_gradient_clipper(
        model,
        method="adagc",
        clip_norm=1.0,
        adagc_beta=0.99,
        adagc_relative_clip=1.04,
        adagc_warmup_updates=1,
    )
    return model, optimizer, scheduler, clipper


def _adagc_engine() -> tuple[TeacherFixture, TrainingEngine]:
    """Build the engine around the exact AdaGC resume fixture."""

    model, optimizer, scheduler, clipper = _adagc_components()
    return model, TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
        gradient_clipper=clipper,
    )


def _decay_engine(
    *,
    max_updates: int = 4,
    weight_decay_end: float | None = 0.02,
) -> tuple[nn.Linear, TrainingEngine]:
    """Build a tiny engine with an update-zero cosine decay schedule."""

    model = nn.Linear(2, 1)
    optimizer = training.build_optimizer(
        model,
        name="adam",
        learning_rate=1e-2,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.2,
    )
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-2,
        min_lr=1e-4,
        warmup_updates=1,
        max_updates=max_updates,
    )
    decay = training.build_weight_decay_scheduler(
        optimizer,
        schedule="cosine",
        weight_decay_end=weight_decay_end,
        max_updates=max_updates,
    )
    return model, TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
        weight_decay_scheduler=decay,
    )


def _teacher_step(engine: TrainingEngine, model: TeacherFixture) -> None:
    """Consume RNG and execute one tiny teacher-backed training update."""

    batches = [torch.randn(2, 2), torch.randn(1, 2)]
    engine.step(
        batches,
        lambda value: Result(model(value).square().sum(), value.shape[0]),
    )


def test_preemption_cli_returns_requeue_code_only_after_checkpoint_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a valid preemption checkpoint surfacing as an uncaught traceback or success."""

    from a2v2.slurm import TrainingPreempted

    checkpoint_path = Path("checkpoint_last.pt")

    def preempted(*_: object, **__: object) -> Path:
        """Model a workflow that already wrote its valid safe-point checkpoint."""

        raise TrainingPreempted(checkpoint_path, update=7)

    monkeypatch.setattr(workflows, "run_training", preempted)

    assert workflows.train_main([
        "--config",
        "tests/fixtures/tiny_pretrain.yaml",
        "--device",
        "cpu",
    ]) == 75


@pytest.mark.parametrize("format_version", (1, 2))
def test_checkpoint_resume_matches_uninterrupted_next_update(
    tmp_path: Path,
    format_version: int,
) -> None:
    """Check checkpoint resume matches uninterrupted next update."""
    torch.manual_seed(31)
    model, optimizer, scheduler = _components()
    engine = TrainingEngine(model, optimizer, scheduler, clip_norm=1.0, device=torch.device("cpu"))
    _step(engine, model)
    path = tmp_path / "resume.pt"
    payload = engine.checkpoint_payload(stage="pretrain", config={"tiny": True})
    if format_version == 1:
        payload["format_version"] = 1
        for key in (
            "gradient_clipper",
            "weight_decay_scheduler",
            "topology",
            "resume_compatibility",
        ):
            payload.pop(key)
        torch.save(payload, path)
    else:
        save_checkpoint(path, payload)

    _step(engine, model)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    resumed_model, resumed_optimizer, resumed_scheduler = _components()
    resumed_engine = TrainingEngine(
        resumed_model, resumed_optimizer, resumed_scheduler, clip_norm=1.0, device=torch.device("cpu")
    )
    checkpoint = load_checkpoint(path)
    resumed_engine.restore(checkpoint)
    restore_rng_state(checkpoint["rng_state"])
    _step(resumed_engine, resumed_model)

    for name, value in resumed_model.state_dict().items():
        assert torch.equal(value, expected[name])
    assert resumed_engine.update == engine.update == 2


def test_v1_checkpoint_upgrades_without_changing_inference_state(tmp_path: Path) -> None:
    """Catch a v1 reader that rejects legacy payloads or rewrites model tensors."""

    torch.manual_seed(5)
    model, optimizer, scheduler = _components()
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
    )
    legacy = engine.checkpoint_payload(stage="finetune", config={"tiny": True})
    legacy["format_version"] = 1
    for key in (
        "gradient_clipper",
        "weight_decay_scheduler",
        "topology",
        "resume_compatibility",
    ):
        legacy.pop(key, None)
    path = tmp_path / "legacy-v1.pt"
    torch.save(legacy, path)

    loaded = load_checkpoint(path)

    assert loaded["format_version"] == 1
    assert loaded["gradient_clipper"] is None
    assert loaded["weight_decay_scheduler"] is None
    assert loaded["topology"] is None
    assert loaded["resume_compatibility"] is None
    assert loaded["config"] == legacy["config"]
    for name, tensor in legacy["model"].items():
        assert torch.equal(loaded["model"][name], tensor)


@pytest.mark.parametrize("missing_key", sorted(training.LEGACY_REQUIRED_KEYS))
def test_v1_checkpoint_rejects_each_missing_legacy_required_key(
    tmp_path: Path,
    missing_key: str,
) -> None:
    """Catch a v1 validator that checks only a representative legacy subset."""

    model, optimizer, scheduler = _components()
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
    )
    payload = engine.checkpoint_payload(stage="pretrain", config={"tiny": True})
    payload["format_version"] = 1
    for key in training.VERSION_2_STATE_KEYS:
        payload.pop(key)
    payload.pop(missing_key)

    with pytest.raises(training.CheckpointError, match=missing_key):
        save_checkpoint(tmp_path / f"missing-{missing_key}.pt", payload)


@pytest.mark.parametrize("missing_key", sorted(training.VERSION_2_STATE_KEYS))
def test_v2_checkpoint_rejects_each_missing_required_state_slot(
    tmp_path: Path,
    missing_key: str,
) -> None:
    """Catch optional treatment of any required v2 runtime slot."""

    model, engine = _adagc_engine()
    payload = engine.checkpoint_payload(stage="pretrain", config={"active": {}})
    payload.pop(missing_key)

    with pytest.raises(training.CheckpointError, match=missing_key):
        save_checkpoint(tmp_path / f"missing-{missing_key}.pt", payload)


def test_new_checkpoint_save_uses_v2_reserved_state_slots(tmp_path: Path) -> None:
    """Catch new writers that silently retain format v1 or omit reserved state."""

    model, engine = _adagc_engine()
    _teacher_step(engine, model)
    payload = engine.checkpoint_payload(
        stage="pretrain",
        config={"active": {"stage": "pretrain"}},
    )
    path = tmp_path / "v2.pt"
    save_checkpoint(path, payload)
    raw = torch.load(path, map_location="cpu", weights_only=False)

    assert raw["format_version"] == 2
    _assert_tree_equal(raw["gradient_clipper"], engine.gradient_clipper.state_dict())
    assert raw["weight_decay_scheduler"] is None
    assert raw["topology"] is None
    assert "resume_compatibility" in raw


def test_cosine_weight_decay_resume_matches_every_next_state_exactly(
    tmp_path: Path,
) -> None:
    """Catch resume drift in decay, model, optimizer, scheduler, or RNG state."""

    torch.manual_seed(79)
    model, engine = _decay_engine()
    first = engine.step(
        [torch.randn(2, 2)],
        lambda value: Result(model(value).square().sum(), value.shape[0]),
    )
    expected_first_decay = 0.02 + 0.5 * (0.2 - 0.02) * (
        1 + torch.cos(torch.tensor(torch.pi / 4)).item()
    )
    assert first.weight_decay == pytest.approx(expected_first_decay)
    path = tmp_path / "decay-resume.pt"
    save_checkpoint(
        path,
        engine.checkpoint_payload(stage="pretrain", config={"active": {}}),
    )

    engine.step(
        [torch.randn(2, 2)],
        lambda value: Result(model(value).square().sum(), value.shape[0]),
    )
    assert engine.weight_decay_scheduler is not None
    expected = {
        "model": model.state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "weight_decay_scheduler": engine.weight_decay_scheduler.state_dict(),
        "update": engine.update,
        "rng": capture_rng_state(),
    }

    resumed_model, resumed_engine = _decay_engine()
    resumed_engine.restore(load_checkpoint(path))
    resumed_engine.step(
        [torch.randn(2, 2)],
        lambda value: Result(
            resumed_model(value).square().sum(),
            value.shape[0],
        ),
    )
    assert resumed_engine.weight_decay_scheduler is not None
    actual = {
        "model": resumed_model.state_dict(),
        "optimizer": resumed_engine.optimizer.state_dict(),
        "scheduler": resumed_engine.scheduler.state_dict(),
        "weight_decay_scheduler": resumed_engine.weight_decay_scheduler.state_dict(),
        "update": resumed_engine.update,
        "rng": capture_rng_state(),
    }

    _assert_tree_equal(actual, expected)


def test_cosine_weight_decay_resume_recomputes_extended_horizon_before_next_update() -> None:
    """Catch restore that keeps the saved short-horizon decay after extension."""

    torch.manual_seed(83)
    saved_model, saved_engine = _decay_engine(
        max_updates=4,
        weight_decay_end=None,
    )
    zero_batch = torch.ones(1, 2)
    for _ in range(2):
        saved_engine.step(
            [zero_batch],
            lambda value: Result(saved_model(value).sum() * 0.0, value.shape[0]),
        )
    checkpoint = saved_engine.checkpoint_payload(
        stage="pretrain",
        config={"active": {}},
    )
    assert checkpoint["update"] == 2
    assert checkpoint["optimizer"]["param_groups"][0]["weight_decay"] == pytest.approx(0.1)

    resumed_model, resumed_engine = _decay_engine(
        max_updates=8,
        weight_decay_end=None,
    )
    resumed_engine.restore(checkpoint)
    expected_restore_decay = 0.1 * (1 + math.cos(math.pi * 2 / 8))
    expected_restore_lr = 1e-4 + 0.5 * (1e-2 - 1e-4) * (
        1 + math.cos(math.pi * 1 / 7)
    )
    assert resumed_engine.weight_decay_scheduler is not None
    assert resumed_engine.weight_decay_scheduler.state_dict() == {"last_update": 2}
    assert resumed_engine.optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        expected_restore_decay
    )
    assert resumed_engine.optimizer.param_groups[0]["lr"] == pytest.approx(
        expected_restore_lr
    )

    weight_before = resumed_model.weight.detach().clone()
    bias_before = resumed_model.bias.detach().clone()
    result = resumed_engine.step(
        [zero_batch],
        lambda value: Result(resumed_model(value).sum() * 0.0, value.shape[0]),
    )
    expected_next_decay = 0.1 * (1 + math.cos(math.pi * 3 / 8))

    torch.testing.assert_close(
        resumed_model.weight,
        weight_before * (1 - expected_restore_lr * expected_restore_decay),
    )
    assert torch.equal(resumed_model.bias, bias_before)
    assert result.update == 3
    assert result.weight_decay == pytest.approx(expected_next_decay)
    assert resumed_engine.weight_decay_scheduler.state_dict() == {"last_update": 3}


def test_cosine_weight_decay_resume_rejects_missing_past_update_zero_state() -> None:
    """Catch silent reinitialization of a stateful decay clock after update zero."""

    model, engine = _decay_engine()
    pristine = engine.checkpoint_payload(stage="pretrain", config={"active": {}})
    _step(engine, model)
    checkpoint = engine.checkpoint_payload(stage="pretrain", config={"active": {}})
    checkpoint["weight_decay_scheduler"] = None
    assert engine.weight_decay_scheduler is not None
    before = deepcopy(engine.weight_decay_scheduler.state_dict())

    with pytest.raises(
        training.CheckpointError,
        match="weight-decay scheduler.*update 1.*state",
    ):
        engine.restore(checkpoint)

    assert engine.weight_decay_scheduler.state_dict() == before
    pristine["weight_decay_scheduler"] = None
    engine.restore(pristine)
    assert engine.weight_decay_scheduler.state_dict() == {"last_update": 0}
    assert [group["weight_decay"] for group in engine.optimizer.param_groups] == pytest.approx([0.2, 0.0])


def test_cosine_weight_decay_clock_mismatch_does_not_mutate_live_state() -> None:
    """Catch installing decay state before comparing its checkpoint clock."""

    model, engine = _decay_engine()
    _step(engine, model)
    checkpoint = engine.checkpoint_payload(stage="pretrain", config={"active": {}})
    checkpoint["weight_decay_scheduler"] = {"last_update": 2}
    assert engine.weight_decay_scheduler is not None
    before_state = deepcopy(engine.weight_decay_scheduler.state_dict())
    before_decay = [group["weight_decay"] for group in engine.optimizer.param_groups]

    with pytest.raises(training.CheckpointError, match="does not match checkpoint update"):
        engine.restore(checkpoint)

    assert engine.weight_decay_scheduler.state_dict() == before_state
    assert [group["weight_decay"] for group in engine.optimizer.param_groups] == before_decay


def test_adagc_resume_past_update_zero_rejects_missing_state() -> None:
    """Catch a resumed AdaGC run that silently reinitializes historical norms."""

    model, engine = _adagc_engine()
    pristine = engine.checkpoint_payload(stage="pretrain", config={"active": {}})
    _teacher_step(engine, model)
    checkpoint = engine.checkpoint_payload(stage="pretrain", config={"active": {}})
    checkpoint["gradient_clipper"] = None
    before_reject = deepcopy(engine.gradient_clipper.state_dict())

    with pytest.raises(
        training.CheckpointError,
        match="AdaGC.*update 1.*clipper state",
    ):
        engine.restore(checkpoint)
    _assert_tree_equal(engine.gradient_clipper.state_dict(), before_reject)

    pristine["gradient_clipper"] = None
    engine.restore(pristine)
    expected_model, expected_engine = _adagc_engine()
    del expected_model
    _assert_tree_equal(
        engine.gradient_clipper.state_dict(),
        expected_engine.gradient_clipper.state_dict(),
    )


def test_adagc_restore_clock_mismatch_does_not_mutate_live_clipper() -> None:
    """Catch validate-after-install behavior for a clipper/checkpoint clock mismatch."""

    model, engine = _adagc_engine()
    _teacher_step(engine, model)
    checkpoint = engine.checkpoint_payload(stage="pretrain", config={"active": {}})
    checkpoint["gradient_clipper"] = deepcopy(checkpoint["gradient_clipper"])
    checkpoint["gradient_clipper"]["update"] = 2
    before = deepcopy(engine.gradient_clipper.state_dict())

    with pytest.raises(training.CheckpointError, match="does not match checkpoint update"):
        engine.restore(checkpoint)

    _assert_tree_equal(engine.gradient_clipper.state_dict(), before)


def test_adagc_resume_matches_every_next_state_exactly(tmp_path: Path) -> None:
    """Catch resume drift in model, optimizer, clipper, scheduler, teacher, or RNG."""

    torch.manual_seed(73)
    model, engine = _adagc_engine()
    _teacher_step(engine, model)
    path = tmp_path / "adagc-resume.pt"
    save_checkpoint(
        path,
        engine.checkpoint_payload(stage="pretrain", config={"active": {}}),
    )

    _teacher_step(engine, model)
    expected = {
        "model": model.state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "gradient_clipper": engine.gradient_clipper.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "teacher": model.teacher.model.state_dict(),
        "update": engine.update,
        "rng": capture_rng_state(),
    }

    resumed_model, resumed_engine = _adagc_engine()
    resumed_engine.restore(load_checkpoint(path))
    _teacher_step(resumed_engine, resumed_model)
    actual = {
        "model": resumed_model.state_dict(),
        "optimizer": resumed_engine.optimizer.state_dict(),
        "gradient_clipper": resumed_engine.gradient_clipper.state_dict(),
        "scheduler": resumed_engine.scheduler.state_dict(),
        "teacher": resumed_model.teacher.model.state_dict(),
        "update": resumed_engine.update,
        "rng": capture_rng_state(),
    }

    _assert_tree_equal(actual, expected)


def test_training_workflow_wires_adagc_and_persists_resume_fingerprint(
) -> None:
    """Catch a config strategy that never reaches the engine or v2 checkpoint."""

    config = load_config(Path("tests/fixtures/tiny_pretrain.yaml"))
    config = replace(
        config,
        optimization=replace(
            config.optimization,
            max_update=2,
            gradient_clip_method="adagc",
            adagc_warmup_updates=1,
        ),
        optimizer=replace(
            config.optimizer,
            weight_decay=0.2,
            weight_decay_schedule="cosine",
            weight_decay_end=0.02,
        ),
    )
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.2)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.1,
        min_lr=0.01,
        warmup_updates=0,
        max_updates=2,
    )
    clipper = workflows._build_gradient_clipper_for_config(model, config)
    decay = workflows._build_weight_decay_scheduler_for_config(optimizer, config)
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=config.optimization.clip_norm,
        device=torch.device("cpu"),
        gradient_clipper=clipper,
        weight_decay_scheduler=decay,
    )
    engine.step(
        [torch.ones(1, 2)],
        lambda value: Result(model(value).square().sum(), sample_size=1),
    )
    checkpoint = engine.checkpoint_payload(
        stage="pretrain",
        config={"active": config_to_dict(config)},
    )

    assert checkpoint["gradient_clipper"]["update"] == 1
    assert checkpoint["weight_decay_scheduler"] == {"last_update": 1}
    assert decay is not None
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.11)
    assert checkpoint["resume_compatibility"]["optimization.gradient_clip_method"] == "adagc"
    assert checkpoint["resume_compatibility"]["common.seed"] == config.common.seed


def test_optimizer_run_metadata_records_only_selected_bitsandbytes_version() -> None:
    """Catch run summaries that hide the selected optional library version."""

    model = nn.Linear(2, 1)
    native = torch.optim.Adam(model.parameters(), lr=1e-3)
    fake_bnb = torch.optim.Adam(model.parameters(), lr=1e-3)
    fake_bnb._a2v2_bitsandbytes_version = "0.49.1"

    assert workflows._optimizer_run_metadata(native) == {}
    assert workflows._optimizer_run_metadata(fake_bnb) == {
        "bitsandbytes_version": "0.49.1"
    }


def test_resume_compatibility_rejects_every_mathematical_state_mismatch() -> None:
    """Catch omitted optimizer, clipping, decay, model, batching, topology, or seed checks."""

    config = load_config(Path("tests/fixtures/tiny_pretrain.yaml"))
    checkpoint = {
        "config": {"active": config_to_dict(config)},
        "resume_compatibility": None,
    }
    mismatches = {
        "optimizer.name": replace(
            config,
            optimizer=replace(config.optimizer, name="adamw"),
        ),
        "optimization.gradient_clip_method": replace(
            config,
            optimization=replace(config.optimization, gradient_clip_method="none"),
        ),
        "optimization.adagc_beta": replace(
            config,
            optimization=replace(config.optimization, adagc_beta=0.98),
        ),
        "optimizer.weight_decay_schedule": replace(
            config,
            optimizer=replace(config.optimizer, weight_decay_schedule="cosine"),
        ),
        "model.ffn_type": replace(
            config,
            model=replace(config.model, ffn_type="geglu"),
        ),
        "dataset.max_tokens": replace(
            config,
            dataset=replace(config.dataset, max_tokens=config.dataset.max_tokens + 1),
        ),
        "distributed.requested_world_size": replace(
            config,
            distributed=replace(config.distributed, requested_world_size=2),
        ),
        "common.seed": replace(
            config,
            common=replace(config.common, seed=config.common.seed + 1),
        ),
    }

    for path, active in mismatches.items():
        with pytest.raises(training.CheckpointError, match=path.replace(".", r"\.")):
            workflows._validate_resume_compatibility(active, checkpoint)


def test_strict_resume_fingerprint_binds_ordered_task_and_manifest_semantics(
    tmp_path: Path,
) -> None:
    """Reject changed labels, subset, manifest bytes, population, or sizes."""

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = manifests / "train.tsv"
    manifest.write_text("/audio\na.wav\t64\nb.wav\t80\n", encoding="utf-8")
    config = load_config(
        Path("tests/fixtures/tiny_finetune.yaml"),
        overrides=(
            f"task.data={manifests}",
            "checkpoint.resume_policy=strict",
        ),
    )
    dataset = SimpleNamespace(sizes=(64, 80))
    provenance = workflows._training_data_resume_provenance(config, dataset)
    saved = training.resume_compatibility_fingerprint(
        config_to_dict(config),
        training_data=provenance,
    )
    assert saved is not None
    checkpoint = {
        "config": {"active": config_to_dict(config)},
        "resume_compatibility": saved,
    }

    assert saved["task.unique_labels"] == list(config.task.unique_labels)
    assert saved["training_data.manifest.sha256"]
    assert saved["training_data.sampler.population"] == 2

    mutations = {
        "task.unique_labels": replace(
            config,
            task=replace(
                config.task,
                unique_labels=tuple(reversed(config.task.unique_labels)),
            ),
        ),
        "dataset.train_subset": replace(
            config,
            dataset=replace(config.dataset, train_subset="alternate"),
        ),
    }
    (manifests / "alternate.tsv").write_bytes(manifest.read_bytes())
    moved_manifests = tmp_path / "moved-manifests"
    moved_manifests.mkdir()
    (moved_manifests / "train.tsv").write_bytes(manifest.read_bytes())
    mutations["task.data"] = replace(
        config,
        task=replace(config.task, data=moved_manifests),
    )
    for expected_path, active in mutations.items():
        active_provenance = workflows._training_data_resume_provenance(
            active, dataset
        )
        with pytest.raises(training.CheckpointError, match=expected_path.replace(".", r"\.")):
            workflows._validate_resume_compatibility(
                active,
                checkpoint,
                training_data=active_provenance,
            )

    manifest.write_text("/audio\na.wav\t64\nc.wav\t80\n", encoding="utf-8")
    changed_manifest = workflows._training_data_resume_provenance(config, dataset)
    with pytest.raises(training.CheckpointError, match=r"manifest\.sha256"):
        workflows._validate_resume_compatibility(
            config,
            checkpoint,
            training_data=changed_manifest,
        )

    manifest.write_text("/audio\na.wav\t64\nb.wav\t80\n", encoding="utf-8")
    changed_sizes = workflows._training_data_resume_provenance(
        config,
        SimpleNamespace(sizes=(64, 79, 80)),
    )
    with pytest.raises(training.CheckpointError, match=r"sampler\.population"):
        workflows._validate_resume_compatibility(
            config,
            checkpoint,
            training_data=changed_sizes,
        )
    changed_size_values = workflows._training_data_resume_provenance(
        config,
        SimpleNamespace(sizes=(64, 79)),
    )
    with pytest.raises(training.CheckpointError, match=r"sampler\.sizes_sha256"):
        workflows._validate_resume_compatibility(
            config,
            checkpoint,
            training_data=changed_size_values,
        )


def test_strict_resume_rejects_data_mismatch_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Validate provenance before mutable training objects can be restored."""

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "pretrain.tsv").write_text(
        "/audio\na.wav\t64\n", encoding="utf-8"
    )
    saved_config = load_config(
        Path("tests/fixtures/tiny_pretrain.yaml"),
        overrides=(
            f"task.data={manifests}",
            "checkpoint.resume_policy=strict",
        ),
    )
    class DatasetSentinel:
        """Expose only the read-only population used before model construction."""

        sizes = (64,)

        def __len__(self) -> int:
            return 1

    dataset = DatasetSentinel()
    provenance = workflows._training_data_resume_provenance(saved_config, dataset)
    checkpoint = {
        "resume_compatibility": training.resume_compatibility_fingerprint(
            config_to_dict(saved_config), training_data=provenance
        ),
        "topology": None,
        "sampler_state": None,
    }
    pristine_checkpoint = deepcopy(checkpoint)
    active = replace(
        saved_config,
        task=replace(saved_config.task, normalize=not saved_config.task.normalize),
    )
    monkeypatch.setattr(
        workflows,
        "_distributed_device",
        lambda *_: (torch.device("cpu"), 0, 1, False),
    )
    monkeypatch.setattr(workflows, "_make_dataset", lambda *_: dataset)
    monkeypatch.setattr(workflows, "load_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        workflows,
        "_make_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model construction must not run")
        ),
    )

    with pytest.raises(training.CheckpointError, match=r"task\.normalize"):
        workflows._run_training(
            active,
            device_name="cpu",
            resume_path=tmp_path / "resume.pt",
            pretrained_checkpoint=None,
        )
    assert checkpoint == pristine_checkpoint


def test_legacy_multiworker_random_crop_resume_policy_warns_or_rejects() -> None:
    """Do not describe worker-local random crops as exact under strict policy."""

    base = load_config(Path("tests/fixtures/tiny_pretrain.yaml"))
    dataset = SimpleNamespace(sizes=(64, 80))
    compatible = replace(
        base,
        dataset=replace(
            base.dataset,
            num_workers=2,
            max_tokens=200,
            crop_strategy="legacy",
        ),
    )
    sampler = workflows.TokenBatchSampler(
        dataset.sizes,
        max_tokens=compatible.dataset.max_tokens,
        shuffle=False,
    )
    with pytest.warns(RuntimeWarning, match="not bit-exact"):
        workflows._validate_crop_resume_policy(compatible, dataset, sampler)

    strict = replace(
        compatible,
        checkpoint=replace(compatible.checkpoint, resume_policy="strict"),
    )
    with pytest.raises(training.CheckpointError, match="not bit-exact"):
        workflows._validate_crop_resume_policy(strict, dataset, sampler)


def test_workflow_allows_only_v1_update_zero_global_to_adagc_transition(
    tmp_path: Path,
) -> None:
    """Catch workflow compatibility hiding the designed missing-state exception."""

    legacy_config = load_config(Path("tests/fixtures/tiny_pretrain.yaml"))
    active_adagc = replace(
        legacy_config,
        optimization=replace(
            legacy_config.optimization,
            gradient_clip_method="adagc",
        ),
    )
    model, optimizer, scheduler = _components()
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=legacy_config.optimization.clip_norm,
        device=torch.device("cpu"),
    )
    legacy = engine.checkpoint_payload(
        stage="pretrain",
        config={"active": config_to_dict(legacy_config)},
    )
    legacy["format_version"] = 1
    for key in training.VERSION_2_STATE_KEYS:
        legacy.pop(key)
    path = tmp_path / "legacy-update-zero.pt"
    save_checkpoint(path, legacy)
    loaded = load_checkpoint(path)

    workflows._validate_resume_compatibility(active_adagc, loaded)

    loaded_after_update = dict(loaded)
    loaded_after_update["update"] = 1
    with pytest.raises(
        training.CheckpointError,
        match=r"optimization\.gradient_clip_method",
    ):
        workflows._validate_resume_compatibility(active_adagc, loaded_after_update)

    changed_seed = replace(
        active_adagc,
        common=replace(active_adagc.common, seed=active_adagc.common.seed + 1),
    )
    with pytest.raises(training.CheckpointError, match=r"common\.seed"):
        workflows._validate_resume_compatibility(changed_seed, loaded)
