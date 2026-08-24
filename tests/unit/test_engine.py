"""Test update-level training semantics without a full Animal2Vec model. The suite isolates
distributed sample-size reduction and release of deserialized checkpoint tensors."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import a2v2.training as training
import a2v2.workflows as workflows
from a2v2.config import CommonConfig, load_config
from a2v2.workflows import _restore_and_release_checkpoint
from a2v2.training import TrainingEngine
from a2v2.training import CosineUpdateScheduler


class Result:
    """Store the summed loss and sample size expected by the training engine."""
    def __init__(self, loss: torch.Tensor, sample_size: int) -> None:
        self.loss = loss
        self.sample_size = sample_size


ROOT = Path(__file__).parents[2]


def test_compile_policy_forwards_exact_options_without_rewrapping_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile the existing module without changing checkpoint or optimizer identity."""

    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    state_keys = tuple(model.state_dict())
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    observed: list[tuple[nn.Module, dict[str, object]]] = []

    def capture_compile(module: nn.Module, **options: object) -> None:
        """Record the in-place Module.compile receiver and execution options."""

        observed.append((module, options))

    monkeypatch.setattr(nn.Module, "compile", capture_compile)
    common = CommonConfig(
        torch_compile=True,
        torch_compile_backend="aot_eager",
        torch_compile_mode="reduce-overhead",
        torch_compile_fullgraph=True,
        torch_compile_dynamic=False,
    )

    workflows._compile_model_in_place(model, common)

    assert observed == [(
        model,
        {
            "backend": "aot_eager",
            "mode": "reduce-overhead",
            "fullgraph": True,
            "dynamic": False,
        },
    )]
    assert tuple(model.state_dict()) == state_keys
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids


def test_disabled_compile_policy_does_not_touch_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the frozen eager construction path free of compile side effects."""

    model = nn.Linear(2, 1)

    def reject_compile(*_: object, **__: object) -> None:
        """Fail if the opt-in execution policy leaks into legacy construction."""

        raise AssertionError("disabled compile policy called Module.compile")

    monkeypatch.setattr(nn.Module, "compile", reject_compile)

    workflows._compile_model_in_place(model, CommonConfig())


def test_compile_policy_reports_actionable_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the selected compiler policy when Module.compile rejects setup."""

    def fail_compile(*_: object, **__: object) -> None:
        """Model an unsupported backend or mode discovered during setup."""

        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(nn.Module, "compile", fail_compile)
    common = CommonConfig(
        torch_compile=True,
        torch_compile_backend="aot_eager",
        torch_compile_mode="max-autotune",
        torch_compile_fullgraph=True,
        torch_compile_dynamic=None,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"torch\.compile setup failed.*backend=aot_eager.*mode=max-autotune"
            r".*fullgraph=True.*dynamic=None.*compiler unavailable"
        ),
    ):
        workflows._compile_model_in_place(nn.Linear(2, 1), common)


def test_compile_policy_rejects_global_silent_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not let Dynamo suppress compile failures for an explicitly compiled run."""

    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)

    with pytest.raises(
        RuntimeError,
        match=r"torch\.compile.*suppress_errors=True.*silent eager fallback",
    ):
        workflows._compile_model_in_place(
            nn.Linear(2, 1),
            CommonConfig(torch_compile=True, torch_compile_backend="eager"),
        )


def test_training_compiles_after_device_move_before_optimizer_and_ddp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep optimizer parameters and DDP reducer rooted in the compiled module."""

    class ConstructionStopped(RuntimeError):
        """Stop the workflow after observing the DDP construction boundary."""

    events: list[object] = []

    class TrackingModel(nn.Linear):
        """Record the device transition without changing normal module behavior."""

        def to(self, *args: object, **kwargs: object) -> "TrackingModel":
            """Move the module after recording the construction boundary."""

            events.append("to")
            return super().to(*args, **kwargs)  # type: ignore[return-value]

    model = TrackingModel(2, 1)
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    config = replace(
        config,
        common=replace(
            config.common,
            torch_compile=True,
            torch_compile_backend="aot_eager",
            torch_compile_mode="default",
            torch_compile_fullgraph=False,
            torch_compile_dynamic=True,
        ),
        distributed=replace(config.distributed, requested_world_size=2),
    )

    def capture_compile(module: nn.Module, **options: object) -> None:
        """Record the exact in-place compile receiver and options."""

        assert module is model
        events.append(("compile", options))

    def capture_optimizer(module: nn.Module, **_: object) -> torch.optim.Optimizer:
        """Build a real optimizer after recording its construction boundary."""

        assert module is model
        events.append("optimizer")
        return torch.optim.SGD(module.parameters(), lr=0.1)

    def stop_at_ddp(module: nn.Module, **_: object) -> None:
        """Record DDP construction and stop before dataset side effects."""

        assert module is model
        events.append("ddp")
        raise ConstructionStopped("observed DDP construction")

    monkeypatch.setattr(workflows, "_distributed_device", lambda *_: (torch.device("cpu"), 0, 2, False))
    dataset = type(
        "DatasetSentinel",
        (),
        {"sizes": (8, 8), "__len__": lambda self: 2},
    )()
    monkeypatch.setattr(workflows, "_make_dataset", lambda *_: dataset)
    monkeypatch.setattr(
        workflows,
        "_training_data_resume_provenance",
        lambda *_: {"schema": "a2v2.training-data.v1"},
    )
    monkeypatch.setattr(workflows, "_make_model", lambda *_args, **_kwargs: (model, None))
    monkeypatch.setattr(
        workflows,
        "_build_gradient_clipper_for_config",
        lambda *_: events.append("gradient_clipper") or object(),
    )
    monkeypatch.setattr(workflows, "build_optimizer", capture_optimizer)
    monkeypatch.setattr(workflows, "DistributedDataParallel", stop_at_ddp)
    monkeypatch.setattr(nn.Module, "compile", capture_compile)

    with pytest.raises(ConstructionStopped, match="observed DDP construction"):
        workflows._run_training(
            config,
            device_name="cpu",
            resume_path=None,
            pretrained_checkpoint=None,
        )

    assert events == [
        "to",
        (
            "compile",
            {
                "backend": "aot_eager",
                "mode": "default",
                "fullgraph": False,
                "dynamic": True,
            },
        ),
        "gradient_clipper",
        "optimizer",
        "ddp",
    ]


def test_batch_move_keeps_sample_ids_on_cpu() -> None:
    """Avoid accelerator synchronization when NumPy masking reads stable IDs."""

    sample_ids = torch.tensor([101, 102])
    batch: dict[str, object] = {
        "id": sample_ids,
        "source": torch.randn(2, 8),
        "target": torch.zeros(2, 4, 1),
        "padding_mask": torch.zeros(2, 8, dtype=torch.bool),
        "path": ("first.wav", "second.wav"),
    }

    moved = workflows._move_batch(batch, torch.device("meta"))

    assert moved["id"] is sample_ids
    assert moved["id"].device.type == "cpu"
    assert moved["source"].device.type == "meta"
    assert moved["target"].device.type == "meta"
    assert moved["padding_mask"].device.type == "meta"
    assert moved["path"] == batch["path"]


def test_engine_does_not_materialize_loss_per_microbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialize the reduced update loss once instead of syncing every forward."""

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.1,
        min_lr=0.0,
        warmup_updates=0,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=0.0,
        device=torch.device("cpu"),
    )
    loss_storage: set[int] = set()
    materialized_microbatch_losses: list[int] = []
    tensor_float = torch.Tensor.__float__

    def observe_float(value: torch.Tensor) -> float:
        """Count Python conversion of tensors sharing a forward loss storage."""

        if value.data_ptr() in loss_storage:
            materialized_microbatch_losses.append(value.data_ptr())
        return tensor_float(value)

    monkeypatch.setattr(torch.Tensor, "__float__", observe_float)

    def forward(value: torch.Tensor) -> Result:
        """Record the storage of one differentiable microbatch loss."""

        loss = model(value).square().sum()
        loss_storage.add(loss.data_ptr())
        return Result(loss, sample_size=1)

    result = engine.step(
        [torch.ones(1, 1), torch.full((1, 1), 2.0)],
        forward,
    )

    assert materialized_microbatch_losses == []
    assert result.loss == pytest.approx(2.5)


def test_distributed_update_reports_globally_reduced_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check distributed update reports globally reduced loss."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.1,
        min_lr=0.0,
        warmup_updates=0,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=0.0,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: value.mul_(2))

    result = engine.step(
        [torch.ones(1, 1)],
        lambda value: Result(model(value).square().sum(), sample_size=1),
    )

    assert result.sample_size == 2
    assert result.loss == pytest.approx(1.0)


def test_restore_releases_loaded_checkpoint_payload() -> None:
    """Check restore releases loaded checkpoint payload."""
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.1,
        min_lr=0.01,
        warmup_updates=0,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=0.0,
        device=torch.device("cpu"),
    )
    engine.step(
        [torch.ones(1, 1)],
        lambda value: Result(model(value).square().sum(), sample_size=1),
    )
    checkpoint = engine.checkpoint_payload(stage="pretrain", config={"active": {}})

    restored_model = nn.Linear(1, 1, bias=False)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_scheduler = CosineUpdateScheduler(
        restored_optimizer,
        max_lr=0.1,
        min_lr=0.01,
        warmup_updates=0,
        max_updates=2,
    )
    restored = TrainingEngine(
        restored_model,
        restored_optimizer,
        restored_scheduler,
        clip_norm=0.0,
        device=torch.device("cpu"),
    )

    _restore_and_release_checkpoint(restored, checkpoint)

    assert checkpoint == {}
    assert restored.update == 1
    assert torch.equal(restored_model.weight, model.weight)


def test_adagc_state_commits_only_after_optimizer_step_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch clipper clock changes when an optimizer attempt raises or is skipped."""

    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.2)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.1,
        min_lr=0.01,
        warmup_updates=0,
        max_updates=2,
    )
    clipper = training.build_gradient_clipper(
        model,
        method="adagc",
        clip_norm=1.0,
        adagc_warmup_updates=1,
    )
    weight_decay_scheduler = training.build_weight_decay_scheduler(
        optimizer,
        schedule="cosine",
        weight_decay_end=0.02,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
        gradient_clipper=clipper,
        weight_decay_scheduler=weight_decay_scheduler,
    )

    monkeypatch.setattr(
        optimizer,
        "step",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("step failed")),
    )
    with pytest.raises(RuntimeError, match="step failed"):
        engine.step(
            [torch.ones(1, 1)],
            lambda value: Result(model(value).square().sum(), sample_size=1),
        )

    assert engine.update == 0
    assert scheduler.last_update == -1
    assert clipper.state_dict()["update"] == 0
    assert weight_decay_scheduler is not None
    assert weight_decay_scheduler.state_dict() == {"last_update": 0}
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)


def test_distributed_terminal_optimizer_failure_keeps_decay_uncommitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch decay advancement before the rank-wide terminal step decision."""

    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.2)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.1,
        min_lr=0.01,
        warmup_updates=0,
        max_updates=2,
    )
    decay = training.build_weight_decay_scheduler(
        optimizer,
        schedule="cosine",
        weight_decay_end=0.02,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
        weight_decay_scheduler=decay,
    )

    def reduce_like_two_ranks(value: torch.Tensor) -> None:
        """Mirror a successful peer except for this rank's failed step bit."""

        if value.dtype != torch.int32 or int(value.item()) == 1:
            value.mul_(2)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_like_two_ranks)
    monkeypatch.setattr(
        optimizer,
        "step",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("step failed")),
    )

    with pytest.raises(training.DistributedOptimizerStepError, match="terminal"):
        engine.step(
            [torch.ones(1, 1)],
            lambda value: Result(model(value).square().sum(), sample_size=1),
        )

    assert decay is not None
    assert engine.update == 0
    assert scheduler.last_update == -1
    assert decay.state_dict() == {"last_update": 0}
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    with pytest.raises(training.DistributedOptimizerStepError, match="terminal"):
        engine.checkpoint_payload(stage="pretrain", config={"active": {}})
    with pytest.raises(training.DistributedOptimizerStepError, match="terminal"):
        engine.step([], lambda value: value)
