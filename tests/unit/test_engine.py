"""Test update-level training semantics without a full Animal2Vec model. The suite isolates
distributed sample-size reduction and release of deserialized checkpoint tensors."""

import pytest
import torch
from torch import nn

import a2v2.training as training
from a2v2.workflows import _restore_and_release_checkpoint
from a2v2.training import TrainingEngine
from a2v2.training import CosineUpdateScheduler


class Result:
    """Store the summed loss and sample size expected by the training engine."""
    def __init__(self, loss: torch.Tensor, sample_size: int) -> None:
        self.loss = loss
        self.sample_size = sample_size


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
