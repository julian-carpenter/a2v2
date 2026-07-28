"""Test update-level training semantics without a full Animal2Vec model. The suite isolates
distributed sample-size reduction and release of deserialized checkpoint tensors."""

import pytest
import torch
from torch import nn

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
