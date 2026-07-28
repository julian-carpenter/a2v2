"""Test parameter grouping, Fairseq-compatible Adam, and cosine update scheduling. The
suite protects weight-decay exemptions, epsilon placement, warmup indexing, and
scheduler resume."""

import math

import pytest
import torch
from torch import nn

from a2v2.model import PSwish
from a2v2.training import CosineUpdateScheduler, build_optimizer


class GroupFixture(nn.Module):
    """Hold representative parameter shapes for optimizer grouping tests."""
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.norm = nn.LayerNorm(3)
        self.activation = PSwish(3)
        self.alibi_scale = nn.Parameter(torch.ones(1, 1, 1, 1, 1))


def test_optimizer_excludes_official_parameter_types_from_weight_decay() -> None:
    """Check optimizer excludes official parameter types from weight decay."""
    model = GroupFixture()
    optimizer = build_optimizer(
        model, name="adam", learning_rate=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.1
    )
    decay = {id(parameter) for group in optimizer.param_groups if group["weight_decay"] for parameter in group["params"]}
    assert id(model.linear.weight) in decay
    assert id(model.linear.bias) not in decay
    assert id(model.norm.weight) not in decay
    assert id(model.activation.p_swish_alpha) not in decay
    assert id(model.alibi_scale) not in decay


def test_legacy_adam_name_uses_decoupled_weight_decay() -> None:
    """Check legacy adam name uses decoupled weight decay."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=0.1,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
    )
    model.weight.grad = torch.zeros_like(model.weight)

    optimizer.step()

    assert model.weight.item() == pytest.approx(0.99)


def test_legacy_adam_matches_fairseq_epsilon_placement() -> None:
    """Check legacy adam matches fairseq epsilon placement."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=0.1,
        betas=(0.9, 0.98),
        eps=0.01,
        weight_decay=0.0,
    )
    model.weight.grad = torch.full_like(model.weight, 2.0)

    optimizer.step()

    first_moment = 0.2
    second_moment = 0.08
    step_size = 0.1 * math.sqrt(1 - 0.98) / (1 - 0.9)
    expected = 1.0 - step_size * first_moment / (math.sqrt(second_moment) + 0.01)
    assert model.weight.item() == pytest.approx(expected)


def test_cosine_scheduler_matches_fairseq_update_indexing() -> None:
    """Check cosine scheduler matches fairseq update indexing."""
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_updates=2,
        warmup_init_lr=1e-5,
        max_updates=10,
    )
    assert scheduler.step_update(0) == pytest.approx(1e-5)
    assert scheduler.step_update(1) == pytest.approx(0.000505)
    assert scheduler.step_update(2) == pytest.approx(1e-3)
    expected_nine = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * 7 / 8))
    assert scheduler.step_update(9) == pytest.approx(expected_nine)


def test_scheduler_state_round_trip() -> None:
    """Check scheduler state round trip."""
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = CosineUpdateScheduler(optimizer, max_lr=1e-3, min_lr=0, warmup_updates=1, max_updates=5)
    scheduler.step_update(3)
    state = scheduler.state_dict()
    scheduler.step_update(4)
    scheduler.load_state_dict(state)
    assert scheduler.last_update == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(scheduler.lr_at_update(3))
