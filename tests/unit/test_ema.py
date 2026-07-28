"""Test exponential-moving-average decay and teacher state. The assertions protect float32
frozen parameters and the exact update equation used for self-distillation."""

import torch
from torch import nn

from a2v2.model import EMATeacher, ema_decay_at_step


def test_ema_decay_anneals_linearly() -> None:
    """Check EMA decay anneals linearly."""
    assert ema_decay_at_step(0.9, 1.0, 0, 10) == 0.9
    assert ema_decay_at_step(0.9, 1.0, 5, 10) == 0.95
    assert ema_decay_at_step(0.9, 1.0, 10, 10) == 1.0
    assert ema_decay_at_step(0.9, 1.0, 20, 10) == 1.0


def test_ema_teacher_is_fp32_frozen_and_updates_after_step() -> None:
    """Check EMA teacher is FP32 frozen and updates after step."""
    student = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        student.weight.zero_()
    ema = EMATeacher(student)
    assert not any(parameter.requires_grad for parameter in ema.model.parameters())
    with torch.no_grad():
        student.weight.fill_(1)
    ema.update(student, decay=0.9)
    assert torch.allclose(ema.model.weight, torch.full_like(ema.model.weight, 0.1))
    assert ema.model.weight.dtype == torch.float32

