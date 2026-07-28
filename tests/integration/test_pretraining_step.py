"""Run complete tiny mean-teacher pretraining steps. The suite checks student gradients,
optimizer updates, EMA movement, and independent masking of cloned student views."""

from pathlib import Path

import torch

from a2v2.config import load_config
from a2v2.model import Animal2VecPretrainingModel
from a2v2.training import CosineUpdateScheduler, TrainingEngine


ROOT = Path(__file__).parents[2]


def test_tiny_pretraining_forward_backward_optimizer_and_ema() -> None:
    """Check tiny pretraining forward backward optimizer and EMA."""
    torch.manual_seed(12)
    cfg = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    model = Animal2VecPretrainingModel.from_config(cfg).train()
    optimizer = torch.optim.Adam(model.student.parameters(), lr=1e-3)
    waveform = torch.randn(2, 64)
    sample_ids = torch.tensor([100, 101])

    output = model(waveform, sample_ids=sample_ids, update=0)
    assert output.predictions.shape == output.targets.shape
    assert output.predictions.shape[-1] == 16
    assert output.sample_size > 0
    teacher_before = next(model.teacher.model.parameters()).detach().clone()
    (output.loss / output.sample_size).backward()
    assert any(parameter.grad is not None for parameter in model.student.parameters())
    optimizer.step()
    model.update_teacher(update=1)
    teacher_after = next(model.teacher.model.parameters()).detach()
    assert not torch.equal(teacher_before, teacher_after)
    assert torch.isfinite(output.loss)


def test_cloned_student_masks_are_distinct() -> None:
    """Check cloned student masks are distinct."""
    cfg = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    model = Animal2VecPretrainingModel.from_config(cfg).eval()
    output = model(torch.randn(1, 64), sample_ids=torch.tensor([3]), update=4)
    assert output.mask.shape[0] == 2
    assert not torch.equal(output.mask[0], output.mask[1])


def test_pretraining_engine_reports_finite_variance_across_microbatches() -> None:
    """Return collapse diagnostics for the full accumulated update."""

    torch.manual_seed(19)
    cfg = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    model = Animal2VecPretrainingModel.from_config(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-3,
        min_lr=0.0,
        warmup_updates=0,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
    )
    batches = [
        {
            "source": torch.randn(2, 64),
            "id": torch.tensor([100, 101]),
        },
        {
            "source": torch.randn(2, 64),
            "id": torch.tensor([102, 103]),
        },
    ]

    result = engine.step(
        batches,
        lambda batch: model(
            batch["source"],
            sample_ids=batch["id"],
            update=engine.update,
        ),
    )

    assert result.pred_var is not None
    assert result.target_var is not None
    assert result.pred_var > 0
    assert result.target_var > 0
    assert torch.isfinite(torch.tensor([result.pred_var, result.target_var])).all()
