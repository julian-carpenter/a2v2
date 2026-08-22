"""Run complete tiny mean-teacher pretraining steps. The suite checks student gradients,
optimizer updates, EMA movement, and independent masking of cloned student views."""

import math
from pathlib import Path

import pytest
import torch

from a2v2.config import load_config
from a2v2.model import Animal2VecPretrainingModel, MaskInfo
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


def test_cls_decoder_restores_only_frame_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude CLS from the unchanged frame-only decoder restore permutation."""

    cfg = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        ("model.use_cls_token=true",),
    )
    model = Animal2VecPretrainingModel.from_config(cfg).eval()
    observed: list[tuple[int, int]] = []
    prepare_input = model.decoder.prepare_input

    def capture_prepare_input(
        unmasked: torch.Tensor,
        mask_info: MaskInfo,
        *,
        noise_std: float,
    ) -> torch.Tensor:
        """Record the frame counts entering and leaving decoder restoration."""

        restored = prepare_input(unmasked, mask_info, noise_std=noise_std)
        observed.append((unmasked.shape[1], restored.shape[1]))
        return restored

    monkeypatch.setattr(model.decoder, "prepare_input", capture_prepare_input)
    output = model(
        torch.randn(2, 64),
        sample_ids=torch.tensor([10, 11]),
        update=0,
    )

    frame_length = output.mask.shape[1]
    retained_frames = frame_length - int(output.mask[0].sum().item())
    assert observed == [(retained_frames, frame_length)]


def test_cls_regression_combines_weighted_loss_and_gradients() -> None:
    """Regress one CLS per clone without weighting the reported sample count."""

    torch.manual_seed(31)
    cls_weight = 0.25
    cfg = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        (
            "model.use_cls_token=true",
            f"model.cls_loss_weight={cls_weight}",
            "model.checkpoint_activations=true",
        ),
    )
    model = Animal2VecPretrainingModel.from_config(cfg).train()
    batch = 2
    output = model(
        torch.randn(batch, 64),
        sample_ids=torch.tensor([20, 21]),
        update=0,
    )

    masked_frames = int(output.mask.sum().item())
    cls_count = batch * cfg.model.clone_batch
    scale = 1 / math.sqrt(output.predictions.shape[-1])
    frame_error = (
        output.predictions[:masked_frames].float()
        - output.targets[:masked_frames]
    ).square()
    cls_error = (
        output.predictions[masked_frames:].float()
        - output.targets[masked_frames:]
    ).square()
    expected_loss = scale * (frame_error.sum() + cls_weight * cls_error.sum())

    assert output.predictions.shape == output.targets.shape
    assert output.predictions.shape[0] == masked_frames + cls_count
    assert output.sample_size == masked_frames + cls_count
    assert torch.allclose(output.loss, expected_loss)

    (output.loss / output.sample_size).backward()
    assert model.student.cls_token.grad is not None
    assert model.student.cls_token.grad.abs().sum() > 0
    assert model.cls_predictor.weight.grad is not None
    assert model.cls_predictor.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


def test_disabled_cls_preserves_frame_only_output_and_state() -> None:
    """Keep legacy modules absent and diagnostics restricted to masked frames."""

    cfg = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    model = Animal2VecPretrainingModel.from_config(cfg).eval()
    output = model(
        torch.randn(1, 64),
        sample_ids=torch.tensor([30]),
        update=0,
    )

    assert not hasattr(model.student, "cls_token")
    assert not hasattr(model.teacher.model, "cls_token")
    assert not hasattr(model, "cls_predictor")
    assert output.sample_size == int(output.mask.sum().item())
    assert output.predictions.shape[0] == output.sample_size


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
