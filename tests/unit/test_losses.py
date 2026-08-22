"""Test pretraining regression, teacher target normalization, loudness measurement,
waveform mixup, and focal loss against explicit equations."""

import math

import torch
from torch.nn import functional as F

import a2v2.model as model_module
from a2v2.model import (
    RegressionLoss,
    SigmoidFocalLoss,
    a_weighted_level,
    make_teacher_targets,
    mix_waveforms,
)


def test_regression_loss_scales_by_inverse_sqrt_dimension() -> None:
    """Check regression loss scales by inverse sqrt dimension."""
    prediction = torch.zeros(6, 4, requires_grad=True)
    target = torch.ones(6, 4)
    result = RegressionLoss(beta=0.0)(prediction, target)
    assert result.loss.item() == 12.0
    assert result.sample_size == 6
    result.loss.backward()
    assert prediction.grad is not None


def test_smooth_l1_regression_uses_configured_beta_and_scale() -> None:
    """Check smooth l1 regression uses configured beta and scale."""
    prediction = torch.zeros(2, 2)
    target = torch.ones(2, 2)
    result = RegressionLoss(beta=0.5, scale=2.0)(prediction, target)
    assert result.loss.item() == 6.0


def test_teacher_targets_instance_normalize_each_layer_then_average() -> None:
    """Check teacher targets instance normalize each layer then average."""
    first = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]])
    second = first * 2 + 5
    target = make_teacher_targets(
        [first, second], top_k=2, instance_norm_per_layer=True, layer_norm_per_layer=False,
        layer_norm_final=False,
    )
    assert torch.allclose(target.mean(dim=1), torch.zeros(1, 2), atol=1e-6)
    assert torch.allclose(target.var(dim=1, unbiased=False), torch.ones(1, 2), atol=2e-5)


def test_cls_teacher_target_split_keeps_frame_targets_identical() -> None:
    """Prevent the prepended token from changing frame normalization statistics."""

    frame_layers = [
        torch.tensor([[[1.0, 2.0], [3.0, 7.0], [8.0, 4.0]]]),
        torch.tensor([[[2.0, 5.0], [6.0, 1.0], [9.0, 3.0]]]),
    ]
    cls_layers = [
        torch.tensor([[[100.0, -50.0]]]),
        torch.tensor([[[-200.0, 75.0]]]),
    ]
    layers_with_cls = [
        torch.cat((cls, frames), dim=1)
        for cls, frames in zip(cls_layers, frame_layers)
    ]

    frame_target, _ = model_module.make_pretraining_targets(
        layers_with_cls,
        top_k=2,
        instance_norm_per_layer=True,
        layer_norm_per_layer=False,
        layer_norm_final=False,
        use_cls_token=True,
    )
    expected = make_teacher_targets(
        frame_layers,
        top_k=2,
        instance_norm_per_layer=True,
        layer_norm_per_layer=False,
        layer_norm_final=False,
    )

    assert torch.equal(frame_target, expected)


@torch.no_grad()
def test_cls_teacher_target_uses_finite_feature_normalization() -> None:
    """Normalize each CLS vector over features even for instance-norm recipes."""

    first = torch.tensor(
        [
            [
                [1.0, 2.0, 4.0, 8.0],
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 3.0, 5.0, 7.0],
            ],
            [
                [-3.0, 1.0, 5.0, 9.0],
                [2.0, 4.0, 6.0, 8.0],
                [3.0, 6.0, 9.0, 12.0],
            ],
        ]
    )
    second = first.clone()
    second[:, 0] = second[:, 0] * 3 + 7

    _, cls_target = model_module.make_pretraining_targets(
        [first, second],
        top_k=2,
        instance_norm_per_layer=True,
        layer_norm_per_layer=False,
        layer_norm_final=True,
        use_cls_token=True,
    )

    assert cls_target is not None
    assert cls_target.shape == (2, 4)
    assert torch.isfinite(cls_target).all()
    assert torch.allclose(cls_target.mean(dim=-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(
        cls_target.var(dim=-1, unbiased=False),
        torch.ones(2),
        atol=2e-5,
    )


def test_mix_waveforms_matches_between_class_formula_without_gain_weighting() -> None:
    """Check mix waveforms matches between class formula without gain weighting."""
    waveform = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    mixed = mix_waveforms(
        waveform,
        strength=0.25,
        probability=1.0,
        same_ratio=True,
        gain_mode="none",
        sample_rate=8000,
        window_seconds=0.05,
        ratios=torch.tensor([0.25]),
        permutation=torch.tensor([1, 0]),
    )
    normalizer = math.sqrt(0.25**2 + 0.75**2)
    expected = torch.tensor([[2.5, 2.5], [1.5, 1.5]]) / normalizer
    assert torch.allclose(mixed.waveforms, expected)
    assert torch.equal(mixed.permutation, torch.tensor([1, 0]))


def test_partial_per_sample_mixup_draws_ratios_only_for_selected_examples() -> None:
    """Check partial per sample mixup draws ratios only for selected examples."""
    torch.manual_seed(7)
    mixed = mix_waveforms(
        torch.arange(48, dtype=torch.float32).reshape(8, 6),
        strength=0.25,
        probability=0.5,
        same_ratio=False,
        gain_mode="none",
        sample_rate=8000,
        window_seconds=0.05,
    )
    assert 0 < mixed.applied.sum() < mixed.applied.numel()
    assert mixed.ratios.numel() == mixed.applied.sum().item()


def test_a_weighted_level_is_finite_for_silence_and_tones() -> None:
    """Check a weighted level is finite for silence and tones."""
    sample_rate = 8000
    time = torch.arange(800, dtype=torch.float32) / sample_rate
    audio = torch.stack((torch.zeros_like(time), torch.sin(2 * torch.pi * 1000 * time)))
    levels = a_weighted_level(audio, sample_rate, window_seconds=0.05)
    assert levels.shape == (2, 3)
    assert torch.isfinite(levels).all()
    assert torch.all(levels[1] > levels[0])


def test_teacher_targets_and_regression_match_recorded_official_equations() -> None:
    """Retain equation-level parity after removing the archived implementation."""

    torch.manual_seed(24)
    layers = [torch.randn(2, 7, 8) for _ in range(3)]
    normalized = [
        F.instance_norm(layer.float().transpose(1, 2)).transpose(1, 2)
        for layer in layers[-2:]
    ]
    expected_target = (normalized[0] + normalized[1]) / 2

    actual_target = make_teacher_targets(
        layers,
        top_k=2,
        instance_norm_per_layer=True,
        layer_norm_per_layer=False,
        layer_norm_final=False,
    )
    assert torch.allclose(actual_target, expected_target)

    prediction = torch.randn_like(actual_target)
    expected_loss = F.mse_loss(
        prediction.float(),
        expected_target,
        reduction="none",
    )
    expected_loss = (expected_loss * (1 / math.sqrt(8))).sum()
    actual_loss = RegressionLoss()(prediction, actual_target).loss
    assert torch.allclose(actual_loss, expected_loss)


def test_focal_loss_matches_the_recorded_official_equation() -> None:
    """Check focal loss matches the recorded official equation."""
    torch.manual_seed(25)
    logits = torch.randn(3, 5, 4)
    targets = torch.randint(0, 2, logits.shape).float()
    probability = torch.sigmoid(logits.float())
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets,
        reduction="none",
    )
    probability_of_target = (
        probability * targets + (1 - probability) * (1 - targets)
    )
    alpha = 0.25 * targets + 0.75 * (1 - targets)
    expected = (
        cross_entropy * (1 - probability_of_target).square() * alpha
    ).sum()

    actual = SigmoidFocalLoss(reduction="sum")(logits, targets)
    assert torch.allclose(actual, expected)
