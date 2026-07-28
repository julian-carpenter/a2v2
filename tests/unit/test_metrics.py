"""Test additive frame confusion counts and binary average precision. Hand-ranked examples
expose the treatment of class decisions and tied scores."""

import torch
import pytest

import a2v2.training as training
from a2v2.training import FrameCounts, average_precision


def test_frame_counts_and_micro_metrics() -> None:
    """Check frame counts and micro metrics."""
    prediction = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    target = torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.bool)
    counts = FrameCounts.from_predictions(prediction, target)
    assert counts == FrameCounts(true_positive=2, false_positive=1, true_negative=2, false_negative=1)
    assert counts.precision == 2 / 3
    assert counts.recall == 2 / 3
    assert counts.f1 == 2 / 3
    assert counts.accuracy == 4 / 6


def test_counts_accumulate_without_global_state() -> None:
    """Check counts accumulate without global state."""
    first = FrameCounts(1, 2, 3, 4)
    second = FrameCounts(4, 3, 2, 1)
    assert first + second == FrameCounts(5, 5, 5, 5)


def test_average_precision_matches_hand_calculated_ranking() -> None:
    """Check average precision matches hand calculated ranking."""
    scores = torch.tensor([0.9, 0.8, 0.7, 0.1])
    targets = torch.tensor([1, 0, 1, 0])
    assert average_precision(scores, targets) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision(torch.tensor([0.2, 0.1]), torch.zeros(2)) == 0.0


def test_average_precision_groups_tied_scores_like_sklearn() -> None:
    """Check average precision groups tied scores like sklearn."""
    scores = torch.tensor([0.9, 0.5, 0.5, 0.1])
    targets = torch.tensor([1, 1, 0, 0])

    assert average_precision(scores, targets) == pytest.approx(5 / 6)


def test_pretraining_variance_diagnostics_match_archived_sample_std() -> None:
    """Match the historical mean feature-wise sample standard deviation."""

    predictions = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 8.0],
    ])
    targets = torch.tensor([
        [2.0, 1.0],
        [4.0, 5.0],
        [8.0, 7.0],
    ])

    pred_var, target_var = training.pretraining_variance_diagnostics(
        predictions,
        targets,
    )

    # These literals come from the archived equation:
    # mean_d sqrt(sample_variance(z[:, d]) + 1e-6).
    assert pred_var == pytest.approx(2.5275254249572754)
    assert target_var == pytest.approx(3.0550506114959717)


def test_pretraining_variance_diagnostics_reject_bad_shape_or_count() -> None:
    """Reject tensors that cannot define paired feature-wise sample variance."""

    with pytest.raises(ValueError, match="same shape"):
        training.pretraining_variance_diagnostics(
            torch.zeros(3, 2),
            torch.zeros(3, 3),
        )
    with pytest.raises(ValueError, match="at least two"):
        training.pretraining_variance_diagnostics(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
        )
