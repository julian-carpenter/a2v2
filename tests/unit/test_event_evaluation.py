"""Test archived-compatible segment matching and aggregate metrics. Fixtures cover perfect
events, misses, false positives, splits, mergers, strict boundaries, overlapping
classes, and negative clips."""

import pytest
import torch

from a2v2.workflows import (
    SegmentedEvaluation,
    aggregate_segmented_metrics,
    legacy_segmented_evaluation,
)


def test_perfect_event_matches_archived_segment_score_and_iou() -> None:
    """Check perfect event matches archived segment score and IoU."""
    probabilities = torch.tensor([[[0.1], [0.8], [0.9], [0.1]]])
    targets = torch.tensor([[[0], [1], [1], [0]]])

    result = legacy_segmented_evaluation(
        probabilities,
        targets,
        method="avg",
        window_frames=1,
        metric_threshold=0.5,
        iou_threshold=0.5,
    )

    assert result.segmented_targets[0, 0, 0] == 1
    assert result.segmented_scores[0, 0, 0] == pytest.approx(0.8)
    assert result.ious[0, 0, 0] == 1
    assert torch.count_nonzero(result.segmented_targets) == 1


def test_aggregation_matches_legacy_macro_ap_and_selects_focal_threshold() -> None:
    """Include every legacy label column in macro AP and score focal events."""
    result = SegmentedEvaluation(
        segmented_scores=torch.tensor([[[0.9, 0.9], [0.8, 0.7], [0.1, 0.6], [0.0, 0.0]]]),
        segmented_targets=torch.tensor([[[1, 1], [1, 0], [0, 1], [0, 0]]]),
        ious=torch.zeros(1, 2, 2),
        splits=torch.zeros(1, 2, 2, dtype=torch.long),
        mergers=torch.zeros(1, 2, 2, dtype=torch.long),
    )

    metrics = aggregate_segmented_metrics([result], ["call", "focal"])

    assert metrics.classwise_average_precision == pytest.approx({
        "call": 1.0,
        "focal": (1.0 + 2 / 3) / 2,
    })
    assert metrics.macro_average_precision == pytest.approx(11 / 12)
    assert metrics.micro_average_precision == pytest.approx(0.95)
    assert metrics.focal_threshold == pytest.approx(0.6)
    assert metrics.focal_f1 == pytest.approx(0.8)


def test_segmented_aggregation_keeps_false_positive_samples() -> None:
    """Count false-positive segments instead of mistaking them for padding.

    This catches the previous ``targets.sum(...) != 0`` filter. A target-free
    row can hold a real predicted event, so dropping it removes the strongest
    false positive in this fixture and changes AP from 0.5 to 1.0.
    """

    result = SegmentedEvaluation(
        segmented_scores=torch.tensor([[[0.6], [0.9], [0.0], [0.0]]]),
        segmented_targets=torch.tensor([[[1], [0], [0], [0]]]),
        ious=torch.tensor([[[0.75], [0.0]]]),
        splits=torch.zeros(1, 2, 1, dtype=torch.long),
        mergers=torch.zeros(1, 2, 1, dtype=torch.long),
    )

    metrics = aggregate_segmented_metrics(
        [result],
        ["call"],
        metric_threshold=0.5,
    )

    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.classwise_average_precision == pytest.approx({"call": 0.5})
    assert metrics.macro_average_precision == pytest.approx(0.5)
    assert metrics.micro_average_precision == pytest.approx(0.5)


def _evaluate(probabilities: list[float], targets: list[int], *, iou_threshold: float = 0.5):
    """Run one compact event-evaluation fixture with supplied truth and predictions."""
    return legacy_segmented_evaluation(
        torch.tensor(probabilities).view(1, -1, 1),
        torch.tensor(targets).view(1, -1, 1),
        method="avg",
        window_frames=1,
        metric_threshold=0.5,
        iou_threshold=iou_threshold,
    )


def test_missing_prediction_and_missing_target_match_archived_samples() -> None:
    """Check missing prediction and missing target match archived samples."""
    missing_prediction = _evaluate(
        [0.1, 0.2, 0.3, 0.1],
        [0, 1, 1, 0],
    )
    missing_target = _evaluate(
        [0.1, 0.8, 0.9, 0.1],
        [0, 0, 0, 0],
    )

    assert missing_prediction.segmented_targets[0, 0, 0] == 1
    assert missing_prediction.segmented_scores[0, 0, 0] == pytest.approx(0.2)
    assert missing_target.segmented_targets[0, 0, 0] == 0
    assert missing_target.segmented_scores[0, 0, 0] == pytest.approx(0.8)


def test_split_and_merger_counts_match_archived_strict_iou_decisions() -> None:
    """Check split and merger counts match archived strict IoU decisions."""
    split = _evaluate(
        [0.0, 0.8, 0.8, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        iou_threshold=0.1,
    )
    merger = _evaluate(
        [0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0, 0.0],
        [0, 1, 1, 0, 1, 1, 0, 0],
        iou_threshold=0.1,
    )

    assert split.splits[0, 0, 0] == 2
    assert torch.allclose(split.segmented_scores[0, :2, 0], torch.tensor([0.9, 0.8]))
    assert torch.allclose(split.ious[0, :2, 0], torch.tensor([1 / 3, 1 / 6]))
    assert merger.mergers[0, 0, 0] == 2


def test_one_frame_overlap_at_legacy_boundary_is_not_an_overlap() -> None:
    """Check one frame overlap at legacy boundary is not an overlap."""
    result = _evaluate(
        [0.0, 0.0, 0.8, 0.8, 0.0],
        [0, 1, 1, 0, 0],
        iou_threshold=0.0,
    )

    assert torch.count_nonzero(result.ious) == 0
    assert torch.equal(result.segmented_targets[0, :2, 0], torch.tensor([1, 0]))


@pytest.mark.parametrize("method", ["avg", "max"])
def test_archived_pooling_aligns_shifted_partial_final_interval(method: str) -> None:
    """Check archived pooling aligns shifted partial final interval."""
    result = legacy_segmented_evaluation(
        torch.tensor([[[0.0], [0.0], [1.0], [0.0], [0.0]]]),
        torch.tensor([[[0], [0], [1], [1], [1]]]),
        method=method,
        window_frames=3,
        metric_threshold=0.3,
        iou_threshold=0.5,
    )

    assert result.segmented_targets[0, 0, 0] == 1
    assert result.ious[0, 0, 0] == 1


def test_overlapping_classes_are_matched_independently() -> None:
    """Check overlapping classes are matched independently."""
    probabilities = torch.tensor([[[0.0, 0.0], [0.8, 0.9], [0.8, 0.9], [0.0, 0.0]]])
    targets = torch.tensor([[[0, 0], [1, 1], [1, 1], [0, 0]]])

    result = legacy_segmented_evaluation(
        probabilities,
        targets,
        method="avg",
        window_frames=1,
        metric_threshold=0.5,
        iou_threshold=0.5,
    )

    assert torch.equal(result.segmented_targets[0, 0], torch.tensor([1, 1]))
    assert torch.equal(result.ious[0, 0], torch.tensor([1.0, 1.0]))


def test_all_negative_clip_produces_only_padding_samples() -> None:
    """Check all negative clip produces only padding samples."""
    result = _evaluate([0.1, 0.2, 0.3, 0.1], [0, 0, 0, 0])

    assert torch.count_nonzero(result.segmented_scores) == 0
    assert torch.count_nonzero(result.segmented_targets) == 0


def test_all_negative_aggregation_is_finite_and_zero() -> None:
    """Check all negative aggregation is finite and zero."""
    result = SegmentedEvaluation(
        segmented_scores=torch.zeros(1, 4, 2),
        segmented_targets=torch.zeros(1, 4, 2, dtype=torch.long),
        ious=torch.zeros(1, 2, 2),
        splits=torch.zeros(1, 2, 2, dtype=torch.long),
        mergers=torch.zeros(1, 2, 2, dtype=torch.long),
    )

    metrics = aggregate_segmented_metrics([result], ["call", "focal"])

    assert metrics.classwise_average_precision == {"call": 0.0, "focal": 0.0}
    assert metrics.macro_average_precision == 0.0
    assert metrics.micro_average_precision == 0.0
    assert metrics.focal_f1 == 0.0
