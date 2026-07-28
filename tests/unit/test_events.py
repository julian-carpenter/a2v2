"""Test conversion of frame probabilities into event intervals. The suite covers smoothing,
thresholded runs, temporal IoU, and rejection of the unavailable legacy Canny mode."""

import pytest
import torch

from a2v2.workflows import event_iou, fuse_probabilities, pool_probabilities


def test_average_and_max_pooling_preserve_shape() -> None:
    """Check average and max pooling preserve shape."""
    probabilities = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]])
    average = pool_probabilities(probabilities, method="avg", window_frames=3)
    maximum = pool_probabilities(probabilities, method="max", window_frames=3)
    assert average.shape == probabilities.shape
    assert maximum.shape == probabilities.shape
    assert average[2, 0] == pytest.approx(1 / 3)
    assert torch.equal(maximum[:, 0], torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))


def test_thresholding_builds_contiguous_intervals_with_scores() -> None:
    """Check thresholding builds contiguous intervals with scores."""
    probabilities = torch.tensor(
        [[0.1, 0.8], [0.7, 0.9], [0.8, 0.2], [0.1, 0.1], [0.9, 0.1]]
    )
    timestamps = torch.arange(5).float() * 0.1
    events = fuse_probabilities(
        probabilities,
        timestamps,
        threshold=0.5,
        method="avg",
        window_frames=1,
        recording_duration=0.5,
    )
    assert [(event.label_index, event.start_frame, event.end_frame) for event in events] == [
        (0, 1, 3), (0, 4, 5), (1, 0, 2)
    ]
    assert events[0].start_seconds == pytest.approx(0.1)
    assert events[0].end_seconds == pytest.approx(0.3)
    assert events[0].score == pytest.approx(0.75)


def test_event_iou_handles_overlap_and_disjoint_intervals() -> None:
    """Check event IoU handles overlap and disjoint intervals."""
    assert event_iou((1.0, 3.0), (2.0, 4.0)) == pytest.approx(1 / 3)
    assert event_iou((1.0, 2.0), (2.0, 3.0)) == 0.0


def test_rejects_legacy_canny_mode() -> None:
    """Check rejects legacy canny mode."""
    with pytest.raises(ValueError, match="canny"):
        pool_probabilities(torch.ones(4, 1), method="canny", window_frames=3)

