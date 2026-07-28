"""Check the diagnostic utility that locates the first difference between continuous and
resumed checkpoints."""

import torch

from tests.gpu.compare_training_checkpoints import _first_difference


def test_checkpoint_comparison_reports_first_diverging_tensor() -> None:
    """Check checkpoint comparison reports first diverging tensor."""
    left = {"model": {"first": torch.tensor([1.0]), "second": torch.tensor([2.0])}}
    right = {"model": {"first": torch.tensor([1.0]), "second": torch.tensor([3.0])}}

    difference = _first_difference(left, right, "checkpoint")

    assert difference == {
        "path": "checkpoint.model.second",
        "exact": False,
        "max_abs_error": 1.0,
        "different_values": 1,
    }
