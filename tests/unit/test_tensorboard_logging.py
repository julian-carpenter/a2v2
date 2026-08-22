"""Read real TensorBoard event files produced by the native experiment logger."""

from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from a2v2.training import UpdateResult
from a2v2.workflows import SegmentedEvaluation, TensorBoardLogger


def _event_tags(directory: Path) -> dict[str, object]:
    """Load every flushed event in one temporary run directory."""

    accumulator = EventAccumulator(str(directory))
    accumulator.Reload()
    return accumulator.Tags()


def test_tensorboard_logger_records_pretraining_and_segmented_validation(
    tmp_path: Path,
) -> None:
    """Protect scalar, PR-curve, and segment-diagnostic observability."""

    logger = TensorBoardLogger(tmp_path, purge_step=None)
    logger.log_update(
        UpdateResult(
            loss=1.25,
            sample_size=32,
            gradient_norm=0.75,
            learning_rate=3e-4,
            update=7,
            skipped=False,
            pred_var=0.42,
            target_var=0.84,
            weight_decay=0.12,
        ),
        stage="pretrain",
        amp_scale=128.0,
    )
    frame_scores = torch.tensor([
        [0.9, 0.8],
        [0.2, 0.7],
        [0.6, 0.1],
    ])
    frame_targets = torch.tensor([
        [1, 1],
        [0, 1],
        [1, 0],
    ])
    segmented = SegmentedEvaluation(
        segmented_scores=torch.tensor([[
            [0.9, 0.8],
            [0.7, 0.2],
            [0.0, 0.0],
        ]]),
        segmented_targets=torch.tensor([[
            [1, 1],
            [0, 0],
            [0, 0],
        ]]),
        ious=torch.tensor([[[0.8, 0.7], [0.0, 0.0]]]),
        splits=torch.tensor([[[2, 0], [0, 0]]]),
        mergers=torch.tensor([[[0, 2], [0, 0]]]),
    )
    logger.log_validation(
        {
            "loss": 0.5,
            "precision": 0.75,
            "recall": 1.0,
            "f1": 6 / 7,
            "average_precision": 0.9,
            "segmented_precision": 0.5,
            "segmented_recall": 1.0,
            "segmented_f1": 2 / 3,
            "segmented_average_precision": 0.75,
        },
        update=7,
        subset="valid",
        labels=("call", "focal"),
        frame_scores=frame_scores,
        frame_targets=frame_targets,
        segmented_evaluations=(segmented,),
        metric_threshold=0.5,
    )
    logger.close()

    tags = _event_tags(tmp_path)
    scalar_tags = set(tags["scalars"])
    assert {
        "train/loss",
        "train/sample_size",
        "train/gradient_norm",
        "train/learning_rate",
        "train/weight_decay",
        "train/skipped",
        "train/amp_scale",
        "pretrain/pred_var",
        "pretrain/target_var",
        "validation/valid/loss",
        "validation/valid/frame/f1",
        "validation/valid/segmented/f1",
        "validation/valid/frame/average_precision/call",
        "validation/valid/segmented/average_precision/call",
    } <= scalar_tags
    assert {
        "validation/valid/frame/pr_micro",
        "validation/valid/frame/pr/call",
        "validation/valid/segmented/pr_micro",
        "validation/valid/segmented/pr/call",
    } <= set(tags["tensors"])
    assert {
        "validation/valid/segmented/iou/call",
        "validation/valid/segmented/splits/call",
        "validation/valid/segmented/mergers/focal",
    } <= set(tags["histograms"])
