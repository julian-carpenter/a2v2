"""End-to-end A2V2 workflows for the Animal2Vec 1.0 baseline.

This top-level module composes the lower-level configuration, data, model, and
training files. Its sections cover event fusion and scoring, checkpoint-backed
inference, conversion of archived Fairseq checkpoints, training orchestration,
and the three installed commands.

The public command functions are ``train_main``, ``infer_main``, and
``convert_checkpoint_main``. Keeping orchestration in one file makes side
effects such as distributed initialization, checkpoint writes, validation, and
event-file output visible without mixing them into model mathematics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import (
    Animal2VecConfig,
    ConfigError,
    config_from_dict,
    config_from_serialized_dict,
    config_to_dict,
    load_config,
)
from .data import (
    AudioDataset,
    DistributedBatchSampler,
    TokenBatchSampler,
    collate_audio,
    feature_timestamps,
    load_audio,
    normalize_waveform,
    resample_waveform,
)
from .model import (
    Animal2VecFineTuningModel,
    Animal2VecPretrainingModel,
)
from .training import (
    CheckpointError,
    CosineUpdateScheduler,
    FrameCounts,
    TrainingEngine,
    UpdateResult,
    average_precision,
    build_optimizer,
    capture_rng_state,
    gather_rank_rng_states,
    load_checkpoint,
    save_checkpoint,
)


# =============================================================================
# TENSORBOARD EXPERIMENT LOGGING
# =============================================================================

def resolve_tensorboard_directory(config: Animal2VecConfig) -> Path:
    """Resolve a recipe's event directory inside its checkpoint directory."""

    configured = config.common.tensorboard_logdir
    return configured if configured.is_absolute() else config.checkpoint.save_dir / configured


def _tensorboard_label(label: str) -> str:
    """Return a stable tag component for one researcher-defined class name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned or "unnamed"


class TensorBoardLogger:
    """Write compact experiment records through PyTorch's TensorBoard API.

    The training workflow creates this object on rank zero only. Methods accept
    detached CPU tensors at validation boundaries, which keeps event writing
    outside model autograd and avoids retaining accelerator allocations.
    """

    def __init__(self, directory: str | Path, *, purge_step: int | None) -> None:
        self.directory = Path(directory)
        try:
            self.writer = SummaryWriter(
                log_dir=str(self.directory),
                purge_step=purge_step,
            )
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"could not initialize TensorBoard logging in {self.directory}: {error}"
            ) from error
        self._closed = False

    def log_run(self, config: Animal2VecConfig, model: nn.Module, *, update: int) -> None:
        """Record the resolved recipe and model size at the run boundary."""

        serialized = json.dumps(config_to_dict(config), indent=2, sort_keys=True)
        self.writer.add_text("run/config", f"```json\n{serialized}\n```", update)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.writer.add_scalar("run/parameters", parameter_count, update)
        self.writer.add_scalar("run/trainable_parameters", trainable_count, update)
        self.writer.add_text("run/stage", config.stage, update)
        self.writer.flush()

    def log_update(
        self,
        result: UpdateResult,
        *,
        stage: str,
        amp_scale: float | None,
    ) -> None:
        """Record one globally reduced logical optimizer-update result."""

        step = result.update
        scalars = {
            "train/loss": result.loss,
            "train/sample_size": result.sample_size,
            "train/gradient_norm": result.gradient_norm,
            "train/learning_rate": result.learning_rate,
            "train/skipped": float(result.skipped),
        }
        if amp_scale is not None:
            scalars["train/amp_scale"] = amp_scale
        if stage == "pretrain":
            if result.pred_var is not None:
                scalars["pretrain/pred_var"] = result.pred_var
            if result.target_var is not None:
                scalars["pretrain/target_var"] = result.target_var
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def log_validation(
        self,
        metrics: Mapping[str, float],
        *,
        update: int,
        subset: str,
        labels: Sequence[str] = (),
        frame_scores: Tensor | None = None,
        frame_targets: Tensor | None = None,
        segmented_evaluations: Sequence[SegmentedEvaluation] = (),
        metric_threshold: float = 0.5,
    ) -> None:
        """Record validation scalars, PR curves, and event diagnostics."""

        base = f"validation/{_tensorboard_label(subset)}"
        for name, value in metrics.items():
            if name == "loss":
                tag = f"{base}/loss"
            elif name.startswith("segmented_"):
                tag = f"{base}/segmented/{name.removeprefix('segmented_')}"
            else:
                tag = f"{base}/frame/{name}"
            self.writer.add_scalar(tag, value, update)

        if frame_scores is not None and frame_targets is not None:
            scores = frame_scores.detach().float().cpu()
            targets = frame_targets.detach().long().cpu()
            if scores.shape != targets.shape or scores.ndim != 2:
                raise ValueError(
                    "TensorBoard frame scores and targets must share [frames, labels] shape"
                )
            if scores.shape[1] != len(labels):
                raise ValueError("TensorBoard labels must match frame score columns")
            self.writer.add_pr_curve(
                f"{base}/frame/pr_micro",
                targets.reshape(-1),
                scores.reshape(-1),
                global_step=update,
            )
            frame_predictions = scores >= metric_threshold
            for index, label in enumerate(labels):
                safe_label = _tensorboard_label(label)
                self.writer.add_pr_curve(
                    f"{base}/frame/pr/{safe_label}",
                    targets[:, index],
                    scores[:, index],
                    global_step=update,
                )
                self.writer.add_scalar(
                    f"{base}/frame/average_precision/{safe_label}",
                    average_precision(scores[:, index], targets[:, index]),
                    update,
                )
                counts = FrameCounts.from_predictions(
                    frame_predictions[:, index],
                    targets[:, index].bool(),
                )
                self.writer.add_scalar(
                    f"{base}/frame/precision/{safe_label}",
                    counts.precision,
                    update,
                )
                self.writer.add_scalar(
                    f"{base}/frame/recall/{safe_label}",
                    counts.recall,
                    update,
                )
                self.writer.add_scalar(
                    f"{base}/frame/f1/{safe_label}",
                    counts.f1,
                    update,
                )

        if segmented_evaluations:
            segmented = aggregate_segmented_metrics(
                segmented_evaluations,
                labels,
                metric_threshold=metric_threshold,
            )
            segment_scores = torch.cat([
                evaluation.segmented_scores.reshape(
                    -1, evaluation.segmented_scores.shape[-1]
                )
                for evaluation in segmented_evaluations
            ]).float().cpu()
            segment_targets = torch.cat([
                evaluation.segmented_targets.reshape(
                    -1, evaluation.segmented_targets.shape[-1]
                )
                for evaluation in segmented_evaluations
            ]).long().cpu()
            self.writer.add_pr_curve(
                f"{base}/segmented/pr_micro",
                segment_targets.reshape(-1),
                segment_scores.reshape(-1),
                global_step=update,
            )
            for index, label in enumerate(labels):
                safe_label = _tensorboard_label(label)
                self.writer.add_pr_curve(
                    f"{base}/segmented/pr/{safe_label}",
                    segment_targets[:, index],
                    segment_scores[:, index],
                    global_step=update,
                )
                self.writer.add_scalar(
                    f"{base}/segmented/average_precision/{safe_label}",
                    segmented.classwise_average_precision[label],
                    update,
                )
                self.writer.add_scalar(
                    f"{base}/segmented/precision/{safe_label}",
                    segmented.classwise_precision[label],
                    update,
                )
                self.writer.add_scalar(
                    f"{base}/segmented/recall/{safe_label}",
                    segmented.classwise_recall[label],
                    update,
                )
                self.writer.add_scalar(
                    f"{base}/segmented/f1/{safe_label}",
                    segmented.classwise_f1[label],
                    update,
                )
                distributions = {
                    "iou": torch.cat([
                        evaluation.ious[..., index].reshape(-1)
                        for evaluation in segmented_evaluations
                    ]),
                    "splits": torch.cat([
                        evaluation.splits[..., index].reshape(-1)
                        for evaluation in segmented_evaluations
                    ]),
                    "mergers": torch.cat([
                        evaluation.mergers[..., index].reshape(-1)
                        for evaluation in segmented_evaluations
                    ]),
                }
                for name, values in distributions.items():
                    nonzero = values[values != 0].detach().float().cpu()
                    if nonzero.numel():
                        self.writer.add_histogram(
                            f"{base}/segmented/{name}/{safe_label}",
                            nonzero,
                            global_step=update,
                        )
        self.writer.flush()

    def close(self) -> None:
        """Flush and close the event writer once."""

        if self._closed:
            return
        self.writer.flush()
        self.writer.close()
        self._closed = True

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # SummaryWriter owns a background event thread. This fallback releases
        # it if an exception leaves the training loop before its normal close.
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# FRAME PROBABILITY POOLING AND EVENT FUSION
# =============================================================================

@dataclass(frozen=True)
class EventInterval:
    """One thresholded contiguous event in frame and second coordinates."""

    label_index: int
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    score: float


def pool_probabilities(
    probabilities: Tensor,
    *,
    method: Literal["avg", "max"] | str,
    window_frames: int,
) -> Tensor:
    """Smooth frame probabilities with a centered average or maximum window."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be [frames, labels]")
    if method not in {"avg", "max"}:
        raise ValueError(f"unsupported event fusion method {method!r}; canny requires the optional legacy analysis stack")
    window_frames = max(1, int(window_frames))
    if window_frames == 1:
        return probabilities
    # Mathematics: a centered window of width w needs floor((w-1)/2) samples
    # on the left and floor(w/2) on the right to retain T outputs.
    # Interpretation: odd and even smoothing widths remain aligned to the
    # original probability frame grid.
    left = (window_frames - 1) // 2
    right = window_frames // 2
    values = probabilities.transpose(0, 1).unsqueeze(0)
    if method == "avg":
        valid = torch.ones_like(values)
        # Mathematics: at boundaries, pooled(t)=Σ_{j∈valid window}p_j /
        # |valid window|. Separate numerator and validity denominator exclude
        # zero padding from the average.
        # Interpretation: beginning and end events do not receive an artificial
        # probability reduction because half their window lies outside audio.
        numerator = F.avg_pool1d(F.pad(values, (left, right)), window_frames, stride=1) * window_frames
        denominator = F.avg_pool1d(F.pad(valid, (left, right)), window_frames, stride=1) * window_frames
        pooled = numerator / denominator.clamp_min(1)
    else:
        # Mathematics: -∞ is the identity for max, so out-of-range padded values
        # cannot win a window maximum.
        # Interpretation: boundary handling preserves the strongest real frame
        # without inventing a probability outside the recording.
        pooled = F.max_pool1d(F.pad(values, (left, right), value=float("-inf")), window_frames, stride=1)
    return pooled.squeeze(0).transpose(0, 1)


def fuse_probabilities(
    probabilities: Tensor,
    timestamps: Tensor,
    *,
    threshold: float,
    method: Literal["avg", "max"] | str,
    window_frames: int,
    recording_duration: float,
) -> tuple[EventInterval, ...]:
    """Threshold smoothed probabilities and join adjacent active frames."""

    if probabilities.shape[0] != timestamps.numel():
        raise ValueError("one timestamp is required per probability frame")
    pooled = pool_probabilities(probabilities, method=method, window_frames=window_frames)
    # Mathematics: a_{tc}=1[p_{tc}^{pooled} >= τ].
    # Interpretation: the continuous classifier score becomes a binary event
    # occupancy sequence at the requested operating threshold.
    active = pooled >= threshold
    if timestamps.numel() > 1:
        # Mathematics: Δt is the median adjacent center difference, robust to
        # small float rounding differences across concatenated segments.
        # Interpretation: the final active frame extends by one normal frame hop
        # to form a half-open event end time.
        hop = float(torch.median(timestamps[1:] - timestamps[:-1]))
    else:
        hop = recording_duration
    events: list[EventInterval] = []
    for label_index in range(probabilities.shape[1]):
        label_active = active[:, label_index]
        frame = 0
        while frame < len(label_active):
            if not bool(label_active[frame]):
                frame += 1
                continue
            start = frame
            while frame < len(label_active) and bool(label_active[frame]):
                frame += 1
            end = frame
            # Mathematics: a maximal contiguous true run [s,e) maps to seconds
            # [max(0,t_s), min(duration,t_{e-1}+Δt)) and score mean_{s:e} p_t.
            # Interpretation: adjacent active frames form one event whose score
            # reflects the complete run rather than its peak alone.
            events.append(EventInterval(
                label_index=label_index,
                start_frame=start,
                end_frame=end,
                start_seconds=max(0.0, float(timestamps[start])),
                end_seconds=min(recording_duration, float(timestamps[end - 1]) + hop),
                score=float(pooled[start:end, label_index].mean()),
            ))
    return tuple(events)


def event_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return temporal intersection-over-union for two half-open intervals."""

    # Mathematics: IoU(A,B)=|A∩B|/|A∪B| for half-open time intervals.
    # Interpretation: overlap rewards temporal agreement while penalizing both
    # missed duration and excessive predicted duration.
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


# =============================================================================
# LEGACY-COMPATIBLE SEGMENTED EVENT EVALUATION
# =============================================================================

@dataclass(frozen=True)
class SegmentedEvaluation:
    """Per-recording tensors emitted by the archived-compatible event matcher."""

    segmented_scores: Tensor
    segmented_targets: Tensor
    ious: Tensor
    splits: Tensor
    mergers: Tensor


@dataclass(frozen=True)
class SegmentedMetrics:
    """Dataset-level event metrics after archived segment matching."""

    classwise_average_precision: dict[str, float]
    classwise_precision: dict[str, float]
    classwise_recall: dict[str, float]
    classwise_f1: dict[str, float]
    macro_average_precision: float
    micro_average_precision: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    focal_threshold: float | None
    focal_f1: float | None
    focal_precision: float | None
    focal_recall: float | None


def _inclusive_intervals(active: Tensor) -> list[tuple[int, int]]:
    """Return contiguous true runs as inclusive ``(start, end)`` frame pairs."""

    intervals: list[tuple[int, int]] = []
    frame = 0
    while frame < active.numel():
        if not bool(active[frame]):
            frame += 1
            continue
        start = frame
        while frame < active.numel() and bool(active[frame]):
            frame += 1
        intervals.append((start, frame - 1))
    return intervals


def _legacy_bounds(interval: tuple[int, int]) -> tuple[int, int]:
    """Give a one-frame interval nonzero width under archived arithmetic."""

    start, end = interval
    return (start, end + 1) if start == end else interval


def _overlap(first: tuple[int, int], second: tuple[int, int]) -> int:
    """Return archived overlap width for two frame intervals."""

    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def _iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    """Compute the archived intersection-over-union, including its exact-match case."""

    # Mathematics: archived intervals use end-start width except for the
    # exact-match branch, so union width is combined-overlap.
    # Interpretation: this preserves the historical evaluation result even
    # though its endpoint convention differs from modern half-open intervals.
    overlap = _overlap(first, second)
    combined = first[1] - first[0] + second[1] - second[0]
    return 1.0 if combined == overlap else overlap / (combined - overlap)


def _predicted_intervals(
    probabilities: Tensor,
    *,
    method: Literal["avg", "max"],
    window_frames: int,
    threshold: float,
) -> list[tuple[int, int]]:
    """Pool and threshold one class series into shifted predicted intervals."""

    frames = probabilities.numel()
    if window_frames > frames:
        return []
    values = probabilities.view(1, 1, -1)
    if method == "avg":
        pooled = F.avg_pool1d(values, window_frames, stride=1)
    elif method == "max":
        pooled = F.max_pool1d(values, window_frames, stride=1)
    else:
        raise ValueError(f"unsupported event fusion method {method!r}")
    # Mathematics: valid pooling produces T-w+1 values; right padding restores
    # T indices and a round(w/2) shift maps each window score near its center.
    # Interpretation: this intentionally retains the official scorer's
    # asymmetric boundary and shift behavior for result comparability.
    padded = F.pad(pooled.view(-1), (0, frames - pooled.numel()))
    shift = round(window_frames / 2)
    return [
        (start + shift, min(frames - 1, end + shift))
        for start, end in _inclusive_intervals(padded >= threshold)
    ]


def legacy_segmented_evaluation(
    probabilities: Tensor,
    targets: Tensor,
    *,
    method: Literal["avg", "max"],
    window_frames: int,
    metric_threshold: float,
    iou_threshold: float,
) -> SegmentedEvaluation:
    """Reproduce the archived average/maximum event matcher on frame tensors."""
    if probabilities.ndim != 3 or targets.shape != probabilities.shape:
        raise ValueError("probabilities and targets must share [batch, frames, labels] shape")
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    batch_size, frames, classes = probabilities.shape
    # Mathematics: output arrays keep fixed upper-bound shapes, and index
    # cursors populate only detected target/prediction segments.
    # Interpretation: the layout matches archived criterion logs and supports
    # concatenation across recordings without variable-length containers.
    scores = probabilities.new_zeros(probabilities.shape)
    segmented_targets = torch.zeros_like(targets, dtype=torch.long)
    ious = probabilities.new_zeros((batch_size, round(frames / 2), classes))
    split_shape = (batch_size, int((frames // 3) * 2), classes)
    splits = torch.zeros(split_shape, dtype=torch.long, device=probabilities.device)
    mergers = torch.zeros_like(splits)

    for batch in range(batch_size):
        for label in range(classes):
            # Mathematics: thresholded binary target runs and pooled prediction
            # runs become sets of inclusive integer intervals.
            # Interpretation: set construction deduplicates identical intervals
            # before the legacy split/merger matching rules execute.
            truth = {
                _legacy_bounds(interval)
                for interval in _inclusive_intervals(targets[batch, :, label].bool())
            }
            predictions = {
                _legacy_bounds(interval)
                for interval in _predicted_intervals(
                    probabilities[batch, :, label],
                    method=method,
                    window_frames=window_frames,
                    threshold=metric_threshold,
                )
            }
            sample_index = iou_index = split_index = merger_index = -1
            for target_interval in truth:
                overlapping = set(
                    prediction
                    for prediction in sorted(predictions)
                    if _overlap(target_interval, prediction) > 0
                )
                if not overlapping:
                    sample_index += 1
                    segmented_targets[batch, sample_index, label] = 1
                    scores[batch, sample_index, label] = probabilities[
                        batch, target_interval[0]:target_interval[1], label
                    ].mean()
                    continue
                valid_overlaps = 0
                for prediction in overlapping:
                    sample_index += 1
                    iou_index += 1
                    # Mathematics: a pair is a valid match iff its archived IoU
                    # exceeds, rather than equals, iou_threshold.
                    # Interpretation: preserving strict comparison avoids a
                    # metric change for zero-overlap or boundary-valued cases.
                    overlap_iou = _iou(target_interval, prediction)
                    ious[batch, iou_index, label] = overlap_iou
                    if overlap_iou > iou_threshold:
                        valid_overlaps += 1
                        segmented_targets[batch, sample_index, label] = 1
                        interval = prediction
                    else:
                        interval = target_interval
                    scores[batch, sample_index, label] = probabilities[
                        batch, interval[0]:interval[1], label
                    ].mean()
                if valid_overlaps > 1:
                    split_index += 1
                    splits[batch, split_index, label] = valid_overlaps

            for prediction in predictions:
                overlapping = set(
                    target_interval
                    for target_interval in sorted(truth)
                    if _overlap(prediction, target_interval) > 0
                )
                valid_overlaps = sum(
                    _iou(prediction, target_interval) > iou_threshold
                    for target_interval in overlapping
                )
                if valid_overlaps > 1:
                    merger_index += 1
                    mergers[batch, merger_index, label] = valid_overlaps
                if not overlapping:
                    sample_index += 1
                    scores[batch, sample_index, label] = probabilities[
                        batch, prediction[0]:prediction[1], label
                    ].mean()

    return SegmentedEvaluation(scores, segmented_targets, ious, splits, mergers)


def _best_f1(scores: Tensor, targets: Tensor) -> tuple[float, float, float, float]:
    """Find the observed score threshold with the highest binary F1."""

    if scores.numel() == 0:
        return 1.0, 0.0, 0.0, 0.0
    best = (float(scores.max()), -1.0, 0.0, 0.0)
    for candidate in torch.unique(scores).sort(descending=True).values:
        # Mathematics: evaluate every distinct observed score τ using
        # prediction_i=1[score_i>=τ], then maximize harmonic mean F1.
        # Interpretation: the selected focal threshold is data-derived and
        # never depends on an arbitrary threshold grid resolution.
        predicted = scores >= candidate
        positive = targets.bool()
        true_positive = int(torch.count_nonzero(predicted & positive))
        false_positive = int(torch.count_nonzero(predicted & ~positive))
        false_negative = int(torch.count_nonzero(~predicted & positive))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-30)
        if f1 > best[1]:
            best = (float(candidate), f1, precision, recall)
    return best


def aggregate_segmented_metrics(
    evaluations: Sequence[SegmentedEvaluation],
    labels: Sequence[str],
    *,
    metric_threshold: float = 0.5,
    focal_label: str = "focal",
) -> SegmentedMetrics:
    """Aggregate the complete archived segment-sample tensors.

    Rows with no positive target remain in the calculation because they can
    represent false-positive predicted segments. The archived logger also
    retained the unused zero padding in each fixed-size tensor, so this
    function keeps it for numerical compatibility.
    """

    if not evaluations:
        raise ValueError("at least one segmented evaluation is required")
    scores = torch.cat([
        evaluation.segmented_scores.reshape(-1, evaluation.segmented_scores.shape[-1])
        for evaluation in evaluations
    ])
    targets = torch.cat([
        evaluation.segmented_targets.reshape(-1, evaluation.segmented_targets.shape[-1])
        for evaluation in evaluations
    ])
    if scores.shape != targets.shape or scores.shape[1] != len(labels):
        raise ValueError("labels must match the segmented score and target class dimension")
    classwise = {
        label: average_precision(scores[:, index], targets[:, index])
        for index, label in enumerate(labels)
    }
    # Mathematics: the archived ``average_precision_score(ta, pr)`` uses a
    # macro average over all C columns, while flattening gives micro AP.
    # Interpretation: the first value weights labels equally, including the
    # focal channel; the second weights every segment-class decision equally.
    macro = sum(classwise.values()) / len(classwise) if classwise else 0.0
    micro = average_precision(scores.reshape(-1), targets.reshape(-1))

    predicted = scores >= metric_threshold
    positive = targets.bool()
    total_counts = FrameCounts.from_predictions(predicted, positive)
    class_counts = [
        FrameCounts.from_predictions(predicted[:, index], positive[:, index])
        for index in range(scores.shape[1])
    ]
    classwise_precision = {
        label: class_counts[index].precision
        for index, label in enumerate(labels)
    }
    classwise_recall = {
        label: class_counts[index].recall
        for index, label in enumerate(labels)
    }
    classwise_f1 = {
        label: class_counts[index].f1
        for index, label in enumerate(labels)
    }
    if focal_label in labels:
        focal_index = labels.index(focal_label)
        threshold, f1, precision, recall = _best_f1(
            scores[:, focal_index], targets[:, focal_index]
        )
    else:
        threshold = f1 = precision = recall = None
    return SegmentedMetrics(
        classwise_average_precision=classwise,
        classwise_precision=classwise_precision,
        classwise_recall=classwise_recall,
        classwise_f1=classwise_f1,
        macro_average_precision=macro,
        micro_average_precision=micro,
        precision=total_counts.precision,
        recall=total_counts.recall,
        f1=total_counts.f1,
        accuracy=total_counts.accuracy,
        focal_threshold=threshold,
        focal_f1=f1,
        focal_precision=precision,
        focal_recall=recall,
    )


# =============================================================================
# CHECKPOINT-BACKED WINDOWED INFERENCE
# =============================================================================

@dataclass(frozen=True)
class InferenceResult:
    """Concatenated frame outputs and fused events for one recording."""

    probabilities: Tensor
    embeddings: Tensor
    timestamps: Tensor
    events: tuple[EventInterval, ...]


def _trim_frames_to_duration(
    probabilities: Tensor,
    embeddings: Tensor,
    timestamps: Tensor,
    *,
    end_seconds: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Remove padded-frontend frames whose centers lie beyond real audio."""

    frame_count = timestamps.numel()
    if probabilities.shape[0] != frame_count or embeddings.shape[0] != frame_count:
        raise ValueError("probabilities, embeddings, and timestamps must align")
    keep = timestamps < end_seconds
    return probabilities[keep], embeddings[keep], timestamps[keep]


class InferenceRunner:
    """Load a native fine-tuning checkpoint and process long recordings.

    Recordings are resampled and split into bounded segments. Frame timestamps
    retain each segment's absolute offset, and padded frames from the final
    partial segment are removed before event fusion.
    """

    def __init__(self, checkpoint_path: str | Path, *, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
        if checkpoint["stage"] != "finetune":
            raise CheckpointError("inference requires a fine-tuning checkpoint")
        stored_config = checkpoint["config"]
        if not isinstance(stored_config, dict) or "active" not in stored_config or "pretrained" not in stored_config:
            raise CheckpointError("fine-tuning checkpoint lacks active/pretrained native configs")
        self.config = config_from_serialized_dict(stored_config["active"])
        self.pretrained_config = config_from_serialized_dict(stored_config["pretrained"])
        self.model = Animal2VecFineTuningModel.from_config(
            self.config, pretrained_config=self.pretrained_config
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.update = int(checkpoint["update"])

    @torch.inference_mode()
    def run_tensor(
        self,
        waveform: Tensor,
        sample_rate: int,
        *,
        channel: int | None = None,
        segment_seconds: float = 10.0,
        threshold: float | None = None,
        event_method: str | None = None,
        fusion_window_seconds: float | None = None,
    ) -> InferenceResult:
        """Run inference on mono or channel-first audio in memory."""

        if waveform.ndim == 2:
            if channel is None:
                waveform = waveform.mean(dim=0)
            else:
                if not 0 <= channel < waveform.shape[0]:
                    raise ValueError(f"channel {channel} is outside recording with {waveform.shape[0]} channels")
                waveform = waveform[channel]
        if waveform.ndim != 1:
            raise ValueError("waveform must be [samples] or [channels, samples]")
        # Mathematics: duration=S/f_s remains the reference endpoint even when
        # the waveform is later resampled to the model rate.
        # Interpretation: output events stay in the original recording's time
        # coordinates and cannot extend into final-segment padding.
        recording_duration = waveform.shape[-1] / sample_rate
        target_rate = self.config.task.sample_rate
        waveform = resample_waveform(waveform.float(), sample_rate, target_rate)
        # Mathematics: each model segment contains round(τ f_target) samples.
        # Interpretation: long recordings use bounded inference memory while
        # preserving the requested segment duration.
        segment_samples = round(segment_seconds * target_rate)
        if segment_samples <= 0:
            raise ValueError("segment_seconds must produce at least one sample")

        probability_parts: list[Tensor] = []
        embedding_parts: list[Tensor] = []
        timestamp_parts: list[Tensor] = []
        for start in range(0, waveform.shape[-1], segment_samples):
            segment = normalize_waveform(waveform[start: start + segment_samples]).unsqueeze(0).to(self.device)
            output = self.model(segment, update=self.update)
            # Mathematics: independent class probability p_{tc}=σ(logit_{tc});
            # embedding z_t is the arithmetic mean of the final K layer outputs.
            # Interpretation: event decisions and reusable representations come
            # from the same forward pass and frame grid.
            probabilities = torch.sigmoid(output.logits[0]).cpu()
            layer_count = min(self.config.model.average_top_k_layers, len(output.layer_outputs))
            embeddings = torch.stack(output.layer_outputs[-layer_count:]).mean(dim=0)[0].cpu()
            # Mathematics: local frame centers receive absolute offset
            # start/f_target before segments are concatenated.
            # Interpretation: splitting for memory does not reset time to zero
            # at each segment boundary.
            timestamps = feature_timestamps(
                probabilities.shape[0], target_rate, self.config.task.conv_feature_layers
            ) + start / target_rate
            segment_end = min(
                recording_duration,
                (start + segment.shape[-1]) / target_rate,
            )
            probabilities, embeddings, timestamps = _trim_frames_to_duration(
                probabilities,
                embeddings,
                timestamps,
                end_seconds=segment_end,
            )
            if timestamps.numel() == 0:
                continue
            probability_parts.append(probabilities)
            embedding_parts.append(embeddings)
            timestamp_parts.append(timestamps)

        if not probability_parts:
            label_count = len(self.config.task.unique_labels)
            empty_probabilities = torch.empty(0, label_count)
            empty_embeddings = torch.empty(0, self.pretrained_config.model.embed_dim)
            return InferenceResult(empty_probabilities, empty_embeddings, torch.empty(0), ())
        probabilities = torch.cat(probability_parts)
        embeddings = torch.cat(embedding_parts)
        timestamps = torch.cat(timestamp_parts)
        threshold = self.config.criterion.metric_threshold if threshold is None else threshold
        event_method = self.config.criterion.event_method if event_method is None else event_method
        window_seconds = self.config.criterion.sigma_s if fusion_window_seconds is None else fusion_window_seconds
        stride = 1
        for _, _, layer_stride in self.config.task.conv_feature_layers:
            stride *= layer_stride
        # Mathematics: total convolution stride q=Π_l s_l gives nominal frame
        # rate f_frame=f_target/q and smoothing width round(τ_window f_frame).
        # Interpretation: a fusion window specified in seconds retains the same
        # physical duration across frontend stride choices.
        frame_rate = target_rate / stride
        events = fuse_probabilities(
            probabilities,
            timestamps,
            threshold=threshold,
            method=event_method,
            window_frames=max(1, round(window_seconds * frame_rate)),
            recording_duration=recording_duration,
        )
        return InferenceResult(probabilities, embeddings, timestamps, events)

    def write_events(
        self,
        path: str | Path,
        events: tuple[EventInterval, ...],
        *,
        audition: bool = False,
    ) -> None:
        """Write fused events in the native TSV or Adobe Audition marker schema.

        The default format exposes onset, offset, and the full-precision event
        score for programmatic analysis. ``audition=True`` reproduces the six
        columns used by the official Animal2Vec 1.0 inference script so Adobe
        Audition can import each event as a range marker. The format switch
        changes serialization only; both branches receive the same event set.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if audition:
            lines = ["Name\tStart\tDuration\tTime Format\tType\tDescription"]
            # Mathematics: order events by the lexicographic numeric key
            # (t_start, t_end, class_index), which is a total order for the
            # finite event coordinates produced by inference.
            # Interpretation: Audition displays markers in recording order,
            # even though event fusion groups its internal output by class.
            ordered_events = sorted(
                events,
                key=lambda event: (
                    event.start_seconds,
                    event.end_seconds,
                    event.label_index,
                ),
            )
            for event in ordered_events:
                label = self.config.task.unique_labels[event.label_index]
                # Mathematics: a range marker stores onset t_s and duration
                # Δt=t_e-t_s. ``timedelta`` maps each real-valued second count
                # to the legacy [D day[s], ]H:MM:SS[.ffffff] representation.
                # Interpretation: Adobe reconstructs the event offset by adding
                # Duration to Start; it does not consume an explicit End field.
                start = str(timedelta(seconds=event.start_seconds))
                duration = str(timedelta(
                    seconds=event.end_seconds - event.start_seconds
                ))
                # Mathematics: Python's fixed-point format rounds score p to
                # three digits after the decimal point, matching ``{:1.03f}``
                # in the Animal2Vec 1.0 writer.
                # Interpretation: researchers can read the model confidence in
                # the marker description while reviewing the waveform.
                description = f"{event.score:.3f}"
                lines.append("\t".join((
                    label,
                    start,
                    duration,
                    "decimal",
                    "Cue",
                    description,
                )))
            path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            return

        lines = ["label\tstart_seconds\tend_seconds\tscore"]
        for event in events:
            label = self.config.task.unique_labels[event.label_index]
            lines.append(
                f"{label}\t{event.start_seconds:.6f}\t{event.end_seconds:.6f}\t{event.score:.8f}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# ARCHIVED FAIRSEQ CHECKPOINT CONVERSION
# =============================================================================

# Public conversion result and legacy checkpoint access

class CheckpointConversionError(ValueError):
    """Raised when an official Fairseq checkpoint cannot map losslessly."""


@dataclass(frozen=True)
class ConversionReport:
    """Auditable summary of renamed, omitted, and missing state tensors."""

    stage: str
    mapped: int
    renamed: tuple[tuple[str, str], ...]
    omitted: tuple[str, ...]
    missing: tuple[str, ...]


def legacy_num_updates(checkpoint: Mapping[str, object]) -> int:
    """Read the update counter from either native-like or Fairseq layout."""

    direct = checkpoint.get("num_updates")
    if direct is not None:
        return int(direct)
    history = checkpoint.get("optimizer_history")
    if isinstance(history, (list, tuple)) and history:
        latest = history[-1]
        if isinstance(latest, Mapping) and latest.get("num_updates") is not None:
            return int(latest["num_updates"])
    return 0


def load_legacy_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load an official checkpoint without importing Fairseq."""

    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise CheckpointConversionError(
            "the checkpoint contains Python config objects unavailable in this environment; "
            "export it in the archived Fairseq environment with OmegaConf.to_container(cfg, resolve=True), "
            "then save a plain {'model': state['model'], 'cfg': plain_cfg} dictionary"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointConversionError(f"cannot load legacy checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise CheckpointConversionError("legacy checkpoint root must be a dictionary")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise CheckpointConversionError("legacy checkpoint model state must be a mapping")
    return checkpoint


# State-key translation

def map_legacy_encoder_key(key: str) -> str | None:
    """Translate one official encoder state key to the native flat model."""

    prefix = "modality_encoders.AUDIO."
    if key == prefix + "alibi_scale":
        return "alibi_scale"
    # Mathematics: the regex captures layer index i, archived component code k,
    # and remaining parameter suffix s, then maps (i,k,s) bijectively when the
    # native architecture contains an equivalent tensor.
    # Interpretation: conversion renames checkpoint coordinates without
    # guessing from tensor order or shape alone.
    match = re.fullmatch(
        re.escape(prefix) + r"local_encoder\.conv_layers\.(\d+)\.(0|2\.1|3)\.(.+)",
        key,
    )
    if match:
        index, component, suffix = match.groups()
        native_component = {"0": "conv", "2.1": "norm", "3": "activation"}[component]
        return f"local_encoder.conv_layers.{index}.{native_component}.{suffix}"
    if key.startswith(prefix + "project_features.1."):
        return "project_norm." + key.removeprefix(prefix + "project_features.1.")
    if key.startswith(prefix + "project_features.2."):
        return "project_features." + key.removeprefix(prefix + "project_features.2.")
    match = re.fullmatch(
        re.escape(prefix) + r"relative_positional_encoder\.(\d+)\.(0|3)\.(.+)",
        key,
    )
    if match:
        legacy_index, component, suffix = match.groups()
        block_index = int(legacy_index) - 1
        if block_index < 0:
            return None
        native_component = "conv" if component == "0" else "norm"
        return f"positional_encoder.blocks.{block_index}.{native_component}.{suffix}"
    if key.startswith(prefix + "context_encoder.blocks."):
        return "prenet.blocks." + key.removeprefix(prefix + "context_encoder.blocks.")
    if key.startswith(prefix + "context_encoder.norm."):
        return "prenet.norm." + key.removeprefix(prefix + "context_encoder.norm.")
    if key.startswith("blocks."):
        return "transformer.blocks." + key.removeprefix("blocks.")
    if key.startswith("norm."):
        return "transformer.norm." + key.removeprefix("norm.")
    return None


def _map_decoder_key(key: str) -> str | None:
    """Translate one official reconstruction-decoder parameter name."""

    prefix = "modality_encoders.AUDIO.decoder."
    match = re.fullmatch(re.escape(prefix) + r"blocks\.(\d+)\.0\.(.+)", key)
    if match:
        index, suffix = match.groups()
        return f"decoder.blocks.{index}.conv.{suffix}"
    if key.startswith(prefix + "proj."):
        return "decoder.proj." + key.removeprefix(prefix + "proj.")
    return None


def _plain(value: object) -> object:
    """Convert legacy config objects into primitive mappings and sequences."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}  # type: ignore[union-attr]
    raise CheckpointConversionError(f"cannot convert config value of type {type(value).__name__} to a plain container")


# Configuration extraction and tensor assignment

def _config_from_legacy_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    stage: str,
) -> tuple[Animal2VecConfig, Animal2VecConfig | None]:
    """Recover active and optional pretrained configs embedded in a checkpoint."""

    if "cfg" not in checkpoint:
        raise CheckpointConversionError("legacy checkpoint has no cfg; pass --config")
    plain = _plain(checkpoint["cfg"])
    if not isinstance(plain, dict):
        raise CheckpointConversionError("legacy cfg is not a mapping; pass --config")
    model_section = plain.get("model", {})
    pretrained = None
    if stage == "finetune" and isinstance(model_section, dict) and "w2v_args" in model_section:
        nested = model_section.pop("w2v_args")
        if isinstance(nested, dict):
            pretrained = config_from_dict(nested)
    try:
        active = config_from_dict(plain)
    except ValueError as exc:
        raise CheckpointConversionError(
            f"the resolved legacy cfg contains unsupported defaults ({exc}); pass the matching repository YAML with --config"
        ) from exc
    return active, pretrained


def _assign(
    converted: dict[str, Tensor],
    target: Mapping[str, Tensor],
    source_name: str,
    target_name: str,
    tensor: object,
    renamed: list[tuple[str, str]],
) -> None:
    """Validate and copy one mapped tensor into the converted state."""

    if not isinstance(tensor, Tensor):
        return
    if target_name not in target:
        return
    # Mathematics: assignment requires shape(source)=shape(target); values are
    # cast only to the target dtype and never reshaped, transposed, or sliced.
    # Interpretation: a superficially similar but incompatible architecture
    # fails conversion instead of producing a silent degraded baseline.
    if target[target_name].shape != tensor.shape:
        raise CheckpointConversionError(
            f"shape mismatch for {target_name} mapped from {source_name}: "
            f"expected {tuple(target[target_name].shape)}, received {tuple(tensor.shape)}"
        )
    converted[target_name] = tensor.detach().to(dtype=target[target_name].dtype)
    renamed.append((source_name, target_name))


def _convert_pretraining_state(
    legacy: Mapping[str, object],
    model: Animal2VecPretrainingModel,
) -> tuple[dict[str, Tensor], list[tuple[str, str]], list[str]]:
    """Map student, decoder, and EMA teacher tensors for pretraining."""

    target = model.state_dict()
    converted: dict[str, Tensor] = {}
    renamed: list[tuple[str, str]] = []
    omitted: list[str] = []
    for source_name, tensor in legacy.items():
        if source_name == "_ema":
            continue
        encoder_name = map_legacy_encoder_key(source_name)
        decoder_name = _map_decoder_key(source_name)
        target_name = f"student.{encoder_name}" if encoder_name is not None else decoder_name
        if target_name is None or target_name not in target:
            omitted.append(source_name)
            continue
        _assign(converted, target, source_name, target_name, tensor, renamed)

    # Mathematics: before an explicit archived EMA payload is applied, initialize
    # teacher coordinate \bar θ_k from the corresponding converted student θ_k.
    # Interpretation: checkpoints lacking separate EMA values still construct a
    # complete model, while later EMA entries override this fallback.
    for target_name in target:
        if not target_name.startswith("teacher.model."):
            continue
        student_name = "student." + target_name.removeprefix("teacher.model.")
        if student_name in converted:
            converted[target_name] = converted[student_name].clone()
            renamed.append((student_name, target_name))

    ema_state = legacy.get("_ema", {})
    if isinstance(ema_state, Mapping) and isinstance(ema_state.get("params"), Mapping):
        ema_state = ema_state["params"]
    if isinstance(ema_state, Mapping):
        for source_name, tensor in ema_state.items():
            encoder_name = map_legacy_encoder_key(str(source_name))
            target_name = f"teacher.model.{encoder_name}" if encoder_name is not None else None
            if target_name is None or target_name not in target:
                omitted.append(f"_ema.{source_name}")
                continue
            _assign(converted, target, f"_ema.{source_name}", target_name, tensor, renamed)
    return converted, renamed, omitted


def _convert_finetuning_state(
    legacy: Mapping[str, object],
    model: Animal2VecFineTuningModel,
) -> tuple[dict[str, Tensor], list[tuple[str, str]], list[str]]:
    """Map the encoder wrapper and classification head for fine-tuning."""

    target = model.state_dict()
    converted: dict[str, Tensor] = {}
    renamed: list[tuple[str, str]] = []
    omitted: list[str] = []
    wrapper_prefixes = (
        "w2v_encoder.w2v_model.",
        "encoder.w2v_model.",
        "w2v_model.",
    )
    for source_name, tensor in legacy.items():
        target_name = None
        if source_name in {"w2v_encoder.proj.weight", "encoder.proj.weight", "proj.weight"}:
            target_name = "classifier.weight"
        elif source_name in {"w2v_encoder.proj.bias", "encoder.proj.bias", "proj.bias"}:
            target_name = "classifier.bias"
        else:
            stripped = source_name
            for prefix in wrapper_prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped.removeprefix(prefix)
                    break
            encoder_name = map_legacy_encoder_key(stripped)
            if encoder_name is not None:
                target_name = "encoder." + encoder_name
        if target_name is None or target_name not in target:
            omitted.append(source_name)
            continue
        _assign(converted, target, source_name, target_name, tensor, renamed)
    return converted, renamed, omitted


# Public conversion operation

def convert_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    config_path: str | Path | None = None,
    pretrained_config_path: str | Path | None = None,
    stage: str | None = None,
) -> ConversionReport:
    """Convert an official pretraining or fine-tuning checkpoint.

    Conversion is strict about every tensor expected by the native model. The
    output retains the original update count but intentionally starts with no
    optimizer state because Fairseq optimizer internals are not portable.
    """

    checkpoint = load_legacy_checkpoint(source)
    legacy_state = checkpoint["model"]
    assert isinstance(legacy_state, Mapping)
    if stage is None:
        stage = "finetune" if any("w2v_encoder" in str(key) for key in legacy_state) else "pretrain"
    if stage not in {"pretrain", "finetune"}:
        raise CheckpointConversionError("stage must be pretrain or finetune")

    if config_path is not None:
        active_config = load_config(config_path)
        pretrained_config = load_config(pretrained_config_path) if pretrained_config_path is not None else None
    else:
        active_config, pretrained_config = _config_from_legacy_checkpoint(checkpoint, stage=stage)
    if active_config.stage != stage:
        raise CheckpointConversionError(
            f"config stage {active_config.stage} does not match requested conversion stage {stage}"
        )

    if stage == "pretrain":
        model: Animal2VecPretrainingModel | Animal2VecFineTuningModel = Animal2VecPretrainingModel.from_config(active_config)
        converted, renamed, omitted = _convert_pretraining_state(legacy_state, model)
    else:
        if pretrained_config is None:
            raise CheckpointConversionError("fine-tuning conversion requires --pretrained-config or checkpoint model.w2v_args")
        model = Animal2VecFineTuningModel.from_config(active_config, pretrained_config=pretrained_config)
        converted, renamed, omitted = _convert_finetuning_state(legacy_state, model)

    target = model.state_dict()
    # Mathematics: strict conversion requires keys(target) \ keys(converted)=∅.
    # Interpretation: every parameter and persistent buffer needed by native
    # inference must have an auditable origin in the archived checkpoint.
    missing = sorted(set(target) - set(converted))
    if missing:
        raise CheckpointConversionError(f"conversion is missing core tensors: {missing}")
    model.load_state_dict(converted, strict=True)
    serialized_config: dict[str, object] = {"active": config_to_dict(active_config)}
    if pretrained_config is not None:
        serialized_config["pretrained"] = config_to_dict(pretrained_config)
    teacher = model.teacher.model.state_dict() if isinstance(model, Animal2VecPretrainingModel) else None
    payload = {
        "format_version": 1,
        "stage": stage,
        "config": serialized_config,
        "model": model.state_dict(),
        "teacher": teacher,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "update": legacy_num_updates(checkpoint),
        "epoch": 0,
        "batch_in_epoch": 0,
        "rng_state": capture_rng_state(),
        "sampler_state": None,
        "best_metric": None,
    }
    save_checkpoint(destination, payload)
    return ConversionReport(stage, len(converted), tuple(renamed), tuple(sorted(omitted)), tuple(missing))


# =============================================================================
# TRAINING AND VALIDATION ORCHESTRATION
# =============================================================================

# Command line and distributed launch

def build_training_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for both training stages."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument(
        "--stop-at-update",
        type=int,
        help="stop at this absolute update without changing the configured scheduler horizon",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _distributed_device(requested: str) -> tuple[torch.device, int, int, bool]:
    """Resolve rank-local device state and initialize a process group if needed."""

    # Mathematics: torchrun defines global rank r∈[0,W), local rank l, and
    # world size W through environment variables.
    # Interpretation: one command works for a CPU process, one GPU, or an
    # eight-GPU launch without embedding cluster-specific addresses in recipes.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        initialized_here = True
    return device, rank, world_size, initialized_here


def _checkpoint_process_group(
    device: torch.device,
    world_size: int,
) -> dist.ProcessGroup | None:
    """Create a CPU control plane for object checkpoint collectives when needed."""

    if world_size > 1 and device.type == "cuda":
        return dist.new_group(backend="gloo")
    return None


def _ddp_find_unused_parameters(config: Animal2VecConfig) -> bool:
    """Return the fixed traversal setting used by exact-resume verification."""

    # Fine-tuning intentionally leaves the frozen backbone unused. During
    # pretraining, the traversal also keeps reduction scheduling identical
    # before and after a process restart when paired with one explicit bucket.
    return True


def _ddp_bucket_cap_mb_list(config: Animal2VecConfig) -> list[int] | None:
    """Return stable pretraining bucket limits for PyTorch's DDP reducer."""

    # Explicit caps make first-iteration and rebuilt pretraining buckets use
    # the same limits, including after a process restart.
    return [4096] if config.stage == "pretrain" else None


# Model, checkpoint, and dataset construction

def _load_pretrained(
    checkpoint_path: Path,
) -> tuple[Animal2VecConfig, dict[str, Tensor]]:
    """Load a native pretraining checkpoint and extract its student encoder."""

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["stage"] != "pretrain":
        raise CheckpointError("--pretrained-checkpoint must point to a pretraining checkpoint")
    stored = checkpoint["config"]
    if not isinstance(stored, dict) or "active" not in stored:
        raise CheckpointError("pretraining checkpoint lacks a native active config")
    config = config_from_serialized_dict(stored["active"])
    # Mathematics: remove the injective prefix "student." from every student
    # state key and discard decoder, teacher, and regression entries.
    # Interpretation: fine-tuning initializes only the shared waveform encoder
    # from a pretraining checkpoint.
    state = {
        key.removeprefix("student."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("student.")
    }
    if not state:
        raise CheckpointError("pretraining checkpoint contains no student encoder tensors")
    return config, state


def _make_model(
    config: Animal2VecConfig,
    *,
    pretrained_checkpoint: Path | None,
    resume_checkpoint: dict[str, Any] | None,
) -> tuple[nn.Module, Animal2VecConfig | None]:
    """Construct the configured stage and resolve fine-tuning encoder weights."""

    if config.stage == "pretrain":
        return Animal2VecPretrainingModel.from_config(config), None
    pretrained_config = None
    encoder_state = None
    if resume_checkpoint is not None:
        stored = resume_checkpoint["config"]
        if isinstance(stored, dict) and "pretrained" in stored:
            pretrained_config = config_from_serialized_dict(stored["pretrained"])
    if pretrained_config is None:
        path = pretrained_checkpoint or (Path(config.model.w2v_path) if config.model.w2v_path else None)
        if path is None:
            raise CheckpointError("fine-tuning requires --pretrained-checkpoint or model.w2v_path")
        pretrained_config, encoder_state = _load_pretrained(path)
    model = Animal2VecFineTuningModel.from_config(
        config,
        pretrained_config=pretrained_config,
        encoder_state=encoder_state,
    )
    return model, pretrained_config


def _make_dataset(config: Animal2VecConfig, subset: str | None = None) -> AudioDataset:
    """Build one manifest dataset with labels only for fine-tuning."""

    subset = config.dataset.train_subset if subset is None else subset
    manifest = config.task.data / f"{subset}.tsv"
    return AudioDataset(
        manifest,
        sample_rate=config.task.sample_rate,
        conv_layers=config.task.conv_feature_layers,
        normalize=config.task.normalize,
        labels=config.task.unique_labels if config.stage == "finetune" else None,
        min_sample_size=config.task.min_sample_size,
        max_sample_size=config.task.max_sample_size,
        min_label_size=config.task.min_label_size,
    )


# Validation and metric selection

def _validation_due(config: Animal2VecConfig, *, update: int, epoch: int | None = None) -> bool:
    """Decide whether update-based or epoch-based validation is due."""

    dataset = config.dataset
    if dataset.disable_validation or update < dataset.validate_after_updates:
        return False
    if dataset.validate_after_updates > 0 and update == dataset.validate_after_updates:
        return True
    if dataset.validate_interval_updates > 0 and update % dataset.validate_interval_updates == 0:
        return True
    # Mathematics: epoch validation occurs when e mod interval = 0; earlier
    # branches give update-based rules precedence.
    # Interpretation: the workflow supports both official cadence styles
    # without running validation twice at the same update.
    return epoch is not None and dataset.validate_interval > 0 and epoch % dataset.validate_interval == 0


@torch.inference_mode()
def _validate(
    model: nn.Module,
    config: Animal2VecConfig,
    *,
    device: torch.device,
    update: int,
    tensorboard_logger: TensorBoardLogger | None = None,
) -> dict[str, float]:
    """Evaluate one subset with framewise and segmented event measurements."""

    dataset = _make_dataset(config, config.dataset.valid_subset)
    if len(dataset) == 0:
        raise ValueError("validation manifest contains no usable examples")
    sampler = TokenBatchSampler(
        dataset.sizes,
        max_tokens=config.dataset.max_tokens,
        shuffle=False,
        required_batch_size_multiple=config.dataset.required_batch_size_multiple,
    )
    crop_generator = torch.Generator().manual_seed(config.common.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.dataset.num_workers,
        collate_fn=partial(
            collate_audio,
            max_sample_size=config.task.max_sample_size,
            pad=config.task.enable_padding,
            conv_layers=config.task.conv_feature_layers,
            generator=crop_generator,
        ),
    )
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_sample_size = 0
    counts = FrameCounts()
    score_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    segmented_evaluations: list[SegmentedEvaluation] = []
    try:
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            padding = batch.get("padding_mask")
            if config.stage == "pretrain":
                output = model(
                    batch["source"],
                    sample_ids=batch["id"],
                    update=update,
                    padding_mask=padding,
                )
            else:
                output = model(
                    batch["source"],
                    target=batch["target"],
                    sample_ids=batch["id"],
                    update=update,
                    padding_mask=padding,
                )
            if output.loss is None:
                raise ValueError("validation model did not return a loss")
            # Mathematics: validation accumulates summed losses L_k and token
            # counts N_k, then reports ΣL_k/max(ΣN_k,1).
            # Interpretation: metrics remain independent of how the token
            # sampler partitions the validation set into batches.
            total_loss += float(output.loss)
            total_sample_size += int(output.sample_size)
            if config.stage == "finetune":
                scores = torch.sigmoid(output.logits)
                targets = output.targets
                if targets is None:
                    raise ValueError("fine-tuning validation did not return targets")
                for sample_index in range(scores.shape[0]):
                    if output.padding_mask is None:
                        valid_frames = torch.ones(
                            scores.shape[1],
                            dtype=torch.bool,
                            device=scores.device,
                        )
                    else:
                        valid_frames = ~output.padding_mask[sample_index]
                    sample_scores = scores[sample_index, valid_frames]
                    sample_targets = targets[sample_index, valid_frames]
                    if sample_scores.numel() == 0:
                        continue
                    cpu_scores = sample_scores.detach().float().cpu()
                    cpu_targets = sample_targets.detach().float().cpu()
                    score_parts.append(cpu_scores)
                    target_parts.append(cpu_targets)
                    # Mathematics: threshold p>=τ and target y>=0.5 before
                    # adding binary confusion counts over every valid
                    # frame-class pair.
                    # Interpretation: padded audio cannot appear as a true
                    # negative, while soft labels retain a stable boundary.
                    counts += FrameCounts.from_predictions(
                        cpu_scores >= config.criterion.metric_threshold,
                        cpu_targets >= 0.5,
                    )

                    source_padding = batch.get("padding_mask")
                    if isinstance(source_padding, Tensor):
                        source_samples = int(
                            (~source_padding[sample_index]).sum().item()
                        )
                    else:
                        source_samples = int(batch["source"].shape[-1])
                    # Mathematics: archived feature rate is
                    # f_e=(T/S)f_s, hence pooling width
                    # w=round(f_e sigma_s). The scorer applies max(w,1)
                    # because PyTorch pooling requires a positive kernel.
                    # Interpretation: sigma_s retains its meaning in seconds
                    # for every frontend geometry and cropped recording.
                    feature_rate = (
                        cpu_scores.shape[0]
                        / max(source_samples, 1)
                        * config.task.sample_rate
                    )
                    window_frames = max(
                        1,
                        round(feature_rate * config.criterion.sigma_s),
                    )
                    segmented_evaluations.append(
                        legacy_segmented_evaluation(
                            cpu_scores.unsqueeze(0),
                            (cpu_targets >= 0.5).long().unsqueeze(0),
                            method=config.criterion.event_method,  # type: ignore[arg-type]
                            window_frames=window_frames,
                            metric_threshold=config.criterion.metric_threshold,
                            iou_threshold=config.criterion.iou_threshold,
                        )
                    )
    finally:
        model.train(was_training)

    metrics = {"loss": total_loss / max(total_sample_size, 1)}
    if config.stage == "finetune":
        frame_scores = torch.cat(score_parts) if score_parts else torch.empty(
            0, len(config.task.unique_labels)
        )
        frame_targets = torch.cat(target_parts) if target_parts else torch.empty_like(
            frame_scores
        )
        metrics.update({
            "precision": counts.precision,
            "recall": counts.recall,
            "f1": counts.f1,
            "accuracy": counts.accuracy,
            "average_precision": average_precision(
                frame_scores.reshape(-1), frame_targets.reshape(-1)
            ) if score_parts else 0.0,
        })
        segmented = aggregate_segmented_metrics(
            segmented_evaluations,
            config.task.unique_labels,
            metric_threshold=config.criterion.metric_threshold,
        )
        metrics.update({
            "segmented_precision": segmented.precision,
            "segmented_recall": segmented.recall,
            "segmented_f1": segmented.f1,
            "segmented_accuracy": segmented.accuracy,
            "segmented_average_precision": segmented.macro_average_precision,
            "segmented_micro_average_precision": segmented.micro_average_precision,
        })
        if segmented.focal_threshold is not None:
            metrics.update({
                "segmented_focal_threshold": segmented.focal_threshold,
                "segmented_focal_f1": segmented.focal_f1 or 0.0,
                "segmented_focal_precision": segmented.focal_precision or 0.0,
                "segmented_focal_recall": segmented.focal_recall or 0.0,
            })
        if tensorboard_logger is not None:
            tensorboard_logger.log_validation(
                metrics,
                update=update,
                subset=config.dataset.valid_subset,
                labels=config.task.unique_labels,
                frame_scores=frame_scores,
                frame_targets=frame_targets,
                segmented_evaluations=segmented_evaluations,
                metric_threshold=config.criterion.metric_threshold,
            )
    elif tensorboard_logger is not None:
        tensorboard_logger.log_validation(
            metrics,
            update=update,
            subset=config.dataset.valid_subset,
        )
    return metrics


def _tracked_metric(config: Animal2VecConfig, metrics: dict[str, float]) -> tuple[str, float, bool]:
    """Resolve the configured best-checkpoint metric and optimization direction."""

    requested = config.checkpoint.best_checkpoint_metric.rsplit("/", 1)[-1]
    aliases = {"ap": "average_precision", "mAP": "average_precision"}
    name = aliases.get(requested, requested)
    if name not in metrics:
        available = ", ".join(sorted(metrics))
        raise ValueError(
            f"best checkpoint metric {config.checkpoint.best_checkpoint_metric!r} is unavailable; "
            f"choose one of {available}"
        )
    return name, metrics[name], name != "loss"


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move tensor fields to one rank's device and preserve metadata fields."""

    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _restore_and_release_checkpoint(
    engine: TrainingEngine,
    checkpoint: dict[str, Any],
) -> None:
    """Restore mutable training state without retaining duplicate model tensors."""
    engine.restore(checkpoint)
    checkpoint.clear()


# Training orchestration

def _run_training(
    config: Animal2VecConfig,
    *,
    device_name: str,
    resume_path: Path | None,
    pretrained_checkpoint: Path | None,
    stop_at_update: int | None = None,
) -> Path:
    """Run native pretraining or fine-tuning and return the last checkpoint."""

    started_at = time.perf_counter()
    device, rank, world_size, initialized_here = _distributed_device(device_name)
    if config.distributed.requested_world_size != world_size:
        if initialized_here:
            dist.destroy_process_group()
        raise ValueError(
            "distributed_training.distributed_world_size="
            f"{config.distributed.requested_world_size} but the launch created {world_size} process(es); "
            "use torchrun with the configured worker count or override the field"
        )
    checkpoint_group = _checkpoint_process_group(device, world_size)
    # Mathematics: rank r starts each global RNG at seed+r; its exact evolved
    # states later enter distributed checkpoints.
    # Interpretation: workers draw distinct augmentations during a run but
    # reproduce their own stream after restart.
    random.seed(config.common.seed + rank)
    np.random.seed(config.common.seed + rank)
    torch.manual_seed(config.common.seed + rank)
    resume_checkpoint = load_checkpoint(resume_path, map_location=device) if resume_path is not None else None
    model, pretrained_config = _make_model(
        config,
        pretrained_checkpoint=pretrained_checkpoint,
        resume_checkpoint=resume_checkpoint,
    )
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = build_optimizer(
        model,
        name=config.optimizer.name,
        learning_rate=config.optimization.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    # Mathematics: scheduler horizon remains config.max_update even when
    # stop_at_update requests an earlier diagnostic termination.
    # Interpretation: a short verification run exercises the same learning
    # rates as the corresponding prefix of full paper training.
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=config.optimization.learning_rate,
        min_lr=config.scheduler.min_lr,
        warmup_updates=config.scheduler.warmup_updates,
        warmup_init_lr=config.scheduler.warmup_init_lr,
        max_updates=config.optimization.max_update,
    )
    wrapped: nn.Module = model
    if world_size > 1:
        # Mathematics: DDP averages each parameter gradient over W ranks before
        # the engine rescales by global sample count.
        # Interpretation: every GPU owns a full model replica and consumes a
        # distinct token batch in each synchronized update.
        wrapped = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=_ddp_find_unused_parameters(config),
            bucket_cap_mb_list=_ddp_bucket_cap_mb_list(config),
        )
    engine = TrainingEngine(
        wrapped,
        optimizer,
        scheduler,
        clip_norm=config.optimization.clip_norm,
        device=device,
        use_amp=config.common.fp16,
        amp_init_scale=config.common.fp16_init_scale,
        amp_min_scale=config.common.min_loss_scale,
    )
    if resume_checkpoint is not None:
        if resume_checkpoint["optimizer"] is None:
            raise CheckpointError("converted inference checkpoints cannot resume optimization")

    dataset = _make_dataset(config)
    sampler = TokenBatchSampler(
        dataset.sizes,
        max_tokens=config.dataset.max_tokens,
        seed=config.common.seed,
        shuffle=True,
        required_batch_size_multiple=config.dataset.required_batch_size_multiple,
    )
    if resume_checkpoint is not None and resume_checkpoint["sampler_state"] is not None:
        sampler.load_state_dict(resume_checkpoint["sampler_state"])
    if resume_checkpoint is not None:
        _restore_and_release_checkpoint(engine, resume_checkpoint)
    batch_sampler: Any = sampler
    if world_size > 1:
        batch_sampler = DistributedBatchSampler(sampler, rank=rank, world_size=world_size)
    # Mathematics: DataLoader worker seeding derives from a private generator
    # initialized at seed+rank and does not consume the model RNG stream.
    # Interpretation: worker creation and prefetch do not perturb dropout or
    # mask draws in the training model.
    loader_generator = torch.Generator().manual_seed(config.common.seed + rank)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.dataset.num_workers,
        generator=loader_generator,
        collate_fn=partial(
            collate_audio,
            max_sample_size=config.task.max_sample_size,
            pad=config.task.enable_padding,
            conv_layers=config.task.conv_feature_layers,
        ),
    )
    if len(dataset) == 0:
        raise ValueError("training manifest contains no usable examples")

    serialized: dict[str, object] = {"active": config_to_dict(config)}
    if pretrained_config is not None:
        serialized["pretrained"] = config_to_dict(pretrained_config)
    output_directory = config.checkpoint.save_dir
    output_directory.mkdir(parents=True, exist_ok=True)
    last_path = output_directory / "checkpoint_last.pt"
    last_validation_update = -1
    last_result: UpdateResult | None = None
    tensorboard_logger = (
        TensorBoardLogger(
            resolve_tensorboard_directory(config),
            purge_step=engine.update + 1 if engine.update > 0 else None,
        )
        if rank == 0
        else None
    )
    if tensorboard_logger is not None:
        tensorboard_logger.log_run(
            config,
            engine.unwrapped_model,
            update=engine.update,
        )

    def write_checkpoint(path: Path) -> None:
        """Gather rank RNG states and let rank zero write one checkpoint."""

        # Mathematics: rank r serializes RNG state R_r; gather forms the ordered
        # vector (R_0,...,R_{W-1}) on rank zero.
        # Interpretation: one checkpoint file can restore each worker's distinct
        # stochastic stream after a multi-GPU restart.
        payload = engine.checkpoint_payload(stage=config.stage, config=serialized)
        if world_size > 1:
            local_rng_state = payload["rng_state"]
            if not isinstance(local_rng_state, Mapping):
                raise CheckpointError("local RNG checkpoint state is not a mapping")
            gathered = gather_rank_rng_states(
                local_rng_state,
                world_size=world_size,
                rank=rank,
                group=checkpoint_group,
            )
            if rank == 0:
                if gathered is None:
                    raise CheckpointError("rank zero received no distributed RNG state")
                payload["rng_state"] = gathered
        if rank == 0:
            save_checkpoint(path, payload)

    def validate_and_checkpoint(*, epoch: int | None = None) -> None:
        """Run scheduled validation and update the best checkpoint."""

        nonlocal last_validation_update
        if not _validation_due(config, update=engine.update, epoch=epoch):
            return
        if last_validation_update == engine.update:
            return
        current = 0.0
        improved = False
        if rank == 0:
            metrics = _validate(
                engine.unwrapped_model,
                config,
                device=device,
                update=engine.update,
                tensorboard_logger=tensorboard_logger,
            )
            print(json.dumps({
                "validation": config.dataset.valid_subset,
                "update": engine.update,
                **metrics,
            }), flush=True)
            _, current, maximize = _tracked_metric(config, metrics)
            # Mathematics: loss improves under <, whereas AP/F1/accuracy-style
            # metrics improve under >.
            # Interpretation: the configured metric chooses checkpoint_best
            # without assuming that every useful measurement has one direction.
            improved = engine.best_metric is None or (
                current > engine.best_metric if maximize else current < engine.best_metric
            )
        if world_size > 1:
            decision = torch.tensor(
                [current, float(improved)],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(decision, src=0)
            current = float(decision[0])
            improved = bool(decision[1])
        if improved:
            engine.best_metric = current
            write_checkpoint(output_directory / "checkpoint_best.pt")
        last_validation_update = engine.update

    def record_update(result: UpdateResult) -> None:
        """Log one attempt and trigger update-based save and validation work."""

        nonlocal last_result
        last_result = result
        if rank == 0 and result.update % config.common.log_interval == 0:
            record: dict[str, object] = {
                "update": result.update,
                "loss": result.loss,
                "sample_size": result.sample_size,
                "gradient_norm": result.gradient_norm,
                "learning_rate": result.learning_rate,
                "skipped": result.skipped,
                "amp_scale": engine.scaler.get_scale() if engine.scaler is not None else None,
            }
            # Interpretation: fine-tuning has no masked mean-teacher tensors,
            # so its JSON schema remains unchanged. Pretraining records retain
            # the historical keys used by collapse-monitoring dashboards.
            if result.pred_var is not None and result.target_var is not None:
                record.update({
                    "pred_var": result.pred_var,
                    "target_var": result.target_var,
                })
            print(json.dumps(record), flush=True)
            if tensorboard_logger is not None:
                tensorboard_logger.log_update(
                    result,
                    stage=config.stage,
                    amp_scale=(
                        engine.scaler.get_scale()
                        if engine.scaler is not None
                        else None
                    ),
                )
        if result.skipped:
            return
        if config.checkpoint.save_interval_updates > 0 and result.update % config.checkpoint.save_interval_updates == 0:
            write_checkpoint(output_directory / f"checkpoint_{result.update}.pt")
        validate_and_checkpoint()

    # Mathematics: terminal update is min(stop_at_update,U_max) when a stop is
    # supplied, while the schedule continues to use U_max as its period.
    # Interpretation: verification can end early without changing the numerical
    # prefix it is intended to test.
    terminal_update = config.optimization.max_update
    if stop_at_update is not None:
        terminal_update = min(stop_at_update, terminal_update)

    while engine.update < terminal_update:
        iteration_epoch = sampler.epoch
        consumed_next_batch = sampler.next_batch
        # Mathematics: K_e uses update_freq[min(e-1,last)] microbatches per
        # logical optimizer update.
        # Interpretation: recipes can change effective batch size by epoch and
        # retain their last setting for later epochs.
        update_frequency = config.optimization.update_freq[
            min(engine.epoch - 1, len(config.optimization.update_freq) - 1)
        ]
        pending: list[dict[str, object]] = []
        made_progress = False
        loader_exhausted = True

        def forward(batch: dict[str, object]) -> object:
            """Dispatch a device batch to the active training-stage model."""

            padding = batch.get("padding_mask")
            if config.stage == "pretrain":
                return wrapped(
                    batch["source"],
                    sample_ids=batch["id"],
                    update=engine.update,
                    padding_mask=padding,
                )
            return wrapped(
                batch["source"],
                target=batch["target"],
                sample_ids=batch["id"],
                update=engine.update,
                padding_mask=padding,
            )

        for cpu_batch in loader:
            # DataLoader workers prefetch from the batch sampler. Track the
            # batches delivered to training separately so a checkpoint does
            # not skip queued-but-unconsumed batches after resume.
            # Mathematics: one delivered DDP round advances the global sampler
            # cursor by W complete token batches.
            # Interpretation: the saved cursor follows batches consumed by the
            # model instead of batches already queued by DataLoader workers.
            consumed_next_batch += world_size
            engine.sampler_state = {
                "epoch": iteration_epoch,
                "next_batch": consumed_next_batch,
            }
            pending.append(_move_batch(cpu_batch, device))
            if len(pending) < update_frequency:
                continue

            result = engine.step(pending, forward)
            pending = []
            made_progress = True
            record_update(result)
            if engine.update >= terminal_update:
                loader_exhausted = False
                break
        # Mathematics: an epoch remainder of 1,...,K-1 microbatches still forms
        # one update normalized by its own total sample count.
        # Interpretation: the final examples in an epoch are trained rather
        # than discarded because they do not fill an accumulation group.
        if pending and engine.update < terminal_update:
            result = engine.step(pending, forward)
            made_progress = True
            record_update(result)
        if loader_exhausted:
            engine.sampler_state = sampler.state_dict()
            validate_and_checkpoint(epoch=engine.epoch)
            if config.checkpoint.save_interval > 0 and engine.epoch % config.checkpoint.save_interval == 0:
                write_checkpoint(output_directory / f"checkpoint_epoch_{engine.epoch}.pt")
            engine.epoch += 1
        if not made_progress:
            # A checkpoint saved after the final batch resumes at the exhausted
            # iterator position. Iterating once advances the sampler epoch.
            continue

    write_checkpoint(last_path)
    if world_size > 1:
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary: dict[str, object] = {
        "rank": rank,
        "world_size": world_size,
        "final_update": engine.update,
        "terminal_update": terminal_update,
        "configured_max_update": config.optimization.max_update,
        "elapsed_seconds": time.perf_counter() - started_at,
        "amp_scale": engine.scaler.get_scale() if engine.scaler is not None else None,
    }
    if (
        last_result is not None
        and last_result.pred_var is not None
        and last_result.target_var is not None
    ):
        # Interpretation: short memory and resume burn-ins often stop before
        # log_interval. The terminal record still exposes their final collapse
        # diagnostics without adding mutable state to a checkpoint.
        summary.update({
            "pred_var": last_result.pred_var,
            "target_var": last_result.target_var,
        })
    if device.type == "cuda":
        summary.update({
            "cuda_device": device.index,
            "cuda_memory_allocated": torch.cuda.memory_allocated(device),
            "cuda_memory_reserved": torch.cuda.memory_reserved(device),
            "cuda_peak_memory_allocated": torch.cuda.max_memory_allocated(device),
            "cuda_peak_memory_reserved": torch.cuda.max_memory_reserved(device),
        })
    if tensorboard_logger is not None:
        tensorboard_logger.close()
        summary["tensorboard_logdir"] = str(tensorboard_logger.directory)
    print(json.dumps({"training_summary": summary}), flush=True)
    return last_path


def run_training(
    config: Animal2VecConfig,
    *,
    device_name: str,
    resume_path: Path | None,
    pretrained_checkpoint: Path | None,
    stop_at_update: int | None = None,
) -> Path:
    """Run training and release only a process group initialized by this call."""

    caller_owned_group = dist.is_initialized()
    try:
        return _run_training(
            config,
            device_name=device_name,
            resume_path=resume_path,
            pretrained_checkpoint=pretrained_checkpoint,
            stop_at_update=stop_at_update,
        )
    finally:
        if not caller_owned_group and dist.is_initialized():
            dist.destroy_process_group()


def train_main(argv: Sequence[str] | None = None) -> int:
    """Load a recipe, apply diagnostic limits, and start training."""

    parser = build_training_parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config, arguments.override)
        if arguments.max_updates is not None:
            if arguments.max_updates <= 0:
                raise ConfigError("--max-updates must be positive")
            config = replace(
                config,
                optimization=replace(config.optimization, max_update=arguments.max_updates),
            )
        if arguments.stop_at_update is not None and arguments.stop_at_update <= 0:
            raise ConfigError("--stop-at-update must be positive")
        run_training(
            config,
            device_name=arguments.device,
            resume_path=arguments.resume,
            pretrained_checkpoint=arguments.pretrained_checkpoint,
            stop_at_update=arguments.stop_at_update,
        )
    except (ConfigError, CheckpointError, ValueError) as exc:
        parser.error(str(exc))
    return 0


# =============================================================================
# CHECKPOINT-CONVERSION COMMAND
# =============================================================================

def build_checkpoint_conversion_parser() -> argparse.ArgumentParser:
    """Create the official-checkpoint conversion command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pretrained-config", type=Path)
    parser.add_argument("--stage", choices=("pretrain", "finetune"))
    return parser


def convert_checkpoint_main(argv: Sequence[str] | None = None) -> int:
    """Convert one checkpoint and print its tensor mapping report as JSON."""

    arguments = build_checkpoint_conversion_parser().parse_args(argv)
    report = convert_checkpoint(
        arguments.source,
        arguments.destination,
        config_path=arguments.config,
        pretrained_config_path=arguments.pretrained_config,
        stage=arguments.stage,
    )
    print(json.dumps(dataclasses.asdict(report), indent=2))
    return 0


# =============================================================================
# INFERENCE COMMAND
# =============================================================================

def build_inference_parser() -> argparse.ArgumentParser:
    """Create the native recording-inference command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--channel", type=int)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--method", choices=("avg", "max"))
    parser.add_argument("--fusion-window-seconds", type=float)
    parser.add_argument(
        "--audition",
        action="store_true",
        help=(
            "write an Adobe Audition marker CSV using the Animal2Vec 1.0 "
            "six-column schema instead of the native event TSV"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def infer_main(argv: Sequence[str] | None = None) -> int:
    """Run recording inference and write the selected event-file format."""

    arguments = build_inference_parser().parse_args(argv)
    waveform, sample_rate = load_audio(arguments.audio)
    runner = InferenceRunner(arguments.checkpoint, device=torch.device(arguments.device))
    result = runner.run_tensor(
        waveform,
        sample_rate,
        channel=arguments.channel,
        segment_seconds=arguments.segment_seconds,
        threshold=arguments.threshold,
        event_method=arguments.method,
        fusion_window_seconds=arguments.fusion_window_seconds,
    )
    # Mathematics: serialization applies the identity map to the event tuple;
    # only the coordinate representation changes from (start,end) seconds to
    # (start,duration) timedeltas when audition=True.
    # Interpretation: one flag adapts the file to a researcher's annotation
    # tool without rerunning or modifying the model's event decisions.
    runner.write_events(
        arguments.output,
        result.events,
        audition=arguments.audition,
    )
    return 0
