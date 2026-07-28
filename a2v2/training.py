"""Optimization, scheduling, checkpointing, metrics, and update execution.

This file contains the stateful machinery around the models. It implements the
Fairseq-compatible Adam variant, the update-indexed cosine schedule, complete
random-state checkpoints, frame metrics, automatic mixed precision, gradient
accumulation, distributed synchronization, and exact resume.

An ``update`` means one optimizer step, not one micro-batch. The distinction
controls learning-rate schedules, EMA updates, checkpoint numbering, and
gradient accumulation. Mathematical comments specify the optimizer equations
and reduction rules before conceptual comments connect them to reproducibility.
"""

from __future__ import annotations

import io
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor, nn
import torch.distributed as dist


# =============================================================================
# CHECKPOINT FORMAT AND RANDOM-STATE CAPTURE
# =============================================================================

FORMAT_VERSION = 1
REQUIRED_KEYS = {
    "format_version", "stage", "config", "model", "teacher", "optimizer",
    "scheduler", "scaler", "update", "epoch", "batch_in_epoch", "rng_state",
    "sampler_state", "best_metric",
}


class CheckpointError(ValueError):
    """Raised when a native checkpoint is unreadable or incompatible."""


def capture_rng_state() -> dict[str, object]:
    """Capture Python, NumPy, CPU PyTorch, and all CUDA generator states."""

    # Mathematics: each stochastic subsystem has an independent generator
    # state whose transition function determines its next random draw.
    # Interpretation: exact resume needs Python sampling, NumPy layerdrop,
    # PyTorch masks/dropout, and every CUDA device, not one global seed.
    numpy_state = np.random.get_state()
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def serialize_rng_state(state: Mapping[str, object]) -> bytes:
    """Wrap tensor-bearing RNG state for safe distributed object transport."""

    buffer = io.BytesIO()
    torch.save(dict(state), buffer)
    return buffer.getvalue()


def deserialize_rng_state(encoded: bytes) -> dict[str, object]:
    """Decode RNG state received through distributed object transport."""

    state = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise CheckpointError("distributed RNG payload is not a dictionary")
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore local or rank-specific random generators exactly."""

    if "by_rank" in state:
        # Mathematics: a distributed checkpoint stores states R_0,...,R_{W-1};
        # rank r must restore R_r under the same world size W.
        # Interpretation: workers consume different dropout and augmentation
        # streams, so broadcasting rank zero's state would diverge after resume.
        saved_world_size = int(state.get("world_size", 0))
        initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if initialized else 1
        rank = torch.distributed.get_rank() if initialized else 0
        ranked = state["by_rank"]
        if saved_world_size != world_size:
            raise CheckpointError(
                f"checkpoint RNG state has world size {saved_world_size}, current launch has {world_size}"
            )
        if not isinstance(ranked, (list, tuple)) or len(ranked) != saved_world_size:
            raise CheckpointError("invalid rank-specific RNG state")
        selected = ranked[rank]
        if not isinstance(selected, Mapping):
            raise CheckpointError(f"invalid RNG state for rank {rank}")
        state = selected
    # Mathematics: restoring generator state places every pseudorandom stream
    # at the exact point immediately after the checkpointed transition.
    # Interpretation: the next mask, crop, layer drop, and shuffle matches the
    # uninterrupted run rather than merely sharing its initial seed.
    random.setstate(state["python"])  # type: ignore[arg-type]
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise CheckpointError("invalid NumPy RNG state")
    keys = numpy_state["keys"]
    if not isinstance(keys, torch.Tensor):
        raise CheckpointError("invalid NumPy RNG key array")
    np.random.set_state((
        str(numpy_state["algorithm"]),
        keys.cpu().numpy().astype(np.uint32, copy=False),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise CheckpointError("invalid PyTorch RNG state")
    # A checkpoint loaded with map_location="cuda" moves every tensor,
    # including generator states, onto CUDA. Generator state setters require
    # CPU byte tensors even when restoring a CUDA generator.
    torch.random.set_rng_state(torch_state.cpu())
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, (list, tuple)) or not all(
            isinstance(item, torch.Tensor) for item in cuda_states
        ):
            raise CheckpointError("invalid CUDA RNG state")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def validate_checkpoint(payload: Mapping[str, object]) -> None:
    """Reject incomplete, unknown-version, or unknown-stage checkpoints."""

    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        raise CheckpointError(f"checkpoint is missing required keys: {missing}")
    if payload["format_version"] != FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format {payload['format_version']}; expected {FORMAT_VERSION}"
        )
    if payload["stage"] not in {"pretrain", "finetune"}:
        raise CheckpointError(f"unknown checkpoint stage: {payload['stage']}")


def save_checkpoint(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically write a validated native checkpoint."""

    validate_checkpoint(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mathematics: os.replace is an atomic name swap on the same filesystem;
    # readers observe either the previous complete file or the new complete file.
    # Interpretation: interruption during torch.save cannot leave the canonical
    # checkpoint path pointing at a partially written archive.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a native checkpoint on the requested device."""

    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a dictionary")
    validate_checkpoint(payload)
    return payload


# =============================================================================
# FAIRSEQ-COMPATIBLE ADAM AND COSINE SCHEDULE
# =============================================================================

class FairseqCompatibleAdam(torch.optim.Optimizer):
    """AdamW update with the epsilon placement used by Fairseq 0.12."""

    def __init__(
        self,
        params: object,
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        })

    @torch.no_grad()
    def step(self, closure: object = None) -> object:
        """Apply one decoupled-weight-decay Adam update.

        The denominator uses ``sqrt(v) + eps`` before multiplication by the
        bias-corrected step size. Preserving this order is required for
        optimizer-update parity with the archived implementation.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Fairseq-compatible Adam does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                first_moment = state["exp_avg"]
                second_moment = state["exp_avg_sq"]
                # Mathematics: m_t=β1 m_{t-1}+(1-β1)g_t and
                # v_t=β2 v_{t-1}+(1-β2)g_t².
                # Interpretation: the optimizer tracks smoothed gradient
                # direction and scale for each trainable coordinate.
                first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                # Mathematics: α_t=lr sqrt(1-β2^t)/(1-β1^t) applies bias
                # correction to moments initialized at zero.
                # Interpretation: early updates have the intended magnitude
                # before the moving averages reach steady state.
                step_size = (
                    group["lr"]
                    * math.sqrt(1 - beta2 ** state["step"])
                    / (1 - beta1 ** state["step"])
                )
                if group["weight_decay"]:
                    # Mathematics: decoupled decay applies θ <- (1-lr λ)θ
                    # independently of the adaptive gradient denominator.
                    # Interpretation: regularization does not get amplified or
                    # suppressed by Adam's per-parameter variance estimate.
                    parameter.add_(
                        parameter,
                        alpha=-group["weight_decay"] * group["lr"],
                    )
                # Mathematics: θ_t = θ_decay - α_t m_t/(sqrt(v_t)+eps);
                # Fairseq places eps inside the denominator before α_t.
                # Interpretation: keeping this operation order is necessary for
                # numerical parity with archived optimizer checkpoints.
                denominator = second_moment.sqrt().add_(group["eps"])
                parameter.addcdiv_(first_moment, denominator, value=-step_size)
        return loss


def build_optimizer(
    model: nn.Module,
    *,
    name: str,
    learning_rate: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build optimizer groups with biases and normalization scales un-decayed."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Mathematics: one-dimensional parameters, biases, ALiBi scales, and
        # parametric-Swish coefficients belong to the λ=0 group; remaining
        # trainable tensors belong to the configured λ group.
        # Interpretation: normalization and calibration parameters avoid weight
        # decay while matrix and convolution kernels remain regularized.
        if (
            parameter.ndim == 1
            or parameter_name.endswith(".bias")
            or "alibi_scale" in parameter_name
            or "p_swish" in parameter_name
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    normalized_name = name.lower()
    optimizer_class: type[torch.optim.Optimizer]
    if normalized_name == "adam":
        optimizer_class = FairseqCompatibleAdam
    elif normalized_name == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f"unsupported optimizer: {name}")
    return optimizer_class(groups, lr=learning_rate, betas=betas, eps=eps)


class CosineUpdateScheduler:
    """Cosine schedule with Fairseq's update-number convention."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_lr: float,
        min_lr: float,
        warmup_updates: int,
        max_updates: int,
        warmup_init_lr: float | None = None,
        period_updates: int | None = None,
        period_multiplier: float = 1.0,
        lr_shrink: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min(min_lr, max_lr)
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        self.warmup_init_lr = self.min_lr if warmup_init_lr is None or warmup_init_lr < 0 else warmup_init_lr
        self.period = period_updates if period_updates is not None and period_updates > 0 else max_updates - warmup_updates
        if self.period <= 0:
            raise ValueError("cosine period must be positive")
        self.period_multiplier = period_multiplier
        self.lr_shrink = lr_shrink
        self.last_update = -1
        self._set_lr(self.warmup_init_lr)

    def _set_lr(self, learning_rate: float) -> None:
        """Write one learning rate into every optimizer parameter group."""

        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def lr_at_update(self, update: int) -> float:
        """Calculate the warmup or cosine rate for an absolute update number."""

        if update < self.warmup_updates:
            if self.warmup_updates == 0:
                return self.max_lr
            # Mathematics: lr(u)=lr_0+u(lr_max-lr_0)/U_warm for u<U_warm.
            # Interpretation: linear warmup limits early updates before Adam's
            # moments and transformer activations have stabilized.
            step = (self.max_lr - self.warmup_init_lr) / self.warmup_updates
            return self.warmup_init_lr + update * step
        current = update - self.warmup_updates
        if self.period_multiplier != 1:
            cycle = math.floor(
                math.log(1 - current / self.period * (1 - self.period_multiplier), self.period_multiplier)
            )
            cycle_length = self.period_multiplier**cycle * self.period
            cycle_position = current - (1 - self.period_multiplier**cycle) / (1 - self.period_multiplier) * self.period
        else:
            cycle = math.floor(current / self.period)
            cycle_length = self.period
            cycle_position = current - self.period * cycle
        # Mathematics: within cycle c,
        # lr = lr_min,c + 1/2(lr_max,c-lr_min,c)(1+cos(πp/P_c)),
        # with both bounds multiplied by shrink^c.
        # Interpretation: the update rate falls smoothly to its minimum and can
        # restart on longer or shorter cycles when a recipe requests it.
        shrink = self.lr_shrink**cycle
        minimum = self.min_lr * shrink
        maximum = self.max_lr * shrink
        return minimum + 0.5 * (maximum - minimum) * (
            1 + math.cos(math.pi * cycle_position / cycle_length)
        )

    def step_update(self, update: int) -> float:
        """Apply and record the learning rate for ``update``."""

        learning_rate = self.lr_at_update(update)
        self._set_lr(learning_rate)
        self.last_update = update
        return learning_rate

    def state_dict(self) -> dict[str, int | float]:
        """Serialize the last applied update; fixed settings come from config."""

        return {"last_update": self.last_update}

    def load_state_dict(self, state: dict[str, int | float]) -> None:
        """Restore the update index and recompute its learning rate."""

        self.last_update = int(state["last_update"])
        self._set_lr(self.lr_at_update(self.last_update))


# =============================================================================
# FRAME-LEVEL METRICS
# =============================================================================

@dataclass(frozen=True)
class FrameCounts:
    """Additive binary confusion counts with common derived metrics."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @classmethod
    def from_predictions(cls, prediction: Tensor, target: Tensor) -> "FrameCounts":
        """Count binary outcomes for two same-shaped tensors."""

        prediction = prediction.bool()
        target = target.bool()
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        # Mathematics: TP=Σ[p∧y], FP=Σ[p∧¬y], TN=Σ[¬p∧¬y], FN=Σ[¬p∧y].
        # Interpretation: reducing to four additive counts lets validation
        # combine arbitrary batches without retaining every frame.
        return cls(
            true_positive=int((prediction & target).sum()),
            false_positive=int((prediction & ~target).sum()),
            true_negative=int((~prediction & ~target).sum()),
            false_negative=int((~prediction & target).sum()),
        )

    def __add__(self, other: "FrameCounts") -> "FrameCounts":
        return FrameCounts(
            self.true_positive + other.true_positive,
            self.false_positive + other.false_positive,
            self.true_negative + other.true_negative,
            self.false_negative + other.false_negative,
        )

    @property
    def precision(self) -> float:
        """Return the fraction of predicted positives that are correct."""

        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Return the fraction of target positives recovered by predictions."""

        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        """Return the harmonic mean of precision and recall."""

        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def accuracy(self) -> float:
        """Return the fraction of all binary decisions that are correct."""

        total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        return (self.true_positive + self.true_negative) / total if total else 0.0


def average_precision(scores: Tensor, targets: Tensor) -> float:
    """Compute binary average precision with tied scores handled as a group.

    Grouping equal scores matches threshold-based AP: reordering examples that
    share the same score cannot change the result.
    """

    scores = scores.detach().reshape(-1).float()
    targets = targets.detach().reshape(-1).bool()
    positives = int(targets.sum())
    if positives == 0:
        return 0.0
    # Mathematics: sort by decreasing score, then evaluate precision and recall
    # only at the final member of each equal-score group.
    # Interpretation: arbitrary ordering among tied predictions cannot change
    # the area under the precision-recall staircase.
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    group_end = torch.ones_like(sorted_targets)
    group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    cumulative_positive = sorted_targets.cumsum(dim=0)[group_end].to(torch.float64)
    cumulative_count = torch.arange(
        1, len(sorted_targets) + 1, device=scores.device, dtype=torch.float64
    )[group_end]
    precision = cumulative_positive / cumulative_count
    recall = cumulative_positive / positives
    # Mathematics: AP = Σ_k (R_k-R_{k-1}) P_k; reversed recall plus a terminal
    # zero expresses the same Riemann sum through negative finite differences.
    # Interpretation: the metric weights precision by each newly recovered
    # fraction of positive animal frames.
    decreasing_recall = torch.cat([recall.flip(0), recall.new_zeros(1)])
    return float(-(torch.diff(decreasing_recall) * precision.flip(0)).sum())


# =============================================================================
# UPDATE-BASED TRAINING ENGINE
# =============================================================================

class LossOutput(Protocol):
    """Structural type returned by a model-specific training forward call."""

    loss: Tensor
    sample_size: int


@dataclass(frozen=True)
class UpdateResult:
    """Reduced measurements and optimizer outcome for one logical update."""

    loss: float
    sample_size: int
    gradient_norm: float
    learning_rate: float
    update: int
    skipped: bool


class TrainingEngine:
    """Own gradient accumulation, AMP, distributed reduction, and resume state.

    The engine is intentionally independent of datasets and model stages.
    Callers supply microbatches and a forward function returning summed loss
    plus Fairseq-compatible ``sample_size``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineUpdateScheduler,
        *,
        clip_norm: float,
        device: torch.device,
        use_amp: bool = False,
        amp_init_scale: float = 128.0,
        amp_min_scale: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_norm = clip_norm
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_min_scale = amp_min_scale
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True, init_scale=amp_init_scale)
            if self.use_amp else None
        )
        self.update = 0
        self.epoch = 1
        self.batch_in_epoch = 0
        self.best_metric: float | None = None
        self.sampler_state: dict[str, object] | None = None

    @property
    def unwrapped_model(self) -> nn.Module:
        """Return the user model beneath an optional DDP wrapper."""

        return self.model.module if hasattr(self.model, "module") else self.model

    def step(
        self,
        microbatches: Iterable[Any],
        forward: Callable[[Any], LossOutput],
    ) -> UpdateResult:
        """Accumulate microbatches and perform one logical optimizer update."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_sample_size = 0
        total_loss = 0.0
        # Mathematics: if an update contains K microbatches, backpropagate each
        # summed loss before a single normalization and optimizer step.
        # Interpretation: gradient accumulation emulates a larger batch without
        # allocating all waveforms and activations at once.
        batches = list(microbatches)
        for microbatch in batches:
            amp_context = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_context:
                output = forward(microbatch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError(f"non-finite loss at update {self.update}")
            if self.scaler is None:
                output.loss.backward()
            else:
                self.scaler.scale(output.loss).backward()
            # Mathematics: accumulate L_sum=Σ_k L_k and N_sum=Σ_k N_k.
            # Interpretation: later normalization weights every frame token
            # equally even when microbatches contain different lengths.
            total_sample_size += int(output.sample_size)
            total_loss += float(output.loss.detach())
            self.batch_in_epoch += 1

        totals = torch.tensor(
            [float(total_sample_size), total_loss],
            device=self.device,
            dtype=torch.float64,
        )
        world_size = 1
        # Mathematics: all-reduce sums [N_r,L_r] over ranks r=0,...,W-1.
        # Interpretation: logging and normalization describe the global batch,
        # not the local shard seen by one GPU.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(totals)
        global_sample_size = max(float(totals[0].item()), 1.0)
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        # Mathematics: DDP has already averaged gradients by 1/W, so multiplying
        # by W/N_global yields ∇(Σ_r L_r / N_global).
        # Interpretation: one update is invariant to world size and uneven
        # frame counts across rank-local token batches.
        multiplier = world_size / global_sample_size
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(multiplier)
        # Mathematics: for threshold c>0, gradients scale by
        # min(1,c/||g||_2); c<=0 maps to infinity and leaves them unchanged.
        # Interpretation: clipping limits rare unstable updates while recording
        # the pre-clipping norm for diagnostics.
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip_norm if self.clip_norm > 0 else float("inf")
        )
        if not torch.isfinite(gradient_norm):
            if self.scaler is None:
                raise FloatingPointError(f"non-finite gradients at update {self.update}")
            # GradScaler records non-finite gradients during unscale_. Let it
            # skip the optimizer step and reduce the scale without advancing
            # scheduler, model-update, or EMA state.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < self.amp_min_scale:
                raise FloatingPointError(
                    f"AMP loss scale {self.scaler.get_scale()} is below minimum {self.amp_min_scale}"
                )
            return UpdateResult(
                loss=float(totals[1].item()) / global_sample_size,
                sample_size=int(totals[0].item()),
                gradient_norm=float(gradient_norm),
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
                update=self.update,
                skipped=True,
            )
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < self.amp_min_scale:
                raise FloatingPointError(
                    f"AMP loss scale {self.scaler.get_scale()} is below minimum {self.amp_min_scale}"
                )
        # Mathematics: optimizer state transitions first, then update index
        # u<-u+1, learning rate lr(u), and teacher EMA at the same u.
        # Interpretation: scheduler and teacher clocks advance only after a
        # successful parameter update; AMP-skipped attempts leave them fixed.
        self.update += 1
        learning_rate = self.scheduler.step_update(self.update)
        update_teacher = getattr(self.unwrapped_model, "update_teacher", None)
        if callable(update_teacher):
            update_teacher(self.update)
        return UpdateResult(
            loss=float(totals[1].item()) / global_sample_size,
            sample_size=int(totals[0].item()),
            gradient_norm=float(gradient_norm),
            learning_rate=learning_rate,
            update=self.update,
            skipped=False,
        )

    def checkpoint_payload(
        self,
        *,
        stage: str,
        config: Mapping[str, object],
    ) -> dict[str, object]:
        """Create a complete, versioned, exactly resumable checkpoint payload."""

        raw_model = self.unwrapped_model
        teacher = getattr(raw_model, "teacher", None)
        # Mathematics: this payload captures all state variables needed to make
        # the next transition F(state,batch) identical after a restart.
        # Interpretation: model weights alone support inference, while optimizer,
        # counters, sampler cursor, scaler, RNGs, and best metric support resume.
        return {
            "format_version": 1,
            "stage": stage,
            "config": dict(config),
            "model": raw_model.state_dict(),
            "teacher": teacher.model.state_dict() if teacher is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "update": self.update,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "rng_state": capture_rng_state(),
            "sampler_state": self.sampler_state,
            "best_metric": self.best_metric,
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore model, optimizer, scheduler, counters, scaler, and RNGs."""

        # Mathematics: restore every mutable component before the next batch;
        # strict model loading enforces a bijection between saved and live keys.
        # Interpretation: a checkpoint created by a different architecture or
        # incomplete implementation fails instead of training with mixed state.
        self.unwrapped_model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler is not None and checkpoint["scaler"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.update = int(checkpoint["update"])
        self.epoch = int(checkpoint["epoch"])
        self.batch_in_epoch = int(checkpoint["batch_in_epoch"])
        self.sampler_state = checkpoint["sampler_state"]
        self.best_metric = checkpoint["best_metric"]
        restore_rng_state(checkpoint["rng_state"])
