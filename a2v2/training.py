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

import importlib
import io
import math
import os
import random
import re
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

FORMAT_VERSION = 2
RANK_LOCAL_RNG_SCHEMA = "a2v2.rng.rank-local.v1"
LEGACY_REQUIRED_KEYS = {
    "format_version", "stage", "config", "model", "teacher", "optimizer",
    "scheduler", "scaler", "update", "epoch", "batch_in_epoch", "rng_state",
    "sampler_state", "best_metric",
}
VERSION_2_STATE_KEYS = {
    "gradient_clipper",
    "weight_decay_scheduler",
    "topology",
    "resume_compatibility",
}
REQUIRED_KEYS = LEGACY_REQUIRED_KEYS | VERSION_2_STATE_KEYS


class CheckpointError(ValueError):
    """Raised when a native checkpoint is unreadable or incompatible."""


class DistributedOptimizerStepError(RuntimeError):
    """Mark a distributed engine terminal after any rank's optimizer failure."""


class OptimizerSetupError(RuntimeError):
    """Raised when a selected optional optimizer cannot be initialized safely."""


_DISTRIBUTED_OPTIMIZER_FAILURE = (
    "optimizer step failed on one or more ranks; distributed training is terminal "
    "and must recover from the last atomic checkpoint"
)


def _all_ranks_true(local_value: bool, device: torch.device) -> bool:
    """Return one rank-consistent Boolean over the active process group."""

    if not (dist.is_available() and dist.is_initialized()):
        return local_value
    decision = torch.tensor(int(local_value), dtype=torch.int32, device=device)
    dist.all_reduce(decision)
    return int(decision.item()) == dist.get_world_size()


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


def gather_rank_rng_states(
    state: Mapping[str, object],
    *,
    world_size: int,
    rank: int,
    group: dist.ProcessGroup | None,
) -> dict[str, object] | None:
    """Gather serialized per-rank RNG states through a selected process group."""

    encoded = serialize_rng_state(state)
    gathered: list[object] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(encoded, gathered, dst=0, group=group)
    if rank != 0:
        return None
    if (
        gathered is None
        or len(gathered) != world_size
        or any(not isinstance(item, bytes) for item in gathered)
    ):
        raise CheckpointError(
            "distributed RNG gather did not return one byte payload per rank"
        )
    return {
        "schema": RANK_LOCAL_RNG_SCHEMA,
        "world_size": world_size,
        "by_rank": [deserialize_rng_state(item) for item in gathered],
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore local or rank-specific random generators exactly."""

    if "by_rank" in state:
        # Mathematics: a distributed checkpoint stores states R_0,...,R_{W-1};
        # rank r must restore R_r under the same world size W.
        # Interpretation: workers consume different dropout and augmentation
        # streams, so broadcasting rank zero's state would diverge after resume.
        schema = state.get("schema")
        if schema is not None and schema != RANK_LOCAL_RNG_SCHEMA:
            raise CheckpointError(
                f"unsupported rank-local RNG schema {schema!r}; "
                f"expected {RANK_LOCAL_RNG_SCHEMA}"
            )
        saved_world_size_value = state.get("world_size")
        if type(saved_world_size_value) is not int:
            raise CheckpointError("rank-specific RNG world size must be an integer")
        saved_world_size = saved_world_size_value
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
    required = {"python", "numpy", "torch"}
    missing = sorted(required - set(state))
    if missing:
        raise CheckpointError(f"RNG state is missing required keys: {missing}")

    python_state = state["python"]
    try:
        python_probe = random.Random()
        python_probe.setstate(python_state)  # type: ignore[arg-type]
    except Exception as error:
        raise CheckpointError(f"invalid python RNG state: {error}") from error

    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise CheckpointError("invalid numpy RNG state")
    numpy_required = {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    numpy_missing = sorted(numpy_required - set(numpy_state))
    if numpy_missing:
        raise CheckpointError(
            f"invalid numpy RNG state; missing keys: {numpy_missing}"
        )
    keys = numpy_state.get("keys")
    if not isinstance(keys, torch.Tensor):
        raise CheckpointError("invalid numpy RNG key array")
    try:
        validated_numpy_state = (
            str(numpy_state["algorithm"]),
            keys.cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(validated_numpy_state)
    except Exception as error:
        raise CheckpointError(f"invalid numpy RNG state: {error}") from error

    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise CheckpointError("invalid torch RNG state")
    try:
        validated_torch_state = torch_state.cpu().clone()
        torch_probe = torch.Generator(device="cpu")
        torch_probe.set_state(validated_torch_state)
    except Exception as error:
        raise CheckpointError(f"invalid torch RNG state: {error}") from error

    cuda_states: list[Tensor] | None = None
    if torch.cuda.is_available() and "cuda" in state:
        raw_cuda_states = state["cuda"]
        if not isinstance(raw_cuda_states, (list, tuple)) or not all(
            isinstance(item, torch.Tensor) for item in raw_cuda_states
        ):
            raise CheckpointError("invalid cuda RNG state")
        visible_devices = torch.cuda.device_count()
        if len(raw_cuda_states) != visible_devices:
            raise CheckpointError(
                f"checkpoint has {len(raw_cuda_states)} CUDA RNG states but the "
                f"local rank has {visible_devices} visible CUDA devices"
            )
        cuda_states = [item.cpu().clone() for item in raw_cuda_states]
        try:
            for device_index, cuda_state in enumerate(cuda_states):
                cuda_probe = torch.Generator(
                    device=torch.device("cuda", device_index)
                )
                cuda_probe.set_state(cuda_state)
        except Exception as error:
            raise CheckpointError(f"invalid cuda RNG state: {error}") from error
    # Mathematics: restoring generator state places every pseudorandom stream
    # at the exact point immediately after the checkpointed transition.
    # Interpretation: the next mask, crop, layer drop, and shuffle matches the
    # uninterrupted run rather than merely sharing its initial seed.
    random.setstate(python_state)  # type: ignore[arg-type]
    np.random.set_state(validated_numpy_state)
    # A checkpoint loaded with map_location="cuda" moves every tensor,
    # including generator states, onto CUDA. Generator state setters require
    # CPU byte tensors even when restoring a CUDA generator.
    torch.random.set_rng_state(validated_torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def validate_checkpoint(payload: Mapping[str, object]) -> None:
    """Reject incomplete, unknown-version, or unknown-stage checkpoints."""

    if "format_version" not in payload:
        raise CheckpointError("checkpoint is missing required keys: ['format_version']")
    version = payload.get("format_version")
    if version not in {1, FORMAT_VERSION}:
        raise CheckpointError(
            f"unsupported checkpoint format {version}; expected 1 or {FORMAT_VERSION}"
        )
    required = LEGACY_REQUIRED_KEYS if version == 1 else REQUIRED_KEYS
    missing = sorted(required - set(payload))
    if missing:
        raise CheckpointError(f"checkpoint is missing required keys: {missing}")
    if payload["stage"] not in {"pretrain", "finetune"}:
        raise CheckpointError(f"unknown checkpoint stage: {payload['stage']}")


def normalize_checkpoint(payload: Mapping[str, object]) -> dict[str, Any]:
    """Add v2 runtime slots to a validated v1 payload using legacy defaults."""

    normalized = dict(payload)
    if normalized["format_version"] == 1:
        for key in VERSION_2_STATE_KEYS:
            normalized.setdefault(key, None)
    return normalized


def resume_compatibility_fingerprint(
    active_config: Mapping[str, object],
) -> dict[str, object] | None:
    """Select mathematical and batching fields that must match on resume."""

    fingerprint: dict[str, object] = {}

    def add_leaf(path: str, value: object) -> None:
        """Flatten nested config mappings while preserving sequence values."""

        if isinstance(value, Mapping):
            for key in sorted(value):
                add_leaf(f"{path}.{key}", value[key])
        else:
            fingerprint[path] = value

    try:
        add_leaf("model", active_config["model"])
        add_leaf("optimizer", active_config["optimizer"])
        add_leaf("scheduler", active_config["scheduler"])
        add_leaf("optimization", active_config["optimization"])
        # The established CLI permits extending max_update on resume. It is a
        # scheduler horizon override, not saved optimizer or clipping state.
        fingerprint.pop("optimization.max_update", None)
        common = active_config["common"]
        dataset = active_config["dataset"]
        distributed = active_config["distributed"]
        if not all(isinstance(group, Mapping) for group in (common, dataset, distributed)):
            return None
        fingerprint["common.seed"] = common["seed"]  # type: ignore[index]
        fingerprint["dataset.max_tokens"] = dataset["max_tokens"]  # type: ignore[index]
        fingerprint["dataset.required_batch_size_multiple"] = dataset[
            "required_batch_size_multiple"
        ]  # type: ignore[index]
        fingerprint["distributed.requested_world_size"] = distributed[
            "requested_world_size"
        ]  # type: ignore[index]
    except (KeyError, TypeError):
        return None
    return fingerprint


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
    return normalize_checkpoint(payload)


# =============================================================================
# TRANSACTIONAL GRADIENT CLIPPING
# =============================================================================

@dataclass(frozen=True)
class GradientClipCandidate:
    """Gradient diagnostic plus tentative state for one optimizer attempt."""

    gradient_norm: Tensor
    finite: bool
    clipped_tensors: int = 0
    largest_scale: float = 1.0
    source_update: int | None = None
    norm_emas: Mapping[str, Tensor] | None = None


class GradientClipper(Protocol):
    """Clip gradients now and commit state only after an optimizer update."""

    def clip(self) -> GradientClipCandidate:
        """Return one pre-step diagnostic and tentative state transition."""

        ...

    def commit(self, candidate: GradientClipCandidate) -> None:
        """Commit a candidate after the corresponding optimizer step."""

        ...

    def state_dict(self) -> dict[str, object] | None:
        """Serialize strategy state or return None for stateless clipping."""

        ...

    def load_state_dict(self, state: Mapping[str, object] | None) -> None:
        """Restore and validate strategy state."""

        ...


def _dense_parameters(parameters: Iterable[nn.Parameter]) -> tuple[nn.Parameter, ...]:
    """Materialize parameters and reject every non-strided gradient layout."""

    materialized = tuple(parameters)
    if any(
        parameter.grad is not None and parameter.grad.layout != torch.strided
        for parameter in materialized
    ):
        raise RuntimeError("gradient clipping does not support sparse gradients")
    return materialized


class GlobalGradientClipper:
    """Preserve the legacy single-global-norm clipping operation exactly."""

    def __init__(self, parameters: Iterable[nn.Parameter], *, clip_norm: float) -> None:
        self.parameters = tuple(parameters)
        self.clip_norm = clip_norm

    @torch.no_grad()
    def clip(self) -> GradientClipCandidate:
        """Apply the archived global operation and return its diagnostic."""

        parameters = _dense_parameters(self.parameters)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.clip_norm if self.clip_norm > 0 else float("inf"),
        )
        return GradientClipCandidate(
            gradient_norm=gradient_norm,
            finite=bool(torch.isfinite(gradient_norm)),
        )

    def commit(self, candidate: GradientClipCandidate) -> None:
        """Fixed clipping has no state to commit."""

    def state_dict(self) -> None:
        """Return no checkpoint state for the legacy fixed strategy."""

        return None

    def load_state_dict(self, state: Mapping[str, object] | None) -> None:
        """Reject state attached to a stateless fixed strategy."""

        if state is not None:
            raise CheckpointError("global gradient clipping does not accept clipper state")


class NoGradientClipper(GlobalGradientClipper):
    """Measure the legacy global diagnostic without bounding gradients."""

    def __init__(self, parameters: Iterable[nn.Parameter]) -> None:
        super().__init__(parameters, clip_norm=0.0)


@dataclass(frozen=True)
class AdaGradientClipperState:
    """Validated but not yet installed AdaGC checkpoint state."""

    update: int
    norm_emas: Mapping[str, Tensor]


class AdaGradientClipper:
    """Paper AdaGC with GlobalGC minimum initialization and tentative state."""

    ALGORITHM_VERSION = 1

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        *,
        clip_norm: float,
        beta: float = 0.99,
        relative_clip: float = 1.04,
        warmup_updates: int = 100,
    ) -> None:
        self.named_parameters = tuple(
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad
        )
        self.parameter_names = tuple(name for name, _ in self.named_parameters)
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("AdaGC parameter names must be unique")
        self.clip_norm = clip_norm
        self.beta = beta
        self.relative_clip = relative_clip
        self.warmup_updates = warmup_updates
        self.update = 0
        # Positive infinity represents a named tensor with no observation yet.
        self.norm_emas = {
            name: torch.tensor(float("inf"), dtype=torch.float32)
            for name in self.parameter_names
        }

    def _gradient_norms(self) -> tuple[list[tuple[str, nn.Parameter, Tensor]], Tensor]:
        """Return dense per-tensor norms and their pre-clipping global L2 norm."""

        _dense_parameters(parameter for _, parameter in self.named_parameters)
        gradients = [
            (name, parameter, torch.linalg.vector_norm(parameter.grad.detach(), ord=2))
            for name, parameter in self.named_parameters
            if parameter.grad is not None
        ]
        if not gradients:
            return gradients, torch.tensor(0.0)
        device = gradients[0][2].device
        global_norm = torch.linalg.vector_norm(
            torch.stack([norm.to(device) for _, _, norm in gradients]),
            ord=2,
        )
        return gradients, global_norm

    @torch.no_grad()
    def clip(self) -> GradientClipCandidate:
        """Mutate gradients while leaving the running norms tentative."""

        gradients, pre_clip_global_norm = self._gradient_norms()
        if not bool(torch.isfinite(pre_clip_global_norm)):
            return GradientClipCandidate(
                gradient_norm=pre_clip_global_norm,
                finite=False,
                source_update=self.update,
            )

        candidate_norms = {
            name: value.clone()
            for name, value in self.norm_emas.items()
        }
        clipped_tensors = 0
        largest_scale = 0.0 if gradients else 1.0
        if self.update < self.warmup_updates:
            # The existing GlobalGC operation is retained during updates
            # 0,...,T_start-1. State observes norms only after that operation.
            pre_clip_global_norm = torch.nn.utils.clip_grad_norm_(
                tuple(parameter for _, parameter in self.named_parameters),
                self.clip_norm if self.clip_norm > 0 else float("inf"),
            )
            for name, parameter, original_norm in gradients:
                clipped_norm = torch.linalg.vector_norm(parameter.grad.detach(), ord=2).float().cpu()
                candidate_norms[name] = torch.minimum(candidate_norms[name], clipped_norm)
                original = float(original_norm.float().cpu())
                tensor_scale = min(float(clipped_norm) / original, 1.0) if original else 1.0
                largest_scale = max(largest_scale, tensor_scale)
                if float(clipped_norm) < float(original_norm.float().cpu()):
                    clipped_tensors += 1
        else:
            for name, parameter, original_norm in gradients:
                norm = original_norm.float().cpu()
                previous = candidate_norms[name]
                if float(norm) == 0.0 or bool(torch.isinf(previous)):
                    scale = 1.0
                else:
                    scale = min(
                        self.relative_clip * float(previous) / float(norm),
                        1.0,
                    )
                parameter.grad.mul_(scale)
                clipped_norm = torch.linalg.vector_norm(parameter.grad.detach(), ord=2).float().cpu()
                if bool(torch.isinf(previous)):
                    candidate_norms[name] = clipped_norm
                else:
                    candidate_norms[name] = (
                        previous * self.beta + clipped_norm * (1.0 - self.beta)
                    )
                if scale < 1.0:
                    clipped_tensors += 1
                largest_scale = max(largest_scale, scale)

        return GradientClipCandidate(
            gradient_norm=pre_clip_global_norm,
            finite=True,
            clipped_tensors=clipped_tensors,
            largest_scale=largest_scale,
            source_update=self.update,
            norm_emas=candidate_norms,
        )

    def commit(self, candidate: GradientClipCandidate) -> None:
        """Commit exactly one finite candidate from the current update."""

        if not candidate.finite or candidate.norm_emas is None:
            raise CheckpointError("cannot commit non-finite AdaGC candidate state")
        if candidate.source_update != self.update:
            raise CheckpointError(
                "cannot commit stale AdaGC candidate: "
                f"candidate update {candidate.source_update}, clipper update {self.update}"
            )
        self.norm_emas = {
            name: candidate.norm_emas[name].detach().cpu().to(torch.float32).clone()
            for name in self.parameter_names
        }
        self.update += 1

    def state_dict(self) -> dict[str, object]:
        """Serialize ordered names and name-keyed CPU FP32 running norms."""

        return {
            "algorithm_version": self.ALGORITHM_VERSION,
            "update": self.update,
            "parameter_names": list(self.parameter_names),
            "norm_emas": {
                name: value.clone()
                for name, value in self.norm_emas.items()
            },
        }

    def pristine_state(self) -> AdaGradientClipperState:
        """Build explicit update-zero state for a checkpoint with no history."""

        return AdaGradientClipperState(
            update=0,
            norm_emas={
                name: torch.tensor(float("inf"), dtype=torch.float32)
                for name in self.parameter_names
            },
        )

    def validate_state_dict(
        self,
        state: Mapping[str, object] | None,
    ) -> AdaGradientClipperState:
        """Parse checkpoint state without mutating the live clipping history."""

        if not isinstance(state, Mapping):
            raise CheckpointError("AdaGC checkpoint is missing clipper state")
        expected_keys = {
            "algorithm_version",
            "update",
            "parameter_names",
            "norm_emas",
        }
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            extra = sorted(set(state) - expected_keys)
            raise CheckpointError(
                f"AdaGC clipper state keys are malformed; missing={missing}, extra={extra}"
            )
        algorithm_version = state["algorithm_version"]
        if type(algorithm_version) is not int or algorithm_version != self.ALGORITHM_VERSION:
            raise CheckpointError(
                "unsupported AdaGC clipper algorithm version "
                f"{algorithm_version!r}; expected {self.ALGORITHM_VERSION}"
            )
        update = state["update"]
        if type(update) is not int or update < 0:
            raise CheckpointError("AdaGC clipper update must be a nonnegative integer")
        names = state["parameter_names"]
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise CheckpointError("AdaGC clipper parameter names must be an ordered string list")
        if tuple(names) != self.parameter_names:
            raise CheckpointError(
                "AdaGC clipper parameter names do not match the active model"
            )
        norms = state["norm_emas"]
        if not isinstance(norms, Mapping) or set(norms) != set(self.parameter_names):
            raise CheckpointError(
                "AdaGC clipper norm state must contain every active parameter name exactly once"
            )
        restored: dict[str, Tensor] = {}
        for name in self.parameter_names:
            value = norms[name]
            if (
                not isinstance(value, Tensor)
                or value.shape != torch.Size([])
                or value.dtype != torch.float32
                or not (bool(torch.isfinite(value)) or bool(torch.isposinf(value)))
                or (bool(torch.isfinite(value)) and float(value) < 0.0)
            ):
                raise CheckpointError(
                    f"AdaGC norm for {name!r} must be a nonnegative float32 scalar"
                )
            # map_location may move every checkpoint tensor to CUDA. AdaGC
            # always reclaims ownership of its scalars on CPU.
            restored[name] = value.detach().cpu().clone()
        return AdaGradientClipperState(update=update, norm_emas=restored)

    def install_state(self, state: AdaGradientClipperState) -> None:
        """Install one already-validated checkpoint candidate atomically."""

        self.update = state.update
        self.norm_emas = {
            name: state.norm_emas[name].clone()
            for name in self.parameter_names
        }

    def load_state_dict(self, state: Mapping[str, object] | None) -> None:
        """Validate completely, then atomically install AdaGC checkpoint state."""

        self.install_state(self.validate_state_dict(state))


def build_gradient_clipper(
    model: nn.Module,
    *,
    method: str,
    clip_norm: float,
    adagc_beta: float = 0.99,
    adagc_relative_clip: float = 1.04,
    adagc_warmup_updates: int = 100,
) -> GradientClipper:
    """Build the configured clipping strategy over canonical model parameters."""

    if method == "global":
        return GlobalGradientClipper(model.parameters(), clip_norm=clip_norm)
    if method == "none":
        return NoGradientClipper(model.parameters())
    if method == "adagc":
        # named_parameters removes duplicate/tied objects by default. The first
        # traversal name is the stable canonical checkpoint key.
        return AdaGradientClipper(
            model.named_parameters(),
            clip_norm=clip_norm,
            beta=adagc_beta,
            relative_clip=adagc_relative_clip,
            warmup_updates=adagc_warmup_updates,
        )
    raise ValueError(f"unsupported gradient clipping method: {method}")


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


_BITSANDBYTES_VERSION_RANGE = "bitsandbytes>=0.49,<0.50"


def _bitsandbytes_optimizer_class(
    bitsandbytes: object,
    optimizer_name: str,
) -> tuple[type[torch.optim.Optimizer], str]:
    """Validate the optional package and return its selected optimizer class."""

    version = str(getattr(bitsandbytes, "__version__", "unknown"))
    supported = re.fullmatch(
        r"0\.49(?:\.\d+)?(?:\.post\d+)?(?:\+[A-Za-z0-9.-]+)?",
        version,
    )
    if supported is None:
        raise OptimizerSetupError(
            f"optimizer {optimizer_name} requires {_BITSANDBYTES_VERSION_RANGE}; "
            f"found bitsandbytes {version}"
        )

    optim_module = getattr(bitsandbytes, "optim", None)
    if optim_module is None:
        raise OptimizerSetupError(
            f"bitsandbytes {version} does not expose bitsandbytes.optim; "
            f"reinstall the optional dependency with pip install 'a2v2[bnb]'"
        )
    class_name = "Adam8bit" if optimizer_name == "adam8bit" else "AdamW8bit"
    optimizer_class = getattr(optim_module, class_name, None)
    if not callable(optimizer_class):
        raise OptimizerSetupError(
            f"bitsandbytes {version} does not expose bitsandbytes.optim.{class_name}; "
            f"reinstall a compatible {_BITSANDBYTES_VERSION_RANGE} build"
        )

    cextension = getattr(bitsandbytes, "cextension", None)
    native_library = getattr(cextension, "lib", None)
    if not bool(getattr(native_library, "compiled_with_cuda", False)):
        pytorch_cuda = torch.version.cuda or "unknown"
        raise OptimizerSetupError(
            f"bitsandbytes {version} did not load a CUDA native library for "
            f"PyTorch CUDA {pytorch_cuda}; install or build a compatible "
            "bitsandbytes binary. If another compatible CUDA toolkit is "
            "installed, set BNB_CUDA_VERSION to its numeric version (for "
            "example, 130 for CUDA 13.0) before starting A2V2"
        )
    return optimizer_class, version


def build_optimizer(
    model: nn.Module,
    *,
    name: str,
    learning_rate: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    min_8bit_size: int = 4096,
    device: torch.device | str | None = None,
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
    elif normalized_name in {"adam8bit", "adamw8bit"}:
        if isinstance(min_8bit_size, bool) or min_8bit_size <= 0:
            raise ValueError("min_8bit_size must be positive for 8-bit optimizers")
        selected_device = torch.device(device) if device is not None else next(
            (
                parameter.device
                for group in groups
                for parameter in group["params"]
            ),
            torch.device("cpu"),
        )
        if selected_device.type != "cuda":
            raise RuntimeError(
                f"optimizer {normalized_name} requires a CUDA device; "
                "use adam or adamw on CPU"
            )
        try:
            bitsandbytes = importlib.import_module("bitsandbytes")
        except (ImportError, OSError) as error:
            raise OptimizerSetupError(
                f"optimizer {normalized_name} requires bitsandbytes; "
                "install the optional dependency with pip install 'a2v2[bnb]'"
            ) from error
        optimizer_class, bitsandbytes_version = _bitsandbytes_optimizer_class(
            bitsandbytes,
            normalized_name,
        )
        try:
            optimizer = optimizer_class(
                groups,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                min_8bit_size=min_8bit_size,
            )
        except (CheckpointError, DistributedOptimizerStepError, OptimizerSetupError):
            raise
        except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise OptimizerSetupError(
                f"could not initialize {normalized_name} with bitsandbytes "
                f"{bitsandbytes_version}; check the installed CUDA, PyTorch, "
                "and bitsandbytes binary compatibility"
            ) from error
        setattr(
            optimizer,
            "_a2v2_bitsandbytes_version",
            bitsandbytes_version,
        )
        return optimizer
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


class CosineWeightDecayScheduler:
    """Cosine decay on the engine's successful-update clock."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        weight_decay_end: float | None,
        max_updates: int,
    ) -> None:
        if max_updates <= 0:
            raise ValueError("weight-decay schedule max_updates must be positive")
        self.optimizer = optimizer
        self.weight_decay_end = weight_decay_end
        self.max_updates = max_updates
        self.initial_weight_decays = tuple(
            float(group.get("weight_decay", 0.0))
            for group in optimizer.param_groups
        )
        for group, initial in zip(
            self.optimizer.param_groups,
            self.initial_weight_decays,
            strict=True,
        ):
            group["initial_weight_decay"] = initial
        self.last_update = 0
        self._set_weight_decay(0)

    def _decay_at_update(self, initial: float, update: int) -> float:
        """Calculate one group's decay while preserving exempt zero groups."""

        if initial == 0.0:
            return 0.0
        end = 0.0 if self.weight_decay_end is None else self.weight_decay_end
        position = min(update, self.max_updates) / self.max_updates
        return end + 0.5 * (initial - end) * (
            1 + math.cos(math.pi * position)
        )

    def _set_weight_decay(self, update: int) -> float:
        """Apply one update's decay to every optimizer parameter group."""

        live = 0.0
        for group, initial in zip(
            self.optimizer.param_groups,
            self.initial_weight_decays,
            strict=True,
        ):
            decay = self._decay_at_update(initial, update)
            group["initial_weight_decay"] = initial
            group["weight_decay"] = decay
            if initial != 0.0 and live == 0.0:
                live = decay
        return live

    def step_update(self, update: int) -> float:
        """Apply and record decay for a nonnegative absolute update."""

        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise ValueError("weight-decay scheduler update must be a nonnegative integer")
        live = self._set_weight_decay(update)
        self.last_update = update
        return live

    def state_dict(self) -> dict[str, int]:
        """Serialize the last applied successful-update index."""

        return {"last_update": self.last_update}

    def validate_state_dict(self, state: Mapping[str, object]) -> int:
        """Validate scheduler state without mutating optimizer groups."""

        if set(state) != {"last_update"}:
            raise CheckpointError(
                "weight-decay scheduler state must contain exactly last_update"
            )
        last_update = state["last_update"]
        if (
            isinstance(last_update, bool)
            or not isinstance(last_update, int)
            or last_update < 0
        ):
            raise CheckpointError(
                "weight-decay scheduler last_update must be a nonnegative integer"
            )
        return last_update

    def install_state(self, last_update: int) -> None:
        """Install one already-validated scheduler clock."""

        self.step_update(last_update)

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate completely, then install decay state atomically."""

        self.install_state(self.validate_state_dict(state))


def build_weight_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    weight_decay_end: float | None,
    max_updates: int,
) -> CosineWeightDecayScheduler | None:
    """Build opt-in decay scheduling without touching the legacy default."""

    if schedule == "constant":
        return None
    if schedule == "cosine":
        return CosineWeightDecayScheduler(
            optimizer,
            weight_decay_end=weight_decay_end,
            max_updates=max_updates,
        )
    raise ValueError(f"unsupported weight-decay schedule: {schedule}")


def _live_weight_decay(optimizer: torch.optim.Optimizer) -> float:
    """Return the first nonzero group's live decay, or zero when all are exempt."""

    for group in optimizer.param_groups:
        decay = float(group.get("weight_decay", 0.0))
        if decay != 0.0:
            return decay
    return 0.0


# =============================================================================
# PRETRAINING COLLAPSE DIAGNOSTICS AND FRAME-LEVEL METRICS
# =============================================================================

@torch.no_grad()
def pretraining_variance_diagnostics(
    predictions: Tensor,
    targets: Tensor,
) -> tuple[float, float]:
    """Return the archived ``pred_var`` and ``target_var`` diagnostics.

    The historical names say variance, while the returned values are the mean
    feature-wise sample standard deviation after adding the archived ``1e-6``
    stabilizer. Inputs may have any leading dimensions; their final dimension
    is the learned feature coordinate.
    """

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.ndim < 2 or predictions.shape[-1] == 0:
        raise ValueError("diagnostic tensors need a nonempty feature dimension")

    # Mathematics: flatten z from [...,D] to [N,D], then retain the three
    # sufficient statistics N, Σ_i z_i, and Σ_i z_i² for each coordinate.
    # Interpretation: diagnostic work detaches from autograd and uses FP32,
    # matching the archived code without retaining full activation histories.
    predictions = predictions.detach().reshape(-1, predictions.shape[-1]).float()
    targets = targets.detach().reshape(-1, targets.shape[-1]).float()
    count = predictions.new_tensor(float(predictions.shape[0]))
    statistics = torch.cat((
        count.view(1),
        predictions.sum(dim=0),
        predictions.square().sum(dim=0),
        targets.sum(dim=0),
        targets.square().sum(dim=0),
    ))

    # Mathematics: sums are linear, so all-reducing sufficient statistics
    # before evaluating variance gives the sample variance of the union of all
    # rank-local masked vectors. Averaging local variances would be incorrect.
    # Interpretation: every worker reports one diagnostic for the same global
    # microbatch even though each GPU observed a different audio shard.
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(statistics)

    feature_dim = predictions.shape[-1]
    count = statistics[0]
    if float(count) < 2:
        raise ValueError("variance diagnostics require at least two vectors")
    offset = 1
    pred_sum = statistics[offset: offset + feature_dim]
    offset += feature_dim
    pred_square_sum = statistics[offset: offset + feature_dim]
    offset += feature_dim
    target_sum = statistics[offset: offset + feature_dim]
    offset += feature_dim
    target_square_sum = statistics[offset: offset + feature_dim]

    def archived_value(total: Tensor, square_total: Tensor) -> float:
        """Evaluate the archived unbiased variance identity and square root."""

        # Mathematics: s² = Σz²/(N-1) - (Σz)²/[N(N-1)], followed by
        # D^{-1}Σ_d sqrt(s_d²+10^{-6}).
        # Interpretation: a value approaching zero warns that feature
        # coordinates are becoming constant across masked training examples.
        variance = (
            square_total / (count - 1)
            - total.square() / (count * (count - 1))
        )
        return float(torch.sqrt(variance + 1e-6).mean())

    return (
        archived_value(pred_sum, pred_square_sum),
        archived_value(target_sum, target_square_sum),
    )


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


def sequence_classification_metrics(
    scores: Tensor,
    targets: Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    """Measure flattened recording-class decisions for multilabel validation."""

    if scores.ndim != 2 or targets.shape != scores.shape:
        raise ValueError("sequence scores and targets must share [batch, classes] shape")
    counts = FrameCounts.from_predictions(scores >= threshold, targets >= 0.5)
    return {
        "sequence_precision": counts.precision,
        "sequence_recall": counts.recall,
        "sequence_f1": counts.f1,
        "sequence_accuracy": counts.accuracy,
        "sequence_average_precision": average_precision(scores, targets),
    }


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
    pred_var: float | None = None
    target_var: float | None = None
    weight_decay: float = 0.0


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
        gradient_clipper: GradientClipper | None = None,
        weight_decay_scheduler: CosineWeightDecayScheduler | None = None,
        use_amp: bool = False,
        amp_init_scale: float = 128.0,
        amp_min_scale: float = 0.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_norm = clip_norm
        self.device = device
        self.gradient_clipper = (
            gradient_clipper
            if gradient_clipper is not None
            else GlobalGradientClipper(model.parameters(), clip_norm=clip_norm)
        )
        self.weight_decay_scheduler = weight_decay_scheduler
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
        self._terminal_optimizer_failure: BaseException | None = None

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

        if self._terminal_optimizer_failure is not None:
            raise DistributedOptimizerStepError(
                _DISTRIBUTED_OPTIMIZER_FAILURE
            ) from self._terminal_optimizer_failure
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_sample_size = 0
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        pred_var_sum = 0.0
        target_var_sum = 0.0
        variance_microbatches = 0
        # Mathematics: if an update contains K microbatches, backpropagate each
        # summed loss before a single normalization and optimizer step.
        # Interpretation: gradient accumulation emulates a larger batch without
        # allocating all waveforms and activations at once.
        batches = list(microbatches)
        for microbatch in batches:
            amp_context = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_context:
                output = forward(microbatch)
            if not _all_ranks_true(bool(torch.isfinite(output.loss)), self.device):
                raise FloatingPointError(
                    f"non-finite loss on one or more ranks at update {self.update}"
                )
            predictions = getattr(output, "predictions", None)
            diagnostic_targets = getattr(output, "targets", None)
            if predictions is not None:
                if diagnostic_targets is None:
                    raise ValueError(
                        "a pretraining output with predictions must also provide targets"
                    )
                # Mathematics: the archived logger evaluates one global
                # standard-deviation scalar per microbatch, then its metrics
                # meter takes their arithmetic mean over an accumulated update.
                # Interpretation: diagnostics retain the original scale even
                # when update_freq groups several forwards into one optimizer
                # step, and their detached collective cannot alter gradients.
                pred_var, target_var = pretraining_variance_diagnostics(
                    predictions,
                    diagnostic_targets,
                )
                pred_var_sum += pred_var
                target_var_sum += target_var
                variance_microbatches += 1
            if self.scaler is None:
                output.loss.backward()
            else:
                self.scaler.scale(output.loss).backward()
            # Mathematics: accumulate L_sum=Σ_k L_k and N_sum=Σ_k N_k.
            # Interpretation: later normalization weights every frame token
            # equally even when microbatches contain different lengths.
            total_sample_size += int(output.sample_size)
            total_loss = total_loss + output.loss.detach().to(dtype=torch.float64)
            self.batch_in_epoch += 1

        totals = torch.stack((
            torch.tensor(
                float(total_sample_size),
                device=self.device,
                dtype=torch.float64,
            ),
            total_loss,
        ))
        world_size = 1
        # Mathematics: all-reduce sums [N_r,L_r] over ranks r=0,...,W-1.
        # Interpretation: logging and normalization describe the global batch,
        # not the local shard seen by one GPU.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(totals)
        global_sample_size = max(float(totals[0].item()), 1.0)
        pred_var = (
            pred_var_sum / variance_microbatches
            if variance_microbatches
            else None
        )
        target_var = (
            target_var_sum / variance_microbatches
            if variance_microbatches
            else None
        )
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
        # Mathematics: every strategy returns the pre-clipping global norm;
        # its state transition remains tentative until optimizer.step succeeds.
        # Interpretation: fixed clipping keeps the archived operation exactly,
        # while AdaGC cannot advance on an AMP-overflow attempt.
        clip_candidate = self.gradient_clipper.clip()
        gradient_norm = clip_candidate.gradient_norm
        gradients_finite = _all_ranks_true(
            clip_candidate.finite and bool(torch.isfinite(gradient_norm)),
            self.device,
        )
        if not gradients_finite:
            if self.scaler is None:
                raise FloatingPointError(
                    f"non-finite gradients on one or more ranks at update {self.update}"
                )
            # A rank-local overflow must skip every optimizer. Apply the same
            # public GradScaler backoff and reset its successful-growth clock
            # without calling any rank's optimizer.
            new_scale = self.scaler.get_scale() * self.scaler.get_backoff_factor()
            self.scaler.update(new_scale=new_scale)
            scaler_state = self.scaler.state_dict()
            scaler_state["_growth_tracker"] = 0
            self.scaler.load_state_dict(scaler_state)
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
                pred_var=pred_var,
                target_var=target_var,
                weight_decay=_live_weight_decay(self.optimizer),
            )
        local_step_error: Exception | None = None
        try:
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
        except Exception as error:
            local_step_error = error
        if dist.is_available() and dist.is_initialized():
            all_steps_succeeded = _all_ranks_true(
                local_step_error is None,
                self.device,
            )
            if not all_steps_succeeded:
                terminal_error = DistributedOptimizerStepError(
                    _DISTRIBUTED_OPTIMIZER_FAILURE
                )
                self._terminal_optimizer_failure = local_step_error or terminal_error
                raise terminal_error from local_step_error
        elif local_step_error is not None:
            raise local_step_error
        if self.scaler is not None:
            self.scaler.update()
            if self.scaler.get_scale() < self.amp_min_scale:
                raise FloatingPointError(
                    f"AMP loss scale {self.scaler.get_scale()} is below minimum "
                    f"{self.amp_min_scale}"
                )
        # Mathematics: optimizer state transitions first, then update index
        # u<-u+1, learning rate lr(u), and teacher EMA at the same u.
        # Interpretation: scheduler and teacher clocks advance only after a
        # successful parameter update; AMP-skipped attempts leave them fixed.
        self.gradient_clipper.commit(clip_candidate)
        self.update += 1
        learning_rate = self.scheduler.step_update(self.update)
        weight_decay = (
            self.weight_decay_scheduler.step_update(self.update)
            if self.weight_decay_scheduler is not None
            else _live_weight_decay(self.optimizer)
        )
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
            pred_var=pred_var,
            target_var=target_var,
            weight_decay=weight_decay,
        )

    def checkpoint_payload(
        self,
        *,
        stage: str,
        config: Mapping[str, object],
    ) -> dict[str, object]:
        """Create a complete, versioned, exactly resumable checkpoint payload."""

        if self._terminal_optimizer_failure is not None:
            raise DistributedOptimizerStepError(
                _DISTRIBUTED_OPTIMIZER_FAILURE
            ) from self._terminal_optimizer_failure
        raw_model = self.unwrapped_model
        teacher = getattr(raw_model, "teacher", None)
        # Mathematics: this payload captures all state variables needed to make
        # the next transition F(state,batch) identical after a restart.
        # Interpretation: model weights alone support inference, while optimizer,
        # counters, sampler cursor, scaler, RNGs, and best metric support resume.
        return {
            "format_version": FORMAT_VERSION,
            "stage": stage,
            "config": dict(config),
            "model": raw_model.state_dict(),
            "teacher": teacher.model.state_dict() if teacher is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "gradient_clipper": self.gradient_clipper.state_dict(),
            "weight_decay_scheduler": (
                self.weight_decay_scheduler.state_dict()
                if self.weight_decay_scheduler is not None
                else None
            ),
            # Task 9 fills the remaining reserved v2 runtime slot.
            "topology": None,
            "resume_compatibility": (
                resume_compatibility_fingerprint(config["active"])
                if isinstance(config.get("active"), Mapping)
                else None
            ),
            "update": self.update,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "rng_state": capture_rng_state(),
            "sampler_state": self.sampler_state,
            "best_metric": self.best_metric,
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore model, optimizer, scheduler, counters, scaler, and RNGs."""

        checkpoint_update = int(checkpoint["update"])
        clipper_state = checkpoint.get("gradient_clipper")
        weight_decay_state = checkpoint.get("weight_decay_scheduler")
        adagc_candidate: AdaGradientClipperState | None = None
        weight_decay_candidate: int | None = None
        if isinstance(self.gradient_clipper, AdaGradientClipper):
            if clipper_state is None:
                if checkpoint_update > 0:
                    raise CheckpointError(
                        "AdaGC cannot resume at update "
                        f"{checkpoint_update} without clipper state"
                    )
                adagc_candidate = self.gradient_clipper.pristine_state()
            else:
                adagc_candidate = self.gradient_clipper.validate_state_dict(clipper_state)
                if adagc_candidate.update != checkpoint_update:
                    raise CheckpointError(
                        "AdaGC clipper update "
                        f"{adagc_candidate.update} does not match checkpoint update "
                        f"{checkpoint_update}"
                    )
        else:
            self.gradient_clipper.load_state_dict(clipper_state)
        if self.weight_decay_scheduler is None:
            if weight_decay_state is not None:
                raise CheckpointError(
                    "constant weight-decay schedule cannot restore scheduler state"
                )
        elif weight_decay_state is None:
            if checkpoint_update > 0:
                raise CheckpointError(
                    "weight-decay scheduler cannot resume at update "
                    f"{checkpoint_update} without state"
                )
            weight_decay_candidate = 0
        elif not isinstance(weight_decay_state, Mapping):
            raise CheckpointError("weight-decay scheduler state must be a mapping")
        else:
            weight_decay_candidate = self.weight_decay_scheduler.validate_state_dict(
                weight_decay_state
            )
            if weight_decay_candidate != checkpoint_update:
                raise CheckpointError(
                    "weight-decay scheduler update "
                    f"{weight_decay_candidate} does not match checkpoint update "
                    f"{checkpoint_update}"
                )
        # Mathematics: restore every mutable component before the next batch;
        # strict model loading enforces a bijection between saved and live keys.
        # Interpretation: a checkpoint created by a different architecture or
        # incomplete implementation fails instead of training with mixed state.
        self.unwrapped_model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler is not None and checkpoint["scaler"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.update = checkpoint_update
        self.epoch = int(checkpoint["epoch"])
        self.batch_in_epoch = int(checkpoint["batch_in_epoch"])
        self.sampler_state = checkpoint["sampler_state"]
        self.best_metric = checkpoint["best_metric"]
        restore_rng_state(checkpoint["rng_state"])
        if adagc_candidate is not None:
            self.gradient_clipper.install_state(adagc_candidate)
        if weight_decay_candidate is not None:
            self.weight_decay_scheduler.install_state(weight_decay_candidate)
