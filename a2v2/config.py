"""Typed configuration and parsing of Animal2Vec 1.0 baseline recipes.

This file translates published YAML values and dotted command-line
overrides into immutable dataclasses. Validation occurs before model or
dataset construction so a misspelled reproduction setting cannot be
silently ignored. Comments distinguish exact parsing rules from their
research meaning, especially for convolution geometry and schedules.
"""

from __future__ import annotations

import ast
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml


# =============================================================================
# ERRORS AND TYPED RECIPE SCHEMA
# =============================================================================

class ConfigError(ValueError):
    """Raised when a configuration cannot be translated safely."""


# Mathematics: each convolution layer is the ordered triple
# (output channels C_out, kernel width K, stride s), with positive integers.
# Interpretation: keeping geometry in a fixed tuple prevents a recipe parser
# from swapping values that determine both model shape and label timing.
ConvLayerSpec = tuple[int, int, int]


# Each group below mirrors one top-level section in the published YAML recipes.
# Keeping the groups separate makes the translation explicit without retaining
# Fairseq's Hydra/OmegaConf object graph.

@dataclass(frozen=True)
class CommonConfig:
    """Process-wide precision, reproducibility, and logging settings."""

    fp16: bool = False
    fp16_init_scale: float = 128.0
    min_loss_scale: float = 0.0
    seed: int = 1
    log_format: str = "json"
    log_interval: int = 100
    tensorboard_logdir: Path = Path("tensorboard")


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint cadence and retention settings."""

    save_dir: Path = Path("checkpoints")
    save_interval: int = 1
    save_interval_updates: int = 0
    keep_last_epochs: int = 1
    best_checkpoint_metric: str = "loss"


@dataclass(frozen=True)
class TaskConfig:
    """Audio and label metadata shared by datasets and models."""

    name: str = "audio_ccas"
    data: Path = Path(".")
    sample_rate: int = 16_000
    normalize: bool = True
    with_labels: bool = False
    unique_labels: tuple[str, ...] = ()
    conv_feature_layers: tuple[ConvLayerSpec, ...] = ()
    min_sample_size: int = 1
    max_sample_size: int | None = None
    min_label_size: int = 0
    enable_padding: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    """DataLoader, batching, and validation cadence settings."""

    num_workers: int = 0
    max_tokens: int = 1_000_000
    train_subset: str = "train"
    valid_subset: str = "valid"
    validate_interval: int = 1
    validate_interval_updates: int = 0
    validate_after_updates: int = 0
    disable_validation: bool = False
    required_batch_size_multiple: int = 8


@dataclass(frozen=True)
class DistributedConfig:
    """Distributed launch settings after environment-variable resolution."""

    world_size: int = 1
    requested_world_size: int = 1
    backend: str = "nccl"


@dataclass(frozen=True)
class CriterionConfig:
    """Fine-tuning loss and event-evaluation settings."""

    name: str = "expanded_model"
    use_focal_loss: bool = True
    label_smoothing: float = 0.0
    metric_threshold: float = 0.5
    iou_threshold: float = 0.0
    event_method: str = "avg"
    sigma_s: float = 0.1
    maxfilt_s: float = 0.1
    max_duration_s: float = 0.5
    low_probability: float = 0.125
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass(frozen=True)
class OptimizationConfig:
    """Update accumulation, clipping, and base learning-rate settings."""

    update_freq: tuple[int, ...] = (1,)
    max_update: int = 0
    clip_norm: float = 0.0
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class OptimizerConfig:
    """Parameters needed to construct the native PyTorch optimizer."""

    name: str = "adam"
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    """Warmup and cosine-decay scheduler settings."""

    name: str = "cosine"
    warmup_updates: int = 0
    warmup_init_lr: float = 0.0
    min_lr: float = 0.0


@dataclass(frozen=True)
class DecoderConfig:
    """Convolutional reconstruction decoder used during pretraining."""

    input_dropout: float = 0.0
    decoder_dim: int = 384
    decoder_groups: int = 16
    decoder_kernel: int = 7
    decoder_layers: int = 4


@dataclass(frozen=True)
class AudioModelConfig:
    """Audio-frontend, masking, ALiBi, and decoder settings."""

    sinc_input: bool = True
    apply_window_to_root: bool = False
    use_pswish: bool = True
    sinc_norm: str = "layer_norm"
    conv_pos_depth: int = 5
    conv_pos_width: int = 95
    conv_pos_groups: int = 16
    prenet_depth: int = 0
    mask_prob: float = 0.65
    mask_length: int = 10
    mask_prob_adjust: float = 0.0
    inverse_mask: bool = False
    mask_noise_std: float = 0.0
    mask_dropout: float = 0.0
    add_masks: bool = False
    ema_local_encoder: bool = False
    use_alibi_encoder: bool = True
    prenet_layerdrop: float = 0.0
    prenet_dropout: float = 0.0
    learned_alibi_scale: bool = True
    learned_alibi_scale_per_head: bool = True
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass(frozen=True)
class ModelConfig:
    """Shared encoder plus stage-specific model settings."""

    name: str = "data2vec_multi"
    depth: int = 12
    embed_dim: int = 768
    num_heads: int = 12
    clone_batch: int = 1
    ema_decay: float = 0.9997
    ema_end_decay: float = 1.0
    ema_anneal_end_step: int = 300_000
    ema_encoder_only: bool = False
    average_top_k_layers: int = 12
    instance_norm_target_layer: bool = True
    layer_norm_target_layer: bool = False
    layer_norm_targets: bool = False
    layerdrop: float = 0.0
    norm_eps: float = 1e-5
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    mlp_ratio: float = 4.0
    layer_norm_first: bool = False
    end_of_block_targets: bool = False
    start_drop_path_rate: float = 0.0
    end_drop_path_rate: float = 0.0
    loss_beta: float = 0.0
    loss_scale: float | None = None
    target_mixup: bool = False
    source_mixup: float = 0.0
    mixup_prob: float = 0.0
    same_mixup: bool = True
    mixing_window_length: float = 0.05
    gain_mode: str = "A_weighting"
    w2v_path: str | None = None
    freeze_finetune_updates: int = 0
    feature_grad_mult: float = 0.0
    apply_mask: bool = False
    mask_prob: float = 0.0
    mask_length: int = 10
    mask_channel_prob: float = 0.0
    mask_channel_length: int = 10
    dropout: float = 0.0
    dropout_input: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    final_dropout: float = 0.0
    drop_path: float = 0.0
    load_pretrain_weights: bool = True
    checkpoint_activations: bool = False
    audio: AudioModelConfig = field(default_factory=AudioModelConfig)


@dataclass(frozen=True)
class Animal2VecConfig:
    """Complete validated configuration consumed by the native runtime.

    ``stage`` is inferred as ``"pretrain"`` or ``"finetune"`` while loading
    the published recipes. All filesystem values are normalized to ``Path``
    objects and all sequence-valued settings to tuples.
    """

    common: CommonConfig
    checkpoint: CheckpointConfig
    task: TaskConfig
    dataset: DatasetConfig
    distributed: DistributedConfig
    criterion: CriterionConfig
    optimization: OptimizationConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    model: ModelConfig
    stage: str


# =============================================================================
# SAFE PARSING HELPERS
# =============================================================================

def _eval_conv_node(node: ast.AST) -> object:
    """Evaluate the restricted list arithmetic used for convolution recipes.

    Published recipes express the frontend as strings such as
    ``[(512, 10, 5)] + [(512, 3, 2)] * 4``. Supporting only integer constants,
    lists, tuples, addition, and repetition reproduces that syntax without
    executing arbitrary Python.
    """

    if isinstance(node, ast.Expression):
        return _eval_conv_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_eval_conv_node(value) for value in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # Mathematics: list addition forms the concatenation A || B.
        # Interpretation: official recipes build a heterogeneous prefix and
        # suffix without executing arbitrary Python expressions.
        left = _eval_conv_node(node.left)
        right = _eval_conv_node(node.right)
        if isinstance(left, list) and isinstance(right, list):
            return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        # Mathematics: list repetition A * n forms n ordered copies of A.
        # Interpretation: recipes can describe repeated convolution stages in
        # the same compact notation used by archived Fairseq YAML files.
        left = _eval_conv_node(node.left)
        right = _eval_conv_node(node.right)
        if isinstance(left, list) and isinstance(right, int):
            return left * right
        if isinstance(left, int) and isinstance(right, list):
            return right * left
    raise ConfigError("invalid convolution expression")


def parse_conv_feature_layers(value: object) -> tuple[ConvLayerSpec, ...]:
    """Parse a YAML list or the restricted arithmetic used by legacy recipes."""

    try:
        parsed = _eval_conv_node(ast.parse(value, mode="eval")) if isinstance(value, str) else value
    except (SyntaxError, ValueError) as exc:
        raise ConfigError("invalid convolution expression") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ConfigError("convolution layers must be a list")
    result: list[ConvLayerSpec] = []
    for index, layer in enumerate(parsed):
        if not isinstance(layer, (list, tuple)) or len(layer) != 3:
            raise ConfigError(f"convolution layer {index} must contain channels, kernel, stride")
        channels, kernel, stride = layer
        if not all(isinstance(item, int) and item > 0 for item in layer):
            raise ConfigError(f"convolution layer {index} values must be positive integers")
        # Mathematics: C, K, s ∈ positive integers is the domain required by
        # Conv1d and by the output-length recurrence in a2v2.data.
        # Interpretation: validation fails at configuration load rather than
        # much later during model allocation or target rasterization.
        result.append((channels, kernel, stride))
    return tuple(result)


def _literal_sequence(value: object, field_name: str) -> tuple[Any, ...]:
    """Normalize a YAML or string-encoded list/tuple into a tuple."""

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ConfigError(f"{field_name} must be a literal sequence") from exc
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{field_name} must be a sequence")
    return tuple(value)


def _check_keys(section: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    """Reject the first unsupported field in a recipe section."""

    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown configuration field {section}.{unknown[0]}")


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply one dotted CLI override to a mutable raw configuration tree."""

    if "=" not in override:
        raise ConfigError(f"override must have section.key=value form: {override}")
    dotted, encoded = override.split("=", 1)
    keys = dotted.split(".")
    if len(keys) < 2 or any(not key for key in keys):
        raise ConfigError(f"override must have section.key=value form: {override}")
    # Mathematics: keys k_0...k_n describe a path through a rooted mapping;
    # the assignment replaces raw[k_0]...[k_n] with YAML-decoded value v.
    # Interpretation: command-line experiments modify one recipe leaf while
    # keeping every other published value visible and unchanged.
    cursor: dict[str, Any] = raw
    for key in keys[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise ConfigError(f"cannot override nested field below {key}")
        cursor = child
    # Mathematics: safe_load maps scalar syntax to its typed value, so "false",
    # "3", and "[1,2]" do not remain truthy or numeric-looking strings.
    # Interpretation: overrides behave like edits to the YAML file itself.
    cursor[keys[-1]] = yaml.safe_load(encoded)


def _mapping(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Read an optional top-level mapping and return an independent dictionary."""

    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return dict(value)


# =============================================================================
# PUBLISHED RECIPE TRANSLATION
# =============================================================================

def config_from_dict(raw: Mapping[str, object]) -> Animal2VecConfig:
    """Translate and validate a published Fairseq-style recipe.

    Unsupported keys are rejected deliberately. Silent acceptance would make
    a recipe appear reproducible while dropping a behavior that Fairseq once
    supplied. A few obsolete logging-only keys are accepted and ignored.
    """

    top_allowed = {
        "common", "checkpoint", "task", "dataset", "distributed_training",
        "criterion", "optimization", "optimizer", "lr_scheduler", "model", "hydra",
    }
    _check_keys("root", raw, top_allowed)

    common_raw = _mapping(raw, "common")
    _check_keys("common", common_raw, {
        "fp16", "seed", "log_format", "log_interval", "tensorboard_logdir",
        "min_loss_scale", "fp16_no_flatten_grads", "fp16_init_scale", "all_gather_list_size",
    })
    # Mathematics: fp16_effective = fp16_requested ∧ CUDA_available.
    # Interpretation: CPU verification follows the same recipe without asking
    # PyTorch to execute unsupported half-precision training kernels.
    tensorboard_logdir = Path(common_raw.get("tensorboard_logdir", "tensorboard"))
    if not str(tensorboard_logdir):
        raise ConfigError("common.tensorboard_logdir must not be empty")
    common = CommonConfig(
        fp16=bool(common_raw.get("fp16", False) and torch.cuda.is_available()),
        fp16_init_scale=float(common_raw.get("fp16_init_scale", 128.0)),
        min_loss_scale=float(common_raw.get("min_loss_scale", 0.0)),
        seed=int(common_raw.get("seed", 1)),
        log_format=str(common_raw.get("log_format", "json")),
        log_interval=int(common_raw.get("log_interval", 100)),
        tensorboard_logdir=tensorboard_logdir,
    )

    checkpoint_raw = _mapping(raw, "checkpoint")
    _check_keys("checkpoint", checkpoint_raw, {
        "save_dir", "save_interval", "save_interval_updates", "keep_last_epochs",
        "best_checkpoint_metric",
    })
    checkpoint = CheckpointConfig(
        save_dir=Path(checkpoint_raw.get("save_dir", "checkpoints")),
        save_interval=int(checkpoint_raw.get("save_interval", 1)),
        save_interval_updates=int(checkpoint_raw.get("save_interval_updates", 0)),
        keep_last_epochs=int(checkpoint_raw.get("keep_last_epochs", 1)),
        best_checkpoint_metric=str(checkpoint_raw.get("best_checkpoint_metric", "loss")),
    )

    task_raw = _mapping(raw, "task")
    _check_keys("task", task_raw, {
        "_name", "data", "sample_rate", "normalize", "with_labels", "unique_labels",
        "conv_feature_layers", "min_sample_size", "max_sample_size", "min_label_size",
        "enable_padding", "verbose_tensorboard_logging",
    })
    # Mathematics: label index c is the position of its name in this immutable
    # tuple; convolution triples retain their exact ordered sequence.
    # Interpretation: classifier columns, rasterized targets, and output event
    # names share one stable indexing authority.
    labels = _literal_sequence(task_raw.get("unique_labels", ()), "task.unique_labels")
    conv_layers = parse_conv_feature_layers(task_raw.get("conv_feature_layers", ()))
    task = TaskConfig(
        name=str(task_raw.get("_name", "audio_ccas")),
        data=Path(task_raw.get("data", ".")),
        sample_rate=int(task_raw.get("sample_rate", 16_000)),
        normalize=bool(task_raw.get("normalize", True)),
        with_labels=bool(task_raw.get("with_labels", False)),
        unique_labels=tuple(str(label) for label in labels),
        conv_feature_layers=conv_layers,
        min_sample_size=int(task_raw.get("min_sample_size", 1)),
        max_sample_size=(int(task_raw["max_sample_size"]) if task_raw.get("max_sample_size") is not None else None),
        min_label_size=int(task_raw.get("min_label_size", 0)),
        enable_padding=bool(task_raw.get("enable_padding", False)),
    )

    dataset_raw = _mapping(raw, "dataset")
    _check_keys("dataset", dataset_raw, {
        "num_workers", "max_tokens", "train_subset", "valid_subset", "validate_interval",
        "validate_interval_updates", "validate_after_updates", "disable_validation",
        "required_batch_size_multiple", "skip_invalid_size_inputs_valid_test",
    })
    dataset = DatasetConfig(
        num_workers=int(dataset_raw.get("num_workers", 0)),
        max_tokens=int(dataset_raw.get("max_tokens", 1_000_000)),
        train_subset=str(dataset_raw.get("train_subset", "train")),
        valid_subset=str(dataset_raw.get("valid_subset", "valid")),
        validate_interval=int(dataset_raw.get("validate_interval", 1)),
        validate_interval_updates=int(dataset_raw.get("validate_interval_updates", 0)),
        validate_after_updates=int(dataset_raw.get("validate_after_updates", 0)),
        disable_validation=bool(dataset_raw.get("disable_validation", False)),
        required_batch_size_multiple=int(dataset_raw.get("required_batch_size_multiple", 8)),
    )

    distributed_raw = _mapping(raw, "distributed_training")
    _check_keys("distributed_training", distributed_raw, {"distributed_world_size", "ddp_backend"})
    # Mathematics: requested_world_size belongs to the recipe, whereas the
    # effective world size W comes from torchrun's WORLD_SIZE environment.
    # Interpretation: retaining both values lets orchestration reject an
    # accidental one-GPU launch of an eight-GPU reproduction recipe.
    requested_world_size = int(distributed_raw.get("distributed_world_size", 1))
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = DistributedConfig(
        world_size=launched_world_size,
        requested_world_size=requested_world_size,
        backend=str(distributed_raw.get("ddp_backend", "nccl")),
    )

    criterion_raw = _mapping(raw, "criterion")
    _check_keys("criterion", criterion_raw, {
        "_name", "use_focal_loss", "label_smoothing", "metric_threshold", "method",
        "iou_threshold", "sigma_s", "maxfilt_s", "max_duration_s", "lowP", "segmentation_metrics",
        "report_accuracy", "log_keys", "focal_alpha", "focal_gamma",
    })
    criterion = CriterionConfig(
        name=str(criterion_raw.get("_name", "expanded_model")),
        use_focal_loss=bool(criterion_raw.get("use_focal_loss", True)),
        label_smoothing=float(criterion_raw.get("label_smoothing", 0.0)),
        metric_threshold=float(criterion_raw.get("metric_threshold", 0.5)),
        iou_threshold=float(criterion_raw.get("iou_threshold", 0.0)),
        event_method=str(criterion_raw.get("method", "avg")),
        sigma_s=float(criterion_raw.get("sigma_s", 0.1)),
        maxfilt_s=float(criterion_raw.get("maxfilt_s", 0.1)),
        max_duration_s=float(criterion_raw.get("max_duration_s", 0.5)),
        low_probability=float(criterion_raw.get("lowP", 0.125)),
        focal_alpha=float(criterion_raw.get("focal_alpha", 0.25)),
        focal_gamma=float(criterion_raw.get("focal_gamma", 2.0)),
    )

    optimization_raw = _mapping(raw, "optimization")
    _check_keys("optimization", optimization_raw, {"update_freq", "max_update", "clip_norm", "lr"})
    # Mathematics: update_freq[e] is the number of microbatches accumulated
    # into one optimizer update during epoch e, with the final value reused.
    # Interpretation: scheduler and EMA time advance by optimizer updates,
    # while this tuple controls how much data contributes to each one.
    update_freq = tuple(int(value) for value in _literal_sequence(optimization_raw.get("update_freq", (1,)), "optimization.update_freq"))
    lr_values = _literal_sequence(optimization_raw["lr"], "optimization.lr") if "lr" in optimization_raw else ()

    optimizer_raw = _mapping(raw, "optimizer")
    _check_keys("optimizer", optimizer_raw, {
        "_name", "adam_betas", "adam_eps", "weight_decay", "dynamic_groups", "groups",
    })
    effective_optimizer = optimizer_raw
    embedded_scheduler: Mapping[str, Any] = {}
    embedded_lr: float | None = None
    if optimizer_raw.get("_name") == "composite":
        # Mathematics: the archived composite has one active parameter group
        # named default, whose optimizer, scheduler, and scalar lr define the
        # same update as a non-composite configuration.
        # Interpretation: flattening this wrapper removes Fairseq plumbing
        # without changing any numerical optimizer setting.
        groups = optimizer_raw.get("groups", {})
        if not isinstance(groups, dict) or not isinstance(groups.get("default"), dict):
            raise ConfigError("optimizer.groups.default must be a mapping")
        default_group = groups["default"]
        effective_optimizer = dict(default_group.get("optimizer", {}))
        embedded_scheduler = default_group.get("lr_scheduler", {})
        embedded_lr = float(default_group.get("lr_float", 1e-4))
    betas = _literal_sequence(effective_optimizer.get("adam_betas", (0.9, 0.98)), "optimizer.adam_betas")
    if len(betas) != 2:
        raise ConfigError("optimizer.adam_betas must contain two values")
    # Mathematics: lr uses the first explicit optimization.lr value, then the
    # composite group's lr_float, then 10^-4 as the schema default.
    # Interpretation: this precedence reproduces both recipe styles found in
    # the official configuration directory.
    learning_rate = float(lr_values[0]) if lr_values else (embedded_lr if embedded_lr is not None else 1e-4)
    optimization = OptimizationConfig(
        update_freq=update_freq,
        max_update=int(optimization_raw.get("max_update", 0)),
        clip_norm=float(optimization_raw.get("clip_norm", 0.0)),
        learning_rate=learning_rate,
    )
    optimizer = OptimizerConfig(
        name=str(effective_optimizer.get("_name", "adam")),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(effective_optimizer.get("adam_eps", 1e-8)),
        weight_decay=float(effective_optimizer.get("weight_decay", 0.0)),
    )

    scheduler_value = raw.get("lr_scheduler", {})
    scheduler_raw = dict(scheduler_value) if isinstance(scheduler_value, dict) else {}
    if embedded_scheduler:
        scheduler_raw = dict(embedded_scheduler)
    _check_keys("lr_scheduler", scheduler_raw, {"_name", "warmup_updates", "warmup_init_lr", "min_lr"})
    scheduler = SchedulerConfig(
        name=str(scheduler_raw.get("_name", "cosine")),
        warmup_updates=int(scheduler_raw.get("warmup_updates", 0)),
        warmup_init_lr=float(scheduler_raw.get("warmup_init_lr", 0.0)),
        min_lr=float(scheduler_raw.get("min_lr", 0.0)),
    )

    model_raw = _mapping(raw, "model")
    _check_keys("model", model_raw, {
        "_name", "loss_beta", "loss_scale", "depth", "embed_dim", "num_heads", "clone_batch",
        "ema_decay", "ema_end_decay", "ema_anneal_end_step", "ema_encoder_only",
        "average_top_k_layers", "instance_norm_target_layer", "layer_norm_target_layer",
        "layer_norm_targets", "layerdrop", "norm_eps", "supported_modality", "target_mixup",
        "norm_affine", "encoder_dropout", "post_mlp_drop", "mlp_ratio", "layer_norm_first",
        "end_of_block_targets", "start_drop_path_rate", "end_drop_path_rate",
        "source_mixup", "mixup_prob", "same_mixup", "mixing_window_length", "gain_mode",
        "modalities", "w2v_path", "freeze_finetune_updates", "feature_grad_mult", "apply_mask",
        "mask_prob", "mask_length", "mask_channel_prob", "mask_channel_length", "dropout",
        "dropout_input", "activation_dropout", "attention_dropout", "final_dropout", "drop_path",
        "load_pretrain_weights", "checkpoint_activations",
    })
    modalities = model_raw.get("modalities", {})
    if modalities is None:
        modalities = {}
    if not isinstance(modalities, dict):
        raise ConfigError("model.modalities must be a mapping")
    _check_keys("model.modalities", modalities, {"audio"})
    audio_raw = dict(modalities.get("audio", {}))
    _check_keys("model.modalities.audio", audio_raw, {
        "sinc_input", "apply_window_to_root", "use_pswish", "sinc_norm", "conv_pos_depth",
        "conv_pos_width", "conv_pos_groups", "prenet_depth", "mask_prob", "mask_length",
        "mask_prob_adjust", "inverse_mask", "mask_noise_std", "mask_dropout", "add_masks",
        "ema_local_encoder", "use_alibi_encoder", "prenet_layerdrop", "prenet_dropout",
        "learned_alibi_scale", "learned_alibi_scale_per_head", "decoder",
    })
    decoder_raw = dict(audio_raw.get("decoder", {}))
    _check_keys("model.modalities.audio.decoder", decoder_raw, {
        "input_dropout", "decoder_dim", "decoder_groups", "decoder_kernel", "decoder_layers",
    })
    # Mathematics: each absent decoder coordinate takes its dataclass default;
    # present coordinates retain their YAML values and types.
    # Interpretation: a partial decoder block behaves like Fairseq config
    # composition without importing Hydra.
    decoder = DecoderConfig(**{
        key: decoder_raw.get(key, getattr(DecoderConfig(), key))
        for key in DecoderConfig.__dataclass_fields__
    })
    audio_kwargs = {
        key: audio_raw.get(key, getattr(AudioModelConfig(), key))
        for key in AudioModelConfig.__dataclass_fields__ if key != "decoder"
    }
    audio = AudioModelConfig(**audio_kwargs, decoder=decoder)
    # Mathematics: archived base recipes average K=12 layers and use depth 12,
    # while large recipes use K=16 and depth 16 when depth is omitted.
    # Interpretation: the target-layer count supplies the architecture default
    # that Hydra formerly inherited from a named model preset.
    average_layers = int(model_raw.get("average_top_k_layers", 12))
    inferred_depth = 16 if average_layers == 16 else 12
    inferred_embed = 1024 if inferred_depth == 16 else 768
    model_defaults = ModelConfig()
    model_kwargs = {
        key: model_raw.get(key, getattr(model_defaults, key))
        for key in ModelConfig.__dataclass_fields__ if key not in {"name", "audio", "depth", "embed_dim", "num_heads"}
    }
    model_kwargs["norm_eps"] = float(model_kwargs["norm_eps"])
    model = ModelConfig(
        name=str(model_raw.get("_name", "data2vec_multi")),
        depth=int(model_raw.get("depth", inferred_depth)),
        embed_dim=int(model_raw.get("embed_dim", inferred_embed)),
        num_heads=int(model_raw.get("num_heads", inferred_depth)),
        audio=audio,
        **model_kwargs,
    )
    # Mathematics: stage = finetune iff labels are requested or the criterion
    # name marks a fine-tuning criterion; all remaining recipes are pretraining.
    # Interpretation: one validated field chooses the model and workflow while
    # retaining compatibility with both official recipe conventions.
    stage = "finetune" if task.with_labels or "finetune" in criterion.name else "pretrain"
    return Animal2VecConfig(
        common=common,
        checkpoint=checkpoint,
        task=task,
        dataset=dataset,
        distributed=distributed,
        criterion=criterion,
        optimization=optimization,
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        stage=stage,
    )


# =============================================================================
# FILE AND CHECKPOINT SERIALIZATION
# =============================================================================

def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Animal2VecConfig:
    """Load a YAML recipe and apply ``section.key=value`` overrides."""

    path = Path(path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load config {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    raw = dict(loaded)
    # Mathematics: overrides compose left to right; a later assignment to the
    # same path replaces the earlier value.
    # Interpretation: command order remains visible and deterministic.
    for override in overrides:
        _apply_override(raw, override)
    return config_from_dict(raw)


def config_to_dict(config: Animal2VecConfig) -> dict[str, object]:
    """Return a checkpoint-safe dictionary containing no framework objects."""

    def normalize(value: object) -> object:
        """Recursively replace paths and tuples with checkpoint-safe values."""

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        return value

    # Mathematics: the recursion is structure-preserving except for the
    # isomorphisms Path -> str and tuple -> list.
    # Interpretation: torch.save receives plain portable containers instead of
    # Python objects tied to this module's import path.
    return normalize(asdict(config))  # type: ignore[return-value]


def config_from_serialized_dict(raw: Mapping[str, object]) -> Animal2VecConfig:
    """Restore the native dataclass representation stored in checkpoints."""

    try:
        common_values = dict(raw["common"])  # type: ignore[arg-type]
        if "tensorboard_logdir" in common_values:
            common_values["tensorboard_logdir"] = Path(common_values["tensorboard_logdir"])
        common = CommonConfig(**common_values)
        checkpoint_values = dict(raw["checkpoint"])  # type: ignore[arg-type]
        checkpoint_values["save_dir"] = Path(checkpoint_values["save_dir"])
        checkpoint = CheckpointConfig(**checkpoint_values)
        task_values = dict(raw["task"])  # type: ignore[arg-type]
        task_values["data"] = Path(task_values["data"])
        task_values["unique_labels"] = tuple(task_values["unique_labels"])
        # Mathematics: serialization maps nested tuples to lists; this inverse
        # restores ((C_0,K_0,s_0),...,(C_n,K_n,s_n)).
        # Interpretation: equality checks against live recipe geometry work
        # after a checkpoint round trip.
        task_values["conv_feature_layers"] = tuple(tuple(layer) for layer in task_values["conv_feature_layers"])
        task = TaskConfig(**task_values)
        dataset = DatasetConfig(**dict(raw["dataset"]))  # type: ignore[arg-type]
        distributed = DistributedConfig(**dict(raw["distributed"]))  # type: ignore[arg-type]
        criterion = CriterionConfig(**dict(raw["criterion"]))  # type: ignore[arg-type]
        optimization_values = dict(raw["optimization"])  # type: ignore[arg-type]
        optimization_values["update_freq"] = tuple(optimization_values["update_freq"])
        optimization = OptimizationConfig(**optimization_values)
        optimizer_values = dict(raw["optimizer"])  # type: ignore[arg-type]
        optimizer_values["betas"] = tuple(optimizer_values["betas"])
        optimizer = OptimizerConfig(**optimizer_values)
        scheduler = SchedulerConfig(**dict(raw["scheduler"]))  # type: ignore[arg-type]
        model_values = dict(raw["model"])  # type: ignore[arg-type]
        audio_values = dict(model_values.pop("audio"))
        decoder = DecoderConfig(**dict(audio_values.pop("decoder")))
        audio = AudioModelConfig(**audio_values, decoder=decoder)
        model = ModelConfig(**model_values, audio=audio)
        stage = str(raw["stage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid serialized native configuration: {exc}") from exc
    return Animal2VecConfig(
        common=common,
        checkpoint=checkpoint,
        task=task,
        dataset=dataset,
        distributed=distributed,
        criterion=criterion,
        optimization=optimization,
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        stage=stage,
    )
