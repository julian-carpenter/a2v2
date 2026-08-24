"""Test safe translation of published YAML recipes into native dataclasses. The suite
covers restricted convolution expressions, defaults, overrides, AMP values, DDP
settings, and unknown fields."""

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

import a2v2.config as config_module
from a2v2.config import (
    ConfigError,
    config_from_dict,
    config_from_serialized_dict,
    config_to_dict,
    load_config,
    parse_conv_feature_layers,
)
from a2v2.workflows import _ddp_bucket_cap_mb_list, _ddp_find_unused_parameters


ROOT = Path(__file__).parents[2]


def test_modern_example_configs_select_compatible_architecture_and_policies() -> None:
    """Load the paired modern recipes with one checkpoint-compatible encoder."""

    pretrain = load_config(ROOT / "configs/modern/rope_cls_geglu_pretrain.yaml")
    finetune = load_config(ROOT / "configs/modern/rope_cls_geglu_finetune.yaml")

    assert pretrain.stage == "pretrain"
    assert finetune.stage == "finetune"
    for config in (pretrain, finetune):
        assert config_module.resolve_position_encoding(config.model) == "rope"
        assert config_module.resolve_attention_backend(config.model) == "flash"
        assert config.model.use_cls_token is True
        assert config.model.ffn_type == "geglu"
        assert config.model.initialization == "deepscale_lm"
        assert config.model.checkpoint_activations is True
        assert config.common.torch_compile is True
        assert config.common.torch_compile_fullgraph is False
        assert config.optimization.gradient_clip_method == "adagc"
        assert config.optimization.adagc_beta == 0.99
        assert config.optimization.adagc_relative_clip == 1.04
        assert config.optimization.adagc_warmup_updates == 100
        assert config.optimizer.name == "adamw8bit"
        assert config.optimizer.min_8bit_size == 4096
        assert config.optimizer.weight_decay_schedule == "cosine"
        assert config.optimizer.weight_decay == 0.01
        assert config.optimizer.weight_decay_end == 0.0

    assert pretrain.model.classification_head == "frame"
    assert pretrain.model.cls_loss_weight == 1.0
    assert finetune.model.classification_head == "cls"
    assert (
        pretrain.model.depth,
        pretrain.model.embed_dim,
        pretrain.model.num_heads,
        pretrain.model.audio.prenet_depth,
        pretrain.model.position_encoding,
        pretrain.model.attention_backend,
        pretrain.model.rope_theta,
        pretrain.model.use_cls_token,
        pretrain.model.ffn_type,
        pretrain.model.initialization,
        pretrain.task.sample_rate,
        pretrain.task.normalize,
        pretrain.task.conv_feature_layers,
    ) == (
        finetune.model.depth,
        finetune.model.embed_dim,
        finetune.model.num_heads,
        finetune.model.audio.prenet_depth,
        finetune.model.position_encoding,
        finetune.model.attention_backend,
        finetune.model.rope_theta,
        finetune.model.use_cls_token,
        finetune.model.ffn_type,
        finetune.model.initialization,
        finetune.task.sample_rate,
        finetune.task.normalize,
        finetune.task.conv_feature_layers,
    )


def test_published_recipes_and_local_reproduction_driver_keep_frozen_hashes() -> None:
    """Reject edits to the published controls while modern examples evolve."""

    expected = {
        "configs/MeerKAT/a2v_large_pretrain_best.yaml": "c5eb23d979cd12704f0dd3977fac1b6031cb4cbf7d5868930eaa6c2816f36a29",
        "configs/MeerKAT/finetune_mixup_001.yaml": "384eed9a25da913258e618425c9526cffdaccdacc2a16f912f8e1719fe15a9e0",
        "configs/MeerKAT/finetune_mixup_025.yaml": "31daab6409907238af025a480bf236fd2bb82fdd9442a9a1e4d3b0552e0d3f4a",
        "configs/MeerKAT/finetune_mixup_100.yaml": "c31c2a3df37d2397ae2cd7cbc502aa849bd819af5ac35f9ff930ab35fb192fd0",
        "configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml": "6a40999452b56e95834d2487860ddc11974c9840b9814eafec1f21468de09102",
        "configs/hyenas/finetune_mixup_100.yaml": "f892d0f9bb52a823a1fe4bcd168aab33a29cbd6bcc395ce0822525b6bc3c07ca",
        "scripts/reproduce_meerkat_paper.sh": "485b4b36fe1045174a392834bd9ee9848538ab496944631a0b2d514e22d4de18",
    }

    observed = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in expected
    }
    assert observed == expected


def test_modern_options_default_to_legacy_and_round_trip() -> None:
    """Keep checked-in recipes on their frozen legacy configuration path."""

    recipe_paths = sorted((ROOT / "configs/MeerKAT").glob("*.yaml")) + sorted(
        (ROOT / "configs/hyenas").glob("*.yaml")
    )
    assert recipe_paths

    for path in recipe_paths:
        config = load_config(path)
        restored = config_from_serialized_dict(config_to_dict(config))

        assert config.common.torch_compile is False
        assert config.common.torch_compile_backend == "inductor"
        assert config.common.torch_compile_mode == "default"
        assert config.common.torch_compile_fullgraph is False
        assert config.common.torch_compile_dynamic is None
        assert config.model.position_encoding == "legacy"
        assert config.model.attention_backend == "legacy"
        assert config.model.rope_theta == 10_000.0
        assert config.model.use_cls_token is False
        assert config.model.cls_loss_weight == 1.0
        assert config.model.classification_head == "frame"
        assert config.model.ffn_type == "mlp"
        assert config.model.initialization == "legacy"
        assert config.optimization.gradient_clip_method == "global"
        assert config.optimization.adagc_beta == 0.99
        assert config.optimization.adagc_relative_clip == 1.04
        assert config.optimization.adagc_warmup_updates == 100
        assert config.optimizer.min_8bit_size == 4096
        assert config.optimizer.weight_decay_schedule == "constant"
        assert config.optimizer.weight_decay_end is None
        assert restored == config

    modern = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=(
            "common.torch_compile=true",
            "common.torch_compile_backend=eager",
            "common.torch_compile_mode=reduce-overhead",
            "common.torch_compile_fullgraph=true",
            "common.torch_compile_dynamic=false",
            "model.position_encoding=rope",
            "model.attention_backend=legacy",
            "model.rope_theta=5000.0",
            "model.use_cls_token=true",
            "model.cls_loss_weight=0.5",
            "model.classification_head=cls",
            "model.ffn_type=geglu",
            "model.initialization=deepscale_lm",
            "optimization.gradient_clip_method=adagc",
            "optimization.adagc_beta=0.95",
            "optimization.adagc_relative_clip=1.2",
            "optimization.adagc_warmup_updates=10",
            "optimizer._name=adamw",
            "optimizer.min_8bit_size=1024",
            "optimizer.weight_decay_schedule=cosine",
            "optimizer.weight_decay_end=0.01",
        ),
    )

    assert modern.common.torch_compile is True
    assert modern.common.torch_compile_backend == "eager"
    assert modern.common.torch_compile_mode == "reduce-overhead"
    assert modern.common.torch_compile_fullgraph is True
    assert modern.common.torch_compile_dynamic is False
    assert config_module.resolve_position_encoding(modern.model) == "rope"
    assert config_module.resolve_attention_backend(modern.model) == "sdpa"
    assert modern.model.classification_head == "cls"
    assert modern.model.ffn_type == "geglu"
    assert modern.model.initialization == "deepscale_lm"
    assert modern.optimization.gradient_clip_method == "adagc"
    assert modern.optimizer.name == "adamw"
    assert modern.optimizer.weight_decay_schedule == "cosine"
    assert config_from_serialized_dict(config_to_dict(modern)) == modern


def test_legacy_serialized_config_supplies_modern_defaults() -> None:
    """Restore checkpoints written before modern strategy fields existed."""

    serialized = config_to_dict(load_config(ROOT / "tests/fixtures/tiny_finetune.yaml"))
    sections = {
        "common": (
            "torch_compile",
            "torch_compile_backend",
            "torch_compile_mode",
            "torch_compile_fullgraph",
            "torch_compile_dynamic",
        ),
        "model": (
            "position_encoding",
            "attention_backend",
            "rope_theta",
            "use_cls_token",
            "cls_loss_weight",
            "classification_head",
            "ffn_type",
            "initialization",
        ),
        "optimization": (
            "gradient_clip_method",
            "adagc_beta",
            "adagc_relative_clip",
            "adagc_warmup_updates",
        ),
        "optimizer": (
            "min_8bit_size",
            "weight_decay_schedule",
            "weight_decay_end",
        ),
    }
    for section, fields in sections.items():
        values = serialized[section]
        assert isinstance(values, dict)
        for field in fields:
            values.pop(field, None)

    restored = config_from_serialized_dict(serialized)

    assert restored.common.torch_compile is False
    assert restored.common.torch_compile_dynamic is None
    assert restored.model.position_encoding == "legacy"
    assert restored.model.attention_backend == "legacy"
    assert restored.model.use_cls_token is False
    assert restored.model.classification_head == "frame"
    assert restored.model.ffn_type == "mlp"
    assert restored.model.initialization == "legacy"
    assert restored.optimization.gradient_clip_method == "global"
    assert restored.optimizer.weight_decay_schedule == "constant"
    assert restored.optimizer.weight_decay_end is None


def test_modern_option_validation_rejects_incompatible_combinations() -> None:
    """Reject invalid strategy selections before runtime construction."""

    raw = yaml.safe_load((ROOT / "tests/fixtures/tiny_finetune.yaml").read_text())
    assert isinstance(raw, dict)
    base_raw = deepcopy(raw)
    model = raw["model"]
    assert isinstance(model, dict)
    model["position_encoding"] = "alibi"
    model["attention_backend"] = "sdpa"

    with pytest.raises(ConfigError, match=r"position_encoding=alibi.*attention_backend"):
        config_from_dict(raw)

    invalid_values = (
        ("common", "torch_compile", "false"),
        ("model", "position_encoding", "absolute"),
        ("model", "rope_theta", 0.0),
        ("model", "use_cls_token", 1),
        ("optimization", "adagc_beta", 1.0),
        ("optimization", "adagc_relative_clip", 0.0),
        ("optimization", "adagc_warmup_updates", -1),
        ("optimizer", "min_8bit_size", -1),
        ("optimizer", "weight_decay_schedule", "linear"),
    )
    for section, field, value in invalid_values:
        invalid = deepcopy(base_raw)
        values = invalid.setdefault(section, {})
        assert isinstance(values, dict)
        values[field] = value

        with pytest.raises(ConfigError, match=rf"{section}\.{field}"):
            config_from_dict(invalid)

    missing_cls_token = deepcopy(base_raw)
    cls_model = missing_cls_token["model"]
    assert isinstance(cls_model, dict)
    cls_model["position_encoding"] = "rope"
    cls_model["classification_head"] = "cls"
    with pytest.raises(ConfigError, match=r"classification_head=cls.*use_cls_token"):
        config_from_dict(missing_cls_token)


def test_checkpoint_activations_defaults_overrides_and_round_trips() -> None:
    """Keep activation checkpointing opt-in and checkpoint-compatible."""

    path = ROOT / "tests/fixtures/tiny_finetune.yaml"
    default = load_config(path)
    explicit_false = load_config(
        path,
        overrides=("model.checkpoint_activations=false",),
    )
    enabled = load_config(
        path,
        overrides=("model.checkpoint_activations=true",),
    )

    assert default.model.checkpoint_activations is False
    assert explicit_false.model.checkpoint_activations is False
    assert enabled.model.checkpoint_activations is True

    serialized = config_to_dict(enabled)
    serialized_model = serialized["model"]
    assert isinstance(serialized_model, dict)
    assert serialized_model["checkpoint_activations"] is True
    assert config_from_serialized_dict(serialized).model.checkpoint_activations is True

    legacy = deepcopy(serialized)
    legacy_model = legacy["model"]
    assert isinstance(legacy_model, dict)
    legacy_model.pop("checkpoint_activations")
    assert config_from_serialized_dict(legacy).model.checkpoint_activations is False


@pytest.mark.parametrize("invalid", (0, 1, [], [False], "false"))
def test_checkpoint_activations_rejects_invalid_recipe_values(
    invalid: object,
) -> None:
    """Reject non-boolean checkpoint policy values in published recipes."""

    path = ROOT / "tests/fixtures/tiny_finetune.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    model = raw["model"]
    assert isinstance(model, dict)
    model["checkpoint_activations"] = invalid

    with pytest.raises(ConfigError, match=r"model\.checkpoint_activations"):
        config_from_dict(raw)


@pytest.mark.parametrize(
    "override",
    (
        "model.checkpoint_activations=0",
        "model.checkpoint_activations=1",
        "model.checkpoint_activations=[]",
        "model.checkpoint_activations=[false]",
        "model.checkpoint_activations='false'",
        'model.checkpoint_activations="false"',
    ),
)
def test_checkpoint_activations_rejects_invalid_overrides(override: str) -> None:
    """Reject CLI overrides that do not decode to Python booleans."""

    with pytest.raises(ConfigError, match=r"model\.checkpoint_activations"):
        load_config(ROOT / "tests/fixtures/tiny_finetune.yaml", overrides=(override,))


@pytest.mark.parametrize("invalid", (0, 1, [], [False], "false"))
def test_checkpoint_activations_rejects_invalid_serialized_values(
    invalid: object,
) -> None:
    """Reject non-boolean checkpoint policy values in native snapshots."""

    serialized = config_to_dict(load_config(ROOT / "tests/fixtures/tiny_finetune.yaml"))
    model = serialized["model"]
    assert isinstance(model, dict)
    model["checkpoint_activations"] = invalid

    with pytest.raises(ConfigError, match=r"model\.checkpoint_activations"):
        config_from_serialized_dict(serialized)


def test_parses_legacy_conv_expression_without_eval() -> None:
    """Check parses legacy conv expression without eval."""
    value = "[(127, 63, 1)] + [(512, 3, 2)] * 3 + [(512, 2, 1)]"
    assert parse_conv_feature_layers(value) == (
        (127, 63, 1),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 1),
    )


def test_rejects_executable_conv_expression() -> None:
    """Check rejects executable conv expression."""
    with pytest.raises(ConfigError, match="convolution"):
        parse_conv_feature_layers("__import__('os').getcwd()")


def test_loads_meerkat_pretraining_recipe() -> None:
    """Check loads meerkat pretraining recipe."""
    cfg = load_config(ROOT / "configs/MeerKAT/a2v_large_pretrain_best.yaml")

    assert cfg.stage == "pretrain"
    assert cfg.task.sample_rate == 8000
    assert cfg.model.depth == 16
    assert cfg.model.embed_dim == 1024
    assert cfg.model.audio.prenet_depth == 8
    assert cfg.model.audio.decoder.decoder_dim == 768
    assert cfg.model.norm_eps == 1e-5
    assert isinstance(cfg.model.norm_eps, float)
    assert len(cfg.task.conv_feature_layers) == 8
    assert cfg.common.fp16 is torch.cuda.is_available()
    assert cfg.common.fp16_init_scale == 1.0
    assert cfg.common.min_loss_scale == 1e-6
    assert cfg.common.tensorboard_logdir == Path("tb")
    assert cfg.distributed.world_size == 1
    assert cfg.distributed.requested_world_size == 4


def test_loads_hyena_finetuning_recipe_and_overrides() -> None:
    """Check loads hyena finetuning recipe and overrides."""
    cfg = load_config(
        ROOT / "configs/hyenas/finetune_mixup_100.yaml",
        overrides=("model.depth=3", "task.data=/tmp/manifests"),
    )

    assert cfg.stage == "finetune"
    assert cfg.task.sample_rate == 24000
    assert cfg.task.data == Path("/tmp/manifests")
    assert cfg.model.depth == 3
    assert cfg.model.average_top_k_layers == 12
    assert cfg.model.mask_channel_prob == 0
    assert cfg.dataset.required_batch_size_multiple == 8
    assert cfg.optimizer.betas == (0.9, 0.98)
    assert cfg.optimization.learning_rate == 0.0001


def test_finetuning_recipe_uses_fairseq_amp_scale_default() -> None:
    """Check finetuning recipe uses fairseq AMP scale default."""
    cfg = load_config(ROOT / "configs/MeerKAT/finetune_mixup_100.yaml")

    assert cfg.common.fp16_init_scale == 128.0
    assert cfg.criterion.iou_threshold == 0.0


def test_ddp_uses_resume_stable_pretraining_reduction_settings() -> None:
    """Check DDP uses resume stable pretraining reduction settings."""
    pretrain = load_config(ROOT / "configs/MeerKAT/a2v_large_pretrain_best.yaml")
    finetune = load_config(ROOT / "configs/MeerKAT/finetune_mixup_100.yaml")

    assert _ddp_find_unused_parameters(pretrain)
    assert _ddp_find_unused_parameters(finetune)
    assert _ddp_bucket_cap_mb_list(pretrain) == [4096]
    assert _ddp_bucket_cap_mb_list(finetune) is None


def test_loads_tiny_native_recipe() -> None:
    """Check loads tiny native recipe."""
    cfg = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")

    assert cfg.common.seed == 7
    assert cfg.common.tensorboard_logdir == Path("tensorboard")
    assert cfg.task.conv_feature_layers == ((8, 7, 1), (16, 4, 2), (16, 4, 2))
    assert cfg.model.audio.decoder.decoder_layers == 2


def test_rejects_unknown_field(tmp_path: Path) -> None:
    """Check rejects unknown field."""
    config = tmp_path / "bad.yaml"
    config.write_text("task:\n  sample_rate: 8000\n  mystery: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"task\.mystery"):
        load_config(config)


def test_rejects_unknown_override() -> None:
    """Check rejects unknown override."""
    with pytest.raises(ConfigError, match=r"model\.not_a_field"):
        load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml", ("model.not_a_field=1",))
