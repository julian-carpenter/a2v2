"""Test safe translation of published YAML recipes into native dataclasses. The suite
covers restricted convolution expressions, defaults, overrides, AMP values, DDP
settings, and unknown fields."""

from pathlib import Path

import pytest
import torch

from a2v2.config import ConfigError, load_config, parse_conv_feature_layers
from a2v2.workflows import _ddp_bucket_cap_mb_list, _ddp_find_unused_parameters


ROOT = Path(__file__).parents[2]


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
    assert cfg.task.unique_labels[-1] == "focal"
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
