"""Load each checked-in experiment recipe through the strict native translator. These tests
detect unsupported fields and protect the small CPU profiles used for workflow smoke
tests."""

from pathlib import Path

import pytest

from a2v2.data import conv_output_length
from a2v2.config import load_config


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("path", sorted((ROOT / "configs/MeerKAT").glob("*.yaml")) + sorted((ROOT / "configs/hyenas").glob("*.yaml")))
def test_every_published_recipe_loads(path: Path) -> None:
    """Check every published recipe loads."""
    config = load_config(path)
    assert config.task.sample_rate in {8000, 24000}
    assert config.model.average_top_k_layers in {12, 16}
    one_second_frames = conv_output_length(config.task.sample_rate, config.task.conv_feature_layers)
    assert one_second_frames in {200, 201}
    if config.stage == "pretrain":
        assert config.model.clone_batch in {8, 12}
        assert config.model.depth in {12, 16}
    else:
        assert config.model.freeze_finetune_updates == 10_000


def test_cpu_smoke_profiles_load_with_tiny_architecture() -> None:
    """Check CPU smoke profiles load with tiny architecture."""
    pretrain = load_config(ROOT / "configs/cpu_smoke_pretraining.yaml")
    finetune = load_config(ROOT / "configs/cpu_smoke_finetuning.yaml")
    assert pretrain.model.embed_dim == 16
    assert pretrain.model.depth == 2
    assert pretrain.model.clone_batch == 2
    assert finetune.stage == "finetune"
    assert finetune.model.average_top_k_layers == 2

