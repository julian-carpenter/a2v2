"""Protect checkpoint-visible model structure while source files move."""

from __future__ import annotations

import hashlib
from pathlib import Path

from torch import nn

import a2v2.config as config_module
from a2v2.config import load_config
from a2v2.model import Animal2VecFineTuningModel, Animal2VecPretrainingModel


ROOT = Path(__file__).parents[2]


def _state_signature(model: nn.Module) -> tuple[int, str]:
    """Hash checkpoint names, shapes, dtypes, and their serialized order."""

    rows = (
        f"{name}:{tuple(value.shape)}:{value.dtype}"
        for name, value in model.state_dict().items()
    )
    encoded = "\n".join(rows).encode()
    return len(model.state_dict()), hashlib.sha256(encoded).hexdigest()


def test_pretraining_checkpoint_structure_survives_source_reorganization() -> None:
    """Check pretraining checkpoint structure survives source reorganization."""
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    model = Animal2VecPretrainingModel.from_config(config)

    assert config_module.resolve_position_encoding(config.model) == "alibi"
    assert config_module.resolve_attention_backend(config.model) == "manual"
    assert _state_signature(model) == (
        120,
        "5efbec43fd1c1c392b8b4278cee6a21513f68f3095353bb22524d2ada2ecf6ad",
    )


def test_finetuning_checkpoint_structure_survives_source_reorganization() -> None:
    """Check finetuning checkpoint structure survives source reorganization."""
    pretraining = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    fine_tuning = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(
        fine_tuning,
        pretrained_config=pretraining,
    )

    assert config_module.resolve_position_encoding(fine_tuning.model) == "alibi"
    assert config_module.resolve_attention_backend(fine_tuning.model) == "manual"
    assert _state_signature(model) == (
        59,
        "b7b2ea2ad9017ec94d56516fbda49a932866b472a48633804619b7232f3d05bb",
    )
