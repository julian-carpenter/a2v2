"""Test individual official-to-native checkpoint mapping rules. The suite covers encoder
names, model-state validation, and recovery of Fairseq's optimizer update counter."""

from pathlib import Path

import pytest
import torch

from a2v2.workflows import (
    CheckpointConversionError,
    legacy_num_updates,
    map_legacy_encoder_key,
)


@pytest.mark.parametrize(
    ("legacy", "native"),
    [
        (
            "modality_encoders.AUDIO.local_encoder.conv_layers.0.0.low_hz_",
            "local_encoder.conv_layers.0.conv.low_hz_",
        ),
        (
            "modality_encoders.AUDIO.local_encoder.conv_layers.2.2.1.weight",
            "local_encoder.conv_layers.2.norm.weight",
        ),
        (
            "modality_encoders.AUDIO.project_features.1.bias",
            "project_norm.bias",
        ),
        (
            "modality_encoders.AUDIO.project_features.2.weight",
            "project_features.weight",
        ),
        (
            "modality_encoders.AUDIO.relative_positional_encoder.3.0.weight",
            "positional_encoder.blocks.2.conv.weight",
        ),
        (
            "modality_encoders.AUDIO.context_encoder.blocks.1.attn.qkv.weight",
            "prenet.blocks.1.attn.qkv.weight",
        ),
        (
            "modality_encoders.AUDIO.context_encoder.norm.weight",
            "prenet.norm.weight",
        ),
        ("blocks.4.mlp.fc2.bias", "transformer.blocks.4.mlp.fc2.bias"),
    ],
)
def test_maps_official_encoder_keys(legacy: str, native: str) -> None:
    """Check maps official encoder keys."""
    assert map_legacy_encoder_key(legacy) == native


def test_rejects_non_tensor_model_state(tmp_path: Path) -> None:
    """Check rejects non tensor model state."""
    from a2v2.workflows import load_legacy_checkpoint

    path = tmp_path / "bad.pt"
    torch.save({"model": "invalid"}, path)
    with pytest.raises(CheckpointConversionError, match="model state"):
        load_legacy_checkpoint(path)


def test_reads_fairseq_update_from_optimizer_history_when_root_field_is_absent() -> None:
    """Check reads fairseq update from optimizer history when root field is absent."""
    checkpoint = {
        "optimizer_history": [
            {"optimizer_name": "FP16Optimizer", "num_updates": 408_402},
        ],
    }

    assert legacy_num_updates(checkpoint) == 408_402
