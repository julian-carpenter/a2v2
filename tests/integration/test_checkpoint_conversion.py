"""Convert synthetic official-style checkpoints through complete native models. These tests
cover pretraining teacher state, fine-tuning wrappers, and strict shape rejection across
module boundaries."""

from pathlib import Path
import re

import torch

from a2v2.workflows import convert_checkpoint
from a2v2.config import load_config
from a2v2.model import Animal2VecFineTuningModel
from a2v2.model import Animal2VecPretrainingModel
from a2v2.training import load_checkpoint


ROOT = Path(__file__).parents[2]


def _legacy_encoder_key(native: str) -> str:
    """Translate a native encoder key back to its synthetic official name."""
    if native == "alibi_scale":
        return "modality_encoders.AUDIO.alibi_scale"
    match = re.match(r"local_encoder\.conv_layers\.(\d+)\.(conv|norm|activation)\.(.+)", native)
    if match:
        index, kind, suffix = match.groups()
        component = {"conv": "0", "norm": "2.1", "activation": "3"}[kind]
        return f"modality_encoders.AUDIO.local_encoder.conv_layers.{index}.{component}.{suffix}"
    if native.startswith("project_norm."):
        return "modality_encoders.AUDIO.project_features.1." + native.removeprefix("project_norm.")
    if native.startswith("project_features."):
        return "modality_encoders.AUDIO.project_features.2." + native.removeprefix("project_features.")
    match = re.match(r"positional_encoder\.blocks\.(\d+)\.conv\.(.+)", native)
    if match:
        index, suffix = match.groups()
        return f"modality_encoders.AUDIO.relative_positional_encoder.{int(index) + 1}.0.{suffix}"
    if native.startswith("prenet.blocks."):
        return "modality_encoders.AUDIO.context_encoder.blocks." + native.removeprefix("prenet.blocks.")
    if native.startswith("prenet.norm."):
        return "modality_encoders.AUDIO.context_encoder.norm." + native.removeprefix("prenet.norm.")
    if native.startswith("transformer.blocks."):
        return "blocks." + native.removeprefix("transformer.blocks.")
    raise AssertionError(f"test fixture cannot map {native}")


def _legacy_decoder_key(native: str) -> str:
    """Translate a native decoder key back to its synthetic official name."""
    match = re.match(r"blocks\.(\d+)\.conv\.(.+)", native)
    if match:
        index, suffix = match.groups()
        return f"modality_encoders.AUDIO.decoder.blocks.{index}.0.{suffix}"
    if native.startswith("proj."):
        return "modality_encoders.AUDIO.decoder." + native
    raise AssertionError(f"test fixture cannot map decoder {native}")


def test_converts_synthetic_pretraining_checkpoint_with_ema(tmp_path: Path) -> None:
    """Check converts synthetic pretraining checkpoint with EMA."""
    config_path = ROOT / "tests/fixtures/tiny_pretrain.yaml"
    config = load_config(config_path)
    native = Animal2VecPretrainingModel.from_config(config)
    legacy: dict[str, object] = {}
    ema: dict[str, torch.Tensor] = {}
    for key, tensor in native.student.state_dict().items():
        legacy[_legacy_encoder_key(key)] = tensor.clone()
    for key, tensor in native.decoder.state_dict().items():
        legacy[_legacy_decoder_key(key)] = tensor.clone()
    for key, tensor in native.teacher.model.state_dict().items():
        if key.startswith(("local_encoder.", "project_norm.", "project_features.")):
            continue
        ema[_legacy_encoder_key(key)] = tensor.clone()
    legacy["_ema"] = ema
    source = tmp_path / "legacy_pretrain.pt"
    destination = tmp_path / "native_pretrain.pt"
    torch.save({"model": legacy}, source)

    report = convert_checkpoint(source, destination, config_path=config_path, stage="pretrain")
    converted = load_checkpoint(destination)
    restored = Animal2VecPretrainingModel.from_config(config)
    restored.load_state_dict(converted["model"], strict=True)
    assert not report.missing
    assert report.mapped == len(restored.state_dict())


def test_converts_synthetic_finetuning_wrapper(tmp_path: Path) -> None:
    """Check converts synthetic finetuning wrapper."""
    pretrain_path = ROOT / "tests/fixtures/tiny_pretrain.yaml"
    finetune_path = ROOT / "tests/fixtures/tiny_finetune.yaml"
    pretrain = load_config(pretrain_path)
    finetune = load_config(finetune_path)
    native = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain)
    legacy: dict[str, torch.Tensor] = {}
    for key, tensor in native.encoder.state_dict().items():
        legacy["w2v_encoder.w2v_model." + _legacy_encoder_key(key)] = tensor.clone()
    legacy["w2v_encoder.proj.weight"] = native.classifier.weight.detach().clone()
    legacy["w2v_encoder.proj.bias"] = native.classifier.bias.detach().clone()
    source = tmp_path / "legacy_finetune.pt"
    destination = tmp_path / "native_finetune.pt"
    torch.save({"model": legacy}, source)

    report = convert_checkpoint(
        source,
        destination,
        config_path=finetune_path,
        pretrained_config_path=pretrain_path,
        stage="finetune",
    )
    converted = load_checkpoint(destination)
    restored = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain)
    restored.load_state_dict(converted["model"], strict=True)
    assert not report.missing
    assert report.mapped == len(restored.state_dict())
    native.eval()
    restored.eval()
    waveform = torch.randn(2, 64)
    assert torch.allclose(native(waveform, update=2).logits, restored(waveform, update=2).logits)


def test_conversion_fails_on_core_shape_mismatch(tmp_path: Path) -> None:
    """Check conversion fails on core shape mismatch."""
    config_path = ROOT / "tests/fixtures/tiny_pretrain.yaml"
    source = tmp_path / "bad_shape.pt"
    torch.save(
        {"model": {"modality_encoders.AUDIO.project_features.2.weight": torch.zeros(3, 3)}},
        source,
    )
    try:
        convert_checkpoint(source, tmp_path / "out.pt", config_path=config_path, stage="pretrain")
    except Exception as exc:
        assert "project_features.weight" in str(exc)
    else:
        raise AssertionError("shape mismatch was accepted")
