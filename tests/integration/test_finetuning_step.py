"""Run a complete tiny fine-tuning update around the encoder freeze boundary. The
assertions connect the configured update number to which backbone parameters receive
gradients."""

from pathlib import Path

import torch

from a2v2.config import load_config
from a2v2.model import Animal2VecFineTuningModel


ROOT = Path(__file__).parents[2]


def test_finetuning_freezes_then_unfreezes_backbone() -> None:
    """Check finetuning freezes then unfreezes backbone."""
    torch.manual_seed(18)
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain).train()
    waveform = torch.randn(2, 64)
    target = torch.randint(0, 2, (2, 16, 2)).float()

    frozen = model(waveform, target=target, update=0, sample_ids=torch.tensor([1, 2]))
    (frozen.loss / frozen.sample_size).backward()
    assert model.classifier.weight.grad is not None
    assert not any(parameter.grad is not None for parameter in model.encoder.parameters())

    model.zero_grad(set_to_none=True)
    unfrozen = model(waveform, target=target, update=1, sample_ids=torch.tensor([1, 2]))
    (unfrozen.loss / unfrozen.sample_size).backward()
    assert any(parameter.grad is not None for parameter in model.encoder.transformer.parameters())
    assert not any(parameter.grad is not None for parameter in model.encoder.local_encoder.parameters())
    assert torch.isfinite(unfrozen.loss)
