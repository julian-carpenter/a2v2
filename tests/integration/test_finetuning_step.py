"""Run a complete tiny fine-tuning update around the encoder freeze boundary. The
assertions connect the configured update number to which backbone parameters receive
gradients."""

from pathlib import Path

import torch

from a2v2.config import load_config
from a2v2.model import Animal2VecFineTuningModel
from a2v2.training import CosineUpdateScheduler, TrainingEngine


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


def test_finetuning_engine_omits_pretraining_variance_diagnostics() -> None:
    """Keep collapse diagnostics exclusive to masked mean-teacher training."""

    torch.manual_seed(20)
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-3,
        min_lr=0.0,
        warmup_updates=0,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
    )
    batch = {
        "source": torch.randn(2, 64),
        "target": torch.randint(0, 2, (2, 16, 2)).float(),
        "id": torch.tensor([1, 2]),
    }

    result = engine.step(
        [batch],
        lambda value: model(
            value["source"],
            target=value["target"],
            sample_ids=value["id"],
            update=engine.update,
        ),
    )

    assert result.pred_var is None
    assert result.target_var is None
