"""Test focal loss, target mixup, layer averaging, and pretrained encoder validation. These
are the main equations and state boundaries specific to supervised adaptation."""

from pathlib import Path

import torch

from a2v2.config import load_config
from a2v2.model import Animal2VecFineTuningModel, AudioEncoder, mix_targets
from a2v2.model import SigmoidFocalLoss


ROOT = Path(__file__).parents[2]


def test_audio_encoder_propagates_activation_checkpointing_to_both_stacks() -> None:
    """Apply one config value to the prenet and main Transformer stacks."""

    config = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.checkpoint_activations=true",),
    )

    encoder = AudioEncoder.from_config(config)

    assert encoder.prenet.checkpoint_activations is True
    assert encoder.transformer.checkpoint_activations is True


def test_active_finetuning_config_controls_encoder_checkpointing() -> None:
    """Do not let an older pretrained config disable the active policy."""

    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=("model.checkpoint_activations=true",),
    )

    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )

    assert pretrain.model.checkpoint_activations is False
    assert model.encoder.prenet.checkpoint_activations is True
    assert model.encoder.transformer.checkpoint_activations is True


def test_sigmoid_focal_loss_matches_hand_calculation() -> None:
    """Check sigmoid focal loss matches hand calculation."""
    logits = torch.zeros(2)
    targets = torch.tensor([1.0, 0.0])
    loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="sum")(logits, targets)
    expected = -torch.log(torch.tensor(0.5)) * 0.25 * (0.25 + 0.75)
    assert torch.allclose(loss, expected)


def test_target_mixup_uses_unadjusted_sampling_ratio() -> None:
    """Check target mixup uses unadjusted sampling ratio."""
    targets = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    mixed = mix_targets(
        targets,
        ratios=torch.tensor([0.25]),
        permutation=torch.tensor([1, 0]),
        applied=torch.tensor([True, True]),
        same_ratio=True,
    )
    assert torch.allclose(mixed[0], torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(mixed[1], torch.tensor([[0.75, 0.25]]))


def test_target_mixup_accepts_one_ratio_per_selected_example() -> None:
    """Check target mixup accepts one ratio per selected example."""
    targets = torch.eye(4).unsqueeze(1)
    mixed = mix_targets(
        targets,
        ratios=torch.tensor([0.25, 0.75]),
        permutation=torch.tensor([1, 0, 3, 2]),
        applied=torch.tensor([True, False, True, False]),
        same_ratio=False,
    )
    assert torch.allclose(mixed[0, 0], torch.tensor([0.25, 0.75, 0.0, 0.0]))
    assert torch.allclose(mixed[2, 0], torch.tensor([0.0, 0.0, 0.75, 0.25]))
    assert torch.equal(mixed[1], targets[1])
    assert torch.equal(mixed[3], targets[3])


def test_finetuning_averages_requested_encoder_layers() -> None:
    """Check finetuning averages requested encoder layers."""
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain).eval()
    output = model(torch.randn(2, 64), update=2)
    expected_features = torch.stack(output.layer_outputs[-2:]).mean(dim=0)
    expected_logits = model.classifier(model.final_dropout(expected_features))
    assert torch.allclose(output.logits, expected_logits)
    assert output.logits.shape == (2, 16, 2)


def test_pretrained_encoder_state_rejects_incompatible_tensors() -> None:
    """Check pretrained encoder state rejects incompatible tensors."""
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain)
    state = model.encoder.state_dict()
    state["project_features.weight"] = torch.zeros(3, 3)
    try:
        model.load_pretrained_encoder(state)
    except ValueError as exc:
        assert "project_features.weight" in str(exc)
    else:
        raise AssertionError("incompatible encoder state was accepted")
