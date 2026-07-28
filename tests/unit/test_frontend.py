"""Test waveform frontend geometry and shared encoder internals. The suite covers
convolution lengths, layer outputs, initialization, dropout, mask ordering, ALiBi memory
ordering, and parameter estimates."""

from dataclasses import replace
from pathlib import Path

import torch

from a2v2.data import conv_output_length
from a2v2.config import load_config
from a2v2.model import (
    AudioEncoder,
    ConvFeatureEncoder,
    _select_and_scale_alibi,
)
from a2v2.model import make_mask_info


ROOT = Path(__file__).parents[2]


def test_conv_feature_encoder_shapes_and_padding_lengths() -> None:
    """Check conv feature encoder shapes and padding lengths."""
    layers = ((8, 7, 1), (16, 4, 2), (16, 4, 2))
    encoder = ConvFeatureEncoder(layers, sample_rate=8000, sinc_input=True, use_pswish=True)
    waveforms = torch.randn(2, 64)
    output = encoder(waveforms)
    assert output.shape == (2, 16, 16)
    assert conv_output_length(torch.tensor([64, 60]), layers).tolist() == [16, 15]


def test_audio_encoder_returns_prenet_and_shared_layers() -> None:
    """Check audio encoder returns prenet and shared layers."""
    cfg = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    encoder = AudioEncoder.from_config(cfg)
    waveform = torch.randn(2, 64)
    padding = torch.zeros(2, 64, dtype=torch.bool)
    padding[1, 60:] = True
    result = encoder(waveform, padding)
    assert result.x.shape == (2, 16, 16)
    assert result.padding_mask.shape == (2, 16)
    assert result.padding_mask[1, 15]
    assert len(result.prenet_layers) == 1
    assert len(result.layer_outputs) == 2
    assert encoder.prenet.norm is not None
    assert encoder.prenet.norm_before
    assert encoder.transformer.norm is None


def test_audio_feature_projection_uses_the_official_post_bert_reset() -> None:
    """Check audio feature projection uses the official post bert reset."""
    torch.manual_seed(17)
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")

    encoder = AudioEncoder.from_config(config)

    assert encoder.project_features.weight.std() > 0.08


def test_shared_transformer_uses_configured_input_dropout() -> None:
    """Check shared transformer uses configured input dropout."""
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    config = replace(config, model=replace(config.model, dropout_input=0.35))

    encoder = AudioEncoder.from_config(config)

    assert encoder.transformer.dropout.p == 0.35


def test_masked_positional_encoder_receives_zeroed_mask_positions() -> None:
    """Check masked positional encoder receives zeroed mask positions."""
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    encoder = AudioEncoder.from_config(config).eval()
    projected = torch.randn(2, 6, config.model.embed_dim)
    mask = torch.tensor([
        [False, True, True, False, False, False],
        [True, False, False, False, True, False],
    ])
    mask_info = make_mask_info(projected, mask)
    positional_inputs: list[torch.Tensor] = []
    handle = encoder.positional_encoder.register_forward_pre_hook(
        lambda _module, values: positional_inputs.append(values[0].detach().clone())
    )

    encoder.encode_projected(projected, mask_info=mask_info)
    handle.remove()

    expected = projected.masked_fill(mask.unsqueeze(-1), 0)
    assert len(positional_inputs) == 1
    assert torch.equal(positional_inputs[0], expected)


def test_selecting_alibi_before_scaling_preserves_forward_and_scale_gradient() -> None:
    """Check selecting ALiBi before scaling preserves forward and scale gradient."""
    batch, heads, time = 3, 2, 7
    base = torch.randn(1, heads, time, time).expand(batch, -1, -1, -1)
    keep = torch.tensor([[0, 2, 5], [1, 4, 6], [0, 3, 6]])
    reference_scale = torch.randn(1, 1, heads, 1, 1, requires_grad=True)
    actual_scale = reference_scale.detach().clone().requires_grad_(True)

    reference = base * reference_scale.clamp_min(0).squeeze(0).to(base)
    rows = keep[:, None, :, None].expand(-1, heads, -1, time)
    reference = torch.gather(reference, -2, rows)
    columns = keep[:, None, None, :].expand(-1, heads, reference.shape[-2], -1)
    reference = torch.gather(reference, -1, columns)
    actual = _select_and_scale_alibi(base, actual_scale, keep, heads)

    assert torch.equal(actual, reference)
    reference.square().sum().backward()
    actual.square().sum().backward()
    assert torch.equal(actual_scale.grad, reference_scale.grad)


def test_published_frontends_have_the_official_frame_counts() -> None:
    """Check published frontends have the official frame counts."""
    meerkat = load_config(ROOT / "configs/MeerKAT/a2v_large_pretrain_best.yaml")
    hyena = load_config(ROOT / "configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml")
    assert conv_output_length(8000, meerkat.task.conv_feature_layers) == 200
    assert conv_output_length(24000, hyena.task.conv_feature_layers) == 201


def test_published_profiles_can_be_counted_without_allocating_models() -> None:
    """Check published profiles can be counted without allocating models."""
    meerkat = load_config(ROOT / "configs/MeerKAT/a2v_large_pretrain_best.yaml")
    hyena = load_config(ROOT / "configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml")
    assert AudioEncoder.estimated_parameter_count(meerkat) > 300_000_000
    assert AudioEncoder.estimated_parameter_count(hyena) > 80_000_000
