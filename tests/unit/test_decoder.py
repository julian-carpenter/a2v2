"""Test restoration of removed mask positions and convolutional teacher feature decoding.
The suite isolates ordering, output shape, and input dropout placement."""

import torch
from torch.nn import functional as F

from a2v2.config import DecoderConfig
from a2v2.model import ConvDecoder, restore_masked_features
from a2v2.model import make_mask_info


def test_restore_masked_features_preserves_unmasked_positions() -> None:
    """Check restore masked features preserves unmasked positions."""
    features = torch.arange(48).reshape(2, 6, 4).float()
    mask = torch.tensor(
        [[False, True, False, True, False, True], [True, False, True, False, True, False]]
    )
    info = make_mask_info(features, mask)
    restored = restore_masked_features(info.x_unmasked, info, noise_std=0.0)
    assert restored.shape == features.shape
    assert torch.equal(restored[~mask], features[~mask])
    assert torch.equal(restored[mask], torch.zeros_like(restored[mask]))


def test_convolutional_decoder_restores_full_time_axis_and_dimension() -> None:
    """Check convolutional decoder restores full time axis and dimension."""
    cfg = DecoderConfig(
        input_dropout=0.0,
        decoder_dim=8,
        decoder_groups=2,
        decoder_kernel=4,
        decoder_layers=2,
    )
    decoder = ConvDecoder(cfg, input_dim=4).eval()
    value = torch.randn(2, 7, 4)
    output = decoder(value)
    assert output.shape == value.shape


def test_decoder_input_dropout_precedes_mask_restoration() -> None:
    """Check decoder input dropout precedes mask restoration."""
    config = DecoderConfig(
        input_dropout=0.5,
        decoder_dim=8,
        decoder_groups=2,
        decoder_kernel=3,
        decoder_layers=1,
    )
    decoder = ConvDecoder(config, input_dim=4).train()
    features = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
    mask = torch.tensor([[False, True, False, True, False, True, False, True]])
    mask_info = make_mask_info(features, mask)

    torch.manual_seed(5)
    expected = restore_masked_features(
        F.dropout(mask_info.x_unmasked, p=0.5, training=True),
        mask_info,
        noise_std=0.0,
    )
    torch.manual_seed(5)
    actual = decoder.prepare_input(mask_info.x_unmasked, mask_info, noise_std=0.0)

    assert torch.equal(actual, expected)
