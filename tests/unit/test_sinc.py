"""Test the analytic raw-waveform Sinc filterbank. The assertions cover shape, symmetry,
normalization, frequency limits, gradient flow, checkpoint layout, and invalid
architectures."""

import torch

from a2v2.model import SincConv1d


def test_sinc_shape_same_padding_and_gradients() -> None:
    """Check sinc shape same padding and gradients."""
    layer = SincConv1d(
        out_channels=8,
        kernel_size=31,
        stride=1,
        sample_rate=8000,
        padding="same",
    )
    waveform = torch.randn(2, 128, requires_grad=True)
    output = layer(waveform)
    assert output.shape == (2, 8, 128)
    output.square().mean().backward()
    assert waveform.grad is not None
    assert layer.low_hz_.grad is not None
    assert layer.band_hz_.grad is not None


def test_sinc_filters_are_symmetric_normalized_and_bounded() -> None:
    """Check sinc filters are symmetric normalized and bounded."""
    layer = SincConv1d(12, 31, sample_rate=8000, padding="valid")
    filters = layer.filters()
    assert filters.shape == (12, 1, 31)
    assert torch.allclose(filters, filters.flip(-1), atol=1e-6)
    assert torch.allclose(filters[..., 15], torch.ones(12, 1), atol=1e-6)
    low, high = layer.frequency_bounds()
    assert torch.all(low >= layer.min_low_hz)
    assert torch.all(high <= layer.sample_rate / 2)
    assert torch.all(high > low)


def test_sinc_official_state_layout_round_trip() -> None:
    """Check sinc official state layout round trip."""
    source = SincConv1d(8, 31, sample_rate=8000)
    target = SincConv1d(8, 31, sample_rate=8000)
    state = source.state_dict()
    assert set(state) == {"low_hz_", "band_hz_"}
    with torch.no_grad():
        source.low_hz_.add_(3)
    target.load_state_dict(source.state_dict())
    assert torch.equal(target.low_hz_, source.low_hz_)


def test_learnable_root_filter_layout_and_absolute_output() -> None:
    """Check learnable root filter layout and absolute output."""
    layer = SincConv1d(
        8,
        31,
        sample_rate=8000,
        learnable_filters=True,
        apply_window_to_root=True,
        return_abs=True,
    )
    assert set(layer.state_dict()) == {"kernel"}
    output = layer(torch.randn(1, 64))
    assert torch.all(output >= 0)


def test_sinc_rejects_invalid_channel_and_kernel_combinations() -> None:
    """Check sinc rejects invalid channel and kernel combinations."""
    try:
        SincConv1d(7, 30, in_channels=2)
    except ValueError as exc:
        assert "odd" in str(exc) or "divisible" in str(exc)
    else:
        raise AssertionError("invalid SincConv1d parameters were accepted")
