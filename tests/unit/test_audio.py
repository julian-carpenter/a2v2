"""Test waveform loading, normalization, convolution timing, timestamps, and native
resampling. These functions define the mapping between recording samples and the frame
axis used by labels and inference."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from a2v2.data import (
    conv_output_length,
    feature_timestamps,
    load_audio,
    normalize_waveform,
    resample_waveform,
)


def test_convolution_output_length_for_scalar_and_tensor() -> None:
    """Check convolution output length for scalar and tensor."""
    layers = ((8, 7, 1), (16, 4, 2), (16, 4, 2))
    assert conv_output_length(64, layers) == 16
    assert torch.equal(
        conv_output_length(torch.tensor([64, 60]), layers),
        torch.tensor([16, 15]),
    )


def test_feature_timestamps_follow_receptive_field_centers() -> None:
    """Check feature timestamps follow receptive field centers."""
    layers = ((8, 3, 2), (16, 3, 2))
    times = feature_timestamps(3, sample_rate=8, layers=layers)
    assert torch.allclose(times, torch.tensor([0.0, 0.5, 1.0]))


def test_normalization_is_per_example() -> None:
    """Check normalization is per example."""
    waveform = torch.tensor([[1.0, 2.0, 3.0], [10.0, 12.0, 14.0]])
    normalized = normalize_waveform(waveform)
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized.var(dim=-1, unbiased=False), torch.ones(2), atol=2e-5)


def test_load_audio_preserves_channels_and_float32(tmp_path: Path) -> None:
    """Check load audio preserves channels and float32."""
    path = tmp_path / "stereo.wav"
    samples = np.stack((np.linspace(-0.5, 0.5, 20), np.linspace(0.5, -0.5, 20)), axis=1)
    sf.write(path, samples, 8000, subtype="FLOAT")

    waveform, sample_rate = load_audio(path)

    assert sample_rate == 8000
    assert waveform.shape == (2, 20)
    assert waveform.dtype == torch.float32
    assert torch.allclose(waveform[0], torch.from_numpy(samples[:, 0]).float())


def test_identity_resampling_returns_equal_values() -> None:
    """Check identity resampling returns equal values."""
    waveform = torch.randn(2, 123)
    result = resample_waveform(waveform, 8000, 8000)
    assert torch.equal(result, waveform)
    assert result.data_ptr() != waveform.data_ptr()


def test_resampling_has_deterministic_length_and_preserves_tone() -> None:
    """Check resampling has deterministic length and preserves tone."""
    source_rate = 48_000
    target_rate = 8_000
    time = torch.arange(source_rate, dtype=torch.float32) / source_rate
    waveform = torch.sin(2 * torch.pi * 440 * time)

    result = resample_waveform(waveform, source_rate, target_rate)

    assert result.shape == (target_rate,)
    spectrum = torch.fft.rfft(result)
    peak_bin = int(spectrum.abs().argmax())
    assert peak_bin == pytest.approx(440, abs=1)
    assert result.square().mean().sqrt().item() == pytest.approx(2**-0.5, rel=0.03)


def test_resampling_rejects_invalid_rates() -> None:
    """Check resampling rejects invalid rates."""
    with pytest.raises(ValueError, match="sample rates"):
        resample_waveform(torch.ones(10), 0, 8000)
