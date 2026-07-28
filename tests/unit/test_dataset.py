"""Test manifest parsing, waveform and HDF5 loading, record filtering, and batch collation.
Crop and padding checks protect sample-to-frame alignment between audio and labels."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import soundfile as sf
import torch

from a2v2.data import AudioDataset, ManifestError, collate_audio, read_manifest


LAYERS = ((8, 7, 1), (16, 4, 2), (16, 4, 2))


def _write_example(root: Path, name: str, length: int, sample_rate: int = 8000) -> None:
    """Write one paired waveform and HDF5 label fixture to disk."""
    audio_path = root / "wav" / name
    label_path = root / "lbl" / Path(name).with_suffix(".h5")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.stack((np.linspace(-1, 1, length), np.linspace(1, -1, length)), axis=1)
    sf.write(audio_path, samples, sample_rate, subtype="FLOAT")
    with h5py.File(label_path, "w") as handle:
        handle["start_frame_lbl"] = [8]
        handle["end_frame_lbl"] = [length - 8]
        handle["lbl_cat"] = [0]
        handle["foc"] = [1]


def _manifest(tmp_path: Path) -> Path:
    """Write a root-plus-TSV manifest for the supplied audio example."""
    _write_example(tmp_path, "a.wav", 64)
    _write_example(tmp_path, "b.wav", 80)
    path = tmp_path / "train.tsv"
    path.write_text(f"{tmp_path / 'wav'}\na.wav\t64\nb.wav\t80\n", encoding="utf-8")
    return path


def test_reads_manifest_with_resolved_paths(tmp_path: Path) -> None:
    """Check reads manifest with resolved paths."""
    records = read_manifest(_manifest(tmp_path))
    assert [record.num_samples for record in records] == [64, 80]
    assert records[0].audio_path == tmp_path / "wav/a.wav"
    assert records[0].label_path == tmp_path / "lbl/a.h5"


def test_manifest_reports_line_for_malformed_or_missing_audio(tmp_path: Path) -> None:
    """Check manifest reports line for malformed or missing audio."""
    malformed = tmp_path / "bad.tsv"
    malformed.write_text(f"{tmp_path}\nmissing.wav\tnot-an-int\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="line 2"):
        read_manifest(malformed)


def test_dataset_loads_mono_normalized_audio_and_labels(tmp_path: Path) -> None:
    """Check dataset loads mono normalized audio and labels."""
    dataset = AudioDataset(
        _manifest(tmp_path), sample_rate=8000, conv_layers=LAYERS,
        normalize=True, labels=("call", "focal"),
    )
    item = dataset[0]
    assert item["source"].shape == (64,)
    assert item["source"].mean().abs() < 1e-5
    assert item["target"].shape == (16, 2)
    assert item["id"] == 0


def test_dataset_rejects_sample_rate_mismatch(tmp_path: Path) -> None:
    """Check dataset rejects sample rate mismatch."""
    _write_example(tmp_path, "bad.wav", 64, sample_rate=16000)
    manifest = tmp_path / "bad.tsv"
    manifest.write_text(f"{tmp_path / 'wav'}\nbad.wav\t64\n", encoding="utf-8")
    dataset = AudioDataset(manifest, sample_rate=8000, conv_layers=LAYERS)
    with pytest.raises(ValueError, match="sample rate"):
        dataset[0]


def test_pretraining_min_label_size_filters_records_without_label_files(tmp_path: Path) -> None:
    """Check pretraining min label size filters records without label files."""
    manifest = _manifest(tmp_path)
    (tmp_path / "lbl/b.h5").unlink()

    dataset = AudioDataset(
        manifest,
        sample_rate=8000,
        conv_layers=LAYERS,
        labels=None,
        min_label_size=1,
    )

    assert len(dataset) == 1
    assert dataset.records[0].audio_path.name == "a.wav"


def test_collate_crops_waveforms_and_targets_with_same_offset(tmp_path: Path) -> None:
    """Check collate crops waveforms and targets with same offset."""
    dataset = AudioDataset(
        _manifest(tmp_path), sample_rate=8000, conv_layers=LAYERS,
        normalize=False, labels=("call", "focal"),
    )
    items = [dataset[0], dataset[1]]
    batch = collate_audio(
        items, max_sample_size=56, pad=False, conv_layers=LAYERS,
        generator=torch.Generator().manual_seed(4),
    )
    assert batch["source"].shape == (2, 56)
    assert batch["target"].shape == (2, 14, 2)
    assert batch["crop_offsets"].shape == (2,)
    assert "padding_mask" not in batch


def test_collate_right_pads_audio_targets_and_mask(tmp_path: Path) -> None:
    """Check collate right pads audio targets and mask."""
    dataset = AudioDataset(
        _manifest(tmp_path), sample_rate=8000, conv_layers=LAYERS,
        normalize=False, labels=("call", "focal"),
    )
    batch = collate_audio([dataset[0], dataset[1]], max_sample_size=100, pad=True, conv_layers=LAYERS)
    assert batch["source"].shape == (2, 80)
    assert batch["padding_mask"][0, 64:].all()
    assert batch["target"].shape == (2, 20, 2)
    assert not batch["target"][0, 16:].any()
