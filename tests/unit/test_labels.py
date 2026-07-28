"""Test HDF5 event loading and projection from sample intervals to dense frontend frames.
The suite covers path conventions, focal labels, empty files, and the official zero-
origin grid."""

from pathlib import Path

import h5py
import torch

from a2v2.data import LabelEvents, derive_label_path, load_label_events, rasterize_labels


def test_derives_label_path_from_audio_component() -> None:
    """Check derives label path from audio component."""
    assert derive_label_path(Path("/dataset/wav/day/a.flac")) == Path("/dataset/lbl/day/a.h5")
    assert derive_label_path(Path("/dataset/audio/day/a.wav")) == Path("/dataset/lbl/day/a.h5")


def test_loads_hdf5_events_and_rasterizes_overlap_and_focal(tmp_path: Path) -> None:
    """Check loads HDF5 events and rasterizes overlap and focal."""
    path = tmp_path / "labels.h5"
    with h5py.File(path, "w") as handle:
        handle["start_frame_lbl"] = [0, 12]
        handle["end_frame_lbl"] = [20, 32]
        handle["lbl_cat"] = [0, 1]
        handle["foc"] = [1, 0]
    events = load_label_events(path)
    target = rasterize_labels(
        events,
        waveform_length=40,
        sample_rate=8,
        conv_layers=((4, 3, 1),),
        num_labels=3,
        focal_label_index=2,
    )
    assert target.shape == (40, 3)
    assert torch.all(target[:20, 0] == 1)
    assert torch.all(target[:20, 2] == 1)
    assert torch.all(target[12:32, 1] == 1)
    assert torch.all(target[20:, 2] == 0)
    assert torch.all(target[12:20, :2] == 1)


def test_empty_events_create_zero_targets_at_exact_frontend_length() -> None:
    """Check empty events create zero targets at exact frontend length."""
    events = LabelEvents((), (), (), ())
    target = rasterize_labels(
        events,
        waveform_length=64,
        sample_rate=8000,
        conv_layers=((8, 7, 1), (16, 4, 2), (16, 4, 2)),
        num_labels=4,
        focal_label_index=3,
    )
    assert target.shape == (16, 4)
    assert not target.any()


def test_rasterization_uses_the_official_zero_origin_grid() -> None:
    """Check rasterization uses the official zero origin grid."""
    events = LabelEvents((0,), (1,), (0,), (0,))
    target = rasterize_labels(
        events,
        waveform_length=20,
        sample_rate=8000,
        conv_layers=((4, 3, 1), (4, 4, 2), (4, 2, 1)),
        num_labels=1,
        focal_label_index=None,
    )

    assert target.shape == (10, 1)
    assert target[0, 0] == 1
    assert target[1:, 0].sum() == 0
