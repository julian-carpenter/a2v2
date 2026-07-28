"""Test the native checkpoint schema and random-state transport. The suite covers local and
rank-specific RNG replay, safe object serialization, atomic round trips, and schema
rejection."""

from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from a2v2.training import (
    CheckpointError,
    capture_rng_state,
    deserialize_rng_state,
    load_checkpoint,
    restore_rng_state,
    serialize_rng_state,
    save_checkpoint,
)


def test_rng_state_round_trip() -> None:
    """Check RNG state round trip."""
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_ranked_rng_state_restores_the_current_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check ranked RNG state restores the current worker."""
    torch.manual_seed(7)
    rank_zero = capture_rng_state()
    torch.manual_seed(8)
    rank_one = capture_rng_state()
    expected = torch.rand(1)
    torch.manual_seed(99)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    restore_rng_state({"world_size": 2, "by_rank": [rank_zero, rank_one]})

    assert torch.equal(torch.rand(1), expected)


def test_rng_state_has_an_opaque_distributed_transport() -> None:
    """Check RNG state has an opaque distributed transport."""
    state = capture_rng_state()

    encoded = serialize_rng_state(state)
    decoded = deserialize_rng_state(encoded)

    assert isinstance(encoded, bytes)
    assert torch.equal(decoded["torch"], state["torch"])


def test_native_checkpoint_round_trip(tmp_path: Path) -> None:
    """Check native checkpoint round trip."""
    path = tmp_path / "checkpoint.pt"
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    payload = {
        "format_version": 1,
        "stage": "pretrain",
        "config": {"name": "tiny"},
        "model": model.state_dict(),
        "teacher": None,
        "optimizer": optimizer.state_dict(),
        "scheduler": {"last_update": 2},
        "scaler": None,
        "update": 2,
        "epoch": 1,
        "batch_in_epoch": 3,
        "rng_state": capture_rng_state(),
        "sampler_state": {"epoch": 1, "next_batch": 3},
        "best_metric": 0.5,
    }
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    assert loaded["format_version"] == 1
    assert loaded["update"] == 2
    assert set(loaded["model"]) == set(model.state_dict())


def test_checkpoint_rejects_missing_schema_keys(tmp_path: Path) -> None:
    """Check checkpoint rejects missing schema keys."""
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 1}, path)
    with pytest.raises(CheckpointError, match="missing"):
        load_checkpoint(path)
