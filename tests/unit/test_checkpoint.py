"""Test the native checkpoint schema and random-state transport. The suite covers local and
rank-specific RNG replay, safe object serialization, atomic round trips, and schema
rejection."""

from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

import a2v2.training as training
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


def test_rank_rng_gather_uses_selected_group_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check rank RNG gathering uses its control-plane group in rank order."""
    torch.manual_seed(7)
    rank_zero = capture_rng_state()
    torch.manual_seed(8)
    rank_one = capture_rng_state()
    encoded_by_rank = [serialize_rng_state(rank_zero), serialize_rng_state(rank_one)]
    selected_group = object()

    def fake_gather_object(
        encoded: bytes,
        gathered: list[bytes | None] | None,
        *,
        dst: int,
        group: object,
    ) -> None:
        """Return two encoded rank states through the selected test group."""
        assert encoded == encoded_by_rank[0]
        assert dst == 0
        assert group is selected_group
        assert gathered is not None
        gathered[:] = encoded_by_rank

    monkeypatch.setattr(training.dist, "gather_object", fake_gather_object)

    gathered = training.gather_rank_rng_states(
        rank_zero,
        world_size=2,
        rank=0,
        group=selected_group,
    )

    assert gathered is not None
    assert gathered["world_size"] == 2
    assert torch.equal(gathered["by_rank"][0]["torch"], rank_zero["torch"])
    assert torch.equal(gathered["by_rank"][1]["torch"], rank_one["torch"])


@pytest.mark.parametrize(
    "received",
    [
        [serialize_rng_state(capture_rng_state())],
        [serialize_rng_state(capture_rng_state()), None],
    ],
)
def test_rank_rng_gather_rejects_incomplete_or_non_byte_payloads(
    monkeypatch: pytest.MonkeyPatch,
    received: list[bytes | None],
) -> None:
    """Check rank RNG gathering rejects a partial distributed checkpoint."""

    def fake_gather_object(
        encoded: bytes,
        gathered: list[bytes | None] | None,
        *,
        dst: int,
        group: object,
    ) -> None:
        """Return the parametrized malformed gather result."""
        assert isinstance(encoded, bytes)
        assert dst == 0
        assert group is selected_group
        assert gathered is not None
        gathered[:] = received

    selected_group = object()
    monkeypatch.setattr(training.dist, "gather_object", fake_gather_object)

    with pytest.raises(CheckpointError, match="one byte payload per rank"):
        training.gather_rank_rng_states(
            capture_rng_state(),
            world_size=2,
            rank=0,
            group=selected_group,
        )


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
