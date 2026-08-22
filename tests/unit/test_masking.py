"""Test deterministic Data2Vec span masks and their restoration indices. The suite covers
padding, dropout, spacing, short spans, clone independence, and reproducibility."""

from pathlib import Path

import pytest
import torch

from a2v2.config import load_config
from a2v2.model import (
    AudioEncoder,
    MaskInfo,
    compute_mask_indices,
    make_mask_info,
    masks_for_cloned_batch,
)


ROOT = Path(__file__).parents[2]


def test_span_masks_are_deterministic_and_equalized() -> None:
    """Check span masks are deterministic and equalized."""
    kwargs = dict(
        shape=(3, 40),
        padding_mask=None,
        mask_prob=0.55,
        mask_length=3,
        seed=9,
        epoch=4,
        indices=torch.tensor([10, 11, 12]),
    )
    first = compute_mask_indices(**kwargs)
    second = compute_mask_indices(**kwargs)
    assert torch.equal(first, second)
    assert first.dtype == torch.bool
    assert first.sum(dim=1).unique().numel() == 1
    assert 0 < first.sum() < first.numel()


def test_padding_is_never_masked() -> None:
    """Check padding is never masked."""
    padding = torch.zeros(2, 30, dtype=torch.bool)
    padding[0, 23:] = True
    padding[1, 27:] = True
    mask = compute_mask_indices(
        (2, 30), padding, 0.5, 2, seed=3, epoch=1, indices=torch.tensor([1, 2])
    )
    assert not torch.any(mask & padding)


def test_mask_dropout_removes_masked_positions() -> None:
    """Check mask dropout removes masked positions."""
    common = dict(
        shape=(2, 50), padding_mask=None, mask_prob=0.5, mask_length=2,
        seed=8, epoch=2, indices=torch.tensor([4, 5]),
    )
    full = compute_mask_indices(**common)
    dropped = compute_mask_indices(**common, mask_dropout=0.5)
    assert dropped.sum() < full.sum()


def test_no_overlap_respects_minimum_space() -> None:
    """Check no overlap respects minimum space."""
    mask = compute_mask_indices(
        (1, 80), None, 0.35, 4, no_overlap=True, min_space=2,
        seed=5, epoch=2, indices=torch.tensor([13]), require_same_masks=False,
    )[0]
    starts = torch.nonzero(mask & ~torch.roll(mask, 1), as_tuple=False).flatten()
    assert torch.all(starts[1:] - starts[:-1] >= 6)


def test_probability_above_one_is_supported_for_short_spans() -> None:
    """Check probability above one is supported for short spans."""
    mask = compute_mask_indices(
        (2, 100), None, 1.5, 2, seed=1, epoch=1, indices=torch.tensor([1, 2])
    )
    assert torch.all(mask.sum(dim=1) > 50)
    assert torch.all(mask.sum(dim=1) < 100)


def test_mask_info_gathers_kept_features_and_can_restore_order() -> None:
    """Check mask info gathers kept features and can restore order."""
    features = torch.arange(24).view(2, 6, 2).float()
    mask = torch.tensor(
        [[False, True, False, True, False, True], [True, False, True, False, True, False]]
    )
    info = make_mask_info(features, mask)
    assert info.x_unmasked.shape == (2, 3, 2)
    shuffled = torch.cat((info.x_unmasked, torch.full_like(info.x_unmasked, -1)), dim=1)
    restored = torch.gather(shuffled, 1, info.ids_restore)
    assert torch.equal(restored[~mask], features[~mask])
    assert torch.all(restored[mask] == -1)


def test_masked_encoder_gathers_original_shuffled_position_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep teacher frame coordinates instead of renumbering student tokens."""

    config = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        ("model.position_encoding=rope", "model.attention_backend=manual"),
    )
    encoder = AudioEncoder.from_config(config).eval()
    projected = torch.randn(1, 4, config.model.embed_dim)
    keep = torch.tensor([[3, 0, 2]])
    ids_keep = keep.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
    mask_info = MaskInfo(
        x_unmasked=torch.gather(projected, 1, ids_keep),
        mask=torch.tensor([[False, True, False, False]]),
        ids_restore=torch.arange(4).view(1, 4, 1).expand_as(projected).long(),
        ids_keep=ids_keep,
    )
    observed: list[torch.Tensor | None] = []

    def capture_positions(stack: torch.nn.Module) -> None:
        """Record IDs at each encoder stack boundary without replacing its work."""

        original = stack.forward

        def forward(
            value: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
            alibi: torch.Tensor | None = None,
            alibi_scale: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            """Capture one stack boundary and preserve the original execution."""

            observed.append(None if position_ids is None else position_ids.detach().clone())
            if position_ids is None:
                return original(value, padding_mask, alibi, alibi_scale)
            return original(value, padding_mask, alibi, alibi_scale, position_ids)

        monkeypatch.setattr(stack, "forward", forward)

    capture_positions(encoder.prenet)
    capture_positions(encoder.transformer)

    encoder.encode_projected(projected, mask_info=mask_info)

    expected = torch.tensor([[3, 0, 2]])
    assert len(observed) == 2
    assert all(position_ids is not None for position_ids in observed)
    assert all(torch.equal(position_ids, expected) for position_ids in observed)


def test_cloned_batches_receive_distinct_reproducible_masks() -> None:
    """Check cloned batches receive distinct reproducible masks."""
    sample_ids = torch.tensor([21, 22])
    first = masks_for_cloned_batch(
        batch_size=2, length=40, clone_count=3, mask_prob=0.5, mask_length=2,
        global_seed=7, update=10, sample_ids=sample_ids,
    )
    second = masks_for_cloned_batch(
        batch_size=2, length=40, clone_count=3, mask_prob=0.5, mask_length=2,
        global_seed=7, update=10, sample_ids=sample_ids,
    )
    assert torch.equal(first, second)
    assert first.shape == (6, 40)
    assert not torch.equal(first[0], first[2])


def test_mask_matches_the_recorded_archived_fairseq_fixture() -> None:
    """Keep the archived reference result without keeping archived source."""

    expected = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    actual = compute_mask_indices(
        (2, 20),
        None,
        0.5,
        3,
        seed=7,
        epoch=2,
        indices=torch.tensor([11, 12]),
    )

    assert torch.equal(actual, expected)
