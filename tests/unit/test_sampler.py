"""Test token-budget packing and distributed batch assignment. The suite emphasizes
deterministic epoch shuffle, required multiples, tail handling, rank-independent
cursors, and exact resume."""

import pytest

from a2v2.data import (
    DistributedBatchSampler,
    StatelessCropBatchSampler,
    TokenBatchSampler,
)


def test_token_budget_sampler_packs_by_maximum_length() -> None:
    """Check token budget sampler packs by maximum length."""
    sampler = TokenBatchSampler([5, 6, 10, 11, 20], max_tokens=24, shuffle=False)
    batches = list(sampler)
    assert batches == [[0, 1], [2, 3], [4]]
    assert all(max([5, 6, 10, 11, 20][index] for index in batch) * len(batch) <= 24 for batch in batches)


def test_sampler_epoch_shuffle_is_reproducible() -> None:
    """Check sampler epoch shuffle is reproducible."""
    first = TokenBatchSampler([5, 6, 10, 11, 20, 21], max_tokens=42, seed=9, shuffle=True)
    second = TokenBatchSampler([5, 6, 10, 11, 20, 21], max_tokens=42, seed=9, shuffle=True)
    assert list(first) == list(second)


def test_batch_multiple_carries_the_remainder_into_the_next_batch() -> None:
    """Check batch multiple carries the remainder into the next batch."""
    sampler = TokenBatchSampler(
        [8] * 110,
        max_tokens=424,
        shuffle=False,
        required_batch_size_multiple=8,
    )

    batches = list(sampler)

    assert [len(batch) for batch in batches] == [48, 48]
    assert batches[0] == list(range(48))
    assert batches[1] == list(range(48, 96))


def test_sampler_rejects_a_batch_multiple_that_drops_the_entire_dataset() -> None:
    """Check sampler rejects a batch multiple that drops the entire dataset."""
    sampler = TokenBatchSampler(
        [8] * 4,
        max_tokens=64,
        shuffle=False,
        required_batch_size_multiple=8,
    )

    with pytest.raises(ValueError, match="batch-size multiple"):
        iter(sampler).__next__()


def test_sampler_resume_yields_remaining_batches() -> None:
    """Check sampler resume yields remaining batches."""
    sampler = TokenBatchSampler([5, 6, 10, 11, 20], max_tokens=24, shuffle=False)
    iterator = iter(sampler)
    assert next(iterator) == [0, 1]
    state = sampler.state_dict()
    resumed = TokenBatchSampler([5, 6, 10, 11, 20], max_tokens=24, shuffle=False)
    resumed.load_state_dict(state)
    assert list(resumed) == [[2, 3], [4]]


def test_stateless_crop_coordinates_match_after_sampler_resume() -> None:
    """Bind each crop to its absolute ordered occurrence, not worker RNG."""

    sizes = [5, 6, 10, 11, 20]
    base = TokenBatchSampler(sizes, max_tokens=24, seed=9, shuffle=False)
    wrapped = StatelessCropBatchSampler(base, token_sampler=base, seed=31)
    iterator = iter(wrapped)
    first = next(iterator)
    state = base.state_dict()
    uninterrupted_next = next(iterator)

    resumed_base = TokenBatchSampler(
        sizes, max_tokens=24, seed=9, shuffle=False
    )
    resumed_base.load_state_dict(state)
    resumed = StatelessCropBatchSampler(
        resumed_base,
        token_sampler=resumed_base,
        seed=31,
    )

    assert next(iter(resumed)) == uninterrupted_next
    assert all(item.crop_seed >= 0 for item in first + uninterrupted_next)
    assert len({item.crop_seed for item in first + uninterrupted_next}) == len(
        first + uninterrupted_next
    )


def test_sampler_resume_after_last_yield_starts_the_next_epoch() -> None:
    """Check sampler resume after last yield starts the next epoch."""
    sampler = TokenBatchSampler([5, 6, 10, 11, 20], max_tokens=24, shuffle=False)
    iterator = iter(sampler)
    assert next(iterator) == [0, 1]
    assert next(iterator) == [2, 3]
    assert next(iterator) == [4]
    state = sampler.state_dict()
    resumed = TokenBatchSampler([5, 6, 10, 11, 20], max_tokens=24, shuffle=False)

    resumed.load_state_dict(state)

    assert resumed.state_dict() == {"epoch": 1, "next_batch": 0}
    assert next(iter(resumed)) == [0, 1]


def test_distributed_sampler_assigns_disjoint_complete_batches() -> None:
    """Check distributed sampler assigns disjoint complete batches."""
    base = TokenBatchSampler([5, 6, 10, 11, 20, 21], max_tokens=42, shuffle=False)
    rank_zero = list(DistributedBatchSampler(base, rank=0, world_size=2))
    base = TokenBatchSampler([5, 6, 10, 11, 20, 21], max_tokens=42, shuffle=False)
    rank_one = list(DistributedBatchSampler(base, rank=1, world_size=2))
    assert rank_zero
    assert rank_one
    assert not {tuple(batch) for batch in rank_zero} & {tuple(batch) for batch in rank_one}


def test_distributed_sampler_drops_unmatched_tail_batch() -> None:
    """Check distributed sampler drops unmatched tail batch."""
    sizes = [5, 6, 10, 11, 20]
    rank_zero_sampler = DistributedBatchSampler(
        TokenBatchSampler(sizes, max_tokens=24, shuffle=False), rank=0, world_size=2
    )
    rank_one_sampler = DistributedBatchSampler(
        TokenBatchSampler(sizes, max_tokens=24, shuffle=False), rank=1, world_size=2
    )

    rank_zero = list(rank_zero_sampler)
    rank_one = list(rank_one_sampler)

    assert len(rank_zero) == len(rank_one) == 1
    assert len(rank_zero_sampler) == len(rank_one_sampler) == 1
    assert not {tuple(batch) for batch in rank_zero} & {tuple(batch) for batch in rank_one}


def test_distributed_sampler_records_a_rank_independent_resume_position() -> None:
    """Check distributed sampler records a rank independent resume position."""
    sizes = [5, 6, 10, 11, 20, 21]
    rank_zero_base = TokenBatchSampler(sizes, max_tokens=42, shuffle=False)
    rank_one_base = TokenBatchSampler(sizes, max_tokens=42, shuffle=False)

    next(iter(DistributedBatchSampler(rank_zero_base, rank=0, world_size=2)))
    next(iter(DistributedBatchSampler(rank_one_base, rank=1, world_size=2)))

    assert rank_zero_base.state_dict() == rank_one_base.state_dict()
    assert rank_zero_base.state_dict()["next_batch"] == 2


def test_distributed_resume_skips_a_dropped_tail_without_an_empty_epoch() -> None:
    """Check distributed resume skips a dropped tail without an empty epoch."""
    base = TokenBatchSampler([5, 6, 10, 11, 20], max_tokens=24, shuffle=False)
    base.load_state_dict({"epoch": 0, "next_batch": 2})

    resumed = DistributedBatchSampler(base, rank=0, world_size=2)

    assert base.state_dict() == {"epoch": 1, "next_batch": 0}
    assert next(iter(resumed)) == [0, 1]


def test_distributed_sampler_rejects_fewer_batches_than_workers() -> None:
    """Check distributed sampler rejects fewer batches than workers."""
    base = TokenBatchSampler([5, 6], max_tokens=20, shuffle=False)

    with pytest.raises(ValueError, match="fewer complete batches"):
        DistributedBatchSampler(base, rank=0, world_size=2)
