"""Test parameter grouping, Fairseq-compatible Adam, and cosine update scheduling. The
suite protects weight-decay exemptions, epsilon placement, warmup indexing, and
scheduler resume."""

import math
from copy import deepcopy

import pytest
import torch
from torch import nn

import a2v2.training as training
from a2v2.model import PSwish
from a2v2.training import CosineUpdateScheduler, build_optimizer


class GroupFixture(nn.Module):
    """Hold representative parameter shapes for optimizer grouping tests."""
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.norm = nn.LayerNorm(3)
        self.activation = PSwish(3)
        self.alibi_scale = nn.Parameter(torch.ones(1, 1, 1, 1, 1))


class GradientFixture(nn.Module):
    """Expose two named tensors plus an optional unused tensor for clipping."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.zeros(2))
        self.second = nn.Parameter(torch.zeros(1))
        self.unused = nn.Parameter(torch.zeros(1))


def _set_gradients(
    model: GradientFixture,
    first: tuple[float, float],
    second: float | None,
) -> None:
    """Assign literal gradients without invoking optimizer or model formulas."""

    model.first.grad = torch.tensor(first)
    model.second.grad = None if second is None else torch.tensor([second])
    model.unused.grad = None


def test_optimizer_excludes_official_parameter_types_from_weight_decay() -> None:
    """Check optimizer excludes official parameter types from weight decay."""
    model = GroupFixture()
    optimizer = build_optimizer(
        model, name="adam", learning_rate=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.1
    )
    decay = {id(parameter) for group in optimizer.param_groups if group["weight_decay"] for parameter in group["params"]}
    assert id(model.linear.weight) in decay
    assert id(model.linear.bias) not in decay
    assert id(model.norm.weight) not in decay
    assert id(model.activation.p_swish_alpha) not in decay
    assert id(model.alibi_scale) not in decay


def test_legacy_adam_name_uses_decoupled_weight_decay() -> None:
    """Check legacy adam name uses decoupled weight decay."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=0.1,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
    )
    model.weight.grad = torch.zeros_like(model.weight)

    optimizer.step()

    assert model.weight.item() == pytest.approx(0.99)


def test_legacy_adam_matches_fairseq_epsilon_placement() -> None:
    """Check legacy adam matches fairseq epsilon placement."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=0.1,
        betas=(0.9, 0.98),
        eps=0.01,
        weight_decay=0.0,
    )
    model.weight.grad = torch.full_like(model.weight, 2.0)

    optimizer.step()

    first_moment = 0.2
    second_moment = 0.08
    step_size = 0.1 * math.sqrt(1 - 0.98) / (1 - 0.9)
    expected = 1.0 - step_size * first_moment / (math.sqrt(second_moment) + 0.01)
    assert model.weight.item() == pytest.approx(expected)


def test_cosine_scheduler_matches_fairseq_update_indexing() -> None:
    """Check cosine scheduler matches fairseq update indexing."""
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_updates=2,
        warmup_init_lr=1e-5,
        max_updates=10,
    )
    assert scheduler.step_update(0) == pytest.approx(1e-5)
    assert scheduler.step_update(1) == pytest.approx(0.000505)
    assert scheduler.step_update(2) == pytest.approx(1e-3)
    expected_nine = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * 7 / 8))
    assert scheduler.step_update(9) == pytest.approx(expected_nine)


def test_scheduler_state_round_trip() -> None:
    """Check scheduler state round trip."""
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = CosineUpdateScheduler(optimizer, max_lr=1e-3, min_lr=0, warmup_updates=1, max_updates=5)
    scheduler.step_update(3)
    state = scheduler.state_dict()
    scheduler.step_update(4)
    scheduler.load_state_dict(state)
    assert scheduler.last_update == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(scheduler.lr_at_update(3))


def test_gradient_clipper_factory_preserves_global_and_none_behavior() -> None:
    """Catch a strategy factory that changes fixed clipping or no-clip diagnostics."""

    actual = GradientFixture()
    expected = deepcopy(actual)
    _set_gradients(actual, (3.0, 4.0), 12.0)
    _set_gradients(expected, (3.0, 4.0), 12.0)
    expected_norm = torch.nn.utils.clip_grad_norm_(expected.parameters(), 6.5)

    global_clipper = training.build_gradient_clipper(
        actual,
        method="global",
        clip_norm=6.5,
    )
    candidate = global_clipper.clip()

    assert torch.equal(candidate.gradient_norm, expected_norm)
    assert torch.equal(actual.first.grad, expected.first.grad)
    assert torch.equal(actual.second.grad, expected.second.grad)
    global_clipper.commit(candidate)
    assert global_clipper.state_dict() is None

    _set_gradients(actual, (3.0, 4.0), 12.0)
    before = actual.first.grad.clone(), actual.second.grad.clone()
    no_clipper = training.build_gradient_clipper(actual, method="none", clip_norm=1.0)
    no_clip_candidate = no_clipper.clip()

    assert float(no_clip_candidate.gradient_norm) == pytest.approx(13.0)
    assert torch.equal(actual.first.grad, before[0])
    assert torch.equal(actual.second.grad, before[1])


def test_adagc_uses_global_warmup_then_previous_ema_and_clipped_norm() -> None:
    """Catch wrong warm-up boundaries, pre-clip EMA updates, or current-EMA clipping."""

    model = GradientFixture()
    clipper = training.build_gradient_clipper(
        model,
        method="adagc",
        clip_norm=6.5,
        adagc_beta=0.99,
        adagc_relative_clip=1.04,
        adagc_warmup_updates=100,
    )

    _set_gradients(model, (3.0, 4.0), 12.0)
    first = clipper.clip()
    assert float(first.gradient_norm) == pytest.approx(13.0)
    assert torch.linalg.vector_norm(model.first.grad).item() == pytest.approx(2.5)
    assert torch.linalg.vector_norm(model.second.grad).item() == pytest.approx(6.0)
    clipper.commit(first)

    for _ in range(1, 100):
        _set_gradients(model, (3.6, 4.8), 8.0)
        warmup = clipper.clip()
        clipper.commit(warmup)

    state_at_100 = clipper.state_dict()
    assert state_at_100["update"] == 100
    assert state_at_100["parameter_names"] == ["first", "second", "unused"]
    assert state_at_100["norm_emas"]["first"].item() == pytest.approx(2.5)
    assert state_at_100["norm_emas"]["second"].item() == pytest.approx(5.2)
    assert torch.isinf(state_at_100["norm_emas"]["unused"])

    _set_gradients(model, (3.12, 4.16), 5.2)
    adaptive = clipper.clip()

    assert float(adaptive.gradient_norm) == pytest.approx(math.sqrt(5.2**2 + 5.2**2))
    assert torch.linalg.vector_norm(model.first.grad).item() == pytest.approx(2.6)
    assert torch.linalg.vector_norm(model.second.grad).item() == pytest.approx(5.2)
    assert adaptive.clipped_tensors == 1
    assert adaptive.largest_scale == pytest.approx(1.0)
    clipper.commit(adaptive)
    state_at_101 = clipper.state_dict()
    assert state_at_101["update"] == 101
    assert state_at_101["norm_emas"]["first"].item() == pytest.approx(2.501)
    assert state_at_101["norm_emas"]["second"].item() == pytest.approx(5.2)


def test_adagc_zero_missing_nonfinite_and_sparse_gradients_are_explicit() -> None:
    """Catch unsafe zero division, missing-state drift, partial mutation, or sparse use."""

    model = GradientFixture()
    clipper = training.build_gradient_clipper(
        model,
        method="adagc",
        clip_norm=1.0,
        adagc_warmup_updates=1,
    )
    _set_gradients(model, (0.0, 0.0), None)
    zero = clipper.clip()
    clipper.commit(zero)
    state = clipper.state_dict()
    assert float(zero.gradient_norm) == 0.0
    assert state["norm_emas"]["first"].item() == 0.0
    assert torch.isinf(state["norm_emas"]["second"])

    _set_gradients(model, (float("inf"), 0.0), 3.0)
    finite_gradient_before = model.second.grad.clone()
    nonfinite = clipper.clip()
    assert not nonfinite.finite
    assert torch.equal(model.second.grad, finite_gradient_before)
    with pytest.raises(training.CheckpointError, match="non-finite"):
        clipper.commit(nonfinite)
    unchanged = clipper.state_dict()
    assert unchanged["update"] == state["update"]
    for name in state["parameter_names"]:
        assert torch.equal(unchanged["norm_emas"][name], state["norm_emas"][name])

    with pytest.warns(UserWarning, match="Sparse invariant checks"):
        model.first.grad = torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1.0]),
            size=model.first.shape,
            check_invariants=True,
        )
    model.second.grad = None
    with pytest.raises(RuntimeError, match="sparse gradients"):
        clipper.clip()


def test_adagc_state_is_name_keyed_exact_and_tied_parameters_are_tracked_once() -> None:
    """Catch positional state restore or duplicate tracking of one shared parameter."""

    class TiedFixture(nn.Module):
        """Expose two attribute names for one shared parameter object."""

        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Parameter(torch.zeros(2))
            self.alias = self.shared

    model = TiedFixture()
    model.shared.grad = torch.tensor([3.0, 4.0])
    clipper = training.build_gradient_clipper(
        model,
        method="adagc",
        clip_norm=2.5,
        adagc_warmup_updates=100,
    )
    candidate = clipper.clip()
    clipper.commit(candidate)
    saved = clipper.state_dict()

    restored_model = TiedFixture()
    restored = training.build_gradient_clipper(
        restored_model,
        method="adagc",
        clip_norm=2.5,
        adagc_warmup_updates=100,
    )
    restored.load_state_dict(saved)
    round_trip = restored.state_dict()

    assert saved["parameter_names"] == ["shared"]
    assert round_trip.keys() == saved.keys()
    assert round_trip["algorithm_version"] == saved["algorithm_version"]
    assert round_trip["update"] == saved["update"]
    assert round_trip["parameter_names"] == saved["parameter_names"]
    assert torch.equal(round_trip["norm_emas"]["shared"], saved["norm_emas"]["shared"])

    malformed = dict(saved)
    malformed["parameter_names"] = ["renamed"]
    with pytest.raises(training.CheckpointError, match="parameter names"):
        restored.load_state_dict(malformed)


@pytest.mark.parametrize(
    ("case", "mutate"),
    (
        ("missing-version", lambda state: state.pop("algorithm_version")),
        ("extra-top-level", lambda state: state.__setitem__("extra", None)),
        ("boolean-version", lambda state: state.__setitem__("algorithm_version", True)),
        ("wrong-version", lambda state: state.__setitem__("algorithm_version", 2)),
        ("missing-update", lambda state: state.pop("update")),
        ("boolean-update", lambda state: state.__setitem__("update", True)),
        ("negative-update", lambda state: state.__setitem__("update", -1)),
        ("missing-names", lambda state: state.pop("parameter_names")),
        ("names-not-list", lambda state: state.__setitem__("parameter_names", ("first", "second", "unused"))),
        ("renamed-parameter", lambda state: state.__setitem__("parameter_names", ["renamed", "second", "unused"])),
        ("missing-norms", lambda state: state.pop("norm_emas")),
        ("missing-norm", lambda state: state["norm_emas"].pop("first")),
        ("extra-norm", lambda state: state["norm_emas"].__setitem__("extra", torch.tensor(1.0))),
        ("vector-norm", lambda state: state["norm_emas"].__setitem__("first", torch.tensor([1.0]))),
        ("float64-norm", lambda state: state["norm_emas"].__setitem__("first", torch.tensor(1.0, dtype=torch.float64))),
        ("nan-norm", lambda state: state["norm_emas"].__setitem__("first", torch.tensor(float("nan")))),
        ("negative-infinite-norm", lambda state: state["norm_emas"].__setitem__("first", torch.tensor(float("-inf")))),
    ),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_adagc_rejects_every_malformed_state_without_mutation(
    case: str,
    mutate: object,
) -> None:
    """Catch permissive schema parsing or partial state installation on reject."""

    del case
    model = GradientFixture()
    model.first.grad = torch.tensor([3.0, 4.0])
    model.second.grad = torch.tensor([1.0])
    clipper = training.build_gradient_clipper(
        model,
        method="adagc",
        clip_norm=2.5,
        adagc_warmup_updates=100,
    )
    clipper.commit(clipper.clip())
    before = clipper.state_dict()
    malformed = deepcopy(before)
    mutate(malformed)  # type: ignore[operator]

    with pytest.raises(training.CheckpointError):
        clipper.load_state_dict(malformed)

    actual = clipper.state_dict()
    assert actual["algorithm_version"] == before["algorithm_version"]
    assert actual["update"] == before["update"]
    assert actual["parameter_names"] == before["parameter_names"]
    for name in before["parameter_names"]:
        assert torch.equal(actual["norm_emas"][name], before["norm_emas"][name])


def test_adagc_state_accepts_positive_infinity_only_as_unobserved_sentinel() -> None:
    """Catch schema hardening that rejects the designed no-gradient sentinel."""

    source_model = GradientFixture()
    source = training.build_gradient_clipper(
        source_model,
        method="adagc",
        clip_norm=2.5,
    )
    state = source.state_dict()
    restored = training.build_gradient_clipper(
        GradientFixture(),
        method="adagc",
        clip_norm=2.5,
    )

    restored.load_state_dict(state)

    for value in restored.state_dict()["norm_emas"].values():
        assert torch.isposinf(value)
