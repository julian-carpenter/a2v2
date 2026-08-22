"""Test parameter grouping, Fairseq-compatible Adam, and cosine update scheduling. The
suite protects weight-decay exemptions, epsilon placement, warmup indexing, and
scheduler resume."""

import importlib
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib

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


def test_native_optimizers_do_not_import_bitsandbytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch eager optional imports on the native optimizer paths."""

    real_import = importlib.import_module

    def unexpected_import(name: str, *args: object, **kwargs: object) -> object:
        """Delegate every import except the optional package under test."""

        if name == "bitsandbytes":
            pytest.fail("native optimizer unexpectedly imported bitsandbytes")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    model = GroupFixture()

    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
    )

    assert type(optimizer) is training.FairseqCompatibleAdam
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0]


def test_bitsandbytes_optimizer_rejects_cpu_before_optional_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch unsupported CPU construction or dependency probing before platform checks."""

    real_import = importlib.import_module

    def unexpected_import(name: str, *args: object, **kwargs: object) -> object:
        """Fail if CPU validation reaches optional-package import."""

        if name == "bitsandbytes":
            pytest.fail("CPU rejection unexpectedly imported bitsandbytes")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unexpected_import)

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        build_optimizer(
            GroupFixture(),
            name="adam8bit",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
            min_8bit_size=1024,
            device=torch.device("cpu"),
        )


def test_bitsandbytes_optimizer_missing_extra_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a raw import error that does not identify the install extra."""

    def missing(name: str) -> object:
        """Simulate only the selected optional package being unavailable."""

        assert name == "bitsandbytes"
        raise ModuleNotFoundError("No module named 'bitsandbytes'")

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(training.OptimizerSetupError, match=r"a2v2\[bnb\]"):
        build_optimizer(
            GroupFixture(),
            name="adam8bit",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
            min_8bit_size=1024,
            device=torch.device("cuda"),
        )


def test_bitsandbytes_optimizer_rejects_nonpositive_minimum_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch an invalid tensor-size threshold reaching the optional package."""

    real_import = importlib.import_module

    def unexpected_import(name: str, *args: object, **kwargs: object) -> object:
        """Fail if threshold validation reaches optional-package import."""

        if name == "bitsandbytes":
            pytest.fail("invalid min_8bit_size unexpectedly imported bitsandbytes")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unexpected_import)

    with pytest.raises(ValueError, match="min_8bit_size must be positive"):
        build_optimizer(
            GroupFixture(),
            name="adam8bit",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
            min_8bit_size=0,
            device=torch.device("cuda"),
        )


@pytest.mark.parametrize(
    ("name", "expected_class_name"),
    (("adam8bit", "FakeAdam8bit"), ("adamw8bit", "FakeAdamW8bit")),
)
def test_bitsandbytes_optimizer_selection_forwards_groups_and_adam_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    expected_class_name: str,
) -> None:
    """Catch wrong 8-bit class selection, grouping, or omitted Adam arguments."""

    class FakeOptimizer:
        """Record constructor inputs at the optional-library boundary."""

        def __init__(self, groups: list[dict[str, object]], **kwargs: object) -> None:
            self.param_groups = groups
            self.kwargs = kwargs

    class FakeAdam8bit(FakeOptimizer):
        """Stand in for bitsandbytes.optim.Adam8bit."""

    class FakeAdamW8bit(FakeOptimizer):
        """Stand in for bitsandbytes.optim.AdamW8bit."""

    fake_module = SimpleNamespace(
        __version__="0.49.1",
        cextension=SimpleNamespace(
            lib=SimpleNamespace(compiled_with_cuda=True),
        ),
        optim=SimpleNamespace(
            Adam8bit=FakeAdam8bit,
            AdamW8bit=FakeAdamW8bit,
        ),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda imported: fake_module if imported == "bitsandbytes" else None,
    )
    model = GroupFixture()

    optimizer = build_optimizer(
        model,
        name=name,
        learning_rate=3e-4,
        betas=(0.8, 0.95),
        eps=2e-8,
        weight_decay=0.12,
        min_8bit_size=2048,
        device=torch.device("cuda"),
    )

    assert type(optimizer).__name__ == expected_class_name
    assert optimizer.kwargs == {
        "lr": 3e-4,
        "betas": (0.8, 0.95),
        "eps": 2e-8,
        "min_8bit_size": 2048,
    }
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.12, 0.0]
    assert optimizer._a2v2_bitsandbytes_version == "0.49.1"


@pytest.mark.parametrize("version", ("0.48.9", "0.50.0"))
def test_bitsandbytes_optimizer_rejects_versions_outside_pinned_range(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    """Catch constructing against an unvalidated bitsandbytes release."""

    class AvailableOptimizer:
        """Stand in for an optimizer that must not be constructed."""

        def __init__(self, groups: object, **kwargs: object) -> None:
            self.param_groups = groups

    fake_module = SimpleNamespace(
        __version__=version,
        cextension=SimpleNamespace(
            lib=SimpleNamespace(compiled_with_cuda=True),
        ),
        optim=SimpleNamespace(Adam8bit=AvailableOptimizer),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    with pytest.raises(
        training.OptimizerSetupError,
        match=r"bitsandbytes>=0\.49,<0\.50",
    ):
        build_optimizer(
            GroupFixture(),
            name="adam8bit",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
            device=torch.device("cuda"),
        )


@pytest.mark.parametrize(
    ("optional_api", "message"),
    (
        (SimpleNamespace(), "bitsandbytes.optim"),
        (SimpleNamespace(optim=SimpleNamespace()), "Adam8bit"),
    ),
)
def test_bitsandbytes_optimizer_rejects_missing_optimizer_api(
    monkeypatch: pytest.MonkeyPatch,
    optional_api: SimpleNamespace,
    message: str,
) -> None:
    """Catch leaking raw attribute errors for an incomplete optional package."""

    fake_module = SimpleNamespace(
        __version__="0.49.2",
        cextension=SimpleNamespace(
            lib=SimpleNamespace(compiled_with_cuda=True),
        ),
        **vars(optional_api),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    with pytest.raises(training.OptimizerSetupError, match=message):
        build_optimizer(
            GroupFixture(),
            name="adam8bit",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
            device=torch.device("cuda"),
        )


def test_bitsandbytes_optimizer_rejects_unloaded_cuda_native_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch deferring a missing CUDA binary until the first optimizer step."""

    class AvailableOptimizer:
        """Stand in for an optimizer hidden behind a failed native load."""

        def __init__(self, groups: object, **kwargs: object) -> None:
            self.param_groups = groups

    fake_module = SimpleNamespace(
        __version__="0.49.2",
        cextension=SimpleNamespace(
            lib=SimpleNamespace(compiled_with_cuda=False),
        ),
        optim=SimpleNamespace(Adam8bit=AvailableOptimizer),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    with pytest.raises(
        training.OptimizerSetupError,
        match="CUDA native library",
    ) as raised:
        build_optimizer(
            GroupFixture(),
            name="adam8bit",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
            device=torch.device("cuda"),
        )
    assert "compatible bitsandbytes binary" in str(raised.value)
    assert "BNB_CUDA_VERSION" in str(raised.value)


def test_bitsandbytes_optimizer_translates_constructor_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch exposing an optional backend's raw constructor exception."""

    class BrokenOptimizer:
        """Simulate a constructor rejected by the installed backend ABI."""

        def __init__(self, groups: object, **kwargs: object) -> None:
            raise TypeError("incompatible backend constructor")

    fake_module = SimpleNamespace(
        __version__="0.49.2",
        cextension=SimpleNamespace(
            lib=SimpleNamespace(compiled_with_cuda=True),
        ),
        optim=SimpleNamespace(Adam8bit=BrokenOptimizer),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    with pytest.raises(
        training.OptimizerSetupError,
        match="could not initialize adam8bit",
    ) as raised:
        build_optimizer(
            GroupFixture(),
            name="adam8bit",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
            device=torch.device("cuda"),
        )
    assert isinstance(raised.value.__cause__, TypeError)


def test_bitsandbytes_optimizer_does_not_translate_terminal_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch converting A2V2 terminal failures into recoverable setup errors."""

    terminal = training.DistributedOptimizerStepError("terminal optimizer state")

    class TerminalOptimizer:
        """Raise an existing A2V2 terminal exception during construction."""

        def __init__(self, groups: object, **kwargs: object) -> None:
            raise terminal

    fake_module = SimpleNamespace(
        __version__="0.49.2",
        cextension=SimpleNamespace(
            lib=SimpleNamespace(compiled_with_cuda=True),
        ),
        optim=SimpleNamespace(Adam8bit=TerminalOptimizer),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _: fake_module)

    with pytest.raises(training.DistributedOptimizerStepError) as raised:
        build_optimizer(
            GroupFixture(),
            name="adam8bit",
            learning_rate=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
            device=torch.device("cuda"),
        )
    assert raised.value is terminal


def test_bitsandbytes_extra_uses_the_pinned_compatible_range() -> None:
    """Catch packaging that omits or broadens the validated optional dependency."""

    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["optional-dependencies"]["bnb"] == [
        "bitsandbytes>=0.49,<0.50"
    ]


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


def test_constant_weight_decay_schedule_preserves_legacy_optimizer_state_exactly() -> None:
    """Catch default scheduling that adds group metadata or checkpoint state."""

    model = GroupFixture()
    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
    )
    before = deepcopy(optimizer.state_dict())

    scheduler = training.build_weight_decay_scheduler(
        optimizer,
        schedule="constant",
        weight_decay_end=None,
        max_updates=4,
    )

    assert scheduler is None
    assert optimizer.state_dict() == before
    assert type(optimizer) is training.FairseqCompatibleAdam
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0]
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]] == [
        id(model.linear.weight)
    ]
    assert [id(parameter) for parameter in optimizer.param_groups[1]["params"]] == [
        id(model.alibi_scale),
        id(model.linear.bias),
        id(model.norm.weight),
        id(model.norm.bias),
        id(model.activation.p_swish_alpha),
        id(model.activation.p_swish_beta),
    ]


def test_cosine_weight_decay_applies_start_midpoint_end_clamp_and_zero_group() -> None:
    """Catch a wrong cosine clock, endpoint, clamp, or decayed exemption group."""

    model = GroupFixture()
    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.2,
    )
    scheduler = training.build_weight_decay_scheduler(
        optimizer,
        schedule="cosine",
        weight_decay_end=0.02,
        max_updates=4,
    )

    assert scheduler is not None
    assert scheduler.last_update == 0
    assert [group["initial_weight_decay"] for group in optimizer.param_groups] == [0.2, 0.0]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.2, 0.0]
    assert scheduler.step_update(2) == pytest.approx(0.11)
    assert [group["weight_decay"] for group in optimizer.param_groups] == pytest.approx([0.11, 0.0])
    assert scheduler.step_update(4) == pytest.approx(0.02)
    assert [group["weight_decay"] for group in optimizer.param_groups] == pytest.approx([0.02, 0.0])
    assert scheduler.step_update(9) == pytest.approx(0.02)
    assert [group["weight_decay"] for group in optimizer.param_groups] == pytest.approx([0.02, 0.0])


def test_cosine_weight_decay_omitted_end_defaults_to_zero() -> None:
    """Catch an opt-in cosine schedule that keeps fixed decay without an endpoint."""

    model = GroupFixture()
    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.2,
    )
    scheduler = training.build_weight_decay_scheduler(
        optimizer,
        schedule="cosine",
        weight_decay_end=None,
        max_updates=4,
    )

    assert scheduler is not None
    assert scheduler.step_update(2) == pytest.approx(0.1)
    assert [group["weight_decay"] for group in optimizer.param_groups] == pytest.approx([0.1, 0.0])
    assert scheduler.step_update(4) == pytest.approx(0.0)
    assert [group["weight_decay"] for group in optimizer.param_groups] == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    "malformed",
    (
        {},
        {"last_update": 0, "extra": None},
        {"last_update": True},
        {"last_update": -1},
        {"last_update": 1.5},
    ),
)
def test_cosine_weight_decay_rejects_malformed_state_without_mutation(
    malformed: dict[str, object],
) -> None:
    """Catch permissive state parsing or validate-after-install restoration."""

    model = GroupFixture()
    optimizer = build_optimizer(
        model,
        name="adam",
        learning_rate=1e-3,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.2,
    )
    scheduler = training.build_weight_decay_scheduler(
        optimizer,
        schedule="cosine",
        weight_decay_end=0.02,
        max_updates=4,
    )
    assert scheduler is not None
    scheduler.step_update(2)
    before_state = scheduler.state_dict()
    before_decay = [group["weight_decay"] for group in optimizer.param_groups]

    with pytest.raises(training.CheckpointError, match="weight-decay scheduler"):
        scheduler.load_state_dict(malformed)

    assert scheduler.state_dict() == before_state
    assert [group["weight_decay"] for group in optimizer.param_groups] == before_decay


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
