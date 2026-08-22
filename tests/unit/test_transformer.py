"""Test the archived pre-norm and post-norm Transformer residual order. The suite also
checks that the stack returns one teacher-target tensor for each block that executes."""

from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest
import torch

import a2v2.model as model_module
from a2v2.model import TransformerBlock, TransformerStack
from a2v2.model import alibi_bias


def test_post_norm_block_matches_official_residual_sequence() -> None:
    """Check post norm block matches official residual sequence."""
    torch.manual_seed(5)
    block = TransformerBlock(
        8, 2, layer_norm_first=False, ffn_targets=True,
        dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
        post_mlp_dropout=0.0,
    ).eval()
    value = torch.randn(2, 5, 8)
    bias = alibi_bias(2, 5).unsqueeze(0).expand(2, -1, -1, -1)

    output, target = block(value, alibi=bias)

    attended = value + block.attn(value, alibi=bias)
    residual = block.norm1(attended)
    expected_target = block.mlp(residual)
    expected = block.norm2(residual + block.post_mlp_dropout(expected_target))
    assert torch.allclose(target, expected_target)
    assert torch.allclose(output, expected)


def test_pre_norm_block_preserves_legacy_sequence() -> None:
    """Check pre norm block preserves legacy sequence."""
    torch.manual_seed(6)
    block = TransformerBlock(
        8, 2, layer_norm_first=True, ffn_targets=False,
        dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
        post_mlp_dropout=0.0,
    ).eval()
    value = torch.randn(1, 4, 8)
    attended = value + block.attn(block.norm1(value))
    mlp = block.mlp(block.norm2(attended))
    expected = mlp + mlp
    output, target = block(value)
    assert torch.allclose(output, expected)
    assert torch.allclose(target, expected)


def test_stack_returns_one_target_per_executed_layer() -> None:
    """Check stack returns one target per executed layer."""
    stack = TransformerStack(8, 2, depth=3, layerdrop=0.0).eval()
    output, layers = stack(torch.randn(2, 7, 8))
    assert output.shape == (2, 7, 8)
    assert len(layers) == 3
    assert all(layer.shape == output.shape for layer in layers)


@pytest.mark.parametrize("checkpoint_activations", (False, True))
def test_stack_passes_same_position_ids_to_every_block_and_recomputation(
    checkpoint_activations: bool,
) -> None:
    """Preserve explicit frame coordinates through ordinary and checkpointed blocks."""

    stack = TransformerStack(
        8,
        2,
        depth=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        post_mlp_dropout=0.0,
        checkpoint_activations=checkpoint_activations,
    ).train()
    position_ids = torch.tensor([[4, 1, 7, 2], [8, 3, 6, 0]])
    observed: list[torch.Tensor] = []
    handles = [
        block.register_forward_pre_hook(
            lambda _module, args: observed.append(args[3].detach().clone())
        )
        for block in stack.blocks
    ]
    value = torch.randn(2, 4, 8, requires_grad=True)

    output, targets = stack(value, position_ids=position_ids)
    (output.square().sum() + sum(target.square().sum() for target in targets)).backward()
    for handle in handles:
        handle.remove()

    assert len(observed) >= len(stack.blocks)
    assert all(torch.equal(actual, position_ids) for actual in observed)


@pytest.mark.parametrize(
    ("enabled", "training", "with_grad", "expected_calls"),
    [
        (True, True, True, 2),
        (False, True, True, 0),
        (True, False, True, 0),
        (True, True, False, 0),
    ],
)
def test_stack_checkpoints_only_training_grad_blocks(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    training: bool,
    with_grad: bool,
    expected_calls: int,
) -> None:
    """Gate recomputation on the option, training mode, and autograd."""

    calls: list[dict[str, object]] = []

    def checkpoint_spy(function: object, *args: object, **kwargs: object) -> object:
        """Record checkpoint options while executing the intercepted block."""

        calls.append(dict(kwargs))
        return function(*args)  # type: ignore[operator]

    monkeypatch.setattr(model_module, "activation_checkpoint", checkpoint_spy)
    stack = TransformerStack(
        8,
        2,
        depth=2,
        checkpoint_activations=enabled,
    )
    stack.train(training)
    value = torch.randn(2, 5, 8, requires_grad=with_grad)
    context = nullcontext() if with_grad else torch.no_grad()

    with context:
        stack(value)

    assert len(calls) == expected_calls
    assert all(
        call == {"use_reentrant": False, "preserve_rng_state": True}
        for call in calls
    )


def test_checkpointed_stack_matches_stochastic_forward_backward_and_rng() -> None:
    """Match ordinary block values, gradients, and post-backward RNG state."""

    torch.manual_seed(41)
    ordinary = TransformerStack(
        8,
        2,
        depth=2,
        dropout=0.2,
        attention_dropout=0.2,
        activation_dropout=0.2,
        post_mlp_dropout=0.2,
        drop_path_rates=(0.1, 0.2),
        input_dropout=0.2,
        layerdrop=0.0,
    ).train()
    checkpointed = deepcopy(ordinary)
    checkpointed.checkpoint_activations = True
    base_value = torch.randn(2, 5, 8)
    ordinary_value = base_value.clone().requires_grad_(True)
    checkpointed_value = base_value.clone().requires_grad_(True)
    padding = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, False, True],
    ])
    bias = alibi_bias(2, 5).unsqueeze(0).expand(2, -1, -1, -1)
    ordinary_scale = torch.ones(2, 1, 2, 1, 1, requires_grad=True)
    checkpointed_scale = ordinary_scale.detach().clone().requires_grad_(True)

    def run(
        stack: TransformerStack,
        value: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Run one stack from a fixed RNG state and capture its results."""

        torch.manual_seed(314)
        output, targets = stack(value * 1.0, padding, bias, scale)
        loss = output.square().sum() + sum(
            target.square().sum() for target in targets
        )
        loss.backward()
        return (
            output.detach().clone(),
            [target.detach().clone() for target in targets],
            torch.get_rng_state().clone(),
        )

    ordinary_output, ordinary_targets, ordinary_rng = run(
        ordinary,
        ordinary_value,
        ordinary_scale,
    )
    checkpointed_output, checkpointed_targets, checkpointed_rng = run(
        checkpointed,
        checkpointed_value,
        checkpointed_scale,
    )

    torch.testing.assert_close(checkpointed_output, ordinary_output, rtol=0, atol=0)
    assert len(checkpointed_targets) == len(ordinary_targets)
    for actual, expected in zip(checkpointed_targets, ordinary_targets, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        checkpointed_value.grad,
        ordinary_value.grad,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        checkpointed_scale.grad,
        ordinary_scale.grad,
        rtol=0,
        atol=0,
    )
    for (actual_name, actual), (expected_name, expected) in zip(
        checkpointed.named_parameters(),
        ordinary.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        assert actual.grad is not None
        assert expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert torch.equal(checkpointed_rng, ordinary_rng)


def test_checkpoint_backward_does_not_repeat_numpy_layerdrop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw each NumPy layerdrop decision once outside recomputation."""

    draws = iter((0.1, 0.9, 0.2))
    observed: list[float] = []

    def random_draw() -> float:
        """Return and record the next deterministic NumPy layerdrop draw."""

        value = next(draws)
        observed.append(value)
        return value

    monkeypatch.setattr(np.random, "random", random_draw)
    stack = TransformerStack(
        8,
        2,
        depth=3,
        layerdrop=0.5,
        checkpoint_activations=True,
    ).train()
    value = torch.randn(2, 5, 8, requires_grad=True)

    output, targets = stack(value)
    loss = output.square().sum() + sum(
        target.square().sum() for target in targets
    )
    loss.backward()

    assert observed == [0.1, 0.9, 0.2]
    assert len(targets) == 1
