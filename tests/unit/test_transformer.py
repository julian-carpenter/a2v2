"""Test the archived pre-norm and post-norm Transformer residual order. The suite also
checks that the stack returns one teacher-target tensor for each block that executes."""

from contextlib import nullcontext
from copy import deepcopy
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import a2v2.model as model_module
from a2v2.model import AudioEncoder, TransformerBlock, TransformerStack
from a2v2.model import alibi_bias


class _ScaleAttention(nn.Module):
    """Deterministic attention branch used to isolate residual arithmetic."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, value: torch.Tensor, *_: object) -> torch.Tensor:
        """Scale the real block input without adding parameters or randomness."""

        return self.scale * value


class _ScaleFFN(nn.Module):
    """Deterministic FFN branch used to isolate residual arithmetic."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale the real intermediate block tensor."""

        return self.scale * value


def _deepscale_branch_std(dimension: int, dropout: float) -> float:
    """Convert the design's branch-weight variance into a standard deviation."""

    variance = math.sqrt((1.0 - dropout) / 2.0) / dimension
    return math.sqrt(variance)


def test_packed_geglu_matches_direct_equation_shapes_and_gradients() -> None:
    """Use one packed affine map, split value/gate, and backpropagate both halves."""

    geglu = model_module.PackedGEGLU(3, 2, dropout=0.0).eval()
    with torch.no_grad():
        geglu.fc1.weight.copy_(torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [-0.2, 0.1, 0.3],
            [0.7, -0.4, 0.2],
        ]))
        geglu.fc1.bias.copy_(torch.tensor([0.05, -0.1, 0.2, -0.3]))
        geglu.fc2.weight.copy_(torch.tensor([
            [0.3, -0.5],
            [0.7, 0.2],
            [-0.4, 0.6],
        ]))
        geglu.fc2.bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
    value = torch.tensor(
        [[[0.2, -0.1, 0.5], [0.7, 0.3, -0.4]]],
        requires_grad=True,
    )

    packed = F.linear(value, geglu.fc1.weight, geglu.fc1.bias)
    direct_value, direct_gate = packed.chunk(2, dim=-1)
    expected = F.linear(
        direct_value * F.gelu(direct_gate),
        geglu.fc2.weight,
        geglu.fc2.bias,
    )
    output = geglu(value)
    output.square().sum().backward()

    assert geglu.fc1.weight.shape == (4, 3)
    assert geglu.fc1.bias.shape == (4,)
    assert geglu.fc2.weight.shape == (3, 2)
    assert len([module for module in geglu.modules() if isinstance(module, nn.Linear)]) == 2
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert value.grad is not None and value.grad.abs().sum() > 0
    assert geglu.fc1.weight.grad is not None
    assert geglu.fc1.weight.grad[:2].abs().sum() > 0
    assert geglu.fc1.weight.grad[2:].abs().sum() > 0
    assert geglu.fc2.weight.grad is not None and geglu.fc2.weight.grad.abs().sum() > 0


def test_packed_geglu_drops_gated_hidden_before_output_projection() -> None:
    """Apply activation dropout to the gated hidden coordinates before fc2."""

    geglu = model_module.PackedGEGLU(2, 3, dropout=1.0).train()
    with torch.no_grad():
        geglu.fc1.weight.fill_(1.0)
        geglu.fc1.bias.fill_(1.0)
        geglu.fc2.weight.fill_(1.0)
        geglu.fc2.bias.copy_(torch.tensor([2.0, 3.0]))
    projected: list[tuple[torch.Tensor, torch.Tensor]] = []
    handle = geglu.fc2.register_forward_hook(
        lambda _module, args, output: projected.append(
            (args[0].detach().clone(), output.detach().clone())
        )
    )

    output = geglu(torch.ones(1, 1, 2))
    handle.remove()

    assert len(projected) == 1
    assert torch.count_nonzero(projected[0][0]) == 0
    torch.testing.assert_close(projected[0][1], torch.tensor([[[2.0, 3.0]]]))
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("layer_norm_first", (False, True))
def test_deepscale_residual_coefficients_scale_both_branches(
    layer_norm_first: bool,
) -> None:
    """Apply lambda to both skips and beta to both branch outputs."""

    total_depth = 8
    block = TransformerBlock(
        4,
        2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        post_mlp_dropout=0.0,
        layer_norm_first=layer_norm_first,
        initialization="deepscale_lm",
        total_depth=total_depth,
    ).eval()
    block.norm1 = nn.Identity()
    block.norm2 = nn.Identity()
    block.attn = _ScaleAttention(2.0)
    block.mlp = _ScaleFFN(3.0)
    value = torch.tensor([[[1.0, -2.0, 0.5, 4.0]]])

    output, target = block(value)

    expected_lambda = math.sqrt(1.0 - 2.0 / total_depth)
    expected_beta = math.sqrt(2.0 / total_depth)
    after_attention = expected_lambda * value + expected_beta * (2.0 * value)
    expected_target = 3.0 * after_attention
    expected_output = (
        expected_lambda * after_attention + expected_beta * expected_target
    )
    assert block.residual_lambda == pytest.approx(expected_lambda)
    assert block.residual_beta == pytest.approx(expected_beta)
    torch.testing.assert_close(target, expected_target)
    torch.testing.assert_close(output, expected_output)


def test_deepscale_rejects_total_encoder_depth_below_two() -> None:
    """Refuse undefined residual coefficients before constructing any block."""

    with pytest.raises(ValueError, match=r"DeepScaleLM.*total.*at least two"):
        TransformerStack(
            8,
            2,
            depth=1,
            initialization="deepscale_lm",
            total_depth=1,
        )


def test_deepscale_initializes_roles_with_branch_dropout_variances() -> None:
    """Initialize packed QKV slices and branch weights from their specified roles."""

    torch.manual_seed(83)
    dimension = 256
    attention_branch_dropout = 0.19
    ffn_branch_dropout = 0.36
    encoder = AudioEncoder(
        ((64, 3, 2),),
        sample_rate=8_000,
        dimension=dimension,
        num_heads=8,
        depth=2,
        prenet_depth=1,
        conv_pos_depth=1,
        conv_pos_width=3,
        conv_pos_groups=16,
        sinc_input=False,
        use_pswish=False,
        mlp_ratio=2.0,
        encoder_dropout=attention_branch_dropout,
        attention_dropout=0.47,
        activation_dropout=0.61,
        post_mlp_dropout=ffn_branch_dropout,
        prenet_dropout=0.25,
        dropout_input=0.0,
        use_alibi=False,
        use_cls_token=True,
        ffn_type="geglu",
        initialization="deepscale_lm",
    )
    blocks = [*encoder.prenet.blocks, *encoder.transformer.blocks]
    qk_std = 1.0 / math.sqrt(dimension)
    attention_std = _deepscale_branch_std(
        dimension,
        attention_branch_dropout,
    )
    ffn_std = _deepscale_branch_std(dimension, ffn_branch_dropout)

    # Each checked matrix has at least 65,536 samples. A Gaussian sample
    # standard deviation then has relative standard error below 0.3%; 2.5%
    # leaves ample deterministic margin while distinguishing every role.
    for block in blocks:
        query, key, projected_value = block.attn.qkv.weight.chunk(3, dim=0)
        assert query.std().item() == pytest.approx(qk_std, rel=0.025)
        assert key.std().item() == pytest.approx(qk_std, rel=0.025)
        assert projected_value.std().item() == pytest.approx(
            attention_std,
            rel=0.025,
        )
        assert block.attn.proj.weight.std().item() == pytest.approx(
            attention_std,
            rel=0.025,
        )
        assert block.mlp.fc1.weight.std().item() == pytest.approx(
            ffn_std,
            rel=0.025,
        )
        assert block.mlp.fc2.weight.std().item() == pytest.approx(
            ffn_std,
            rel=0.025,
        )
        for linear in (block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2):
            assert linear.bias is not None
            assert torch.count_nonzero(linear.bias) == 0
        assert block.residual_lambda == pytest.approx(math.sqrt(1.0 / 3.0))
        assert block.residual_beta == pytest.approx(math.sqrt(2.0 / 3.0))

    # CLS is the only learned embedding at its token position and is followed
    # by prenet dropout p=0.25, so the embedding variance is 1-p. With 256
    # coordinates its sample-std relative error is about 4.4%; 15% is robust.
    assert encoder.cls_token.std().item() == pytest.approx(
        math.sqrt(1.0 - 0.25),
        rel=0.15,
    )
    # The acoustic projection remains under nn.Linear.reset_parameters rather
    # than receiving any Transformer-role distribution.
    projection_std = math.sqrt(1.0 / (3.0 * encoder.project_features.in_features))
    assert encoder.project_features.weight.std().item() == pytest.approx(
        projection_std,
        rel=0.08,
    )


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


@pytest.mark.parametrize("ffn_type", ("mlp", "geglu"))
def test_checkpointed_stack_matches_stochastic_forward_backward_and_rng(
    ffn_type: str,
) -> None:
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
        ffn_type=ffn_type,
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
