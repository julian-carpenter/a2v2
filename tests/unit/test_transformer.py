"""Test the archived pre-norm and post-norm Transformer residual order. The suite also
checks that the stack returns one teacher-target tensor for each block that executes."""

import torch

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

