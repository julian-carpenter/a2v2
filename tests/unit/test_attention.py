"""Isolate ALiBi construction and combined-QKV self-attention. Hand calculations make head
slopes, score masking, and projection behavior visible without running the full encoder."""

import torch
from torch.nn import functional as F

from a2v2.model import MultiheadAttention, alibi_bias, alibi_slopes


def test_alibi_slopes_and_symmetric_bias() -> None:
    """Check ALiBi slopes and symmetric bias."""
    slopes = alibi_slopes(2)
    assert torch.allclose(slopes, torch.tensor([1 / 16, 1 / 256]))
    bias = alibi_bias(2, 3)
    expected_distance = -torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]]).float()
    assert bias.shape == (2, 3, 3)
    assert torch.allclose(bias[0], expected_distance / 16)
    assert torch.allclose(bias[1], expected_distance / 256)


def test_attention_matches_direct_combined_qkv_calculation() -> None:
    """Check attention matches direct combined QKV calculation."""
    torch.manual_seed(3)
    layer = MultiheadAttention(8, 2, qkv_bias=True, attention_dropout=0.0, projection_dropout=0.0)
    value = torch.randn(2, 5, 8)
    bias = alibi_bias(2, 5).unsqueeze(0).expand(2, -1, -1, -1)

    output = layer(value, alibi=bias)

    qkv = layer.qkv(value).reshape(2, 5, 3, 2, 4).permute(2, 0, 3, 1, 4)
    query, key, projected_value = qkv.unbind(0)
    weights = (query * (4**-0.5)) @ key.transpose(-2, -1)
    weights = (weights + bias).softmax(dim=-1, dtype=torch.float32)
    reference = (weights @ projected_value).transpose(1, 2).reshape(2, 5, 8)
    reference = F.linear(reference, layer.proj.weight, layer.proj.bias)
    assert torch.allclose(output, reference, atol=1e-6)


def test_attention_padding_mask_blocks_keys() -> None:
    """Check attention padding mask blocks keys."""
    torch.manual_seed(11)
    layer = MultiheadAttention(8, 2).eval()
    value = torch.randn(1, 4, 8)
    padding = torch.tensor([[False, False, False, True]])
    changed = value.clone()
    changed[:, -1] = 10_000
    assert torch.allclose(
        layer(value, padding_mask=padding)[:, :-1],
        layer(changed, padding_mask=padding)[:, :-1],
        atol=1e-5,
    )


def test_attention_rejects_invalid_dimensions() -> None:
    """Check attention rejects invalid dimensions."""
    try:
        MultiheadAttention(10, 3)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("invalid head dimension was accepted")
