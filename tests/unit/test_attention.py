"""Isolate positional attention math and combined-QKV self-attention.

Hand calculations make rotary phases, ALiBi slopes, score masking, and
projection behavior visible without running the full encoder.
"""

import math
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

import a2v2.model as model_module
from a2v2.model import MultiheadAttention, alibi_bias, alibi_slopes


def test_rope_matches_hand_calculated_two_dimensional_rotations() -> None:
    """Rotate Q and K by their original scalar frame IDs."""

    query = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])
    key = torch.tensor([[[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]])
    position_ids = torch.tensor([[0, 1, 3]])
    expected_query = torch.tensor(
        [[[[1.0, 0.0], [math.cos(1.0), math.sin(1.0)], [math.cos(3.0), math.sin(3.0)]]]]
    )
    expected_key = torch.tensor(
        [[[[0.0, 1.0], [-math.sin(1.0), math.cos(1.0)], [-math.sin(3.0), math.cos(3.0)]]]]
    )

    actual_query, actual_key = model_module.apply_rotary_position_embedding(
        query,
        key,
        position_ids,
        theta=10_000.0,
    )

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_rope_preserves_qk_shape_dtype_device_and_gradients(dtype: torch.dtype) -> None:
    """Keep rotary attention tensors in the projection's numerical frame."""

    query = torch.randn(2, 3, 4, 6, dtype=dtype, requires_grad=True)
    key = torch.randn(2, 3, 4, 6, dtype=dtype, requires_grad=True)
    position_ids = torch.tensor([[0, 3, 1, 5], [7, 2, 4, 6]])

    rotated_query, rotated_key = model_module.apply_rotary_position_embedding(
        query,
        key,
        position_ids,
        theta=5_000.0,
    )
    (rotated_query.float().square().sum() + rotated_key.float().square().sum()).backward()

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert rotated_query.dtype == dtype
    assert rotated_key.dtype == dtype
    assert rotated_query.device == query.device
    assert rotated_key.device == key.device
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert key.grad is not None and torch.isfinite(key.grad).all()


def test_rope_rejects_odd_attention_head_dimension() -> None:
    """Reject a head width that cannot form rotary coordinate pairs."""

    query = torch.randn(1, 2, 3, 5)
    key = torch.randn(1, 2, 3, 5)

    with pytest.raises(ValueError, match="head dimension must be even"):
        model_module.apply_rotary_position_embedding(
            query,
            key,
            torch.tensor([[0, 1, 2]]),
            theta=10_000.0,
        )


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


def test_rope_attention_rejects_alibi_bias() -> None:
    """Reject simultaneous rotary phases and an additive ALiBi prior."""

    layer = MultiheadAttention(
        8,
        2,
        position_encoding="rope",
        attention_backend="manual",
    )

    with pytest.raises(ValueError, match=r"RoPE.*ALiBi.*mutually exclusive"):
        layer(
            torch.randn(1, 4, 8),
            alibi=alibi_bias(2, 4),
            position_ids=torch.tensor([[0, 3, 1, 2]]),
        )


def _attention_outputs_and_gradients(
    layer: MultiheadAttention,
    value: torch.Tensor,
    upstream_gradient: torch.Tensor,
    *,
    training: bool,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Return output, input gradient, and ordered parameter gradients."""

    layer.train(training)
    sample = value.detach().clone().requires_grad_(True)
    torch.manual_seed(seed)
    output = layer(sample)
    gradients = torch.autograd.grad(
        (output * upstream_gradient).sum(),
        (sample, *layer.parameters()),
    )
    return (output.detach(), *(gradient.detach() for gradient in gradients))


def _mean_attention_outputs_and_gradients(
    layer: MultiheadAttention,
    value: torch.Tensor,
    upstream_gradient: torch.Tensor,
    *,
    draws: int,
    seed_offset: int,
) -> tuple[torch.Tensor, ...]:
    """Average train-mode dropout outputs and gradients over independent draws."""

    totals: list[torch.Tensor] | None = None
    for draw in range(draws):
        result = _attention_outputs_and_gradients(
            layer,
            value,
            upstream_gradient,
            training=True,
            seed=seed_offset + draw,
        )
        if totals is None:
            totals = [torch.zeros_like(item) for item in result]
        for total, item in zip(totals, result, strict=True):
            total.add_(item)
    assert totals is not None
    return tuple(total / draws for total in totals)


@pytest.mark.parametrize("case", ("no_padding", "key_padding", "rope"))
def test_sdpa_matches_manual_attention_forward_and_backward(case: str) -> None:
    """Match the reference path for padding and rotary coordinates."""

    torch.manual_seed(29)
    position_encoding = "rope" if case == "rope" else "none"
    manual = MultiheadAttention(
        12,
        3,
        qkv_bias=True,
        qk_scale=0.37,
        attention_dropout=0.0,
        projection_dropout=0.0,
        position_encoding=position_encoding,
        attention_backend="manual",
    ).eval()
    sdpa = deepcopy(manual)
    sdpa.attention_backend = "sdpa"
    manual_value = torch.randn(2, 5, 12, requires_grad=True)
    sdpa_value = manual_value.detach().clone().requires_grad_(True)
    padding = None
    if case == "key_padding":
        padding = torch.tensor(
            [[False, False, False, False, False], [False, False, False, True, True]]
        )
    position_ids = None
    if case == "rope":
        position_ids = torch.tensor([[0, 4, 1, 7, 2], [8, 3, 6, 2, 5]])

    manual_output = manual(manual_value, padding, position_ids=position_ids)
    sdpa_output = sdpa(sdpa_value, padding, position_ids=position_ids)
    manual_output.float().square().sum().backward()
    sdpa_output.float().square().sum().backward()

    torch.testing.assert_close(sdpa_output, manual_output, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(sdpa_value.grad, manual_value.grad, rtol=3e-5, atol=3e-6)
    for (actual_name, actual), (expected_name, expected) in zip(
        sdpa.named_parameters(),
        manual.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        assert actual.grad is not None
        assert expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=4e-5, atol=4e-6)


@pytest.mark.parametrize(("training", "expected_dropout"), ((False, 0.0), (True, 0.25)))
def test_sdpa_branch_routes_configured_training_dropout(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
    expected_dropout: float,
) -> None:
    """Call real SDPA with configured train dropout and zero eval dropout."""

    original = F.scaled_dot_product_attention
    observed: list[tuple[float, float | None]] = []

    def spy(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """Record SDPA policy arguments while executing the real operator."""

        observed.append((dropout_p, scale))
        return original(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    layer = MultiheadAttention(
        8,
        2,
        qk_scale=0.41,
        attention_dropout=0.25,
        attention_backend="sdpa",
    )
    layer.train(training)

    layer(torch.randn(2, 4, 8))

    assert observed == [(expected_dropout, 0.41)]


def test_attention_dropout_eval_matches_manual_forward_and_backward() -> None:
    """Disable attention dropout in eval for both implementations."""

    torch.manual_seed(77)
    manual = MultiheadAttention(
        8,
        2,
        qkv_bias=True,
        attention_dropout=0.3,
        projection_dropout=0.0,
        attention_backend="manual",
    )
    sdpa = deepcopy(manual)
    sdpa.attention_backend = "sdpa"
    value = torch.randn(2, 4, 8)
    upstream_gradient = torch.randn(2, 4, 8)

    manual_first = _attention_outputs_and_gradients(
        manual, value, upstream_gradient, training=False, seed=101
    )
    manual_second = _attention_outputs_and_gradients(
        manual, value, upstream_gradient, training=False, seed=102
    )
    sdpa_first = _attention_outputs_and_gradients(
        sdpa, value, upstream_gradient, training=False, seed=201
    )
    sdpa_second = _attention_outputs_and_gradients(
        sdpa, value, upstream_gradient, training=False, seed=202
    )

    for first, second in zip(manual_first, manual_second, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for first, second in zip(sdpa_first, sdpa_second, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for actual, expected in zip(sdpa_first, manual_first, strict=True):
        torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_attention_dropout_training_matches_in_expectation_forward_and_backward() -> None:
    """Match unbiased dropout outputs and gradients across independent RNG streams."""

    torch.manual_seed(77)
    manual = MultiheadAttention(
        8,
        2,
        qkv_bias=True,
        attention_dropout=0.3,
        projection_dropout=0.0,
        attention_backend="manual",
    )
    sdpa = deepcopy(manual)
    sdpa.attention_backend = "sdpa"
    value = torch.randn(2, 4, 8)
    upstream_gradient = torch.randn(2, 4, 8)
    reference = _attention_outputs_and_gradients(
        manual,
        value,
        upstream_gradient,
        training=False,
        seed=1,
    )

    manual_first = _attention_outputs_and_gradients(
        manual, value, upstream_gradient, training=True, seed=1_000
    )
    manual_second = _attention_outputs_and_gradients(
        manual, value, upstream_gradient, training=True, seed=1_001
    )
    sdpa_first = _attention_outputs_and_gradients(
        sdpa, value, upstream_gradient, training=True, seed=5_000
    )
    sdpa_second = _attention_outputs_and_gradients(
        sdpa, value, upstream_gradient, training=True, seed=5_001
    )
    assert not torch.equal(manual_first[0], manual_second[0])
    assert not torch.equal(manual_first[1], manual_second[1])
    assert not torch.equal(sdpa_first[0], sdpa_second[0])
    assert not torch.equal(sdpa_first[1], sdpa_second[1])

    manual_mean = _mean_attention_outputs_and_gradients(
        manual,
        value,
        upstream_gradient,
        draws=512,
        seed_offset=1_000,
    )
    sdpa_mean = _mean_attention_outputs_and_gradients(
        sdpa,
        value,
        upstream_gradient,
        draws=512,
        seed_offset=5_000,
    )

    for actual in (*manual_mean, *sdpa_mean):
        assert torch.isfinite(actual).all()
    for actual, expected in zip(manual_mean, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.12)
    for actual, expected in zip(sdpa_mean, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.12)
    for actual, expected in zip(sdpa_mean, manual_mean, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.12)


@pytest.mark.parametrize("attention_backend", ("sdpa", "flash"))
def test_fused_attention_rejects_alibi(attention_backend: str) -> None:
    """Keep dense ALiBi on the exact manual implementation."""

    with pytest.raises(ValueError, match=r"ALiBi.*manual"):
        MultiheadAttention(
            8,
            2,
            position_encoding="alibi",
            attention_backend=attention_backend,
        )


def test_strict_flash_fails_with_actionable_context_on_cpu() -> None:
    """Fail instead of falling back when FlashAttention cannot run."""

    layer = MultiheadAttention(8, 2, attention_backend="flash").eval()

    with pytest.raises(
        RuntimeError,
        match=(
            r"strict FlashAttention failed.*backend=flash.*dtype=torch.float32"
            r".*device=cpu.*head_dimension=4.*length=5.*padding=False"
        ),
    ):
        layer(torch.randn(2, 5, 8))
