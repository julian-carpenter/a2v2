"""Test small framework replacements used throughout the model. The suite covers float32
normalization, gradient scaling, padding, transposition, P-Swish, stochastic depth, and
BERT initialization."""

import torch
from torch import nn

from a2v2.model import (
    DropPath,
    GradMultiply,
    PSwish,
    SamePad,
    TransposeLast,
    init_bert_params,
)
from a2v2.model import Fp32GroupNorm, Fp32LayerNorm


def test_fp32_layer_norm_restores_input_dtype() -> None:
    """Check FP32 layer norm restores input dtype."""
    layer = Fp32LayerNorm(4)
    value = torch.randn(2, 3, 4, dtype=torch.float64)
    output = layer(value)
    expected = torch.nn.functional.layer_norm(
        value.float(), (4,), layer.weight.float(), layer.bias.float(), layer.eps
    ).to(value.dtype)
    assert output.dtype == value.dtype
    assert torch.allclose(output, expected)


def test_fp32_group_norm_restores_input_dtype() -> None:
    """Check FP32 group norm restores input dtype."""
    layer = Fp32GroupNorm(2, 4)
    value = torch.randn(2, 4, 7, dtype=torch.float64)
    assert layer(value).dtype == torch.float64


def test_grad_multiply_changes_only_backward() -> None:
    """Check grad multiply changes only backward."""
    value = torch.tensor([2.0], requires_grad=True)
    output = GradMultiply.apply(value, 0.25)
    assert output.item() == 2.0
    output.square().sum().backward()
    assert value.grad.item() == 1.0


def test_same_pad_and_transpose_last() -> None:
    """Check same pad and transpose last."""
    value = torch.arange(12).view(1, 3, 4)
    assert SamePad(4)(value).shape == (1, 3, 3)
    assert torch.equal(TransposeLast()(value), value.transpose(-1, -2))
    assert torch.equal(TransposeLast(-3)(value), value.transpose(-1, -3))


def test_pswish_starts_as_identity_and_has_legacy_parameter_names() -> None:
    """Check pswish starts as identity and has legacy parameter names."""
    layer = PSwish(3)
    value = torch.randn(2, 3, 5)
    assert torch.allclose(layer(value), value)
    assert set(layer.state_dict()) == {"p_swish_alpha", "p_swish_beta"}


def test_drop_path_is_identity_in_eval_and_drops_whole_samples() -> None:
    """Check drop path is identity in eval and drops whole samples."""
    value = torch.ones(64, 3, 2)
    layer = DropPath(0.5)
    assert torch.equal(layer.eval()(value), value)
    torch.manual_seed(2)
    output = layer.train()(value)
    assert set(output.unique().tolist()) <= {0.0, 2.0}
    assert torch.equal(output[:, :1, :1].expand_as(output), output)


def test_bert_initialization_handles_linear_and_embedding() -> None:
    """Check bert initialization handles linear and embedding."""
    torch.manual_seed(4)
    linear = nn.Linear(128, 64)
    embedding = nn.Embedding(20, 16, padding_idx=0)
    linear.apply(init_bert_params)
    embedding.apply(init_bert_params)
    assert torch.equal(linear.bias, torch.zeros_like(linear.bias))
    assert linear.weight.std().item() < 0.03
    assert torch.equal(embedding.weight[0], torch.zeros_like(embedding.weight[0]))

