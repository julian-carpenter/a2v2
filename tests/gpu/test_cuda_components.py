"""Check CUDA implementations of isolated numerical building blocks. The suite compares CPU
and GPU values or gradients where exact device agreement is meaningful, and checks
finite AMP behavior where half precision requires tolerance-based evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from a2v2.config import load_config
from a2v2.model import AudioEncoder
from a2v2.model import MultiheadAttention, alibi_bias
from a2v2.model import ConvDecoder
from a2v2.model import masks_for_cloned_batch
from a2v2.model import SincConv1d
from a2v2.model import mix_waveforms


ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


def _report_error(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int | str]:
    """Summarize absolute and relative tensor error for assertion diagnostics."""
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    difference = (actual_cpu - expected_cpu).abs()
    relative = difference / expected_cpu.abs().clamp_min(1e-12)
    worst_flat = int(difference.reshape(-1).argmax()) if difference.numel() else 0
    report: dict[str, float | int | str] = {
        "name": name,
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "max_rel": float(relative.max()) if relative.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
        "actual_norm": float(torch.linalg.vector_norm(actual_cpu)),
        "expected_norm": float(torch.linalg.vector_norm(expected_cpu)),
        "worst_flat_index": worst_flat,
    }
    print(json.dumps(report, sort_keys=True))
    return report


def test_sinc_fp32_cuda_matches_cpu_forward_and_backward(cuda_device: torch.device) -> None:
    """Check sinc FP32 CUDA matches CPU forward and backward."""
    cpu_layer = SincConv1d(8, 31, sample_rate=8000, padding="same").train()
    cuda_layer = deepcopy(cpu_layer).to(cuda_device).train()
    cpu_waveform = torch.randn(2, 128, requires_grad=True)
    cuda_waveform = cpu_waveform.detach().to(cuda_device).requires_grad_(True)

    cpu_output = cpu_layer(cpu_waveform)
    cuda_output = cuda_layer(cuda_waveform)
    cpu_output.square().mean().backward()
    cuda_output.square().mean().backward()

    _report_error("sinc.output.fp32", cuda_output, cpu_output)
    _report_error("sinc.input_grad.fp32", cuda_waveform.grad, cpu_waveform.grad)
    assert torch.allclose(cuda_output.cpu(), cpu_output, atol=1e-5, rtol=1e-4)
    assert torch.allclose(cuda_waveform.grad.cpu(), cpu_waveform.grad, atol=2e-5, rtol=2e-4)
    for name, cpu_parameter in cpu_layer.named_parameters():
        cuda_parameter = dict(cuda_layer.named_parameters())[name]
        assert cpu_parameter.grad is not None
        assert cuda_parameter.grad is not None
        _report_error(f"sinc.{name}.grad.fp32", cuda_parameter.grad, cpu_parameter.grad)
        assert torch.allclose(cuda_parameter.grad.cpu(), cpu_parameter.grad, atol=2e-5, rtol=2e-4)


def test_sinc_fp16_autocast_has_finite_forward_backward_and_update(
    cuda_device: torch.device,
) -> None:
    """Check sinc FP16 autocast has finite forward backward and update."""
    layer = SincConv1d(8, 31, sample_rate=8000, padding="same").to(cuda_device).train()
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
    waveform = torch.randn(2, 128, device=cuda_device, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.float16):
        output = layer(waveform)
        loss = output.square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert waveform.grad is not None and torch.isfinite(waveform.grad).all()
    assert all(parameter.grad is not None for parameter in layer.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())
    scaler.step(optimizer)
    scaler.update()

    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert scaler.get_scale() > 0


def test_frontend_fp32_cuda_matches_cpu_with_padding_and_backward(
    cuda_device: torch.device,
) -> None:
    """Check frontend FP32 CUDA matches CPU with padding and backward."""
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    cpu_encoder = AudioEncoder.from_config(config).eval()
    cuda_encoder = deepcopy(cpu_encoder).to(cuda_device).eval()
    cpu_waveform = torch.randn(2, 64)
    cpu_padding = torch.zeros(2, 64, dtype=torch.bool)
    cpu_padding[1, 60:] = True
    cuda_waveform = cpu_waveform.to(cuda_device).requires_grad_(True)

    with torch.no_grad():
        cpu_output = cpu_encoder(cpu_waveform, cpu_padding)
    cuda_output = cuda_encoder(cuda_waveform, cpu_padding.to(cuda_device))
    cuda_output.x.square().mean().backward()

    _report_error("frontend.local_features.fp32", cuda_output.local_features, cpu_output.local_features)
    _report_error("frontend.context.fp32", cuda_output.x, cpu_output.x)
    assert torch.equal(cuda_output.padding_mask.cpu(), cpu_output.padding_mask)
    assert cuda_output.padding_mask[1, 15]
    assert torch.allclose(cuda_output.local_features.cpu(), cpu_output.local_features, atol=1e-5, rtol=1e-4)
    assert torch.allclose(cuda_output.x.detach().cpu(), cpu_output.x, atol=2e-5, rtol=2e-4)
    assert cuda_waveform.grad is not None and torch.isfinite(cuda_waveform.grad).all()


def test_alibi_attention_cuda_matches_cpu_with_padding_and_backward(
    cuda_device: torch.device,
) -> None:
    """Check ALiBi attention CUDA matches CPU with padding and backward."""
    cpu_layer = MultiheadAttention(16, 4, qkv_bias=True).train()
    cuda_layer = deepcopy(cpu_layer).to(cuda_device).train()
    cpu_value = torch.randn(2, 7, 16, requires_grad=True)
    cuda_value = cpu_value.detach().to(cuda_device).requires_grad_(True)
    padding = torch.tensor(
        [[False, False, False, False, False, False, False], [False, False, False, False, False, True, True]]
    )
    cpu_bias = alibi_bias(4, 7).unsqueeze(0).expand(2, -1, -1, -1)
    cuda_bias = cpu_bias.to(cuda_device)

    cpu_output = cpu_layer(cpu_value, padding, cpu_bias)
    cuda_output = cuda_layer(cuda_value, padding.to(cuda_device), cuda_bias)
    cpu_output.square().mean().backward()
    cuda_output.square().mean().backward()

    _report_error("attention.output.fp32", cuda_output, cpu_output)
    _report_error("attention.input_grad.fp32", cuda_value.grad, cpu_value.grad)
    assert torch.allclose(cuda_output.cpu(), cpu_output, atol=1e-5, rtol=1e-4)
    assert torch.allclose(cuda_value.grad.cpu(), cpu_value.grad, atol=2e-5, rtol=2e-4)


def test_alibi_attention_fp16_autocast_is_finite(cuda_device: torch.device) -> None:
    """Check ALiBi attention FP16 autocast is finite."""
    layer = MultiheadAttention(16, 4, qkv_bias=True).to(cuda_device).train()
    value = torch.randn(2, 7, 16, device=cuda_device, requires_grad=True)
    padding = torch.tensor(
        [[False, False, False, False, False, False, False], [False, False, False, False, False, True, True]],
        device=cuda_device,
    )
    bias = alibi_bias(4, 7, device=cuda_device).unsqueeze(0).expand(2, -1, -1, -1)

    with torch.autocast("cuda", dtype=torch.float16):
        output = layer(value, padding, bias)
        loss = output.square().mean()
    loss.backward()

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_decoder_block_keeps_archived_fp32_layer_norm_output_under_autocast(
    cuda_device: torch.device,
) -> None:
    """Check decoder block keeps archived FP32 layer norm output under autocast."""
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    decoder = ConvDecoder(config.model.audio.decoder, config.model.embed_dim).to(cuda_device).eval()
    block_outputs: list[torch.Tensor] = []
    handle = decoder.blocks[0].register_forward_hook(
        lambda _module, _inputs, output: block_outputs.append(output)
    )

    with torch.autocast("cuda", dtype=torch.float16):
        output = decoder(torch.randn(2, 8, config.model.embed_dim, device=cuda_device))
    handle.remove()

    assert len(block_outputs) == 1
    assert block_outputs[0].dtype == torch.float32
    assert output.dtype == torch.float16


def test_mixup_and_clone_masks_replay_exactly_on_cuda(cuda_device: torch.device) -> None:
    """Check mixup and clone masks replay exactly on CUDA."""
    waveform = torch.randn(4, 256, device=cuda_device)
    ratios = torch.tensor([0.7], device=cuda_device)
    permutation = torch.tensor([1, 0, 3, 2], device=cuda_device)
    arguments = {
        "strength": 0.5,
        "probability": 1.0,
        "same_ratio": True,
        "gain_mode": "A_weighting",
        "sample_rate": 8000,
        "window_seconds": 0.01,
        "ratios": ratios,
        "permutation": permutation,
    }
    first = mix_waveforms(waveform, **arguments)
    second = mix_waveforms(waveform, **arguments)
    assert torch.equal(first.waveforms, second.waveforms)
    assert torch.equal(first.ratios, second.ratios)
    assert torch.equal(first.permutation, second.permutation)
    assert torch.equal(first.applied, second.applied)

    sample_ids = torch.tensor([10, 11], device=cuda_device)
    first_mask = masks_for_cloned_batch(
        batch_size=2,
        length=16,
        clone_count=3,
        mask_prob=0.5,
        mask_length=2,
        global_seed=7,
        update=4,
        sample_ids=sample_ids,
    )
    second_mask = masks_for_cloned_batch(
        batch_size=2,
        length=16,
        clone_count=3,
        mask_prob=0.5,
        mask_length=2,
        global_seed=7,
        update=4,
        sample_ids=sample_ids,
    )
    assert torch.equal(first_mask, second_mask)
    assert first_mask.shape == (6, 16)
    assert not torch.equal(first_mask[0], first_mask[1])
