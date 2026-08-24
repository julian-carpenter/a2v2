"""Benchmark eager manual attention against strict Flash on one CUDA device.

This is a bounded diagnostic, not a throughput acceptance test.  It measures
the same forward, FP32 square-mean loss, and backward scope for both backends.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import time

import torch

from a2v2.model import MultiheadAttention


def _arguments() -> argparse.Namespace:
    """Parse the bounded benchmark shape and iteration counts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=2_048)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _validate(arguments: argparse.Namespace) -> None:
    """Reject shapes that cannot construct the requested attention module."""

    positive = ("batch", "length", "dimension", "heads", "iterations")
    for name in positive:
        if getattr(arguments, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if arguments.warmups < 0:
        raise ValueError("--warmups must be nonnegative")
    if arguments.dimension % arguments.heads:
        raise ValueError("--dimension must be divisible by --heads")


def _module(
    *,
    dimension: int,
    heads: int,
    backend: str,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> MultiheadAttention:
    """Build one fresh eager attention module with the shared weights."""

    module = MultiheadAttention(
        dimension,
        heads,
        attention_dropout=0.0,
        projection_dropout=0.0,
        position_encoding="none",
        attention_backend=backend,
    )
    module.load_state_dict(deepcopy(state), strict=True)
    return module.to(device=device, dtype=torch.float16).eval()


def _measure(
    *,
    backend: str,
    arguments: argparse.Namespace,
    state: dict[str, torch.Tensor],
    input_value: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    """Measure synchronized forward and backward iterations for one backend."""

    module = _module(
        dimension=arguments.dimension,
        heads=arguments.heads,
        backend=backend,
        state=state,
        device=device,
    )

    def forward_backward() -> tuple[torch.Tensor, torch.Tensor]:
        """Run one fresh-input training objective and retain its tensors."""

        module.zero_grad(set_to_none=True)
        value = input_value.detach().clone().requires_grad_(True)
        output = module(value)
        output.float().square().mean().backward()
        if value.grad is None:
            raise RuntimeError(f"{backend} produced no input gradient")
        return output.detach(), value.grad.detach()

    for _ in range(arguments.warmups):
        forward_backward()
    module.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    output = input_gradient = None
    for _ in range(arguments.iterations):
        output, input_gradient = forward_backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    if output is None or input_gradient is None:
        raise RuntimeError("no measured benchmark iteration ran")
    parameter_gradients = [
        parameter.grad for parameter in module.parameters() if parameter.grad is not None
    ]
    finite = (
        bool(torch.isfinite(output).all())
        and bool(torch.isfinite(input_gradient).all())
        and all(bool(torch.isfinite(gradient).all()) for gradient in parameter_gradients)
    )
    if not finite:
        raise RuntimeError(f"{backend} produced a non-finite output or gradient")

    metrics: dict[str, object] = {
        "backend": backend,
        "kernel_mode": (
            "strict FlashAttention through SDPBackend.FLASH_ATTENTION"
            if backend == "flash"
            else "legacy manual dense attention body"
        ),
        "total_seconds": elapsed,
        "milliseconds_per_iteration": elapsed * 1_000 / arguments.iterations,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "output_norm": float(output.float().norm()),
        "input_gradient_norm": float(input_gradient.float().norm()),
        "finite_output_and_gradients": finite,
    }
    return metrics, output.cpu(), input_gradient.cpu()


def main() -> int:
    """Run the fixed-scope eager comparison and print one JSON observation."""

    arguments = _arguments()
    _validate(arguments)
    if not torch.cuda.is_available():
        raise RuntimeError("the attention benchmark requires CUDA")
    device = torch.device(arguments.device)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")

    torch.manual_seed(1_201)
    torch.cuda.manual_seed_all(1_202)
    reference = MultiheadAttention(
        arguments.dimension,
        arguments.heads,
        attention_dropout=0.0,
        projection_dropout=0.0,
        position_encoding="none",
        attention_backend="manual",
    )
    shared_state = deepcopy(reference.state_dict())
    input_value = torch.randn(
        arguments.batch,
        arguments.length,
        arguments.dimension,
        device=device,
        dtype=torch.float16,
    )

    observations: dict[str, dict[str, object]] = {}
    raw: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for backend in ("manual", "flash"):
        metrics, output, input_gradient = _measure(
            backend=backend,
            arguments=arguments,
            state=shared_state,
            input_value=input_value,
            device=device,
        )
        observations[backend] = metrics
        raw[backend] = output, input_gradient

    output_difference = (raw["manual"][0] - raw["flash"][0]).abs().float()
    gradient_difference = (raw["manual"][1] - raw["flash"][1]).abs().float()
    report = {
        "benchmark": "bounded manual versus strict Flash attention",
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "shape": {
            "batch": arguments.batch,
            "frames": arguments.length,
            "dimension": arguments.dimension,
            "heads": arguments.heads,
            "layers": 1,
        },
        "dtype": "float16",
        "warmup_iterations": arguments.warmups,
        "measured_iterations": arguments.iterations,
        "scope": "eager forward + FP32 square-mean loss + backward",
        "compiler": {"enabled": False, "mode": "eager"},
        "position_encoding": "none",
        "observations": observations,
        "raw_differences": {
            "output_max_abs": float(output_difference.max()),
            "output_mean_abs": float(output_difference.mean()),
            "input_gradient_max_abs": float(gradient_difference.max()),
            "input_gradient_mean_abs": float(gradient_difference.mean()),
        },
        "claim_scope": "one bounded diagnostic; no performance threshold",
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
