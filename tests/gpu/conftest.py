"""Shared deterministic CUDA setup for single-GPU pytest modules.

Each GPU test starts from fixed CPU and CUDA seeds, deterministic kernels, and
disabled TF32. The fixture restores global backend flags afterward so running
the tests does not alter another suite's numerical policy.
"""

from __future__ import annotations

from collections.abc import Iterator
import os

import pytest
import torch


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def deterministic_cuda() -> Iterator[None]:
    """Configure deterministic CUDA execution or skip on a CPU-only host."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    matmul_tf32_before = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32_before = torch.backends.cudnn.allow_tf32

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32_before
        torch.backends.cudnn.allow_tf32 = cudnn_tf32_before
        torch.cuda.empty_cache()


@pytest.fixture
def cuda_device() -> torch.device:
    """Provide the first visible CUDA device selected by the test launcher."""

    return torch.device("cuda", 0)
