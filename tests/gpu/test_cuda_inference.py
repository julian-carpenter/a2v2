"""Exercise long-recording inference on a real CUDA device. The tests protect segment
concatenation, stereo handling, partial window bounds, and shaped empty outputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from a2v2.config import config_to_dict, load_config
from a2v2.workflows import InferenceRunner
from a2v2.model import Animal2VecFineTuningModel
from a2v2.training import capture_rng_state, save_checkpoint
from a2v2 import workflows


ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("head", ["frame", "cls"])
def test_strict_flash_validation_uses_half_precision(
    cuda_device: torch.device, monkeypatch: pytest.MonkeyPatch, head: str,
) -> None:
    """Catch full-precision validation failing at the first scheduled update."""
    overrides = (
        "model.position_encoding=rope", "model.attention_backend=flash",
        "model.use_cls_token=true", "model.modalities.audio.use_alibi_encoder=false",
    )
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml", overrides=overrides)
    config = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml", overrides=(
        *overrides, f"model.classification_head={head}",
        "dataset.required_batch_size_multiple=1", "dataset.num_workers=0",
    ))
    config = replace(config, common=replace(config.common, fp16=True))
    model = Animal2VecFineTuningModel.from_config(config, pretrained_config=pretrain).to(cuda_device)

    class ValidationDataset(torch.utils.data.Dataset):
        """Supply one real collatable labeled recording."""
        sizes = (64,)

        def __len__(self) -> int:
            """Return the fixture recording count."""
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            """Return waveform and frame targets for actual model evaluation."""
            return {"id": index, "source": torch.randn(64), "target": torch.zeros(16, 2), "path": "sample.wav"}

    monkeypatch.setattr(workflows, "_make_dataset", lambda *_: ValidationDataset())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        metrics = workflows._validate(model, config, device=cuda_device, update=10000)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert model.training
    assert any("_scaled_dot_product_flash_attention" in event.key for event in profile.key_averages())


def _checkpoint(path: Path, *, flash: bool = False) -> Path:
    """Create the tiny native checkpoint fixture required by this module."""
    overrides = (
        "model.position_encoding=rope", "model.attention_backend=flash",
        "model.modalities.audio.use_alibi_encoder=false",
    ) if flash else ()
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml", overrides=overrides)
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml", overrides=overrides)
    model = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([8.0, -8.0]))
    save_checkpoint(path, {
        "format_version": 1,
        "stage": "finetune",
        "config": {
            "active": config_to_dict(finetune),
            "pretrained": config_to_dict(pretrain),
        },
        "model": model.state_dict(),
        "teacher": None,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "update": 2,
        "epoch": 1,
        "batch_in_epoch": 0,
        "rng_state": capture_rng_state(),
        "sampler_state": None,
        "best_metric": None,
    })
    return path


@pytest.mark.parametrize("flash", [False, True])
def test_chunked_stereo_inference_runs_on_cuda_and_bounds_final_segment(
    cuda_device: torch.device,
    tmp_path: Path,
    flash: bool,
) -> None:
    """Check chunked stereo inference runs on CUDA and bounds final segment."""
    runner = InferenceRunner(
        _checkpoint(tmp_path / "finetuned.pt", flash=flash),
        device=cuda_device,
    )
    assert next(runner.model.parameters()).device == cuda_device

    sample_rate = 16_000
    duration = 1.1
    time = torch.arange(round(sample_rate * duration)) / sample_rate
    stereo = torch.stack((torch.sin(2 * torch.pi * 440 * time), torch.zeros_like(time)))
    result = runner.run_tensor(
        stereo,
        sample_rate,
        channel=0,
        segment_seconds=0.4,
        threshold=0.5,
        event_method="avg",
        fusion_window_seconds=0.01,
    )

    assert result.probabilities.device.type == "cpu"
    assert result.embeddings.device.type == "cpu"
    assert result.probabilities.shape[1] == 2
    assert result.embeddings.shape[1] == 16
    assert len(result.timestamps) == len(result.probabilities)
    assert torch.isfinite(result.probabilities).all()
    assert torch.isfinite(result.embeddings).all()
    assert torch.isfinite(result.timestamps).all()
    assert result.timestamps[-1] < duration
    assert result.events
    assert max(event.end_seconds for event in result.events) <= duration
    assert all(event.label_index == 0 for event in result.events)
    print(json.dumps({
        "device": str(cuda_device),
        "frames": len(result.timestamps),
        "events": len(result.events),
        "last_timestamp": float(result.timestamps[-1]),
        "duration": duration,
    }, sort_keys=True))


def test_empty_recording_returns_shaped_empty_cpu_outputs_on_cuda_runner(
    cuda_device: torch.device,
    tmp_path: Path,
) -> None:
    """Check empty recording returns shaped empty CPU outputs on CUDA runner."""
    runner = InferenceRunner(
        _checkpoint(tmp_path / "finetuned.pt"),
        device=cuda_device,
    )
    result = runner.run_tensor(torch.empty(0), 16_000)

    assert result.probabilities.shape == (0, 2)
    assert result.embeddings.shape == (0, 16)
    assert result.timestamps.shape == (0,)
    assert result.events == ()
    assert result.probabilities.device.type == "cpu"
    assert result.embeddings.device.type == "cpu"
