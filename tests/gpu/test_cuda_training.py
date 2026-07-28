"""Exercise pretraining, fine-tuning, AMP, checkpoint, and overflow behavior on CUDA. The
tests compare complete state trees when exact continuation is required and report
numerical error for device comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from a2v2.config import config_to_dict, load_config
from a2v2.model import Animal2VecFineTuningModel
from a2v2.model import Animal2VecPretrainingModel
from a2v2.training import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from a2v2.training import TrainingEngine
from a2v2.training import CosineUpdateScheduler, build_optimizer


ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.gpu


def _report_error(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Summarize absolute and relative tensor error for assertion diagnostics."""
    difference = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
    print(json.dumps({
        "name": name,
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
    }, sort_keys=True))


def _clone_tree(value: Any) -> Any:
    """Detach tensors and copy a nested state tree for later comparison."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return deepcopy(value)


def _assert_tree_equal(actual: Any, expected: Any, path: str = "root") -> None:
    """Compare nested tensors and containers with exact value equality."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        actual_cpu = actual.detach().cpu()
        if not torch.equal(actual_cpu, expected):
            difference = (actual_cpu.float() - expected.float()).abs()
            pytest.fail(
                f"{path} differs: max_abs={float(difference.max())}, "
                f"mean_abs={float(difference.mean())}"
            )
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_tree_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected)), path
        assert len(actual) == len(expected), path
        for index, item in enumerate(expected):
            _assert_tree_equal(actual[index], item, f"{path}[{index}]")
        return
    assert actual == expected, path


def _pretraining_engine(
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[object, Animal2VecPretrainingModel, TrainingEngine]:
    """Build a tiny pretraining engine on the requested device and precision."""
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    config = replace(
        config,
        common=replace(
            config.common,
            fp16=use_amp,
            fp16_init_scale=1.0,
            min_loss_scale=1e-6,
        ),
    )
    model = Animal2VecPretrainingModel.from_config(config).to(device)
    optimizer = build_optimizer(
        model,
        name=config.optimizer.name,
        learning_rate=config.optimization.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=config.optimization.learning_rate,
        min_lr=config.scheduler.min_lr,
        warmup_updates=config.scheduler.warmup_updates,
        warmup_init_lr=config.scheduler.warmup_init_lr,
        max_updates=4,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=config.optimization.clip_norm,
        device=device,
        use_amp=use_amp,
        amp_init_scale=config.common.fp16_init_scale,
        amp_min_scale=config.common.min_loss_scale,
    )
    return config, model, engine


def _random_pretraining_step(
    model: Animal2VecPretrainingModel,
    engine: TrainingEngine,
    device: torch.device,
) -> object:
    """Generate deterministic synthetic audio and perform one pretraining update."""
    batch = {
        "source": torch.randn(2, 64, device=device),
        "id": torch.tensor([100, 101], device=device),
    }
    return engine.step(
        [batch],
        lambda value: model(value["source"], sample_ids=value["id"], update=engine.update),
    )


def _engine_snapshot(model: Animal2VecPretrainingModel, engine: TrainingEngine) -> dict[str, Any]:
    """Capture all mutable engine and model state needed for exact comparison."""
    return _clone_tree({
        "model": model.state_dict(),
        "teacher": model.teacher.model.state_dict(),
        "optimizer": engine.optimizer.state_dict(),
        "scheduler": engine.scheduler.state_dict(),
        "scaler": engine.scaler.state_dict() if engine.scaler is not None else None,
        "update": engine.update,
        "epoch": engine.epoch,
        "batch_in_epoch": engine.batch_in_epoch,
        "sampler_state": engine.sampler_state,
        "best_metric": engine.best_metric,
        "rng_state": capture_rng_state(),
    })


def test_tiny_pretraining_fp32_cuda_matches_cpu_and_updates_teacher(
    cuda_device: torch.device,
) -> None:
    """Check tiny pretraining FP32 CUDA matches CPU and updates teacher."""
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    cpu_model = Animal2VecPretrainingModel.from_config(config).eval()
    cuda_model = deepcopy(cpu_model).to(cuda_device).eval()
    waveform = torch.randn(2, 64)
    sample_ids = torch.tensor([100, 101])

    cpu_output = cpu_model(waveform, sample_ids=sample_ids, update=0)
    cuda_output = cuda_model(
        waveform.to(cuda_device),
        sample_ids=sample_ids.to(cuda_device),
        update=0,
    )
    _report_error("pretraining.predictions.fp32", cuda_output.predictions, cpu_output.predictions)
    _report_error("pretraining.targets.fp32", cuda_output.targets, cpu_output.targets)
    _report_error("pretraining.loss.fp32", cuda_output.loss, cpu_output.loss)
    assert torch.equal(cuda_output.mask.cpu(), cpu_output.mask)
    assert cuda_output.sample_size == cpu_output.sample_size
    assert torch.allclose(cuda_output.predictions.cpu(), cpu_output.predictions, atol=3e-5, rtol=3e-4)
    assert torch.allclose(cuda_output.targets.cpu(), cpu_output.targets, atol=3e-5, rtol=3e-4)
    assert torch.allclose(cuda_output.loss.cpu(), cpu_output.loss, atol=2e-4, rtol=3e-4)

    cuda_model.train()
    optimizer = torch.optim.Adam(cuda_model.student.parameters(), lr=1e-3)
    teacher_before = next(cuda_model.teacher.model.parameters()).detach().clone()
    training_output = cuda_model(
        waveform.to(cuda_device),
        sample_ids=sample_ids.to(cuda_device),
        update=0,
    )
    (training_output.loss / training_output.sample_size).backward()
    optimizer.step()
    cuda_model.update_teacher(1)
    teacher_after = next(cuda_model.teacher.model.parameters()).detach()
    assert not torch.equal(teacher_before, teacher_after)
    assert torch.isfinite(training_output.loss)


def test_amp_checkpoint_cpu_load_gpu_restore_matches_next_update_exactly(
    cuda_device: torch.device,
    tmp_path: Path,
) -> None:
    """Check AMP checkpoint CPU load gpu restore matches next update exactly."""
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    config, model, engine = _pretraining_engine(cuda_device, use_amp=True)
    first = _random_pretraining_step(model, engine, cuda_device)
    assert first.update == 1
    assert engine.scaler is not None and engine.scaler.get_scale() == 1.0
    engine.sampler_state = {"epoch": 1, "next_batch": 1}
    engine.best_metric = 0.25

    path = tmp_path / "cuda-resume.pt"
    save_checkpoint(
        path,
        engine.checkpoint_payload(
            stage="pretrain",
            config={"active": config_to_dict(config)},
        ),
    )
    cpu_checkpoint = load_checkpoint(path, map_location="cpu")
    assert all(tensor.device.type == "cpu" for tensor in cpu_checkpoint["model"].values())
    assert cpu_checkpoint["scaler"]["scale"] == 1.0

    expected_result = _random_pretraining_step(model, engine, cuda_device)
    expected = _engine_snapshot(model, engine)

    _, resumed_model, resumed_engine = _pretraining_engine(cuda_device, use_amp=True)
    resumed_engine.restore(cpu_checkpoint)
    actual_result = _random_pretraining_step(resumed_model, resumed_engine, cuda_device)
    actual = _engine_snapshot(resumed_model, resumed_engine)

    assert actual_result.loss == expected_result.loss
    assert actual_result.sample_size == expected_result.sample_size
    assert actual_result.gradient_norm == expected_result.gradient_norm
    assert actual_result.learning_rate == expected_result.learning_rate
    _assert_tree_equal(actual, expected)


def test_checkpoint_loaded_directly_on_cuda_restores_cpu_and_cuda_rng(
    cuda_device: torch.device,
    tmp_path: Path,
) -> None:
    """Check checkpoint loaded directly on CUDA restores CPU and CUDA RNG."""
    torch.manual_seed(91)
    torch.cuda.manual_seed_all(92)
    config, _, engine = _pretraining_engine(cuda_device, use_amp=True)
    path = tmp_path / "cuda-map-location.pt"
    save_checkpoint(
        path,
        engine.checkpoint_payload(
            stage="pretrain",
            config={"active": config_to_dict(config)},
        ),
    )
    expected_cpu = torch.rand(8)
    expected_cuda = torch.rand(8, device=cuda_device)

    checkpoint = load_checkpoint(path, map_location=cuda_device)
    assert checkpoint["rng_state"]["torch"].device == cuda_device
    restore_rng_state(checkpoint["rng_state"])

    assert torch.equal(torch.rand(8), expected_cpu)
    assert torch.equal(torch.rand(8, device=cuda_device), expected_cuda)


def test_finetuning_amp_freezes_then_unfreezes_transformer(
    cuda_device: torch.device,
) -> None:
    """Check finetuning AMP freezes then unfreezes transformer."""
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    finetune = replace(
        finetune,
        common=replace(
            finetune.common,
            fp16=True,
            fp16_init_scale=1.0,
            min_loss_scale=1e-6,
        ),
    )
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    ).to(cuda_device)
    optimizer = build_optimizer(
        model,
        name=finetune.optimizer.name,
        learning_rate=finetune.optimization.learning_rate,
        betas=finetune.optimizer.betas,
        eps=finetune.optimizer.eps,
        weight_decay=finetune.optimizer.weight_decay,
    )
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=finetune.optimization.learning_rate,
        min_lr=finetune.scheduler.min_lr,
        warmup_updates=finetune.scheduler.warmup_updates,
        warmup_init_lr=finetune.scheduler.warmup_init_lr,
        max_updates=finetune.optimization.max_update,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=cuda_device,
        use_amp=True,
        amp_init_scale=finetune.common.fp16_init_scale,
        amp_min_scale=finetune.common.min_loss_scale,
    )
    waveform = torch.randn(2, 64, device=cuda_device)
    target = torch.randint(0, 2, (2, 16, 2), device=cuda_device).float()
    sample_ids = torch.tensor([1, 2], device=cuda_device)

    frozen = engine.step(
        [(waveform, target)],
        lambda values: model(
            values[0], target=values[1], update=engine.update, sample_ids=sample_ids
        ),
    )
    assert frozen.update == 1
    assert model.classifier.weight.grad is not None
    assert not any(parameter.grad is not None for parameter in model.encoder.parameters())

    unfrozen = engine.step(
        [(waveform, target)],
        lambda values: model(
            values[0], target=values[1], update=engine.update, sample_ids=sample_ids
        ),
    )
    assert unfrozen.update == 2
    assert any(parameter.grad is not None for parameter in model.encoder.transformer.parameters())
    assert not any(parameter.grad is not None for parameter in model.encoder.local_encoder.parameters())
    assert torch.isfinite(torch.tensor(unfrozen.loss))


def test_amp_overflow_skips_update_and_reduces_scale(
    cuda_device: torch.device,
) -> None:
    """Check AMP overflow skips update and reduces scale."""
    model = torch.nn.Linear(1, 1, bias=False).to(cuda_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.1,
        min_lr=0.01,
        warmup_updates=0,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=0.0,
        device=cuda_device,
        use_amp=True,
        amp_init_scale=128.0,
        amp_min_scale=1.0,
    )
    value = torch.ones(1, 1, device=cuda_device)
    weight_before = model.weight.detach().clone()

    overflow = engine.step(
        [value],
        lambda batch: SimpleNamespace(
            loss=model(batch).float().sum() * 1.0e37,
            sample_size=1,
        ),
    )

    assert overflow.skipped
    assert overflow.update == 0
    assert engine.update == 0
    assert engine.scheduler.last_update == -1
    assert engine.scaler is not None and engine.scaler.get_scale() == 64.0
    assert torch.equal(model.weight, weight_before)

    recovered = engine.step(
        [value],
        lambda batch: SimpleNamespace(
            loss=model(batch).float().square().sum(),
            sample_size=1,
        ),
    )
    assert not recovered.skipped
    assert recovered.update == 1
    assert engine.update == 1
    assert not torch.equal(model.weight, weight_before)
