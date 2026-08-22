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

import a2v2.training as training
from a2v2.config import config_to_dict, load_config
from a2v2.model import Animal2VecFineTuningModel
from a2v2.model import Animal2VecPretrainingModel, TransformerStack, alibi_bias
from a2v2.training import (
    capture_rng_state,
    load_checkpoint,
    pretraining_variance_diagnostics,
    restore_rng_state,
    save_checkpoint,
)
from a2v2.training import TrainingEngine
from a2v2.training import CosineUpdateScheduler, build_gradient_clipper, build_optimizer


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
    cpu_variances = pretraining_variance_diagnostics(
        cpu_output.predictions,
        cpu_output.targets,
    )
    cuda_variances = pretraining_variance_diagnostics(
        cuda_output.predictions,
        cuda_output.targets,
    )
    assert cuda_variances[0] == pytest.approx(cpu_variances[0], rel=5e-4, abs=5e-5)
    assert cuda_variances[1] == pytest.approx(cpu_variances[1], rel=5e-4, abs=5e-5)

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
    assert actual_result.pred_var == expected_result.pred_var
    assert actual_result.target_var == expected_result.target_var
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
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=("model.checkpoint_activations=true",),
    )
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


def test_transformer_activation_checkpointing_autocast_cuda_parity(
    cuda_device: torch.device,
) -> None:
    """Match direct and checkpointed stochastic Transformer execution in AMP."""

    torch.manual_seed(701)
    torch.cuda.manual_seed_all(702)
    direct = TransformerStack(
        8,
        2,
        depth=2,
        dropout=0.2,
        attention_dropout=0.2,
        activation_dropout=0.2,
        post_mlp_dropout=0.2,
        drop_path_rates=(0.1, 0.2),
        input_dropout=0.2,
    ).to(cuda_device).train()
    checkpointed = deepcopy(direct)
    checkpointed.checkpoint_activations = True
    base_value = torch.randn(2, 5, 8, device=cuda_device)
    direct_value = base_value.clone().requires_grad_(True)
    checkpointed_value = base_value.clone().requires_grad_(True)
    padding = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, False, False, True],
        ],
        device=cuda_device,
    )
    bias = alibi_bias(2, 5).unsqueeze(0).expand(2, -1, -1, -1).to(cuda_device)
    direct_scale = torch.ones(2, 1, 2, 1, 1, device=cuda_device, requires_grad=True)
    checkpointed_scale = direct_scale.detach().clone().requires_grad_(True)

    def run(
        stack: TransformerStack,
        value: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run from fixed CPU/CUDA RNG states and retain final CUDA RNG."""

        torch.manual_seed(703)
        torch.cuda.manual_seed_all(704)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output, targets = stack(value * 1.0, padding, bias, scale)
            loss = output.square().sum() + sum(
                target.square().sum() for target in targets
            )
        loss.backward()
        return (
            output.detach().clone(),
            [target.detach().clone() for target in targets],
            loss.detach().clone(),
            torch.cuda.get_rng_state(cuda_device).clone(),
        )

    direct_output, direct_targets, direct_loss, direct_rng = run(
        direct,
        direct_value,
        direct_scale,
    )
    checkpointed_output, checkpointed_targets, checkpointed_loss, checkpointed_rng = run(
        checkpointed,
        checkpointed_value,
        checkpointed_scale,
    )

    value_rtol, value_atol = 2e-3, 2e-3
    gradient_rtol, gradient_atol = 3e-3, 3e-4
    torch.testing.assert_close(
        checkpointed_output,
        direct_output,
        rtol=value_rtol,
        atol=value_atol,
    )
    assert len(checkpointed_targets) == len(direct_targets)
    for actual, expected in zip(checkpointed_targets, direct_targets, strict=True):
        torch.testing.assert_close(actual, expected, rtol=value_rtol, atol=value_atol)
    torch.testing.assert_close(
        checkpointed_loss,
        direct_loss,
        rtol=value_rtol,
        atol=value_atol,
    )
    torch.testing.assert_close(
        checkpointed_value.grad,
        direct_value.grad,
        rtol=gradient_rtol,
        atol=gradient_atol,
    )
    torch.testing.assert_close(
        checkpointed_scale.grad,
        direct_scale.grad,
        rtol=gradient_rtol,
        atol=gradient_atol,
    )
    for (actual_name, actual), (expected_name, expected) in zip(
        checkpointed.named_parameters(),
        direct.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        assert actual.grad is not None
        assert expected.grad is not None
        torch.testing.assert_close(
            actual.grad,
            expected.grad,
            rtol=gradient_rtol,
            atol=gradient_atol,
        )
    assert torch.equal(checkpointed_rng, direct_rng)


@pytest.mark.parametrize("gradient_clip_method", ("global", "adagc"))
def test_amp_overflow_skips_update_and_reduces_scale(
    cuda_device: torch.device,
    gradient_clip_method: str,
) -> None:
    """Check AMP overflow skips update and reduces scale."""
    model = torch.nn.Linear(1, 1, bias=False).to(cuda_device)
    model.register_buffer("teacher_update", torch.zeros((), device=cuda_device))

    def update_teacher(update: int) -> None:
        """Record the real successful-update callback in module state."""

        model.teacher_update.fill_(update)

    model.update_teacher = update_teacher
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.2)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=0.1,
        min_lr=0.01,
        warmup_updates=0,
        max_updates=2,
    )
    gradient_clipper = build_gradient_clipper(
        model,
        method=gradient_clip_method,
        clip_norm=0.0,
        adagc_warmup_updates=1,
    )
    weight_decay_scheduler = training.build_weight_decay_scheduler(
        optimizer,
        schedule="cosine",
        weight_decay_end=0.02,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=0.0,
        device=cuda_device,
        gradient_clipper=gradient_clipper,
        weight_decay_scheduler=weight_decay_scheduler,
        use_amp=True,
        amp_init_scale=128.0,
        amp_min_scale=1.0,
    )
    value = torch.ones(1, 1, device=cuda_device)
    weight_before = model.weight.detach().clone()
    clipper_before = _clone_tree(engine.gradient_clipper.state_dict())

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
    assert weight_decay_scheduler is not None
    assert weight_decay_scheduler.state_dict() == {"last_update": 0}
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert engine.scaler is not None and engine.scaler.get_scale() == 64.0
    assert torch.equal(model.weight, weight_before)
    assert model.teacher_update.item() == 0.0
    _assert_tree_equal(engine.gradient_clipper.state_dict(), clipper_before)

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
    assert recovered.weight_decay == pytest.approx(0.11)
    assert weight_decay_scheduler.state_dict() == {"last_update": 1}
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.11)
    assert not torch.equal(model.weight, weight_before)
    assert model.teacher_update.item() == 1.0
    if gradient_clip_method == "adagc":
        assert engine.gradient_clipper.state_dict()["update"] == 1


@pytest.mark.parametrize("optimizer_name", ("adam8bit", "adamw8bit"))
def test_bitsandbytes_optimizer_cuda_state_resume(
    cuda_device: torch.device,
    tmp_path: Path,
    optimizer_name: str,
) -> None:
    """Require both 8-bit optimizers to continue through normal state interfaces."""

    bitsandbytes = pytest.importorskip(
        "bitsandbytes",
        reason="bitsandbytes extra is not installed; install a2v2[bnb] to run this CUDA gate",
    )
    torch.manual_seed(811)
    model = torch.nn.Linear(128, 128, bias=False, device=cuda_device)
    optimizer = build_optimizer(
        model,
        name=optimizer_name,
        learning_rate=3e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
        min_8bit_size=1,
        device=cuda_device,
    )
    model.weight.grad = torch.randn_like(model.weight)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / f"{optimizer_name}.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        path,
    )

    resumed_model = torch.nn.Linear(128, 128, bias=False, device=cuda_device)
    resumed_optimizer = build_optimizer(
        resumed_model,
        name=optimizer_name,
        learning_rate=3e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.1,
        min_8bit_size=1,
        device=cuda_device,
    )
    checkpoint = torch.load(path, map_location=cuda_device, weights_only=False)
    resumed_model.load_state_dict(checkpoint["model"], strict=True)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    next_gradient = torch.randn_like(model.weight)
    model.weight.grad = next_gradient.clone()
    resumed_model.weight.grad = next_gradient.clone()

    optimizer.step()
    resumed_optimizer.step()

    assert getattr(optimizer, "_a2v2_bitsandbytes_version") == bitsandbytes.__version__
    assert torch.equal(resumed_model.weight, model.weight)
    _assert_tree_equal(
        _clone_tree(resumed_optimizer.state_dict()),
        _clone_tree(optimizer.state_dict()),
    )
