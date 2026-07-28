"""Compare an uninterrupted optimizer update with one performed after native checkpoint
restoration. The test covers model, teacher, optimizer, scheduler, counters, and random
state together."""

from pathlib import Path

import torch
from torch import nn

from a2v2.training import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from a2v2.training import TrainingEngine
from a2v2.training import CosineUpdateScheduler


class Result:
    """Store the summed loss and sample size expected by the training engine."""
    def __init__(self, loss: torch.Tensor, sample_size: int) -> None:
        self.loss = loss
        self.sample_size = sample_size


def _components() -> tuple[nn.Linear, torch.optim.Optimizer, CosineUpdateScheduler]:
    """Construct the tiny model, optimizer, scheduler, and engine under test."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = CosineUpdateScheduler(
        optimizer, max_lr=1e-2, min_lr=1e-4, warmup_updates=1, max_updates=4
    )
    return model, optimizer, scheduler


def _step(engine: TrainingEngine, model: nn.Linear) -> None:
    """Perform one synthetic training update and return its measured result."""
    batches = [torch.randn(2, 2), torch.randn(1, 2)]

    def forward(value: torch.Tensor) -> Result:
        """Run the minimal model-specific forward path used by this test fixture."""
        prediction = model(value)
        return Result(prediction.square().sum(), value.shape[0])

    engine.step(batches, forward)


def test_checkpoint_resume_matches_uninterrupted_next_update(tmp_path: Path) -> None:
    """Check checkpoint resume matches uninterrupted next update."""
    torch.manual_seed(31)
    model, optimizer, scheduler = _components()
    engine = TrainingEngine(model, optimizer, scheduler, clip_norm=1.0, device=torch.device("cpu"))
    _step(engine, model)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, engine.checkpoint_payload(stage="pretrain", config={"tiny": True}))

    _step(engine, model)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    resumed_model, resumed_optimizer, resumed_scheduler = _components()
    resumed_engine = TrainingEngine(
        resumed_model, resumed_optimizer, resumed_scheduler, clip_norm=1.0, device=torch.device("cpu")
    )
    checkpoint = load_checkpoint(path)
    resumed_engine.restore(checkpoint)
    restore_rng_state(checkpoint["rng_state"])
    _step(resumed_engine, resumed_model)

    for name, value in resumed_model.state_dict().items():
        assert torch.equal(value, expected[name])
    assert resumed_engine.update == engine.update == 2
