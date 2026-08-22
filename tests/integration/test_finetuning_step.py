"""Run a complete tiny fine-tuning update around the encoder freeze boundary. The
assertions connect the configured update number to which backbone parameters receive
gradients."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from a2v2.config import load_config
from a2v2.model import Animal2VecFineTuningModel, TransformerStack
from a2v2.training import CosineUpdateScheduler, TrainingEngine
import a2v2.workflows as workflows


ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.filterwarnings(
    "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning"
)


def test_transformer_fullgraph_compile_matches_eager_without_graph_breaks() -> None:
    """Capture padded legacy attention as one graph with exact eager gradients."""

    torch.manual_seed(401)
    eager = TransformerStack(
        8,
        2,
        depth=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        post_mlp_dropout=0.0,
        layerdrop=0.0,
        input_dropout=0.0,
        position_encoding="none",
        attention_backend="manual",
    ).train()
    compiled = deepcopy(eager).train()
    base = torch.randn(2, 5, 8)
    padding = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, False, True],
    ])

    def forward_backward(
        stack: TransformerStack,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run a deterministic pure-stack loss and populate every gradient."""

        output, targets = stack(value * 1.0, padding)
        loss = output.square().sum() + sum(target.square().sum() for target in targets)
        loss.backward()
        return output.detach(), [target.detach() for target in targets]

    eager_value = base.clone().requires_grad_(True)
    eager_output, eager_targets = forward_backward(eager, eager_value)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled.compile(backend="eager", fullgraph=True, dynamic=False)
    compiled_value = base.clone().requires_grad_(True)
    compiled_output, compiled_targets = forward_backward(compiled, compiled_value)

    torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)
    assert len(compiled_targets) == len(eager_targets)
    for actual, expected in zip(compiled_targets, eager_targets, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled_value.grad, eager_value.grad, rtol=0, atol=0)
    for actual, expected in zip(compiled.parameters(), eager.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0


def test_dynamic_compiled_cls_rope_geglu_checkpointed_updates_match_eager() -> None:
    """Match two dynamic-length whole-model updates across modern strategies."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=(
            "model.position_encoding=rope",
            "model.attention_backend=sdpa",
            "model.use_cls_token=true",
            "model.ffn_type=geglu",
        ),
    )
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=(
            "model.use_cls_token=true",
            "model.classification_head=cls",
            "model.checkpoint_activations=true",
            "model.freeze_finetune_updates=0",
            "model.apply_mask=false",
        ),
    )
    torch.manual_seed(402)
    eager = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    ).train()
    compiled = deepcopy(eager).train()
    compiled.compile(backend="eager", fullgraph=False, dynamic=True)
    eager_optimizer = torch.optim.SGD(eager.parameters(), lr=1e-3)
    compiled_optimizer = torch.optim.SGD(compiled.parameters(), lr=1e-3)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    for update, length in enumerate((64, 72), start=1):
        generator = torch.Generator().manual_seed(500 + update)
        waveform = torch.randn(2, length, generator=generator)
        target = torch.randint(0, 2, (2, 16, 2), generator=generator).float()
        sample_ids = torch.tensor([10 + update, 20 + update])

        eager_optimizer.zero_grad(set_to_none=True)
        eager_output = eager(
            waveform,
            target=target,
            sample_ids=sample_ids,
            update=update,
        )
        assert eager_output.loss is not None
        eager_output.loss.backward()

        compiled_optimizer.zero_grad(set_to_none=True)
        compiled_output = compiled(
            waveform,
            target=target,
            sample_ids=sample_ids,
            update=update,
        )
        assert compiled_output.loss is not None
        compiled_output.loss.backward()

        torch.testing.assert_close(
            compiled_output.logits,
            eager_output.logits,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            compiled_output.loss,
            eager_output.loss,
            rtol=0,
            atol=0,
        )
        for actual, expected in zip(compiled.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        eager_optimizer.step()
        compiled_optimizer.step()

    assert tuple(compiled.state_dict()) == tuple(eager.state_dict())
    for actual, expected in zip(compiled.parameters(), eager.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1


def test_finetuning_freezes_then_unfreezes_backbone() -> None:
    """Check finetuning freezes then unfreezes backbone."""
    torch.manual_seed(18)
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=("model.checkpoint_activations=true",),
    )
    model = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain).train()
    waveform = torch.randn(2, 64)
    target = torch.randint(0, 2, (2, 16, 2)).float()

    frozen = model(waveform, target=target, update=0, sample_ids=torch.tensor([1, 2]))
    (frozen.loss / frozen.sample_size).backward()
    assert model.classifier.weight.grad is not None
    assert not any(parameter.grad is not None for parameter in model.encoder.parameters())

    model.zero_grad(set_to_none=True)
    unfrozen = model(waveform, target=target, update=1, sample_ids=torch.tensor([1, 2]))
    (unfrozen.loss / unfrozen.sample_size).backward()
    assert any(parameter.grad is not None for parameter in model.encoder.transformer.parameters())
    assert not any(parameter.grad is not None for parameter in model.encoder.local_encoder.parameters())
    assert torch.isfinite(unfrozen.loss)


def test_finetuning_engine_omits_pretraining_variance_diagnostics() -> None:
    """Keep collapse diagnostics exclusive to masked mean-teacher training."""

    torch.manual_seed(20)
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineUpdateScheduler(
        optimizer,
        max_lr=1e-3,
        min_lr=0.0,
        warmup_updates=0,
        max_updates=2,
    )
    engine = TrainingEngine(
        model,
        optimizer,
        scheduler,
        clip_norm=1.0,
        device=torch.device("cpu"),
    )
    batch = {
        "source": torch.randn(2, 64),
        "target": torch.randint(0, 2, (2, 16, 2)).float(),
        "id": torch.tensor([1, 2]),
    }

    result = engine.step(
        [batch],
        lambda value: model(
            value["source"],
            target=value["target"],
            sample_ids=value["id"],
            update=engine.update,
        ),
    )

    assert result.pred_var is None
    assert result.target_var is None


def test_cls_validation_reports_sequence_multilabel_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Report clip decisions and AP without invoking event segmentation."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.use_cls_token=true",),
    )
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=(
            "task.enable_padding=true",
            "dataset.required_batch_size_multiple=1",
            "model.use_cls_token=true",
            "model.classification_head=cls",
        ),
    )
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.logit(torch.tensor([0.9, 0.2])))

    first_target = torch.zeros(16, 2)
    first_target[2, 0] = 1
    second_target = torch.zeros(8, 2)
    second_target[3, 1] = 1

    class SequenceDataset(torch.utils.data.Dataset):
        """Expose two unequal-length labeled recordings to real collation."""

        sizes = (64, 32)

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, object]:
            return (
                {
                    "id": 1,
                    "source": torch.randn(64),
                    "target": first_target,
                    "path": "first.wav",
                }
                if index == 0
                else {
                    "id": 2,
                    "source": torch.randn(32),
                    "target": second_target,
                    "path": "second.wav",
                }
            )

    monkeypatch.setattr(workflows, "_make_dataset", lambda *_: SequenceDataset())

    def reject_event_scoring(*_: object, **__: object) -> None:
        """Fail if sequence validation reaches frame-event postprocessing."""

        raise AssertionError("sequence validation routed through event scoring")

    monkeypatch.setattr(workflows, "legacy_segmented_evaluation", reject_event_scoring)

    logger = workflows.TensorBoardLogger(tmp_path, purge_step=None)
    try:
        metrics = workflows._validate(
            model,
            finetune,
            device=torch.device("cpu"),
            update=2,
            tensorboard_logger=logger,
        )
    finally:
        logger.close()

    assert metrics["sequence_precision"] == pytest.approx(0.5)
    assert metrics["sequence_recall"] == pytest.approx(0.5)
    assert metrics["sequence_f1"] == pytest.approx(0.5)
    assert metrics["sequence_accuracy"] == pytest.approx(0.5)
    assert metrics["sequence_average_precision"] == pytest.approx(0.5)
    assert "precision" not in metrics
    assert not any(name.startswith("segmented_") for name in metrics)
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert "validation/valid/sequence/f1" in events.Tags()["scalars"]
    assert "validation/valid/sequence/pr_micro" in events.Tags()["tensors"]
    assert "validation/valid/frame/f1" not in events.Tags()["scalars"]
