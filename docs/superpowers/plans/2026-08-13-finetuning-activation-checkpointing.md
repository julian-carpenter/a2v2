# Fine-Tuning Activation Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the eight-A100 40 GB MeerKAT reproduction complete unfrozen fine-tuning from its update-10,000 checkpoint without changing batch topology, sample order, optimizer scheduling, or model mathematics.

**Architecture:** Add an execution-only `ModelConfig.checkpoint_activations` flag with a false default. Propagate the active fine-tuning value into both encoder Transformer stacks, then checkpoint each selected block with PyTorch's non-reentrant API while preserving PyTorch RNG state and keeping NumPy layerdrop selection outside recomputation. The reproduction driver enables the flag only for fine-tuning and retains `max_tokens=960000` with `update_freq=[2]`.

**Tech Stack:** Python 3.10+, PyTorch 2.3+ autograd/checkpointing, immutable dataclass configuration, Bash, pytest, torchrun DDP, NCCL, Gloo, eight NVIDIA A100-SXM4-40GB GPUs.

**Design:** `docs/superpowers/specs/2026-08-13-finetuning-activation-checkpointing-design.md`

## Global Constraints

- Work on the current `main` branch, as authorized by the user. Preserve unrelated user changes.
- Prefix shell commands with `rtk`, including test, Git, Bash, CUDA, and inspection commands.
- Use `apply_patch` for repository edits.
- Write each regression test before the source change that makes it pass.
- Keep `checkpoint_activations` false unless a config or override enables it.
- Let the active fine-tuning config control the encoder execution policy even when the stored pretrained config omits the field or records false.
- Keep `dataset.max_tokens=960000`, `optimization.update_freq=[2]`, eight ranks, the saved sampler cursor, and the 30,000-update scheduler horizon.
- Add no model parameter or buffer. State-dict names, shapes, dtypes, and order must remain unchanged.
- Use `torch.utils.checkpoint.checkpoint(..., use_reentrant=False, preserve_rng_state=True)` on each non-dropped Transformer block.
- Keep NumPy layerdrop selection and layer-specific ALiBi scaling outside the checkpointed callable.
- Skip checkpointing unless the stack has the option enabled, the stack is training, and autograd is enabled.
- Do not combine this work with ALiBi allocation changes, Flash Attention, FSDP, offload, automatic OOM retries, or new dependencies.
- Write production-probe outputs under a fresh `/tmp/a2v2-finetune-checkpointing.*` directory. Do not modify or delete the source checkpoint.
- Cap the production probe at 30 minutes. Stop and report the evidence if checkpointing fails or leaves less than 1 GiB of allocator-reserved headroom on any rank.
- Preserve probe artifacts until the user has reviewed the result.

## File Structure

- Modify `a2v2/config.py`: define, parse, serialize, and restore the execution-policy field.
- Modify `a2v2/model.py`: propagate the field and checkpoint selected Transformer blocks.
- Modify `tests/unit/test_config.py`: protect the default, CLI override, serialization, and old-snapshot behavior.
- Modify `tests/unit/test_transformer.py`: protect the runtime gate, stochastic parity, gradients, RNG state, and layerdrop placement.
- Modify `tests/unit/test_finetuning.py`: protect encoder and active-fine-tuning propagation.
- Modify `tests/integration/test_finetuning_step.py`: exercise checkpointing across the freeze boundary on CPU.
- Modify `tests/gpu/test_cuda_training.py`: exercise the checkpointed unfrozen AMP path on one GPU.
- Modify `scripts/reproduce_meerkat_paper.sh`: enable checkpointing only in the fine-tuning command and record it in provenance.
- Modify `tests/integration/test_reproduction_driver.py`: protect the command graph, summary, and run profile.
- Modify `docs/reproducing-paper.md`: explain the 40 GB execution profile, exact-resume constraint, and failed token-budget workaround.
- Modify `docs/code-guide.md`: document configuration, Transformer execution, fine-tuning propagation, and the update-10,000 boundary.
- Do not modify checked-in YAML recipes. Their canonical defaults remain unchanged.

---

### Task 1: Add the Configuration Field and Propagate the Active Policy

**Files:**
- Modify: `a2v2/config.py:193-243,586-650`
- Modify: `a2v2/model.py:1376-1425,1643-1778,2130-2170`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_finetuning.py`
- Verify: `tests/unit/test_model_state_contract.py`

**Interfaces:**
- Consumes: Existing `ModelConfig`, `config_from_dict`, `config_to_dict`, `config_from_serialized_dict`, `AudioEncoder.from_config`, and `Animal2VecFineTuningModel` construction.
- Produces: `ModelConfig.checkpoint_activations: bool`; `TransformerStack.checkpoint_activations: bool`; `AudioEncoder(..., checkpoint_activations: bool = False)`; active fine-tuning policy copied into the encoder model derived from `pretrained_config`.

- [ ] **Step 1: Write the failing configuration regression**

Add `deepcopy`, `config_from_serialized_dict`, and `config_to_dict` imports in `tests/unit/test_config.py`, then add:

```python
from copy import deepcopy

from a2v2.config import (
    ConfigError,
    config_from_serialized_dict,
    config_to_dict,
    load_config,
    parse_conv_feature_layers,
)


def test_checkpoint_activations_defaults_overrides_and_round_trips() -> None:
    """Keep activation checkpointing opt-in and checkpoint-compatible."""

    path = ROOT / "tests/fixtures/tiny_finetune.yaml"
    default = load_config(path)
    explicit_false = load_config(
        path,
        overrides=("model.checkpoint_activations=false",),
    )
    enabled = load_config(
        path,
        overrides=("model.checkpoint_activations=true",),
    )

    assert default.model.checkpoint_activations is False
    assert explicit_false.model.checkpoint_activations is False
    assert enabled.model.checkpoint_activations is True

    serialized = config_to_dict(enabled)
    serialized_model = serialized["model"]
    assert isinstance(serialized_model, dict)
    assert serialized_model["checkpoint_activations"] is True
    assert config_from_serialized_dict(serialized).model.checkpoint_activations is True

    legacy = deepcopy(serialized)
    legacy_model = legacy["model"]
    assert isinstance(legacy_model, dict)
    legacy_model.pop("checkpoint_activations")
    assert config_from_serialized_dict(legacy).model.checkpoint_activations is False
```

- [ ] **Step 2: Run the configuration test and confirm the red state**

Run:

```bash
rtk python -m pytest -q \
  tests/unit/test_config.py::test_checkpoint_activations_defaults_overrides_and_round_trips
```

Expected: FAIL because `model.checkpoint_activations` is not in the strict model allowlist or `ModelConfig`.

- [ ] **Step 3: Add the minimal configuration field**

Add the field to `ModelConfig` in `a2v2/config.py`:

```python
    load_pretrain_weights: bool = True
    checkpoint_activations: bool = False
    audio: AudioModelConfig = field(default_factory=AudioModelConfig)
```

Add the name to the strict `model` key set in `config_from_dict`:

```python
        "load_pretrain_weights", "checkpoint_activations",
```

Do not add custom serialization code. `config_to_dict` already uses `asdict`, and `ModelConfig(**model_values, audio=audio)` already applies dataclass defaults when an older serialized mapping omits the field.

- [ ] **Step 4: Run the configuration regression**

Run:

```bash
rtk python -m pytest -q tests/unit/test_config.py
```

Expected: PASS, including the existing unknown-field checks.

- [ ] **Step 5: Write failing encoder-propagation regressions**

Extend the imports in `tests/unit/test_finetuning.py`:

```python
from a2v2.model import Animal2VecFineTuningModel, AudioEncoder, mix_targets
```

Add:

```python
def test_audio_encoder_propagates_activation_checkpointing_to_both_stacks() -> None:
    """Apply one config value to the prenet and main Transformer stacks."""

    config = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.checkpoint_activations=true",),
    )

    encoder = AudioEncoder.from_config(config)

    assert encoder.prenet.checkpoint_activations is True
    assert encoder.transformer.checkpoint_activations is True


def test_active_finetuning_config_controls_encoder_checkpointing() -> None:
    """Do not let an older pretrained config disable the active policy."""

    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=("model.checkpoint_activations=true",),
    )

    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )

    assert pretrain.model.checkpoint_activations is False
    assert model.encoder.prenet.checkpoint_activations is True
    assert model.encoder.transformer.checkpoint_activations is True
```

- [ ] **Step 6: Run the propagation tests and confirm the red state**

Run:

```bash
rtk python -m pytest -q \
  tests/unit/test_finetuning.py::test_audio_encoder_propagates_activation_checkpointing_to_both_stacks \
  tests/unit/test_finetuning.py::test_active_finetuning_config_controls_encoder_checkpointing
```

Expected: FAIL because the stack and encoder constructors do not expose the field.

- [ ] **Step 7: Add the minimal propagation path**

Add a keyword and plain attribute to `TransformerStack` in `a2v2/model.py`:

```python
        norm_after: bool | None = None,
        checkpoint_activations: bool = False,
    ) -> None:
        super().__init__()
        self.checkpoint_activations = checkpoint_activations
```

Add the same keyword to `AudioEncoder.__init__`:

```python
        learned_alibi_scale_per_head: bool = True,
        checkpoint_activations: bool = False,
    ) -> None:
```

Pass it to both stacks:

```python
        self.prenet = TransformerStack(
            # existing arguments stay unchanged
            checkpoint_activations=checkpoint_activations,
            **common,
        )

        self.transformer = TransformerStack(
            # existing arguments stay unchanged
            checkpoint_activations=checkpoint_activations,
            **common,
        )
```

Pass the config value from `AudioEncoder.from_config`:

```python
            learned_alibi_scale_per_head=audio.learned_alibi_scale_per_head,
            checkpoint_activations=model.checkpoint_activations,
        )
```

Copy the active value while `Animal2VecFineTuningModel` derives its encoder model from the pretrained config:

```python
        encoder_model = replace(
            pretrained_config.model,
            # existing fine-tuning runtime replacements stay unchanged
            checkpoint_activations=fine_model.checkpoint_activations,
            audio=replace(
                pretrained_config.model.audio,
                prenet_layerdrop=fine_model.layerdrop,
                prenet_dropout=fine_model.dropout,
            ),
        )
```

- [ ] **Step 8: Run focused propagation and state-schema tests**

Run:

```bash
rtk python -m pytest -q \
  tests/unit/test_config.py \
  tests/unit/test_finetuning.py \
  tests/unit/test_model_state_contract.py
```

Expected: PASS. The model-state signature hashes must stay unchanged because the policy adds no parameter or buffer.

- [ ] **Step 9: Commit the configuration and propagation change**

Run:

```bash
rtk git add \
  a2v2/config.py \
  a2v2/model.py \
  tests/unit/test_config.py \
  tests/unit/test_finetuning.py
rtk git diff --cached --check
rtk git commit -m "feat: plumb activation checkpointing policy"
```

Expected: one commit containing the field, its propagation, and its unit regressions.

---

### Task 2: Checkpoint Transformer Blocks with Stochastic Parity

**Files:**
- Modify: `a2v2/model.py:23-32,1430-1460`
- Modify: `tests/unit/test_transformer.py`
- Modify: `tests/integration/test_finetuning_step.py`
- Modify: `tests/gpu/test_cuda_training.py`

**Interfaces:**
- Consumes: `TransformerStack.checkpoint_activations` from Task 1 and existing `TransformerBlock.forward(value, padding_mask, alibi) -> tuple[Tensor, Tensor]`.
- Produces: module-level `activation_checkpoint` alias and the runtime gate `checkpoint_activations and self.training and torch.is_grad_enabled()`.

- [ ] **Step 1: Write the failing runtime-gate regression**

Add these imports to `tests/unit/test_transformer.py`:

```python
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest
import torch

import a2v2.model as model_module
```

Add:

```python
@pytest.mark.parametrize(
    ("enabled", "training", "with_grad", "expected_calls"),
    [
        (True, True, True, 2),
        (False, True, True, 0),
        (True, False, True, 0),
        (True, True, False, 0),
    ],
)
def test_stack_checkpoints_only_training_grad_blocks(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    training: bool,
    with_grad: bool,
    expected_calls: int,
) -> None:
    """Gate recomputation on the option, training mode, and autograd."""

    calls: list[dict[str, object]] = []

    def checkpoint_spy(function: object, *args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return function(*args)  # type: ignore[operator]

    monkeypatch.setattr(model_module, "activation_checkpoint", checkpoint_spy)
    stack = TransformerStack(
        8,
        2,
        depth=2,
        checkpoint_activations=enabled,
    )
    stack.train(training)
    value = torch.randn(2, 5, 8, requires_grad=with_grad)
    context = nullcontext() if with_grad else torch.no_grad()

    with context:
        stack(value)

    assert len(calls) == expected_calls
    assert all(
        call == {"use_reentrant": False, "preserve_rng_state": True}
        for call in calls
    )
```

- [ ] **Step 2: Write the failing forward, gradient, and RNG parity regression**

Add to `tests/unit/test_transformer.py`:

```python
def test_checkpointed_stack_matches_stochastic_forward_backward_and_rng() -> None:
    """Match ordinary block values, gradients, and post-backward RNG state."""

    torch.manual_seed(41)
    ordinary = TransformerStack(
        8,
        2,
        depth=2,
        dropout=0.2,
        attention_dropout=0.2,
        activation_dropout=0.2,
        post_mlp_dropout=0.2,
        drop_path_rates=(0.1, 0.2),
        input_dropout=0.2,
        layerdrop=0.0,
    ).train()
    checkpointed = deepcopy(ordinary)
    checkpointed.checkpoint_activations = True
    base_value = torch.randn(2, 5, 8)
    ordinary_value = base_value.clone().requires_grad_(True)
    checkpointed_value = base_value.clone().requires_grad_(True)
    padding = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, False, True],
    ])
    bias = alibi_bias(2, 5).unsqueeze(0).expand(2, -1, -1, -1)
    ordinary_scale = torch.ones(2, 1, 2, 1, 1, requires_grad=True)
    checkpointed_scale = ordinary_scale.detach().clone().requires_grad_(True)

    def run(
        stack: TransformerStack,
        value: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        torch.manual_seed(314)
        output, targets = stack(value * 1.0, padding, bias, scale)
        loss = output.square().sum() + sum(
            target.square().sum() for target in targets
        )
        loss.backward()
        return (
            output.detach().clone(),
            [target.detach().clone() for target in targets],
            torch.get_rng_state().clone(),
        )

    ordinary_output, ordinary_targets, ordinary_rng = run(
        ordinary,
        ordinary_value,
        ordinary_scale,
    )
    checkpointed_output, checkpointed_targets, checkpointed_rng = run(
        checkpointed,
        checkpointed_value,
        checkpointed_scale,
    )

    torch.testing.assert_close(checkpointed_output, ordinary_output, rtol=0, atol=0)
    assert len(checkpointed_targets) == len(ordinary_targets)
    for actual, expected in zip(checkpointed_targets, ordinary_targets, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        checkpointed_value.grad,
        ordinary_value.grad,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        checkpointed_scale.grad,
        ordinary_scale.grad,
        rtol=0,
        atol=0,
    )
    for (actual_name, actual), (expected_name, expected) in zip(
        checkpointed.named_parameters(),
        ordinary.named_parameters(),
        strict=True,
    ):
        assert actual_name == expected_name
        assert actual.grad is not None
        assert expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
    assert torch.equal(checkpointed_rng, ordinary_rng)
```

The loss must include every returned teacher-target tensor. Fine-tuning consumes those tensors, and omitting them could let non-reentrant early-stop skip a branch that production needs.

- [ ] **Step 3: Write the failing layerdrop-placement regression**

Add:

```python
def test_checkpoint_backward_does_not_repeat_numpy_layerdrop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw each NumPy layerdrop decision once outside recomputation."""

    draws = iter((0.1, 0.9, 0.2))
    observed: list[float] = []

    def random_draw() -> float:
        value = next(draws)
        observed.append(value)
        return value

    monkeypatch.setattr(np.random, "random", random_draw)
    stack = TransformerStack(
        8,
        2,
        depth=3,
        layerdrop=0.5,
        checkpoint_activations=True,
    ).train()
    value = torch.randn(2, 5, 8, requires_grad=True)

    output, targets = stack(value)
    loss = output.square().sum() + sum(
        target.square().sum() for target in targets
    )
    loss.backward()

    assert observed == [0.1, 0.9, 0.2]
    assert len(targets) == 1
```

- [ ] **Step 4: Run all three regressions and confirm the red state**

Run:

```bash
rtk python -m pytest -q \
  tests/unit/test_transformer.py::test_stack_checkpoints_only_training_grad_blocks \
  tests/unit/test_transformer.py::test_checkpointed_stack_matches_stochastic_forward_backward_and_rng \
  tests/unit/test_transformer.py::test_checkpoint_backward_does_not_repeat_numpy_layerdrop
```

Expected: FAIL because `a2v2.model.activation_checkpoint` does not exist and the stack still calls each block directly.

- [ ] **Step 5: Implement the per-block checkpoint call**

Add the import in `a2v2/model.py`:

```python
from torch.utils.checkpoint import checkpoint as activation_checkpoint
```

Keep the existing layerdrop decision and ALiBi scaling in `TransformerStack.forward`. Replace only the direct block invocation:

```python
            should_checkpoint = (
                self.checkpoint_activations
                and self.training
                and torch.is_grad_enabled()
            )
            if should_checkpoint:
                value, target = activation_checkpoint(
                    block,
                    value,
                    padding_mask,
                    layer_bias,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                value, target = block(value, padding_mask, layer_bias)
            layer_outputs.append(target)
```

Pass the module directly. Do not close over the loop variable in a lambda because backward could then recompute with the last block.

- [ ] **Step 6: Run Transformer regressions**

Run:

```bash
rtk python -m pytest -q tests/unit/test_transformer.py
```

Expected: PASS with exact CPU output, gradient, and RNG equality.

- [ ] **Step 7: Exercise the complete fine-tuning freeze boundary with the option enabled**

In `tests/integration/test_finetuning_step.py`, change the fine-tuning load in `test_finetuning_freezes_then_unfreezes_backbone` to:

```python
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=("model.checkpoint_activations=true",),
    )
```

Make the same change in `tests/gpu/test_cuda_training.py` inside `test_finetuning_amp_freezes_then_unfreezes_transformer`:

```python
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=("model.checkpoint_activations=true",),
    )
```

These tests retain their existing frozen-gradient, unfrozen-gradient, AMP, and finite-loss assertions. Other tests continue to exercise the false default.

- [ ] **Step 8: Run CPU and one-GPU fine-tuning regressions**

Run:

```bash
rtk python -m pytest -q \
  tests/integration/test_finetuning_step.py \
  tests/unit/test_finetuning.py \
  tests/unit/test_model_state_contract.py
rtk env CUDA_VISIBLE_DEVICES=0 \
  python -m pytest -q \
  tests/gpu/test_cuda_training.py::test_finetuning_amp_freezes_then_unfreezes_transformer
```

Expected: PASS. The CUDA test must complete one frozen and one checkpointed unfrozen AMP update.

- [ ] **Step 9: Commit the Transformer execution change**

Run:

```bash
rtk git add \
  a2v2/model.py \
  tests/unit/test_transformer.py \
  tests/integration/test_finetuning_step.py \
  tests/gpu/test_cuda_training.py
rtk git diff --cached --check
rtk git commit -m "fix: checkpoint finetuning transformer activations"
```

Expected: one commit containing the recomputation path and its CPU/CUDA regressions.

---

### Task 3: Enable and Record Checkpointing in the Reproduction Driver

**Files:**
- Modify: `scripts/reproduce_meerkat_paper.sh:190-250,295-310,383-405`
- Test: `tests/integration/test_reproduction_driver.py:469-550,650-760`

**Interfaces:**
- Consumes: strict CLI override `model.checkpoint_activations=true` from Task 1.
- Produces: a fine-tuning-only override, launch-summary field, and `finetune_checkpoint_activations=true` run-profile field.

- [ ] **Step 1: Write the failing dry-run command-graph regression**

Add this test after `test_dry_run_is_one_fold_and_uses_all_eight_gpus` in `tests/integration/test_reproduction_driver.py`:

```python
def test_dry_run_enables_activation_checkpointing_only_for_finetuning(
    tmp_path: Path,
) -> None:
    """Enable recomputation only for the memory-bound fine-tuning stage."""

    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    _placeholder_manifests(manifests)

    completed = _run_driver(str(manifests), str(output), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    launches = [
        line.replace("\\", "")
        for line in completed.stdout.splitlines()
        if line.startswith("Launching:") and "a2v2-train" in line
    ]
    pretraining = [
        line for line in launches if "a2v_large_pretrain_best.yaml" in line
    ]
    finetuning = [
        line for line in launches if "finetune_mixup_100.yaml" in line
    ]

    assert len(pretraining) == 2
    assert len(finetuning) == 1
    assert all("model.checkpoint_activations=true" not in line for line in pretraining)
    assert "--override model.checkpoint_activations=true" in finetuning[0]
    assert (
        "  finetune: dataset.max_tokens=960000, "
        "optimization.update_freq=[2], model.checkpoint_activations=true"
        in completed.stdout
    )
```

- [ ] **Step 2: Extend the failing real-mode provenance regression**

In `test_real_driver_burns_in_resumes_and_appends_phase_logs`, add this assertion beside the current OMP assertion:

```python
    assert "finetune_checkpoint_activations=true\n" in run_profile
```

- [ ] **Step 3: Run the two driver tests and confirm the red state**

Run:

```bash
rtk python -m pytest -q \
  tests/integration/test_reproduction_driver.py::test_dry_run_enables_activation_checkpointing_only_for_finetuning \
  tests/integration/test_reproduction_driver.py::test_real_driver_burns_in_resumes_and_appends_phase_logs
```

Expected: FAIL because the driver does not render or record the setting.

- [ ] **Step 4: Add one fixed fine-tuning execution-policy constant**

Place this beside the fine-tuning token and accumulation constants in `scripts/reproduce_meerkat_paper.sh`:

```bash
readonly FINETUNE_CHECKPOINT_ACTIVATIONS=true
```

Do not add an environment variable. The eight-A100 40 GB driver owns this hardware execution profile.

- [ ] **Step 5: Add the summary and run-profile fields**

Extend the existing fine-tuning summary line:

```bash
    "  finetune: dataset.max_tokens=${FINETUNE_MAX_TOKENS}, optimization.update_freq=[${FINETUNE_UPDATE_FREQ}], model.checkpoint_activations=${FINETUNE_CHECKPOINT_ACTIVATIONS}" \
```

Add this line to the `run-profile.txt` writer after `finetune_update_freq`:

```bash
        printf 'finetune_checkpoint_activations=%s\n' "${FINETUNE_CHECKPOINT_ACTIVATIONS}"
```

- [ ] **Step 6: Add the fine-tuning-only CLI override**

Add this pair to `FINETUNE_COMMAND` after the accumulation override:

```bash
    --override "model.checkpoint_activations=${FINETUNE_CHECKPOINT_ACTIVATIONS}"
```

Do not add it to `PRETRAIN_BURN_IN_COMMAND` or `PRETRAIN_RESUME_COMMAND`.

- [ ] **Step 7: Run shell and driver regressions**

Run:

```bash
rtk bash -n scripts/reproduce_meerkat_paper.sh
rtk python -m pytest -q tests/integration/test_reproduction_driver.py
```

Expected: PASS. Existing assertions for `max_tokens=960000`, `update_freq=[2]`, resume selection, and command counts must remain unchanged.

- [ ] **Step 8: Commit the driver change**

Run:

```bash
rtk git add \
  scripts/reproduce_meerkat_paper.sh \
  tests/integration/test_reproduction_driver.py
rtk git diff --cached --check
rtk git commit -m "fix: enable checkpointing in meerkat finetuning"
```

Expected: one commit containing driver wiring and provenance tests.

---

### Task 4: Document the Unfreeze Memory Boundary and Recovery Contract

**Files:**
- Modify: `docs/reproducing-paper.md:40-135,245-285`
- Modify: `docs/code-guide.md:80-115,270-305`
- Test: `tests/unit/test_documentation.py`

**Interfaces:**
- Consumes: the final config field, runtime gate, and driver behavior from Tasks 1 through 3.
- Produces: operator guidance that preserves the existing sampler topology and identifies activation checkpointing as the 40 GB memory control.

- [ ] **Step 1: Replace the incorrect fine-tuning headroom and fallback guidance**

In `docs/reproducing-paper.md`, retain the published and driver batch equations, then replace the headroom and 800,000-token fallback text with this content:

````markdown
The driver keeps the fine-tuning global token target while each rank processes
a larger microbatch:

```text
published: 426,667 × 4 ranks × 9 accumulation = 15,360,012 tokens/update
driver:    960,000 × 8 ranks × 2 accumulation = 15,360,000 tokens/update
```

Fine-tuning update 10,000 is the first update with a trainable Transformer
backbone. The checkpoint saved at update 10,000 contains the last frozen
forward; its first resumed forward retains activations for all eight prenet and
16 main Transformer blocks. On 40 GB A100s, the driver sets
`model.checkpoint_activations=true` for fine-tuning. PyTorch retains block
boundaries and recomputes each block during backward. Frozen training,
validation, evaluation, and inference bypass recomputation because they run
without encoder gradients.

Do not use `A2V2_FINETUNE_MAX_TOKENS=800000` as an OOM recovery setting for the
published MeerKAT manifests. All 53,114 training records contain 80,000 samples,
and the inherited batch-size multiple makes both 960,000 and 800,000 produce
eight-record microbatches. Lower token budgets also repartition shuffled
batches. Applying the saved numeric sampler cursor to that new partition would
duplicate and skip examples, so changing `max_tokens` is not an exact resume.
````

Remove the old fallback command. Keep the existing batch-control table unchanged. Add after the table:

```markdown
Activation checkpointing is a fixed fine-tuning override in this 40 GB driver,
not an environment-variable batch control. It adds computation during the
unfrozen backward pass while preserving the batch and checkpoint cursor.
```

- [ ] **Step 2: Clarify resume and provenance behavior**

Extend the fine-tuning resume paragraph in `docs/reproducing-paper.md` with:

```markdown
The activation-checkpointing flag changes execution policy and adds no
checkpoint tensors. The driver can therefore enable it while resuming the
untouched update-10,000 checkpoint under the same sampler topology.
```

Change the provenance phrase from `batch profile provenance` to `batch and activation-memory profile provenance`.

- [ ] **Step 3: Document optional checkpointing in the standalone fine-tuning section**

After the four-rank fine-tuning command in `docs/reproducing-paper.md`, add:

```markdown
On 40 GB devices, add
`--override model.checkpoint_activations=true` to trade block recomputation for
lower unfrozen-backbone activation memory. The field defaults to false, so
canonical YAML files and inference retain their existing execution path.
```

- [ ] **Step 4: Document the configuration and Transformer contracts**

After the serialization paragraph in `docs/code-guide.md`, add:

```markdown
`ModelConfig.checkpoint_activations` defaults to false. The field changes
autograd execution and creates no parameter or buffer, so serialized configs
record it while model state dictionaries remain unchanged. Older serialized
configs that omit it restore the false default.
```

After the `TransformerStack` paragraph, add:

```markdown
When `checkpoint_activations`, training mode, and gradient tracking are all
active, `TransformerStack` runs each selected block through non-reentrant
PyTorch activation checkpointing with RNG preservation. The stack draws NumPy
layerdrop decisions before entering the checkpointed block, so backward
recomputation cannot select a different layer path. Evaluation and frozen
fine-tuning call blocks directly.
```

- [ ] **Step 5: Document active fine-tuning propagation and troubleshooting**

Extend the `Animal2VecFineTuningModel` paragraph in `docs/code-guide.md`:

```markdown
Fine-tuning copies the active `checkpoint_activations` value into the encoder
model derived from the pretrained configuration. An older pretrained snapshot
therefore cannot disable the current execution policy.

### Fine-tuning activation memory

A fine-tuning checkpoint at update 10,000 has not completed the 30,000-update
recipe. Its next forward is the first trainable-backbone forward and retains
dense attention activations that the frozen phase discarded. On the eight-A100
40 GB reproduction profile, the driver preserves the saved batch topology and
uses block recomputation to control that peak. Allocator mapping warnings near
the failure report exhausted capacity; changing allocator fragmentation
settings cannot replace the activation-memory policy.
```

- [ ] **Step 6: Run documentation and stale-guidance checks**

Run:

```bash
rtk python -m pytest -q \
  tests/unit/test_documentation.py \
  tests/integration/test_reproduction_driver.py
rtk rg -n "A2V2_FINETUNE_MAX_TOKENS=800000|substantially more measured headroom" \
  README.md docs/reproducing-paper.md docs/code-guide.md scripts tests
```

Expected: tests PASS and `rg` returns no stale 800,000-token fallback or unfreeze-headroom claim.

- [ ] **Step 7: Commit the documentation change**

Run:

```bash
rtk git add docs/reproducing-paper.md docs/code-guide.md
rtk git diff --cached --check
rtk git commit -m "docs: explain finetuning activation memory"
```

Expected: one documentation commit with no YAML hash changes.

---

### Task 5: Run Complete Automated Verification

**Files:**
- Verify: all files changed in Tasks 1 through 4
- Verify: `tests/gpu/test_cuda_components.py`
- Verify: `tests/gpu/test_cuda_inference.py`
- Verify: `tests/gpu/test_cuda_training.py`

**Interfaces:**
- Consumes: the committed implementation and documentation.
- Produces: clean CPU, one-GPU, shell, state-schema, and driver evidence before the production-sized DDP probe.

- [ ] **Step 1: Recheck the runtime after any node restart**

Run:

```bash
rtk git status --short --branch
rtk which a2v2-train
rtk python -c \
  'import a2v2, torch, soundfile, h5py, yaml, tensorboard; print(a2v2.__file__); print(torch.__version__)'
rtk nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader
```

Expected: `a2v2` resolves inside `/abyss/home/ml/a2v2`, all imports succeed, and all eight A100s have no resident training process. If the node restart removed the editable installation, restore it with:

```bash
rtk python -m pip install -e '.[test]'
```

Then repeat the import and entry-point checks.

- [ ] **Step 2: Run focused regressions**

Run:

```bash
rtk python -m pytest -q \
  tests/unit/test_config.py \
  tests/unit/test_transformer.py \
  tests/unit/test_finetuning.py \
  tests/unit/test_model_state_contract.py \
  tests/integration/test_finetuning_step.py \
  tests/integration/test_reproduction_driver.py
rtk bash -n scripts/reproduce_meerkat_paper.sh
rtk git diff --check
```

Expected: PASS with no shell or whitespace error.

- [ ] **Step 3: Run the complete CPU suite without visible GPUs**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES= python -m pytest -q
```

Expected: all CPU tests pass and GPU-marked tests skip.

- [ ] **Step 4: Run the complete one-GPU suite**

Run:

```bash
rtk env CUDA_VISIBLE_DEVICES=0 \
  python -m pytest -q -m gpu \
  tests/gpu/test_cuda_components.py \
  tests/gpu/test_cuda_inference.py \
  tests/gpu/test_cuda_training.py
```

Expected: all one-GPU component, inference, resume, AMP, and checkpointed unfreeze tests pass.

- [ ] **Step 5: Inspect the committed implementation**

Run:

```bash
rtk git log -5 --oneline
rtk git status --short --branch
rtk git show --stat --oneline HEAD~4..HEAD
rtk rg -n "checkpoint_activations|activation_checkpoint" \
  a2v2 scripts tests docs
```

Expected: the four task commits follow the design commit, the worktree is clean, the field defaults to false, the reproduction fine-tuning command enables it, and pretraining commands do not.

---

### Task 6: Prove Two Production-Sized Unfrozen Updates on Eight GPUs

**Files:**
- Read only: `experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt`
- Read only: `experiments/a2v2-meerkat-fold0-260728/pretrain/checkpoint_last.pt`
- Read only: `/local/datasets/MeerKAT_10s_2024-06-12/manifests`
- Create at runtime: `/tmp/a2v2-finetune-checkpointing.*/`

**Interfaces:**
- Consumes: the exact update-10,000 checkpoint, the production manifest, the corrected driver profile, and training-summary peak-memory fields.
- Produces: an isolated update-10,002 checkpoint, eight rank summaries, internal peak-memory measurements, source-immutability evidence, and post-run GPU-idle evidence.

- [ ] **Step 1: Create an isolated evidence directory and name the immutable inputs**

Run from `/abyss/home/ml/a2v2`:

```bash
A2V2_PROBE_SOURCE=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt
A2V2_PROBE_PRETRAIN=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/pretrain/checkpoint_last.pt
A2V2_PROBE_MANIFESTS=/local/datasets/MeerKAT_10s_2024-06-12/manifests
A2V2_PROBE_ROOT="$(rtk proxy mktemp -d /tmp/a2v2-finetune-checkpointing.XXXXXXXX)"
rtk proxy printf '%s\n' "$A2V2_PROBE_ROOT" \
  | rtk proxy tee /tmp/a2v2-finetune-checkpointing.current
rtk proxy mkdir -p "$A2V2_PROBE_ROOT/finetune" "$A2V2_PROBE_ROOT/nccl"
rtk proxy printf '%s\n' "$A2V2_PROBE_ROOT"
```

Expected: one fresh path under `/tmp/a2v2-finetune-checkpointing.*`. Keep the printed path in the task record. Each later command block reloads this pointer because command shells may not preserve variables between steps.

- [ ] **Step 2: Record source identity and run the production preflight**

Run:

```bash
A2V2_PROBE_SOURCE=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt
A2V2_PROBE_ROOT="$(rtk proxy cat /tmp/a2v2-finetune-checkpointing.current)"
rtk proxy sha256sum -- "$A2V2_PROBE_SOURCE" \
  | rtk proxy tee "$A2V2_PROBE_ROOT/source.sha256.before"
rtk proxy stat -c '%n|size=%s|mtime=%y|inode=%i' -- "$A2V2_PROBE_SOURCE" \
  | rtk proxy tee "$A2V2_PROBE_ROOT/source.stat.before"
rtk proxy nvidia-smi \
  --query-compute-apps=pid,gpu_uuid,used_memory \
  --format=csv,noheader,nounits \
  | rtk proxy tee "$A2V2_PROBE_ROOT/compute-apps.before.csv"
rtk proxy test ! -s "$A2V2_PROBE_ROOT/compute-apps.before.csv"
rtk proxy env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python scripts/check_reproduction_environment.py \
  --output-dir "$A2V2_PROBE_ROOT" \
  --checkpoint "$A2V2_PROBE_SOURCE" \
  --expected-stage finetune \
  | rtk proxy tee "$A2V2_PROBE_ROOT/preflight.before.json"
rtk proxy jq -e \
  '.pass and .checkpoint.stage=="finetune" and .checkpoint.update==10000 and .checkpoint.rng_world_size==8 and (.cuda_devices|length)==8' \
  "$A2V2_PROBE_ROOT/preflight.before.json"
```

Expected: source SHA-256 `47c545c5dd5a7c5d27bc3911407ddfaa4bdaf864508a6e2170b567e5e2b6a14f`, update 10,000, eight-rank RNG state, eight idle A100s, and at least 64 GiB of output headroom.

- [ ] **Step 3: Recheck NCCL and Gloo transport after the node restart**

Run:

```bash
A2V2_PROBE_ROOT="$(rtk proxy cat /tmp/a2v2-finetune-checkpointing.current)"
rtk proxy timeout --foreground --signal=TERM --kill-after=30s 5m \
  env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  tests/gpu/nccl_probe.py \
  --output-dir "$A2V2_PROBE_ROOT/nccl"
rtk proxy jq -s -e \
  'length==8 and ([.[].rank]|sort)==[0,1,2,3,4,5,6,7] and all(.[]; .pass and .world_size==8 and .rank_rng_restore_exact)' \
  "$A2V2_PROBE_ROOT"/nccl/rank-*.json
```

Expected: eight passing rank reports with exact RNG restoration.

- [ ] **Step 4: Launch the bounded update-10,000 to update-10,002 probe**

Start one-second supplementary sampling, then run torchrun with a 29-minute TERM boundary and 60-second kill grace:

```bash
A2V2_PROBE_SOURCE=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt
A2V2_PROBE_PRETRAIN=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/pretrain/checkpoint_last.pt
A2V2_PROBE_MANIFESTS=/local/datasets/MeerKAT_10s_2024-06-12/manifests
A2V2_PROBE_ROOT="$(rtk proxy cat /tmp/a2v2-finetune-checkpointing.current)"
rtk bash -o pipefail -c '
probe_root=$1
probe_pretrain=$2
probe_manifests=$3
probe_source=$4

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  -l 1 \
  > "$probe_root/nvidia-smi.csv" &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

timeout --foreground --signal=TERM --kill-after=60s 29m \
  env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 \
  python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  /usr/local/bin/a2v2-train \
  --config /abyss/home/ml/a2v2/configs/MeerKAT/finetune_mixup_100.yaml \
  --pretrained-checkpoint "$probe_pretrain" \
  --override "task.data=$probe_manifests" \
  --override dataset.train_subset=train_0 \
  --override dataset.valid_subset=valid_0 \
  --override "checkpoint.save_dir=$probe_root/finetune" \
  --override distributed_training.distributed_world_size=8 \
  --override dataset.max_tokens=960000 \
  --override "optimization.update_freq=[2]" \
  --override model.checkpoint_activations=true \
  --device cuda \
  --resume "$probe_source" \
  --stop-at-update 10002 \
  2>&1 | tee "$probe_root/train.log"
exit "${PIPESTATUS[0]}"
' _ \
  "$A2V2_PROBE_ROOT" \
  "$A2V2_PROBE_PRETRAIN" \
  "$A2V2_PROBE_MANIFESTS" \
  "$A2V2_PROBE_SOURCE"
```

Expected: torchrun exits zero within 30 minutes. The source checkpoint path never appears as a save directory.

- [ ] **Step 5: Validate all rank summaries and the 1 GiB memory margin**

Run:

```bash
A2V2_PROBE_ROOT="$(rtk proxy cat /tmp/a2v2-finetune-checkpointing.current)"
rtk proxy jq -Rsc \
  '[split("\n")[] | fromjson? | select(type=="object" and has("training_summary")) | .training_summary]' \
  "$A2V2_PROBE_ROOT/train.log" \
  | rtk proxy tee "$A2V2_PROBE_ROOT/training-summaries.json"
A2V2_TOTAL_MIN="$(rtk proxy jq '[.cuda_devices[].total_bytes]|min' "$A2V2_PROBE_ROOT/preflight.before.json")"
A2V2_PEAK_CAP=$((A2V2_TOTAL_MIN - 1073741824))
rtk proxy jq -e --argjson cap "$A2V2_PEAK_CAP" \
  'length==8 and ([.[].rank]|sort)==[0,1,2,3,4,5,6,7] and ([.[].cuda_device]|sort)==[0,1,2,3,4,5,6,7] and all(.[]; .world_size==8 and .final_update==10002 and .terminal_update==10002 and .configured_max_update==30000 and .cuda_peak_memory_allocated <= .cuda_peak_memory_reserved and .cuda_peak_memory_reserved <= $cap)' \
  "$A2V2_PROBE_ROOT/training-summaries.json"
```

Expected: eight summaries at update 10,002. Each rank's PyTorch peak reserved memory stays at least 1 GiB below that device's reported capacity. Treat these allocator metrics as authoritative; `nvidia-smi.csv` only supplies a time series.

- [ ] **Step 6: Validate checkpoint advancement, optimizer creation, and exact cursor use**

Run this strict validator for the known production checkpoint:

```bash
A2V2_PROBE_SOURCE=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt
A2V2_PROBE_ROOT="$(rtk proxy cat /tmp/a2v2-finetune-checkpointing.current)"
rtk python - \
  "$A2V2_PROBE_SOURCE" \
  "$A2V2_PROBE_ROOT/finetune/checkpoint_last.pt" \
  <<'PY' | rtk proxy tee "$A2V2_PROBE_ROOT/checkpoint-acceptance.json"
import json
import sys

from a2v2.training import load_checkpoint


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


source = load_checkpoint(sys.argv[1], map_location="cpu")
result = load_checkpoint(sys.argv[2], map_location="cpu")
active = result["config"]["active"]
require(result["stage"] == "finetune", "wrong stage")
require(result["update"] == 10002, "wrong update")
require(
    result["scheduler"]["last_update"] == 10002,
    "scheduler did not advance twice",
)
require(
    active["model"]["checkpoint_activations"] is True,
    "checkpointing was not serialized",
)
require(active["optimization"]["max_update"] == 30000, "scheduler horizon changed")
require(active["optimization"]["update_freq"] == [2], "accumulation changed")
require(active["dataset"]["max_tokens"] == 960000, "batch topology changed")
require(
    active["distributed"]["world_size"]
    == active["distributed"]["requested_world_size"]
    == 8,
    "world size changed",
)
require(result["epoch"] == source["epoch"] == 23, "engine epoch changed")
require(
    result["batch_in_epoch"] == source["batch_in_epoch"] + 4 == 19990,
    "unexpected attempted microbatch count",
)
require(
    result["sampler_state"] == {"epoch": 24, "next_batch": 752},
    "unexpected sampler cursor",
)
require(
    result["best_metric"] == source["best_metric"],
    "unexpected validation or best-metric change",
)
require(set(result["model"]) == set(source["model"]), "model state schema changed")
require(
    result["scaler"]["scale"] == source["scaler"]["scale"] == 64.0,
    "AMP scale changed",
)
require(
    result["scaler"]["_growth_tracker"]
    == source["scaler"]["_growth_tracker"] + 2
    == 1911,
    "AMP attempts were not exactly two finite updates",
)
source_state = source["optimizer"]["state"]
result_state = result["optimizer"]["state"]
source_ids = set(source_state)
new_ids = set(result_state) - source_ids
require(source_ids == {111, 333}, "unexpected source optimizer state")
require(
    all(int(result_state[key]["step"]) == 10002 for key in source_ids),
    "classifier Adam state did not advance twice",
)
new_steps = [int(result_state[key]["step"]) for key in new_ids]
require(
    new_steps and set(new_steps) <= {1, 2} and 2 in new_steps,
    "backbone Adam state was not created and reused",
)
rng = result["rng_state"]
require(
    rng["world_size"] == 8 and len(rng["by_rank"]) == 8,
    "incomplete rank RNG state",
)
print(json.dumps({
    "pass": True,
    "update": result["update"],
    "sampler_state": result["sampler_state"],
    "optimizer_state_entries": len(result_state),
    "new_optimizer_state_entries": len(new_ids),
    "new_optimizer_steps": sorted(set(new_steps)),
    "amp_scale": result["scaler"]["scale"],
    "amp_growth_tracker": result["scaler"]["_growth_tracker"],
}, sort_keys=True))
PY
rtk proxy test ! -e "$A2V2_PROBE_ROOT/finetune/checkpoint_last.pt.tmp"
```

Expected: the validator prints `"pass": true`. Four microbatches advance the exact saved partition, both updates stay finite, and the backbone Adam state exists and reaches its second step.

- [ ] **Step 7: Re-run checkpoint preflight and prove source immutability**

Run:

```bash
A2V2_PROBE_SOURCE=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt
A2V2_PROBE_ROOT="$(rtk proxy cat /tmp/a2v2-finetune-checkpointing.current)"
rtk proxy sha256sum -c "$A2V2_PROBE_ROOT/source.sha256.before"
rtk proxy stat -c '%n|size=%s|mtime=%y|inode=%i' -- "$A2V2_PROBE_SOURCE" \
  | rtk proxy tee "$A2V2_PROBE_ROOT/source.stat.after"
rtk proxy diff -u \
  "$A2V2_PROBE_ROOT/source.stat.before" \
  "$A2V2_PROBE_ROOT/source.stat.after"
rtk proxy env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python scripts/check_reproduction_environment.py \
  --output-dir "$A2V2_PROBE_ROOT" \
  --checkpoint "$A2V2_PROBE_ROOT/finetune/checkpoint_last.pt" \
  --expected-stage finetune \
  | rtk proxy tee "$A2V2_PROBE_ROOT/preflight.after.json"
rtk proxy jq -e \
  '.pass and .checkpoint.stage=="finetune" and .checkpoint.update==10002 and .checkpoint.rng_world_size==8' \
  "$A2V2_PROBE_ROOT/preflight.after.json"
```

Expected: source hash, size, mtime, and inode match the pre-run record. The temporary checkpoint passes the normal eight-rank fine-tuning preflight at update 10,002.

- [ ] **Step 8: Prove that all GPU processes exited and retain the evidence**

Run:

```bash
A2V2_PROBE_ROOT="$(rtk proxy cat /tmp/a2v2-finetune-checkpointing.current)"
rtk proxy nvidia-smi \
  --query-compute-apps=pid,gpu_uuid,used_memory \
  --format=csv,noheader,nounits \
  | rtk proxy tee "$A2V2_PROBE_ROOT/compute-apps.after.csv"
rtk proxy test ! -s "$A2V2_PROBE_ROOT/compute-apps.after.csv"
rtk git status --short --branch
rtk proxy printf 'probe_evidence=%s\n' "$A2V2_PROBE_ROOT"
```

Expected: no resident compute processes, a clean source worktree, and one retained evidence directory. Do not delete it during this task.

---

## Completion Criteria

- The config field defaults to false, survives explicit true/false parsing, serializes, and restores false from old snapshots that omit it.
- The active fine-tuning config enables both encoder stacks even when the pretrained config records false.
- Checkpointed and direct Transformer paths match CPU output, targets, input gradients, parameter gradients, ALiBi-scale gradients, and final PyTorch RNG state.
- NumPy layerdrop draws occur once per stack layer and do not repeat during backward.
- Frozen fine-tuning, evaluation, and no-grad execution bypass checkpointing.
- The reproduction driver enables the flag only for fine-tuning, records it, and retains `960000/[2]`.
- Model state-dict signatures remain unchanged.
- The complete CPU and one-GPU suites pass.
- Eight production-sized ranks complete updates 10,001 and 10,002 without OOM or collective failure.
- Every rank retains at least 1 GiB of peak reserved-memory headroom.
- The temporary checkpoint contains complete eight-rank resume state, advances the exact sampler cursor, creates and reuses backbone Adam state, and records the active flag.
- The source update-10,000 checkpoint remains byte-for-byte and metadata identical.
- All GPU processes exit, and the evidence directory remains available for review.
