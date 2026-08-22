# Task 7 Report: Weight-Decay Scheduling and 8-Bit Optimizers

## Status

Task 7 adds opt-in cosine weight-decay scheduling, exact scheduler resume,
Adam8bit and AdamW8bit builders, live decay metrics, and the pinned `bnb`
extra. The default `adam` plus `constant` schedule path keeps the previous
optimizer class, parameter groups, order, optimizer state, decay values, and
`weight_decay_scheduler=None` checkpoint slot.

Production changes stay in `a2v2/training.py`, `a2v2/workflows.py`, and
`pyproject.toml`. Test changes cover optimizer components, engine transactions,
checkpoint resume, CLI and TensorBoard output, and bounded CUDA execution.
Frozen reproduction scripts and recipe values did not change.

## Strict TDD evidence

The pre-change optimizer, engine, and resume baseline passed:

```text
rtk python -m pytest -q tests/unit/test_optim.py tests/unit/test_engine.py tests/integration/test_resume.py
58 passed in 7.26s
```

### Decay scheduler and resume RED/GREEN

The first decay tests covered the constant no-op path, cosine start/midpoint/end
and clamp values, zero-group preservation, malformed state, checkpoint resume,
and optimizer-failure transaction behavior. The initial RED failed at the
absent scheduler factory:

```text
rtk python -m pytest -q tests/unit/test_optim.py -k weight_decay tests/unit/test_engine.py::test_adagc_state_commits_only_after_optimizer_step_succeeds tests/integration/test_resume.py -k weight_decay
10 failed, 3 passed, 53 deselected
```

The independent optimizer-failure RED also failed at the missing factory:

```text
rtk python -m pytest -q tests/unit/test_engine.py::test_adagc_state_commits_only_after_optimizer_step_succeeds
1 failed
```

After the scheduler and engine integration, the decay selection passed:

```text
14 passed, 53 deselected
```

The independent optimizer-failure transaction test passed:

```text
1 passed
```

Workflow, JSON, and TensorBoard tests then started RED:

```text
rtk python -m pytest -q tests/integration/test_resume.py::test_training_workflow_wires_adagc_and_persists_resume_fingerprint tests/unit/test_tensorboard_logging.py::test_tensorboard_logger_records_pretraining_and_segmented_validation tests/integration/test_cli.py::test_cli_stop_at_update_preserves_configured_scheduler_horizon
3 failed
```

The failures identified the absent workflow factory, TensorBoard scalar, JSON
field, and checkpoint state. The first GREEN run exposed one test fixture whose
SGD decay did not match its config. Correcting that fixture produced the final
workflow GREEN, and the expanded focused suite later passed 91 tests.

### Optional optimizer RED/GREEN

Builder and packaging tests began with five intended feature failures: the old
builder rejected the new arguments and `pyproject.toml` lacked the `bnb` extra.
One native-lazy-import test also intercepted Triton's unrelated use of
`importlib`; the test now delegates every import except `bitsandbytes`.

```text
rtk python -m pytest -q tests/unit/test_optim.py -k bitsandbytes
6 failed
```

After lazy builder selection and packaging, the corrected selection passed:

```text
6 passed, 35 deselected
```

A separate threshold test proved that zero reached the optional import before
validation:

```text
rtk python -m pytest -q tests/unit/test_optim.py::test_bitsandbytes_optimizer_rejects_nonpositive_minimum_before_import
1 failed
```

The pre-import positive-size check made the same test pass. Run-summary version
metadata followed its own missing-helper RED and one-test GREEN cycle.

## Schedule formula and clock

For each parameter group with starting decay `w_0`, configured end `w_end`,
horizon `U`, and successful update `u`, the scheduler applies:

```text
w(u) = w_end + 0.5 * (w_0 - w_end) * (1 + cos(pi * min(u,U) / U))
```

The constructor writes `w(0)` before the first optimizer step. A successful
optimizer step increments the engine update, advances the learning-rate
scheduler and decay scheduler to that update, then advances the teacher. AMP
overflow, local optimizer exceptions, and the rank-wide terminal optimizer
decision leave the decay clock and group values unchanged. Tests cover global
clipping and AdaGC overflow paths plus a simulated rank-terminal failure that
blocks checkpointing and future steps.

Each group stores `initial_weight_decay`; the scheduler also owns an immutable
tuple of those starting values. A group that starts at zero remains zero. An
omitted `weight_decay_end` uses that group's starting value, preserving the
current fixed decay. Values clamp at `w_end` after `max_update`.

Cosine state contains one strict field, `last_update`. The engine saves it in
checkpoint v2, validates its exact schema and clock before mutating live state,
rejects missing state after update zero, and installs validated state after the
optimizer restore. The resume fingerprint already covers schedule, start, end,
optimizer name, and Adam settings. It continues to allow the established
`optimization.max_update` horizon override.

## bitsandbytes behavior and CUDA evidence

`pyproject.toml` now defines:

```toml
bnb = ["bitsandbytes>=0.49,<0.50"]
```

Native Adam and AdamW paths never import bitsandbytes. Adam8bit and AdamW8bit
reject non-CUDA devices before import, reject nonpositive `min_8bit_size`, and
name `a2v2[bnb]` when the package is missing. Both builders receive the same
ordered decay/no-decay groups plus `lr`, `betas`, `eps`, and `min_8bit_size`.
They use normal optimizer `state_dict` and `load_state_dict` interfaces. The run
summary records the selected bitsandbytes version.

The host originally had bitsandbytes 0.50.0. A short bounded install replaced
it with pinned-range version 0.49.2. The 0.49.2 wheel lacks a CUDA 13.3 binary,
so its first native step reported the package's CUDA-version mismatch. The
wheel includes CUDA 13.0, and the package-supported override loaded that binary
on this CUDA 13.3 host:

```text
rtk env BNB_CUDA_VERSION=130 python -c "import bitsandbytes, torch; print(f'bitsandbytes={bitsandbytes.__version__} torch_cuda={torch.version.cuda}')"
bitsandbytes=0.49.2 torch_cuda=13.3
```

Pinned-range combined CUDA evidence:

```text
rtk env BNB_CUDA_VERSION=130 python -m pytest -q tests/gpu/test_cuda_training.py -k 'bitsandbytes or overflow'
4 passed, 5 deselected in 6.95s
```

The two bitsandbytes cases run Adam8bit and AdamW8bit updates, save model and
optimizer state, restore through normal interfaces, and compare the next model
and optimizer states exactly. The other two cases prove the decay clock stays
fixed on AMP overflow for global clipping and AdaGC.

## Legacy compatibility and final CPU evidence

The default schedule factory returns `None` before touching optimizer groups.
The parity test checks the Fairseq-compatible Adam class, exact group order,
group keys and values, and pre-scheduler optimizer state. Existing tests retain
the Fairseq epsilon placement, decoupled decay equation, exact legacy resume,
v1 upgrade, frozen reproduction driver, and checkpoint conversion contracts.

Expanded focused regression:

```text
rtk python -m pytest -q tests/unit/test_optim.py tests/unit/test_engine.py tests/unit/test_tensorboard_logging.py tests/integration/test_resume.py tests/integration/test_cli.py
93 passed, 8 warnings in 10.40s
```

Full CPU regression after the implementation and test-fixture corrections:

```text
rtk python -m pytest -q tests/unit tests/integration
389 passed, 8 warnings in 32.91s
```

The warnings are the maintained multiprocessing `fork()` deprecations from
`tests/integration/test_cli.py`.

## Commit status and concerns

This report and the scoped changes belong in the Task 7 feature commit. The
handoff records the exact commit hash after commit creation.

The pinned 0.49.2 wheel does not ship a CUDA 13.3 binary. This host requires
`BNB_CUDA_VERSION=130`; a CUDA version covered directly by the wheel does not
need that override. Building bitsandbytes from source offers the other path for
CUDA 13.3. No Task 7 correctness blocker remains on CPU or the tested CUDA 13.0
binary path.
