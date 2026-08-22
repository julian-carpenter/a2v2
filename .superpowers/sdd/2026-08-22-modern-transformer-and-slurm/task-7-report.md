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
omitted `weight_decay_end` uses `0.0` for an opt-in cosine schedule. The
constant default preserves the current fixed decay. Values clamp at `w_end`
after `max_update`.

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

## Fix Round 1

### Findings and TDD evidence

The review found that the cosine implementation treated an omitted endpoint as
the starting decay. The new unit and extended-horizon resume tests failed at
update 2 of 4 because both observed `0.2` instead of `0.1`:

```text
rtk python -m pytest -q tests/unit/test_optim.py::test_cosine_weight_decay_omitted_end_defaults_to_zero tests/integration/test_resume.py::test_cosine_weight_decay_resume_recomputes_extended_horizon_before_next_update
2 failed in 7.02s
```

Changing the cosine-only fallback to `0.0` made both new tests pass. The same
run included the legacy constant control and same-horizon exact-resume test:

```text
4 passed in 6.59s
```

The bitsandbytes review tests then exposed seven setup-boundary failures. The
old builder returned a generic missing-extra error, accepted versions 0.48.9
and 0.50.0, leaked `AttributeError` for missing optimizer APIs, ignored an
unloaded CUDA native library, and leaked a constructor `TypeError`:

```text
rtk python -m pytest -q tests/unit/test_optim.py -k bitsandbytes
7 failed, 7 passed, 35 deselected in 4.11s
```

The CUDA 13.3 negative gate reproduced the installed wheel's missing
`libbitsandbytes_cuda133.so` and failed before A2V2 defined a targeted setup
error. After implementation, the unit boundary and host CUDA gate passed:

```text
rtk python -m pytest -q tests/unit/test_optim.py -k bitsandbytes
14 passed, 35 deselected in 3.74s

rtk env -u BNB_CUDA_VERSION python -m pytest -q tests/gpu/test_cuda_training.py::test_bitsandbytes_cuda_missing_native_binary_fails_during_setup
1 passed in 6.90s
```

### Corrected endpoint and extended-horizon resume

Cosine scheduling now uses `w_end=0.0` when the endpoint is omitted. Constant
scheduling still returns `None` before it touches optimizer groups, so the
legacy default keeps its class, group order, update-zero decay, optimizer state,
checkpoint slot, and numerical resume behavior.

The allowed `optimization.max_update` extension uses the active horizon when
the engine restores scheduler state. The test saves update `u=2` under `U=4`,
where omitted-end cosine decay is `w(2)=0.1`. Restoring under `U=8` keeps the
saved clock `last_update=2` and recomputes the live decay as:

```text
w(2) = 0.1 * (1 + cos(pi * 2 / 8))
```

The LR scheduler uses its existing one-update warm-up clock, so restore applies
`lr(2) = 1e-4 + 0.5 * (1e-2 - 1e-4) * (1 + cos(pi * 1 / 7))`. A zero-gradient
optimizer update then proves `theta_3 = theta_2 * (1 - lr(2) * w(2))`. Only
after that successful optimizer step does the decay scheduler commit
`w(3) = 0.1 * (1 + cos(pi * 3 / 8))`. The same-horizon test still compares each
next model, optimizer, LR scheduler, decay scheduler, and teacher state exactly.

### bitsandbytes setup-error contract

`OptimizerSetupError` now covers these selected-optimizer setup failures:

- import failures name the `a2v2[bnb]` extra;
- installed versions outside `bitsandbytes>=0.49,<0.50` report the required and
  detected versions;
- missing `bitsandbytes.optim`, `Adam8bit`, or `AdamW8bit` symbols identify the
  missing API;
- an unloaded CUDA native library reports the bitsandbytes version and PyTorch
  CUDA version, then recommends a compatible install or source build and the
  `BNB_CUDA_VERSION` override where a compatible toolkit exists;
- constructor and backend compatibility exceptions retain their cause under a
  targeted initialization message.

The builder validates `min_8bit_size` and the selected device before importing
bitsandbytes, so those A2V2 messages keep their prior types and text. It also
re-raises `CheckpointError`, `DistributedOptimizerStepError`,
`OptimizerSetupError`, `KeyboardInterrupt`, and `SystemExit` without translation.
The unit suite asserts the distributed terminal exception object by identity.

With `BNB_CUDA_VERSION=130`, bitsandbytes 0.49.2 loads its CUDA 13.0 native
binary on this CUDA 13.3 host. Both 8-bit optimizers and both AMP-overflow
transaction cases pass; the no-override negative test skips under the selected
override:

```text
rtk env BNB_CUDA_VERSION=130 python -m pytest -q tests/gpu/test_cuda_training.py -k 'bitsandbytes or overflow'
4 passed, 1 skipped, 5 deselected in 6.99s
```

### Regression evidence and commit status

The focused optimizer, transaction, resume, TensorBoard, and CLI suite passed:

```text
101 passed, 8 warnings in 9.96s
```

The explicit packaging, lazy-import, constant-schedule, and global-clipping
controls passed `4 passed in 3.62s`. The full CPU unit and integration run
exited with 397 passing tests; collection confirmed `397 tests collected in
5.72s`. `git diff --check` reported no whitespace errors.

The repository records the source, tests, and this report in one Fix Round 1
commit. The handoff supplies its immutable hash because a commit cannot contain
its own hash. The remaining deployment constraint is the upstream 0.49.2 wheel:
it lacks a CUDA 13.3 native binary, so this host needs the tested CUDA 13.0
override or a compatible source build.
