# Task 6 Report: AdaGC and Checkpoint Format v2

## Status

Implemented and verified on the directly authorized `main` checkout. Production
changes are limited to `a2v2/training.py` and `a2v2/workflows.py`; coverage is
limited to the optimizer, engine, resume, bounded CUDA, and maintained NCCL
resume tests named by the Task 6 scope. Frozen reproduction scripts and recipes
were not changed.

This report and the scoped implementation are included in the Task 6 feature
commit; the exact commit hash is recorded in the handoff after commit creation.

## Strict TDD evidence

The pre-change focused baseline was:

```text
rtk python -m pytest -q tests/unit/test_optim.py tests/unit/test_engine.py tests/integration/test_resume.py
8 passed in 6.97s
```

### Clipper factory and AdaGC RED/GREEN

Tests for fixed/global/none construction, paper defaults, warm-up, post-warm-up
scaling, state round trips, tied/no/zero/nonfinite/sparse gradients, and malformed
state were added before the implementation.

```text
rtk python -m pytest -q tests/unit/test_optim.py -k 'gradient_clipper or adagc'
4 failed, 5 deselected
```

The intended RED was the absent `build_gradient_clipper` API. After the minimal
implementation, the same command produced:

```text
4 passed, 5 deselected
```

### Optimizer transaction RED/GREEN

An engine test first injected AdaGC and an optimizer failure and asserted that
the tentative clip state, engine update, scheduler, and teacher clocks did not
commit.

```text
rtk python -m pytest -q tests/unit/test_engine.py -k adagc
1 failed, 2 deselected
```

The intended RED was `TrainingEngine.__init__()` rejecting the new
`gradient_clipper` argument. After transaction integration:

```text
1 passed, 2 deselected
```

### Checkpoint v2 and exact resume RED/GREEN

The v1 upgrade, new-v2-save, required AdaGC-state, and AdaGC exact-resume tests
were added before checkpoint implementation.

```text
rtk python -m pytest -q tests/integration/test_resume.py -k 'v1 or v2 or adagc'
4 failed, 1 deselected
```

Failures were the intended missing behaviors: no v1 normalization, new payloads
still used v1, missing AdaGC state was accepted, and interrupted AdaGC state
drifted from uninterrupted training. After implementation:

```text
4 passed, 1 deselected
```

Workflow factory wiring and compatibility fingerprints then began RED:

```text
rtk python -m pytest -q tests/integration/test_resume.py -k 'workflow_wires or compatibility_rejects'
2 failed, 5 deselected
```

Both failures were missing workflow/checkpoint compatibility helpers. After
wiring and early compatibility validation:

```text
2 passed, 5 deselected
```

### Review-driven RED/GREEN

Self-review found that the diagnostic `largest_scale` was reporting the most
aggressive scaling factor instead of the literal largest tensor scale. The
existing equation test was strengthened first:

```text
rtk python -m pytest -q tests/unit/test_optim.py::test_adagc_uses_global_warmup_then_previous_ema_and_clipped_norm
1 failed
```

The observed value was approximately `0.5` while one tensor was unscaled and
the required largest value was `1.0`. After the narrow correction:

```text
1 passed
```

## Algorithm and behavior

`build_gradient_clipper` provides three constructor-constant strategies:

- `global` retains the exact prior `torch.nn.utils.clip_grad_norm_` call,
  parameter order, configured maximum norm, optimizer ordering, and stateless
  checkpoint behavior.
- `none` computes the infinity-norm diagnostic without scaling gradients.
- `adagc` defaults to `beta=0.99`, `lambda_rel=1.04`, and 100 successful warm-up
  updates, following arXiv:2502.11034.

During warm-up, AdaGC applies the same global clipping path. Each observed
per-tensor running minimum is initialized from that tensor's post-global-clip
norm. For tensor `i` after warm-up, using only the previously committed state:

```text
h_(t,i) = min(lambda_rel * gamma_(t-1,i) / ||g_(t,i)||, 1)
g'_(t,i) = h_(t,i) * g_(t,i)
gamma_(t,i) = beta * gamma_(t-1,i) + (1 - beta) * ||g'_(t,i)||
```

The state owns one CPU FP32 scalar per canonical named trainable tensor. Tied or
shared parameters are tracked once under the first canonical name. No-gradient
tensors remain unobserved (`+inf`) and do not change state; their first later
observation is unscaled and initializes the EMA. A zero norm uses scale one and
updates the EMA with zero. Sparse gradients are rejected actionably before any
mutation. All dense norms are preflighted before mutation, so any nonfinite norm
returns a noncommittable candidate without changing finite gradients or state.

## Transaction and distributed evidence

Clipping produces a tentative `GradientClipCandidate`; AdaGC state and its
successful-update clock commit only after `optimizer.step()` returns. AMP first
unscales, then clips. On GradScaler overflow/nonfinite gradients, the skipped
step updates the scaler but leaves model, optimizer, clipper, scheduler,
teacher/EMA, and engine update coherent. An injected CPU optimizer exception
independently proves that a failed optimizer step cannot commit those clocks.

The CUDA overflow test is parameterized for both the frozen global behavior and
AdaGC. It verifies unchanged model/update/scheduler/teacher state and exact
AdaGC state on overflow, followed by one coherent successful update.

The maintained two-rank NCCL resume harness uses AdaGC, saves after update one,
resumes to update two, and independently all-gathers the AdaGC-state digest over
the Gloo checkpoint group. Final evidence showed `local_exact=true`,
`all_ranks_exact=true`, and `all_ranks_adagc_state=true` on both ranks, with the
shared clipper digest
`6ad554301ec6f72fafa49579394ac4803ae90f438a679f94728102ef79afc012`.

Two harness defects were exposed before that GREEN result: the digest helper
could not byte-view scalar EMA tensors, then map-location loading returned those
scalars on CUDA while the clipper requires CPU ownership. The helper now safely
flattens scalars and the strict loader validates mapped values before reclaiming
CPU FP32 ownership.

## Checkpoint format and legacy compatibility

All production writers now emit format v2. Its required state slots are
`gradient_clipper`, `weight_decay_scheduler`, `topology`, and
`resume_compatibility`; the latter two are reserved for later scoped tasks.
Atomic replacement and distributed rank-zero save behavior were not changed.

The loader accepts v1, validates the complete legacy required-key set, and
normalizes missing v2 slots to legacy defaults while retaining source format
provenance. Parameterized v1 tests cover missing optimizer, update, and RNG
entries. V2 validation rejects every missing new slot. Stateful AdaGC resume
fails closed if clipper state is absent after update zero, or if its algorithm
version, update, parameter names, keys, scalar shape/dtype/value, or update clock
is incompatible. Stateless global/none strategies reject unexpected state.

The resume compatibility fingerprint covers mathematical model, optimizer,
scheduler, clipping, seed, relevant dataset batching, and requested world-size
fields. It deliberately excludes `optimization.max_update`, preserving the
existing supported contract of extending the training horizon at resume. V1
checkpoints derive legacy-default compatibility values; mismatch errors identify
the first actionable field. Exact legacy global resume numerics and existing
inference, conversion, checkpoint, CLI, and reproduction paths remain green.

## Final verification

Final focused CPU command:

```text
rtk python -m pytest -q tests/unit/test_optim.py tests/unit/test_engine.py tests/integration/test_resume.py
20 passed in 7.14s
```

Legacy checkpoint, inference, conversion, and frozen reproduction coverage:

```text
rtk python -m pytest -q tests/unit/test_checkpoint.py tests/integration/test_inference.py tests/integration/test_checkpoint_conversion.py tests/integration/test_reproduction_driver.py
56 passed in 23.46s
```

Full CPU regression:

```text
rtk python -m pytest -q tests/unit tests/integration
331 passed, 8 warnings in 32.75s
```

The eight warnings are the existing multiprocessing `fork()` deprecation
warnings from `tests/integration/test_cli.py`.

Bounded CUDA overflow coverage:

```text
rtk python -m pytest -q tests/gpu/test_cuda_training.py -k 'adagc or overflow'
2 passed, 5 deselected in 5.60s
```

Two-rank exact resume and independent clipper-state agreement:

```text
rtk python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task6-nccl-final.SyMQRa
exit 0; both ranks local_exact/all_ranks_exact/all_ranks_adagc_state true
```

The brief's illustrative `tests/gpu/test_distributed_smoke.py` path does not
exist in this repository; the maintained `tests/gpu/nccl_resume.py` harness was
used for the required distributed resume gate.

Static and diff checks:

```text
rtk python -m compileall -q a2v2 tests
rtk git diff --check
```

Both exited successfully without output.

## Concerns

- NCCL emitted the harness's existing device-selection and
  `find_unused_parameters=True` warnings; exact results and cross-rank AdaGC
  state agreement were unaffected.
- The compatibility fingerprint intentionally stores the complete normalized
  values needed for early mismatch detection. Later Task 9 topology work should
  populate the reserved topology slot rather than creating another format
  boundary.
- No unresolved Task 6 correctness blocker remains.
