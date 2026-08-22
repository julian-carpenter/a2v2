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
provenance. Parameterized v1 tests cover every missing legacy required key.
V2 validation rejects every missing new slot. Stateful AdaGC resume
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

## Fix Round 1

### Status and design rationale

Independent review identified three real gaps: step/skip decisions were local to
each rank, workflow compatibility hid the designed v1/update-zero AdaGC
exception, and restore installed clipper state before comparing clocks. The
bounded correction touches the same two production files and three existing
Task 6 test files. Frozen reproduction files remain unchanged.

A rank-local optimizer exception may occur after other ranks have already
mutated their model and optimizer. Snapshotting every model and optimizer tensor
before every step would impose unacceptable steady-state memory and throughput
cost. The implemented contract is therefore process-terminal:

- every rank attempts its local optimizer step, then reduces a success bit
  before any GradScaler, clipper, engine-update, scheduler, or teacher clock can
  commit;
- if any rank reports failure, every rank raises the same
  `DistributedOptimizerStepError`, preserves the original exception as
  `__cause__` on the failing rank, and marks its engine terminal;
- terminal engines reject every later `step()` and `checkpoint_payload()` call,
  so partially advanced model/optimizer state cannot become a new checkpoint;
- recovery is from the last atomic checkpoint in a new process, not by silently
  continuing a potentially divergent run.

Loss finiteness and gradient finiteness/AMP skip decisions also reduce rank-wide.
A rank-local AMP overflow skips every optimizer, applies the same scale backoff,
and leaves model, optimizer, clipper, scheduler, teacher, and update state fixed.

### Distributed transaction RED/GREEN

The two-rank optimizer-failure mode was written first. Initial RED:

```text
rtk python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task6-fix-red-opt.utdXpo --failure-mode optimizer --failure-rank 1
exit 1
rank 0: no error, clocks_unchanged=false, checkpoint/future step allowed
rank 1: injected RuntimeError, clocks_unchanged=true
```

After adding the common terminal decision, the test was strengthened to include
the GradScaler checkpoint clock. That produced a second RED: both ranks raised
the terminal error, but rank 0 reported `clocks_unchanged=false` because its
scaler updated before global success. Moving `GradScaler.update()` behind the
collective produced final GREEN:

```text
rtk python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task6-fix-red-opt.utdXpo --failure-mode optimizer --failure-rank 1
exit 0
both ranks: all_ranks_terminal/checkpoint_blocked/future_step_blocked/clocks_unchanged=true
rank 1 local cause: injected rank-local optimizer failure
```

The rank-local AMP overflow test began RED with rank 0 committing update 1 at
scale 8 while rank 1 skipped at update 0 and backed off to scale 4:

```text
rtk python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task6-fix-red-amp.Vlh45k --failure-mode amp-overflow --failure-rank 1
exit 1; all_ranks_skipped=false
```

Final GREEN:

```text
rtk python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task6-fix-red-amp.Vlh45k --failure-mode amp-overflow --failure-rank 1
exit 0; both ranks skipped=true, update=0, scale=4.0, state_unchanged=true
```

Loss finiteness received its own test-first cycle. The RED harness safely matched
the missing collectives so it reported the bug rather than hanging: rank 1
raised local nonfinite loss while rank 0 proceeded to the later global-gradient
error. After the rank-wide pre-backward decision, both ranks failed identically:

```text
rtk python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task6-fix-red-loss.eE12zu --failure-mode nonfinite-loss --failure-rank 1
RED: exit 1; rank 0 nonfinite gradients, rank 1 nonfinite loss
GREEN: exit 0; both ranks FloatingPointError at update 0, state_unchanged=true
```

### Schema, workflow, and atomic restore RED/GREEN

The initial exhaustive schema/restore selection was:

```text
rtk python -m pytest -q tests/unit/test_optim.py::test_adagc_rejects_every_malformed_state_without_mutation tests/integration/test_resume.py -k 'malformed_state or missing_legacy or missing_required_state or clock_mismatch or allows_only_v1 or past_update_zero'
7 failed, 31 passed, 7 deselected
```

The intended failures showed accepted extra/Boolean AdaGC fields, a missing
`format_version` error that did not name the key, and the blocked workflow
transition. Two restore cases initially had a missing test-only `deepcopy`
import; after correcting that setup, the restore/workflow RED was:

```text
rtk python -m pytest -q tests/integration/test_resume.py -k 'past_update_zero or clock_mismatch or allows_only_v1'
3 failed, 25 deselected in 6.90s
```

The failures proved that missing update-zero state retained live update 1,
clock mismatch installed update 2 before rejecting checkpoint update 1, and
workflow validation rejected `global -> adagc` before restore.

Final GREEN for the combined schema/restore selection:

```text
rtk python -m pytest -q tests/unit/test_optim.py::test_adagc_rejects_every_malformed_state_without_mutation tests/integration/test_resume.py -k 'malformed_state or missing_legacy or missing_required_state or clock_mismatch or allows_only_v1 or past_update_zero'
38 passed, 7 deselected in 7.29s
```

AdaGC now parses a strict, exact-key schema into a frozen candidate without
touching live state. It rejects missing/extra keys, Boolean/wrong versions,
Boolean/negative updates, wrong name order/type, missing/extra norm entries,
non-scalar or non-FP32 values, NaN, and negative infinity. Positive infinity
remains the explicit unobserved/no-gradient sentinel. Restore compares candidate
and checkpoint clocks first, restores the other checkpoint components, and only
then installs the candidate. Missing update-zero state installs explicit
pristine update-zero/+infinity state; missing state after update zero still
fails closed.

The workflow exception is deliberately narrow: source format must be v1,
checkpoint update must be the integer zero, saved method must be `global`, and
active method must be `adagc`. Update greater than zero and every unrelated
fingerprint mismatch remain rejected.

### Fix Round 1 verification

Focused CPU:

```text
rtk python -m pytest -q tests/unit/test_optim.py tests/unit/test_engine.py tests/integration/test_resume.py
58 passed in 4.65s
```

Full CPU:

```text
rtk python -m pytest -q tests/unit tests/integration
369 passed, 8 warnings in 33.25s
```

The warnings remain the existing multiprocessing `fork()` deprecations in the
CLI integration tests.

Bounded CUDA:

```text
rtk python -m pytest -q tests/gpu/test_cuda_training.py -k 'adagc or overflow'
2 passed, 5 deselected in 5.52s
```

Successful two-rank exact resume:

```text
rtk python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task6-fix-nccl-success.hhpkpA
exit 0; both ranks local_exact/all_ranks_exact/all_ranks_adagc_state=true
```

The clipper state digest remains
`6ad554301ec6f72fafa49579394ac4803ae90f438a679f94728102ef79afc012`.
`compileall` and `git diff --check` also exited successfully without output.

### Fix Round 1 concerns and commit status

- Terminal coordination covers ordinary Python optimizer exceptions that allow
  every process to reach the success collective. A process death, CUDA context
  loss, or collective failure remains the launcher/process group's fatal-error
  domain; no in-process protocol can guarantee a final collective in that case.
- A terminal failure deliberately does not roll back model/optimizer mutations
  on ranks whose local step returned. The engine makes those mutations
  uncheckpointable and unusable, and restart uses the last atomic checkpoint.
- NCCL retains the harness's existing device-selection and unused-parameter
  warnings; all exactness and failure-contract assertions pass.
- This report and the scoped fix are included in the Fix Round 1 commit; its
  exact hash is recorded in the handoff after commit creation.
