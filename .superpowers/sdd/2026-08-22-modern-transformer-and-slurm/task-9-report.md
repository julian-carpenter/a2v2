# Task 9 Report: SLURM Runtime Contracts and Coordinated Preemption

## Status and scope

Implemented and verified the pure scheduler/topology/runtime layer, checkpoint
topology and rank-local RNG state, shared-output ownership, and completed-update
preemption integration. The work deliberately excludes shell launchers and
documentation, which remain Tasks 10 and 11. Frozen reproduction scripts and
recipes were not changed.

The plan's illustrative CPU invocation of `tests/gpu/nccl_resume.py` was not
used: that harness is intentionally CUDA/NCCL-only. A dedicated two-rank Gloo
integration exercises safe-point/preemption behavior on CPU, while the existing
NCCL harnesses retain bounded CUDA-specific coverage.

This report and the scoped implementation are included in the Task 9 feature
commit; the exact commit hash is recorded in the parent handoff after commit
creation.

## Strict TDD evidence

### Pure scheduler and contract RED/GREEN

The module-presence test began with the expected missing-module failure:

```text
tests/unit/test_slurm.py::test_slurm_runtime_module_is_available
1 failed (`find_spec("a2v2.slurm") is None`)
```

After adding an importable empty module, the new environment, rendezvous, and
run-contract tests were added before their APIs:

```text
13 failed, 1 passed
```

The failures named absent `SlurmEnvironment`, homogeneous world-size,
rendezvous, and run-contract APIs. Minimal pure implementations produced:

```text
14 passed
```

Torchrun parsing, early SLURM/CUDA validation, and checkpoint-topology tests
then began RED:

```text
15 failed, 14 passed
```

Two test fixtures initially encoded invalid local-rank arithmetic and were
corrected before production behavior was changed. The complete selection then
passed:

```text
29 passed
```

### Output locking RED/GREEN

Ownership, contention, concurrent acquisition, and explicit stale recovery
tests were written before lock APIs:

```text
4 failed, 29 passed
```

Atomic candidate/link ownership and fingerprinted recovery made the selection
green:

```text
33 passed
```

Final self-review found a release/recovery race: an old owner could validate its
inode, pause, and later unlink a replacement owner's lock. A deterministic
threaded test paused release after its first owner read and demonstrated RED:

```text
rtk python -m pytest -q tests/unit/test_slurm.py::test_output_lock_release_cannot_unlink_a_recovered_new_owner
1 failed
assert not replacement_done.wait(timeout=0.2)
```

Serializing acquire, recovery, and release on the output-directory inode fixed
the race:

```text
rtk python -m pytest -q tests/unit/test_slurm.py::test_output_lock_release_cannot_unlink_a_recovered_new_owner
1 passed in 2.63s
```

The final pure/invariant selection was:

```text
rtk python -m pytest -q tests/unit/test_slurm.py tests/unit/test_documentation.py tests/unit/test_repository_layout.py
56 passed in 6.37s
```

### Signals, RNG, and coordinated checkpoint RED/GREEN

Signal-handler, handler-lifetime, local exit-contract, rank-local RNG, and CUDA
cardinality tests were added first:

```text
6 failed, 33 passed
```

The failures were the missing flag/exit APIs, absent rank-local RNG schema, and
the historical comparison of CUDA RNG count to global world size. After the
narrow runtime and restore changes:

```text
39 passed
```

The initial two-rank Gloo integration failed because the centralized workflow
checkpoint writer did not exist. After that writer was introduced, the test was
strengthened and failed again on the missing completed-update safe-point helper.
The workflow integration tests then independently began RED with:

```text
- uncaught `TrainingPreempted` in the local resume/preemption path;
- `topology=None` in an ordinary new checkpoint; and
- no shared-output preparation/ownership helper in the Gloo path.
```

The final implementation makes all ranks reduce the request, participate in
state gathers, observe rank zero's write result, barrier, flush, and raise the
same exit contract. The dedicated final gate is:

```text
rtk python -m pytest -q tests/integration/test_slurm_launcher.py
1 passed in 14.35s
```

Its real two-process Gloo run proves that a request set only on global rank one
causes both ranks to checkpoint after completed update one, exactly one writer
creates the v2 checkpoint, both ranks restore their own exact RNG state, the
topology contains both ranks, both flush, and both observe exit code 75.

## State schemas and compatibility

No additional checkpoint format boundary was created. Existing v2 payloads now
populate the reserved `topology` slot. V1 checkpoints and earlier v2/local
checkpoints with `topology=None` remain readable.

Topology uses schema `a2v2.topology.v1`:

```text
{
  "schema": "a2v2.topology.v1",
  "world_size": W,
  "by_rank": [
    {
      "hostname", "global_rank", "local_rank", "local_world_size",
      "visible_cuda_devices", "selected_cuda_device",
      "cuda_device_name", "cuda_device_uuid"
    }
  ]
}
```

Records are ordered by contiguous global rank and are JSON-safe. Resume rejects
schema, rank-array, or world-size incompatibility. Host, local assignment,
visible-device, ordinal, name, and UUID changes are surfaced as explicit
numerical-equivalence warnings rather than using hostname order to infer rank.

Distributed RNG uses schema `a2v2.rng.rank-local.v1`:

```text
{
  "schema": "a2v2.rng.rank-local.v1",
  "world_size": W,
  "by_rank": [python/NumPy/CPU-Torch/local-CUDA RNG state for each global rank]
}
```

The reader accepts the prior untagged rank wrapper and the unchanged local
single-process RNG mapping. Restore validates the whole selected state before
mutating generators. CUDA RNG cardinality is compared with the rank's locally
visible CUDA devices, never global world size.

The output lock uses schema `a2v2.output-lock.v1` with exact `job_id`, `run_id`,
`hostname`, and `pid`. Run contracts contain job/phase topology, rendezvous,
output directory, and manifest fingerprint fields and use canonical JSON SHA256
fingerprints plus deterministic per-field differences.

## Runtime topology and ownership semantics

An ordinary process with no torchrun variables retains rank/local-rank zero and
world/local-world size one. Any partial torchrun environment fails and names
the missing variables. Complete launches validate rank bounds,
`rank % local_world_size == local_rank`, and homogeneous divisibility before
device selection or process-group initialization.

When SLURM variables are present, job/node/process counts are required and the
derived torchrun node, world, and local-world topology must match the allocation.
CUDA launches additionally require a valid local ordinal and matching visible
device cardinality. A non-SLURM local single-GPU launch may intentionally leave
additional devices visible.

Only global rank zero creates the shared output directory and, for a SLURM run,
acquires `.a2v2-output.lock`. It broadcasts success or the actionable ownership
failure, then all ranks barrier and independently confirm directory visibility.
Lock creation writes and fsyncs a complete candidate, then hard-links it
exclusively to the deterministic lock name. Acquire, recovery, and release are
serialized with an advisory lock on the stable output-directory inode.

Contention never auto-steals ownership. Explicit recovery requires the exact
owner fingerprint and proven-dead same-host PID; a live owner, changed record,
or unprovable remote-host liveness fails closed. Release verifies the exact
inode and owner record while holding the directory mutex, so stale owners cannot
unlink replacements. Locks are released from the public workflow's `finally`
path after success, preemption, or error.

## Signal and safe-point semantics

`SIGUSR1` and best-effort `SIGTERM` handlers only assign primitive process-local
flag/signal fields. They perform no allocation, I/O, logging, checkpointing, or
collective. Prior process handlers are restored after the training call.

Every successful completed update reaches the same safe point before periodic
save or validation work. Ranks reduce request bits with `MAX`; CUDA jobs use the
timed Gloo checkpoint/control group and CPU tensors, while CPU Gloo jobs use the
default group. If requested, all ranks build checkpoint payloads and gather
rank-local RNG and topology state. Only global rank zero atomically writes
`checkpoint_last.pt`.

A rank-wide `MIN` success decision follows the write. Any failure raises a
checkpoint error and refuses the requeue-friendly exit. Success is followed by
a barrier and process-local log flush, then every rank raises
`TrainingPreempted`. The CLI records the valid checkpoint/update and returns 75;
no rank checkpoints from signal context or takes this exit without the common
checkpoint decision. Task 6 terminal-engine checkpoint guards remain in place.

## Final verification

Full CPU unit and integration regression, including ordinary non-SLURM CLI,
checkpoint, exact-resume, reproduction-driver, and two-rank Gloo paths:

```text
rtk python -m pytest -q tests/unit tests/integration
450 passed, 8 warnings in 59.40s
```

The eight warnings are the existing multiprocessing `fork()` deprecation
warnings from multiworker CLI tests. An earlier full run exposed only two new
repository invariants (nested test-helper docstrings and the module-layout
allowlist); both were corrected narrowly before this final run.

Task 6 optimizer/engine/resume focus was also rerun during implementation:

```text
53 passed in 7.29s
```

One earlier combined run had a non-reproducible pointer-address probe failure;
the isolated probe and the complete Task 6 focus immediately passed without a
production change. No Task 6 terminal, checkpoint, or numerical guard was
relaxed.

Bounded two-GPU NCCL collective, topology, and rank-local RNG probe:

```text
rtk env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_probe.py --output-dir /tmp/a2v2-task9-nccl-probe.4hpizk
exit 0; both ranks pass=true, rank_rng_restore_exact=true, topology_ok=true
```

Bounded two-GPU exact checkpoint/resume regression:

```text
rtk env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task9-nccl-resume.Pftgzj
exit 0; both ranks local_exact/all_ranks_exact/all_ranks_adagc_state/topology_ok=true
```

The Task 6 AdaGC digest remains
`6ad554301ec6f72fafa49579394ac4803ae90f438a679f94728102ef79afc012`.
Both CUDA harnesses report topology schema `a2v2.topology.v1` and RNG schema
`a2v2.rng.rank-local.v1`.

Static and patch checks:

```text
rtk python -m compileall -q a2v2 tests
rtk git diff --check
```

Both exited successfully without diagnostic output.

## Unrun site-dependent gate and concerns

- This node cannot certify real multi-node SLURM behavior. The required
  one-node launcher parity and at least 2-node by 2-GPU SLURM acceptance run for
  NCCL/Gloo transport, shared-filesystem locking, checkpoint/resume, and real
  scheduler preemption remains explicitly unrun.
- The lock protocol requires the site's shared filesystem to provide the POSIX
  hard-link, atomic-replacement, and advisory-lock semantics used by the local
  race tests. The real-cluster gate must verify those properties.
- NCCL emitted the maintained harness's device-selection and
  `find_unused_parameters=True` warnings; exact results and cross-rank state
  agreement were unaffected.
- `SIGTERM` remains best effort by design; exact recovery is advertised through
  scheduler-provided `SIGUSR1` lead time in the later launcher/documentation
  tasks.
- No unresolved Task 9 correctness blocker remains within locally testable
  scope.

## Fix Round 1

### Review findings and root cause

Independent review found that the initial distributed writer had multiple
exception boundaries between collectives. A rank-local payload or RNG
serialization failure could send that rank to the preemption success
`all_reduce` while peers remained in `gather_object`; rank-zero RNG decode could
skip the later topology gather; and a rank-zero periodic/final save failure had
no outcome broadcast, so peers could continue toward a terminal barrier. The
initial post-write `MIN` description above is therefore superseded by the
staged protocol in this fix round.

Shared-output visibility also remained a rank-local decision after the creation
barrier. Finally, RNG restore validated types up front but installed Python and
NumPy state before Torch/CUDA state setters had proved their payloads valid.

### Distributed checkpoint and visibility RED

The real two-process Gloo harness gained bounded process-group timeouts and
rank-specific failure injection. A nonzero payload-preparation exception first
demonstrated the mismatched payload-gather/preemption-all-reduce path:

```text
rtk python -m pytest -q tests/integration/test_slurm_launcher.py::test_two_rank_gloo_preemption_checkpoint_failures_are_common[prepare]
1 failed in 23.62s
rank 0: connection closed in preemption all_reduce
rank 1: timed out in preemption all_reduce while its peer had entered gather_object
```

Corrupt rank-one RNG bytes then demonstrated rank zero leaving before the
topology gather:

```text
rtk python -m pytest -q tests/integration/test_slurm_launcher.py::test_two_rank_gloo_preemption_checkpoint_failures_are_common[merge]
1 failed in 24.27s
rank 0/1: mismatched-collective timeout/connection closure
```

Writer and visibility coverage was added in the same test-first cycle:

```text
rtk python -m pytest -q tests/integration/test_slurm_launcher.py::test_two_rank_gloo_preemption_checkpoint_failures_are_common[write] tests/integration/test_slurm_launcher.py::test_two_rank_gloo_ordinary_writer_failure_is_common tests/integration/test_slurm_launcher.py::test_two_rank_gloo_output_visibility_failure_is_common
2 failed, 1 passed in 36.27s
```

The existing preemption wrapper happened to synchronize a rank-zero write
failure, so that case passed. The ordinary writer failed RED with rank zero
raising the injected `OSError` while rank one continued, and visibility failed
RED with rank zero reporting success while only rank one reported the missing
directory.

### Staged checkpoint protocol and failure matrix

Every distributed checkpoint call, including periodic, epoch, best, final, and
preemption saves, now uses one protocol:

1. Each rank builds its engine payload, serializes its local RNG, and prepares
   JSON-safe topology inside a local `try` block.
2. All ranks `all_gather_object` readiness/error records. Any local failure
   becomes the same deterministic `CheckpointError` on every rank before state
   gathering starts.
3. Ready ranks enter one symmetric `gather_object` carrying both encoded RNG and
   topology. Combining them removes the decode-between-gathers hazard.
4. Rank zero alone validates/merges every bundle, decodes RNG, builds the two
   tagged schemas, validates the complete v2 payload, serializes it, and performs
   the atomic write.
5. Rank zero broadcasts its sole writer outcome. Every rank either returns from
   the writer on common success or raises the same error. Only common success may
   reach the preemption barrier/log flush/exit 75 or an ordinary later barrier.

Protocol collectives are deliberately outside local and writer exception
handlers. Collective transport failure or process death therefore remains
launcher-fatal rather than attempting another collective on a damaged group.

| Injected boundary | Common result | Checkpoint/requeue result |
| --- | --- | --- |
| Rank-one payload preparation | Preparation error naming rank one on both ranks | No state gather, file, flush, or requeue exit |
| Rank-one corrupt RNG / rank-zero decode | Merge error broadcast from rank zero | No file, flush, or requeue exit |
| Rank-zero preemption save | Write error broadcast from rank zero | No valid file, flush, or requeue exit |
| Rank-zero ordinary save | Write error broadcast from rank zero | No peer continues toward the terminal barrier |
| Rank-one output invisibility | Missing rank/hostname list on both ranks | No rank enters training |

Output creation/lock acquisition still belongs only to global rank zero. After
its creation-status broadcast and barrier, every rank reports `(rank, hostname,
visible)` through a common gather. Missing or malformed reports produce the same
actionable list on every rank.

Final Gloo GREEN for the happy path plus all five failures:

```text
rtk python -m pytest -q tests/integration/test_slurm_launcher.py
6 passed in 72.15s
```

The happy case retains exactly one writer, tagged RNG/topology round trip,
completed update one, synchronized flush, and exit 75 on both ranks.

### Atomic RNG restore RED/GREEN

Mutation sentinels capture Python, NumPy, and CPU Torch globals, then corrupt a
later NumPy field, CPU Torch byte state, or CUDA byte state. The first invocation
had a test-only missing `random` import (`3 failed`); after correcting that
setup, behavior-level RED was:

```text
rtk python -m pytest -q tests/unit/test_slurm.py::test_rng_restore_validates_every_late_state_before_mutating_any_generator
3 failed in 5.78s
```

The malformed NumPy value raised raw `ValueError` after Python installation;
the malformed Torch bytes raised raw `RuntimeError` after Python and NumPy
installation; and malformed CUDA bytes reached the global CUDA installer after
all CPU generators changed.

Restore now performs a complete validation phase with isolated instances:

- `random.Random().setstate` validates Python state;
- an independent `numpy.random.RandomState` validates the fully converted NumPy
  tuple and key array;
- an independent CPU `torch.Generator` validates cloned CPU bytes; and
- one independent CUDA `torch.Generator` per locally visible device validates
  cloned rank-local CUDA bytes.

Only after every probe succeeds are Python, NumPy, CPU Torch, and CUDA global
states installed in that order. Malformed payloads are normalized to
`CheckpointError` and leave all earlier globals exact.

```text
rtk python -m pytest -q tests/unit/test_slurm.py::test_rng_restore_validates_every_late_state_before_mutating_any_generator
3 passed in 6.09s

rtk python -m pytest -q tests/unit/test_slurm.py tests/unit/test_checkpoint.py tests/unit/test_documentation.py tests/unit/test_repository_layout.py
67 passed in 7.94s
```

An older CUDA-cardinality test used one-byte placeholder states. Its CUDA
generator dependency was corrected to accept those fixtures so that the test
continues to isolate its intended local-visible-device versus global-world-size
contract; malformed-byte behavior is now covered separately.

### Fix Round 1 final verification

Task 6 optimizer transaction, terminal-engine, checkpoint, and exact local
resume focus:

```text
rtk python -m pytest -q tests/unit/test_optim.py tests/unit/test_engine.py tests/integration/test_resume.py
94 passed in 7.22s
```

Complete CPU regression, including ordinary non-SLURM paths and all six Gloo
subprocess cases:

```text
rtk python -m pytest -q tests/unit tests/integration
458 passed, 8 warnings in 100.94s
```

The warnings remain the existing multiworker `fork()` deprecations.

Bounded two-GPU NCCL collective/topology/RNG validation:

```text
rtk env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_probe.py --output-dir /tmp/a2v2-task9-fix1-nccl-probe.VlS7BI
exit 0; both ranks pass/rank_rng_restore_exact/topology_ok=true
```

Bounded two-GPU exact resume:

```text
rtk env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task9-fix1-nccl-resume.L4ZIgF
exit 0; both ranks local_exact/all_ranks_exact/all_ranks_adagc_state/topology_ok=true
```

Both CUDA harnesses retained schemas `a2v2.rng.rank-local.v1` and
`a2v2.topology.v1`. The AdaGC digest remains
`6ad554301ec6f72fafa49579394ac4803ae90f438a679f94728102ef79afc012`.

`rtk python -m compileall -q a2v2 tests` and `rtk git diff --check`
both exited cleanly before the report append and are rerun on the final staged
diff before commit.

The real multi-node SLURM/shared-filesystem/scheduler-preemption gate remains
unrun on this node. Fix Round 1 does not change that site-dependent concern. No
job longer than the bounded local CPU/CUDA gates was launched.

This appendix is included in the Fix Round 1 commit; its exact hash and clean
post-commit status are recorded in the parent handoff.
