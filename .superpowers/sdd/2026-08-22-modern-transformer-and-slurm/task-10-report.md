# Task 10 Report: Separate SLURM Launchers

## Status and scope

Implemented and locally verified the two planned launch paths:

- `scripts/a2v2_slurm_node.sh`: one torchrun agent per allocated node;
- `scripts/reproduce_meerkat_slurm.sh`: separate pretraining, fine-tuning,
  and final-evaluation orchestration.

The change is limited to those launchers, their integration scaffolding, and
this report. Task 11 prose documentation and modern example configs remain
out of scope. The canonical local
`scripts/reproduce_meerkat_paper.sh` was not edited.

The scripts are included in the Task 10 commit. The exact commit hash is
reported in the parent handoff after commit creation.

## Strict TDD evidence

### Initial launcher RED

Mocked scheduler/Python tests, dry-run command tests, topology validation,
phase handoff, signal forwarding, recovery, and the frozen-driver hash guard
were added before either launcher existed:

```text
13 failed, 1 passed, 42 deselected in 15.93s
```

All 13 failures named the missing
`scripts/a2v2_slurm_node.sh` or
`scripts/reproduce_meerkat_slurm.sh`. The one passing test was the new
frozen-byte guard.

The node wrapper was implemented first. Its initial focused GREEN was:

```text
5 passed, 14 deselected in 7.67s
```

The first orchestration selections then passed:

```text
6 passed, 13 deselected in 9.50s
2 signal tests passed, 17 deselected in 8.85s
```

### Review-driven RED/GREEN cycles

Self-review identified a cross-node lock-preflight race. A deterministic test
put an already-created lock in front of node rank one and failed RED because
the slow peer mistook rank zero's current-run lock for contention:

```text
test_nonzero_node_accepts_the_lock_rank_zero_may_have_just_acquired
1 failed
ERROR: output lock already exists ...
```

Only node rank zero now checks pre-launch contention; all nodes still verify
shared path visibility. The real-contention and slow-peer cases passed
together:

```text
2 passed, 18 deselected in 4.84s
```

The preemption tests were strengthened for torchrun normalizing a worker's
exit 75 to an agent exit 1. The new case failed RED with process return code 1
instead of 75. The launcher now records that it forwarded `SIGUSR1`, requires
a newly atomically replaced checkpoint, validates its stage/world topology,
and only then returns 75:

```text
1 failed, 2 passed
4 passed, 18 deselected in 14.60s
```

The final four cases cover raw exit 75, normalized exit 1, checkpoint
validation failure, and refusal when no new checkpoint was published.

An explicit pretraining handoff initially failed RED because pretraining still
owned the default output directory while fine-tuning consumed the custom
checkpoint:

```text
test_explicit_pretraining_handoff_owns_the_directory_pretraining_writes
1 failed
```

The custom `checkpoint_last.pt` now determines pretraining's output owner and
the same exact file feeds fine-tuning. That case passed. A separate RED case
then proved that a custom handoff could collide with the fine-tuning directory;
the launcher now rejects identical stage output owners:

```text
test_pretraining_and_finetuning_cannot_share_one_output_owner
1 failed
1 passed after the distinct-owner validation
```

## Launcher topology and command contract

The outer training step renders one synchronous `srun` task per node:

```text
srun
  --nodes=N
  --ntasks=N
  --ntasks-per-node=1
  --gpus-per-node=G
  scripts/a2v2_slurm_node.sh PHASE ...
```

Each node wrapper invokes:

```text
python -m torch.distributed.run
  --nnodes=N
  --node-rank=SLURM_NODEID
  --nproc-per-node=G
  --rdzv-backend=c10d
  --rdzv-endpoint=HOST:PORT
  --rdzv-id=JOB_ID-PHASE
```

Therefore the configured global worker count is exactly `N * G`. The outer
launcher passes that literal through
`distributed_training.distributed_world_size`; the node wrapper checks local
`SLURM_GPUS_ON_NODE` and the cardinality of the inherited
`CUDA_VISIBLE_DEVICES`. Neither script assigns or rewrites
`CUDA_VISIBLE_DEVICES`.

The first hostname returned by
`scontrol show hostnames SLURM_JOB_NODELIST` is the real-mode default
rendezvous host. Users may instead provide an exact endpoint. The default port
is `15000 + cksum(JOB_ID) % 20000`, hence bounded to 15000..34999 and stable
across a requeue. Job 4815 rendered port 19780 in the bounded dry run.
Rendezvous IDs remain phase-unique even though sequential phases reuse the
same endpoint.

Both scripts use strict Bash mode and quoted arrays. Dry-run records use
`printf %q` and never `eval`; tests parse those records back with
`shlex.split` and cover paths containing spaces. The bounded two-node,
four-GPU dry run required no SLURM variables, CUDA, data, checkpoints, or
output directory and rendered:

- pretraining with world size 8;
- pretraining-checkpoint validation before fine-tuning;
- fine-tuning with world size 8 and the intended pretraining checkpoint;
- a distinct one-node, one-task, one-GPU final evaluation.

## Run contract, output ownership, and recovery

Every node receives the same job, phase, node/GPU counts, world size,
rendezvous endpoint/ID, absolute output directory, and manifest fingerprint.
The wrapper hashes those fields into
`A2V2_SLURM_CONTRACT_FINGERPRINT` and exports the exact contract fields to
workers. Real launches SHA-256 fingerprint the selected shared manifests;
dry-run deliberately records `dry-run-unverified`.

`A2V2_RUN_ID` is phase-specific:

```text
JOB-meerkat-fFOLD-FRACTION/pretrain
JOB-meerkat-fFOLD-FRACTION/finetune
```

Task 9 consumes that identity for its atomic
`.a2v2-output.lock`. Pretraining and fine-tuning own distinct directories and
therefore distinct `checkpoint_last.pt` files. Rank zero alone preflights an
existing lock before torchrun; Task 9 still performs the authoritative atomic
acquisition after distributed initialization.

Contention never auto-steals. `--recover-lock PHASE:FINGERPRINT` is explicit,
requires a 64-character lowercase SHA-256 owner fingerprint, and calls Task
9's proven-dead-owner recovery before any launch. Remote/unprovable liveness
continues to fail closed.

## Resume and phase handoff

Pretraining resumes `checkpoint_last.pt` automatically when present or uses
an explicitly selected pretraining-stage resume. Fine-tuning always receives
the intended pretraining `checkpoint_last.pt` through
`--pretrained-checkpoint`, while its own `checkpoint_last.pt` is the only
automatic fine-tuning resume. Selected checkpoints are validated for native
loadability, exact stage, and distributed RNG/topology world size before use.

Successful stages write a separate completion marker only after checkpoint
validation. A requeued `all` run can therefore skip a completed stage without
confusing checkpoint presence with completion. Preemption never writes a
completion marker.

## Signal and exit contract

The batch header requests `SIGUSR1` with 300 seconds of lead time:

```text
#SBATCH --signal=B:USR1@300
```

The batch shell runs synchronous `srun` in the background so its trap can
forward `SIGUSR1`. Official Slurm behavior forwards that signal from
synchronous `srun` to the step tasks. Task 9 workers only set a flag in signal
context and checkpoint together at the next completed-update safe point.

After the step exits, the shell accepts either raw status 75 or a nonzero
torchrun/srun status following its forwarded signal. It requires the checkpoint
inode/signature to differ from the pre-step state, loads and validates the new
checkpoint, and then returns 75. Missing, unchanged, wrong-stage, wrong-world,
or unloadable checkpoints produce exit 2 instead. The launcher never invokes
`scontrol requeue`; site policy decides whether exit 75 is requeued, avoiding
an unrecoverable automatic loop.

## Frozen local control

`scripts/reproduce_meerkat_paper.sh` remains byte-identical:

```text
SHA256 485b4b36fe1045174a392834bd9ee9848538ab496944631a0b2d514e22d4de18
```

The existing exact three-training-command launch-graph assertion and the new
byte guard both pass. The SLURM path is wholly separate.

## Final verification

Focused launcher, two-rank Gloo, reproduction-driver, quoting, recovery, and
frozen-control gate:

```text
rtk python -m pytest -q tests/integration/test_slurm_launcher.py tests/integration/test_reproduction_driver.py
62 passed in 119.71s
```

Fresh full CPU regression after all changes:

```text
rtk python -m pytest -q tests/unit tests/integration
478 passed, 8 warnings in 131.03s
```

The eight warnings are the existing multiprocessing `fork()` deprecation
warnings in multiworker CLI tests.

Static and bounded local checks:

```text
rtk bash -n scripts/a2v2_slurm_node.sh
rtk bash -n scripts/reproduce_meerkat_slurm.sh
rtk git diff --check
rtk bash scripts/a2v2_slurm_node.sh ... --dry-run
rtk bash scripts/reproduce_meerkat_slurm.sh ... --dry-run
```

All exited successfully. No real `sbatch`, CUDA training, or multi-node step
was started.

## Unrun site gate and concerns

- The required one-node real-SLURM parity and at least 2-node by 2-GPU
  NCCL/Gloo, shared-filesystem, resume, contention, and scheduler-preemption
  gate remains unrun.
- The site's SLURM version must support the requested batch/step signal and GPU
  options and expose integer `SLURM_GPUS_ON_NODE` plus a homogeneous
  `SLURM_GPUS_PER_NODE` form accepted by the launcher.
- The shared filesystem must provide the POSIX hard-link, atomic replacement,
  inode, advisory lock, and visibility semantics Task 9 and the checkpoint
  freshness gate assume.
- Actual torchrun/srun exit normalization is handled locally through the
  forwarded-signal plus new-valid-checkpoint contract, but must still be
  observed on the target site.
- Checkpoint validation loads the native checkpoint on the batch host with CPU
  mapping. The target submit node must have the repository environment and
  enough memory for that pre-requeue safety check.

## Fix Round 1

### Review findings and strict RED/GREEN evidence

Independent review correctly identified three trust-boundary defects in the
first implementation: existence-only completion markers, a Bash newline hash
that was not Task 9's canonical `RunContract`, and a preemption flag cleared
immediately before each step.

Marker/runtime-contract tests were added before their APIs. The first RED was:

```text
0 passed, 9 failed
ImportError: missing CompletionMarkerError, write_stage_completion,
validate_stage_completion, and config_fingerprint
```

The marker round-trip, partial/mismatch/stale checkpoint, interrupted atomic
write, and per-rank mismatch cases then passed:

```text
9 passed
```

A separate training-path test failed RED because model construction was
reached before exported contract validation. After wiring validation directly
into _`run_training`, before model construction and output locking:

```text
10 passed
```

The first runtime-manifest mutation test also failed RED because a changed
shared manifest was accepted:

```text
0 passed, 1 failed
Failed: DID NOT RAISE TopologyError
```

Every worker now recomputes its phase's shared-manifest identity. The runtime
config, manifest, and rank mismatch slice passed:

```text
4 passed
```

The first combined signal run exposed four heavyweight canonical-helper
processes before `srun`, causing two deterministic prelaunch tests to time out.
Both stage contracts are now produced in one canonical Python invocation, and
nodes do not recompute an outer-supplied bundle. The final marker/signal slice,
covering prelaunch, handoff, active-success, raw 75, normalized nonzero,
checkpoint failure, and missing checkpoint, passed:

```text
8 passed
```

### Atomic completed-stage record

The marker schema is `a2v2.stage-completion.v1`. Its exact JSON fields are:

- `schema`;
- the complete canonical `run_contract`;
- `run_contract_fingerprint`;
- the fully resolved `config_fingerprint`;
- `checkpoint`, containing absolute `path`, `size_bytes`, and SHA-256.

The stage, world size, output directory, rendezvous, and phase-specific
manifest identity are therefore bound through the embedded `RunContract`.
Writing uses a same-directory candidate, complete JSON plus newline, file
`fsync`, atomic `os.replace`, and directory `fsync`. An injected
replacement failure preserves the old marker and cleans the candidate.

Every standalone or `all` stage treats any existing marker as authoritative
state that must parse and validate. Wrong schema/keys, partial JSON, a
different contract/config, an unexpected checkpoint path, or changed
checkpoint bytes fails closed and asks for explicit removal or repair. A valid
record is reusable. Completion records are published only after the Task 9
checkpoint stage/world validation succeeds.

Final evaluation requires the active fine-tuning completion record and uses
its validated `checkpoint_last.pt`. It no longer prefers an unrelated
pre-existing `checkpoint_best.pt`; the mocked stale-best test verifies the
exact evaluation command.

### Canonical launcher/runtime contract

`python -m a2v2.slurm launcher-contracts` now uses Task 9's
`RunContract.to_mapping()` and `RunContract.fingerprint()` canonical JSON.
One call produces independent pretraining and fine-tuning identities. Each
manifest digest includes its logical name, absolute path, and file SHA-256.
Dry-run represents a missing manifest explicitly as null, so it remains
deterministic without data.

Each stage config fingerprint hashes `config_to_dict(load_config(...,
overrides))`, using the exact ordered overrides later passed to training.
Consequently `--phase all`, `--phase pretrain`, and `--phase finetune`
produce the same identity for the same selected stage. Integration tests
compare the node export to Python `RunContract.fingerprint()` and compare
`all` against both standalone selections.

The node exports the complete JSON, canonical fingerprint, config fingerprint,
and manifest fingerprint. On every worker _`run_training` checks complete
presence, parses the exact RunContract keys, recomputes the RunContract and
resolved-config fingerprints, recomputes the active manifests, and compares
phase, world size, output directory, rendezvous, node/process counts, and
scheduler job. This occurs before model construction or output-lock
acquisition. Rank-zero and rank-one mismatch tests both fail early.

Both launchers now accept an explicit bracketed IPv6 endpoint such as
`[2001:db8::1]:23451`, matching Task 9's parser.

### Latched preemption state machine

`PREEMPTION_REQUESTED` is monotonic and is never cleared. `SIGUSR1`
received during hashing, validation, recovery, checkpoint handoff, an active
step, or final marker publication therefore remains visible at the next safe
boundary.

Before and after every selected stage, a latched request exits 75 without
launching more work. If an already-completed stage exists, its marker and
checkpoint are revalidated first. During an active training step, the signal
is forwarded to synchronous `srun`; even status 0 requires checkpoint bytes
to differ from the pre-step SHA-256 and requires stage/world validation before
exit 75. A successful step publishes its completion record before that exit.
Raw 75 and scheduler-normalized nonzero behavior remain supported. The
launcher still never calls `scontrol requeue`.

### Fix Round 1 verification and status

Focused gates:

```text
tests/unit/test_slurm.py: 54 passed
tests/integration/test_slurm_launcher.py: 31 passed
tests/integration/test_reproduction_driver.py: 37 passed
marker/signal final slice: 8 passed
```

The first fresh full CPU run reached `494 passed, 1 failed, 8 warnings`; the
sole failure was the repository documentation gate naming two missing nested
test-helper docstrings. After adding those docstrings, the exact documentation
gate passed. The second fresh full CPU run completed cleanly:

```text
495 passed
```

Bash syntax and `git diff --check` pass. The frozen local driver remains:

```text
SHA256 485b4b36fe1045174a392834bd9ee9848538ab496944631a0b2d514e22d4de18
```

No `sbatch`, real multi-node step, CUDA training, auto-requeue, or
long-running job was started. The real-site gate and the site-specific
concerns listed above remain unrun/unchanged. Fix Round 1 is committed as the
single commit reported in the parent handoff; the report intentionally does
not embed a self-referential commit hash.

## Fix Round 2

### Root cause and RED evidence

The phase-specific `REQUIRED_MANIFESTS` validation was correct, but the next
`launcher-contracts` invocation unconditionally passed both configs and all
three manifests. The helper's parser also marked every stage option required,
so a valid standalone stage could not avoid loading the unrelated stage.

Real-mode regression tests were added before changing either interface. They
use the real canonical contract and marker implementations while mocking only
unavailable `srun` and native checkpoint loading. The RED result was:

```text
6 passed, 3 failed
pretrain: ERROR: fine-tuning config does not exist
finetune/evaluate: ERROR: pretraining config does not exist
```

The six passing cases proved selected pretraining, fine-tuning, and `all`
inputs still failed actionably when their own config or manifest was missing.

### Phase-selective single-process contract construction

`launcher-contracts` now accepts one or more explicit `--phase` values.
It validates and loads only the options for those phases, rejects duplicate
phase requests, and keeps the canonical four-line record per stage:

- manifest fingerprint;
- canonical RunContract JSON;
- RunContract fingerprint;
- resolved-config fingerprint.

The outer launcher selects contracts as follows:

```text
pretrain          -> pretrain
finetune          -> finetune
evaluate          -> finetune
all               -> pretrain + finetune (one Python process)
```

Standalone pretraining therefore does not inspect the fine-tuning config or
train/validation manifests. Standalone fine-tuning and final evaluation do not
inspect the pretraining config or manifest. Evaluation requires both
fine-tuning train and validation manifests because both are part of the exact
completed fine-tuning RunContract it validates.

The helper uses the identical stage-building branch whether a stage is
selected alone or beside the other stage. Existing Python-canonical and
`all`-versus-standalone identity tests remain unchanged and pass.

### GREEN and verification evidence

The exact Round 2 regression slice passed:

```text
9 passed
```

Canonical identity, Python fingerprint parity, and dry-run topology:

```text
3 passed
```

Complete focused SLURM/Gloo/frozen-driver gate:

```text
131 passed
```

Fresh full CPU gate:

```text
504 passed
```

`bash -n`, `git diff --check`, and a bounded standalone pretraining
dry-run with an inaccessible fine-tuning config all exited zero. The frozen
local reproduction driver remains byte-identical:

```text
SHA256 485b4b36fe1045174a392834bd9ee9848538ab496944631a0b2d514e22d4de18
```

Atomic completion markers, per-rank runtime validation, signal latching,
quoting, lock recovery, bracketed IPv6, and no-auto-requeue behavior were not
changed. No real `sbatch`, multi-node step, CUDA training, or long-running
job was started; the same real-site gate remains unrun. Fix Round 2 is
committed as the single commit reported in the parent handoff; the report
avoids a self-referential commit hash.
