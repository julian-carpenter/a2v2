# A2V2 SLURM guide

This guide covers the separate SLURM launch path in
`scripts/reproduce_meerkat_slurm.sh` and
`scripts/a2v2_slurm_node.sh`. The local paper control remains
`scripts/reproduce_meerkat_paper.sh`.

The implementation passed deterministic dry-run, mocked scheduler, Bash
syntax, two-rank Gloo checkpoint, lock race, resume, frozen-driver, and CPU
regression tests. No real SLURM allocation ran during development. Treat the
site acceptance sections as release gates.

## Site requirements

Each compute node needs the same repository revision and Python environment.
The batch host also needs enough CPU memory to load a native checkpoint for
stage, world-size, and freshness checks.

The launcher expects:

- a homogeneous allocation with the same GPU count on each node;
- integer `SLURM_GPUS_ON_NODE` and a compatible
  `SLURM_GPUS_PER_NODE` value;
- `srun`, `scontrol`, Bash, Python, PyTorch, and the installed A2V2 package;
- one shared manifest directory visible at the same path on all nodes; and
- one shared output path with POSIX hard links, advisory locks, atomic
  replacement, inode identity, directory `fsync`, and prompt cross-node
  visibility.

Add account, partition, wall-time, QoS, module, container, and environment
directives through the site submit wrapper. The checked-in script requests:

```text
#SBATCH --signal=B:USR1@300
```

This requests `SIGUSR1` with 300 seconds of lead time.

## Command surface

Show orchestration options:

```bash
bash scripts/reproduce_meerkat_slurm.sh /path/to/manifests /path/to/output --help
```

Show node-wrapper options:

```bash
bash scripts/a2v2_slurm_node.sh pretrain --help
```

The orchestrator accepts `pretrain`, `finetune`, `evaluate`, and `all`.
It selects inputs by phase. Standalone pretraining does not load fine-tuning
inputs. Fine-tuning does not load the pretraining YAML. The launcher `evaluate`
phase runs the legacy framewise/event evaluator and reports frame and event
metrics. `a2v2-evaluate-sequence` runs the CLS sequence evaluator and reports
sequence-level metrics. Both evaluators restore model construction from the
stored pretraining config in their checkpoint.

### Pretrain

```bash
sbatch scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase pretrain --nodes 2 --gpus-per-node 4
```

### Fine-tune

```bash
sbatch scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase finetune --nodes 2 --gpus-per-node 4
```

Fine-tuning reads
`/shared/runs/meerkat/pretrain/checkpoint_last.pt` unless
`--pretrain-checkpoint` selects another `checkpoint_last.pt`. Its own
automatic resume path is
`/shared/runs/meerkat/finetune/checkpoint_last.pt`.

### All stages

```bash
sbatch scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase all --nodes 2 --gpus-per-node 4
```

The `all` path runs pretraining, fine-tuning, and final evaluation in order.
Final evaluation uses one node, one task, and one GPU.

The final evaluator expects framewise event logits. For a modern
`classification_head=cls` run, use `pretrain` and `finetune` as separate
phases, then call the sequence-classification evaluator. The all-stage dry-run
can render a CLS command graph. Its final event-evaluation step will reject
sequence logits.

### Modern CLS sequence evaluation

Run modern CLS pretraining and fine-tuning as separate phases. Evaluate the
resulting native fine-tuning checkpoint on one device:

```bash
a2v2-evaluate-sequence \
  /shared/runs/modern-cls/finetune/checkpoint_last.pt \
  --trust-checkpoint \
  --config configs/modern/rope_cls_geglu_finetune.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override dataset.valid_subset=valid_0 \
  --device cuda
```

Use one SLURM task and one GPU for the same command inside an allocation:

```bash
srun --nodes=1 --ntasks=1 --gpus=1 \
  a2v2-evaluate-sequence \
  /shared/runs/modern-cls/finetune/checkpoint_last.pt \
  --trust-checkpoint \
  --config configs/modern/rope_cls_geglu_finetune.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override dataset.valid_subset=valid_0 \
  --device cuda
```

PyTorch writes native `.pt` checkpoints in a pickle-backed format.
`--trust-checkpoint` permits deserialization. Pass it for a checkpoint from
a source you trust. The flag does not make pickle safe.

The command requires a strict fine-tuning config with
`model.classification_head=cls`. It restores encoder construction from the
checkpoint's stored pretraining config. It checks label order, sample rate,
normalization, convolution geometry, layer averaging, CLS selection, and focal
loss parameters, then uses a strict model-state load. Data path, validation
subset, worker count, batch budget, metric threshold, and selected device
remain evaluation choices. The evaluator neither restores optimizer state nor
requires the original training world size.

### Dry run

```bash
bash scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase all --nodes 2 --gpus-per-node 4 \
  --job-id dryrun-4815 --dry-run
```

Dry-run needs no SLURM variables, CUDA devices, manifests, checkpoints, or
output directories. It prints shell-quoted commands and marks manifest
identity as `dry-run-unverified`. It does not validate site storage or GPU
transport.

## Topology

The outer script starts one torchrun agent per node:

```text
srun
  --nodes=N
  --ntasks=N
  --ntasks-per-node=1
  --gpus-per-node=G
  scripts/a2v2_slurm_node.sh PHASE ...
```

Each node wrapper starts one worker per local GPU:

```text
python -m torch.distributed.run
  --nnodes=N
  --node-rank=SLURM_NODEID
  --nproc-per-node=G
  --rdzv-backend=c10d
  --rdzv-endpoint=HOST:PORT
  --rdzv-id=JOB_ID-PHASE
```

The global world size equals `N * G`. The outer script passes that value
through `distributed_training.distributed_world_size`. The node wrapper
checks the allocation, local rank, visible GPU count, and configured world
size before model construction. Both scripts preserve scheduler-provided
`CUDA_VISIBLE_DEVICES`.

Final evaluation uses a separate `srun` step with one node, one task, and one
GPU. It does not enter the distributed training topology.

## Rendezvous

The first hostname from
`scontrol show hostnames "$SLURM_JOB_NODELIST"` supplies the default
rendezvous host. The default port is:

```text
15000 + cksum(SLURM_JOB_ID) % 20000
```

The result stays in 15000 through 34999 and remains stable across a requeue.
Each phase appends its name to the rendezvous ID, so sequential stages can
reuse one endpoint without sharing a rendezvous identity.

Use `--rdzv-endpoint HOST:PORT` to supply an endpoint. Bracket IPv6 literals,
for example `[2001:db8::1]:23451`. Check firewalls and name resolution from
each allocated node before a real launch.

## Run contracts and runtime validation

Every node receives one canonical contract with:

- scheduler job ID and phase;
- node count, processes per node, and global world size;
- rendezvous endpoint and phase-specific ID;
- absolute output directory;
- manifest fingerprint; and
- run identity.

The outer script exports canonical JSON plus SHA-256 fingerprints for the run
contract, resolved config, and manifest set. Each worker recomputes those
values before model construction or output-lock acquisition. A changed
manifest, config override, phase, world size, output directory, rendezvous
value, rank, or scheduler job fails closed.

For the default run identity:

```text
A2V2_RUN_ID=JOB-meerkat-fFOLD-FRACTION/pretrain
A2V2_RUN_ID=JOB-meerkat-fFOLD-FRACTION/finetune
```

`--run-id` changes the shared prefix. Keep one run ID across resume and
requeue attempts.

## Checkpoints, resume, and phase handoff

New training saves use native checkpoint format v2. The reader accepts format
v1 and supplies legacy defaults. Format v2 carries the gradient-clipper state,
weight-decay clock, distributed topology, resume fingerprint, rank-local RNG,
and the existing model, optimizer, scheduler, scaler, sampler, and EMA state.

Resume requires the same mathematical config, world size, batching and sampler
fingerprint, seed, data identity, and checkpoint schema. Hardware and transport
changes produce numerical-equivalence warnings. Hostname order does not assign
torchrun ranks.

The shipped modern recipes use strict v2 provenance and stateless crop
coordinates. That combination validates the selected manifest, retained
record identities, sampler population, loss and AMP settings, and tracked
metric before model construction. Distributed ranks exchange their preflight
status and fingerprints over the control group before any rank constructs a
model. Frozen paper recipes retain compatible legacy cropping. Resume warns
when any batch in the sampler epoch can crop, regardless of worker count, and
does not claim bit-exact crops across process restart. Older checkpoints
without v2 data provenance remain usable in compatible mode with a warning.

The launcher resolves the pretraining manifest from the pretraining config's
`dataset.train_subset`. A neighboring `pretrain.tsv` does not enter the
pretraining RunContract when another subset is configured.

Pretraining follows this order:

1. Use `--pretrain-resume` when supplied.
2. Resume the pretraining `checkpoint_last.pt` when it exists.
3. Start a new run when neither path exists.

Fine-tuning receives the intended pretraining
`checkpoint_last.pt` through `--pretrained-checkpoint`. Its automatic resume
path uses the fine-tuning `checkpoint_last.pt`. The pretraining checkpoint
supplies encoder initialization. The two stages must own distinct output
directories.

Evaluation consumes the checkpoint bound to the validated fine-tuning
completion record. A stale `checkpoint_best.pt` cannot replace that handoff.

## Completion markers

A successful stage writes
`checkpoint_last.pt.stage-complete` after native checkpoint validation.
Checkpoint presence does not mean stage completion. Preemption writes no
completion marker.

The marker schema is `a2v2.stage-completion.v1`. It records:

- the complete canonical run contract;
- the run-contract fingerprint;
- the resolved-config fingerprint; and
- the checkpoint's absolute path, byte size, and SHA-256.

The writer uses a same-directory candidate, file and directory `fsync`, and
atomic replacement. An `all` rerun validates a marker and checkpoint before
skipping a completed stage. Partial JSON, changed checkpoint bytes, changed
contract or config, wrong schema, and an unexpected checkpoint path fail
closed. Remove or repair a bad marker after investigating its provenance.

## Signals, checkpoint freshness, and requeue

The batch shell runs synchronous `srun` in the background so its trap can
forward `SIGUSR1` to step tasks. Worker signal handlers set a process-local
flag and perform no allocation, logging, file I/O, checkpoint write, or
collective.

At the next completed optimizer update, all ranks reduce the request. Each rank
prepares checkpoint and rank-local RNG/topology data. Rank zero writes one
atomic `checkpoint_last.pt`; all ranks receive the same write result, enter a
barrier, flush logs, and leave through the preemption contract.

Torchrun or `srun` can return raw status 75 or normalize a worker exit to a
different nonzero status after the forwarded signal. The outer launcher accepts
either case after it proves all of these conditions:

- the checkpoint inode or SHA-256 changed after the step began;
- the file loads as a native checkpoint;
- checkpoint stage and world size match the active contract; and
- all ranks reached the coordinated success path.

The script then returns exit 75. Missing, unchanged, unloadable, wrong-stage,
or wrong-world checkpoints produce exit 2. A latched preemption request stays
set across hashing, validation, handoff, marker publication, and stage
boundaries.

Configure requeue in your site policy, submit wrapper, or workflow manager.
Site administrators decide whether exit 75 triggers a requeue. Test that policy
with a bounded job before enabling production retries. Avoid a rule that
requeues exit 2, because exit 2 marks failed checkpoint validation.

`SIGTERM` remains best effort. Use the scheduler's `SIGUSR1` lead-time
contract for coordinated recovery.

## Output locking and recovery

The distributed checkpoint runtime gives each training output directory one
`.a2v2-output.lock`. Rank zero creates the directory and acquires the lock
after distributed initialization; peers synchronize and verify shared
visibility. The lock binds job ID, `A2V2_RUN_ID`, hostname, PID, and an owner
fingerprint.

Contention does not steal the lock. A second job targeting the same phase
directory stops before it can write `checkpoint_last.pt.tmp`.

Recover a stale lock with:

```bash
bash scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase pretrain --nodes 2 --gpus-per-node 4 \
  --recover-lock pretrain:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

Use the exact 64-character lowercase SHA-256 owner fingerprint from the error.
Recovery requires the recorded same-host PID to be dead and the owner record to
match. Live owners, remote hosts whose liveness cannot be proved, and changed
records fail closed. Check the scheduler and process table before recovery.

The filesystem protocol serializes acquire, recovery, and release with an
advisory lock on the output-directory inode. Release checks the owner record
and lock inode so an old process cannot unlink a replacement owner's lock.

## Output layout and logs

The default output root contains:

```text
OUTPUT_DIR/
├── pretrain/
│   ├── checkpoint_last.pt
│   ├── checkpoint_last.pt.stage-complete
│   └── tb/
├── finetune/
│   ├── checkpoint_last.pt
│   ├── checkpoint_last.pt.stage-complete
│   └── tb/
└── final-evaluation/
    ├── final-evaluation-report.json
    └── tensorboard/
```

Rank zero writes TensorBoard and aggregate training records. Each worker emits
rank-aware terminal state. Keep SLURM stdout and stderr beside this shared
tree, and record the job ID, run ID, commit, resolved configs, environment,
allocation, GPU inventory, and rendezvous endpoint.

## Troubleshooting

### World-size or GPU-cardinality mismatch

Compare `--nodes * --gpus-per-node` with the resolved
`distributed_training.distributed_world_size`. Inspect
`SLURM_GPUS_ON_NODE` and count entries in the inherited
`CUDA_VISIBLE_DEVICES`. Request a homogeneous allocation.

### Rendezvous timeout

Resolve the host from each node, test the selected port, inspect firewall
rules, and check that each phase uses the same endpoint with a distinct
rendezvous ID. Use an explicit endpoint when the first allocation hostname
cannot accept peer connections.

### Manifest or config fingerprint mismatch

Stop the job. Compare absolute manifest paths, file hashes, selected configs,
and ordered overrides on every node. Do not update shared manifests while a
run or requeue attempt uses them.

### Output lock exists

Identify the owner job and PID from the error. Wait for a live job, choose a
new output directory, or use explicit recovery after proving the same-host
owner dead. Do not delete the lock from one node during an active run.

### Completion marker rejects a stage

Compare the marker contract, config fingerprint, checkpoint path, byte count,
and SHA-256 with the current request. A marker can expose a reused output
directory or changed checkpoint. Investigate before removal.

### Exit 75 did not requeue

The launcher has completed its checkpoint contract. Configure the scheduler or
submit wrapper to treat 75 as requeue-worthy. The script does not invoke
`scontrol requeue`.

### Exit 2 after a signal

The checkpoint freshness or validation gate failed. Inspect the stage,
world-size, load error, and file signature. Keep exit 2 outside automatic
requeue policy.

### Shared path appears on rank zero alone

Stop the run and fix mount or visibility behavior. All ranks must see the
directory and manifests before training. The lock and atomic checkpoint
protocol depends on shared POSIX semantics.

## One-node parity

Run this gate on the target site before a multi-node job:

1. Record SLURM, PyTorch, CUDA, NCCL, driver, GPU, filesystem, and repository
   versions.
2. Use one node and one GPU with bounded configs that finish within the site
   test window.
3. Run the same configs and data through local torchrun and the SLURM launcher.
4. Compare resolved configs, updates, model and optimizer state, sampler/RNG
   state, metrics, and output layout.
5. Resume both paths for one more update and compare the next state.
6. Send `SIGUSR1`, verify a fresh valid checkpoint and exit 75, then exercise
   the site's requeue rule.

The local and SLURM launchers have different orchestration code. One-node
parity supplies site evidence that scheduler GPU visibility, rendezvous,
storage, signals, and exit handling preserve the runtime contract.

## Two-node acceptance matrix

Run at least two nodes with two GPUs per node. Record each result and artifact
path.

| Gate | Procedure | Required evidence |
| --- | --- | --- |
| Allocation | Launch one agent per node and one worker per GPU. | Four unique global ranks, correct local ranks, unchanged GPU visibility, matching topology schema. |
| NCCL data path | Complete bounded pretraining and fine-tuning updates. | Common loss/sample-size reductions, no rank hang, one rank-zero checkpoint writer. |
| Gloo control path | Trigger checkpoint and validation control traffic. | Common success result and bounded completion on all ranks. |
| Shared filesystem | Create manifests, lock, checkpoint candidate, atomic replacement, and completion marker. | All nodes observe exact bytes, inode changes, hard-link exclusivity, advisory lock behavior, and directory persistence. |
| Exact resume | Use strict provenance and stateless crops, save v2 state, stop, relaunch with the same world size, and run the next update. | Matching model, optimizer, AdaGC, decay, sampler, rank-local RNG, topology, manifest, crop, and loss state. |
| Contention | Point a second live job at one phase directory. | Second job fails closed and cannot alter checkpoint or lock bytes. |
| Proven-dead recovery | Stop a same-host owner and use its fingerprint. | Recovery succeeds once; stale owner cannot release the new lock. |
| Preemption | Send `SIGUSR1` during prelaunch and during an active step. | Latched request, fresh validated checkpoint after an active step, no marker on interrupted work, exit 75. |
| Site requeue | Apply the site's exit-75 policy. | New job resumes the bound checkpoint without a loop; exit 2 stays terminal. |
| Phase handoff | Complete pretraining, fine-tuning, and evaluation. | Valid stage markers, intended pretraining handoff, distinct output owners, evaluation bound to the fine-tuning completion record. |
| Rendezvous | Requeue with the same job/run identity and phase. | Stable endpoint, phase-specific c10d ID, no collision across sequential stages. |

## Unrun real-site gate

The development node did not run one-node SLURM parity or the Two-node
acceptance matrix. It also did not test a site's scheduler requeue policy,
shared filesystem, firewall, modules, container stack, or signal
normalization. Multi-node support remains implemented; local and mock tests
passed, with site validation pending.

Do not claim target-cluster validation until the one-node gate and every
applicable matrix row has dated evidence from that cluster.
