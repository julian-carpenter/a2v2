# Distributed Validation Control Design

## Problem

The eight-rank MeerKAT fine-tuning job reaches its first scheduled validation at update 10,000. Rank 0 evaluates all 13,283 `valid_0` recordings while ranks 1 through 7 wait for the validation decision. The current code makes those ranks wait in a CUDA broadcast on the default NCCL process group.

PyTorch configures the default NCCL collective timeout as 10 minutes. A standalone run of the same `_validate` function, checkpoint, manifest, batch limit, and worker count completed successfully in 53 minutes and 47 seconds. The waiting ranks therefore time out before rank 0 reaches the broadcast. The reported NCCL failure is a consequence of the rank-asymmetric validation schedule; validation data and model inference both complete successfully.

The failed runs also expose a recovery gap in `scripts/reproduce_meerkat_paper.sh`. Fine-tuning writes periodic epoch checkpoints but writes `checkpoint_last.pt` only when training exits normally. The script ignores periodic checkpoints when `checkpoint_last.pt` is absent, so each retry starts fine-tuning at update 0. The preserved `checkpoint_epoch_20.pt` contains complete optimizer, scheduler, AMP scaler, sampler, and eight-rank RNG state at update 8,295.

## Requirements

- Preserve single-rank validation and its current metric calculations.
- Keep gradient and model collectives on NCCL.
- Synchronize the validation metric and best-checkpoint decision through a CPU-capable process group.
- Allow at least two hours for rank-0 validation before the control collective times out.
- Preserve distributed RNG checkpoint gathering through the same control group.
- Resume fine-tuning from `checkpoint_last.pt` when it exists.
- Otherwise resume from the newest complete periodic or best fine-tuning checkpoint.
- Validate a selected recovery checkpoint before launching eight-rank training.
- Keep a fresh experiment directory on the existing from-scratch path.
- Do not change optimizer settings, validation cadence, metric definitions, or paper-reproduction batch geometry.

## Training Control Path

CUDA training will create one auxiliary Gloo process group with an explicit two-hour timeout. The existing checkpoint RNG-state gather already uses this group. The validation decision will use it as well.

Rank 0 will run `_validate` as it does today. Every rank will then construct a two-element CPU `float64` tensor containing the tracked metric and improvement flag. `torch.distributed.broadcast` will synchronize that tensor through the auxiliary Gloo group. No NCCL work will remain pending while rank 0 validates.

CPU distributed training will continue to use its default Gloo group. Single-process training will skip the broadcast. The two-hour limit applies only to auxiliary control operations and does not weaken the default NCCL watchdog for training collectives.

The code will keep the existing `_checkpoint_process_group` boundary because checkpoint state and validation decisions are both low-volume training control data. Its docstring and tests will describe the broader role.

## Fine-Tuning Recovery

The reproduction driver will choose a fine-tuning resume checkpoint before it assembles the launch command:

1. Use `finetune/checkpoint_last.pt` when present.
2. Otherwise consider `checkpoint_best.pt`, `checkpoint_<update>.pt`, and `checkpoint_epoch_<epoch>.pt`.
3. Choose the candidate with the newest modification time. Checkpoint writes are atomic, so a published filename represents a complete save; temporary files do not match these patterns.
4. Run the existing checkpoint preflight against the selected file and require stage `finetune`.
5. Append `--resume <selected-path>` only after preflight succeeds.

The pretraining checkpoint preflight will require stage `pretrain`. A stage mismatch will stop the script before torchrun starts. Existing checks for optimizer, scheduler, eight-rank RNG state, free GPU memory, and disk space remain in force.

## Failure Handling

The auxiliary Gloo group retains a finite timeout. A rank that crashes or a validation that exceeds two hours will still terminate the job instead of waiting forever. The original exception and normal process-group cleanup behavior remain unchanged.

If the newest recovery checkpoint is unreadable, missing training state, recorded for another world size, or from the wrong stage, preflight will fail with the checkpoint path and reason. The driver will not silently fall back to an older artifact because doing so could discard progress without the operator noticing.

## Tests

Regression tests will prove these behaviors before implementation:

- Distributed CUDA setup creates the auxiliary Gloo group with a two-hour timeout.
- Validation decisions use a CPU tensor and the auxiliary group.
- Checkpoint preflight rejects a stage mismatch.
- The reproduction driver resumes from a periodic epoch checkpoint when `checkpoint_last.pt` is absent and validates that checkpoint before launch.
- `checkpoint_last.pt` remains the preferred resume source when present.

After focused tests pass, verification will run the full CPU test suite and the
existing eight-GPU NCCL preflight. The first bounded production run will resume
the update-8,295 checkpoint and stop at update 10,000. The CPU Threading
Amendment below records that run's result and supersedes its acceptance
procedure. Verification will not continue the full 30,000-update recipe.

## Out of Scope

This change will not shard validation across ranks, alter metric aggregation, raise the default NCCL timeout, or optimize the 54-minute validation implementation. Those changes carry numerical or operational tradeoffs that the timeout repair does not require.

## CPU Threading Amendment

### Verification Finding

The first bounded production verification proved the control-plane repair but
hit its 7,200-second wall cap before rank 0 completed validation. The run
resumed correctly, reached the same update-9,747 metrics as the interrupted
job, and kept all ranks alive for about 88 minutes of rank-asymmetric
validation without an NCCL, Gloo, CUDA, or rank failure.

Torchrun printed that it had set `OMP_NUM_THREADS=1`. On this host, that
environment value makes `torch.get_num_threads()` return 1. Rank 0 then stayed
at 100 percent of one CPU during the validation tail. The same Python runtime
without the environment override reports 128 PyTorch intra-op threads and had
completed the standalone validation in 53 minutes and 47 seconds. The host has
128 physical cores, 256 logical CPUs, and eight NUMA nodes.

The capped run advanced the periodic recovery checkpoint atomically from
update 8,295 to a complete update-8,710 state. That checkpoint retains the
optimizer, scheduler, AMP scaler, sampler, and eight-rank RNG state and becomes
the next bounded verification's resume point.

### Considered Approaches

The approved approach sets an explicit thread budget in the reproduction
driver. `A2V2_OMP_NUM_THREADS` defaults to 8, and both distributed training
commands receive `OMP_NUM_THREADS=8`. Eight ranks can therefore use at most 64
intra-op threads. The recipes configure 20 DataLoader workers per rank, so the
combined upper bound of 224 threads fits within the host's 256 logical CPUs.

Leaving the variable unset would preserve torchrun's one-thread fallback and
repeat the measured bottleneck. Setting 16 or more threads per rank would put
128 or more intra-op threads beside 160 DataLoader workers and oversubscribe
this host. Changing `torch.set_num_threads` only around `_validate` would hide a
deployment constraint inside the training workflow and add asymmetric runtime
state that the paper driver can express directly.

### Driver Contract

The driver will:

- resolve `TRAIN_OMP_NUM_THREADS` from `A2V2_OMP_NUM_THREADS`, defaulting to 8;
- require a positive integer before preflight or torchrun;
- pass `OMP_NUM_THREADS=<resolved value>` to pretraining and fine-tuning;
- print the resolved value in the launch summary;
- record `omp_num_threads=<resolved value>` in `environment/run-profile.txt`;
- list the override in `docs/reproducing-paper.md`.

This setting changes host CPU parallelism only. It does not change model
parameters, batches, optimizer state, validation metrics, checkpoint format,
or distributed topology. Operators can lower the setting for a constrained
host without editing the script.

### Amendment Tests and Acceptance

Driver tests will verify the default value, a valid environment override, and
rejection of zero or non-numeric values. The dry-run command graph must show
the resolved value in both training launches. The fake real-mode test must
record the value in the run profile.

After the complete automated suite and eight-rank transport probe pass, the
bounded production verification will resume the complete update-8,710 epoch
checkpoint and stop at update 10,000. The same 7,200-second cap remains in
force. Acceptance requires rank-zero validation to finish, every rank to
synchronize without a collective error, and `checkpoint_best.pt` and
`checkpoint_last.pt` to load as complete eight-rank fine-tuning checkpoints at
update 10,000.
