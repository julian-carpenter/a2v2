# Distributed Checkpoint Reliability

## Goal

Make the eight-A100 MeerKAT driver save and resume checkpoints without routing
Python RNG objects through NCCL. Fail fast before training when the interpreter,
GPU allocation, collective topology, or output filesystem cannot support the
run. Release distributed resources after success and exceptions.

## Failure evidence

The reported traceback came from commit `5e5359c`. Its line numbers identify
this call path:

```text
run_training
  -> epoch 10 checkpoint branch
  -> write_checkpoint
  -> torch.distributed.gather_object
  -> NCCL Error 1: unhandled cuda error
```

The experiment directory dates the start to 2026-07-28 and the exception to
2026-07-29, about 22 hours later. The paper recipe saves every ten epochs, so
the timing agrees with the first scheduled epoch checkpoint.

Three shorter eight-rank checks passed on the same host:

- the CUDA/NCCL collective and RNG checkpoint probe;
- the exact distributed resume probe; and
- one full production-model update followed by `checkpoint_last.pt`.

The production update reserved as much as 37.51 GB on one 40 GB A100. These
results isolate the failure to the long-lived epoch checkpoint boundary. They
do not implicate model mathematics, ordinary DDP reductions, or basic
eight-GPU connectivity.

PyTorch implements NCCL object collectives by serializing objects to byte
tensors and moving the internal tensors to the current CUDA device. Its
documentation warns that object collectives can have unexpected memory
behavior and out-of-memory failures. The checkpoint payload already serializes
each rank's RNG dictionary to CPU bytes, so GPU transport gives no benefit.

The warning about `destroy_process_group()` has a separate cause.
`run_training` destroys a group on its normal return path, while exceptions
leave the group initialized.

## Architecture

### Checkpoint control plane

Distributed CUDA training will use two process groups:

- NCCL carries model tensors, gradients, diagnostics, broadcasts, and barriers.
- Gloo carries serialized rank RNG bytes during checkpoint construction.

`run_training` will create the Gloo group after NCCL initialization. Every
rank will enter the same `gather_object` call with that group. Rank zero will
require exactly `world_size` byte payloads before deserializing them and saving
the checkpoint.

CPU distributed training already uses Gloo, so it will reuse the default
group. Single-process training will keep its current local RNG dictionary.
The checkpoint schema and `by_rank` ordering will remain unchanged.

### Distributed cleanup

The public `run_training` function will record whether its caller already owns
a process group. A private implementation will retain the training body. The
public wrapper will destroy only a group created during its call, including
when model construction, data loading, training, validation, or checkpointing
raises.

This ownership check preserves Python callers that initialize their own group.
The CLI will no longer print the leaked-process-group warning after a training
exception.

### Driver preflight

`scripts/reproduce_meerkat_paper.sh` will run these checks before the first
training command:

1. Use the selected Python interpreter for the default distributed launcher
   through `python -m torch.distributed.run`.
2. Import A2V2 and each runtime dependency in that interpreter.
3. Require eight distinct A100 devices with at least 39 GiB total memory and
   at least 38 GiB free memory per device.
4. Require at least 64 GiB free on the output filesystem. One measured
   pretraining checkpoint occupies 4.7 GiB, and atomic saves need room for a
   temporary file.
5. Run the repository's eight-rank collective, checkpoint, and RNG-restore
   probe. The probe will use NCCL for tensors and Gloo for RNG bytes, matching
   production.

Dry-run mode will print the resolved preflight and training commands without
requiring CUDA.

### Burn-in and resume

On a fresh pretraining directory, the driver will:

1. launch the production model and batch profile with `--stop-at-update 1`;
2. require and load `checkpoint_last.pt`; and
3. launch the full pretraining command with `--resume checkpoint_last.pt`.

The burn-in takes about 70 seconds on this host. It exercises model allocation,
forward and backward passes, AMP overflow handling, optimizer state,
rank-specific RNG gathering, atomic checkpoint writing, and resume loading.

An existing `checkpoint_last.pt` skips the burn-in and resumes through the
current path. The driver will not change topology or batch controls during
resume.

Training logs will append across burn-in, resume, and operator reruns. Each
launch will add a timestamped command header. A failed rerun will no longer
overwrite the evidence from the preceding run.

## Error handling

The driver will stop before training and name the failed requirement when:

- the selected interpreter cannot import A2V2 or a runtime dependency;
- fewer or more than eight requested GPUs are visible;
- a selected GPU has the wrong model or insufficient free memory;
- the output filesystem lacks the minimum free space;
- the eight-rank probe fails;
- the burn-in does not write a valid native checkpoint; or
- an existing checkpoint cannot load.

The training process will propagate checkpoint gather errors after releasing
its process groups. Rank zero will reject missing or malformed RNG payloads
before writing a checkpoint.

## Tests

Test-first implementation will add these contracts:

1. A focused distributed helper test requires CUDA training to choose a Gloo
   group for RNG-object gathering.
2. Checkpoint payload assembly rejects a missing or non-byte rank payload.
3. A forced training exception destroys a process group that `run_training`
   created and preserves one owned by its caller.
4. Driver tests require dependency, free-memory, disk-space, eight-rank probe,
   burn-in, checkpoint validation, full resume, and append-only logging
   commands.
5. The eight-rank NCCL probe and exact-resume probe pass after the change.
6. A full production-model burn-in writes a checkpoint, resumes to update two,
   and exits cleanly.

The final verification will include focused tests, the complete CPU suite,
shell syntax, eight-rank probes, and the bounded production burn-in. A full
384,230-update reproduction requires separate user approval because it exceeds
two hours.

## Scope

This change does not alter model equations, optimizer arithmetic, scheduler
updates, batch topology, checkpoint fields, or paper recipes. It does not add
checkpoint retries or reduce the training token budget.
