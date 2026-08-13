# Fine-Tuning Activation Checkpointing Design

## Problem

The MeerKAT reproduction reaches fine-tuning update 10,000 and saves a
complete checkpoint after validation. The next forward pass fails on every
A100 40 GB rank while allocating dense attention logits:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.91 GiB.
```

This is the first unfrozen-backbone forward, not a resume or allocator defect.
`FineTuningModel` runs the encoder under `torch.no_grad()` while
`update < freeze_finetune_updates`. The saved checkpoint has `update=10000`,
and the recipe sets `freeze_finetune_updates=10000`, so its first resumed
forward retains activations for the full encoder.

Every `train_0` recording contains 80,000 waveform samples. The current sampler
forms batches of eight recordings, and the convolutional frontend produces
2,000 time steps per recording. One FP32 attention tensor therefore occupies

```text
8 batches * 16 heads * 2000 * 2000 * 4 bytes = 1.907 GiB.
```

That value matches the failed allocation at the ALiBi addition. At failure,
PyTorch had allocated about 36.8 GiB per rank and retained little unused cache,
which rules out fragmentation as the primary cause.

The earlier production burn-in stopped at update 10,000. Its final training
forward used update 9,999 and kept the backbone frozen, so it did not exercise
the memory transition. The paper recipe used four ranks. The reproduction host
uses eight A100 40 GB GPUs, and the current eight-rank mapping needs an
activation-memory strategy for the unfrozen phase. The official animal2vec
Fairseq wrapper also exposes activation checkpointing, so this strategy is
consistent with the source implementation's supported execution modes.

## Measured Constraints

Changing only `dataset.max_tokens` does not provide a practical fix:

| `max_tokens` | Records per rank | Accumulation tested | Result at update 10,000 |
| ---: | ---: | ---: | --- |
| 960,000 | 8 | 2 | OOM, 1.91 GiB request |
| 480,000 | 6 | 4 | OOM, 1.43 GiB request |
| 400,000 | 5 | 5 | OOM in attention dropout |
| 320,000 | 4 | 6 | OOM, 978 MiB request |
| 240,000 | 3 | 8 | OOM before completing an update |

The documented 800,000 fallback still produces batches of eight because all
records have the same length and the sampler rounds to the required batch-size
multiple. It has the same activation footprint as 960,000.

Batch sizes one or two might fit, but they would require 24 or 12 accumulated
microbatches per optimizer update. More importantly, changing the batch
partition while resuming would reinterpret the saved numeric sampler cursor
against a different batch list. That would duplicate and skip training samples.
The fix must retain `max_tokens=960000` and `update_freq=[2]` so the current
checkpoint remains an exact continuation.

## Requirements

- Complete unfrozen fine-tuning on eight A100 40 GB GPUs.
- Preserve the paper driver's current sample order, sampler cursor, optimizer
  schedule, effective batch, and numerical model operations.
- Resume the existing update-10,000 checkpoint without altering its contents.
- Recompute stochastic block operations with the same PyTorch RNG draws used
  by an ordinary forward and backward pass.
- Avoid recomputation during frozen fine-tuning, validation, evaluation, and
  inference.
- Keep activation checkpointing disabled by default for existing configs and
  callers.
- Make the reproduction driver's memory mode explicit in its launch command
  and recorded run profile.
- Prove the fix with complete unfrozen optimizer updates, not a forward-only
  probe.

## Selected Design

### Configuration

`ModelConfig` will gain one field:

```python
checkpoint_activations: bool = False
```

The normal strict config parser will accept `model.checkpoint_activations`.
YAML files and CLI overrides will supply a parsed boolean value. Configuration
serialization will include the resolved value. The false default means old
YAML files, old active-config snapshots, library callers, and non-reproduction
commands keep their current behavior.

`AudioEncoder.from_config` will pass the value into the encoder. The encoder
will pass it to both `TransformerStack` instances: the prenet and the main
transformer.

Fine-tuning requires one additional propagation step. Its constructor derives
the encoder architecture from `pretrained_config.model` and selectively copies
runtime settings from the active fine-tuning model. That replacement will copy
`fine_model.checkpoint_activations` into the derived encoder model. The active
fine-tuning config therefore controls recomputation even when the pretrained
config predates the field or records false. The option adds no parameters or
buffers, so state-dict keys and checkpoint model weights remain unchanged.

### Transformer Execution

Each `TransformerStack` will store the option and decide whether to checkpoint
a block from runtime state:

```text
checkpoint_activations
and stack.training
and torch.is_grad_enabled()
```

When the condition is true, the stack will call
`torch.utils.checkpoint.checkpoint` for each non-dropped block with
`use_reentrant=False` and `preserve_rng_state=True`. The block inputs remain
`value`, `padding_mask`, and the layer-specific ALiBi bias. Both returned
tensors, the next value and teacher-target value, remain part of the stack's
normal output contract.

Non-reentrant checkpointing supports the existing block signature and modern
autograd behavior without requiring an input tensor to have
`requires_grad=True`. RNG preservation makes block-local dropout and drop-path
draws replay during backward instead of consuming a second stochastic stream.

Layerdrop selection remains outside the checkpointed function. The stack will
draw from NumPy once per layer, exactly as it does now, and only execute the
selected block through checkpointing. Backward recomputation therefore cannot
repeat or change the layerdrop decision.

The checkpoint condition naturally disables recomputation in the frozen
fine-tuning phase because its encoder forward runs under `torch.no_grad()`.
Evaluation and validation also run without gradient tracking. Pretraining and
other callers remain unchanged unless they explicitly enable the new field.

### Reproduction Driver

The fine-tuning launch in `scripts/reproduce_meerkat_paper.sh` will add:

```text
--override model.checkpoint_activations=true
```

The pretraining launch will not enable it. The driver will print the resolved
fine-tuning setting in its launch summary and write
`finetune_checkpoint_activations=true` to
`environment/run-profile.txt`. The setting will be fixed for this hardware
profile instead of adding another environment-variable branch.

The driver will retain `dataset.max_tokens=960000` and
`optimization.update_freq=[2]`. When it finds the existing fine-tuning
checkpoint, the current preflight and resume logic will run unchanged. The
sampler will resume at epoch 24, batch 720 under the same batch topology.

## Memory Behavior

Ordinary autograd retains the dense attention and feed-forward intermediates
from all eight prenet blocks and sixteen transformer blocks until backward.
Activation checkpointing retains block boundaries and recomputes block
internals as backward reaches each block. This exchanges additional compute for
a lower peak activation footprint without changing attention mathematics,
model parameters, optimizer state, or data order.

The implementation will not combine this change with an in-place ALiBi
optimization, broadcasted-bias rewrite, Flash Attention, FSDP, CPU offload, or
a smaller batch. Those approaches would make the acceptance result harder to
attribute and introduce separate numerical or operational risks.

## Checkpoint and Resume Compatibility

The new option changes execution policy only. It creates no state-dict entries
and requires no checkpoint migration. A checkpoint written with the option
enabled can load with it disabled, and the existing update-10,000 checkpoint
can load with it enabled.

The active config stored by the next save will record
`checkpoint_activations=true`. All optimizer, scheduler, AMP scaler, model,
sampler, and per-rank RNG states will continue from the existing checkpoint.
Because batch geometry stays fixed, no special sampler reset or epoch restart
is allowed.

## Failure Handling

Activation checkpointing will not catch or retry CUDA OOM exceptions. A real
capacity failure will follow the existing distributed shutdown path and leave
the source checkpoint untouched. The reproduction driver will not silently
reduce batch size or change accumulation.

The bounded verification will write only to a temporary save directory. If
the selected design still exceeds 40 GB, implementation will stop and report
the measured failure. A separate design would then consider reducing the
materialized ALiBi bias or changing the attention implementation.

## Tests

Tests will be written before implementation and cover these contracts:

- Config loading accepts true and false values, defaults to false, serializes
  the resolved value, and still rejects unknown model keys.
- Encoder construction propagates the option to the prenet and main
  transformer stacks.
- Fine-tuning construction uses the active fine-tuning value when the
  pretrained config omits the field or resolves it to false.
- A training forward and backward with checkpointing produces the same output,
  loss, input gradients, parameter gradients, and final PyTorch RNG state as
  the ordinary path under identical seeds and stochastic settings.
- Layerdrop consumes the same NumPy draws and selects the same blocks in both
  modes.
- Checkpointing runs only when the option, training mode, and gradient tracking
  are all active. Evaluation and `torch.no_grad()` bypass it.
- An old config without the field and a representative old checkpoint remain
  loadable.
- The reproduction dry run includes the fine-tuning override but not a
  pretraining override.
- The launch summary and run profile record activation checkpointing.
- Existing driver, configuration, model, CPU integration, and GPU smoke tests
  remain green.

## Production Acceptance

After focused and full automated tests pass, one bounded production probe will:

1. Verify that all eight GPUs are idle and the production checkpoint timestamps
   are unchanged.
2. Resume the untouched fine-tuning `checkpoint_last.pt` at update 10,000.
3. Use the original eight-rank `max_tokens=960000` and `update_freq=[2]`
   settings with activation checkpointing enabled.
4. Save only under a new directory in `/tmp`.
5. Stop at update 10,002 under a wall-clock cap of at most 30 minutes.
6. Capture PyTorch's per-rank peak allocated and reserved memory from each
   training summary. Periodic `nvidia-smi` sampling may supplement these values
   but will not serve as the peak measurement.

Acceptance requires two completed unfrozen optimizer updates, including the
first creation and reuse of backbone Adam state. Every rank must exit normally
without CUDA OOM or collective failure. The temporary checkpoint must report
stage `finetune`, update 10,002, complete optimizer and scheduler state, and
eight-rank RNG state. GPU memory must return to its idle level after exit.
Every rank's peak reserved memory must remain at least 1 GiB below the
PyTorch-visible device capacity. The acceptance report will record both peak
allocated and peak reserved bytes for every rank.

This probe does not continue the complete 30,000-update experiment. Once it
passes, the operator can restart the reproduction script, which will resume the
production checkpoint through the normal driver path.

## Documentation Changes

`docs/reproducing-paper.md` will explain why the 40 GB profile enables
activation checkpointing at the unfreeze boundary and will remove the
ineffective 800,000-token fallback. It will state that lowering `max_tokens`
changes batch topology during resume and is not an exact recovery mechanism.

`docs/code-guide.md` will document the new model field, its runtime gate, the
compute-for-memory tradeoff, and the fact that checkpointed blocks preserve RNG
state. The reproduction troubleshooting section will identify update 10,000 as
the first unfrozen forward rather than treating the symptom as allocator
fragmentation.

## Out of Scope

This work will not alter attention equations, ALiBi construction, mixed
precision, validation scheduling, freeze duration, learning-rate scheduling,
data sampling, or the published model architecture. It will not add general
automatic OOM recovery or claim parity between the four-rank paper recipe and
the eight-GPU 40 GB reproduction host beyond the existing documented batch
mapping.
