# Configuring an experiment

Start with the commented pair:

- [Annotated pretraining config](../configs/examples/annotated_pretrain.yaml)
- [Annotated fine-tuning config](../configs/examples/annotated_finetune.yaml)

These examples explain each included parameter at its point of use. `TUNE`
marks likely changes for a new corpus. `KEEP` marks encoder geometry or
architecture that you must retain when loading a checkpoint. Both examples
use a MeerKAT-shaped encoder and a nominal 1000-second batch on four GPUs.
They illustrate configuration; they do not reproduce a published experiment.
Use the frozen [paper recipes](../configs/README.md) and
[reproduction instructions](reproducing-paper.md) for that purpose.

Change `task.data`, label vocabulary/fold, and checkpoint paths first. Keep
`task.min_label_size=0` for unlabeled pretraining: a positive label-byte filter
excludes rows without label files, including in archived corpus recipes. Keep
pretraining and fine-tuning sample rates, frontend geometry, and architecture
compatible. The loader rejects unknown keys; comments do not set values.
Use `--override section.key=value` for a recorded experiment change, for example
`--override 'optimization.update_freq=[5]'`. Do not edit the control recipe.

## Sinc filterbank and convolution tuples

`task.conv_feature_layers` lists `(output_channels, kernel_samples, stride)`.
With `model.modalities.audio.sinc_input: true`, the **first tuple** describes
the learnable Sinc bandpass filterbank. `(127, 63, 1)` means 127 filters with
63-sample kernels and stride 1. At 8000 Hz, a kernel spans 7.875 ms. Sinc
kernels require an odd width. With `sinc_input: false`, the first tuple
describes an ordinary convolution instead.

Subsequent tuples describe ordinary convolutions. Their stride product `q`
defines the output hop `q / sample_rate`; it does not describe the full
receptive field. The example's `q=40` at 8000 Hz gives 200 feature frames/s.
Use `feature_timestamps` to account for kernel widths and padding when mapping
frames to time. Changing channels changes parameter shapes; changing stride
or sample rate changes label timing as well as memory use.

## Batch size in audio seconds

Here, a **token means one resampled waveform sample**, not one Transformer
feature frame. Let:

- `M = dataset.max_tokens`: per-rank, per-microbatch waveform-sample budget;
- `f = task.sample_rate`: waveform samples per second;
- `W = distributed_training.distributed_world_size`: total training processes;
- `A = optimization.update_freq[0]`: microbatches accumulated per rank/update.

The nominal global audio budget per optimizer update is:

```text
B_seconds = (M / f) × W × A
A_for_1000_seconds = ceil(1000 × f / (M × W))
```

For `M=400000`, `f=8000`, `W=4`, and `A=5`, the budget is 1000 seconds,
or 100 ten-second recordings. The integer choice of `A` means you cannot hit
every target exactly. Pick the nearest acceptable budget and record it.
If you need at least 1000 nominal seconds, use the ceiling formula. A list
such as `[2, 3]` uses 2 in the first epoch, then 3 in later epochs; the trainer
reuses the last list entry.

Treat this as a budget, not an exact duration measurement. The sampler packs
whole recordings with a longest-item-based cost. Cropping, padding, recording
lengths, `required_batch_size_multiple`, partial accumulation at epoch ends,
and distributed padding can change the actual audio seen. Ten-second clips
at 8000 Hz contain 80000 samples: `M=408000` still fits five such clips, so
the waveform duration is 50 seconds per microbatch, not 51. Multiplying
`model.clone_batch` into `B_seconds` would count repeated masked views as
new audio; it increases student work/memory, not unique recording exposure.

Set `W` to the number of launched workers, not GPU count per node. On two
nodes with eight workers each, `W=16`. A YAML world size does not launch
workers: use `torchrun` or the [SLURM launcher](slurm.md).

### Finding a safe per-GPU budget

1. Start with a small `M` and the intended precision, architecture, and device.
2. Run several successful updates, including checkpoint/resume. Increase `M`
   until memory headroom becomes small or a disposable probe runs out of memory.
3. Back off to leave headroom for variable lengths, optimizer state, compilation,
   validation, and other processes. A one-update success does not prove safety.
4. For fine-tuning, test **after encoder unfreezing**; frozen-encoder memory
   measurements underestimate the later training peak. Include validation too.
5. Adjust integer `A` to obtain the target global duration without increasing
   the per-forward-pass memory burden. Recompute budgets if `W` changes.

Do not find the limit by repeatedly crashing your long experiment. Use a
separate smoke output directory and `--stop-at-update` to preserve the intended
learning-rate/EMA schedule; `--max-updates` changes the schedule itself.
Gradient accumulation trades additional microbatch passes for a larger global
update; it does not make a too-large individual microbatch fit. Eight-bit
optimizers reduce optimizer state, not all activations. Compare complete
steady-state profiles before changing a paper batch.

## Converting desired epochs to max_update

`optimization.max_update` counts **successful optimizer updates**, not
microbatches or epochs. Learning-rate, EMA, and weight-decay schedules use this
clock. For retained training duration `D` seconds and nominal global batch
duration `B`:

```text
updates_per_epoch ≈ D / B
max_update ≈ desired_epochs × D / B
```

For one million retained seconds, a 1000-second batch, and 50 epochs, the
estimate is 50000 updates. Sum the selected manifest's resampled/cropped
training lengths, not the duration of excluded files or the validation split.
For fixed ten-second clips with `N` retained rows, use `D=10×N`.

For an exact count, instantiate the same dataset and samplers as training,
including filtering, crop limits, recording-count rounding, rank padding, and
shuffle seed. The global sampler yields `G` microbatches; each rank receives
`R=ceil(G/W)` batches after distributed padding. An epoch then needs
`ceil(R/A)` optimizer attempts, including its final partial accumulation.
Variable-length packing can change `G` between epochs, and skipped AMP
overflows consume batches without advancing successful updates. Monitor
checkpoint `epoch`, `sampler_state`, and logs instead of assuming the duration
estimate proves an exact epoch count.

If you want an early stop but retain the paper LR/EMA trajectory, keep
`max_update` unchanged and use `--stop-at-update N`. If you design a new
full schedule, set `max_update`, `warmup_updates`, and
`ema_anneal_end_step` together. Archive that distinction in your experiment.

## Parameters to tune first

| Choice | How to choose it |
| --- | --- |
| `task.data`, `train_subset`, `valid_subset`, `unique_labels` | Verify manifests, annotation names/order, and disjoint splits. Run the corpus audit before training. |
| `sample_rate`, `max_sample_size` | Retain the frequencies and event durations needed for your corpus. A training crop limit uses resampled samples; 80000/8000 is 10 seconds. Fine-tuning must match the pretrained frontend rate. |
| `max_tokens`, world size, `update_freq` | Measure memory after warm-up and unfreezing; use the duration formula above for the global update. |
| `max_update`, warm-up, EMA endpoint | Derive the budget from dataset/sampler size and desired epochs; count successful updates. |
| `lr`, `weight_decay`, dropout, mixup | Compare held-out metrics across recorded trials. Inspect the unfreezing boundary; do not select from training loss alone. |
| Time/channel masks | Convert lengths through the feature hop. Pretraining `mask_prob` is an archived span-count parameter and may exceed 1; inspect realized masking and variance. |
| `clone_batch` | Trade masked student views against compute/memory. Keep the unique-audio budget separate. |
| `freeze_finetune_updates`, `feature_grad_mult` | Choose how long to train only the classifier and whether local convolutions update; test memory after the freeze ends. |
| `metric_threshold`, `sigma_s`, `method` | Tune on validation against precision/recall and event duration; keep the test split untouched. |
| Architecture (`KEEP` fields) | Start a new checkpoint family for RoPE/ALiBi, CLS, FFN, width/depth, or initialization changes. Keep a matched baseline. |
| Compile/checkpointing/optimizer execution | Benchmark the full training lifecycle. Checkpointing recomputes activations; strict Flash needs supported CUDA/dtype/masking; 8-bit optimizers need the optional dependency. |

See the [modern-option reference](../configs/README.md#opt-in-modern-examples)
for supported strategy values. Fine-tuning restores encoder architecture from
the stored pretraining configuration; matching an active YAML filename alone
does not prove compatible weights. A checkpoint is pickle-backed: load only
files from a source you trust.

## Running the examples

Replace `/path/to/manifests`, verify labels, and choose budgets for your GPUs.
The commands below launch four workers to match the examples:

```bash
torchrun --standalone --nproc-per-node=4 /usr/local/bin/a2v2-train \
  --config configs/examples/annotated_pretrain.yaml --device cuda

torchrun --standalone --nproc-per-node=4 /usr/local/bin/a2v2-train \
  --config configs/examples/annotated_finetune.yaml \
  --pretrained-checkpoint checkpoints/tutorial_pretrain/checkpoint_last.pt \
  --device cuda
```

Use your installed `a2v2-train` path (`command -v a2v2-train`) if it differs.
These commands train full experiments; use a separate output directory and
`--stop-at-update` for smoke tests.
