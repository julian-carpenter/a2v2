# Reproducing Animal2Vec 1.0

This guide separates compatibility checks that can run on a CPU from the full
training runs needed to reproduce the paper's reported MeerKAT and NIPS4Bplus
results. The checked-in MeerKAT and hyena recipes are the **Animal2Vec 1.0
reproduction baselines** within A2V2 and preserve the official
hyperparameters. They are controls for the published method, not recipes for a
new A2V2 method. A full reproduction still requires the datasets, published
splits, and multi-GPU compute used by the original experiments. See the
[configuration-family guide](../configs/README.md) before adding or changing
recipes.

Comments in each file under `configs/` explain the stage, sample rate, batch
topology, masking, freeze boundaries, and required paths beside their exact
values. The [code guide](code-guide.md) defines repository terminology and
maps the principal equations to their source implementations.

## One-fold approximate reproduction on eight A100 GPUs

For the practical eight-GPU run requested for this repository, use the
deployment driver from the repository root:

```bash
bash scripts/reproduce_meerkat_paper.sh \
  /datasets/MeerKAT/manifests \
  /experiments/a2v2-meerkat-fold0
```

The default first verifies the selected Python environment, output filesystem,
and all eight GPUs. It then runs an eight-rank collective/checkpoint probe. A
fresh pretraining directory performs one production update, validates the
resulting native checkpoint, resumes the complete pretraining run, performs
one 100%-label fine-tuning run for fold 0, and runs one final validation. Both
training stages use all devices in
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` with eight `torchrun` ranks. The
validation step uses one GPU because it does not perform distributed gradient
updates. Its report is written to:

```text
/experiments/a2v2-meerkat-fold0/final-evaluation/final-evaluation-report.json
```

The driver is a **same-order approximate reproduction**, not an exact paper
run. The paper recipes used four ranks. On the verified A100 host, published
pretraining already reserved about 37.1 GiB per rank, so adding GPUs cannot
safely increase its per-rank allocation. The driver retains 408,000 tokens per
rank and changes accumulation from five to three:

```text
published: 408,000 × 4 ranks × 5 accumulation = 8,160,000 tokens/update
driver:    408,000 × 8 ranks × 3 accumulation = 9,792,000 tokens/update
```

Fine-tuning had substantially more measured headroom. The default raises its
per-rank token budget to 960,000 and reduces accumulation to two:

```text
published: 426,667 × 4 ranks × 9 accumulation = 15,360,012 tokens/update
driver:    960,000 × 8 ranks × 2 accumulation = 15,360,000 tokens/update
```

The fine-tuning global token batch is therefore essentially unchanged, while
each rank processes a larger microbatch. These quantities are padded-sample
budgets; actual memory also depends on recording lengths, packing, PyTorch,
and the update-10,000 transformer unfreeze. If the deployed data produces a
genuine out-of-memory error, lower only the relevant token budget and retain
the emitted log as part of the experiment record:

```bash
A2V2_FINETUNE_MAX_TOKENS=800000 \
bash scripts/reproduce_meerkat_paper.sh \
  /datasets/MeerKAT/manifests \
  /experiments/a2v2-meerkat-fold0
```

The complete batch controls are:

| Environment variable | Default |
| --- | ---: |
| `A2V2_PRETRAIN_MAX_TOKENS` | 408000 |
| `A2V2_PRETRAIN_UPDATE_FREQ` | 3 |
| `A2V2_FINETUNE_MAX_TOKENS` | 960000 |
| `A2V2_FINETUNE_UPDATE_FREQ` | 2 |
| `A2V2_EVAL_MAX_TOKENS` | 320000 |
| `A2V2_EVAL_WORKERS` | 20 |

The deployment controls are:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `A2V2_GPUS` | `0,1,2,3,4,5,6,7` | Eight physical GPU IDs exposed to each launch |
| `A2V2_PYTHON` | `python` | Interpreter used for preflight, `torch.distributed.run`, and evaluation |
| `A2V2_TORCHRUN` | unset | Optional single launcher executable replacing `python -m torch.distributed.run` |
| `A2V2_TRAIN_ENTRY` | `a2v2-train` | Training script passed to the distributed launcher |

Use `--fold N` for another single fold and `--fraction 025` or
`--fraction 001` for one reduced-label recipe. Use `--dry-run` to check
manifest resolution and print the full preflight → probe → burn-in → pretrain
resume → fine-tune → validation command graph without requiring CUDA.

Preflight requires exactly eight visible A100 devices, at least 39 GiB total
and 38 GiB currently free on every device, and at least 64 GiB free on the
output filesystem. The distributed probe exercises NCCL tensor collectives and
Gloo transport for the serialized per-rank RNG checkpoint state. A fresh run's
one-update burn-in uses the full paper model, data, token budget, accumulation,
AMP, and eight-rank topology; `--stop-at-update 1` changes only the stopping
point, not the configured scheduler horizon.

The script validates and resumes a stage when its own `checkpoint_last.pt`
exists. Do not change world size or batch variables between an interrupted run
and its resume: sampler and optimizer state belong to the stored topology.
Validation prefers `checkpoint_best.pt`, falls back to `checkpoint_last.pt`
with a warning, and records the selected checkpoint hash. Logs contain UTC
phase headers and append across invocations. The output directory also retains
package, GPU, manifest, recipe, and batch profile provenance.

## 1. Select the published recipe

The repository includes four primary recipes:

| Stage | Dataset | Recipe |
| --- | --- | --- |
| Pretraining | MeerKAT, 8 kHz | `configs/MeerKAT/a2v_large_pretrain_best.yaml` |
| Fine-tuning | MeerKAT, 8 kHz | `configs/MeerKAT/finetune_mixup_100.yaml` |
| Pretraining | Spotted hyena, 24 kHz | `configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml` |
| Fine-tuning | Spotted hyena, 24 kHz | `configs/hyenas/finetune_mixup_100.yaml` |

The 25% and 1% MeerKAT labeled-data recipes are
`configs/MeerKAT/finetune_mixup_025.yaml` and
`configs/MeerKAT/finetune_mixup_001.yaml`.

The checked-in files omit Hydra and Fairseq compatibility metadata. For a
future A2V2 experiment:

1. copy the nearest Animal2Vec 1.0 control rather than editing the control;
2. retain every value whose change affects data, tensor geometry, forward
   computation, optimization, validation, or saved output;
3. do not restore compatibility-only component names, log switches, composite
   optimizer wrappers, or unsupported Canny parameters; and
4. archive TensorBoard's `run/config` entry with the checkpoint, because it
   contains the recipe after command-line overrides and native defaults.

## 2. Prepare manifests and labels

Create one TSV for each split named by `dataset.train_subset` and
`dataset.valid_subset`. Its first line is the audio root. Later rows contain the
relative audio path, a tab, and the number of waveform samples:

```text
/datasets/MeerKAT
wav/00000001.wav\t80000
wav/00000002.wav\t80000
```

Pretraining needs only audio. Fine-tuning expects each `wav/name.wav` to have a
matching `lbl/name.h5` with these datasets:

| HDF5 dataset | Meaning |
| --- | --- |
| `start_frame_lbl` | Inclusive onset in waveform samples |
| `end_frame_lbl` | Exclusive offset in waveform samples |
| `lbl_cat` | Zero-based index into `task.unique_labels` |
| `foc` | One for a focal-animal event, zero otherwise |

Keep the published split membership unchanged. The MeerKAT recipes use
ten-second, 80,000-sample recordings. The frontend's total stride is 40 samples,
which gives an asymptotic frame rate of 200 Hz. The hyena recipes use 24 kHz
audio and a total stride of 120 samples, also 200 Hz.

`task.min_label_size` is a supervised-data filter: a labeled record is retained
only when its HDF5 file is larger than the configured byte count. Pretraining
recipes must leave this value at zero because their datasets deliberately have
no label paths. The cleaned Hyena pretraining recipe therefore retains every
otherwise valid row in `pretrain.tsv`; its earlier copied
`min_label_size: 1` value would have removed the entire unlabeled dataset in
the native loader.

Before a long run, use the CPU recipes with a few representative files to catch
manifest, sample-rate, and HDF5 errors.

## 3. Pretrain

The MeerKAT paper model has 16 shared transformer layers, embedding width 1024,
16 attention heads, an eight-layer context prenet, 12 independently masked
student clones, and a four-layer convolutional decoder. It trains for 384,230
optimizer updates. The EMA coefficient increases from 0.9997 to 1.0 through
update 300,000.

Launch the original four-worker layout with PyTorch DDP:

```bash
torchrun --standalone --nproc-per-node=4 "$(command -v a2v2-train)" \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/meerkat-pretrain \
  --device cuda
```

Important recipe values are preserved:

- `max_tokens: 408000` per worker
- `required_batch_size_multiple: 1`, overriding Fairseq's inherited default of
  eight for pretraining batches
- gradient accumulation of five microbatches
- Adam at `1e-4`, betas `(0.9, 0.98)`, epsilon `1e-6`
- weight decay `0.01`, gradient clipping at `1.0`
- 10,000 warmup updates followed by cosine decay
- A-weighted between-class waveform mixup on every example
- approximately 93% time masking from the published overlapping-span settings
- instance normalization of each teacher layer followed by a 16-layer average

The loss is the summed masked-position regression loss scaled by
`1 / sqrt(embed_dim)` and normalized by the global masked-position count before
the optimizer step.

Resume without changing the recipe or update target:

```bash
torchrun --standalone --nproc-per-node=4 "$(command -v a2v2-train)" \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/meerkat-pretrain \
  --resume /checkpoints/meerkat-pretrain/checkpoint_last.pt \
  --device cuda
```

## 4. Fine-tune

Fine-tuning averages all 16 shared-transformer layer outputs. For the first
10,000 updates, only the classifier is trained. The convolutional feature
encoder remains frozen; the transformer is unfrozen for the remaining 20,000
updates. The recipe applies time and channel masks, A-weighted waveform and
target mixup, and sigmoid focal loss.

Because the fine-tuning YAMLs omit `required_batch_size_multiple`, config
translation uses the legacy default of eight. Set the value when you want
different batch construction; the CPU smoke recipes use one for their tiny
datasets.

```bash
torchrun --standalone --nproc-per-node=4 "$(command -v a2v2-train)" \
  --config configs/MeerKAT/finetune_mixup_100.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/meerkat-finetune \
  --pretrained-checkpoint /checkpoints/meerkat-pretrain/checkpoint_last.pt \
  --device cuda
```

Validation follows `validate_after_updates` and `validate_interval_updates`.
Each pass logs normalized loss, micro precision, recall, F1, accuracy, and micro
average precision. It also runs the archived-compatible event matcher and logs
segmented precision, recall, F1, accuracy, macro AP, micro AP, classwise PR
curves, IoU distributions, splits, and mergers. `checkpoint_best.pt` tracks the
recipe's `checkpoint.best_checkpoint_metric`; the published MeerKAT recipe
selects framewise F1. Set the metric suffix to `segmented_f1` when an experiment
should select checkpoints by temporal event overlap.

The published recipes write TensorBoard events below each training directory
at `tb/`. For example:

```bash
tensorboard --logdir /checkpoints/meerkat-finetune/tb
```

The one-fold reproduction driver also writes standalone evaluation events to
`final-evaluation/tensorboard/`. The adjacent JSON report records that absolute
path and contains the same aggregate segmented metrics.

For the labeled-data ablations, point each recipe at a manifest directory whose
`train_0.tsv` contains the corresponding published 25% or 1% training split.
Do not randomly resample it on each run.

## 5. Evaluate and run inference

Run inference on the held-out recordings with the selected native fine-tuning
checkpoint:

```bash
a2v2-infer \
  /checkpoints/meerkat-finetune/checkpoint_best.pt \
  /datasets/MeerKAT/test_recording.wav \
  predictions.tsv \
  --segment-seconds 10 \
  --device cuda
```

The default threshold and average event-fusion settings come from the
fine-tuning YAML. Frame probabilities, embeddings, and timestamps are available
through `a2v2.workflows.InferenceRunner` when a complete evaluation script
needs them. Use the paper's unchanged test split and aggregation convention when
comparing with its reported micro average precision and focal-call F1.

Add `--audition` and use a `.csv` output path when researchers need to review
the same events as Adobe Audition range markers:

```bash
a2v2-infer \
  /checkpoints/meerkat-finetune/checkpoint_best.pt \
  /datasets/MeerKAT/test_recording.wav \
  predictions.csv \
  --audition \
  --segment-seconds 10 \
  --device cuda
```

The resulting file matches the six-column, tab-delimited marker layout from
the Animal2Vec 1.0 inference script. `Start` contains onset,
`Duration = offset - onset`, `Type` is `Cue`, and `Description` contains the
score rounded to three decimal places. The flag changes the file layout and
retains the A2V2 event set.

## 6. Checkpoint conversion instead of retraining

Published Fairseq weights can be converted and checked before committing to a
full training run. See `checkpoint-conversion.md`. The converter instantiates the
architecture from the matching YAML, maps every required tensor, and rejects a
checkpoint with missing core state or incompatible shapes.

## Reproducibility controls

Native checkpoints store the model, EMA teacher, optimizer, scheduler, AMP
scaler, update and epoch counters, sampler position, and each rank's Python,
NumPy, CPU, and CUDA random states. The distributed token sampler assigns
complete batches and drops an unmatched tail so every rank performs the same
number of backward passes.

For a defensible comparison:

1. Record the exact commit and YAML file.
2. Keep the published audio sample rate, split membership, and class order.
3. Use the recipe's worker count, token budget, and accumulation factor.
4. Resume only from a native checkpoint produced by the same world size.
5. Report results from the same validation or test aggregation used in the
   paper, including whether the focal output is included.

## Verification boundary

The automated suite compares native operations against the archived source for
Sinc convolution, ALiBi attention, masking, target construction, pretraining
loss, focal loss, and converted fine-tuning logits. It also runs pretraining,
resume, fine-tuning, validation, checkpointing, and inference on CPU fixtures.

The current development machine has no GPU and does not contain the full
datasets or published checkpoint files. It can establish implementation and
small-model numerical parity, but it cannot independently confirm the paper's
large-run metrics. Treat those metrics as reproduced only after completing the
full data and compute procedure above.
