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

## Local control and separate SLURM path

`scripts/reproduce_meerkat_paper.sh` remains the canonical local eight-A100
control. The modern Transformer work and SLURM support kept that file
byte-identical at SHA-256
`485b4b36fe1045174a392834bd9ee9848538ab496944631a0b2d514e22d4de18`.
It retains the same three training commands, resolved MeerKAT recipes,
fine-tuning activation-checkpoint override, and final evaluation flow.

Those frozen recipes use the legacy crop and compatible-resume defaults. Their
resume path remains supported. A process restart with random cropping is not
bit-exact because checkpoints omit the legacy collator's crop-generator state
and worker prefetch state. The driver emits a runtime warning if any batch in
the complete sampler epoch can crop, regardless of the restored cursor or
worker count. Use the separate modern recipes, which select stateless crops
and strict v2 provenance, when exact data-path resume is required. The frozen
YAML and local driver bytes remain unchanged.

`scripts/reproduce_meerkat_slurm.sh` provides a separate scheduler entry
point. It uses the frozen MeerKAT recipes by default, starts one torchrun agent
per node, runs one worker per GPU, creates phase-specific rendezvous IDs, and
runs final evaluation as a one-node, one-GPU step. It does not change the local
driver or establish paper equivalence for a new topology.
The pretraining RunContract reads `dataset.train_subset` from the resolved
pretraining config and fingerprints that manifest only.

The SLURM implementation passed Bash syntax, deterministic dry-run, mocked
scheduler, two-rank Gloo safe-point, frozen-driver, and full CPU tests. No real
`sbatch`, multi-node SLURM step, or scheduler preemption ran on the
development node. The unrun real-site gate covers one-node parity and at least
two nodes with two GPUs each for transport, shared-filesystem semantics,
checkpoint/resume, lock contention, and preemption/requeue behavior. Record
the site, software versions, allocation, date, and results before claiming
cluster validation.

Read [the SLURM guide](slurm.md) for commands, marker and lock contracts,
handoff rules, recovery, and the acceptance matrix.

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

The driver keeps the fine-tuning global token target while each rank processes
a larger microbatch:

```text
published: 426,667 × 4 ranks × 9 accumulation = 15,360,012 tokens/update
driver:    960,000 × 8 ranks × 2 accumulation = 15,360,000 tokens/update
```

Fine-tuning update 10,000 is the first update with a trainable Transformer
backbone. The checkpoint saved at update 10,000 contains the last frozen
forward; its first resumed forward retains activations for all eight prenet and
16 main Transformer blocks. On 40 GB A100s, the driver sets
`model.checkpoint_activations=true` for fine-tuning. PyTorch retains block
boundaries and recomputes each block during backward. Frozen training,
validation, evaluation, and inference bypass recomputation because they run
without encoder gradients.

Do not use `A2V2_FINETUNE_MAX_TOKENS=800000` as an OOM recovery setting for the
published MeerKAT manifests. All 53,114 training records contain 80,000 samples,
and the inherited batch-size multiple makes both 960,000 and 800,000 produce
eight-record microbatches. Lower token budgets also repartition shuffled
batches. Applying the saved numeric sampler cursor to that new partition would
duplicate and skip examples, so changing `max_tokens` is not an exact resume.

The complete batch controls are:

| Environment variable | Default |
| --- | ---: |
| `A2V2_PRETRAIN_MAX_TOKENS` | 408000 |
| `A2V2_PRETRAIN_UPDATE_FREQ` | 3 |
| `A2V2_FINETUNE_MAX_TOKENS` | 960000 |
| `A2V2_FINETUNE_UPDATE_FREQ` | 2 |
| `A2V2_EVAL_MAX_TOKENS` | 320000 |
| `A2V2_EVAL_WORKERS` | 20 |

Activation checkpointing is a fixed fine-tuning override in this 40 GB driver,
not an environment-variable batch control. It adds computation during the
unfrozen backward pass while preserving the batch and checkpoint cursor.

The deployment controls are:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `A2V2_GPUS` | `0,1,2,3,4,5,6,7` | Eight physical GPU IDs exposed to each launch |
| `A2V2_PYTHON` | `python` | Interpreter used for preflight, `torch.distributed.run`, and evaluation |
| `A2V2_TORCHRUN` | unset | Optional single launcher executable replacing `python -m torch.distributed.run` |
| `A2V2_TRAIN_ENTRY` | `a2v2-train` | Training script passed to the distributed launcher |
| `A2V2_OMP_NUM_THREADS` | `8` | Per-rank PyTorch CPU thread budget for distributed training and rank-zero validation |

The default gives the eight training ranks 64 PyTorch intra-op threads in
total. Together with the configured 20 data-loader workers per rank, the
steady-state training count is 224 threads on the 256-logical-CPU reproduction
host. During scheduled rank-zero validation, the validation loader can add
another 20 workers temporarily while the training loaders remain alive, for a
simple modeled count of 244 threads before runtime and helper threads.
Override `A2V2_OMP_NUM_THREADS` only with a positive integer; values of 16 or
more can oversubscribe this host during distributed loading.

Use `--fold N` for another single fold and `--fraction 025` or
`--fraction 001` for one reduced-label recipe. Use `--dry-run` to check
manifest resolution and print the full preflight → probe → burn-in → pretrain
resume → fine-tune → validation command graph without requiring CUDA.

Preflight requires exactly eight visible A100 devices, at least 39 GiB total
and 38 GiB currently free on every device, and at least 64 GiB free on the
output filesystem. The distributed probe exercises NCCL tensor collectives and
Gloo transport for the serialized per-rank RNG checkpoint state. A fresh run's
two-update burn-in uses the full paper model, data, token budget, accumulation,
AMP, and eight-rank topology; `--stop-at-update 2` changes only the stopping
point, not the configured scheduler horizon. Reaching update 2 is important:
it exercises a forward after AdamW8bit has allocated persistent optimizer
state, which a one-update burn-in cannot validate.

The script validates and resumes pretraining from `checkpoint_last.pt`. For
fine-tuning it prefers `checkpoint_last.pt`; when that file is absent after an
interruption, it selects the newest `checkpoint_best.pt`,
`checkpoint_<update>.pt`, or `checkpoint_epoch_<epoch>.pt`. Before torchrun,
preflight requires the selected file to match the training stage and contain
optimizer, scheduler, and one RNG state for each of the eight ranks. Do not
change world size or batch variables between an interrupted run and its resume:
sampler and optimizer state belong to the stored topology.

The activation-checkpointing flag changes execution policy and adds no
checkpoint tensors. The driver can therefore enable it while resuming the
untouched update-10,000 checkpoint under the same sampler topology.

Scheduled training validation runs on rank zero. CUDA workers exchange the
result through a CPU Gloo control group with a two-hour timeout, so the other
ranks do not hold an NCCL collective while rank zero evaluates the full split.

Validation prefers `checkpoint_best.pt`, falls back to `checkpoint_last.pt`
with a warning, and records the selected checkpoint hash. Logs contain UTC
phase headers and append across invocations. The output directory also retains
package, GPU, manifest, recipe, and batch and activation-memory profile
provenance.

## Modern eight-A100 benchmark

`scripts/animal2vec2_benchmark.sh` is the performance-oriented companion to
the frozen Animal2Vec 1.0 control. It follows the same preflight, eight-rank
NCCL probe, two-update burn-in, checkpoint validation, full pretraining,
fine-tuning, and final-evaluation sequence. It does not modify or replace
`scripts/reproduce_meerkat_paper.sh`.

Launch fold 0 from the repository root:

```bash
bash scripts/animal2vec2_benchmark.sh \
  /local/datasets/MeerKAT_10s_2024-06-12/manifests \
  /experiments/animal2vec2-meerkat-fold0
```

The benchmark accepts `--fold N` and intentionally has no label-fraction
option. It always uses `train_<fold>.tsv` and `valid_<fold>.tsv`, so it runs
the approved 100% labeled-data condition only. `--dry-run` checks those paths
and prints the complete command graph without allocating a GPU.

The launcher uses the paired files in `configs/modern/` and pins their defining
choices again as command-line overrides:

- RoPE with strict FlashAttention and ALiBi disabled;
- frame plus CLS regression during pretraining and a CLS sequence head during
  fine-tuning;
- packed GEGLU and DeepScaleLM initialization;
- AdaGC, AdamW8bit, and cosine weight-decay annealing from `0.01` to `0.0`;
- dynamic regional pretraining compilation plus complete-model fine-tuning
  compilation; and
- no activation checkpointing.

Pretraining uses `408000 × 8 × 3 = 9,792,000` tokens per optimizer update for
384,230 updates. Fine-tuning uses `960000 × 8 × 2 = 15,360,000` tokens per
optimizer update for 30,000 updates. These preserve the successful local
reproduction's eight-GPU effective batch and training horizon. Do not change
token, accumulation, or world-size settings when resuming a checkpoint.

The final memory acceptance run used eight A100-SXM4-40GB GPUs,
bitsandbytes 0.50.0, the production manifests, strict FlashAttention, no
activation checkpointing, and a clean dynamic stack-compilation policy. A
fresh run reached update 2, then a checkpoint resume reached update 10 across
changing retained-token lengths. There were no graph breaks, exact-length
recompiles, OOMs, or secondary NCCL timeouts. The worst reported peak was
34.20 GiB allocated and 34.81 GiB reserved on a 39.49 GiB device.

`612000 × 2` is not safe on this host. Its first optimizer update can fit, but
the next forward fails after AdamW8bit state is resident; the same failure was
reproduced with compilation disabled, so allocator settings and compiler graph
caches are not the root cause. The former one-update burn-in therefore gave a
false positive. The launcher now uses `408000 × 3` and burns in through update
2. Both partitions have the same nominal effective token budget, but only the
new default retains adequate post-optimizer headroom.

Variable `ids_keep` lengths did expose a separate compiler issue: compiling the
whole pretraining model specialized its continuation and accumulated large
graphs. The current policy leaves masking eager, compiles the four student and
teacher Transformer stacks, and marks their token axes as dynamic.
Expected finite variants cover student/teacher grad mode, prenet/main depth,
and aligned/unaligned Inductor kernels; they do not specialize exact token
lengths. Never override the default batch upward without a two-update burn-in,
and never change either batch value while resuming an existing checkpoint.

Fine-tuning was tested separately with its backbone unfrozen from update zero.
The default `960000 × 2` completed one full update with a worst reserved peak
of 29.29 GiB. The batch-equivalent `1920000 × 1` profile directly OOMed during
the compiled forward with 39.40 GiB in use. Keep the fine-tuning default on
40GB A100s. These bounded measurements describe this host and software stack;
they are evidence for the launcher's defaults, not a throughput guarantee for
other datasets or GPUs.

The environment preflight additionally requires bitsandbytes `>=0.50,<0.51`
and confirms that its native CUDA library loaded. The default
`A2V2_BNB_CUDA_VERSION=auto` omits `BNB_CUDA_VERSION`, allowing version 0.50.1
to select its compatible packaged CUDA binary. Set a numeric suffix only when
deliberately selecting another compatible binary. Every training rank uses
the same persistent cache at
`OUTPUT_DIR/environment/torchinductor-cache`; set
`A2V2_TORCHINDUCTOR_CACHE_DIR` to move it to another fast local filesystem.

The final step evaluates `checkpoint_best.pt` with
`a2v2-evaluate-sequence`, falling back to `checkpoint_last.pt` when needed.
It writes atomic JSON to
`OUTPUT_DIR/final-evaluation/final-evaluation-report.json`. These are
recording-level CLS metrics. They are not frame-event metrics from the frozen
Animal2Vec 1.0 evaluator, so compare the two model families with the matching
metric definition.

Before evaluation, the launcher compares `checkpoint_best.pt` with the current
last or resume checkpoint's strict run fingerprint. It falls back to the
current checkpoint when an old best file belongs to another manifest or run.
The exact evaluated checkpoint is recorded in
`evaluation-checkpoint-sha256.txt`; stage, update, size, and preflight results
are stored in `evaluation-checkpoint-preflight.json`. The checkpoint hash is
checked again before the metrics report is published.

The benchmark supports the same batch and process environment variables as
the local control, plus:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `A2V2_BNB_CUDA_VERSION` | `auto` | Automatic bitsandbytes CUDA-binary selection; set a numeric suffix for an explicit compatible override |
| `A2V2_TORCHINDUCTOR_CACHE_DIR` | `OUTPUT_DIR/environment/torchinductor-cache` | Persistent compile cache shared by benchmark stages |
| `A2V2_SEQUENCE_EVAL_ENTRY` | `a2v2-evaluate-sequence` | Sequence-evaluator executable |

Strict FlashAttention fails at launch or first forward if PyTorch cannot
dispatch its Flash kernel. The script does not fall back to the math backend,
does not enable activation checkpointing, and does not reduce batch size after
an OOM. Those failures indicate that the host no longer matches this benchmark
profile and should be investigated rather than hidden by a silent policy
change.

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

On 40 GB devices, add
`--override model.checkpoint_activations=true` to trade block recomputation for
lower unfrozen-backbone activation memory. The field defaults to false, so
canonical YAML files and inference retain their existing execution path.

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
