# A2V2

A2V2 is an extensible, native-PyTorch codebase for bioacoustic representation
learning. Its first resident method is a compatibility-focused rewrite of
Animal2Vec 1.0. It implements the published architecture, self-supervised
pretraining objective, supervised fine-tuning path, official checkpoint
conversion, long-recording inference, event evaluation, mixed precision,
distributed training, and native checkpoint resume without Fairseq. See the
opt-in strict policy below for exact data-path resume.

The published recipes under `configs/MeerKAT/` and `configs/hyenas/` are
**Animal2Vec 1.0 reproduction baselines**. They are frozen controls intended
for reproducing the original paper and comparing future A2V2 methods against
it; they are not configurations for a new method called Animal2Vec 2.0. The
[configuration guide](configs/README.md) records this boundary and reserves a
clear home for later experiment families.

The implementation lives in the flat `a2v2/` directory at the repository
root. The distribution name is `a2v2`, and installed commands begin with
`a2v2-`. Future architectures, objectives, and workflows can therefore live
under the same project identity without renaming the Animal2Vec 1.0 control.

> **Verification status, 2026-07-23:** CPU, one-A100 FP32/FP16, official
> checkpoint conversion, layer/output/gradient/update parity, two- and
> four-rank NCCL, exact distributed resume, and unchanged four-rank recipe
> memory gates passed. Data/evaluator verification is partial because the
> publisher release does not include authoritative split manifests. Full paper
> training and all paper metrics have not been completed. The paper has
> **not** been reproduced. See
> [the complete A100 report](docs/gpu-verification-20260723.md).

## Contents

- [What this repository is](#what-this-repository-is)
- [Official paper, code, weights, and datasets](#official-paper-code-weights-and-datasets)
- [Source snapshot and exclusions](#source-snapshot-and-exclusions)
- [Feature overview](#feature-overview)
- [Repository map](#repository-map)
- [Code reading guide](docs/code-guide.md)
- [Architecture at a glance](#architecture-at-a-glance)
- [Installation](#installation)
- [Opt-in modern recipes](#opt-in-modern-recipes)
- [SLURM launch](#slurm-launch)
- [Data and manifest preparation](#data-and-manifest-preparation)
- [Downloading the official artifacts](#downloading-the-official-artifacts)
- [Converting official Fairseq weights](#converting-official-fairseq-weights)
- [Starting pretraining](#starting-pretraining)
- [Resuming pretraining](#resuming-pretraining)
- [Starting fine-tuning](#starting-fine-tuning)
- [Resuming fine-tuning](#resuming-fine-tuning)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Native checkpoint semantics](#native-checkpoint-semantics)
- [Configuration and overrides](#configuration-and-overrides)
- [Testing and GPU verification](#testing-and-gpu-verification)
- [Documentation guide](#documentation-guide)
- [Troubleshooting](#troubleshooting)
- [Reproducing the paper](#reproducing-the-paper)
- [Citation and license](#citation-and-license)

## What this repository is

Animal2Vec is a raw-waveform bioacoustic representation learner designed for
long recordings in which biologically interesting events are sparse, brief,
noisy, and strongly class-imbalanced. Pretraining uses a masked student and an
exponential-moving-average teacher in a Data2Vec-style self-distillation
scheme. Fine-tuning turns the pretrained encoder into a framewise multilabel
classifier and then fuses frame probabilities into event intervals.

The original public implementation was tied to an archived Fairseq, Hydra, and
OmegaConf stack. This rewrite preserves the architecture, published YAML
recipes, tensor mappings, masking behavior, data conventions, and numerical
equations while replacing that runtime with:

- ordinary PyTorch modules;
- typed configuration parsing;
- standard `torchrun` distributed launch;
- native AMP and optimizer/scheduler management;
- a versioned, plain-container checkpoint schema;
- strict conversion of official Fairseq checkpoints; and
- dependency-light training, inference, and event evaluation.

The native runtime has five direct dependencies:

- PyTorch;
- NumPy;
- SoundFile;
- h5py; and
- PyYAML.

It does not require Fairseq, Hydra, OmegaConf, torchaudio, timm,
Transformers, TensorFlow, pandas, SciPy, scikit-learn, or librosa. The
standalone tree contains no legacy runtime or parity exporter that imports
those packages.

## Official paper, code, weights, and datasets

These are the authoritative public resources associated with Animal2Vec and
the paper:

| Resource | Link | Notes |
| --- | --- | --- |
| Published article | [Methods in Ecology and Evolution, DOI 10.1111/2041-210X.70218](https://doi.org/10.1111/2041-210X.70218) | Peer-reviewed article published in 2026 |
| Open preprint | [arXiv:2406.01253](https://arxiv.org/abs/2406.01253) | 2024 preprint and downloadable PDF |
| Original project repository | [github.com/livingingroups/animal2vec](https://github.com/livingingroups/animal2vec) | Original Fairseq-based implementation |
| Archived original code | [Zenodo DOI 10.5281/zenodo.17640321](https://doi.org/10.5281/zenodo.17640321) | Preservation record for the original code |
| MeerKAT audio and labels | [Edmond DOI 10.17617/3.0J0DYB](https://doi.org/10.17617/3.0J0DYB) | Public dataset; consult the record for its CC BY-NC terms |
| Official pretrained and fine-tuned weights | [Edmond DOI 10.17617/3.ETPUKU](https://doi.org/10.17617/3.ETPUKU) | Official Fairseq `.pt` checkpoints; conversion is required for this runtime |
| NIPS4Bplus annotations | [Figshare DOI 10.6084/m9.figshare.6798548](https://doi.org/10.6084/m9.figshare.6798548) | Public temporal annotations used by the NIPS4Bplus benchmark |
| NIPS4Bplus paper | [PeerJ Computer Science DOI 10.7717/peerj-cs.223](https://doi.org/10.7717/peerj-cs.223) | Dataset and protocol description |
| Xeno-canto | [xeno-canto.org](https://xeno-canto.org/) | Source collection used by the paper's bird-transfer pretraining; the exact paper subset is not bundled here |

The official weights record contains:

- `animal2vec_large_pretrained_MeerKAT_240507.pt`; and
- `animal2vec_large_finetuned_MeerKAT_240507.pt`.

Those files are serialized in the original Fairseq format. Do not pass them
directly to `a2v2-infer` or `--pretrained-checkpoint`; convert them first
as described below.

The published article describes Animal2Vec and MeerKAT as open resources. Code,
data, weights, and third-party datasets do not necessarily share one license.
Read the license on each linked record before redistribution or commercial use.
The code in this repository is under the MIT license in [LICENSE](LICENSE).

## Source snapshot and exclusions

This directory was assembled from the verified workspace at Git commit:

```text
b1be8d45eaf9307458dafad9902ada0422a8da52
```

It also includes the narrow working-tree corrections and durable tests produced
during A100 verification. Those corrections are itemized in
[the GPU verification report](docs/gpu-verification-20260723.md), which also
records the original commit, complete diff provenance, environment locks, GPU
UUIDs, test results, checkpoint hashes, and first-divergence investigations.

This source distribution includes:

- native package source;
- published and smoke-test configurations;
- unit, integration, GPU, NCCL, and resume tests;
- conversion, dataset-audit, and manifest-audit utilities;
- a direct public Python API and installed CLI commands;
- reader-focused source documentation; and
- the MIT license and packaging metadata.

The standalone distribution omits the archived `nn/` implementation,
Fairseq-only parity exporters, obsolete compatibility wrappers, and their
analysis dependencies. The final GPU report preserves the numerical evidence
that those archived tools produced.

It deliberately excludes generated or machine-specific state:

- `.git/` history and linked worktrees;
- Python virtual environments and interpreter builds;
- `__pycache__`, pytest caches, and benchmark caches;
- downloaded datasets;
- downloaded or converted model weights;
- native training checkpoints;
- rank logs and generated metrics; and
- the 273 GB A100 verification artifact bundle under the source workspace's
  `results/gpu_verification/` directory.

The documentation records where those artifacts came from and how to regenerate
them. Keeping them outside the source tree prevents accidental publication of
hundreds of gigabytes of data and avoids conflating source control with
experiment storage.

## Feature overview

### Native raw-audio frontend

- WAV/audio loading through SoundFile.
- Mono selection or channel averaging.
- Native windowed-sinc resampling without torchaudio.
- Per-recording waveform normalization.
- Learnable Sinc filters with deterministic CUDA reflection padding.
- The published stack of strided local convolutional feature layers.
- Layer, group, and instance normalization behavior matching archived
  checkpoints.
- P-Swish activation support.
- Exact convolution length arithmetic and frame timestamps.

### Positional and contextual encoding

- Convolutional relative positional encoder.
- Configurable context prenet.
- Shared Transformer stack.
- Multi-head self-attention with ALiBi.
- Learned per-head ALiBi scaling from the published model.
- Memory-bounded ALiBi selection that gathers required rows and columns before
  clone scaling.
- Pre-norm/post-norm and dropout settings translated from the checked-in
  recipes.

### Self-supervised pretraining

- Data2Vec-style student/teacher training.
- EMA teacher state with configurable start/end decay and annealing horizon.
- Independently masked cloned student batches.
- Span masking with the archived overlap and padding behavior.
- Top-layer teacher target collection and normalization.
- Configurable layer averaging.
- A-weighted between-class waveform mixup.
- Convolutional reconstruction decoder.
- Masked-position regression loss with distributed sample-size normalization.
- Exact handling of AMP overflow retries without advancing optimizer,
  scheduler, EMA, update, sampler, or checkpoint state.

### Supervised fine-tuning

- Native initialization from a pretraining checkpoint.
- Layer-averaged Transformer embeddings.
- Framewise multilabel classifier.
- Time and channel masking.
- Source and target mixup.
- Encoder-freeze and unfreeze schedules.
- Permanently frozen local feature encoder when requested by the recipe.
- Sigmoid focal loss and label-smoothing configuration.
- Framewise precision, recall, F1, accuracy, and average precision.
- Best-checkpoint tracking using the configured metric.

### Data pipeline

- Fairseq-style TSV manifests with an audio root and sample counts.
- WAV/HDF5 pair resolution.
- Dense frame targets from interval annotations.
- Token-budget batching.
- Required batch-size multiples.
- Legacy worker-random cropping and opt-in stateless cropping.
- Distributed assignment of complete batches.
- Rank-aware, checkpointable sampler position.
- Correct distinction between prefetched and training-consumed batches.
- Deterministic DataLoader ownership, with sampler-coordinate crop seeds in
  strict stateless runs.

### Optimization and distributed training

- Adam optimizer.
- Cosine update scheduler with warmup and minimum learning rate.
- Gradient accumulation.
- Gradient clipping.
- PyTorch AMP and GradScaler.
- Gloo on distributed CPU and NCCL on distributed CUDA.
- Standard `torchrun` launch.
- Exact world-size validation against the resolved recipe.
- Stable DDP bucket/traversal settings for bitwise resume.
- Structured per-rank elapsed-time and CUDA-memory summaries.

### Opt-in modern Transformer and training modes

- RoPE or ALiBi position encoding with explicit resolution rules.
- PyTorch SDPA with selectable fallback or strict Flash enforcement.
- Direct CLS regression during pretraining and sequence-level CLS fine-tuning.
- Packed GEGLU feed-forward blocks and DeepScaleLM residual scaling.
- AdaGC with checkpointed per-parameter norm history.
- AdamW plus optional bitsandbytes Adam8bit and AdamW8bit.
- Cosine weight-decay annealing on the successful-update clock.
- In-place `torch.compile` policy with recorded model or Transformer-block scope.
- The existing non-reentrant activation-checkpointing option.
- Separate SLURM launchers with torchrun topology and preemption contracts.

### Checkpoints and exact resume

- Versioned native checkpoint schema.
- Model and EMA teacher tensors.
- Optimizer and scheduler state.
- AMP scaler state.
- Update, epoch, batch, and best-metric state.
- Sampler epoch and next-consumed position.
- Python, NumPy, CPU PyTorch, and rank-local CUDA RNG state.
- Multi-rank RNG gathering into one rank-0 checkpoint.
- Device-safe RNG restore after arbitrary `map_location`.
- Exact uninterrupted-versus-resumed comparison utility.

New checkpoints fingerprint the complete mathematical task and dataset
configuration, the resolved training-manifest path and ordered bytes, and the
ordered sampler population and sizes. `checkpoint.resume_policy: strict`
requires that provenance before model, optimizer, scheduler, or clipper
construction. `dataset.crop_strategy: stateless` derives each crop from the
checkpointed sampler epoch and ordered occurrence, so DataLoader worker
prefetch cannot change the resumed crop.

The defaults remain `resume_policy: compatible` and `crop_strategy: legacy`
for old checkpoints and frozen recipes. Compatible mode warns when a
checkpoint lacks full provenance. A legacy random crop with multiple workers
also warns because a restarted process can choose a different window. Do not
describe that combination as bit-exact.

### Official weight conversion

- Strict conversion of official pretraining checkpoints.
- Strict conversion of official fine-tuning checkpoints.
- Explicit architecture construction from matching YAML recipes.
- Student, teacher/EMA, frontend, Transformer, decoder, encoder wrapper, and
  classifier mappings.
- Shape checks and required-key coverage.
- Machine-readable conversion report.
- Plain native result with no Fairseq import at inference time.

### Inference and evaluation

- Strict loading of native fine-tuning checkpoints.
- Long-recording chunking.
- Mono/channel selection and sample-rate conversion.
- Correct final-partial-segment trimming.
- Frame probabilities, averaged embeddings, and timestamps.
- Average or maximum event fusion.
- Event onset, offset, label, and score TSV output.
- Adobe Audition marker CSV output matching the Animal2Vec 1.0 six-column
  schema.
- Empty-recording handling.
- Pure-PyTorch archived event matching and aggregation.
- Strict IoU boundary semantics, split/merger tracking, macro/micro AP, and
  focal-threshold metrics.
- Tied-score average precision matching the archived scikit-learn behavior
  without importing scikit-learn into the runtime.

### Verification assets

- 162-test clean native CPU suite, plus 14 CUDA tests skipped on CPU.
- 14 durable one-A100 GPU-marked tests.
- The final report and hashes from official FP32/FP16
  layer/output/gradient/update comparisons.
- Two- and four-rank NCCL probes.
- Two- and four-rank exact-resume probes.
- Full-state checkpoint comparator that reports the first differing path.
- Exhaustive MeerKAT audio/HDF5 audit.
- Manifest membership and hash audit.
- The final GPU report records the archived/native event-evaluator comparison.

## Repository map

```text
animal2vec2.0/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── a2v2/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   └── workflows.py
├── configs/
│   ├── README.md
│   ├── MeerKAT/
│   ├── hyenas/
│   ├── modern/
│   ├── cpu_smoke_pretraining.yaml
│   └── cpu_smoke_finetuning.yaml
├── docs/
│   ├── code-guide.md
│   ├── slurm.md
│   ├── checkpoint-conversion.md
│   ├── reproducing-paper.md
│   ├── gpu-verification-20260723.md
│   ├── refactoring-design.md
│   ├── refactoring-plan.md
│   ├── documentation-design.md
│   └── documentation-plan.md
├── scripts/
│   ├── audit_meerkat_dataset.py
│   └── audit_meerkat_manifests.py
└── tests/
    ├── fixtures/
    ├── gpu/
    ├── integration/
    └── unit/
```

### Important paths

| Path | Role |
| --- | --- |
| `a2v2/config.py` | Typed recipe schema, YAML translation, overrides, and checkpoint serialization |
| `a2v2/data.py` | Audio geometry and I/O, labels, manifests, collation, and token samplers |
| `a2v2/model.py` | Neural primitives, encoder, objectives, EMA teacher, pretraining, and fine-tuning |
| `a2v2/training.py` | Optimizer, scheduler, checkpoints, metrics, AMP, DDP reduction, and resume |
| `a2v2/workflows.py` | Event evaluation, inference, conversion, training orchestration, and commands |

## Architecture at a glance

The main pretraining data flow is:

```text
raw waveform
  → optional A-weighted between-class mixup
  → learnable Sinc/local convolutional encoder
  → feature projection and normalization
  → convolutional positional encoder
  → context prenet
  ├─→ unmasked EMA teacher Transformer
  │     → normalized top-layer target average
  └─→ independently masked cloned student Transformer
        → convolutional decoder
        → masked-position predictions
  → sample-size-normalized regression loss
```

Fine-tuning reuses the pretrained local/context/Transformer encoder:

```text
raw waveform
  → optional waveform and target mixup
  → pretrained local convolution frontend and feature projection
  → optional time/channel feature masking
  → positional encoder and pretrained Transformer
  → average selected Transformer layers
  → linear multilabel classifier
  → sigmoid focal loss
  → frame probabilities
  → event fusion
```

Read [the code guide](docs/code-guide.md) for a file-by-file walk through these
flows, tensor notation, equations, compatibility-sensitive operation order, and
the current responsibility-based import map.

## Installation

### Requirements

- Python 3.10 or newer.
- A PyTorch build appropriate for the host CPU or NVIDIA driver.
- A C library usable by SoundFile, normally `libsndfile`.
- CUDA/NCCL only for GPU and distributed GPU execution.

The A100 authority environment used:

- CPython 3.14.6;
- PyTorch 2.12.1+cu130;
- CUDA runtime 13.0;
- cuDNN 9.20.0; and
- NCCL 2.29.7.

That is a verified environment, not a requirement that every installation use
those exact versions. For platform-specific PyTorch commands, use the
[official PyTorch installer](https://pytorch.org/get-started/locally/).

### Create an isolated native environment

```bash
cd /path/to/animal2vec2.0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the correct PyTorch wheel for the machine first. Then install the
project:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[test]'
```

The two modern examples select bitsandbytes AdamW8bit. Install its pinned
optional dependency with:

```bash
python -m pip install -e '.[bnb]'
```

The supported range is `bitsandbytes>=0.50,<0.51`. On the verified CUDA 13.3
host, bitsandbytes 0.50.1 automatically loads its compatible packaged CUDA 13.2
binary; no `BNB_CUDA_VERSION` override is needed. An explicit override remains
available for advanced setups that deliberately select another compatible
packaged binary.

Confirm that the four console commands exist:

```bash
a2v2-train --help
a2v2-infer --help
a2v2-evaluate-sequence --help
a2v2-convert-checkpoint --help
```

Python callers can invoke `train_main`, `infer_main`,
`evaluate_sequence_main`, or `convert_checkpoint_main` from
`a2v2.workflows`. Multi-process launches should pass the installed training
script to `torchrun`, as shown below.

### Verified A100 environment

The [A100 verification report](docs/gpu-verification-20260723.md) records the
exact CPython build, CUDA wheel, package freezes, GPU allocation, NCCL
topology, checkpoint hashes, and measured memory. The
[code guide](docs/code-guide.md) lists the GPU tests that should run after a
source change.

Keep Fairseq out of the native environment. If a raw official checkpoint
embeds unavailable Python objects, create a plain tensor/config export in a
separate external environment as described under conversion.

## Opt-in modern recipes

The paired modern examples live outside the frozen reproduction directories:

| Stage | Recipe |
| --- | --- |
| Pretraining with frame and CLS regression | `configs/modern/rope_cls_geglu_pretrain.yaml` |
| Sequence-level fine-tuning | `configs/modern/rope_cls_geglu_finetune.yaml` |

Both files define the same 16-layer, 1,024-dimensional encoder with an
eight-layer prenet, RoPE, strict Flash attention, a CLS token, packed GEGLU,
and DeepScaleLM. They also select activation checkpointing, AdaGC, AdamW8bit,
cosine weight-decay annealing, dynamic `torch.compile`, stateless crop
coordinates, and strict resume provenance. Pretraining keeps update-dependent
masking eager and compiles the four student and teacher Transformer stacks;
fine-tuning compiles the complete model. The stack inputs mark the
retained-token axis as dynamic, avoiding one large graph per masked-token length.

For the full-label MeerKAT benchmark on the verified eight-A100 host, run the
complete pretrain, fine-tune, and sequence-evaluation workflow with:

```bash
bash scripts/animal2vec2_benchmark.sh \
  /local/datasets/MeerKAT_10s_2024-06-12/manifests \
  /experiments/animal2vec2-meerkat-fold0
```

Use `--dry-run` first to inspect every resolved command. The benchmark keeps
the reproduction driver's eight-rank effective token batches and update
horizons, using the hardware-verified `408000 × 3` pretraining partition. It
uses the 100% label split and overrides activation checkpointing off. It
requires bitsandbytes 0.50 with a loaded CUDA backend and defaults to the wheel's
automatic CUDA-binary selection. It also keeps one persistent TorchInductor
cache below the output directory so resumed jobs can reuse compiled artifacts.
The burn-in completes two updates so it exercises a forward after AdamW8bit
has materialized optimizer state. Set `A2V2_BNB_CUDA_VERSION` when you need
another compatible packaged binary. See the
[reproduction guide](docs/reproducing-paper.md#modern-eight-a100-benchmark)
for checkpoint, evaluation, and environment controls.

Train the pretraining example after replacing its manifest path:

```bash
torchrun --standalone --nproc-per-node=4 "$(command -v a2v2-train)" \
  --config configs/modern/rope_cls_geglu_pretrain.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/modern-pretrain \
  --device cuda
```

Pass the resulting native checkpoint to the paired fine-tuning recipe:

```bash
torchrun --standalone --nproc-per-node=4 "$(command -v a2v2-train)" \
  --config configs/modern/rope_cls_geglu_finetune.yaml \
  --pretrained-checkpoint /checkpoints/modern-pretrain/checkpoint_last.pt \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/modern-finetune \
  --device cuda
```

Evaluate its validation manifest on one device:

```bash
a2v2-evaluate-sequence \
  /checkpoints/modern-finetune/checkpoint_last.pt \
  --trust-checkpoint \
  --config configs/modern/rope_cls_geglu_finetune.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override dataset.valid_subset=valid_0 \
  --device cuda
```

PyTorch writes native `.pt` checkpoints in a pickle-backed format.
`--trust-checkpoint` permits deserialization. Pass it for a checkpoint from
a source you trust. The flag does not make pickle safe.

The evaluator restores the encoder architecture from the checkpoint's
pretraining config, uses a strict model-state load, and prints sequence
metrics as JSON. It does not restore training topology or optimizer state.

Let bitsandbytes 0.50 select its compatible packaged CUDA binary by default.
Strict Flash raises an error when PyTorch cannot dispatch the Flash kernel;
select `model.attention_backend=sdpa` for kernel fallback. Read the
[configuration guide](configs/README.md) before changing architecture fields.

CLS and GEGLU add checkpoint tensors, and DeepScaleLM changes residual
equations. The modern pair cannot reinterpret published Animal2Vec 1.0
weights. Fine-tuning construction checks the CLS requirement and strict state
loading checks every key and shape.

## SLURM launch

`scripts/reproduce_meerkat_slurm.sh` runs one torchrun agent per node and one
worker per allocated GPU. The commands below use the frozen MeerKAT recipes.
The SLURM path remains separate from the byte-identical local paper driver.
Its pretraining RunContract resolves the manifest named by the selected
config's `dataset.train_subset`; it does not assume `pretrain.tsv`.

Submit pretraining:

```bash
sbatch scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase pretrain --nodes 2 --gpus-per-node 4
```

Submit fine-tuning after pretraining has produced its handoff checkpoint:

```bash
sbatch scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase finetune --nodes 2 --gpus-per-node 4
```

Submit pretraining, fine-tuning, and one-GPU final evaluation as one staged
batch job:

```bash
sbatch scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase all --nodes 2 --gpus-per-node 4
```

Render the full command graph without SLURM, CUDA, manifests, or checkpoints:

```bash
bash scripts/reproduce_meerkat_slurm.sh \
  /datasets/MeerKAT/manifests /shared/runs/meerkat \
  --phase all --nodes 2 --gpus-per-node 4 \
  --job-id dryrun-4815 --dry-run
```

Add account, partition, wall-time, and site module directives through your
submit environment. The launcher requests `SIGUSR1` with five minutes of lead
time and returns exit 75 after it validates a new coordinated checkpoint. It
does not call `scontrol requeue`; the site or submitter owns retry policy.
Read the [SLURM guide](docs/slurm.md) for rendezvous, locking, resume,
completion markers, troubleshooting, and the unrun real-cluster acceptance
gate.

## Data and manifest preparation

### Manifest format

Training reads one TSV file per split. The first line is the audio root. Every
later line contains a relative audio path, a tab, and the number of waveform
samples:

```text
/datasets/MeerKAT
wav/00000001.wav	80000
wav/00000002.wav	80000
```

For a recipe with:

```yaml
task:
  data: /datasets/MeerKAT/manifests
dataset:
  train_subset: pretrain
```

the trainer opens:

```text
/datasets/MeerKAT/manifests/pretrain.tsv
```

Fine-tuning similarly uses `train_0.tsv`, `valid_0.tsv`, or the few-shot split
name selected by the YAML.

### Recommended dataset layout

```text
/datasets/MeerKAT/
├── wav/
│   ├── 00000001.wav
│   └── 00000002.wav
├── lbl/
│   ├── 00000001.h5
│   └── 00000002.h5
└── manifests/
    ├── pretrain.tsv
    ├── train_0.tsv
    ├── valid_0.tsv
    ├── train_0_few_0.tsv
    └── train_0_few_2.tsv
```

The loader derives a label path by replacing the final `wav`, `audio`, or
`flac` directory component with `lbl` and changing the file suffix to `.h5`.

### HDF5 label schema

Fine-tuning label files contain:

| Dataset | Meaning |
| --- | --- |
| `start_frame_lbl` | Inclusive event start in waveform samples |
| `end_frame_lbl` | Exclusive event end in waveform samples |
| `lbl_cat` | Zero-based index into `task.unique_labels` |
| `foc` | Optional focal-animal flag |
| `start_time_lbl` | Optional onset in seconds |
| `end_time_lbl` | Optional offset in seconds |
| `lbl` | Optional human-readable class label |

The published MeerKAT class order is:

```text
beep, synch, sn, cc, ld, oth, mo, al, soc, agg, eating, focal
```

The first eleven entries are call/event categories. `focal` is an additional
output channel activated when an event's focal flag is set.

### Preparing custom audio

Create WAV/HDF5 pairs and TSV manifests with the data tooling used by your
project. The trainer only requires the schema documented above. Keep
resampling, segmentation, and split generation in a separate preparation
environment when those jobs need analysis packages.

Run both audit scripts after preparation. They catch path, frame-count, label,
fold-membership, and hashing errors before a long training job starts.

### Audit before training

The repository includes:

```bash
python scripts/audit_meerkat_dataset.py \
  /datasets/MeerKAT \
  --workers 32 \
  --batch-size 512 \
  --output dataset-audit.json

python scripts/audit_meerkat_manifests.py \
  /datasets/MeerKAT/manifests \
  --dataset-audit dataset-audit.json \
  --output manifest-audit.json
```

The dataset audit validates every WAV/HDF5 pair, sample rate, channel count,
frame count, HDF5 key/array structure, interval bound, and label category. The
manifest audit checks membership, duplicates, missing files, fold overlap,
few-shot containment, and stable membership hashes.

The released MeerKAT archive audited during GPU verification contained 384,592
audio/label pairs and 1,068.311 hours of 8 kHz audio. It contained 66,397
event-bearing recordings, while the paper reports 66,398 labelled samples.
It also contained 31,273 PCM24 files despite a paper-level 16-bit description.
These discrepancies are documented; do not silently “correct” the release.

### Official split warning

The public release available during verification did not contain the exact
official fold and few-shot TSV manifests. Seed-1612 manifests were regenerated
to exercise loading, memory, DDP, and resume, but they are not authoritative
paper splits. For a paper reproduction, obtain publisher-authoritative
membership and preserve its hashes.

## Downloading the official artifacts

### MeerKAT dataset

Open the [MeerKAT Edmond record](https://doi.org/10.17617/3.0J0DYB), accept the
record's terms, and download the current release. Keep the downloaded archive
outside this repository or in a Git-ignored data mount.

The archive verified on 2026-07-23 was:

```text
MeerKAT_10s_2024-06-12.zip
MD5 6a5065ddda33d2bfaad118a418d9b965
size 24,844,330,626 bytes
```

Verify the checksum before extraction:

```bash
md5sum MeerKAT_10s_2024-06-12.zip
```

Publisher records can change versions. If a current record intentionally
contains a new version, preserve both the publisher version identifier and the
new checksum instead of forcing it to match the historical value above.

### Official weights

Open the [Animal2Vec weights record](https://doi.org/10.17617/3.ETPUKU) and
download the two official `.pt` files.

The publisher MD5 and verification SHA-256 values were:

| File | Publisher MD5 | Verification SHA-256 |
| --- | --- | --- |
| `animal2vec_large_pretrained_MeerKAT_240507.pt` | `c0ae0cb16afd0501f00a5955fb6482ed` | `c048af7f2fb5c149fcc888a4a982a947e92bba0ea657dee82229fe1f4264d1de` |
| `animal2vec_large_finetuned_MeerKAT_240507.pt` | `b377ea79700f3bbc98b6154f21545158` | `2846cefa9e1d696a95d91f9ae316456895eb3075eaafc5e81b3d20eb9c4bb712` |

Check both downloads:

```bash
md5sum animal2vec_large_*_MeerKAT_240507.pt
sha256sum animal2vec_large_*_MeerKAT_240507.pt
```

PyTorch checkpoints are pickle-based. Only deserialize artifacts obtained from
a source you trust and whose checksum you verified.

## Converting official Fairseq weights

### What conversion does

`a2v2-convert-checkpoint`:

1. loads the legacy tensor/config container on CPU;
2. resolves the active and, where needed, pretrained architecture;
3. instantiates the corresponding native model;
4. maps original names to native names;
5. checks every mapped shape;
6. rejects missing required tensors;
7. writes a versioned native checkpoint; and
8. prints a JSON conversion report.

Conversion is strict. Do not patch the tool to use `strict=False` merely to
make an unknown checkpoint load: a partially initialized network can emit
plausible-looking but invalid predictions.

### Convert the official pretraining checkpoint

```bash
a2v2-convert-checkpoint \
  /weights/animal2vec_large_pretrained_MeerKAT_240507.pt \
  /weights/animal2vec_large_pretrained_MeerKAT_240507.native.pt \
  --stage pretrain \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml
```

The converted pretraining checkpoint can initialize native fine-tuning. It can
also supply unmasked student embeddings through `InferenceRunner`; see the
[Python inference API](#python-inference-api).

### Convert the official fine-tuning checkpoint

Fine-tuning conversion needs both the active classifier recipe and the
pretraining architecture:

```bash
a2v2-convert-checkpoint \
  /weights/animal2vec_large_finetuned_MeerKAT_240507.pt \
  /weights/animal2vec_large_finetuned_MeerKAT_240507.native.pt \
  --stage finetune \
  --config configs/MeerKAT/finetune_mixup_100.yaml \
  --pretrained-config configs/MeerKAT/a2v_large_pretrain_best.yaml
```

The converted fine-tuning checkpoint can be passed directly to
`a2v2-infer`.

### If the raw file embeds unavailable Python objects

Some official Fairseq checkpoints contain OmegaConf or Fairseq-era config
objects. A clean native environment may correctly refuse to deserialize them
because those modules are absent.

Create a one-time plain export in a separate Python 3.10/Fairseq/OmegaConf
legacy environment:

```python
from pathlib import Path

import torch
from omegaconf import OmegaConf

source = Path(
    "/weights/animal2vec_large_finetuned_MeerKAT_240507.pt"
)
destination = source.with_suffix(".plain.pt")

state = torch.load(source, map_location="cpu", weights_only=False)
plain_cfg = OmegaConf.to_container(state["cfg"], resolve=True)
num_updates = state.get("num_updates")
if num_updates is None and state.get("optimizer_history"):
    num_updates = state["optimizer_history"][-1].get("num_updates", 0)

torch.save(
    {
        "model": state["model"],
        "cfg": plain_cfg,
        "num_updates": int(num_updates or 0),
    },
    destination,
)
```

Move only the resulting plain file into the native environment and run the same
conversion command against it. Fairseq, Hydra, and OmegaConf are not needed
again after this boundary.

The [GPU verification report](docs/gpu-verification-20260723.md) records the
exact isolated legacy environment used for the official exports. That
environment is external to this standalone native source tree.

### Interpret the conversion report

The JSON report includes:

- `stage`;
- `mapped`;
- `renamed`;
- `omitted`; and
- `missing`.

`missing` must be empty. `omitted` is allowed only for legacy state that is
outside the selected native stage and is listed explicitly in the report.
A shape mismatch is a hard error.

Inspect the native checkpoint:

```bash
python - <<'PY'
from a2v2.training import load_checkpoint

path = "/weights/animal2vec_large_finetuned_MeerKAT_240507.native.pt"
checkpoint = load_checkpoint(path)
print("format:", checkpoint["format_version"])
print("stage:", checkpoint["stage"])
print("update:", checkpoint["update"])
print("model tensors:", len(checkpoint["model"]))
print("config sections:", checkpoint["config"].keys())
PY
```

The official converted files verified on A100 had SHA-256:

```text
pretraining native:
e5cb00a82b2aac92b98e38e77ba377b36a6a687b0b1f857ccbbb2810d1a41d08

fine-tuning native:
b95db2dfa6e13ab9e3f02f593bcc461d67664388d9f3a64ada34b492e65e9795
```

### Conversion is not training resume

Converted official checkpoints contain model/config state, but not a native
optimizer, scheduler, scaler, sampler, or rank-RNG state. Therefore:

- converted pretraining weights are valid fine-tuning initialization;
- converted fine-tuning weights are valid inference weights;
- neither converted file can exactly resume the archived Fairseq optimizer
  run; and
- `a2v2-train --resume` will reject a converted inference-only
  checkpoint.

Use `--resume` only with a native training checkpoint written by this trainer.

For mapping details and failure handling, read
[docs/checkpoint-conversion.md](docs/checkpoint-conversion.md).

## Starting pretraining

### Choose a recipe

| Dataset/profile | Recipe |
| --- | --- |
| Published MeerKAT large pretraining | `configs/MeerKAT/a2v_large_pretrain_best.yaml` |
| Published spotted-hyena example | `configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml` |
| Tiny CPU plumbing check | `configs/cpu_smoke_pretraining.yaml` |

The published MeerKAT model uses:

- four distributed workers;
- 16 Transformer layers;
- embedding width 1,024;
- 16 attention heads;
- 12 independently masked student clones;
- an eight-layer context prenet;
- a four-layer convolutional decoder;
- `max_tokens=408000` per worker;
- five-way gradient accumulation;
- 384,230 optimizer updates;
- Adam at `1e-4`;
- 10,000 warmup updates and cosine decay;
- FP16;
- A-weighted waveform mixup; and
- EMA decay from 0.9997 toward 1.0 through update 300,000.

### Start the published four-GPU MeerKAT recipe

Assuming `pretrain.tsv` is under `/datasets/MeerKAT/manifests`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc-per-node=4 \
  "$(command -v a2v2-train)" \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/meerkat-pretrain \
  --device cuda
```

The worker count must equal
`distributed_training.distributed_world_size` in the resolved recipe. The
trainer rejects a mismatch so an accidental one-rank launch cannot silently
change the effective batch.

The exact recipe was verified to execute on four A100-SXM4-40GB devices without
reducing the batch, token budget, clone count, model size, accumulation factor,
or worker count. The worst observed allocator reservation left 2.40 GiB
against PyTorch-visible capacity. This was a two-update fit/resume burn-in, not
a completed 384,230-update run.

### Monitor representation variance

Pretraining update records include `pred_var` for masked student-decoder
outputs and `target_var` for the matched normalized EMA-teacher targets. The
names match Animal2Vec 1.0 logs. Each scalar contains the mean feature-wise
sample standard deviation:

```text
mean_d sqrt(sample_variance(z[:, d]) + 1e-6)
```

For distributed training, the engine reduces vector count, sum, and squared
sum across ranks before it evaluates the formula. It computes one value per
microbatch and averages the values across the microbatches accumulated into an
optimizer attempt. FP16 activations use FP32 diagnostic arithmetic.

A pretraining update JSON record therefore contains fields such as:

```json
{
  "update": 384,
  "loss": 0.417,
  "pred_var": 0.238,
  "target_var": 0.512,
  "gradient_norm": 1.73,
  "amp_scale": 0.03125
}
```

Plot both series throughout pretraining. A sustained move toward zero warns
that the student predictions or teacher representation are collapsing toward
constant feature vectors. Interpret the trajectory with normalized loss,
gradient norm, EMA decay, learning rate, and AMP scale. The terminal
`training_summary` repeats the most recent pair, so a short burn-in exposes the
diagnostic before it reaches `common.log_interval`.

### TensorBoard monitoring

The trainer writes TensorBoard events on rank zero for pretraining and
fine-tuning. A relative `common.tensorboard_logdir` sits below
`checkpoint.save_dir`. The published recipes use `tb`, so the MeerKAT command
above writes:

```text
/checkpoints/meerkat-pretrain/tb
```

Recipes without `tensorboard_logdir` use
`<checkpoint.save_dir>/tensorboard`. Open the dashboard with:

```bash
tensorboard --logdir /checkpoints/meerkat-pretrain/tb
```

The main training tags are:

| Tag | Meaning |
| --- | --- |
| `train/loss` | Globally normalized update loss |
| `train/sample_size` | Global frame-token count for the update |
| `train/gradient_norm` | Gradient norm before clipping |
| `train/learning_rate` | Post-update scheduler value |
| `train/skipped` | One when AMP rejected an optimizer attempt |
| `train/amp_scale` | Current GradScaler scale when FP16 is active |
| `pretrain/pred_var` | Masked student-output dispersion |
| `pretrain/target_var` | Matched EMA-target dispersion |

`run/config` stores the resolved native recipe. `run/parameters` and
`run/trainable_parameters` record model size. A resumed run purges stale events
after the checkpoint step before it appends new measurements.

Fine-tuning validation adds the `validation/<subset>/frame/` and
`validation/<subset>/segmented/` groups. Those groups contain aggregate and
classwise scalars, precision-recall curves, and event-diagnostic histograms.

### Single-GPU diagnostic pretraining

For a functional diagnostic on one GPU, explicitly override the recipe's world
size:

```bash
CUDA_VISIBLE_DEVICES=0 \
a2v2-train \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/meerkat-pretrain-one-gpu \
  --override distributed_training.distributed_world_size=1 \
  --stop-at-update 1 \
  --device cuda
```

This is not paper-equivalent: the world size and therefore effective batch
topology differ. Label it as a diagnostic.

### `--stop-at-update` versus `--max-updates`

These flags have different purposes:

- `--stop-at-update N` stops at an absolute update while preserving the
  configured scheduler horizon. Use it for memory, smoke, and resume burn-ins.
- `--max-updates N` replaces the configured maximum update and therefore
  changes the scheduler horizon. Use it only when intentionally defining a
  different experiment.

Never replace the paper's maximum update with a small number and then describe
the result as an exact-recipe scheduler test.

### CPU smoke pretraining

The tiny CPU recipe validates data, model, loss, optimizer, checkpoint, and CLI
plumbing:

```bash
a2v2-train \
  --config configs/cpu_smoke_pretraining.yaml \
  --override task.data=/path/to/smoke_data/manifests \
  --override checkpoint.save_dir=/checkpoints/cpu-smoke-pretrain \
  --device cpu
```

It is not a quality or paper-performance test.

## Resuming pretraining

Resume a native checkpoint with the same recipe, world size, data membership,
and output directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc-per-node=4 \
  "$(command -v a2v2-train)" \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/meerkat-pretrain \
  --resume /checkpoints/meerkat-pretrain/checkpoint_last.pt \
  --device cuda
```

For a bounded resume diagnostic, add `--stop-at-update` with the desired
absolute terminal update:

```bash
  --resume /checkpoints/meerkat-pretrain/checkpoint_update100.pt \
  --stop-at-update 101
```

Resume rules:

1. Use a native training checkpoint, not a converted official checkpoint.
2. Keep the same `distributed_training.distributed_world_size`.
3. Keep class order, audio rate, model configuration, optimizer, scheduler, and
   accumulation settings unchanged.
4. Keep manifest membership and sample counts unchanged.
5. Do not reseed the process manually after checkpoint restore.
6. Preserve the configured scheduler horizon.
7. Treat a changed output directory as bookkeeping only; document it when
   comparing checkpoints.

The verified trainer restores:

- student and EMA teacher;
- optimizer and parameter groups;
- cosine scheduler;
- GradScaler;
- update, epoch, and delivered-batch position;
- best validation metric;
- sampler epoch and next-consumed batch; and
- Python, NumPy, CPU PyTorch, and every rank's CUDA RNG state.

The final four-rank verification compared uninterrupted update 2 with a branch
resumed from update 1. After normalizing only `checkpoint.save_dir`, every
logical state value was bitwise exact.

## Starting fine-tuning

### Choose a MeerKAT recipe

| Label fraction | Recipe | Configured updates | Training subset |
| ---: | --- | ---: | --- |
| 100% | `configs/MeerKAT/finetune_mixup_100.yaml` | 30,000 | `train_0` |
| 25% | `configs/MeerKAT/finetune_mixup_025.yaml` | 18,000 | `train_0_few_2` |
| 1% | `configs/MeerKAT/finetune_mixup_001.yaml` | 13,000 | `train_0_few_0` |

Use official split membership when reproducing paper numbers. Do not take an
arbitrary percentage of a regenerated training manifest and call it the paper
split.

### Obtain a native pretraining checkpoint

Fine-tuning needs one of:

- a native checkpoint produced by pretraining in this repository; or
- the official pretraining checkpoint after strict native conversion.

Pass it with `--pretrained-checkpoint`. The trainer extracts the student
encoder and the stored pretrained architecture.

### Start four-GPU 100% MeerKAT fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc-per-node=4 \
  "$(command -v a2v2-train)" \
  --config configs/MeerKAT/finetune_mixup_100.yaml \
  --pretrained-checkpoint /checkpoints/meerkat-pretrain/checkpoint_last.pt \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/meerkat-finetune-fold0 \
  --device cuda
```

To initialize from the converted official weights:

```bash
  --pretrained-checkpoint \
  /weights/animal2vec_large_pretrained_MeerKAT_240507.native.pt
```

The 100% recipe:

- averages all 16 shared Transformer layers;
- freezes the Transformer for the first 10,000 updates;
- keeps the local convolutional feature encoder frozen;
- unfreezes the Transformer for the remaining 20,000 updates;
- applies time and channel masking;
- performs source and target mixup;
- uses sigmoid focal loss;
- trains with `max_tokens=426667` and nine-way accumulation; and
- validates and tracks framewise F1 according to the YAML schedule.

### Fine-tune a different fold

The recipe names `train_0` and `valid_0`. For fold 3, for example:

```bash
torchrun --standalone --nproc-per-node=4 \
  "$(command -v a2v2-train)" \
  --config configs/MeerKAT/finetune_mixup_100.yaml \
  --pretrained-checkpoint /checkpoints/meerkat-pretrain/checkpoint_last.pt \
  --override task.data=/datasets/MeerKAT/manifests \
  --override dataset.train_subset=train_3 \
  --override dataset.valid_subset=valid_3 \
  --override checkpoint.save_dir=/checkpoints/meerkat-finetune-fold3 \
  --device cuda
```

Repeat with the exact authoritative membership for all five folds when
performing paper-level evaluation.

### CPU smoke fine-tuning

```bash
a2v2-train \
  --config configs/cpu_smoke_finetuning.yaml \
  --pretrained-checkpoint \
    /checkpoints/cpu-smoke-pretrain/checkpoint_last.pt \
  --override task.data=/path/to/smoke_data/manifests \
  --override checkpoint.save_dir=/checkpoints/cpu-smoke-finetune \
  --device cpu
```

This exercises the classifier, labels, freeze logic, optimizer, validation,
checkpointing, and inference-compatible state on a tiny model.

## Resuming fine-tuning

Use the same active fine-tuning recipe and pass the native fine-tuning training
checkpoint to `--resume`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc-per-node=4 \
  "$(command -v a2v2-train)" \
  --config configs/MeerKAT/finetune_mixup_100.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/meerkat-finetune-fold0 \
  --resume /checkpoints/meerkat-finetune-fold0/checkpoint_last.pt \
  --device cuda
```

`--pretrained-checkpoint` is not required on resume because a native
fine-tuning checkpoint stores both the active fine-tuning config and the
pretrained encoder config, along with the complete live model state.

Keep the same world size and split membership. This is especially important
near the update-10,000 freeze boundary because optimizer membership and
gradient flow must remain consistent.

`checkpoint_best.pt` is selected by the configured validation metric and is
normally the checkpoint used for inference/evaluation. `checkpoint_last.pt` is
the operational resume point. They may represent different updates.

Do not resume training from the converted official fine-tuning checkpoint. It
has no native optimizer/scheduler/scaler/sampler/RNG state. Use that file for
inference.

## Inference

### CLI inference

Inference requires a native fine-tuning checkpoint:

```bash
a2v2-infer \
  /checkpoints/meerkat-finetune-fold0/checkpoint_best.pt \
  /audio/recording.wav \
  /predictions/recording.events.tsv \
  --segment-seconds 10 \
  --device cuda
```

The output is:

```text
label	start_seconds	end_seconds	score
sn	1.235000	1.410000	0.87344211
focal	1.235000	1.410000	0.79234004
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--channel N` | Select one channel from multichannel input; otherwise channels are averaged |
| `--segment-seconds S` | Chunk duration for bounded-memory inference; default 10 |
| `--threshold P` | Override the checkpoint recipe's event probability threshold |
| `--method avg|max` | Override event fusion method |
| `--fusion-window-seconds S` | Override the smoothing/fusion window |
| `--audition` | Write an Adobe Audition marker CSV instead of the native event TSV |
| `--device cpu|cuda|cuda:N` | Inference device |

The runner:

1. loads audio through SoundFile;
2. selects or averages channels;
3. resamples to the checkpoint's training sample rate;
4. normalizes each chunk;
5. produces frame logits and Transformer layer embeddings;
6. converts logits to probabilities;
7. trims frontend frames beyond a partial final chunk;
8. assigns convolution-aware timestamps; and
9. fuses frame probabilities into bounded event intervals.

No sidecar YAML is required because a native fine-tuning checkpoint stores both
active and pretrained configurations.

The `a2v2-infer` command exports frame events and requires a frame-classifier
checkpoint. Use the Python API below for pretraining embeddings, CLS outputs,
or class-agnostic event voting.

### Adobe Audition marker CSV

Researchers can import A2V2 event predictions as Adobe Audition range markers:

```bash
a2v2-infer \
  /checkpoints/meerkat-finetune-fold0/checkpoint_best.pt \
  /audio/recording.wav \
  /predictions/recording.csv \
  --audition \
  --segment-seconds 10 \
  --device cuda
```

The `--audition` flag reproduces the marker table layout written by the
official Animal2Vec 1.0 inference script:

```text
Name	Start	Duration	Time Format	Type	Description
sn	0:00:01.235000	0:00:00.175000	decimal	Cue	0.873
focal	0:00:01.235000	0:00:00.175000	decimal	Cue	0.792
```

The file uses a `.csv` extension because that is the format Adobe Audition
expects for marker import, but its six fields are separated by tabs to match
the legacy Animal2Vec output. `Start` is the event onset.
`Duration = end_seconds - start_seconds`, so Audition reconstructs the offset
as `Start + Duration`. `Description` contains the model score rounded to three
decimal places.

This flag changes file serialization. It writes the same chronologically
ordered A2V2 events that the native TSV would contain, including an explicit
`focal` marker when that output channel crosses the event threshold. It does
not apply the legacy script's additional competing-label suppression,
focal/non-focal renaming, or secondary-prediction policy.

### Inference with converted official weights

```bash
a2v2-infer \
  /weights/animal2vec_large_finetuned_MeerKAT_240507.native.pt \
  /audio/recording.wav \
  /predictions/recording.events.tsv \
  --segment-seconds 10 \
  --device cuda
```

The converted official model passed strict native FP32 and FP16 inference on an
A100. For a 10.25-second test recording, probabilities had shape `[2048, 12]`,
embeddings had shape `[2048, 1024]`, and the final timestamp remained inside
the real recording.

### Python inference API

Use the API for frame probabilities, embeddings, timestamps, or per-segment
CLS outputs. Pass a native checkpoint from a trusted source; checkpoint loading
uses Python's pickle-backed PyTorch format.

```python
from pathlib import Path

import torch

from a2v2.data import load_audio
from a2v2.workflows import InferenceRunner

checkpoint = Path(
    "/weights/animal2vec_large_finetuned_MeerKAT_240507.native.pt"
)
audio_path = Path("/audio/recording.wav")

waveform, sample_rate = load_audio(audio_path)
runner = InferenceRunner(checkpoint, device=torch.device("cuda:0"))
result = runner.run_tensor(
    waveform,
    sample_rate,
    segment_seconds=10.0,
)

print("probabilities:", result.probabilities.shape)
print("embeddings:", result.embeddings.shape)
print("timestamps:", result.timestamps.shape)
for event in result.events:
    label = runner.labels[event.label_index]
    print(
        label,
        event.start_seconds,
        event.end_seconds,
        event.score,
    )

runner.write_events("/predictions/recording.events.tsv", result.events)

# Write the same events as Adobe Audition range markers.
runner.write_events(
    "/predictions/recording.csv",
    result.events,
    audition=True,
)
```

For a frame classifier, `result.probabilities`, `result.embeddings`, and
`result.timestamps` are CPU tensors aligned along their first dimension.
`result.events` is an immutable tuple of event intervals. The defaults preserve
the class-specific inference behavior above.

#### Pretraining embeddings and optional CLS outputs

```python
runner = InferenceRunner(
    "/checkpoints/pretrain/checkpoint_last.pt",
    device=torch.device("cuda:0"),
    return_cls=True,  # default False; has an effect only if CLS is available
)
result = runner.run_tensor(waveform, sample_rate, segment_seconds=10.0)
print(result.embeddings.shape, result.timestamps.shape)
assert result.probabilities is None and result.events is None
assert result.cls_predictions is None  # pretraining has no trained classifier
if result.cls_embeddings is not None:
    print(result.cls_embeddings.shape)
```

Pretraining inference uses the unmasked **student encoder**, without the EMA
teacher, decoder, or pretraining regression loss. Frame and CLS embeddings use
the arithmetic mean of the last `average_top_k_layers` Transformer outputs.
Frame embeddings exclude the prepended CLS token and retain convolution-aware
timestamps in seconds.

All available arrays are CPU tensors. Let `F` be the total retained frames,
`N` the number of audio segments, `D` the embedding dimension, and `C` the
checkpoint's class count:

| Checkpoint | `probabilities` / `events` | `cls_embeddings` | `cls_predictions` |
| --- | --- | --- | --- |
| Pretraining | `None` / `None` | `[N, D]` if CLS exists and `return_cls=True`; otherwise `None` | `None` |
| Fine-tuned frame classifier | `[F, C]` / event tuple | `[N, D]` if CLS exists and `return_cls=True`; otherwise `None` | `None` |
| Fine-tuned CLS classifier | `None` / `None` | `[N, D]`; enables `return_cls` even if passed as `False` | `[N, C]` class probabilities |

Each checkpoint type returns `embeddings` with shape `[F, D]` and `timestamps`
with shape `[F]`. A CLS row describes one input segment, including a short final
segment, in chronological segment order. Frame timestamps do not index CLS
rows. Changing `segment_seconds` changes the context represented by each CLS
row. For a CLS classifier, use `result.cls_predictions >= threshold` to obtain
segment-level decisions; these predictions do not define frame event boundaries.

#### Class-agnostic event detection

```python
runner = InferenceRunner(
    "/checkpoints/finetune/checkpoint_best.pt",
    device=torch.device("cuda:0"),
    event_detection=True,
    event_vote="max",  # default; alternatives: "mean", "median"
    return_cls=True,
)
result = runner.run_tensor(waveform, sample_rate, threshold=0.5)
if result.events is not None:
    runner.write_events("/predictions/events.tsv", result.events)
```

Set `event_detection=True` to reduce sigmoid probabilities across the class
axis before thresholding and temporal event fusion. `max` selects the largest
probability; `mean` averages all class probabilities; `median` selects the
middle value, averaging the two central values for an even class count.
This is probability aggregation, not voting on already-thresholded labels.

A frame classifier then returns `[F, 1]` probabilities and events with
`label_index=0`. `runner.labels` is `("event",)`, and both TSV and Audition
exports use the label `event`. This can produce multiple event intervals in
a recording. The `event_vote` option reduces **classes**; the existing
`event_method="avg"|"max"` option controls **temporal** fusion.

For a CLS classifier, the same reduction returns `[N, 1]` in
`cls_predictions`; `probabilities` and `events` remain `None`. Pretraining
checkpoints remain embedding-only even with event detection enabled.
The reduction includes all output classes, including auxiliary classes such
as `focal`. It does not add the ability to recognize sounds outside the
model's learned classes. Tune the threshold for your corpus and voting mode.

### Empty and very short recordings

An empty input returns zero-row tensors for available outputs. Unavailable or
disabled outputs remain `None`; a frame classifier returns `events=()`.
`write_events` rejects `events=None` rather than writing a misleading empty
event file. A short final segment does not introduce frame timestamps beyond
the recording. CPU and GPU tests cover these boundaries.

## Evaluation

### Frame metrics

Fine-tuning validation computes normalized loss and framewise:

- micro precision;
- micro recall;
- micro F1;
- accuracy; and
- micro average precision.

The validation cadence and best-checkpoint metric come from the recipe. The
stdout JSON and TensorBoard event file receive the same aggregate values.

### Event metrics

`a2v2/workflows.py` reproduces the archived event algorithm
without importing the analysis stack into the native runtime. It includes:

- average and maximum sliding pooling;
- interval construction;
- strict positive-overlap and strict `IoU > threshold` matching;
- unmatched-target and unmatched-prediction handling;
- split and merger markers;
- interval mean scores;
- per-class AP;
- macro AP over every legacy label column, including `focal`;
- micro AP over every segment-class decision;
- fixed-threshold segmented precision, recall, F1, and accuracy; and
- best-unique-threshold focal precision, recall, and F1.

Every fine-tuning validation invokes this matcher, including scheduled training
validation and `scripts/evaluate_finetuning_checkpoint.py`. The standalone
script writes the aggregate values to its JSON report and puts TensorBoard
events in a `tensorboard` directory beside that report unless
`--tensorboard-dir` selects another location.

The matcher preserves two unusual archived conventions required for result
comparison. Multi-frame intervals use an inclusive endpoint when constructed
but enter half-open overlap arithmetic, and a pair counts only when
`IoU > iou_threshold`. A previous native aggregator discarded all rows with no
positive target. That rule also discarded genuine false-positive segments.
The corrected aggregator retains the complete fixed-size arrays passed to the
legacy classification and AP routines, including their zero padding.

TensorBoard records:

- framewise and segmented micro/classwise precision-recall curves;
- classwise frame and segmented AP;
- segmented IoU histograms per label; and
- split and merger-count histograms per label.

Tests cover perfect matches, misses, false positives, strict boundaries,
splits, mergers, overlapping classes, pooling shifts, and all-negative clips.
The pure-PyTorch average-precision implementation also matches scikit-learn's
archived calculation to floating-point precision on tied-score fixtures.

### Paper-level evaluation caution

Algorithm parity does not replace data/split parity. A paper-level metric
requires:

- authoritative fold membership;
- the same label fractions;
- the same validation/test recordings;
- the same event convention;
- completed full training; and
- the same fold aggregation and reporting convention.

Those requirements were not all available during the 2026-07-23 verification.

## Native checkpoint semantics

### Training checkpoint contents

A native training checkpoint is a validated plain Python/tensor container with:

| Field | Contents |
| --- | --- |
| `format_version` | Native schema version |
| `stage` | `pretrain` or `finetune` |
| `config` | Resolved active config and, for fine-tuning, pretrained config |
| `model` | Student/teacher or fine-tuning model tensors |
| `optimizer` | Adam state and parameter groups |
| `scheduler` | Cosine scheduler state |
| `scaler` | AMP GradScaler state when FP16 is active |
| `update` | Successful optimizer update count |
| `epoch` | Training epoch counter |
| `batch_in_epoch` | Delivered batch position |
| `best_metric` | Current best validation metric |
| `sampler_state` | Sampler epoch and next consumed batch |
| `rng_state` | Python, NumPy, CPU PyTorch, and rank CUDA RNG states |

### Checkpoint files

Depending on recipe intervals, the trainer writes:

- `checkpoint_last.pt`;
- `checkpoint_best.pt`;
- `checkpoint_<update>.pt`; and
- `checkpoint_epoch_<epoch>.pt`.

Rank 0 writes the shared file after collecting each rank's RNG state.
Distributed CUDA training keeps tensor collectives on NCCL but sends the
serialized CPU RNG payloads through a separate Gloo process group. Rank 0
requires exactly one byte payload from every rank before it writes; a partial
gather is an error, not a shortened `by_rank` list. The public training
boundary destroys process groups that it initialized even when training raises
an exception, while preserving a process group supplied by a caller.

### Overflow attempts are not updates

With AMP, a non-finite gradient attempt reduces the scale but does not advance:

- the optimizer;
- scheduler;
- update counter;
- EMA teacher;
- sampler's committed position;
- validation cadence; or
- update-based checkpoint cadence.

This behavior matters for exact resume, especially at the large pretraining
recipe's initial scale.

### World-size contract

Resume with the same world size. Native checkpoints store one RNG payload per
rank and the distributed sampler partitions complete batches according to the
world size. Changing it is a different experiment and is not an exact resume.

## Configuration and overrides

### Checked-in recipes

The repository contains eight YAML files:

```text
configs/MeerKAT/a2v_large_pretrain_best.yaml
configs/MeerKAT/finetune_mixup_100.yaml
configs/MeerKAT/finetune_mixup_025.yaml
configs/MeerKAT/finetune_mixup_001.yaml
configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml
configs/hyenas/finetune_mixup_100.yaml
configs/cpu_smoke_pretraining.yaml
configs/cpu_smoke_finetuning.yaml
```

The native loader safely parses the subset of legacy YAML used by these
recipes. It does not invoke Hydra interpolation or evaluate arbitrary Python.

### Dotted overrides

Repeat `--override` for each replacement:

```bash
a2v2-train \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/datasets/MeerKAT/manifests \
  --override checkpoint.save_dir=/checkpoints/experiment-a \
  --override common.seed=7 \
  --device cuda
```

Overrides are typed according to the config field. Invalid fields or values are
errors.

### Preserve resolved configuration

For any experiment intended for comparison:

1. hash the source YAML;
2. record every override;
3. retain the checkpoint's serialized resolved config;
4. record the package and PyTorch lock;
5. record world size and GPU UUIDs; and
6. preserve manifest membership hashes.

Do not rely on a source YAML alone after applying command-line overrides or
loading checkpoint-derived architecture.

## Testing and GPU verification

### Full native CPU suite

```bash
python -m pytest -q
```

The clean authority environment passed:

```text
162 passed, 14 skipped
```

### One-GPU suite

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m pytest -q -m gpu \
  tests/gpu/test_cuda_components.py \
  tests/gpu/test_cuda_inference.py \
  tests/gpu/test_cuda_training.py
```

The authority run passed 14 tests covering:

- Sinc FP32 CPU/CUDA forward and gradients;
- Sinc FP16 forward/backward/update;
- frontend FP32 parity;
- ALiBi FP32/FP16 behavior;
- decoder autocast dtype;
- exact mixup/mask RNG replay;
- tiny pretraining FP32 CPU/CUDA comparison;
- AMP checkpoint continuation;
- CUDA map-location RNG restore;
- fine-tuning freeze/unfreeze;
- overflow skip semantics;
- chunked inference; and
- empty inference.

### NCCL probe

Two ranks:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc-per-node=2 \
  tests/gpu/nccl_probe.py \
  --output-dir /tmp/a2v2-nccl-2
```

Four ranks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  tests/gpu/nccl_probe.py \
  --output-dir /tmp/a2v2-nccl-4
```

The probes require correct NCCL all-reduce, broadcast, all-gather, and barriers,
plus Gloo-backed rank RNG collection and exact rank-local RNG restoration. Each
rank report records `rng_gather_backend: "gloo"`.

### Durable distributed resume

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  tests/gpu/nccl_resume.py \
  --output-dir /tmp/a2v2-resume-4
```

The probe builds uninterrupted and resumed branches and requires exact state
on every rank.

### Verification results

The A100 verification decision is:

| Gate | Status |
| --- | --- |
| A: package integrity | Pass |
| B: CUDA correctness | Pass |
| C: official checkpoint parity | Pass |
| D: gradient and optimizer-update parity | Pass |
| E: distributed reliability and exact recipe fit | Pass |
| F: data and evaluation parity | Partial / blocked on publication artifacts |
| G: full paper reproduction | Blocked / not run |

Official same-GPU comparisons passed:

- 2,076/2,076 pretraining comparisons in FP32;
- 2,076/2,076 pretraining comparisons in FP16;
- 1,056/1,056 fine-tuning comparisons in FP32; and
- 1,056/1,056 fine-tuning comparisons in FP16.

The production-topology pretraining and 100% fine-tuning stop/continuous/resume
burn-ins were recursively bitwise exact after normalizing only the output
directory. Read [docs/gpu-verification-20260723.md](docs/gpu-verification-20260723.md)
for tolerances, per-rank memory, hashes, first divergences, rejected DDP
experiments, and the precise scope of every claim.

## Documentation guide

| Document | Read it when… |
| --- | --- |
| [Code guide](docs/code-guide.md) | You want a reading order, data-flow trace, or old-to-new import map |
| [Checkpoint conversion](docs/checkpoint-conversion.md) | You are converting official or custom Fairseq weights |
| [Paper reproduction](docs/reproducing-paper.md) | You are preparing manifests and full training/evaluation runs |
| [Final A100 report](docs/gpu-verification-20260723.md) | You need current evidence, outcomes, defects, hashes, memory, commands, and blockers |
| [Refactor design](docs/refactoring-design.md) | You need the rationale and compatibility boundary for the flat package |
| [Documentation design](docs/documentation-design.md) | You need the scope and accuracy rules for source explanations |
| [Documentation plan](docs/documentation-plan.md) | You need the file-by-file execution and verification checklist |

The A100 report describes the pre-refactor file paths used during verification.
Its opening note links those paths to the current flat modules.

## Troubleshooting

### “CUDA was requested but is not available”

Confirm that:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

uses the intended virtual environment and PyTorch wheel. A driver's reported
maximum CUDA compatibility is not the same as the CUDA runtime embedded in the
PyTorch wheel.

### World-size mismatch

If the trainer reports:

```text
distributed_training.distributed_world_size=4 but the launch created 1 process
```

either launch four processes with `torchrun --nproc-per-node=4` or explicitly
override the recipe to one worker for a diagnostic. Do not override it silently
for a paper run.

### Out of memory

First verify:

- no unrelated process occupies the GPU;
- the intended four GPUs are visible;
- the exact config and overrides;
- PyTorch/allocator versions; and
- whether the failure is allocated memory or fragmentation.

The exact published native pretraining recipe fit 40 GB A100s during the
verified burn-in. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as in
the authority run. Do not silently reduce tokens, accumulation, clone count, or
model size. If a reduction is required on different hardware, document it as a
new experiment.

### Official checkpoint cannot import OmegaConf/Fairseq classes

This is expected for some raw files in a clean runtime. Verify the official
checksum, use the isolated legacy environment to export a plain
`{"model", "cfg", "num_updates"}` dictionary, and convert that plain file.
Do not add Fairseq to the native environment.

### Conversion reports missing tensors or a shape mismatch

Check that:

- pretraining weights use the pretraining YAML;
- fine-tuning weights use both the 100% fine-tuning and matching pretraining
  YAMLs;
- the file was not truncated;
- the checksum matches the publisher record; and
- the checkpoint is the expected 2024 MeerKAT model.

Do not bypass the error with non-strict loading.

### Fine-tuning requires a pretrained checkpoint

Pass:

```text
--pretrained-checkpoint /path/to/native-pretraining-checkpoint.pt
```

or set `model.w2v_path` to a native pretraining checkpoint. A raw Fairseq file
must be converted first.

### Converted checkpoint cannot resume optimization

This is by design. Converted files do not contain native optimizer, scheduler,
scaler, sampler, or rank RNG state. Start fine-tuning from converted
pretraining weights, run inference from converted fine-tuning weights, or
resume only from a native training checkpoint.

### Manifest contains no usable examples

Check:

- the first line is the correct audio root;
- later rows contain a literal tab;
- sample counts match actual audio;
- sample rate matches `task.sample_rate`;
- paths are relative to the root;
- labeled rows resolve to HDF5 files; and
- `min_sample_size`, `max_sample_size`, and `min_label_size` do not filter every
  row.

Run the two audit scripts before debugging the model.

### Timestamps extend beyond the recording

Use this native runner, not an earlier checkout. The verified implementation
trims partial-final-segment frames using actual recording duration and
convolution-aware timestamps. If a custom caller concatenates chunks itself,
use `InferenceRunner` rather than reimplementing timestamp arithmetic.

### DDP warns that no unused parameters were found

The verified pretraining path intentionally uses unused-parameter traversal
with one large explicit bucket because that combination made reduction order
identical before and after restart. The warning indicates overhead, not a
failed correctness condition. Do not switch to static-graph DDP without
repeating all-rank optimizer and resume comparisons; the static-graph
production probe was rejected.

## Reproducing the paper

The checked-in recipes preserve the published model sizes and principal
training settings, but source code alone cannot establish paper reproduction.

### Practical one-fold run on eight A100s

The deployment driver performs one pretraining run and one fold-0, 100%-label
fine-tuning run on all eight selected GPUs, then validates the best checkpoint
and prints a final JSON report:

```bash
bash scripts/reproduce_meerkat_paper.sh \
  /datasets/MeerKAT/manifests \
  /experiments/a2v2-meerkat-fold0
```

Before a long launch, the driver checks the selected Python runtime, exactly
eight visible A100s, at least 39 GiB total and 38 GiB currently free on every
GPU, and at least 64 GiB free on the output filesystem. It then runs an
eight-rank NCCL/Gloo checkpoint probe. A fresh pretraining directory runs one
real production update, validates the resulting resumable eight-rank
checkpoint, and starts the full job with `--resume`. Existing checkpoints are
validated before resume. Phase headers and command output append to the stage
logs, so rerunning the driver does not erase the evidence from an earlier
failure.

This is explicitly an approximate, same-order reproduction. It is not the
published four-rank topology and does not include all five folds or the
Xeno-canto/NIPS4Bplus experiment. The driver records this limitation in
`final-evaluation/final-evaluation-report.json`.

Pretraining retains the memory-verified 408,000 tokens per rank and uses three
accumulated microbatches:

```text
408,000 × 8 ranks × 3 = 9,792,000 effective tokens/update
```

Fine-tuning uses 960,000 tokens per rank and two accumulated microbatches:

```text
960,000 × 8 ranks × 2 = 15,360,000 effective tokens/update
```

The latter is essentially equal to the paper recipe's
`426,667 × 4 × 9 = 15,360,012`; it shifts work from accumulation into larger
per-device microbatches. The pretraining batch is 20% larger because its
verified per-rank memory peak leaves too little headroom to increase
`max_tokens`.

Set `A2V2_FINETUNE_MAX_TOKENS` if the deployed manifest's recording-length
distribution needs a smaller or larger memory profile. The corresponding
pretraining, accumulation, evaluation-batch, worker, interpreter, launcher,
and GPU-list controls are documented in
[docs/reproducing-paper.md](docs/reproducing-paper.md). A rerun resumes from
each stage's `checkpoint_last.pt`; keep all topology and batch controls
unchanged while resuming.

### Work already completed

- Clean native dependency audit.
- Full CPU regression suite.
- One-A100 FP32 and FP16 component/training/inference tests.
- Strict conversion of real official pretraining and fine-tuning checkpoints.
- Same-GPU archived/native layer and inference comparison.
- Gradient and one-optimizer-update comparison.
- NCCL on two and four ranks.
- Exact durable distributed resume.
- Exact published-topology memory and two-update resume burn-ins.
- Exhaustive released MeerKAT structural audit.
- Archived/native synthetic event-evaluator equality.

### Work or artifacts still required

- Authoritative publisher split manifests for all five folds and the reported
  1%, 25%, and 100% fractions.
- Resolution of the 66,397-versus-66,398 labelled-recording discrepancy.
- The exact publisher supplement/protocol artifacts.
- The exact Xeno-canto subset, exclusions, hashes, and preprocessing used for
  the bird-transfer experiment.
- Exact NIPS4Bplus sequence and framewise protocol inputs.
- The full 384,230-update MeerKAT pretraining run.
- All 15 authoritative five-fold/fraction fine-tuning runs.
- Full-manifest event outputs and paper aggregation.

Only after Gate G described in the
[GPU verification report](docs/gpu-verification-20260723.md) passes should a
report say that the paper was reproduced.

For the exact procedure, target values, environment capture, data provenance,
and acceptance gates, read:

1. [docs/reproducing-paper.md](docs/reproducing-paper.md); and
2. [docs/gpu-verification-20260723.md](docs/gpu-verification-20260723.md).

## Citation and license

If you use this rewrite, cite the original Animal2Vec work and follow the code,
weights, and dataset licenses on their respective records.

Published article:

```bibtex
@article{schaeferzimmermann2026animal2vec,
  title = {animal2vec and MeerKAT: A self-supervised transformer for
           rare-event raw audio input and a large-scale reference dataset
           for bioacoustics},
  author = {Schäfer-Zimmermann, Julian C. and Demartsev, Vlad and Averly,
            Baptiste and Dhanjal-Adams, Kiran and Duteil, Mathieu and Gall,
            Gabriella and Faiß, Marius and Johnson-Ulrich, Lily and Stowell,
            Dan and Manser, Marta B. and Roch, Marie A. and
            Strandburg-Peshkin, Ariana},
  journal = {Methods in Ecology and Evolution},
  year = {2026},
  doi = {10.1111/2041-210X.70218}
}
```

Open preprint:

```bibtex
@article{schaeferzimmermann2024animal2vec,
  title = {animal2vec and MeerKAT: A self-supervised transformer for
           rare-event raw audio input and a large-scale reference dataset
           for bioacoustics},
  author = {Schäfer-Zimmermann, Julian C. and Demartsev, Vlad and Averly,
            Baptiste and Dhanjal-Adams, Kiran and Duteil, Mathieu and Gall,
            Gabriella and Faiß, Marius and Johnson-Ulrich, Lily and Stowell,
            Dan and Manser, Marta B. and Roch, Marie A. and
            Strandburg-Peshkin, Ariana},
  journal = {arXiv preprint arXiv:2406.01253},
  year = {2024},
  url = {https://arxiv.org/abs/2406.01253}
}
```

MeerKAT dataset:

```text
Schäfer-Zimmermann, J. C. et al. (2024).
MeerKAT: Meerkat Kalahari Audio Transcripts.
Edmond. https://doi.org/10.17617/3.0J0DYB
```

Official weights:

```text
Pretrained models for animal2vec and MeerKAT.
Edmond. https://doi.org/10.17617/3.ETPUKU
```

This repository's source code is licensed under the
[MIT License](LICENSE). The MeerKAT dataset and official weights are published
under separate terms on Edmond; consult and comply with those records.
