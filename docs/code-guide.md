# A2V2 code guide

The current A2V2 runtime contains the complete Animal2Vec 1.0 reproduction
baseline in five implementation files under `a2v2/`, one SLURM contract
module, and one public API file. The baseline branches form the stable control
against which new A2V2 methods can be compared. Each implementation file uses large section
banners to preserve the navigation benefits of smaller modules. Search for a
banner title or use an editor's symbol outline to move within a file.

The package forms this one-way dependency chain:

```text
a2v2.config
      ↓
 a2v2.data
      ↓
 a2v2.model
      ↓
a2v2.training
      ↓
a2v2.workflows

a2v2.slurm supplies pure launch, topology, lock, and marker contracts to
a2v2.workflows and the shell launchers.
```

`a2v2.__init__` re-exports the common configuration, model, and inference
types. It does not own research logic.

## Comment and notation conventions

Result-sensitive blocks use paired comments:

```python
# Mathematics: equation, shape transformation, probability law, or state rule.
# Interpretation: purpose in the Animal2Vec reproduction.
```

The order lets a reader verify the operation before reading its motivation.
Comments name axes and units wherever a mismatch could change a result.

The source uses these symbols:

| Symbol | Meaning |
| --- | --- |
| `B` | batch size |
| `S` | waveform samples per item |
| `T` | convolution or transformer frames |
| `C` | target classes, or convolution channels when stated |
| `D` | transformer embedding dimension |
| `H` | attention heads |
| `d_h` | per-head dimension, `D/H` |
| `M` | masked frame count |
| `W` | distributed world size |
| `K` | layer count, clone count, or accumulation count in local context |

Waveform time uses samples until a function divides by sample rate. Event
intervals use seconds. Padding masks use `True` for positions that contain no
real input. Time masks use `True` for real frames hidden from the student.

## Suggested reading path

Researchers who want the shortest complete path should read:

1. The module introduction and dataclasses in `config.py`.
2. `conv_output_length`, `rasterize_labels`, `collate_audio`, and both samplers
   in `data.py`.
3. The section banners in `model.py`, starting at the Sinc filterbank and
   continuing through attention, masking, `AudioEncoder`, and the two task
   models.
4. `FairseqCompatibleAdam`, `CosineUpdateScheduler`, and
   `TrainingEngine.step` in `training.py`.
5. `run_training`, `InferenceRunner`, and `convert_checkpoint` in
   `workflows.py`.

This order follows runtime data flow. A reader interested only in architecture
can spend most of their time in `model.py`. A reader auditing resume behavior
should start with `training.py` and then inspect the training section in
`workflows.py`.

## `a2v2/config.py`

This file turns published Fairseq-style YAML into immutable Python dataclasses.
It rejects unknown fields so a setting cannot disappear during framework
removal.

### Schema sections

The dataclasses mirror top-level recipe groups:

| Dataclass | Recipe responsibility |
| --- | --- |
| `CommonConfig` | seed, AMP, logging |
| `CheckpointConfig` | save cadence and best metric |
| `TaskConfig` | sample rate, labels, convolution geometry, data path |
| `DatasetConfig` | workers, token budget, subsets, validation cadence |
| `DistributedConfig` | requested and launched world size |
| `CriterionConfig` | focal loss and event scoring |
| `OptimizationConfig` | update accumulation, clipping, learning rate |
| `OptimizerConfig` | Adam coefficients and weight decay |
| `SchedulerConfig` | warmup and cosine floor |
| `DecoderConfig` | reconstruction decoder |
| `AudioModelConfig` | local frontend, masking, ALiBi, decoder |
| `ModelConfig` | transformer, EMA, loss, mixup, fine-tuning |
| `Animal2VecConfig` | complete validated runtime configuration |

`parse_conv_feature_layers` accepts literal lists and the restricted list
addition or repetition syntax used by official recipes. It never evaluates
arbitrary Python.

`config_from_dict` contains the full legacy-to-native translation. Read it when
adding a recipe field. The function checks supported keys, resolves old
composite optimizer wrappers, recovers model defaults that Hydra once supplied,
and infers pretraining versus fine-tuning.

`config_to_dict` converts paths and tuples into portable containers for
checkpoints. `config_from_serialized_dict` performs the inverse operation.

`ModelConfig.checkpoint_activations` defaults to false. The field changes
autograd execution and creates no parameter or buffer, so serialized configs
record it while model state dictionaries remain unchanged. Older serialized
configs that omit it restore the false default.

## `a2v2/data.py`

This file covers the path from an audio filename to a device-ready batch.

### Audio geometry and I/O

`conv_output_length` applies the convolution recurrence

```text
L_next = floor((L + total_padding - kernel) / stride) + 1
```

one layer at a time. Model features, padding masks, labels, and test fixtures
all call this function.

`feature_timestamps` propagates receptive-field center and sample jump through
the convolution stack. It returns center times in seconds.

`load_audio` uses SoundFile and returns `[channels, samples]` float32 data.
`normalize_waveform` reproduces Fairseq's affine-free layer normalization over
the sample axis.

`resample_waveform` implements windowed-sinc resampling in PyTorch. It maps each
output index to a continuous input coordinate, builds a low-pass sinc kernel
under a raised-cosine window, renormalizes boundary taps, and processes output
positions in chunks.

### Labels, manifests, and collation

`load_label_events` reads sparse HDF5 intervals. `rasterize_labels` maps them to
the zero-origin frame grid used by the published dataset code. That grid differs
from receptive-field center timestamps; the source explains and preserves the
difference.

`AudioDataset` reads the root-plus-rows TSV format, resolves label paths, checks
sample rate, averages channels, normalizes waveforms, and creates targets on
demand.

`collate_audio` chooses one batch sample length, applies deterministic random
crops, creates padding masks, and maps each sample crop offset to its target
frame offset.

### Token batch samplers

`TokenBatchSampler` sorts recordings by length and enforces

```text
batch_size × maximum_recording_length ≤ max_tokens
```

It stores the epoch and first undelivered batch index. `DistributedBatchSampler`
assigns one complete token batch to each rank per synchronized round. Both
classes keep their checkpoint state independent of DataLoader prefetch.

## `a2v2/model.py`

This file contains the complete differentiable a2v2. Its section order
matches a bottom-up architecture reading.

### Differentiable primitives

`GradMultiply` leaves forward values unchanged and multiplies only the backward
gradient. `SamePad` trims the extra frame created by symmetric padding with an
even kernel. `PSwish` applies

```text
y[c] = alpha[c] x[c] sigmoid(beta[c] x[c]).
```

`DropPath` draws one Bernoulli variable per sample and residual path.

The FP32 normalization classes perform reduction arithmetic in float32 under
AMP and return the result in the input dtype.

### Sinc filterbank

`SincConv1d` parameterizes each first-layer filter through lower and upper
cutoff frequencies. It constructs a symmetric band-pass response as the
difference between two low-pass sinc functions, applies a Hamming window, and
normalizes by bandwidth.

The explicit reflection-padding branch uses slices and flips. CUDA's generic
reflection-padding backward kernel uses nondeterministic atomics on the
verified environment, so this implementation preserves the equivalent
deterministic operation.

### Attention, RoPE and ALiBi

`alibi_slopes` creates the geometric head slopes from the ALiBi paper.
`alibi_bias` builds

```text
B[h,i,j] = -slope[h] |i-j|.
```

`MultiheadAttention` packs Q, K, and V in one projection to retain checkpoint
layout. It adds ALiBi before masking, assigns negative infinity to padded keys,
normalizes attention weights in float32, and returns to the surrounding dtype.

#### RoPE and ALiBi

`resolve_position_encoding` preserves the legacy rule: `legacy` reads
`audio.use_alibi_encoder`. Explicit `alibi`, `rope`, and `none` bypass
that old flag. RoPE rotates packed Q and K after projection. It calculates
sine and cosine values in FP32 from integer position IDs and needs an even
head dimension.

The masked student gathers frame position IDs with the same `ids_keep` tensor
that gathers features. A CLS-enabled encoder assigns ID zero to CLS and shifts
frame IDs by one. The new ALiBi CLS path gives the CLS row and column zero bias
while frame pairs retain their temporal distance. The CLS-disabled ALiBi path
keeps the legacy calculation.

#### Strict Flash and SDPA

`resolve_attention_backend` keeps `legacy` manual attention for legacy
position settings. An explicit RoPE or no-position model with a legacy backend
resolves to `sdpa`. The `sdpa` branch calls
`scaled_dot_product_attention` and lets PyTorch choose Flash,
memory-efficient, or math kernels. The `flash` branch enables the Flash SDPA
backend alone and raises an actionable setup error when the kernel rejects the
device, dtype, shape, or mask. A test or benchmark report may identify a run
as Flash-backed after strict mode succeeds. ALiBi stays on manual attention
because its dense learned bias does not fit the selected Flash contract.

#### CLS pretraining and sequence fine-tuning

`AudioEncoder` adds one learned `[1, 1, D]` token when
`use_cls_token=true`. It prepends the token after convolutional positional
encoding. Frame masks and decoder restoration retain their frame axis; context
padding adds one valid CLS column.

The pretraining teacher encodes full frames plus CLS. The student encodes CLS
plus retained frames. A conditional predictor compares the final student CLS
state with the normalized EMA-teacher CLS target:

```text
loss = masked_frame_loss + cls_loss_weight * cls_loss
sample_size = masked_frame_count + cls_count
```

The default weight is 1.0. Frame target normalization excludes CLS, so the
added token does not alter frame time statistics. A CLS-disabled model creates
no token or predictor and retains legacy state keys.

`classification_head=cls` averages selected top-layer CLS states and emits
`[B, C]` logits. Fine-tuning reduces frame labels to recording occurrence
before mixup and reports `sample_size=B`. Event fusion rejects sequence
logits because they contain no timing axis. `a2v2-evaluate-sequence` restores
a native CLS fine-tuning checkpoint and reports multilabel precision, recall,
F1, accuracy, and average precision.

#### Packed GEGLU and DeepScaleLM

`ffn_type=geglu` replaces each MLP input projection with one packed
`2 * int(D * mlp_ratio)` projection. The block splits value and gate halves,
applies GELU to the gate, multiplies both halves, and projects back to `D`.
GEGLU changes parameter shapes and requires a matching checkpoint.

`initialization=deepscale_lm` uses total Transformer depth
`N = prenet_depth + depth`, with `N >= 2`:

```text
lambda = sqrt(1 - 2/N)
beta = sqrt(2/N)
residual(x, branch) = lambda * x + beta * branch
```

The initializer handles Q, K, V, attention output, and FFN roles. It leaves the
audio frontend under its acoustic rules. Checkpoint loading overwrites
initialization, while strict tensor keys and shapes enforce architecture
compatibility.

### Masking and decoder

`compute_mask_indices` reproduces the Fairseq Data2Vec span sampler, including
stochastic span-count rounding, overlap modes, padding-aware lengths,
equalization across a batch, and mask dropout. It uses NumPy randomness because
the archived algorithm did.

`make_mask_info` sorts unmasked positions before masked positions and records
the inverse permutation. The student processes the shorter unmasked sequence.
`restore_masked_features` appends mask vectors and gathers through the inverse
permutation before decoding.

`ConvDecoder` applies grouped temporal convolutions and predicts teacher
features at every restored position. The pretraining objective selects only
masked positions.

### Objectives, mixup, and EMA teacher

`RegressionLoss` implements summed MSE or smooth L1, scaled by `1/sqrt(D)` when
the recipe does not provide a scale. It reports frame-token count as
`sample_size`.

`SigmoidFocalLoss` implements independent multilabel focal loss:

```text
p_t = y sigmoid(z) + (1-y)(1-sigmoid(z))
FL = alpha_t BCE(z,y) (1-p_t)^gamma.
```

`make_teacher_targets` normalizes the selected EMA layer outputs and averages
them. `a_weighted_level` and `mix_waveforms` implement perceptual gain-corrected
between-class mixing.

`EMATeacher` holds a float32, evaluation-mode copy of the student:

```text
teacher = decay × teacher + (1-decay) × student.
```

It copies buffers and never enables dropout.

### Shared encoder

`ConvFeatureEncoder` turns `[B,S]` waveforms into `[B,C,T]` local features.
`AudioEncoder.project_waveform` normalizes and projects them to `[B,T,D]`.

`PositionalConvEncoder` adds local relative-position information.
`TransformerStack` runs the optional prenet and main blocks while retaining
intermediate target tensors.

When `checkpoint_activations`, training mode, and gradient tracking are all
active, `TransformerStack` runs each selected block through non-reentrant
PyTorch activation checkpointing with RNG preservation. The stack draws NumPy
layerdrop decisions before entering the checkpointed block, so backward
recomputation cannot select a different layer path. Evaluation and frozen
fine-tuning call blocks directly.

`AudioEncoder.encode_projected` has two paths:

- The teacher and fine-tuning path keeps all projected frames.
- The masked student path zeros hidden positions before positional convolution,
  gathers unmasked projected and positional vectors, and gathers matching ALiBi
  rows and columns before attention.

The gather-before-scale ALiBi order prevents a clone-expanded full bias tensor
that exceeded A100 memory during verification.

### Task models

`Animal2VecPretrainingModel` performs this flow:

```text
waveform
  → optional gain-corrected mixup
  → one local projection
  ├─→ full EMA teacher → normalized top-layer targets
  └─→ cloned masks → shortened student → restored decoder predictions
  → regression at masked positions
```

`Animal2VecFineTuningModel` constructs its encoder architecture from the
pretraining configuration. It applies fine-tuning dropout, masks, mixup, and
freeze schedules, averages final transformer layers, and produces framewise
multilabel logits.

Fine-tuning copies the active `checkpoint_activations` value into the encoder
model derived from the pretrained configuration. An older pretrained snapshot
therefore cannot disable the current execution policy.

### Fine-tuning activation memory

A fine-tuning checkpoint at update 10,000 has not completed the 30,000-update
recipe. Its next forward is the first trainable-backbone forward and retains
dense attention activations that the frozen phase discarded. On the eight-A100
40 GB reproduction profile, the driver preserves the saved batch topology and
uses block recomputation to control that peak. Allocator mapping warnings near
the failure report exhausted capacity; changing allocator fragmentation
settings cannot replace the activation-memory policy.

## `a2v2/training.py`

This file owns state transitions around the models.

### Optimizer and schedule

`FairseqCompatibleAdam` uses the archived epsilon placement:

```text
m_t = beta1 m_(t-1) + (1-beta1) g_t
v_t = beta2 v_(t-1) + (1-beta2) g_t^2
step_t = lr sqrt(1-beta2^t) / (1-beta1^t)
theta_t = theta_(t-1) - step_t m_t / (sqrt(v_t) + epsilon).
```

`build_optimizer` excludes biases, one-dimensional normalization parameters,
ALiBi scales, and P-Swish parameters from weight decay.

`CosineUpdateScheduler` indexes learning rates by successful optimizer update.
It supports linear warmup and cosine cycles. Diagnostic early stops do not
change its full configured horizon.

#### AdaGC

`build_gradient_clipper` selects `global`, `none`, or `adagc`. AdaGC
runs after AMP unscale and distributed sample-size normalization. For updates
before `adagc_warmup_updates`, it uses `clip_norm` as a global threshold and
records each tensor's clipped norm minimum. Later updates clip each tensor
against the prior norm EMA, then update that EMA from the clipped norm.

Defaults are `adagc_beta=0.99`, `adagc_relative_clip=1.04`, and
`adagc_warmup_updates=100`. AdaGC stores one CPU FP32 scalar per named
trainable parameter. AMP overflow and rank-wide optimizer failure leave its
update counter and EMA state unchanged. Resume requires AdaGC state after
update zero.

#### 8-bit optimizers

`build_optimizer` adds PyTorch AdamW plus bitsandbytes Adam8bit and AdamW8bit.
The two 8-bit paths import bitsandbytes after device and
`min_8bit_size` validation. They require CUDA and the
`bitsandbytes>=0.49,<0.50` extra. The builder reports missing APIs, version
mismatch, unloaded CUDA libraries, and constructor failures as setup errors.
All optimizers use the same ordered decay and no-decay parameter groups.

The 0.49.2 wheel lacks a CUDA 13.3 binary. Bounded CUDA tests on the recorded
13.3 host used `BNB_CUDA_VERSION=130` to load the packaged CUDA 13.0 binary.
A source build covers CUDA 13.3 without that wheel override.

#### Cosine weight-decay clock

`CosineWeightDecayScheduler` uses the successful optimizer-update clock:

```text
w(u) = w_end + 0.5 * (w_0 - w_end) * (1 + cos(pi * min(u,U) / U))
```

`weight_decay_schedule=constant` creates no scheduler and preserves the
legacy default. The opt-in cosine schedule uses
`optimizer.weight_decay_end=0.0` when the endpoint is absent. Groups that
start at zero stay zero. A v2 checkpoint stores `last_update`; restore
recomputes the live value against the active, allowed `max_update` horizon.

### Checkpoints and random state

`capture_rng_state` records Python, NumPy, CPU PyTorch, and every CUDA generator.
Distributed training gathers one state per rank into the rank-zero checkpoint.
`restore_rng_state` requires the same world size and returns each rank to its
own stream.

`gather_rank_rng_states` serializes each tensor-bearing state to bytes and
requires exactly one payload in rank order. CUDA training creates a dedicated
Gloo process group with a two-hour timeout for checkpoint and validation
control data. Serialized CPU checkpoint metadata stays off the CUDA allocator,
and CPU validation decisions let nonzero ranks wait without holding a pending
NCCL collective. CPU distributed training already uses Gloo as its default
group.

`save_checkpoint` writes a temporary file and performs an atomic replacement.
`validate_checkpoint` enforces the versioned plain-container schema.

#### Checkpoint format v1 and v2

The reader accepts checkpoint format v1 and supplies legacy defaults for
stateful fields that v1 lacks. New saves use format v2. A v2 payload stores
gradient-clipper state, weight-decay scheduler state, distributed topology, and
the resume-compatibility fingerprint beside the model, optimizer, LR
scheduler, scaler, sampler, EMA teacher, and RNG state.

V1 can resume the legacy global or no-clipping path. It cannot resume AdaGC
after update zero because no norm history exists. Older v2 and local
checkpoints with `topology=None` remain readable. Architecture compatibility
still comes from serialized active and pretrained configs plus strict tensor
keys and shapes.

### Metrics and training engine

`FrameCounts` stores additive TP, FP, TN, and FN counts. `average_precision`
groups tied scores before integrating the precision-recall staircase.

`pretraining_variance_diagnostics` reproduces the Animal2Vec 1.0 collapse
monitor. For masked prediction or teacher matrix \(Z\in\mathbb{R}^{N\times D}\),
it returns:

```text
(1 / D) sum_d sqrt(Var_(N-1)(Z[:, d]) + 1e-6).
```

The historical `pred_var` and `target_var` names refer to this mean
feature-wise sample standard deviation. The helper packs count, coordinate
sums, and coordinate squared sums for student and teacher into one tensor. A
distributed all-reduce combines those moments before the helper calculates the
standard deviation. This order measures the union of rank-local examples.

`TrainingEngine.step` performs one logical optimizer update:

1. Run each microbatch forward and collect detached pretraining diagnostics.
2. Backpropagate each summed microbatch loss.
3. Sum loss and sample size across ranks.
4. Account for DDP's `1/W` gradient average.
5. Normalize gradients by global frame-token count.
6. Clip the global gradient norm.
7. Let GradScaler skip non-finite AMP attempts.
8. Advance optimizer, update counter, schedule, and EMA after success.

The engine averages `pred_var` and `target_var` across the microbatches in one
attempt and returns them in `UpdateResult`. Fine-tuning results leave the
optional fields empty. `workflows.py` adds numeric keys to pretraining update
JSON and terminal summaries without storing them in checkpoints.

An AMP-skipped attempt changes only GradScaler state. It does not advance model,
schedule, teacher, validation, sampler, or checkpoint cadence.

## `a2v2/workflows.py`

This file composes all lower-level pieces and owns external side effects.

### Compile policy and Graph breaks

The workflow moves the model to its device, calls `Module.compile` in place
when enabled, and then builds the clipper, optimizer, and DDP wrapper. In-place
compilation keeps parameter identities and state keys. Compile settings stay
in execution provenance and do not enter the mathematical resume fingerprint.

`torch_compile_fullgraph=false` permits graph breaks around deterministic
NumPy mask creation, dynamic `nonzero` selection, stable sample-ID scalar
access, and mask-count validation. A complete modern pretraining update in the
recorded CUDA run produced seven graph breaks and ten unique graphs.
`fullgraph=true` can reject NumPy layerdrop and serves as a diagnostic for
pure Transformer regions. A two-length dynamic fine-tuning test recorded one
unique graph.

The bounded A100 microbenchmark measured one warmed-up FP16 shape:
`B=2`, `T=128`, `D=64`, four heads, two layers, GEGLU, strict Flash, and
activation checkpointing. Eager took 10.3072 ms per iteration and compiled
took 3.7094 ms for five measured iterations after two warm-ups. That
`2.7787x` ratio describes this microbenchmark on one A100-SXM4-40GB. It makes
no paper-scale or general throughput claim.

Activation checkpointing remains the existing
`model.checkpoint_activations` option. It uses non-reentrant PyTorch
checkpointing with RNG preservation during gradient-enabled training and
creates no model parameter or buffer.

### Events and evaluation

`pool_probabilities` applies centered average or maximum smoothing.
`fuse_probabilities` thresholds pooled frames and converts maximal active runs
to half-open intervals in seconds.

The segmented evaluation section reproduces the archived matcher, including its
inclusive interval arithmetic, strict IoU comparison, split and merger counts,
classwise AP, macro and micro AP, fixed-threshold confusion metrics, and focal
threshold search. It retains all fixed-size output rows because an all-zero
target row can contain a false-positive segment score. Removing target-free
rows inflates AP.

`TensorBoardLogger` owns experiment event files. Rank zero records globally
reduced training scalars at `common.log_interval`. `_validate` supplies
recording-preserving frame tensors and segmented artifacts for scalar metrics,
PR curves, and IoU/split/merger histograms. The standalone checkpoint evaluator
uses the same class and `_validate` call, so it cannot drift to a second metric
implementation.

### Inference

`InferenceRunner` loads a native fine-tuning checkpoint, reconstructs both
active and pretrained configurations, and loads state strictly.

`run_tensor` selects or averages channels, resamples to the model rate, divides
long audio into bounded segments, computes probabilities and averaged
embeddings, assigns absolute timestamps, removes frames beyond the true final
duration, and fuses events.

`write_events` owns the two external event-table representations. Its default
native TSV stores label, onset seconds, offset seconds, and the full score.
Passing `audition=True` writes the Animal2Vec 1.0 Adobe marker schema:
`Name`, `Start`, `Duration`, `Time Format`, `Type`, and `Description`.
The Audition branch converts
\(\left(t_{\mathrm{start}},t_{\mathrm{end}}\right)\) into
\(\left(t_{\mathrm{start}},t_{\mathrm{end}}-t_{\mathrm{start}}\right)\),
formats both coordinates as `datetime.timedelta` values, and rounds the score
to three decimal places. It sorts markers by numeric onset before writing the
tab-delimited `.csv`.

The serializer receives the existing event tuple. It does not suppress
overlapping labels or reinterpret the `focal` channel. Researchers who need a
different event-selection policy can operate on `InferenceResult.events`
before calling `write_events`.

### Checkpoint conversion

The conversion section maps official Fairseq state keys to current model
attributes. It validates destination shapes and complete target-key coverage.
It never reshapes a plausible tensor to force a match.

Converted checkpoints contain inference state and configuration but no portable
Fairseq optimizer state. Use them for inference or initialization, not exact
continuation of the archived training run.

### Training orchestration

`run_training` resolves rank-local devices, seeds each rank, constructs model
and optimizer state, wraps DDP, restores checkpoints, creates token samplers and
DataLoaders, and owns validation and save cadence.

The public `run_training` function is a lifecycle boundary around the private
implementation. It destroys the default process group—and therefore the
auxiliary control group—when this call initialized distributed training,
including exception paths. It does not destroy a default group initialized by
an embedding caller; in that case it destroys only the auxiliary control group
created by the training call.

The workflow records batches delivered to training instead of the sampler
cursor advanced by DataLoader prefetch. It also supplies a private DataLoader
generator so worker setup does not consume the model RNG stream.

Only rank zero creates `TensorBoardLogger`. Relative event paths resolve below
`checkpoint.save_dir`; published `tensorboard_logdir: tb` values therefore
remain recognizable within a self-contained experiment directory. Resume uses
the next optimizer update as TensorBoard's purge boundary.

The installed commands call these functions:

| Command | Function |
| --- | --- |
| `a2v2-train` | `train_main` |
| `a2v2-infer` | `infer_main` |
| `a2v2-evaluate-sequence` | `evaluate_sequence_main` |
| `a2v2-convert-checkpoint` | `convert_checkpoint_main` |

For DDP, pass the installed training script to `torchrun`:

```bash
torchrun --standalone --nproc-per-node=4 \
  "$(command -v a2v2-train)" \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --device cuda
```

## Import migration

New code should import by responsibility:

```python
from a2v2.config import load_config
from a2v2.data import AudioDataset, load_audio
from a2v2.model import (
    Animal2VecFineTuningModel,
    Animal2VecPretrainingModel,
    AudioEncoder,
)
from a2v2.training import TrainingEngine, load_checkpoint
from a2v2.workflows import InferenceRunner, convert_checkpoint
```

The package root supports the common path:

```python
from a2v2 import (
    Animal2VecFineTuningModel,
    Animal2VecPretrainingModel,
    InferenceRunner,
    load_config,
)
```

The intermediate `baseline` namespace and the old `animal2vec` package no
longer exist. The source distribution contains no compatibility shims. A
single `a2v2` namespace gives the Animal2Vec 1.0 control and future research
code one readable home.

## Compatibility contracts

Native checkpoints store dictionaries and tensors rather than pickled model
instances. Source module paths do not enter state keys. Model attribute names
and construction order do.

Tests protect these ordered state signatures:

| Model | Entries | SHA-256 signature |
| --- | ---: | --- |
| Pretraining | 120 | `5efbec43fd1c1c392b8b4278cee6a21513f68f3095353bb22524d2ada2ecf6ad` |
| Fine-tuning | 59 | `b7b2ea2ad9017ec94d56516fbda49a932866b472a48633804619b7232f3d05bb` |

Configuration tests protect semantic YAML hashes. Documentation tests require a
module docstring and docstrings on every named definition. Repository layout
tests require exactly six Python files in `a2v2/`.

## Verification commands

Run the CPU suite:

```bash
python -m pytest -q
```

Run one-GPU tests:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -m gpu
```

Run a four-rank exact-resume probe:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  tests/gpu/nccl_resume.py \
  --output-dir results/gpu_reverification/resume-4
```

The report in `gpu-verification-20260723.md` records the official checkpoint,
FP32, FP16, gradient, optimizer-update, NCCL, memory, and exact-resume evidence
for the implementation before file consolidation. The consolidation preserved
function bodies and state signatures. A GPU agent should rerun the durable
suite after any future change to equations, randomness, model attributes,
checkpoint fields, batching, AMP, or DDP.

Both standalone distributed programs bind `LOCAL_RANK` before initializing
NCCL and record `rng_gather_backend: "gloo"` in their per-rank JSON evidence.
