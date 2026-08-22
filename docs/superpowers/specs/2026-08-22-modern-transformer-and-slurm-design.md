# Modern Transformer and SLURM Design

## Purpose

This change adds modern, selectable training and model features without
changing the verified Animal2Vec 1.0 reproduction. The checked-in MeerKAT and
hyena recipes, `scripts/reproduce_meerkat_paper.sh`, legacy checkpoint tensor
names, and their default numerical paths remain the control. New behavior is
activated only by new configuration values or a separate SLURM launcher.

The work covers:

- rotary position embeddings (RoPE) as an alternative to ALiBi;
- PyTorch scaled-dot-product attention (SDPA), including a strict
  FlashAttention mode;
- an optional prepended CLS token for self-supervised regression and
  sequence-level fine-tuning;
- AdaGC gradient clipping;
- bitsandbytes 8-bit Adam and AdamW;
- the already-supported activation-checkpointing option;
- a packed GEGLU feed-forward network;
- opt-in `torch.compile` execution;
- simplified DeepScaleLM initialization and residual scaling;
- cosine weight-decay annealing; and
- an opt-in SLURM/torchrun launcher with explicit multi-node contracts.

## Non-Negotiable Reproduction Contract

The following invariants gate every implementation task:

1. Existing recipe files are not edited to enable a new feature.
2. An old YAML file produces the same resolved mathematical configuration as
   before this change.
3. The legacy position value resolves through
   `model.modalities.audio.use_alibi_encoder`; it retains the manual attention
   implementation.
4. Existing models retain the same state-dict keys, tensor shapes, parameter
   order, initialization, and optimizer grouping.
5. Existing checkpoints remain loadable without migration. New checkpoint
   fields use defaults when absent.
6. `scripts/reproduce_meerkat_paper.sh` retains its current local eight-A100
   command graph. SLURM support is a separate entry point.
7. Reproduction contract tests compare resolved configs, state signatures,
   dry-run commands, and deterministic legacy forward/update results before
   performance acceptance is considered.

The new example recipe lives outside the frozen reproduction recipe
directories. It is illustrative, not a new paper-reproduction default.

## Approaches Considered

### One automatic modernization pass

This would replace manual attention and the standard MLP in all models. It has
the smallest user-facing surface, but it makes old runs numerically different
and changes checkpoint shapes. It is rejected.

### Fork the entire model into a second implementation

This isolates the baseline but duplicates the convolutional frontend,
masking, teacher, fine-tuning, and conversion logic. The two paths would drift
and double the audit surface. It is rejected.

### Selected: shared implementation with explicit compatibility modes

The existing modules gain small strategy selections. A `legacy` value keeps
the old branch exactly. New values select modern components. Conditional
parameters are constructed only when their feature is enabled. This keeps one
data and training pipeline while making the state and numerical differences
visible in serialized configuration.

## Configuration Surface

### Common execution policy

`CommonConfig` gains:

```python
torch_compile: bool = False
torch_compile_backend: str = "inductor"
torch_compile_mode: str = "default"
torch_compile_fullgraph: bool = False
torch_compile_dynamic: bool | None = None
```

These fields affect execution only. Compilation is applied in place before
DDP wrapping so model state keys remain unchanged. `torch_compile=false`
preserves the current workflow.

### Model architecture and attention

`ModelConfig` gains:

```python
position_encoding: str = "legacy"       # legacy | alibi | rope | none
attention_backend: str = "legacy"       # legacy | manual | sdpa | flash
rope_theta: float = 10_000.0
use_cls_token: bool = False
cls_loss_weight: float = 1.0
classification_head: str = "frame"      # frame | cls
ffn_type: str = "mlp"                    # mlp | geglu
initialization: str = "legacy"           # legacy | deepscale_lm
```

Resolution rules are deliberately asymmetric:

- `position_encoding=legacy` reads `use_alibi_encoder` and keeps manual
  attention. This is the only default path.
- Explicit `alibi` selects ALiBi and forbids `sdpa` and `flash` because the
  current dense, learned per-layer bias is not accepted by the Flash kernel.
- Explicit `rope` or `none` with `attention_backend=legacy` resolves to
  `sdpa`; CUDA may dispatch to FlashAttention when its dtype and shape are
  supported.
- `attention_backend=flash` forces PyTorch's Flash SDPA backend and fails with
  an actionable error instead of silently using the math kernel.
- `manual` remains available for reference and CPU parity tests.
- RoPE and ALiBi are mutually exclusive by construction; one resolved enum is
  passed to the encoder.

`classification_head=cls` requires an encoder checkpoint built with
`use_cls_token=true`. `classification_head=frame` remains the default and
retains the present event-localization output.

### Optimization

`OptimizationConfig` gains:

```python
gradient_clip_method: str = "global"     # global | adagc | none
adagc_beta: float = 0.99
adagc_relative_clip: float = 1.04
adagc_warmup_updates: int = 100
```

The existing `clip_norm` is AdaGC's absolute GlobalGC warm-up threshold.
Legacy configs resolve to `global`. AdaGC uses the paper's tensor-wise
algorithm and defaults from arXiv:2502.11034 v4.

`OptimizerConfig` gains:

```python
min_8bit_size: int = 4096
weight_decay_schedule: str = "constant"  # constant | cosine
weight_decay_end: float | None = None
```

Optimizer names add `adam8bit` and `adamw8bit`. bitsandbytes remains an
optional dependency, imported only for those names. A missing or incompatible
installation raises a targeted setup error. Small tensors stay FP32 by the
bitsandbytes default. The native `adam` implementation remains the Fairseq
compatible baseline.

For `weight_decay_schedule=cosine`, the start is the existing
`optimizer.weight_decay` and the default end is `0.0`. Only parameter groups
whose initial weight decay is nonzero are scheduled. No-decay groups stay
zero. Constant scheduling is the legacy default.

## RoPE and Masked Token Order

RoPE is applied to packed Q and K after the QKV projection and before SDPA or
manual score construction. Head dimensions must be even. Sine and cosine
values are calculated in FP32 from explicit integer position IDs and cast to
Q/K dtype for rotation.

Position IDs are data, not inferred from the shortened student sequence:

- a teacher without CLS receives frame IDs `0..T-1`;
- a masked student gathers those IDs with the exact `ids_keep` tensor used to
  gather the unmasked embeddings;
- a teacher with CLS receives CLS ID `0` and frame IDs `1..T`;
- a masked student prepends CLS ID `0` to the frame IDs gathered by
  `ids_keep`.

This is essential because `ids_keep` can reorder retained tokens. Rotating by
`0..T_keep-1` would assign the wrong phase after mask shuffling. The same
position-ID tensor is passed through the prenet and main Transformer stack.

ALiBi keeps its exact legacy construction when CLS is disabled. For the new
CLS path, frame-to-frame bias is built from the explicit shuffled frame
coordinates, while the CLS row and column are zero. This preserves the exact
temporal prior between frames without treating a global summary token as an
audio sample adjacent to frame zero.

## SDPA and FlashAttention

The QKV projection remains packed, preserving its current checkpoint layout.
The modern branch calls `torch.nn.functional.scaled_dot_product_attention`.
Padding is expressed as an allowed-key boolean mask with a non-padding CLS
column prepended when required. Attention dropout is zero in evaluation and
uses the configured probability in training.

Backend behavior:

- `sdpa` lets PyTorch select FlashAttention, memory-efficient attention, or
  its math implementation;
- `flash` enters `torch.nn.attention.sdpa_kernel` with only
  `SDPBackend.FLASH_ATTENTION` enabled;
- `manual` and `legacy` retain explicit FP32 softmax and dense scores;
- ALiBi always uses the manual implementation.

The strict mode is the acceptance mechanism for the requested Flash path on
A100. SDPA fallback is useful for CPU tests and unsupported padding shapes,
but must not be reported as Flash. GPU tests run strict mode and compare
forward/backward results within dtype-appropriate tolerances against manual
attention.

## CLS Token

### Encoder

When enabled, `AudioEncoder` owns one learned `[1, 1, D]` CLS parameter. The
token is prepended after the convolutional positional encoder has processed
frame features. It is never included in frame masking, `ids_keep`, or
`ids_restore`. Padding prepends a `False` value. All returned Transformer layer
outputs include CLS at index zero.

The feature-padding mask and local feature tensors remain frame-only. Context
padding is token-level and includes CLS. This distinction is explicit in the
encoder output contract.

### Pretraining

The teacher encodes full frames plus CLS. The student encodes CLS plus retained
frames. The convolutional decoder receives only the student frame tokens and
uses the unchanged frame-only `MaskInfo` to restore the original frame axis.
Teacher layer outputs are split before target normalization. Existing
frame-target normalization sees frames only, so prepending CLS does not change
its time statistics. CLS targets use feature-wise LayerNorm for each selected
layer when either per-layer normalization option is active, then follow the
configured final LayerNorm after top-layer averaging.

A conditional linear CLS predictor maps the final student CLS state to teacher
target dimension. Its prediction is compared with the teacher's normalized
top-layer CLS target using the existing regression loss. Masked-frame and CLS
loss sums are combined as:

```text
loss = masked_frame_loss + cls_loss_weight * cls_loss
sample_size = masked_frame_count + cls_count
```

The default weight is one, so one CLS vector is counted like one masked frame
vector. Diagnostics concatenate frame and CLS predictions/targets. No CLS
module exists when the option is false, preserving legacy state dictionaries.

### Fine-tuning

The frame head is unchanged. The CLS head averages the requested top-layer CLS
states, applies final dropout, and returns `[batch, classes]` logits.
Framewise labels are reduced to recording-level occurrence labels with a
maximum over time. For mixup, occurrence labels are reduced before mixing so
the result is the same convex combination used for the waveforms; taking a
maximum after mixing could depend on event alignment.

CLS fine-tuning reports `sample_size=batch`, because time is no longer a loss
axis. Frame-event postprocessing rejects sequence logits with a clear message;
a sequence classification evaluation helper reports multilabel precision,
recall, F1, accuracy, and average precision without pretending to provide
event timing.

## Packed GEGLU

`ffn_type=geglu` replaces each standard MLP with:

```text
[value, gate] = packed_in(x).chunk(2, dim=-1)
h = value * GELU(gate)
out = out_projection(dropout(h))
```

`packed_in` emits `2 * int(embed_dim * mlp_ratio)` coordinates. The output
projection returns to `embed_dim`. Dropout locations match the current MLP as
closely as possible. `ffn_type=mlp` constructs the existing class unchanged.
GEGLU deliberately has a different state shape and parameter count and is
therefore an architecture choice, not a load-time reinterpretation of a
legacy checkpoint.

## DeepScaleLM

The selectable implementation follows the paper's simplified DeepScaleLM
scheme. For total Transformer depth `N = prenet_depth + depth`, with `N >= 2`:

```text
lambda = sqrt(1 - 2/N)
beta = sqrt(2/N)
residual(x, branch) = lambda * x + beta * branch
```

Both attention and FFN residual additions use this rule. The packed QKV slices
are initialized separately: Q and K with variance `1/d`; V, attention output,
and FFN weights with variance
`(1/d) * sqrt((1-p)/2)`, using the relevant branch dropout `p`. Biases are
zero. Audio convolution and projection initialization stay under their
existing rules because the text-embedding variance assumption in the paper
does not map directly to the acoustic frontend.

`initialization=legacy` runs the existing `init_bert_params` and residual code
without an extra multiplication. DeepScaleLM applies only when constructing a
new model. Checkpoint loading remains authoritative over initialization.

## AdaGC and Resume

AdaGC executes after AMP unscale and the existing world-size/sample-size
gradient normalization. During updates `t < adagc_warmup_updates`, it applies
global clipping with `clip_norm`, then records the running minimum of each
clipped tensor norm. After warm-up, each parameter tensor is clipped relative
to the previous EMA and then updates the EMA from the clipped norm.

The clipper owns one CPU FP32 scalar per named trainable parameter. Its state
dict stores the algorithm version, update count, ordered names, and norm EMAs.
State advances only after a successful optimizer update; AMP-overflow attempts
do not commit candidate state. A legacy checkpoint has no clipper state and is
valid only for global/none clipping. Resuming AdaGC without its state fails
closed unless the checkpoint is at update zero.

The logged gradient norm remains the pre-clipping global L2 norm for continuity.
Optional diagnostics add the number of tensor gradients clipped and the
largest tensor scale.

## 8-Bit Optimizers

The optional extra installs a pinned compatible bitsandbytes range. Builder
selection is:

- `adam`: current Fairseq-compatible implementation;
- `adamw`: native PyTorch AdamW;
- `adam8bit`: `bitsandbytes.optim.Adam8bit`;
- `adamw8bit`: `bitsandbytes.optim.AdamW8bit`.

All choices receive the same decay/no-decay parameter groups. Checkpoint save
and restore use the optimizer's normal `state_dict`. The reproduction config
continues to select `adam`; no automatic replacement occurs. A CUDA smoke test
is skipped when the optional dependency is absent and must pass when the
extra is installed.

## Weight-Decay Annealing

A small `CosineWeightDecayScheduler` runs on the same successful-update clock
as the learning-rate scheduler. With start `w_0`, end `w_end`, and horizon
`U`, it applies:

```text
w(u) = w_end + 0.5 * (w_0 - w_end) * (1 + cos(pi * min(u,U) / U))
```

Parameter groups carry an immutable `initial_weight_decay` marker so zero
groups stay zero. The scheduler's last update is checkpointed and restored.
The default constant schedule creates no new numerical transition.

## Activation Checkpointing

Gradient checkpointing is already selectable through
`model.checkpoint_activations`. It is propagated into the prenet and main
Transformer stacks, uses non-reentrant checkpointing with RNG preservation,
and is active only while training with gradients enabled. This work adds
cross-feature tests for RoPE position IDs, CLS, GEGLU, SDPA, and compile. It
does not rename or change the existing field.

## `torch.compile`

Compilation is opt-in and applied with `nn.Module.compile(...)` before DDP
wrapping. In-place compilation avoids `_orig_mod.` state-dict prefixes. The
workflow records the backend, mode, dynamic, and fullgraph values in its
resolved config and training summary.

Compile compatibility changes are limited to tensor-friendly code:

- padding masks are applied without Python `.any()` synchronization in the
  Transformer core;
- RoPE caches are represented as tensor operations or non-persistent buffers;
- strategy branches are constructor/config constants;
- random mask creation, NumPy layerdrop selection, data loading, checkpoint
  I/O, logging, and distributed control remain outside compiled regions.

`fullgraph=true` is a diagnostic contract and may reject unsupported research
settings such as NumPy layerdrop. The default `fullgraph=false` permits graph
breaks. Unit tests compile pure Transformer submodules; CUDA integration tests
compile one complete pretraining and fine-tuning update when the environment
supports Inductor.

## SLURM Launch and Distributed Contract

The existing runtime already consumes torchrun's `RANK`, `LOCAL_RANK`, and
`WORLD_SIZE`. A new launcher runs one torchrun agent per allocated node via
`srun`, preserving SLURM's `CUDA_VISIBLE_DEVICES`:

```text
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 ... \
  scripts/a2v2_slurm_node.sh <phase> -- <a2v2-train arguments>
```

The node wrapper invokes:

```text
python -m torch.distributed.run
  --nnodes=$SLURM_JOB_NUM_NODES
  --node-rank=$SLURM_NODEID
  --nproc-per-node=$SLURM_GPUS_ON_NODE
  --rdzv-backend=c10d
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT
  --rdzv-id=$SLURM_JOB_ID-$phase
```

The first hostname in `SLURM_JOB_NODELIST` is the default rendezvous host;
users may supply an explicit endpoint. The launcher validates homogeneous GPU
counts, `nodes * GPUs-per-node`, shared manifests/output visibility, unique
phase IDs, and the configured global world size. It never overwrites
scheduler-provided GPU visibility.

`scripts/reproduce_meerkat_paper.sh` stays the canonical verified local
control. A separate `scripts/reproduce_meerkat_slurm.sh` reproduces its phase
ordering with explicit multi-node rendezvous and runs final single-GPU
evaluation as a distinct one-node step.

### Checkpoint topology

The legacy single-node checkpoint RNG payload remains readable and unchanged.
New distributed metadata records, per global rank:

- hostname, global/local rank and local world size;
- visible CUDA device count and selected device ordinal;
- CUDA device UUID/name where available; and
- rank-local CUDA RNG state plus a schema tag.

Resume requires identical mathematical config, world size, data/sampler
fingerprint, seed, batching, and checkpoint schema. Hardware and transport
differences are recorded as numerical-equivalence warnings. Rank assignment
is never inferred from a stable hostname ordering because torchrun ranks may
change after an elastic restart.

The SLURM launcher acquires an output ownership lock keyed by job/run ID so two
jobs cannot write the same `checkpoint_last.pt.tmp`. Rank zero creates shared
directories and other ranks synchronize before use.

### Preemption

`SIGUSR1` handlers only set a process-local flag. At a completed optimizer
update, all ranks reduce that flag; if any rank requested preemption, they save
one coordinated checkpoint, barrier, flush logs, and exit with a documented
code. No collective or file write occurs inside a signal handler. `SIGTERM`
is best effort and is not advertised as exact preemption recovery. The batch
template requests sufficient `SIGUSR1` lead time and requeues only after a
valid checkpoint exists.

## Checkpoint Format

The checkpoint format advances only when new stateful runtime features require
it. Readers accept version 1 and supply legacy defaults. New payloads add:

- gradient-clipper state or `None`;
- weight-decay scheduler state or `None`;
- distributed topology metadata; and
- a resume-compatibility fingerprint.

Architecture differences remain self-describing through the serialized active
and pretrained configs and model tensor keys. Loading an incompatible
architecture fails with the existing strict key/shape checks.

## Error Handling

Configuration validation rejects:

- RoPE with odd attention head dimensions;
- ALiBi with SDPA/Flash;
- CLS classification without a CLS-enabled pretrained encoder;
- unsupported backend, FFN, initialization, clip, or decay schedule values;
- nonpositive RoPE theta, AdaGC relative factor, or bitsandbytes minimum size;
- invalid AdaGC beta or warm-up count;
- DeepScaleLM with fewer than two total Transformer layers;
- strict Flash mode on a backend that cannot execute it; and
- SLURM topology or resume-contract mismatches.

Optional dependency errors name the install extra. Performance modes never
silently alter batch size, masking, targets, optimizer choice, or checkpoint
resume state.

## Test Strategy

### Gate 0: frozen control

- full existing CPU suite;
- legacy config serialization and old-snapshot loading;
- baseline model state signatures and deterministic forward/update fixtures;
- paper-driver dry run and frozen config hashes.

### Gate 1: component units

- RoPE analytic rotations, dtype/device behavior, and gathered/shuffled IDs;
- SDPA/manual parity and strict backend errors;
- CLS prepend/padding/ALiBi/RoPE/mask restoration and target selection;
- sequence-label aggregation and mixup order;
- packed GEGLU shapes, gradients, and parameter estimates;
- DeepScaleLM variances, QKV slice initialization, residual coefficients;
- AdaGC warm-up minima, post-warm-up EMA, overflow rollback, state round trip;
- bitsandbytes lazy import and optimizer selection;
- weight-decay values, zero-group preservation, and resume;
- compile config and pure-module fullgraph tests;
- SLURM topology parsing, rendezvous, locks, signals, and compatibility diffs.

### Gate 2: CPU integration

- tiny pretraining updates for rope/no-CLS, rope/CLS, ALiBi/CLS, and GEGLU;
- tiny frame and CLS fine-tuning updates;
- uninterrupted versus resumed AdaGC and decay-annealed runs;
- two-rank Gloo checkpoint and topology round trip;
- mocked `srun`/torchrun command integration;
- old reproduction command assertions remain unchanged.

### Gate 3: bounded CUDA

- one-GPU strict Flash forward/backward for RoPE and no-position attention;
- activation-checkpointed RoPE/CLS/GEGLU step;
- one compiled pretraining and fine-tuning step;
- bitsandbytes Adam8bit and AdamW8bit save/resume when installed;
- 2- and 4-rank NCCL exact-resume tests.

No command expected to exceed two hours is launched without asking the user,
as previously requested.

### Gate 4: SLURM acceptance

On a real allocation, run one-node launcher parity, then at least 2 nodes by 2
GPUs for NCCL/Gloo transport, checkpoint/resume, shared-output locking, and
preemption. Multi-node support is documented as implemented but unvalidated
for a specific cluster until these site-dependent gates run. The README lists
the exact topology, software versions, and date for every completed gate.

## Documentation

- `README.md` gains a concise feature matrix, modern recipe example, and SLURM
  launch pointer while retaining the reproduction-first opening.
- `docs/code-guide.md` documents strategy resolution, tensor shapes, state,
  and compile boundaries.
- `docs/reproducing-paper.md` states that the local control is unchanged and
  describes how SLURM differs.
- `docs/slurm.md` is the authoritative allocation, launch, resume,
  preemption, and troubleshooting guide.
- `configs/README.md` explains which fields are architecture choices versus
  execution policies.

## Primary References

- RoPE: [RoFormer](https://arxiv.org/abs/2104.09864)
- data2vec 2.0: [Efficient Self-supervised Learning with Contextualized Target Representations](https://arxiv.org/abs/2212.07525)
- AdaGC: [arXiv:2502.11034](https://arxiv.org/abs/2502.11034)
- DeepScaleLM: [Transformers Get Stable](https://arxiv.org/abs/2403.09635)
- SDPA/Flash dispatch: [PyTorch SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- bitsandbytes: [8-bit optimizer guide](https://huggingface.co/docs/bitsandbytes/optimizers)
- multi-node torchrun: [PyTorch elastic launch documentation](https://docs.pytorch.org/docs/stable/elastic/run.html)
