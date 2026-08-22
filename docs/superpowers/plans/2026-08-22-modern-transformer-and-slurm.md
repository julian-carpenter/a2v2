# Modern Transformer and SLURM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RoPE, fused SDPA/FlashAttention, CLS pretraining and sequence classification, AdaGC, bitsandbytes optimizers, packed GEGLU, torch.compile, DeepScaleLM, weight-decay annealing, and SLURM multi-node launch support while preserving the verified Animal2Vec paper-reproduction path.

**Architecture:** Keep every existing recipe on an explicit legacy branch and introduce opt-in strategy fields. Position IDs flow from the original frame axis through mask gathers, packed QKV feeds either manual attention or PyTorch SDPA, and conditional CLS/GEGLU state exists only in new architectures. Stateful optimizer policies receive versioned checkpoint state. SLURM uses one torchrun agent per node and a separate launcher, leaving the local eight-A100 script unchanged.

**Tech Stack:** Python 3.10+, PyTorch 2.3+ (SDPA, FlashAttention dispatch, non-reentrant activation checkpointing, `nn.Module.compile`), pytest, optional bitsandbytes, Bash, torchrun, NCCL/Gloo, SLURM `srun`.

**Design:** `docs/superpowers/specs/2026-08-22-modern-transformer-and-slurm-design.md`

## Global Constraints

- Work directly on `main`, as previously authorized.
- Prefix every shell command with `rtk`.
- Use `apply_patch` for edits.
- Write and run a failing behavioral test before each production change.
- Do not edit the numerical values in checked-in MeerKAT or hyena reproduction recipes.
- Do not change `scripts/reproduce_meerkat_paper.sh` unless a test proves a documentation-only or additive no-op change. The intended result is no change.
- Preserve legacy config resolution, model state signatures, parameter order, manual attention, BERT initialization, Fairseq-compatible Adam, fixed clipping, fixed decay, and eager execution.
- Old version-1 checkpoints must load. New stateful features must fail closed when their required resume state is missing after update zero.
- Do not claim a fused Flash kernel unless strict Flash mode completes.
- Do not launch a command expected to run longer than two hours without asking the user.
- Keep site-dependent multi-node SLURM acceptance separate from implementation and mocked/local tests.

## File Structure

- Modify `a2v2/config.py`: add strict opt-in schema, resolver helpers, validation, and old-checkpoint defaults.
- Modify `a2v2/model.py`: add position-ID plumbing, RoPE, SDPA, CLS, sequence head, GEGLU, and DeepScaleLM.
- Modify `a2v2/training.py`: add AdaGC, decay scheduling, optional optimizer selection, metrics, and checkpoint v2 compatibility.
- Modify `a2v2/workflows.py`: wire compile, new optimization state, sequence validation, topology metadata, and graceful preemption.
- Add `a2v2/slurm.py`: pure SLURM environment, rendezvous, run-contract, lock, and signal helpers.
- Modify `pyproject.toml`: add the optional bitsandbytes extra.
- Add `scripts/a2v2_slurm_node.sh` and `scripts/reproduce_meerkat_slurm.sh`.
- Add unit/integration/GPU tests under the existing `tests/` layout.
- Modify `README.md`, `configs/README.md`, `docs/code-guide.md`, and `docs/reproducing-paper.md`; add `docs/slurm.md`.

---

### Task 1: Freeze the Reproduction Control and Add Config Strategies

**Files:**
- Modify: `a2v2/config.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_model_state_contract.py`
- Modify: `tests/integration/test_reproduction_driver.py`

- [ ] **Step 1: Add red compatibility tests**

Add tests that:

- load every checked-in MeerKAT and hyena YAML before and after a
  serialize/restore round trip;
- assert absent new fields resolve to legacy/default values;
- assert the known tiny pretraining and fine-tuning state signatures remain
  unchanged under defaults;
- assert the paper driver's dry-run launch lines remain unchanged;
- reject invalid enum/range/bool values and incompatible combinations.

Use names:

```text
test_modern_options_default_to_legacy_and_round_trip
test_legacy_serialized_config_supplies_modern_defaults
test_modern_option_validation_rejects_incompatible_combinations
test_reproduction_driver_retains_frozen_local_command_graph
```

- [ ] **Step 2: Run the focused tests and confirm red**

```bash
rtk python -m pytest -q tests/unit/test_config.py tests/unit/test_model_state_contract.py tests/integration/test_reproduction_driver.py
```

- [ ] **Step 3: Implement the minimal schema and resolvers**

Add the exact fields from the design. Use strict bool helpers for compile and
CLS flags, explicit enum sets, and numeric validation. Add:

```python
resolve_position_encoding(model: ModelConfig) -> str
resolve_attention_backend(model: ModelConfig) -> str
```

Explicit position selection takes precedence over the deprecated
`use_alibi_encoder`; the legacy value reads it. Restore missing serialized
fields from dataclass defaults before construction.

- [ ] **Step 4: Run the focused tests green**

Run the command from Step 2 and:

```bash
rtk git diff --check
```

- [ ] **Step 5: Commit the schema**

```bash
rtk git add a2v2/config.py tests/unit/test_config.py tests/unit/test_model_state_contract.py tests/integration/test_reproduction_driver.py
rtk git commit -m "feat: add opt-in modern training strategies"
```

---

### Task 2: Add Position IDs, RoPE, and SDPA/Flash Attention

**Files:**
- Modify: `a2v2/model.py`
- Modify: `tests/unit/test_attention.py`
- Modify: `tests/unit/test_masking.py`
- Modify: `tests/unit/test_transformer.py`
- Modify: `tests/gpu/test_cuda_components.py`

- [ ] **Step 1: Write red RoPE and masked-order tests**

Cover:

- a hand-calculated 2-D rotation at positions 0, 1, and 3;
- Q/K dtype, device, shape, and gradient preservation;
- odd head-dimension rejection;
- gathered student position IDs equal the original teacher IDs selected by
  `ids_keep[..., 0]`, including a deliberately shuffled keep order;
- Transformer stacks pass the same IDs into every block and activation
  checkpointing preserves them.

- [ ] **Step 2: Run the new RoPE tests red**

```bash
rtk python -m pytest -q tests/unit/test_attention.py tests/unit/test_masking.py tests/unit/test_transformer.py
```

- [ ] **Step 3: Implement scalar position-ID plumbing and RoPE**

Extend `MultiheadAttention.forward`, `TransformerBlock.forward`, and
`TransformerStack.forward` with optional `position_ids`. Calculate RoPE in
FP32 and cast rotated Q/K back. Gather original frame IDs in
`AudioEncoder.encode_projected` using the same indices as features and
convolutional positions. Do not create persistent cache buffers.

- [ ] **Step 4: Add red SDPA/backend tests**

Test manual-versus-SDPA forward/backward parity with:

- no padding;
- key padding;
- RoPE;
- training/evaluation dropout;
- explicit rejection of ALiBi plus SDPA/Flash;
- strict Flash failure on CPU.

Monkeypatch `F.scaled_dot_product_attention` to assert the modern branch is
actually called; do not inspect source text.

- [ ] **Step 5: Implement SDPA and strict Flash**

Retain the exact legacy/manual body. Convert `True=padding` to an SDPA
allowed-key mask. Use configured scale. Enter
`sdpa_kernel(SDPBackend.FLASH_ATTENTION)` only for strict mode and wrap kernel
errors with the backend, dtype, device, head dimension, length, and padding
status.

- [ ] **Step 6: Run CPU tests green**

```bash
rtk python -m pytest -q tests/unit/test_attention.py tests/unit/test_masking.py tests/unit/test_transformer.py tests/unit/test_model_state_contract.py
```

- [ ] **Step 7: Run bounded A100 strict-Flash tests**

```bash
rtk python -m pytest -q tests/gpu/test_cuda_components.py -k "flash or rope"
```

Assert forward/backward completes under FP16 and BF16, strict mode does not
fall back, outputs agree with manual attention within dtype tolerance, and
peak allocated memory is below manual dense attention for a long sequence.

- [ ] **Step 8: Commit attention**

```bash
rtk git add a2v2/model.py tests/unit/test_attention.py tests/unit/test_masking.py tests/unit/test_transformer.py tests/gpu/test_cuda_components.py
rtk git commit -m "feat: add rotary and fused attention paths"
```

---

### Task 3: Add CLS-Aware Encoding and Pretraining Regression

**Files:**
- Modify: `a2v2/model.py`
- Modify: `tests/unit/test_masking.py`
- Modify: `tests/unit/test_losses.py`
- Modify: `tests/integration/test_pretraining_step.py`

- [ ] **Step 1: Write red encoder/position tests**

Test full and masked encoders for all position modes:

- CLS is exactly index zero and never appears in `MaskInfo`;
- context length is frame length plus one;
- masked student length is retained frames plus one;
- context padding prepends false;
- RoPE receives CLS ID 0 and gathered one-based frame IDs;
- ALiBi has a zero CLS row/column and unchanged gathered frame-frame values;
- the frame-only convolutional positions, feature padding, and decoder restore
  axis stay unchanged.

- [ ] **Step 2: Implement conditional CLS in AudioEncoder**

Construct `cls_token` only when enabled. Prepend it after positional
convolution and frame gathering. Add a helper that constructs the new CLS ALiBi
matrix without changing the no-CLS legacy helper.

- [ ] **Step 3: Write red pretraining objective tests**

Test:

- teacher frame targets are identical to targets produced from the same frame
  layers without a prepended CLS;
- CLS target normalization is finite and feature-normalized;
- decoder consumes only frame tokens and returns exactly `T` frames;
- predictions/targets contain `M + B*clone_batch` rows;
- sample size and weighted summed loss match a direct calculation;
- gradients reach the CLS token and CLS predictor;
- teacher parameters stay gradient-free;
- disabled CLS produces the current exact output/state contract.

- [ ] **Step 4: Implement CLS target and predictor**

Add a target helper that splits CLS/frame layers. Use existing
`make_teacher_targets` for frames and the design's feature normalization for
CLS. Feed `student_context.x[:, 1:]` into decoder restore. Concatenate masked
frame and CLS regression inputs after applying `cls_loss_weight` without
double-normalizing sample size.

- [ ] **Step 5: Run CPU integration green**

```bash
rtk python -m pytest -q tests/unit/test_masking.py tests/unit/test_losses.py tests/integration/test_pretraining_step.py tests/unit/test_model_state_contract.py
```

- [ ] **Step 6: Commit CLS pretraining**

```bash
rtk git add a2v2/model.py tests/unit/test_masking.py tests/unit/test_losses.py tests/integration/test_pretraining_step.py
rtk git commit -m "feat: add cls teacher student regression"
```

---

### Task 4: Add CLS Sequence Fine-Tuning and Evaluation

**Files:**
- Modify: `a2v2/model.py`
- Modify: `a2v2/training.py`
- Modify: `a2v2/workflows.py`
- Modify: `tests/unit/test_finetuning.py`
- Modify: `tests/integration/test_finetuning_step.py`
- Modify: `tests/integration/test_inference.py`

- [ ] **Step 1: Write red sequence-head tests**

Cover:

- frame head with a CLS encoder strips index zero and preserves `[B,T,C]`;
- CLS head averages top-layer CLS states and returns `[B,C]`;
- frame labels reduce to clip occurrence labels before waveform/target mixup;
- sequence focal loss uses sample size `B`;
- CLS mode rejects a pretrained architecture without CLS;
- padded frame labels do not create false clip occurrences;
- event inference rejects sequence logits with a targeted error;
- sequence validation reports multilabel counts and AP.

- [ ] **Step 2: Implement head routing and sequence metrics**

Keep one classifier state layout. Select frame or CLS features before the
linear head. Add a pure `sequence_targets` helper and reuse existing
`FrameCounts`/average precision on flattened batch-class decisions under a
sequence-specific name. Do not route sequence logits through event smoothing.

- [ ] **Step 3: Run focused tests green**

```bash
rtk python -m pytest -q tests/unit/test_finetuning.py tests/integration/test_finetuning_step.py tests/integration/test_inference.py
```

- [ ] **Step 4: Commit sequence fine-tuning**

```bash
rtk git add a2v2/model.py a2v2/training.py a2v2/workflows.py tests/unit/test_finetuning.py tests/integration/test_finetuning_step.py tests/integration/test_inference.py
rtk git commit -m "feat: add cls sequence classification"
```

---

### Task 5: Add Packed GEGLU and DeepScaleLM

**Files:**
- Modify: `a2v2/model.py`
- Modify: `tests/unit/test_transformer.py`
- Modify: `tests/unit/test_finetuning.py`
- Modify: `tests/integration/test_pretraining_step.py`

- [ ] **Step 1: Write red GEGLU tests**

Assert the packed projection equation against a direct tensor calculation,
projection shapes, gradients, dropout placement, parameter estimate, activation
checkpoint parity, and expected strict shape mismatch when loading an MLP
checkpoint into GEGLU.

- [ ] **Step 2: Implement conditional packed GEGLU**

Use the existing `MLP` class unchanged for `ffn_type=mlp`. Add a
`PackedGEGLU` and construct it only for `geglu`. Pass the selection through
both stacks and update `estimated_parameter_count`.

- [ ] **Step 3: Write red DeepScaleLM tests**

For a fixed seed and sufficiently large matrices, assert:

- residual coefficients equal the paper formula for total encoder depth;
- both residual branches use the coefficients;
- Q/K packed slices and V/out/FFN roles have expected standard deviations
  within statistical tolerance;
- biases are zero;
- the student and initial EMA teacher match;
- loading a checkpoint overrides initialization;
- legacy initialization tensors remain exactly unchanged.

- [ ] **Step 4: Implement simplified DeepScaleLM**

Apply a role-aware initializer after module construction. Initialize packed QKV
slices separately. Preserve convolution, acoustic projection reset, decoder,
and classifier initialization. Pass total `prenet_depth + depth` to every
block and reject depths below two.

- [ ] **Step 5: Run focused tests green**

```bash
rtk python -m pytest -q tests/unit/test_transformer.py tests/unit/test_finetuning.py tests/integration/test_pretraining_step.py tests/unit/test_model_state_contract.py
```

- [ ] **Step 6: Commit architecture modules**

```bash
rtk git add a2v2/model.py tests/unit/test_transformer.py tests/unit/test_finetuning.py tests/integration/test_pretraining_step.py
rtk git commit -m "feat: add geglu and deepscale initialization"
```

---

### Task 6: Add AdaGC and Versioned Resume State

**Files:**
- Modify: `a2v2/training.py`
- Modify: `a2v2/workflows.py`
- Modify: `tests/unit/test_optim.py`
- Modify: `tests/unit/test_engine.py`
- Modify: `tests/integration/test_resume.py`
- Modify: `tests/gpu/test_cuda_training.py`
- Modify: `tests/gpu/nccl_resume.py`

- [ ] **Step 1: Write red hand-calculated AdaGC tests**

Use two named tensors and direct norms to prove:

- updates 0-99 use global clipping and track the minimum clipped tensor norm;
- update 100 uses the previous EMA and relative factor 1.04;
- the EMA uses clipped norms and beta 0.99;
- zero/missing gradients are safe;
- non-finite gradients do not mutate state;
- the returned diagnostic is the pre-clipping global norm;
- name-keyed state round trips exactly.

- [ ] **Step 2: Implement a transactional clipper**

Add `GlobalGradientClipper`, `AdaGradientClipper`, and a candidate/commit
result. The engine computes candidates after AMP unscale and distributed
normalization, commits only after a successful optimizer step, and leaves
scheduler/EMA/clipper clocks unchanged on overflow.

- [ ] **Step 3: Write red checkpoint migration/resume tests**

Prove:

- v1 fixed-clipping checkpoints load;
- new saves use v2 and contain clipper/decay/topology slots;
- v1 inference is unchanged;
- AdaGC at update greater than zero rejects missing state;
- uninterrupted and resumed AdaGC produce identical next model, optimizer,
  clipper, scheduler, teacher, and RNG state;
- DDP ranks produce identical AdaGC state.

- [ ] **Step 4: Implement v1-to-v2 normalization**

Normalize loaded v1 payloads by inserting optional state fields. Keep required
inference keys and strict model loading. Add active-config resume checks for
optimizer name, clip method/hyperparameters, decay schedule, mathematical
model strategy, batching, world size, and seed.

- [ ] **Step 5: Run CPU/GPU/DDP tests**

```bash
rtk python -m pytest -q tests/unit/test_optim.py tests/unit/test_engine.py tests/integration/test_resume.py
rtk python -m pytest -q tests/gpu/test_cuda_training.py -k "adagc or overflow"
rtk python -m pytest -q tests/gpu/test_distributed_smoke.py -k resume
```

- [ ] **Step 6: Commit AdaGC**

```bash
rtk git add a2v2/training.py a2v2/workflows.py tests/unit/test_optim.py tests/unit/test_engine.py tests/integration/test_resume.py tests/gpu/test_cuda_training.py tests/gpu/nccl_resume.py
rtk git commit -m "feat: add adaptive gradient clipping"
```

---

### Task 7: Add Weight-Decay Annealing and bitsandbytes Optimizers

**Files:**
- Modify: `a2v2/training.py`
- Modify: `a2v2/workflows.py`
- Modify: `pyproject.toml`
- Modify: `tests/unit/test_optim.py`
- Modify: `tests/integration/test_resume.py`
- Modify: `tests/gpu/test_cuda_training.py`

- [ ] **Step 1: Write red decay schedule tests**

Test constant exact parity; cosine start, midpoint, end, and clamping; permanent
zero no-decay group; successful-update clock; AMP skip; and exact resume.

- [ ] **Step 2: Implement decay scheduling**

Extend the update scheduler or add a coordinated
`CosineWeightDecayScheduler`. Apply update-zero values before the first step
and advance after successful updates with LR. Store initial decay markers on
parameter groups and log live weight decay in JSON/TensorBoard.

- [ ] **Step 3: Write red optional-optimizer tests**

Test lazy import, missing-extra message, CPU rejection, optimizer/group
selection through a fake bitsandbytes module, and no changes to native Adam
group order or semantics.

- [ ] **Step 4: Implement bitsandbytes support**

Add:

```toml
[project.optional-dependencies]
bnb = ["bitsandbytes>=0.49,<0.50"]
```

Select `Adam8bit` or `AdamW8bit` only on CUDA and pass
`min_8bit_size`. Record the library version in the run summary.

- [ ] **Step 5: Run focused tests and optional CUDA smoke**

```bash
rtk python -m pytest -q tests/unit/test_optim.py tests/integration/test_resume.py
rtk python -m pytest -q tests/gpu/test_cuda_training.py -k bitsandbytes
```

The CUDA test skips with an explicit reason when the extra is not installed.

- [ ] **Step 6: Commit optimizer work**

```bash
rtk git add a2v2/training.py a2v2/workflows.py pyproject.toml tests/unit/test_optim.py tests/integration/test_resume.py tests/gpu/test_cuda_training.py
rtk git commit -m "feat: add decay schedules and 8-bit optimizers"
```

---

### Task 8: Wire torch.compile and Low-Risk Throughput Improvements

**Files:**
- Modify: `a2v2/model.py`
- Modify: `a2v2/workflows.py`
- Modify: `tests/unit/test_engine.py`
- Modify: `tests/integration/test_pretraining_step.py`
- Modify: `tests/integration/test_finetuning_step.py`
- Modify: `tests/gpu/test_cuda_training.py`

- [ ] **Step 1: Write red compile-wiring tests**

Monkeypatch `nn.Module.compile` and prove disabled mode does nothing; enabled
mode forwards exact backend/mode/fullgraph/dynamic values; compilation occurs
before DDP; state keys and parameter objects stay unchanged.

- [ ] **Step 2: Implement in-place compile**

Call `model.compile(...)` after `to(device)` and before optimizer/DDP.
Expose compile settings in run/terminal summaries. Treat settings as execution
policy in resume provenance, not mathematical checkpoint state.

- [ ] **Step 3: Write red graph/throughput regression tests**

Cover:

- pure Transformer fullgraph eager/compiled forward/backward;
- RoPE, SDPA, CLS, GEGLU, and activation checkpointing combinations;
- whole tiny fine-tuning with dynamic lengths;
- pretraining under `fullgraph=false`;
- sample IDs remain CPU-resident while other batch tensors move to CUDA;
- GPU logging performs one update-level loss materialization, not one per
  microbatch.

- [ ] **Step 4: Remove avoidable graph breaks/synchronization**

Apply padding masks without tensor-dependent Python `.any()` inside attention.
Keep IDs on CPU for deterministic NumPy masking. Accumulate detached loss as a
device tensor until the update-level reduction. Do not change layerdrop,
masking RNG, sampler, or diagnostic definitions.

- [ ] **Step 5: Run CPU and bounded CUDA tests**

```bash
rtk python -m pytest -q tests/unit/test_engine.py tests/integration/test_pretraining_step.py tests/integration/test_finetuning_step.py
rtk python -m pytest -q tests/gpu/test_cuda_training.py -k compile
```

Record eager/compiled elapsed time and peak memory as diagnostics; do not assert
an unstable absolute speed threshold in pytest.

- [ ] **Step 6: Commit compile/performance**

```bash
rtk git add a2v2/model.py a2v2/workflows.py tests/unit/test_engine.py tests/integration/test_pretraining_step.py tests/integration/test_finetuning_step.py tests/gpu/test_cuda_training.py
rtk git commit -m "feat: add compile execution policy"
```

---

### Task 9: Add SLURM Topology, Contracts, Locking, and Preemption

**Files:**
- Add: `a2v2/slurm.py`
- Modify: `a2v2/training.py`
- Modify: `a2v2/workflows.py`
- Add: `tests/unit/test_slurm.py`
- Add: `tests/integration/test_slurm_launcher.py`
- Modify: `tests/gpu/nccl_probe.py`
- Modify: `tests/gpu/nccl_resume.py`

- [ ] **Step 1: Write red pure SLURM tests**

Test environment parsing, missing variables, homogeneous world-size arithmetic,
rendezvous endpoint/phase IDs, first-host resolution abstraction, run-contract
fingerprints/diffs, output lock ownership/contention, rank topology metadata,
and a signal handler that only sets a flag.

- [ ] **Step 2: Implement pure helpers**

Keep `a2v2/slurm.py` importable without CUDA or SLURM commands. Shell command
execution stays in scripts. Use explicit dataclasses and JSON-safe mappings.
Use an atomic exclusive lock file containing run/job identity and reject stale
lock takeover unless the user supplies a separate recovery command.

- [ ] **Step 3: Write red coordinated-preemption tests**

In a two-rank Gloo process, set the request flag on one rank, reduce it at a
post-update safe point, write one valid checkpoint, barrier, and return the
documented preemption status. Assert no checkpoint/collective runs in the
signal handler itself.

- [ ] **Step 4: Implement safe-point checkpointing and topology state**

Record rank-local CUDA RNG and topology metadata in v2 payloads while retaining
the v1 reader. Fix checkpoint preflight to compare local CUDA-state cardinality
to local visible devices rather than global world size. Rank zero creates
shared output and all ranks synchronize before writing.

- [ ] **Step 5: Run unit and local distributed tests**

```bash
rtk python -m pytest -q tests/unit/test_slurm.py
rtk python -m pytest -q tests/integration/test_resume.py -k preemption
rtk python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --device cpu
```

- [ ] **Step 6: Commit runtime support**

```bash
rtk git add a2v2/slurm.py a2v2/training.py a2v2/workflows.py tests/unit/test_slurm.py tests/integration/test_slurm_launcher.py tests/gpu/nccl_probe.py tests/gpu/nccl_resume.py
rtk git commit -m "feat: add slurm-safe distributed contracts"
```

---

### Task 10: Add the Separate SLURM Launchers

**Files:**
- Add: `scripts/a2v2_slurm_node.sh`
- Add: `scripts/reproduce_meerkat_slurm.sh`
- Modify: `tests/integration/test_slurm_launcher.py`
- Modify: `tests/integration/test_reproduction_driver.py`

- [ ] **Step 1: Write red mocked-command tests**

With mock `scontrol`, `srun`, and Python/torchrun executables, assert:

- one `srun` launcher task per node and one torchrun worker per GPU;
- scheduler `CUDA_VISIBLE_DEVICES` is preserved;
- unique rendezvous ID per phase;
- global world-size override equals nodes times GPUs;
- shared path and lock preflight occurs before launch;
- final evaluation is a separate one-node/one-GPU step;
- `SIGUSR1` forwards a checkpoint request and requeue waits for validation;
- the local paper script output remains unchanged.

- [ ] **Step 2: Implement shell launchers**

Use strict Bash mode, explicit positional/flag parsing, quoted arrays, and
`python -m torch.distributed.run`. Do not embed partition/account/site paths.
Provide `--dry-run`. Require a shared manifest and output path. Derive the
port from a bounded job-ID hash unless explicitly supplied.

- [ ] **Step 3: Run launcher integration tests**

```bash
rtk python -m pytest -q tests/integration/test_slurm_launcher.py tests/integration/test_reproduction_driver.py
rtk bash -n scripts/a2v2_slurm_node.sh
rtk bash -n scripts/reproduce_meerkat_slurm.sh
```

- [ ] **Step 4: Commit launchers**

```bash
rtk git add scripts/a2v2_slurm_node.sh scripts/reproduce_meerkat_slurm.sh tests/integration/test_slurm_launcher.py tests/integration/test_reproduction_driver.py
rtk git commit -m "feat: add slurm torchrun launchers"
```

---

### Task 11: Document and Add an Opt-In Modern Example

**Files:**
- Modify: `README.md`
- Modify: `configs/README.md`
- Modify: `docs/code-guide.md`
- Modify: `docs/reproducing-paper.md`
- Add: `docs/slurm.md`
- Add: `configs/modern/rope_cls_geglu_pretrain.yaml`
- Add: `configs/modern/rope_cls_geglu_finetune.yaml`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Write red example-config tests**

Load both modern examples, assert pretrain/fine-tune architecture compatibility,
strict Flash/RoPE/CLS/GEGLU selection, DeepScale/AdaGC/decay settings, and no
changes to frozen recipe hashes.

- [ ] **Step 2: Add examples and documentation**

Document every new field, architecture versus execution policy, checkpoint
compatibility, Flash strict versus SDPA fallback, CLS loss/sequence semantics,
AdaGC defaults, bitsandbytes installation, compile limitations, and the fact
that activation checkpointing already existed.

`docs/slurm.md` includes allocation examples, one-node and multi-node launch,
shared filesystem requirements, resume compatibility, preemption/requeue,
logging, troubleshooting, and an acceptance matrix. State which paths are
implemented, which were locally tested, and which require a real cluster.

- [ ] **Step 3: Run docs/example checks**

```bash
rtk python -m pytest -q tests/unit/test_config.py
rtk python -m compileall -q a2v2
rtk git diff --check
```

- [ ] **Step 4: Commit documentation**

```bash
rtk git add README.md configs/README.md docs/code-guide.md docs/reproducing-paper.md docs/slurm.md configs/modern tests/unit/test_config.py
rtk git commit -m "docs: explain modern and slurm training modes"
```

---

### Task 12: Full Verification, Review, and Acceptance Report

**Files:**
- Modify as review requires
- Add: `docs/verification/2026-08-22-modern-transformer.md`

- [ ] **Step 1: Run the full CPU suite**

```bash
rtk python -m pytest -q tests/unit tests/integration
```

- [ ] **Step 2: Run static/package checks**

```bash
rtk python -m compileall -q a2v2 tests
rtk python -m pip check
rtk git diff --check
rtk git status --short
```

- [ ] **Step 3: Re-run the frozen reproduction dry run and compare**

Run the existing environment and driver dry-run tests. Compare resolved
MeerKAT configs, state signatures, launch commands, and no-new-field defaults
against the Gate 0 snapshots.

- [ ] **Step 4: Run bounded CUDA and local DDP suites**

```bash
rtk python -m pytest -q tests/gpu/test_cuda_components.py
rtk python -m pytest -q tests/gpu/test_cuda_training.py
rtk python -m pytest -q tests/gpu/test_cuda_inference.py
rtk python -m pytest -q tests/gpu/test_distributed_smoke.py
```

Use only tests whose aggregate runtime is expected below two hours. Ask before
any longer reproduction or benchmark.

- [ ] **Step 5: Benchmark the modern attention path**

On one idle A100, compare manual legacy and strict Flash at representative
frame lengths. Record dtype, batch, heads, sequence length, forward/backward
time, peak allocated/reserved memory, compiler settings, and kernel mode. This
is evidence, not a pass/fail performance threshold.

- [ ] **Step 6: Request code review**

Use `superpowers:requesting-code-review` on the complete diff. Address only
evidence-backed issues, rerun affected and full tests, and record final command
outputs.

- [ ] **Step 7: Write the verification report**

Include:

- Git revision and dirty state;
- exact commands and outcomes;
- frozen reproduction evidence;
- CPU/CUDA/DDP/Flash/bitsandbytes results;
- compile graph limitations;
- SLURM mocked/local results;
- explicit unrun site-dependent multi-node gates;
- no claim that a topology was validated unless it actually ran.

- [ ] **Step 8: Final commit**

```bash
rtk git add docs/verification/2026-08-22-modern-transformer.md
rtk git commit -m "test: record modern training verification"
rtk git status --short --branch
```
