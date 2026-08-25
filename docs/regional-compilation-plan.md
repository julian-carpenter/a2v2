# Regional Pretraining Compilation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent modern pretraining OOMs caused by mask-length-dependent whole-model recompilation while retaining compiled Transformer acceleration.

**Architecture:** Keep `Animal2VecPretrainingModel.forward` eager and compile its student and teacher `TransformerBlock` instances in place with dynamic shapes. Preserve complete-model compilation for fine-tuning and other modules, and report the selected scope in run provenance.

**Tech Stack:** Python 3.12, PyTorch `torch.compile`, TorchDynamo/Inductor, CUDA Flash SDPA, pytest, Bash.

**Spec:** `docs/regional-compilation-design.md`

## Global Constraints

- Do not change `scripts/reproduce_meerkat_paper.sh` or its frozen SHA-256.
- Do not change mathematical checkpoint or resume-fingerprint state.
- Compile modules in place before optimizer and DDP construction.
- Compiled pretraining requires `common.torch_compile_dynamic=true`.
- Do not launch any diagnostic expected to exceed two hours without approval.

---

### Task 1: Regional compile-policy contract

**Files:**
- Modify: `tests/unit/test_engine.py`
- Modify: `a2v2/workflows.py`
- Modify: `a2v2/model.py`

**Interfaces:**
- Consumes: `Animal2VecPretrainingModel`, `TransformerBlock`, `CommonConfig`.
- Produces: `_CompilePolicyReport(scope: str, regions: int)` and `_compile_model_in_place(model, common) -> _CompilePolicyReport`.

- [ ] **Step 1: Write the failing receiver test**

Construct the tiny pretraining model, monkeypatch `nn.Module.compile`, call
`_compile_model_in_place`, and assert that only every block in
`student.prenet`, `student.transformer`, `teacher.model.prenet`, and
`teacher.model.transformer` receives the configured options. Assert the report
is `scope="transformer_blocks"` with the matching region count.

- [ ] **Step 2: Write the failing static-policy test**

Call the policy with compiled pretraining and `torch_compile_dynamic=False`.
Assert an actionable `RuntimeError` is raised before `Module.compile` is called.

- [ ] **Step 3: Run the tests and verify RED**

Run: `pytest tests/unit/test_engine.py -k 'compile_policy' -q`

Expected: the new regional receiver/report assertions fail because the complete
pretraining model is currently the sole compile receiver.

- [ ] **Step 4: Implement the minimal regional dispatcher**

Add the immutable report, enumerate student and teacher block regions, require
dynamic shapes for pretraining, compile each block in place, and retain the
existing generic one-region policy. Remove the now-redundant decorators from
`compute_mask_indices` and `make_mask_info`.

- [ ] **Step 5: Run the tests and verify GREEN**

Run: `pytest tests/unit/test_engine.py -k 'compile_policy' -q`

Expected: all selected tests pass while parameter IDs and state keys remain
unchanged.

### Task 2: Training provenance and multi-update regression

**Files:**
- Modify: `tests/integration/test_pretraining_step.py`
- Modify: `a2v2/workflows.py`

**Interfaces:**
- Consumes: `_CompilePolicyReport` returned during model construction.
- Produces: `training_summary.torch_compile.scope` and `.regions`.

- [ ] **Step 1: Update the integration expectation first**

Set the compiled CPU pretraining override to dynamic shapes and expect scope
`transformer_blocks` plus the exact number of compiled student/teacher blocks.
Add a run covering at least twelve deterministic update numbers whose retained
token lengths vary.

- [ ] **Step 2: Run the focused integration tests and verify RED**

Run: `pytest tests/integration/test_pretraining_step.py -k 'compiled' -q`

Expected: the summary lacks scope/region fields and the new regional contract
is unmet.

- [ ] **Step 3: Thread the report into the summary**

Capture the report returned before optimizer construction and add its immutable
fields to the existing `torch_compile` summary mapping. Do not add them to the
active mathematical config or resume fingerprint.

- [ ] **Step 4: Run the focused integration tests and verify GREEN**

Run: `pytest tests/integration/test_pretraining_step.py -k 'compiled' -q`

Expected: all selected tests pass and checkpoint keys have no `_orig_mod.`
prefixes.

### Task 3: Modern recipe, launcher, and documentation

**Files:**
- Modify: `configs/modern/rope_cls_geglu_pretrain.yaml`
- Modify: `scripts/animal2vec2_benchmark.sh`
- Modify: `tests/integration/test_modern_benchmark_driver.py`
- Modify: `docs/code-guide.md`
- Modify: `docs/reproducing-paper.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: dynamic regional pretraining policy.
- Produces: an explicit modern benchmark command with dynamic compilation and recorded scope.

- [ ] **Step 1: Add failing launcher/config assertions**

Assert the modern YAML sets `torch_compile_dynamic: true`, the launch command
contains `--override common.torch_compile_dynamic=true`, and the run profile
records the pretraining `transformer_blocks` and fine-tuning `model` scopes.

- [ ] **Step 2: Run the launcher tests and verify RED**

Run: `pytest tests/integration/test_modern_benchmark_driver.py -q`

Expected: the dynamic override and scope record assertions fail.

- [ ] **Step 3: Update recipe, launcher, and prose**

Change the pretraining recipe to dynamic shapes, add the explicit launcher
override/profile field, and document why masking stays eager while Transformer
blocks compile. Explain that a later NCCL timeout can be secondary to one-rank
compiler OOM.

- [ ] **Step 4: Run launcher and configuration tests**

Run: `pytest tests/integration/test_modern_benchmark_driver.py tests/unit/test_config.py -q`

Expected: all selected tests pass.

### Task 4: CPU, CUDA, DDP, and frozen-path verification

**Files:**
- Modify: `tests/gpu/test_cuda_training.py` only if the existing CUDA regression cannot express twelve changing mask updates.
- Verify: `scripts/reproduce_meerkat_paper.sh`

**Interfaces:**
- Consumes: completed regional compiler policy and modern launcher.
- Produces: verification evidence; no production API.

- [ ] **Step 1: Run focused CPU suites**

Run: `pytest tests/unit/test_engine.py tests/integration/test_pretraining_step.py tests/integration/test_modern_benchmark_driver.py tests/unit/test_config.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run focused one-GPU strict-Flash tests**

Run the existing modern compile/Flash CUDA selection in
`tests/gpu/test_cuda_training.py`, including multiple mask updates if needed.

Expected: updates finish without graph-limit warnings and strict Flash remains active.

- [ ] **Step 3: Run the bounded eight-GPU resume probe**

Copy the update-1 checkpoint to a temporary output, use a clean Inductor cache,
resume the exact `612000 x 2` command through at least update 10, and enable
`TORCH_LOGS=recompiles`.

Expected: no `ids_keep` recompilations, no compiler OOM, no NCCL timeout, and
peak reserved memory remains below device capacity on every rank.

- [ ] **Step 4: Verify legacy integrity and the full non-GPU suite**

Run the repository's frozen-launcher checksum test and the complete non-GPU
pytest suite.

Expected: the checksum remains
`485b4b36fe1045174a392834bd9ee9848538ab496944631a0b2d514e22d4de18`
and all tests pass.

- [ ] **Step 5: Review and commit**

Inspect `git diff --check`, review the complete diff, and commit the verified
regional-compilation fix directly on the previously authorized `main` branch.
