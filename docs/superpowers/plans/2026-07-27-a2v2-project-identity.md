# A2V2 Project Identity Implementation Plan

**Status:** Completed and verified on 2026-07-27.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the live `baseline` identity with `a2v2` and label the
published configurations as Animal2Vec 1.0 reproduction baselines.

**Architecture:** Keep the six-file flat package and all numerical code
unchanged. Rename package and consumer paths in one migration, then update
packaging, commands, recipe introductions, and active researcher documentation.

**Tech Stack:** Python 3.10+, PyTorch, setuptools, pytest, YAML.

## Global Constraints

- Keep exactly six Python files in the runtime package.
- Preserve all model state names, shapes, dtypes, and serialized order.
- Preserve all YAML values and semantic hashes.
- Do not add compatibility packages or command aliases.
- Keep Fairseq, Hydra, OmegaConf, Transformers, and torchaudio out of runtime
  imports.
- This standalone directory has no Git metadata, so this plan omits commit
  steps.

---

### Task 1: Lock the A2V2 package identity

**Files:**

- Modify: `tests/unit/test_repository_layout.py`
- Modify: `tests/unit/test_documentation.py`
- Modify: `tests/unit/test_model_state_contract.py`

**Interfaces:**

- Consumes: the current six-file package contract and both state signatures.
- Produces: tests that import `a2v2` and reject stale package directories.

- [x] **Step 1: Change the layout test**

Set `PACKAGE = ROOT / "a2v2"`, require `baseline/` and `animal2vec/` to be
absent, and import `a2v2.config`, `a2v2.data`, `a2v2.model`,
`a2v2.training`, and `a2v2.workflows`.

- [x] **Step 2: Change the documentation and state-contract imports**

Scan `ROOT / "a2v2"` in the documentation test. Import `load_config` from
`a2v2.config` and both task models from `a2v2.model`.

- [x] **Step 3: Verify red**

Run:

```bash
python -m pytest -q \
  tests/unit/test_repository_layout.py \
  tests/unit/test_documentation.py \
  tests/unit/test_model_state_contract.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'a2v2'`.

### Task 2: Rename source and Python consumers

**Files:**

- Move: `baseline/` to `a2v2/`
- Modify: all Python files under `tests/` that import `baseline`

**Interfaces:**

- Consumes: the same relative imports inside the six runtime files.
- Produces: importable `a2v2` package with unchanged public objects.

- [x] **Step 1: Move the package directory**

Rename the directory without changing implementation file contents. Update
source comments that name `baseline.data` as a package path to `a2v2.data`.

- [x] **Step 2: Rewrite consumer imports**

Replace `from baseline.` with `from a2v2.` in unit, integration, and GPU Python
files. Replace any `import baseline.` form in the same scope.

- [x] **Step 3: Run the focused package and model tests**

Run:

```bash
python -m pytest -q \
  tests/unit/test_repository_layout.py \
  tests/unit/test_model_state_contract.py \
  tests/unit/test_config.py \
  tests/unit/test_attention.py \
  tests/integration/test_pretraining_step.py \
  tests/integration/test_finetuning_step.py
```

Expected: all selected tests pass and both state signatures retain their exact
hashes.

### Task 3: Rename distribution and commands

**Files:**

- Modify: `pyproject.toml`
- Modify: command examples in active Markdown documentation

**Interfaces:**

- Produces: distribution `a2v2` with `a2v2-train`, `a2v2-infer`, and
  `a2v2-convert-checkpoint`.

- [x] **Step 1: Update setuptools metadata**

Set project name to `a2v2`, update its description, point each script to
`a2v2.workflows`, and change package discovery to `include = ["a2v2"]`.

- [x] **Step 2: Update live commands and imports**

Rewrite active README, code-guide, checkpoint-conversion, and
reproducing-paper examples from `baseline-*` and `baseline.*` to `a2v2-*` and
`a2v2.*`.

- [x] **Step 3: Verify command functions**

Run each parser through the source checkout:

```bash
python -c "from a2v2.workflows import build_training_parser; build_training_parser().parse_args(['--help'])"
python -c "from a2v2.workflows import build_inference_parser; build_inference_parser().parse_args(['--help'])"
python -c "from a2v2.workflows import build_checkpoint_conversion_parser; build_checkpoint_conversion_parser().parse_args(['--help'])"
```

Expected: each command prints help and exits with status zero.

### Task 4: Define Animal2Vec 1.0 recipe scope

**Files:**

- Create: `configs/README.md`
- Modify: the introduction comments in all eight files under `configs/`
- Modify: `README.md`
- Modify: `docs/code-guide.md`
- Modify: `docs/reproducing-paper.md`
- Modify: current-layout notes in historical documents

**Interfaces:**

- Produces: clear terminology for official 1.0 recipes, smoke checks, and
  future A2V2 recipes.

- [x] **Step 1: Add the configuration inventory**

Explain that MeerKAT and Hyena recipes preserve Animal2Vec 1.0 settings, CPU
smoke recipes check workflow behavior, and future A2V2 recipes should use an
`a2v2/` subdirectory.

- [x] **Step 2: Annotate recipe introductions**

Add “Animal2Vec 1.0 reproduction baseline” to each official recipe header.
Describe CPU smoke recipes as reduced checks rather than result-reproduction
recipes. Do not change YAML values.

- [x] **Step 3: Update project explanations**

Describe A2V2 as the shared project roof and the current code as its
Animal2Vec 1.0 baseline implementation. Preserve the historical GPU report's
recorded commands and add the live `a2v2` mapping to its top note.

- [x] **Step 4: Verify YAML values**

Run:

```bash
python -m pytest -q tests/unit/test_documentation.py tests/integration/test_recipe_configs.py
```

Expected: all recipe context and semantic-hash checks pass.

### Task 5: Complete regression and packaging verification

**Files:**

- Verify: entire repository.

**Interfaces:**

- Produces: a clean, installable A2V2 source tree.

- [x] **Step 1: Run the complete CPU suite**

Run `python -m pytest -q` with multiprocessing permissions.

Expected: 162 tests pass and 14 CUDA tests skip.

- [x] **Step 2: Run independent contracts**

Check six-file layout, both state signatures, 186 paired mathematical and
conceptual comments, full docstrings, forbidden imports, and absence of live
`baseline` package paths.

- [x] **Step 3: Build and inspect the wheel**

Build with:

```bash
python -m pip wheel --no-deps --no-build-isolation . \
  --wheel-dir /tmp/a2v2-wheel-verification
```

Require six `a2v2/*.py` files, no `baseline/` or `animal2vec/` package, and
three `a2v2-*` entry points.

- [x] **Step 4: Clean generated artifacts**

Remove build, egg-info, pytest, and bytecode caches created by verification.
