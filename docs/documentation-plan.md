# Repository-wide Documentation Implementation Plan

> **Historical record:** Agents completed this plan before the six-file
> consolidation. Its old package paths are evidence of that work, not
> instructions for the current tree.

> **For agentic workers:** Execute each task in order and mark its checkboxes.
> This plan uses inline execution because the standalone directory has no Git
> metadata and the user requested autonomous progress.

**Goal:** Document each source, test, configuration, and project-control file
for researchers and developers without changing executable behavior.

**Architecture:** Explanations live beside the code or value they describe.
The code guide provides cross-file concepts; source docstrings provide exact
local contracts. A documentation test enforces coverage and protects the YAML
value trees.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, PyYAML, pytest, Markdown, TOML,
and YAML comments.

## Global constraints

- Do not change runtime equations, control flow, public interfaces, tests, or
  configuration values.
- Preserve both model-state signatures recorded in `docs/code-guide.md`.
- Preserve all ten YAML semantic hashes recorded in Task 1.
- Do not edit the MIT license text.
- Preserve the dated GPU report as an audit record.
- Do not introduce documentation generators or runtime dependencies.
- Use plain language and define specialist terms at first use.

---

### Task 1: Add enforceable documentation contracts

**Files:**

- Create: `tests/unit/test_documentation.py`
- Reference: all Python and YAML files in the repository

- [x] Add an AST-based test requiring a module docstring in each Python file.
- [x] Require docstrings on classes and functions, excluding only standard
      double-underscore protocol methods.
- [x] Require every YAML file to begin with an explanatory comment.
- [x] Canonicalize each YAML value tree as sorted JSON and compare its SHA-256
      with the ten pre-documentation hashes.
- [x] Run the new tests and confirm that missing test-module and function
      docstrings make the coverage test fail before documentation edits.

### Task 2: Document the 28 runtime modules

**Files:**

- Modify: `animal2vec/*.py`

- [x] Expand module docstrings with each file's place in the model or runtime.
- [x] Add parameter, unit, shape, return, error, and state semantics to public
      functions and classes.
- [x] Document private helpers whose equations or ordering affect parity.
- [x] Document `forward`, state serialization, sampler iteration, and training
      orchestration methods.
- [x] Add researcher-facing comments for Sinc filters, ALiBi, masking, teacher
      targets, mixup, focal loss, event matching, optimizer equations, AMP,
      DDP, and exact resume.
- [x] Run unit tests for each edited subsystem.

### Task 3: Document audit programs and test infrastructure

**Files:**

- Modify: `scripts/*.py`
- Modify: `tests/gpu/compare_training_checkpoints.py`
- Modify: `tests/gpu/conftest.py`
- Modify: `tests/gpu/nccl_probe.py`
- Modify: `tests/gpu/nccl_resume.py`
- Modify: `tests/gpu/official_native_inference.py`

- [x] Explain dataset assumptions, counters, hashes, and exit status in both
      audit programs.
- [x] Document checkpoint comparison recursion and ignored paths.
- [x] Document GPU marker behavior, NCCL topology, rank-local random state,
      and exact-resume branch construction.
- [x] Document the official native inference artifact format.

### Task 4: Document every behavioral test

**Files:**

- Modify: `tests/unit/*.py`
- Modify: `tests/integration/*.py`
- Modify: `tests/gpu/test_*.py`

- [x] Add a module-level statement of scope to each test file.
- [x] Add a purpose docstring to each test and fixture/helper function.
- [x] Explain numerical tolerances, expected errors, state comparisons, and
      researcher-visible implications beside non-obvious assertions.
- [x] Run the documentation coverage test until no Python definitions remain
      undocumented.

### Task 5: Annotate all YAML recipes and project-control files

**Files:**

- Modify: `configs/**/*.yaml`
- Modify: `tests/fixtures/*.yaml`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [x] Add an experiment header and section explanations to each YAML file.
- [x] Define units and parameter interactions beside non-obvious values.
- [x] Explain packaging metadata, entry points, dependency floors, and pytest
      discovery in `pyproject.toml`.
- [x] Explain why each direct dependency appears in `requirements.txt`.
- [x] Run the YAML semantic-hash test and recipe-loading tests.

### Task 6: Expand the researcher reading path

**Files:**

- Modify: `README.md`
- Modify: `docs/code-guide.md`
- Review: all remaining `docs/*.md`

- [x] Add a glossary of bioacoustic and machine-learning terms.
- [x] Add an equations-to-source map for the paper's main operations.
- [x] Explain the distinction between frames, events, segments, recordings,
      batches, microbatches, clones, and distributed ranks.
- [x] Cross-check commands, paths, test counts, and GPU claims.
- [x] Resolve every local Markdown link and verify fenced code blocks.

### Task 7: Complete regression and packaging verification

**Files:**

- Modify only documentation defects found during verification.

- [x] Compile all Python source.
- [x] Run the complete pytest suite outside socket restrictions.
- [x] Recompute model-state signatures.
- [x] Run recipe tests and all three CLI help commands.
- [x] Build and inspect the wheel.
- [x] Scan for forbidden runtime imports, trailing whitespace, undocumented
      definitions, stale paths, unresolved links, and generated caches.
