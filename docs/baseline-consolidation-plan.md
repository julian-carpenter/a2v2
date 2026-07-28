# Baseline package consolidation implementation plan

> **Historical record:** This completed plan uses the intermediate `baseline`
> package and command names. Do not execute those naming steps against the
> current tree. Continue from the
> [A2V2 identity plan](superpowers/plans/2026-07-27-a2v2-project-identity.md).

Date: 2026-07-27

## Objective

Replace the 28-file `animal2vec` runtime package with the six-file `baseline`
package described in `baseline-consolidation-design.md`. Preserve every
paper-reproduction behavior while improving the source reading path and inline
scientific explanation.

This directory has no Git metadata, so the steps below omit commit boundaries.

## Task 1: Specify the target layout in tests

Edit:

- `tests/unit/test_repository_layout.py`
- `tests/unit/test_model_state_contract.py`
- `tests/unit/test_documentation.py`

Require exactly these files:

```text
baseline/__init__.py
baseline/config.py
baseline/data.py
baseline/model.py
baseline/training.py
baseline/workflows.py
```

Require the old `animal2vec` directory to be absent. Change the model contract
imports to the future modules without changing the expected signatures. Change
the documentation scan root to `baseline`.

Run the three tests and confirm that they fail because the new package does not
exist. This establishes that the tests can detect the requested change.

## Task 2: Build the six-file package

Create `baseline/__init__.py` with the public exports.

Move configuration code into `baseline/config.py`.

Merge audio, labels, dataset, and batching into `baseline/data.py`. Remove
imports between the merged sections and keep the original definition order
where it affects behavior.

Merge layers, normalization, Sinc convolution, attention, masking, decoder,
losses, EMA, front end, pretraining, and fine-tuning into `baseline/model.py`.
Preserve class bodies, registered-submodule assignment order, initialization
order, and tensor operation order.

Merge checkpoint, optimizer, metrics, and engine code into
`baseline/training.py`.

Merge event detection, event evaluation, inference, checkpoint conversion,
training orchestration, and the former command wrappers into
`baseline/workflows.py`. Rename colliding parser and entry functions so each
installed command has a stable target.

Add module introductions and section dividers. Around result-sensitive code,
place a rigorous `Mathematics:` comment before a high-level `Interpretation:`
comment.

Run import, state-signature, and representative unit tests before removing the
old package.

## Task 3: Update consumers and packaging

Rewrite imports in:

- unit, integration, and GPU tests;
- GPU helper programs;
- repository scripts that import runtime code;
- `pyproject.toml`.

Rename the distribution to `animal2vec-baseline`. Install these commands:

```text
baseline-train
baseline-infer
baseline-convert-checkpoint
```

Delete the old `animal2vec` package only after the new imports and focused tests
pass.

## Task 4: Update researcher documentation

Rewrite `README.md` and `docs/code-guide.md` around the five implementation
files. Explain the reading order, section map, shape notation, and command
changes.

Update active commands and imports in:

- `docs/checkpoint-conversion.md`
- `docs/reproducing-paper.md`
- `docs/gpu-verification-20260723.md`

Mark older refactoring and documentation plans as historical records when their
paths no longer describe the live tree. Do not rewrite the recorded GPU results.

Search all Python, TOML, YAML, and active Markdown files for stale runtime
imports or entry points.

## Task 5: Verify behavior and distribution contents

Run focused tests for:

- repository layout and documentation;
- configuration and recipe values;
- model state signatures;
- primitives, Sinc, attention, masking, front end, decoder, losses, EMA;
- pretraining and fine-tuning steps;
- optimizer, engine, checkpoint, and exact resume;
- checkpoint conversion, inference, event evaluation, and CLI behavior.

Run the complete CPU suite. The expected outcome is that all CPU tests pass and
CUDA tests skip because this machine has no GPU.

Build a wheel without dependency isolation if the local build tools allow it.
Inspect the archive and assert that it contains the six `baseline` Python files
and no `animal2vec` package.

Run syntax compilation and AST documentation audits. Recompute the two model
state signatures and all YAML semantic hashes as final independent checks.

Record the exact commands and outcomes in the final handoff.
