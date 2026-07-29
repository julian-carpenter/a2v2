# TensorBoard and Segmented Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record training and validation measurements in TensorBoard and include legacy-compatible segmented metrics in every labeled validation.

**Architecture:** `a2v2.workflows` will own one rank-zero experiment logger and the shared validation data flow. The existing event matcher remains the compatibility implementation; its dataset aggregator will retain false-positive rows and expose fixed-threshold metrics. Both the training loop and standalone evaluator will invoke the same logger and `_validate` function.

**Tech Stack:** Python 3.10+, pure PyTorch metric computation, `torch.utils.tensorboard`, TensorBoard event files, pytest.

## Global Constraints

- Preserve the archived average/maximum pooling, interval endpoints, strict IoU comparison, and segment score selection.
- Compute segmented metrics for every fine-tuning validation.
- Keep stdout JSON and checkpoint schemas backward compatible.
- Open a TensorBoard writer only on rank zero.
- Do not add transformers, torchaudio, SciPy, scikit-learn, or intervaltree to the native runtime.

---

### Task 1: Correct segmented aggregation and protect matcher parity

**Files:**
- Modify: `a2v2/workflows.py`
- Modify: `tests/unit/test_event_evaluation.py`

**Interfaces:**
- Consumes: `legacy_segmented_evaluation(...) -> SegmentedEvaluation`
- Produces: `aggregate_segmented_metrics(evaluations, labels, metric_threshold=...) -> SegmentedMetrics`

- [ ] **Step 1: Write failing aggregation tests**

Add a fixture with one target segment scored below one false-positive segment.
Assert fixed-threshold precision `0.5` and average precision `0.5`. The current
all-zero-target filter must make this test fail.

- [ ] **Step 2: Run the focused test and confirm the expected failure**

Run:

```bash
pytest -q tests/unit/test_event_evaluation.py
```

Expected: failure because `SegmentedMetrics` lacks fixed-threshold fields or
because the false-positive row was removed.

- [ ] **Step 3: Implement complete aggregation**

Retain all flattened rows, compute `FrameCounts` at `metric_threshold`, average
per-class AP for macro AP, flatten all class decisions for micro AP, and retain
the focal-label threshold fields.

- [ ] **Step 4: Add an independent archived matcher characterization**

Transcribe the archived list/interval operations inside the test module and
compare scores, targets, IoUs, splits, and mergers for fixed and seeded-random
fixtures. Keep this oracle independent from production helpers.

- [ ] **Step 5: Run event-evaluation tests**

Run:

```bash
pytest -q tests/unit/test_event_evaluation.py
```

Expected: all tests pass.

### Task 2: Preserve TensorBoard configuration and dependency

**Files:**
- Modify: `a2v2/config.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `CommonConfig.tensorboard_logdir: Path`

- [ ] **Step 1: Write a failing configuration test**

Load a recipe containing `common.tensorboard_logdir: tb` and assert the typed
value equals `Path("tb")`. Assert the default equals `Path("tensorboard")`.

- [ ] **Step 2: Run the test and confirm the missing field**

Run:

```bash
pytest -q tests/unit/test_config.py
```

- [ ] **Step 3: Parse and serialize the field**

Add the typed field, reject empty paths, and retain it through
`config_to_dict`/`config_from_serialized_dict`.

- [ ] **Step 4: Declare TensorBoard**

Add `tensorboard>=2.14` to both runtime dependency lists.

- [ ] **Step 5: Run configuration tests**

Run:

```bash
pytest -q tests/unit/test_config.py tests/unit/test_checkpoint.py
```

### Task 3: Add the shared experiment logger

**Files:**
- Modify: `a2v2/workflows.py`
- Create or modify: `tests/unit/test_tensorboard_logging.py`

**Interfaces:**
- Produces: `TensorBoardLogger`
- Produces: `resolve_tensorboard_directory(config) -> Path`
- Consumes: `UpdateResult`, validation metrics, frame tensors, segmented evaluations

- [ ] **Step 1: Write failing real-event-file tests**

Create a logger in a temporary directory, log one pretraining update and one
fine-tuning validation fixture, close it, then use TensorBoard's
`EventAccumulator` to assert scalar, PR-curve, and histogram tags.

- [ ] **Step 2: Run the test and confirm the logger is absent**

Run:

```bash
pytest -q tests/unit/test_tensorboard_logging.py
```

- [ ] **Step 3: Implement the logger**

Wrap `SummaryWriter`, add configuration text, training scalars, validation
scalars, classwise PR curves, and nonzero segment diagnostic histograms.
Sanitize label names for tag paths and flush after each logical record.

- [ ] **Step 4: Run logger tests**

Run:

```bash
pytest -q tests/unit/test_tensorboard_logging.py
```

### Task 4: Add segmented metrics to shared validation

**Files:**
- Modify: `a2v2/workflows.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `tests/integration/test_reproduction_driver.py`

**Interfaces:**
- Extends: `_validate(..., tensorboard_logger: TensorBoardLogger | None = None) -> dict[str, float]`

- [ ] **Step 1: Write failing validation tests**

Enable validation in the tiny fine-tuning flow and assert the stdout validation
record and standalone JSON report contain `segmented_precision`,
`segmented_recall`, `segmented_f1`, and `segmented_average_precision`.

- [ ] **Step 2: Run the tests and confirm missing metrics**

Run:

```bash
pytest -q tests/integration/test_cli.py tests/integration/test_reproduction_driver.py
```

- [ ] **Step 3: Retain recording axes and evaluate segments**

For each valid sample, remove padding frames, calculate the archived pooling
window, call `legacy_segmented_evaluation`, aggregate the results, and merge
the scalar fields into `_validate`'s return dictionary.

- [ ] **Step 4: Pass artifacts to the logger**

Log validation scalars, frame/class PR curves, segmented PR curves, and
IoU/split/merger histograms before returning.

- [ ] **Step 5: Run integration tests**

Run:

```bash
pytest -q tests/integration/test_cli.py tests/integration/test_reproduction_driver.py
```

### Task 5: Wire training and standalone evaluation

**Files:**
- Modify: `a2v2/workflows.py`
- Modify: `scripts/evaluate_finetuning_checkpoint.py`
- Modify: `scripts/reproduce_meerkat_paper.sh`
- Modify: `tests/integration/test_cli.py`
- Modify: `tests/integration/test_reproduction_driver.py`

**Interfaces:**
- Adds: evaluator option `--tensorboard-dir PATH`

- [ ] **Step 1: Write failing end-to-end event-file tests**

Run one tiny pretraining update and one tiny fine-tuning validation. Parse the
event files and assert training variance, optimizer, frame validation, and
segmented validation tags. Run the evaluator and assert it creates events next
to the report.

- [ ] **Step 2: Confirm event directories are missing**

Run:

```bash
pytest -q tests/integration/test_cli.py tests/integration/test_reproduction_driver.py
```

- [ ] **Step 3: Wire rank-zero training logging**

Create the logger after resume restoration, record run metadata, log each
stdout update at the same cadence, pass it to scheduled validation, and close
it on success or error.

- [ ] **Step 4: Wire evaluator logging**

Resolve the optional directory, create a logger, call shared validation, close
the logger, and add the event directory to report metadata. Update the
reproduction shell to pass an explicit final-evaluation path.

- [ ] **Step 5: Run end-to-end tests**

Run:

```bash
pytest -q tests/integration/test_cli.py tests/integration/test_reproduction_driver.py
```

### Task 6: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/code-guide.md`
- Modify: `docs/reproducing-paper.md`

**Interfaces:**
- Documents: event locations, tag taxonomy, segmented metric semantics, and launch command

- [ ] **Step 1: Document TensorBoard usage**

Include `tensorboard --logdir <save_dir>/tb`, the default directory rule, all
major tag groups, and the distinction between framewise and segmented metrics.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest -q tests/unit/test_event_evaluation.py tests/unit/test_tensorboard_logging.py tests/unit/test_config.py tests/integration/test_cli.py tests/integration/test_reproduction_driver.py
```

- [ ] **Step 3: Compile changed Python files**

Run:

```bash
python -m py_compile a2v2/config.py a2v2/workflows.py scripts/evaluate_finetuning_checkpoint.py
```

- [ ] **Step 4: Run the complete CPU suite**

Run:

```bash
pytest -q
```

GPU-only tests may skip on this host. Any multiprocessing tests blocked by the
sandbox should run once with permission for local process sockets.
