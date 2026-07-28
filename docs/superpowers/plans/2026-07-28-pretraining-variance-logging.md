# Pretraining Variance Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add archived-compatible `pred_var` and `target_var` collapse diagnostics to pretraining update logs and terminal summaries.

**Architecture:** A training helper will compute the archived mean feature-wise sample standard deviation from detached FP32 prediction and target tensors. It will reduce packed sufficient statistics across distributed ranks, while `TrainingEngine.step` will average the per-microbatch scalars and return them in optional `UpdateResult` fields. The workflow will print those fields for pretraining without changing model or checkpoint state.

**Tech Stack:** Python 3.10+, PyTorch tensors and distributed collectives, native JSON logging, pytest.

## Global Constraints

- Preserve the historical JSON keys `pred_var` and `target_var`.
- Use masked decoder predictions and matched normalized EMA-teacher targets.
- Compute \(\operatorname{mean}_d\sqrt{\operatorname{Var}_{N-1}(z_{:,d})+10^{-6}}\).
- Cast diagnostic inputs to FP32 and run all operations under `torch.no_grad()`.
- Reduce count, sum, and squared sum before applying the variance formula.
- Average scalar diagnostics across accumulated microbatches.
- Leave fine-tuning results at `None`.
- Do not add dependencies or checkpoint fields.
- Preserve loss, gradients, optimizer state, EMA state, DDP reduction order, and resume state.

---

### Task 1: Archived variance equation

**Files:**
- Modify: `tests/unit/test_metrics.py`
- Modify: `a2v2/training.py`

**Interfaces:**
- Consumes: prediction and target tensors with identical shape `[..., feature_dim]`.
- Produces: `pretraining_variance_diagnostics(predictions: Tensor, targets: Tensor) -> tuple[float, float]`, ordered as prediction then target.

- [ ] **Step 1: Write failing equation and validation tests**

Add tests that access the intended function through the imported module so the
first run fails with a missing attribute:

```python
import a2v2.training as training


def test_pretraining_variance_diagnostics_match_archived_sample_std() -> None:
    predictions = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 8.0],
    ])
    targets = torch.tensor([
        [2.0, 1.0],
        [4.0, 5.0],
        [8.0, 7.0],
    ])

    pred_var, target_var = training.pretraining_variance_diagnostics(
        predictions, targets
    )

    assert pred_var == pytest.approx(2.5275254249572754)
    assert target_var == pytest.approx(3.0550506114959717)


def test_pretraining_variance_diagnostics_reject_bad_shape_or_count() -> None:
    with pytest.raises(ValueError, match="same shape"):
        training.pretraining_variance_diagnostics(
            torch.zeros(3, 2), torch.zeros(3, 3)
        )
    with pytest.raises(ValueError, match="at least two"):
        training.pretraining_variance_diagnostics(
            torch.zeros(1, 2), torch.zeros(1, 2)
        )
```

The expected values must come from a hand calculation or an independent one-off
PyTorch expression before the production helper exists.

- [ ] **Step 2: Run the tests and verify the missing-feature failure**

Run:

```bash
pytest -q \
  tests/unit/test_metrics.py::test_pretraining_variance_diagnostics_match_archived_sample_std \
  tests/unit/test_metrics.py::test_pretraining_variance_diagnostics_reject_bad_shape_or_count
```

Expected: FAIL because `a2v2.training` has no
`pretraining_variance_diagnostics`.

- [ ] **Step 3: Implement packed distributed sufficient statistics**

Add the helper near the metric section in `a2v2/training.py`:

```python
@torch.no_grad()
def pretraining_variance_diagnostics(
    predictions: Tensor,
    targets: Tensor,
) -> tuple[float, float]:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.ndim < 2:
        raise ValueError("diagnostic tensors need a feature dimension")
    predictions = predictions.detach().reshape(-1, predictions.shape[-1]).float()
    targets = targets.detach().reshape(-1, targets.shape[-1]).float()

    count = predictions.new_tensor(float(predictions.shape[0]))
    statistics = torch.cat((
        count.view(1),
        predictions.sum(0),
        predictions.square().sum(0),
        targets.sum(0),
        targets.square().sum(0),
    ))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(statistics)

    feature_dim = predictions.shape[-1]
    count = statistics[0]
    if count < 2:
        raise ValueError("variance diagnostics require at least two vectors")
    offset = 1
    pred_sum = statistics[offset:offset + feature_dim]
    offset += feature_dim
    pred_square_sum = statistics[offset:offset + feature_dim]
    offset += feature_dim
    target_sum = statistics[offset:offset + feature_dim]
    offset += feature_dim
    target_square_sum = statistics[offset:offset + feature_dim]

    def archived_value(total: Tensor, square_total: Tensor) -> float:
        variance = (
            square_total / (count - 1)
            - total.square() / (count * (count - 1))
        )
        return float(torch.sqrt(variance + 1e-6).mean())

    return (
        archived_value(pred_sum, pred_square_sum),
        archived_value(target_sum, target_square_sum),
    )
```

Use the project's mathematical and conceptual inline-comment style around the
packed tensor and formula.

- [ ] **Step 4: Run focused unit tests**

Run:

```bash
pytest -q tests/unit/test_metrics.py
```

Expected: all metric tests PASS.

### Task 2: Engine aggregation and result contract

**Files:**
- Modify: `tests/integration/test_pretraining_step.py`
- Modify: `tests/gpu/test_cuda_training.py`
- Modify: `tests/gpu/nccl_resume.py`
- Modify: `a2v2/training.py`

**Interfaces:**
- Consumes: `PretrainingOutput.predictions` and `.targets` when both exist.
- Produces: `UpdateResult.pred_var: float | None` and `UpdateResult.target_var: float | None`.

- [ ] **Step 1: Write a failing engine-result test**

Construct a real `TrainingEngine` with the tiny pretraining model and two
microbatches. Assert that one successful result has finite positive
`pred_var` and `target_var`. Run a real fine-tuning step and assert both fields
are `None`. The pretraining assertion catches an engine that forgets to inspect
the forward output, while the fine-tuning assertion protects the optional
contract.

- [ ] **Step 2: Verify the result fields are absent**

Run:

```bash
pytest -q \
  tests/integration/test_pretraining_step.py \
  tests/integration/test_finetuning_step.py
```

Expected: FAIL because `UpdateResult` has no `pred_var` or `target_var`.

- [ ] **Step 3: Add optional fields and microbatch aggregation**

Extend the dataclass:

```python
@dataclass(frozen=True)
class UpdateResult:
    loss: float
    sample_size: int
    gradient_norm: float
    learning_rate: float
    update: int
    skipped: bool
    pred_var: float | None = None
    target_var: float | None = None
```

Within `TrainingEngine.step`, call
`pretraining_variance_diagnostics(output.predictions, output.targets)` only
when both attributes exist. Sum the returned values and divide by the number of
diagnostic-bearing microbatches for both successful and AMP-skipped results.

- [ ] **Step 4: Extend GPU result comparisons**

Add `pred_var` and `target_var` to the explicit result snapshot in
`tests/gpu/nccl_resume.py`. In the CPU/CUDA pretraining test, compare diagnostic
values with a tolerance appropriate to FP32 reduced statistics. These tests
will skip on the CPU host and run on the A100 host.

- [ ] **Step 5: Run engine, integration, and resume tests**

Run:

```bash
pytest -q \
  tests/integration/test_pretraining_step.py \
  tests/integration/test_finetuning_step.py \
  tests/integration/test_resume.py
```

Expected: all selected tests PASS.

### Task 3: JSON update and terminal logging

**Files:**
- Modify: `tests/integration/test_cli.py`
- Modify: `a2v2/workflows.py`

**Interfaces:**
- Consumes: optional variance fields from `UpdateResult`.
- Produces: JSON numeric keys `pred_var` and `target_var` in pretraining update records and the terminal `training_summary`.

- [ ] **Step 1: Write a failing CLI logging test**

Run the existing CPU smoke pretraining configuration for one update and parse
each stdout line with `json.loads`. Select the ordinary update record and the
`training_summary` record. Require both keys in both records, require finite
positive values, and require fine-tuning update records to omit them.

- [ ] **Step 2: Verify JSON keys are missing**

Run:

```bash
pytest -q tests/integration/test_cli.py::test_cli_logs_pretraining_variance_diagnostics
```

Expected: FAIL because update JSON lacks `pred_var` and `target_var`.

- [ ] **Step 3: Add conditional update fields and terminal retention**

Change `record_update` to create a dictionary first, then add the variance
keys when `result.pred_var` and `result.target_var` are not `None`. Track the
last `UpdateResult` in `run_training` and add the same keys to
`training_summary` for a pretraining run.

Flush the update print so monitoring processes receive each record without
waiting for a full stdout buffer.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest -q tests/integration/test_cli.py
```

Expected: all CLI tests PASS.

### Task 4: Research documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/code-guide.md`
- Modify: `tests/gpu/nccl_resume.py`

**Interfaces:**
- Consumes: completed logging behavior.
- Produces: researcher guidance and durable distributed comparison coverage.

- [ ] **Step 1: Document the formula and interpretation**

Add a compact pretraining-diagnostics subsection to the README and code guide.
State that both historical `*_var` keys contain mean feature-wise sample
standard deviation. Explain that values near zero indicate representation
collapse and that researchers should inspect the trajectory with loss, EMA
decay, gradient norm, and AMP scale.

- [ ] **Step 2: Check prose and source consistency**

Run:

```bash
rg -n "pred_var|target_var|sample standard deviation" \
  README.md docs/code-guide.md a2v2 tests
```

Inspect each match for consistent naming and no claim that the scalar is raw
variance.

- [ ] **Step 3: Run syntax and focused verification**

Run:

```bash
python -m py_compile a2v2/training.py a2v2/workflows.py
pytest -q \
  tests/unit/test_metrics.py \
  tests/integration/test_pretraining_step.py \
  tests/integration/test_finetuning_step.py \
  tests/integration/test_cli.py \
  tests/integration/test_resume.py
```

Expected: all selected tests PASS.

- [ ] **Step 4: Run the complete CPU suite**

Run:

```bash
pytest -q
```

Expected: all CPU tests PASS and GPU-marked tests SKIP on the CPU host.

- [ ] **Step 5: Audit the final requirements**

Confirm that the implementation:

- reproduces the archived formula and key names;
- reduces sufficient statistics before standard deviation;
- averages across microbatches;
- omits variance keys for fine-tuning;
- changes no checkpoint field; and
- leaves the full suite green.

This standalone copy has no Git metadata, so no commit step applies.
