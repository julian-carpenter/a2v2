# Distributed Validation Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let eight-rank MeerKAT fine-tuning survive its long rank-zero validation, resume an interrupted reproduction from the newest complete fine-tuning checkpoint, and retain enough CPU parallelism for validation to finish within the approved two-hour acceptance window.

**Architecture:** Keep DDP training collectives on the default NCCL group and send low-volume validation decisions through the existing auxiliary Gloo group with a two-hour timeout. Extend checkpoint preflight with an expected-stage check, then let the shell driver prefer `checkpoint_last.pt` or select the newest periodic recovery artifact and preflight it before torchrun. Give each distributed training rank an explicit eight-thread PyTorch intra-op budget so torchrun does not silently force validation to one CPU thread.

**Tech Stack:** Python 3.12, PyTorch distributed NCCL/Gloo, pytest, Bash, native A2V2 checkpoints.

## Global Constraints

- Preserve single-rank validation and all current metric calculations.
- Keep gradient and model collectives on NCCL.
- Use an explicit two-hour timeout only for the auxiliary control group.
- Preserve distributed RNG checkpoint gathering through that group.
- Prefer `checkpoint_last.pt`; otherwise choose the newest best, update, or epoch checkpoint.
- Reject recovery checkpoints with the wrong stage or incomplete eight-rank training state before torchrun.
- Do not change optimizer settings, validation cadence, metric definitions, or reproduction batch geometry.
- Resolve `A2V2_OMP_NUM_THREADS` to a positive integer with default `8`, store it internally as `TRAIN_OMP_NUM_THREADS`, and pass it only to distributed pretraining and fine-tuning launches.
- Keep the default host budget at 64 PyTorch intra-op threads across eight ranks; together with the configured 160 data-loader workers, the conservative upper bound is 224 threads on this 256-logical-CPU host.
- Prefix repository shell commands with `rtk` as required by `AGENTS.md`.
- Do not launch the full 30,000-update fine-tuning recipe during verification.
- Hard-cap the amended production acceptance run at 7,200 seconds and do not extend it without fresh user approval.

---

### Task 1: Route validation decisions through the timed CPU control group

**Files:**
- Modify: `tests/integration/test_cli.py:24-58`
- Modify: `a2v2/workflows.py:1401-1409`
- Modify: `a2v2/workflows.py:1918-1960`

**Interfaces:**
- Consumes: `_checkpoint_process_group(device: torch.device, world_size: int) -> dist.ProcessGroup | None`
- Produces: `_synchronize_validation_decision(current: float, improved: bool, *, world_size: int, group: dist.ProcessGroup | None) -> tuple[float, bool]`
- Produces: a CUDA auxiliary group created with `dist.new_group(backend="gloo", timeout=timedelta(hours=2))`

- [ ] **Step 1: Add failing process-group and decision-synchronization tests**

Extend `tests/integration/test_cli.py` with:

```python
def test_checkpoint_process_group_uses_timed_gloo_for_distributed_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_group = object()

    def fake_new_group(*, backend: str, timeout: timedelta) -> object:
        assert backend == "gloo"
        assert timeout == timedelta(hours=2)
        return selected_group

    monkeypatch.setattr(workflows.dist, "new_group", fake_new_group)

    assert workflows._checkpoint_process_group(torch.device("cuda", 0), 8) is selected_group


def test_validation_decision_uses_cpu_control_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_group = object()

    def fake_broadcast(tensor: torch.Tensor, *, src: int, group: object) -> None:
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64
        assert src == 0
        assert group is selected_group
        tensor.copy_(torch.tensor([0.75, 1.0], dtype=torch.float64))

    monkeypatch.setattr(workflows.dist, "broadcast", fake_broadcast)

    assert workflows._synchronize_validation_decision(
        0.0,
        False,
        world_size=8,
        group=selected_group,  # type: ignore[arg-type]
    ) == (0.75, True)
```

Import `timedelta` from `datetime` in the test module. Update the existing skip test's fake `new_group` signature to accept `timeout` so it reports an unexpected group without raising a signature error.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
rtk python -m pytest -q \
  tests/integration/test_cli.py::test_checkpoint_process_group_uses_timed_gloo_for_distributed_cuda \
  tests/integration/test_cli.py::test_validation_decision_uses_cpu_control_group
```

Expected: the group test fails because `timeout` is absent, and the decision test fails because `_synchronize_validation_decision` does not exist.

- [ ] **Step 3: Implement the minimal control-plane change**

In `a2v2/workflows.py`, broaden the existing group docstring and set its timeout:

```python
def _checkpoint_process_group(
    device: torch.device,
    world_size: int,
) -> dist.ProcessGroup | None:
    """Create a timed CPU control plane for checkpoint and validation data."""

    if world_size > 1 and device.type == "cuda":
        return dist.new_group(backend="gloo", timeout=timedelta(hours=2))
    return None
```

Add the decision helper beside the process-group helper:

```python
def _synchronize_validation_decision(
    current: float,
    improved: bool,
    *,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> tuple[float, bool]:
    """Broadcast rank zero's validation decision over the CPU control plane."""

    if world_size <= 1:
        return current, improved
    decision = torch.tensor([current, float(improved)], dtype=torch.float64)
    dist.broadcast(decision, src=0, group=group)
    return float(decision[0]), bool(decision[1])
```

Replace the CUDA tensor broadcast in `validate_and_checkpoint` with:

```python
current, improved = _synchronize_validation_decision(
    current,
    improved,
    world_size=world_size,
    group=checkpoint_group,
)
```

- [ ] **Step 4: Run focused and nearby lifecycle tests and verify GREEN**

Run:

```bash
rtk python -m pytest -q tests/integration/test_cli.py
```

Expected: all CLI integration tests pass, including process-group cleanup and CPU/single-rank behavior.

- [ ] **Step 5: Commit the distributed control fix**

```bash
rtk git add a2v2/workflows.py tests/integration/test_cli.py
rtk git commit -m "fix: isolate long validation from NCCL timeout"
```

---

### Task 2: Require the expected training stage during checkpoint preflight

**Files:**
- Modify: `tests/integration/test_reproduction_driver.py:91-130`
- Modify: `scripts/check_reproduction_environment.py:91-122`
- Modify: `scripts/check_reproduction_environment.py:144-179`

**Interfaces:**
- Consumes: `check_training_checkpoint(path: Path, expected_world_size: int, expected_stage: str | None = None)`
- Produces: CLI option `--expected-stage {pretrain,finetune}`

- [ ] **Step 1: Add a failing wrong-stage preflight test**

Change the test checkpoint helper call to pass `expected_stage="pretrain"`, then add this assertion before mutating the optimizer payload:

```python
with pytest.raises(RuntimeError, match=r"checkpoint stage is pretrain; expected finetune"):
    preflight.check_training_checkpoint(
        checkpoint,
        expected_world_size=8,
        expected_stage="finetune",
    )
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
rtk python -m pytest -q \
  tests/integration/test_reproduction_driver.py::test_checkpoint_preflight_requires_distributed_resume_state
```

Expected: FAIL because `check_training_checkpoint` does not accept `expected_stage`.

- [ ] **Step 3: Implement stage validation and CLI plumbing**

Update the helper signature and validate the stored stage immediately after loading:

```python
def check_training_checkpoint(
    path: Path,
    expected_world_size: int,
    expected_stage: str | None = None,
) -> dict[str, object]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    stage = str(checkpoint["stage"])
    if expected_stage is not None and stage != expected_stage:
        raise RuntimeError(
            f"checkpoint stage is {stage}; expected {expected_stage}: {path.resolve()}"
        )
```

Return `stage` in the report, add the parser option, and pass it from `main`:

```python
parser.add_argument("--expected-stage", choices=("pretrain", "finetune"))

report["checkpoint"] = check_training_checkpoint(
    arguments.checkpoint,
    arguments.expected_gpus,
    arguments.expected_stage,
)
```

- [ ] **Step 4: Run the preflight module tests and verify GREEN**

Run:

```bash
rtk python -m pytest -q tests/integration/test_reproduction_driver.py -k preflight
```

Expected: every preflight test passes.

- [ ] **Step 5: Commit the checkpoint-stage guard**

```bash
rtk git add scripts/check_reproduction_environment.py tests/integration/test_reproduction_driver.py
rtk git commit -m "fix: validate reproduction checkpoint stage"
```

---

### Task 3: Recover fine-tuning from periodic checkpoints

**Files:**
- Modify: `tests/integration/test_reproduction_driver.py:226-330`
- Modify: `scripts/reproduce_meerkat_paper.sh:210-216`
- Modify: `scripts/reproduce_meerkat_paper.sh:313-316`
- Modify: `scripts/reproduce_meerkat_paper.sh:355-382`

**Interfaces:**
- Consumes: `FINETUNE_LAST_CHECKPOINT`, `FINETUNE_BEST_CHECKPOINT`, periodic checkpoint filename conventions, and `PREFLIGHT_COMMAND`
- Produces: shell variable `FINETUNE_RESUME_CHECKPOINT`
- Produces: `FINETUNE_CHECKPOINT_PREFLIGHT_COMMAND` when a resume point exists

- [ ] **Step 1: Add failing dry-run recovery tests**

Add two tests to `tests/integration/test_reproduction_driver.py`:

```python
def test_dry_run_recovers_from_newest_periodic_finetune_checkpoint(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    _placeholder_manifests(manifests)
    finetune = output / "finetune"
    finetune.mkdir(parents=True)
    best = finetune / "checkpoint_best.pt"
    update = finetune / "checkpoint_9000.pt"
    epoch = finetune / "checkpoint_epoch_20.pt"
    best.touch()
    update.touch()
    epoch.touch()
    os.utime(best, (1, 1))
    os.utime(update, (2, 2))
    os.utime(epoch, (3, 3))

    completed = _run_driver(str(manifests), str(output), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    rendered = completed.stdout.replace("\\", "")
    assert f"--checkpoint {epoch} --expected-stage finetune" in rendered
    assert f"--resume {epoch}" in rendered


def test_dry_run_prefers_finetune_last_over_newer_periodic_checkpoint(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    _placeholder_manifests(manifests)
    finetune = output / "finetune"
    finetune.mkdir(parents=True)
    last = finetune / "checkpoint_last.pt"
    periodic = finetune / "checkpoint_epoch_20.pt"
    last.touch()
    periodic.touch()
    os.utime(last, (1, 1))
    os.utime(periodic, (2, 2))

    completed = _run_driver(str(manifests), str(output), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    rendered = completed.stdout.replace("\\", "")
    assert f"--resume {last}" in rendered
    assert f"--resume {periodic}" not in rendered
```

Also extend `test_dry_run_is_one_fold_and_uses_all_eight_gpus` to assert the pretraining checkpoint check renders `--expected-stage pretrain`.

- [ ] **Step 2: Run both recovery tests and verify RED**

Run:

```bash
rtk python -m pytest -q \
  tests/integration/test_reproduction_driver.py::test_dry_run_recovers_from_newest_periodic_finetune_checkpoint \
  tests/integration/test_reproduction_driver.py::test_dry_run_prefers_finetune_last_over_newer_periodic_checkpoint
```

Expected: the periodic fallback test fails because the driver emits no fine-tuning `--resume`; the last-checkpoint test lacks the new fine-tuning checkpoint preflight.

- [ ] **Step 3: Implement deterministic checkpoint selection**

After defining the fine-tuning paths, select the recovery point without subprocesses:

```bash
FINETUNE_RESUME_CHECKPOINT=""
if [[ -f "${FINETUNE_LAST_CHECKPOINT}" ]]; then
    FINETUNE_RESUME_CHECKPOINT="${FINETUNE_LAST_CHECKPOINT}"
else
    for candidate in \
        "${FINETUNE_BEST_CHECKPOINT}" \
        "${FINETUNE_DIR}"/checkpoint_[0-9]*.pt \
        "${FINETUNE_DIR}"/checkpoint_epoch_*.pt
    do
        [[ -f "${candidate}" ]] || continue
        if [[ -z "${FINETUNE_RESUME_CHECKPOINT}" \
            || "${candidate}" -nt "${FINETUNE_RESUME_CHECKPOINT}" ]]
        then
            FINETUNE_RESUME_CHECKPOINT="${candidate}"
        fi
    done
fi
```

Add `--expected-stage pretrain` to `CHECKPOINT_PREFLIGHT_COMMAND`. After assembling `FINETUNE_COMMAND`, add the selected resume and its preflight command:

```bash
if [[ -n "${FINETUNE_RESUME_CHECKPOINT}" ]]; then
    FINETUNE_COMMAND+=(--resume "${FINETUNE_RESUME_CHECKPOINT}")
    FINETUNE_CHECKPOINT_PREFLIGHT_COMMAND=(
        "${PREFLIGHT_COMMAND[@]}"
        --checkpoint "${FINETUNE_RESUME_CHECKPOINT}"
        --expected-stage finetune
    )
fi
```

In dry-run mode, print `FINETUNE_CHECKPOINT_PREFLIGHT_COMMAND` before `FINETUNE_COMMAND` when it exists. In real mode, call:

```bash
run_logged \
    finetuning-checkpoint-validation \
    "${FINETUNE_DIR}/train.log" \
    "${FINETUNE_CHECKPOINT_PREFLIGHT_COMMAND[@]}"
```

before the fine-tuning launch when `FINETUNE_RESUME_CHECKPOINT` is non-empty.

- [ ] **Step 4: Run all reproduction-driver tests and verify GREEN**

Run:

```bash
rtk python -m pytest -q tests/integration/test_reproduction_driver.py
```

Expected: the dry-run command graph, fake real-mode orchestration, checkpoint-stage checks, and evaluator tests all pass.

- [ ] **Step 5: Commit the recovery behavior**

```bash
rtk git add scripts/reproduce_meerkat_paper.sh tests/integration/test_reproduction_driver.py
rtk git commit -m "fix: resume periodic finetuning checkpoints"
```

---

### Task 4: Document the control plane and recovery rules

**Files:**
- Modify: `docs/reproducing-paper.md:29-37`
- Modify: `docs/reproducing-paper.md:101-117`
- Modify: `docs/code-guide.md:315-327`
- Modify: `docs/code-guide.md:429-440`

**Interfaces:**
- Consumes: the behavior implemented in Tasks 1 through 3
- Produces: operator guidance for long validation and automatic resume selection

- [ ] **Step 1: Update the reproduction guide**

Replace the resume paragraph with:

```markdown
The script validates and resumes pretraining from `checkpoint_last.pt`. For
fine-tuning it prefers `checkpoint_last.pt`; when that file is absent after an
interruption, it selects the newest `checkpoint_best.pt`,
`checkpoint_<update>.pt`, or `checkpoint_epoch_<epoch>.pt`. Before torchrun,
preflight requires the selected file to match the training stage and contain
optimizer, scheduler, and one RNG state for each of the eight ranks. Do not
change world size or batch variables between an interrupted run and its resume:
sampler and optimizer state belong to the stored topology.
```

Add this paragraph to the validation description:

```markdown
Scheduled training validation runs on rank zero. CUDA workers exchange the
result through a CPU Gloo control group with a two-hour timeout, so the other
ranks do not hold an NCCL collective while rank zero evaluates the full split.
```

- [ ] **Step 2: Update the code guide**

Replace the Gloo-group paragraph with:

```markdown
`gather_rank_rng_states` serializes each tensor-bearing state to bytes and
requires exactly one payload in rank order. CUDA training creates a dedicated
Gloo process group with a two-hour timeout for checkpoint and validation
control data. Serialized CPU checkpoint metadata stays off the CUDA allocator,
and CPU validation decisions let nonzero ranks wait without holding a pending
NCCL collective. CPU distributed training already uses Gloo as its default
group.
```

In the lifecycle paragraph, replace both occurrences of “auxiliary checkpoint
group” with “auxiliary control group.”

- [ ] **Step 3: Check documentation and shell syntax**

Run:

```bash
rtk bash -n scripts/reproduce_meerkat_paper.sh
rtk git diff --check
```

Expected: both commands exit 0 with no diagnostics.

- [ ] **Step 4: Commit the operator documentation**

```bash
rtk git add docs/reproducing-paper.md docs/code-guide.md
rtk git commit -m "docs: explain distributed validation recovery"
```

---

### Task 5: Verify regressions and eight-GPU behavior

**Files:**
- Inspect: `experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_epoch_20.pt`
- Create at runtime: `/tmp/a2v2-validation-control-probe/`
- Update at runtime: `experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_best.pt`
- Update at runtime: `experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt`

**Interfaces:**
- Consumes: the update-8,295 fine-tuning checkpoint and `valid_0` manifest
- Produces: a synchronized update-10,000 checkpoint and validation metrics

- [ ] **Step 1: Run the focused regression suites**

```bash
rtk python -m pytest -q \
  tests/integration/test_cli.py \
  tests/integration/test_reproduction_driver.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete automated suite**

```bash
rtk python -m pytest -q
```

Expected: the suite exits 0 with no failures.

- [ ] **Step 3: Run the existing eight-rank transport and checkpoint probe**

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  tests/gpu/nccl_probe.py \
  --output-dir /tmp/a2v2-validation-control-probe
```

Expected: all eight rank reports contain `"pass": true`, NCCL tensor collectives pass, and the Gloo RNG gather round-trips.

- [ ] **Step 4: Run the bounded production resume through first validation**

Before launch, confirm all eight GPUs are free. Then run:

```bash
rtk env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  /usr/local/bin/a2v2-train \
  --config /abyss/home/ml/a2v2/configs/MeerKAT/finetune_mixup_100.yaml \
  --pretrained-checkpoint /abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/pretrain/checkpoint_last.pt \
  --resume /abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_epoch_20.pt \
  --stop-at-update 10000 \
  --override task.data=/local/datasets/MeerKAT_10s_2024-06-12/manifests \
  --override dataset.train_subset=train_0 \
  --override dataset.valid_subset=valid_0 \
  --override checkpoint.save_dir=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune \
  --override distributed_training.distributed_world_size=8 \
  --override dataset.max_tokens=960000 \
  --override optimization.update_freq=[2] \
  --device cuda
```

Expected within two hours: training resumes at update 8,295; rank 0 completes validation at update 10,000; no NCCL or Gloo watchdog fires; every rank completes the validation decision and checkpoint gathers; `checkpoint_best.pt` and `checkpoint_last.pt` load as stage `finetune`, update 10,000, world size 8.

- [ ] **Step 5: Validate the produced checkpoints and inspect the working tree**

```bash
rtk python scripts/check_reproduction_environment.py \
  --output-dir experiments/a2v2-meerkat-fold0-260728 \
  --checkpoint experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt \
  --expected-stage finetune
rtk git status --short --branch
rtk git log -5 --oneline
```

Expected: preflight reports `"pass": true`, stage `finetune`, update 10,000, and RNG world size 8; the working tree contains no uncommitted source changes.

---

## Approved CPU Threading Amendment

The first Task 5 production attempt resumed successfully and remained healthy
through more than 88 minutes of rank-asymmetric validation: no NCCL or Gloo
watchdog fired, and all nonzero ranks stayed alive in the auxiliary control
group. It reached the 7,200-second cap before rank zero completed validation.
Runtime inspection identified the separate bottleneck: torchrun automatically
set `OMP_NUM_THREADS=1`, rank zero stayed at one fully occupied CPU core, and
the same validation had previously completed in 53 minutes 47 seconds when
PyTorch could use the host's default CPU pool. That attempt advanced the
newest complete fine-tuning recovery checkpoint to update 8,710. Tasks 6 and 7
implement and verify the user-confirmed eight-thread correction; Task 7
supersedes the unfinished production-acceptance portion of Task 5.

### Task 6: Budget distributed training CPU threads explicitly

**Files:**
- Modify: `tests/integration/test_reproduction_driver.py:235-390`
- Modify: `scripts/reproduce_meerkat_paper.sh:185-430`
- Modify: `docs/reproducing-paper.md:78-103`

**Interfaces:**
- Consumes: `require_positive_integer NAME VALUE`
- Consumes: optional environment variable `A2V2_OMP_NUM_THREADS`
- Produces: readonly shell value `TRAIN_OMP_NUM_THREADS`, default `8`
- Produces: `OMP_NUM_THREADS=<resolved value>` in every distributed pretraining and fine-tuning process environment
- Produces: `omp_num_threads=<resolved value>` in `environment/run-profile.txt`

- [ ] **Step 1: Add failing default, override, validation, and provenance tests**

In `test_dry_run_is_one_fold_and_uses_all_eight_gpus`, collect only training
launch lines so the human-readable summary does not affect the count:

```python
    training_launches = [
        line.replace("\\", "")
        for line in stdout.splitlines()
        if line.startswith("Launching:") and "a2v2-train" in line
    ]
    assert len(training_launches) == 3
    assert all("OMP_NUM_THREADS=8" in line for line in training_launches)
    assert "  CPU: OMP_NUM_THREADS=8" in stdout
```

Add an override test:

```python
def test_dry_run_applies_the_requested_training_thread_budget(tmp_path: Path) -> None:
    """Apply one explicit CPU-thread budget to every distributed training rank."""
    manifests = tmp_path / "manifests"
    output = tmp_path / "experiment"
    _placeholder_manifests(manifests)
    environment = {**os.environ, "A2V2_OMP_NUM_THREADS": "4"}

    completed = _run_driver(
        str(manifests),
        str(output),
        "--dry-run",
        environment=environment,
    )

    assert completed.returncode == 0, completed.stderr
    training_launches = [
        line.replace("\\", "")
        for line in completed.stdout.splitlines()
        if line.startswith("Launching:") and "a2v2-train" in line
    ]
    assert len(training_launches) == 3
    assert all("OMP_NUM_THREADS=4" in line for line in training_launches)
    assert "  CPU: OMP_NUM_THREADS=4" in completed.stdout
```

Add validation coverage:

```python
@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_driver_rejects_an_invalid_training_thread_budget(
    tmp_path: Path,
    value: str,
) -> None:
    """Reject thread budgets that cannot produce a positive worker pool."""
    manifests = tmp_path / "manifests"
    _placeholder_manifests(manifests)
    environment = {**os.environ, "A2V2_OMP_NUM_THREADS": value}

    completed = _run_driver(
        str(manifests),
        str(tmp_path / "experiment"),
        "--dry-run",
        environment=environment,
    )

    assert completed.returncode != 0
    assert "A2V2_OMP_NUM_THREADS must be a positive integer" in completed.stderr
```

After the first successful invocation in
`test_real_driver_burns_in_resumes_and_appends_phase_logs`, add:

```python
    run_profile = (output / "environment/run-profile.txt").read_text(
        encoding="utf-8"
    )
    assert "omp_num_threads=8\n" in run_profile
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
rtk python -m pytest -q \
  tests/integration/test_reproduction_driver.py::test_dry_run_is_one_fold_and_uses_all_eight_gpus \
  tests/integration/test_reproduction_driver.py::test_dry_run_applies_the_requested_training_thread_budget \
  tests/integration/test_reproduction_driver.py::test_driver_rejects_an_invalid_training_thread_budget \
  tests/integration/test_reproduction_driver.py::test_real_driver_burns_in_resumes_and_appends_phase_logs
```

Expected: the default and override assertions fail because training launches do
not yet carry `OMP_NUM_THREADS`; the invalid values are not yet rejected; and
the run profile lacks `omp_num_threads`.

- [ ] **Step 3: Implement the validated driver setting**

Immediately after the evaluation controls, resolve the external setting into a
task-specific internal variable:

```bash
readonly TRAIN_OMP_NUM_THREADS="${A2V2_OMP_NUM_THREADS:-8}"
```

Validate it beside the existing token and worker controls:

```bash
require_positive_integer A2V2_OMP_NUM_THREADS "${TRAIN_OMP_NUM_THREADS}"
```

Add this line to the launch summary:

```bash
    "  CPU: OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS}" \
```

Add this line to `environment/run-profile.txt`:

```bash
        printf 'omp_num_threads=%s\n' "${TRAIN_OMP_NUM_THREADS}"
```

In both `PRETRAIN_COMMAND` and `FINETUNE_COMMAND`, add the environment entry
immediately after `PYTORCH_CUDA_ALLOC_CONF`:

```bash
    "OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS}"
```

Do not add it to preflight, the transport probe, or final single-GPU
evaluation. Do not call `torch.set_num_threads` from Python; the launch
environment is the single source of truth for all eight ranks.

- [ ] **Step 4: Document the operator override and host budget**

Add this row to the deployment-controls table in
`docs/reproducing-paper.md`:

```markdown
| `A2V2_OMP_NUM_THREADS` | `8` | Per-rank PyTorch CPU thread budget for distributed training and rank-zero validation |
```

After the table, add:

```markdown
The default gives the eight training ranks 64 PyTorch intra-op threads in
total. Together with the configured 20 data-loader workers per rank, the
conservative upper bound is 224 threads on the 256-logical-CPU reproduction
host. Override `A2V2_OMP_NUM_THREADS` only with a positive integer; values of
16 or more can oversubscribe this host during distributed loading.
```

- [ ] **Step 5: Run focused and complete regressions**

```bash
rtk bash -n scripts/reproduce_meerkat_paper.sh
rtk python -m pytest -q tests/integration/test_reproduction_driver.py
rtk python -m pytest -q
rtk git diff --check
```

Expected: shell syntax is valid, all driver tests pass, the complete suite has
no failures, and the diff has no whitespace errors.

- [ ] **Step 6: Commit the CPU-threading amendment**

```bash
rtk git add \
  scripts/reproduce_meerkat_paper.sh \
  tests/integration/test_reproduction_driver.py \
  docs/reproducing-paper.md
rtk git commit -m "fix: budget CPU threads for distributed validation"
```

### Task 7: Repeat the bounded eight-GPU acceptance run

**Files:**
- Inspect: `experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_epoch_20.pt`
- Create at runtime: `/tmp/a2v2-validation-control-probe-omp8/`
- Update at runtime: `experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_best.pt`
- Update at runtime: `experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt`

**Interfaces:**
- Consumes: the complete stage-`finetune`, update-8,710, world-size-eight recovery checkpoint
- Consumes: `OMP_NUM_THREADS=8` and the existing eight-A100 topology
- Produces: synchronized update-10,000 fine-tuning checkpoints and validation metrics

- [ ] **Step 1: Re-run automated and transport gates**

```bash
rtk python -m pytest -q \
  tests/integration/test_cli.py \
  tests/integration/test_reproduction_driver.py
rtk python -m pytest -q
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  tests/gpu/nccl_probe.py \
  --output-dir /tmp/a2v2-validation-control-probe-omp8
```

Expected: all test suites pass; all eight probe reports contain
`{"pass": true}`; NCCL tensor collectives and the Gloo RNG gather succeed.

- [ ] **Step 2: Preflight the exact recovery checkpoint and free GPUs**

```bash
rtk python scripts/check_reproduction_environment.py \
  --output-dir experiments/a2v2-meerkat-fold0-260728 \
  --checkpoint experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_epoch_20.pt \
  --expected-stage finetune
```

Expected: `"pass": true`, update 8,710, stage `finetune`, optimizer and
scheduler state present, RNG world size 8, and all eight A100s meet the free
memory threshold. Record the checkpoint hash and metadata before launch; do
not delete, restore, or manually rewrite it.

- [ ] **Step 3: Resume through update-10,000 validation with a hard cap**

Run this command from the root controller, which holds the user's explicit
authorization for the bounded eight-GPU job:

```bash
rtk timeout --signal=TERM 7200 env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 \
  python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  /usr/local/bin/a2v2-train \
  --config /abyss/home/ml/a2v2/configs/MeerKAT/finetune_mixup_100.yaml \
  --pretrained-checkpoint /abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/pretrain/checkpoint_last.pt \
  --resume /abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_epoch_20.pt \
  --stop-at-update 10000 \
  --override task.data=/local/datasets/MeerKAT_10s_2024-06-12/manifests \
  --override dataset.train_subset=train_0 \
  --override dataset.valid_subset=valid_0 \
  --override checkpoint.save_dir=/abyss/home/ml/a2v2/experiments/a2v2-meerkat-fold0-260728/finetune \
  --override distributed_training.distributed_world_size=8 \
  --override dataset.max_tokens=960000 \
  --override optimization.update_freq=[2] \
  --device cuda
```

Expected within 7,200 seconds: the first completed update is greater than
8,710; training reaches update 10,000; rank zero completes `valid_0`; no NCCL
or Gloo watchdog fires; all ranks leave the validation decision and checkpoint
gathers; the command exits 0. Stop at the cap and report partial evidence if it
does not finish; do not extend the run without asking the user again.

- [ ] **Step 4: Validate the production artifacts and repository state**

```bash
rtk python scripts/check_reproduction_environment.py \
  --output-dir experiments/a2v2-meerkat-fold0-260728 \
  --checkpoint experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_best.pt \
  --expected-stage finetune
rtk python scripts/check_reproduction_environment.py \
  --output-dir experiments/a2v2-meerkat-fold0-260728 \
  --checkpoint experiments/a2v2-meerkat-fold0-260728/finetune/checkpoint_last.pt \
  --expected-stage finetune
rtk git status --short --branch
rtk git log -10 --oneline
```

Expected: both checkpoints load as stage `finetune`, update 10,000, world size
8, with complete optimizer, scheduler, sampler, scaler, and per-rank RNG state;
the source working tree is clean and `main` contains the amendment commit.
