# Distributed Checkpoint Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep distributed RNG checkpoint traffic off NCCL, release process groups after errors, and make the eight-A100 driver prove its environment and checkpoint path before the long run.

**Architecture:** CUDA workers will keep NCCL as the tensor data plane and create a Gloo group for serialized RNG bytes. A public training wrapper will own process-group cleanup. A focused Python preflight helper and the shell driver will check the runtime, hardware, disk, collectives, production burn-in checkpoint, and resume path.

**Tech Stack:** Python 3.10+, PyTorch distributed NCCL/Gloo, Bash, pytest, eight NVIDIA A100 40 GB GPUs.

## Global Constraints

- Preserve model equations, optimizer and scheduler arithmetic, RNG values, sampler position, checkpoint schema, recipe values, and eight-rank batch topology.
- Gather one RNG payload per rank in global-rank order.
- Use NCCL for CUDA tensor collectives and Gloo for serialized RNG objects.
- Destroy only process groups that A2V2 initialized.
- Keep existing checkpoints readable and exact-resume compatible.
- Ask the user before launching any command expected to run longer than two hours.

---

### Task 1: CPU RNG gather contract

**Files:**
- Modify: `tests/unit/test_checkpoint.py`
- Modify: `a2v2/training.py`

**Interfaces:**
- Produces: `gather_rank_rng_states(state: Mapping[str, object], *, world_size: int, rank: int, group: dist.ProcessGroup | None) -> dict[str, object] | None`

- [ ] **Step 1: Write failing gather-result tests**

Add tests that replace `dist.gather_object` with a deterministic fake:

```python
import a2v2.training as training


def test_rank_rng_gather_uses_requested_group_and_preserves_rank_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = []
    for seed in (17, 23):
        torch.manual_seed(seed)
        states.append(capture_rng_state())
    encoded = [serialize_rng_state(state) for state in states]
    sentinel_group = object()

    def gather(obj, output, *, dst, group):
        assert isinstance(obj, bytes)
        assert dst == 0
        assert group is sentinel_group
        assert output is not None
        output[:] = encoded

    monkeypatch.setattr(torch.distributed, "gather_object", gather)
    result = training.gather_rank_rng_states(
        states[0],
        world_size=2,
        rank=0,
        group=sentinel_group,
    )

    assert result is not None
    assert result["world_size"] == 2
    ranked = result["by_rank"]
    assert isinstance(ranked, list)
    assert torch.equal(ranked[0]["torch"], states[0]["torch"])
    assert torch.equal(ranked[1]["torch"], states[1]["torch"])


@pytest.mark.parametrize("gathered", [[b"valid", None], [b"valid"]])
def test_rank_rng_gather_rejects_incomplete_payloads(
    monkeypatch: pytest.MonkeyPatch,
    gathered: list[object],
) -> None:
    def gather(obj, output, *, dst, group):
        assert output is not None
        output[:] = gathered

    monkeypatch.setattr(torch.distributed, "gather_object", gather)
    with pytest.raises(CheckpointError, match="one byte payload per rank"):
        training.gather_rank_rng_states(
            capture_rng_state(),
            world_size=2,
            rank=0,
            group=object(),
        )
```

- [ ] **Step 2: Run the tests and verify the missing-helper failure**

Run:

```bash
python -m pytest -q \
  tests/unit/test_checkpoint.py::test_rank_rng_gather_uses_requested_group_and_preserves_rank_order \
  tests/unit/test_checkpoint.py::test_rank_rng_gather_rejects_incomplete_payloads
```

Expected: FAIL because `a2v2.training` lacks `gather_rank_rng_states`.

- [ ] **Step 3: Implement validated rank RNG gathering**

Add beside the RNG serialization functions:

```python
def gather_rank_rng_states(
    state: Mapping[str, object],
    *,
    world_size: int,
    rank: int,
    group: dist.ProcessGroup | None,
) -> dict[str, object] | None:
    encoded = serialize_rng_state(state)
    gathered: list[object] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(encoded, gathered, dst=0, group=group)
    if rank != 0:
        return None
    if (
        gathered is None
        or len(gathered) != world_size
        or any(not isinstance(item, bytes) for item in gathered)
    ):
        raise CheckpointError(
            f"checkpoint RNG gather requires one byte payload per rank; "
            f"expected {world_size}"
        )
    return {
        "world_size": world_size,
        "by_rank": [deserialize_rng_state(item) for item in gathered],
    }
```

- [ ] **Step 4: Run the checkpoint unit tests**

Run:

```bash
python -m pytest -q tests/unit/test_checkpoint.py
```

Expected: PASS.

### Task 2: Gloo checkpoint group and exception cleanup

**Files:**
- Modify: `tests/integration/test_cli.py`
- Modify: `a2v2/workflows.py`

**Interfaces:**
- Produces: `_checkpoint_process_group(device: torch.device, world_size: int) -> dist.ProcessGroup | None`
- Splits: `run_training(config: Animal2VecConfig, *, device_name: str, resume_path: Path | None, pretrained_checkpoint: Path | None, stop_at_update: int | None = None) -> Path` and `_run_training` with the same typed parameters
- Consumes: `gather_rank_rng_states` from Task 1

- [ ] **Step 1: Write failing group-selection and cleanup tests**

Add:

```python
import pytest
import a2v2.workflows as workflows


def test_cuda_checkpoint_objects_use_a_gloo_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls = []

    def new_group(*, backend):
        calls.append(backend)
        return sentinel

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    assert workflows._checkpoint_process_group(torch.device("cuda", 3), 8) is sentinel
    assert calls == ["gloo"]
    assert workflows._checkpoint_process_group(torch.device("cpu"), 8) is None
    assert workflows._checkpoint_process_group(torch.device("cuda", 0), 1) is None


def test_run_training_destroys_only_a_group_it_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = False
    destroyed = 0

    def is_initialized() -> bool:
        return initialized

    def fail(*args, **kwargs):
        nonlocal initialized
        initialized = True
        raise RuntimeError("forced failure")

    def destroy() -> None:
        nonlocal initialized, destroyed
        initialized = False
        destroyed += 1

    monkeypatch.setattr(torch.distributed, "is_initialized", is_initialized)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    monkeypatch.setattr(workflows, "_run_training", fail)

    with pytest.raises(RuntimeError, match="forced failure"):
        workflows.run_training(
            load_config(ROOT / "configs/cpu_smoke_pretraining.yaml"),
            device_name="cpu",
            resume_path=None,
            pretrained_checkpoint=None,
        )
    assert destroyed == 1


def test_run_training_preserves_a_caller_owned_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda: pytest.fail("destroyed caller-owned group"),
    )
    monkeypatch.setattr(
        workflows,
        "_run_training",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced failure")),
    )
    with pytest.raises(RuntimeError, match="forced failure"):
        workflows.run_training(
            load_config(ROOT / "configs/cpu_smoke_pretraining.yaml"),
            device_name="cpu",
            resume_path=None,
            pretrained_checkpoint=None,
        )
```

- [ ] **Step 2: Run the tests and verify the expected failures**

Run:

```bash
python -m pytest -q \
  tests/integration/test_cli.py::test_cuda_checkpoint_objects_use_a_gloo_group \
  tests/integration/test_cli.py::test_run_training_destroys_only_a_group_it_created \
  tests/integration/test_cli.py::test_run_training_preserves_a_caller_owned_group
```

Expected: FAIL because the group selector and private implementation do not
exist.

- [ ] **Step 3: Implement group selection and checkpoint routing**

Import `gather_rank_rng_states`. Add:

```python
def _checkpoint_process_group(
    device: torch.device,
    world_size: int,
) -> dist.ProcessGroup | None:
    if world_size <= 1 or device.type != "cuda":
        return None
    return dist.new_group(backend="gloo")
```

Create the group once after world-size validation. In `write_checkpoint`,
replace the local serialization and default-group `gather_object` block with:

```python
ranked_rng = gather_rank_rng_states(
    payload["rng_state"],
    world_size=world_size,
    rank=rank,
    group=checkpoint_group,
)
if rank == 0:
    if ranked_rng is None:
        raise CheckpointError("rank zero did not receive distributed RNG state")
    payload["rng_state"] = ranked_rng
```

- [ ] **Step 4: Add process-group ownership cleanup**

Rename the current body to `_run_training`. Add the public wrapper:

```python
def run_training(
    config: Animal2VecConfig,
    *,
    device_name: str,
    resume_path: Path | None,
    pretrained_checkpoint: Path | None,
    stop_at_update: int | None = None,
) -> Path:
    caller_owned_group = dist.is_initialized()
    try:
        return _run_training(
            config,
            device_name=device_name,
            resume_path=resume_path,
            pretrained_checkpoint=pretrained_checkpoint,
            stop_at_update=stop_at_update,
        )
    finally:
        if not caller_owned_group and dist.is_initialized():
            dist.destroy_process_group()
```

Remove the normal-path destroy at the bottom of `_run_training`; keep
world-size mismatch cleanup safe for a group created in that call.

- [ ] **Step 5: Run CLI and resume tests**

Run:

```bash
python -m pytest -q tests/integration/test_cli.py tests/integration/test_resume.py
```

Expected: PASS.

### Task 3: Testable reproduction environment preflight

**Files:**
- Create: `scripts/check_reproduction_environment.py`
- Modify: `tests/integration/test_reproduction_driver.py`

**Interfaces:**
- Produces: `check_cuda_devices(expected_count: int, min_total_bytes: int, min_free_bytes: int) -> list[dict[str, object]]`
- Produces: `check_output_space(path: Path, min_free_bytes: int) -> dict[str, object]`
- Produces: `check_training_checkpoint(path: Path, expected_world_size: int) -> dict[str, object]`
- CLI prints one JSON report and exits nonzero on a failed check

- [ ] **Step 1: Write failing helper tests**

Load the script with `importlib.util.spec_from_file_location`. Add tests that:

- supply a fake `torch.cuda` facade with eight A100 devices and exact free-byte
  values, then assert the returned metadata;
- change one device to 37 GiB free and require an error naming that index;
- monkeypatch `shutil.disk_usage` below 64 GiB and require an output-space
  error; and
- create a native training checkpoint whose rank RNG world size is eight,
  require a passing checkpoint report, then set `optimizer=None` and require a
  resume-state error.

- [ ] **Step 2: Run the tests and verify the missing-script failure**

Run:

```bash
python -m pytest -q \
  tests/integration/test_reproduction_driver.py -k "environment or checkpoint_preflight"
```

Expected: FAIL because `scripts/check_reproduction_environment.py` does not
exist.

- [ ] **Step 3: Implement the helper**

Use `torch.cuda.get_device_properties`, `torch.cuda.mem_get_info`,
`shutil.disk_usage`, and `a2v2.training.load_checkpoint`. Require:

```text
8 visible devices
"A100" in every device name
total_memory >= 39 * 1024**3
free_memory >= 38 * 1024**3
disk free >= 64 * 1024**3
training checkpoint optimizer/scheduler present
checkpoint rng_state.world_size == 8
```

Support:

```text
--output-dir PATH
--expected-gpus 8
--min-total-gib 39
--min-free-gib 38
--min-disk-gib 64
--checkpoint PATH
```

An omitted checkpoint runs only runtime, CUDA, and disk checks.

- [ ] **Step 4: Run the helper tests**

Run:

```bash
python -m pytest -q tests/integration/test_reproduction_driver.py
```

Expected: PASS for the new helper tests; existing dry-run assertions may still
fail until Task 4 updates the command graph.

### Task 4: Driver probe, burn-in, resume, and append-only logs

**Files:**
- Modify: `scripts/reproduce_meerkat_paper.sh`
- Modify: `tests/integration/test_reproduction_driver.py`

**Interfaces:**
- Default launcher: `"$A2V2_PYTHON" -m torch.distributed.run`
- Override launcher: one executable from `A2V2_TORCHRUN`
- Preflight probe: `tests/gpu/nccl_probe.py`
- Fresh pretraining: burn-in update 1, validate checkpoint, full resume

- [ ] **Step 1: Extend the dry-run command test**

Require the dry-run output to contain:

```text
check_reproduction_environment.py
tests/gpu/nccl_probe.py
--nproc-per-node=8
--stop-at-update 1
--resume <output>/pretrain/checkpoint_last.pt
python -m torch.distributed.run
```

Require four eight-rank commands: the probe, pretraining burn-in, full
pretraining resume, and fine-tuning.

- [ ] **Step 2: Add a real-mode fake-runtime test**

Create executable fake `python` and `nvidia-smi` programs under `tmp_path/bin`.
Use this fake Python body:

```python
#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
record = Path(os.environ["A2V2_FAKE_INVOCATIONS"])
with record.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(arguments) + "\n")

if arguments == ["--version"]:
    print("Python 3.12.3")
elif arguments[:3] == ["-m", "pip", "freeze"]:
    print("a2v2==0.1.0")
elif arguments and arguments[0].endswith("check_reproduction_environment.py"):
    print('{"pass": true}')
elif arguments[:2] == ["-m", "torch.distributed.run"]:
    if any(value.endswith("nccl_probe.py") for value in arguments):
        output = Path(arguments[arguments.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        for rank in range(8):
            (output / f"rank-{rank}.json").write_text(
                json.dumps({"rank": rank, "pass": True}) + "\n",
                encoding="utf-8",
            )
    else:
        overrides = [
            value.removeprefix("checkpoint.save_dir=")
            for value in arguments
            if value.startswith("checkpoint.save_dir=")
        ]
        output = Path(overrides[0])
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoint_last.pt").touch()
        if any("finetune_mixup" in value for value in arguments):
            (output / "checkpoint_best.pt").touch()
    print("fake distributed command")
elif arguments and arguments[0].endswith("evaluate_finetuning_checkpoint.py"):
    output = Path(arguments[arguments.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"pass": true}\n', encoding="utf-8")
    print('{"pass": true}')
else:
    raise SystemExit(f"unexpected fake Python arguments: {arguments}")
```

Use this fake `nvidia-smi` body:

```bash
#!/usr/bin/env bash
printf 'fake nvidia-smi\n'
```

Run the driver without `--dry-run`, using:

```python
environment = {
    **os.environ,
    "PATH": f"{fake_bin}:{os.environ['PATH']}",
    "A2V2_PYTHON": str(fake_python),
    "A2V2_TRAIN_ENTRY": "/bin/true",
}
```

Assert that `pretrain/train.log` contains separate burn-in and resume launch
headers and that a second driver invocation appends rather than truncates the
first invocation.

- [ ] **Step 3: Run the driver tests and verify command-graph failures**

Run:

```bash
python -m pytest -q tests/integration/test_reproduction_driver.py
```

Expected: FAIL until the shell implements the probe and burn-in graph.

- [ ] **Step 4: Implement interpreter-aligned launcher arrays**

Replace the scalar default launcher with:

```bash
if [[ -n "${A2V2_TORCHRUN:-}" ]]; then
    TORCHRUN_COMMAND=("${A2V2_TORCHRUN}")
else
    TORCHRUN_COMMAND=("${PYTHON_BIN}" -m torch.distributed.run)
fi
```

Use `"${TORCHRUN_COMMAND[@]}"` in each distributed command.

- [ ] **Step 5: Implement preflight and probe**

Create output directories, call the helper under the selected GPU mask, then
run:

```bash
env CUDA_VISIBLE_DEVICES="${SELECTED_GPUS}" \
    "${TORCHRUN_COMMAND[@]}" \
    --standalone \
    --nproc-per-node=8 \
    "${REPOSITORY_ROOT}/tests/gpu/nccl_probe.py" \
    --output-dir "${OUTPUT_DIR}/environment/nccl-preflight"
```

Dry-run prints both commands.

- [ ] **Step 6: Implement burn-in and checkpoint validation**

On a fresh pretraining directory, run the normal pretraining command with
`--stop-at-update 1`, require `checkpoint_last.pt`, call the helper with
`--checkpoint`, then run the full command with `--resume`.

For an existing checkpoint, validate it and run only the full resume command.

- [ ] **Step 7: Preserve logs**

Change `run_logged` to add an ISO-8601 UTC header and use `tee -a`. Include a
human-readable phase name so the preflight probe, burn-in, resume, fine-tune,
and evaluation records remain distinct.

- [ ] **Step 8: Run shell and driver tests**

Run:

```bash
bash -n scripts/reproduce_meerkat_paper.sh
python -m pytest -q tests/integration/test_reproduction_driver.py
```

Expected: PASS.

### Task 5: Update durable GPU probes and documentation

**Files:**
- Modify: `tests/gpu/nccl_probe.py`
- Modify: `tests/gpu/nccl_resume.py`
- Modify: `README.md`
- Modify: `docs/reproducing-paper.md`
- Modify: `docs/code-guide.md`

**Interfaces:**
- Probes report `rng_gather_backend: "gloo"`
- Documentation records preflight, burn-in, log, and approval behavior

- [ ] **Step 1: Route probe RNG bytes through Gloo**

Set the CUDA device before NCCL initialization. Create one Gloo group after
the default NCCL group. Replace each probe's local `gather_object` block with:

```python
checkpoint_group = dist.new_group(backend="gloo")
ranked_rng = gather_rank_rng_states(
    payload["rng_state"],
    world_size=world_size,
    rank=rank,
    group=checkpoint_group,
)
if rank == 0:
    if ranked_rng is None:
        raise RuntimeError("rank zero did not receive RNG state")
    payload["rng_state"] = ranked_rng
```

Add `"rng_gather_backend": str(dist.get_backend(checkpoint_group))` to each
rank JSON report.

- [ ] **Step 2: Update researcher instructions**

Document:

- the NCCL tensor data plane and Gloo checkpoint control plane;
- the 64 GiB disk and 38 GiB free-memory gates;
- the eight-rank collective preflight;
- the one-update production burn-in and automatic full resume; and
- append-only log behavior.

State that full training still needs explicit approval in managed agent runs.

- [ ] **Step 3: Run focused CPU verification**

Run:

```bash
python -m py_compile \
  a2v2/training.py \
  a2v2/workflows.py \
  scripts/check_reproduction_environment.py
python -m pytest -q \
  tests/unit/test_checkpoint.py \
  tests/integration/test_cli.py \
  tests/integration/test_resume.py \
  tests/integration/test_reproduction_driver.py
bash -n scripts/reproduce_meerkat_paper.sh
```

Expected: PASS.

### Task 6: Eight-GPU and full regression verification

**Files:**
- Modify only defects exposed by verification.

**Interfaces:**
- Produces fresh eight-rank probe reports and bounded production checkpoints

- [ ] **Step 1: Run the complete CPU suite**

Run:

```bash
python -m pytest -q
```

Expected: all CPU tests pass; GPU-only pytest tests may run on this host.

- [ ] **Step 2: Run the eight-rank collective probe**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  tests/gpu/nccl_probe.py \
  --output-dir /tmp/a2v2-nccl-8-after
```

Expected: all rank reports contain `"pass": true` and
`"rng_gather_backend": "gloo"`.

- [ ] **Step 3: Run eight-rank exact resume**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  tests/gpu/nccl_resume.py \
  --output-dir /tmp/a2v2-resume-8-after
```

Expected: every rank reports exact state.

- [ ] **Step 4: Run bounded production burn-in and resume**

Run the paper pretraining recipe with the driver-equivalent overrides,
`--stop-at-update 1`, and a new temporary output directory. Load its
checkpoint, resume to update two, and compare the resulting state against a
fresh uninterrupted two-update branch with
`tests/gpu/compare_training_checkpoints.py`.

Expected: both branches complete, checkpoint transport uses Gloo, the process
group exits without a warning, and the comparator finds no logical difference
after normalizing the output directory.

- [ ] **Step 5: Audit the final diff**

Run:

```bash
git diff --check
git status --short
```

Review each changed line against the Global Constraints and the approved
design. Do not launch the full 384,230-update reproduction without new user
approval.
