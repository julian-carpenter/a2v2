# Flat Package Readability Refactor Implementation Plan

> **Historical record:** Agents completed this plan against the intermediate
> 28-file package. Do not execute its path or command steps against the current
> tree. A later consolidation plan also uses the superseded `baseline` name.
> Use the [A2V2 identity plan](superpowers/plans/2026-07-27-a2v2-project-identity.md)
> and [code-guide.md](code-guide.md) for the live layout.

> **For agentic workers:** Execute this plan task by task. Use test-first
> development for the structural contract and run the full regression suite
> after each import migration.

**Goal:** Replace the archived and nested source layout with one readable
`animal2vec/` directory while preserving all verified native behavior.

**Architecture:** Move each implementation file from
`src/animal2vec/<domain>/` to the repository-root package and replace nested
imports with direct sibling imports. Remove Fairseq-only source and
verification utilities. Keep model attributes, equations, YAML files,
checkpoint structures, and console command names unchanged.

**Tech stack:** CPython 3.10+, PyTorch 2.3+, NumPy, SoundFile, h5py, PyYAML,
setuptools, and pytest.

## Global constraints

- Preserve the GPU-verified numerical paths described in
  `docs/gpu-verification-20260723.md`.
- Preserve the pretraining state signature of 120 entries with SHA-256
  `5efbec43fd1c1c392b8b4278cee6a21513f68f3095353bb22524d2ada2ecf6ad`.
- Preserve the fine-tuning state signature of 59 entries with SHA-256
  `b7b2ea2ad9017ec94d56516fbda49a932866b472a48633804619b7232f3d05bb`.
- Keep the three installed console command names.
- Keep runtime dependencies unchanged.
- Keep every checked-in YAML file unchanged.
- Do not add compatibility packages that recreate the removed directory tree.
- This standalone folder has no `.git` metadata, so no task includes a commit.

---

### Task 1: Lock the structural and model-state contracts

**Files:**

- Create: `tests/unit/test_repository_layout.py`
- Create: `tests/unit/test_model_state_contract.py`

**Produces:**

- A failing contract for the target flat layout.
- A passing contract for model parameter names, shapes, dtypes, and order.

- [x] Add a layout test that asserts:

```python
ROOT = Path(__file__).parents[2]
PACKAGE = ROOT / "animal2vec"

assert PACKAGE.is_dir()
assert not (ROOT / "src").exists()
assert not (ROOT / "nn").exists()
```

- [x] In the same test, import each target module with
  `importlib.import_module(f"animal2vec.{name}")`.

- [x] Parse package source with `ast` and fail if an import starts with
  `fairseq`, `hydra`, `omegaconf`, `torchaudio`, `timm`, `transformers`, or
  `tensorflow`.

- [x] Add a model-state signature helper:

```python
def state_signature(model: torch.nn.Module) -> tuple[int, str]:
    rows = (
        f"{name}:{tuple(value.shape)}:{value.dtype}"
        for name, value in model.state_dict().items()
    )
    encoded = "\n".join(rows).encode()
    return len(model.state_dict()), hashlib.sha256(encoded).hexdigest()
```

- [x] Construct models from `tests/fixtures/tiny_pretrain.yaml` and
  `tests/fixtures/tiny_finetune.yaml`, then assert the two signatures listed in
  the global constraints.

- [x] Run:

```bash
python -m pytest -q tests/unit/test_repository_layout.py
```

Expected before migration: failure because `animal2vec/` does not exist and
`src/` plus `nn/` do exist.

- [x] Run:

```bash
python -m pytest -q tests/unit/test_model_state_contract.py
```

Expected before migration: pass.

### Task 2: Move the native package to one flat directory

**Files:**

- Move the package files according to this table.

| Old path | New path |
| --- | --- |
| `src/animal2vec/__init__.py` | `animal2vec/__init__.py` |
| `src/animal2vec/config.py` | `animal2vec/config.py` |
| `src/animal2vec/audio.py` | `animal2vec/audio.py` |
| `src/animal2vec/data/dataset.py` | `animal2vec/dataset.py` |
| `src/animal2vec/data/labels.py` | `animal2vec/labels.py` |
| `src/animal2vec/data/sampler.py` | `animal2vec/batching.py` |
| `src/animal2vec/modules/common.py` | `animal2vec/layers.py` |
| `src/animal2vec/modules/normalization.py` | `animal2vec/normalization.py` |
| `src/animal2vec/modules/sinc.py` | `animal2vec/sinc.py` |
| `src/animal2vec/modules/masking.py` | `animal2vec/masking.py` |
| `src/animal2vec/modules/attention.py` | `animal2vec/attention.py` |
| `src/animal2vec/models/frontend.py` | `animal2vec/frontend.py` |
| `src/animal2vec/modules/decoder.py` | `animal2vec/decoder.py` |
| `src/animal2vec/modules/ema.py` | `animal2vec/ema.py` |
| `src/animal2vec/models/pretraining.py` | `animal2vec/pretraining.py` |
| `src/animal2vec/models/finetuning.py` | `animal2vec/finetuning.py` |
| `src/animal2vec/training/losses.py` | `animal2vec/losses.py` |
| `src/animal2vec/training/metrics.py` | `animal2vec/metrics.py` |
| `src/animal2vec/training/optim.py` | `animal2vec/optimizer.py` |
| `src/animal2vec/training/checkpoint.py` | `animal2vec/checkpoint.py` |
| `src/animal2vec/training/engine.py` | `animal2vec/engine.py` |
| `src/animal2vec/checkpoint_conversion.py` | `animal2vec/checkpoint_conversion.py` |
| `src/animal2vec/inference/events.py` | `animal2vec/event_detection.py` |
| `src/animal2vec/evaluation/events.py` | `animal2vec/event_evaluation.py` |
| `src/animal2vec/inference/runner.py` | `animal2vec/inference.py` |
| `src/animal2vec/cli/train.py` | `animal2vec/train.py` |
| `src/animal2vec/cli/infer.py` | `animal2vec/infer.py` |
| `src/animal2vec/cli/convert_checkpoint.py` | `animal2vec/convert_checkpoint.py` |

**Produces:**

- Direct sibling imports under `animal2vec/`.
- No runtime package under `src/`.

- [x] Move files without editing class bodies or model attribute assignments.

- [x] Rewrite relative imports to direct sibling imports, for example:

```python
from .attention import MultiheadAttention, alibi_bias
from .checkpoint import load_checkpoint
from .pretraining import Animal2VecPretrainingModel
```

- [x] Replace imports in tests and scripts with the flat module paths.

- [x] Update `pyproject.toml`:

```toml
[project.scripts]
animal2vec-train = "animal2vec.train:main"
animal2vec-infer = "animal2vec.infer:main"
animal2vec-convert-checkpoint = "animal2vec.convert_checkpoint:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["animal2vec"]

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [x] Expand `animal2vec/__init__.py` to export the six main public objects
  named in the design.

- [x] Run the two Task 1 tests. Both must pass.

- [x] Run:

```bash
python -m pytest -q tests/unit tests/integration
```

Expected outside the restricted sandbox: all CPU tests pass.

### Task 3: Remove Fairseq-only and archived source

**Files:**

- Delete: `nn/`
- Delete: `get_results_for_single_manifest_split.py`
- Delete: `animal2vec_train.py`
- Delete: `animal2vec_inference.py`
- Delete: `tests/parity/conftest.py`
- Delete: `tests/parity/test_attention_parity.py`
- Delete: `tests/parity/test_frontend_parity.py`
- Delete Fairseq and archived-reference utilities under `tests/gpu/`.
- Delete obsolete porting plans and pre-verification handoff documents.

**Produces:**

- A standalone repository containing only executable native code.
- Retained equation fixtures that do not import archived source.

- [x] Move the masking fixture assertion from
  `tests/parity/test_masking_parity.py` into `tests/unit/test_masking.py`.

- [x] Move the teacher-target/regression equation assertion from
  `tests/parity/test_pretraining_parity.py` into `tests/unit/test_losses.py`.

- [x] Move the focal-loss equation assertion from
  `tests/parity/test_finetuning_parity.py` into `tests/unit/test_losses.py`.

- [x] Delete the remaining old parity files after the independent equations
  live in unit tests.

- [x] Remove these archived GPU utilities:

```text
tests/gpu/export_official_checkpoint.py
tests/gpu/legacy_event_evaluator_export.py
tests/gpu/legacy_official_finetuning_export.py
tests/gpu/legacy_official_pretraining_export.py
tests/gpu/legacy_official_pretraining_memory.py
tests/gpu/legacy_reference_export.py
tests/gpu/native_event_evaluator_compare.py
tests/gpu/native_official_finetuning_compare.py
tests/gpu/native_official_pretraining_compare.py
tests/gpu/native_reference_compare.py
```

- [x] Keep the durable native CUDA tests, NCCL probes, checkpoint comparator,
  and official native inference utility.

- [x] Remove these historical working documents because the final GPU report
  supersedes them and their commands require the absent Fairseq checkout:

```text
docs/A100_GPU_VERIFICATION_BOOTSTRAP.md
docs/GPU_VERIFICATION_HANDOFF.md
docs/architecture-map.md
docs/superpowers/
```

- [x] Scan all Python source outside the historical report:

```bash
rg -n '(^|\s)(from|import)\s+(fairseq|hydra|omegaconf|torchaudio|timm|transformers|tensorflow)' \
  animal2vec tests scripts
```

Expected: no matches.

### Task 4: Add reader-facing code documentation

**Files:**

- Create: `docs/code-guide.md`
- Modify: all modules in `animal2vec/`
- Modify: `README.md`
- Modify: `docs/checkpoint-conversion.md`
- Modify: `docs/reproducing-paper.md`
- Modify: `docs/gpu-verification-20260723.md`

**Produces:**

- A stable reading order and old-to-new import migration table.
- Comments around the GPU-sensitive behavior.

- [x] Write a code guide with these reading paths:

```text
Configuration and input:
config.py -> audio.py -> labels.py -> dataset.py -> batching.py

Shared encoder:
sinc.py -> attention.py -> masking.py -> frontend.py

Pretraining:
ema.py -> decoder.py -> losses.py -> pretraining.py

Fine-tuning and inference:
finetuning.py -> event_detection.py -> inference.py -> event_evaluation.py

Runtime:
optimizer.py -> checkpoint.py -> engine.py -> train.py
```

- [x] Add module and public-class docstrings. Document tensor shapes at the
  boundaries where audio, feature, clone, head, and class dimensions change.

- [x] Add section headings to `config.py`, `frontend.py`, `train.py`, and
  `checkpoint_conversion.py`.

- [x] Explain these compatibility-sensitive choices beside their code:

  - deterministic Sinc reflection padding;
  - zeroing masked features before positional convolution;
  - gathering ALiBi rows and columns before clone scaling;
  - FP32 decoder normalization under autocast;
  - skipped AMP updates leaving mutable training state unchanged;
  - committed DataLoader batch positions versus worker prefetch;
  - per-rank DataLoader generator ownership;
  - releasing deserialized checkpoint tensors before training;
  - fixed DDP bucket/traversal settings for exact resume; and
  - tied-score average precision grouping.

- [x] Update all active commands and imports in the README and active docs.

- [x] Add a historical-layout note to the GPU report. Do not rewrite its
  recorded paths or claims.

### Task 5: Validate packaging and behavior

**Files:**

- Modify only files required to fix failures caused by Tasks 2 through 4.

- [x] Compile:

```bash
python -m compileall -q animal2vec tests scripts
```

- [x] Run:

```bash
python -m pytest -q
```

Expected on this CPU host outside sandbox restrictions: 160 native CPU tests
pass and 14 CUDA tests skip.

- [x] Check all YAML files:

```bash
python -m pytest -q tests/integration/test_recipe_configs.py
```

- [x] Test entry points:

```bash
python -m animal2vec.train --help
python -m animal2vec.infer --help
python -m animal2vec.convert_checkpoint --help
```

- [x] Build a wheel and inspect it:

```bash
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/animal2vec-wheel
python -m zipfile -l /tmp/animal2vec-wheel/animal2vec-0.1.0-py3-none-any.whl
```

The wheel must contain `animal2vec/*.py` and must not contain `src/`, `nn/`,
tests, or legacy scripts.

- [x] Recompute both model-state signatures and require exact matches with the
  global constraints.

- [x] Run whitespace, obsolete-path, forbidden-import, and Markdown-link scans.

- [x] Record the GPU follow-up commands in `docs/code-guide.md`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -m gpu
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 tests/gpu/nccl_probe.py --output-dir results/gpu_reverification/nccl-2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 tests/gpu/nccl_resume.py --output-dir results/gpu_reverification/resume-4
```
