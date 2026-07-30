# Minimal Configuration Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Fairseq-era recipe noise with minimal, fully commented YAML files while preserving every behavior-bearing Animal2Vec 1.0 baseline value.

**Architecture:** Keep `a2v2.config` as the compatibility translator for old checkpoints, but treat checked-in YAML as the native researcher interface. A policy test rejects known ignored keys from those YAML files, while existing recipe tests and a before/after native-config comparison protect numerical behavior.

**Tech Stack:** YAML, PyYAML, immutable Python dataclasses, PyTorch, pytest.

## Global Constraints

- Do not change model, optimizer, scheduler, dataset, metric, or checkpoint behavior.
- Continue accepting archived Fairseq fields in `a2v2.config` for embedded-checkpoint compatibility.
- Do not introduce YAML inheritance, anchors, includes, or generated recipes.
- Preserve the paper architecture, effective batch sizes, schedules, AMP values, augmentation, validation, and checkpoint selection.
- Comments must explain mathematical effect and researcher-facing meaning without obscuring values.
- Preserve the six Animal2Vec 1.0 paper-baseline recipes as frozen controls.

---

### Task 1: Define the checked-in recipe policy

**Files:**
- Modify: `tests/unit/test_documentation.py`

**Interfaces:**
- Consumes: all `*.yaml` files discovered below the repository root.
- Produces: `test_checked_in_recipes_exclude_ignored_legacy_options()`, which protects the cleaned researcher-facing contract.

- [ ] **Step 1: Add the failing policy test**

Add a mapping of forbidden dotted paths and recursively inspect each loaded
YAML mapping:

```python
IGNORED_RECIPE_PATHS = {
    "hydra",
    "common.log_format",
    "common.fp16_no_flatten_grads",
    "common.all_gather_list_size",
    "checkpoint.keep_last_epochs",
    "task._name",
    "task.verbose_tensorboard_logging",
    "dataset.skip_invalid_size_inputs_valid_test",
    "distributed_training.ddp_backend",
    "criterion._name",
    "criterion.use_focal_loss",
    "criterion.label_smoothing",
    "criterion.maxfilt_s",
    "criterion.max_duration_s",
    "criterion.lowP",
    "criterion.segmentation_metrics",
    "criterion.report_accuracy",
    "criterion.log_keys",
    "lr_scheduler._name",
    "model._name",
    "model.supported_modality",
    "model.ema_encoder_only",
    "model.load_pretrain_weights",
    "model.modalities.audio.mask_prob_adjust",
    "model.modalities.audio.inverse_mask",
    "model.modalities.audio.add_masks",
    "model.modalities.audio.ema_local_encoder",
}
```

The assertion must report `relative-file:dotted.path` for every violation so a
future author understands why the recipe is rejected.

- [ ] **Step 2: Run the policy test and observe legacy-field failures**

Run:

```bash
python -m pytest tests/unit/test_documentation.py::test_checked_in_recipes_exclude_ignored_legacy_options -q
```

Expected: failure listing the Fairseq/Hydra keys currently present in published,
smoke, and fixture YAML files.

- [ ] **Step 3: Keep the test focused on YAML representation**

Do not remove compatibility fields from dataclasses or parser allowlists. The
test inspects files, not serialized native checkpoints or `config_from_dict`.

- [ ] **Step 4: Commit the policy test with the recipe rewrite in Task 2**

The red test intentionally remains uncommitted until the cleaned recipes make
the repository green.

### Task 2: Rewrite all recipes around behavior-bearing values

**Files:**
- Modify: `configs/MeerKAT/a2v_large_pretrain_best.yaml`
- Modify: `configs/MeerKAT/finetune_mixup_100.yaml`
- Modify: `configs/MeerKAT/finetune_mixup_025.yaml`
- Modify: `configs/MeerKAT/finetune_mixup_001.yaml`
- Modify: `configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml`
- Modify: `configs/hyenas/finetune_mixup_100.yaml`
- Modify: `configs/cpu_smoke_pretraining.yaml`
- Modify: `configs/cpu_smoke_finetuning.yaml`
- Modify: `tests/fixtures/tiny_pretrain.yaml`
- Modify: `tests/fixtures/tiny_finetune.yaml`
- Modify: `configs/README.md`

**Interfaces:**
- Consumes: the legacy-compatible syntax accepted by `a2v2.config.load_config(path, overrides=())`.
- Produces: standalone YAML recipes that resolve to the same `Animal2VecConfig` behavior as the committed versions.

- [ ] **Step 1: Remove every field rejected by the policy test**

Delete only the forbidden paths listed in Task 1. Preserve parser support for
them so old embedded configs remain readable.

- [ ] **Step 2: Flatten both composite pretraining optimizer blocks**

Move `groups.default.lr_float` to `optimization.lr`, move the nested Adam
coordinates into `optimizer`, and move the nested scheduler coordinates into
`lr_scheduler`. The MeerKAT values must resolve to:

```yaml
optimization:
  update_freq: [5]
  max_update: 384230
  clip_norm: 1
  lr: [0.0001]
optimizer:
  _name: adam
  adam_betas: [0.9, 0.98]
  adam_eps: 1.0e-6
  weight_decay: 0.01
lr_scheduler:
  warmup_updates: 10000
```

The Hyena recipe uses the same structure with `update_freq: [2]`,
`max_update: 1417431`, and `lr: [0.00025]`.

- [ ] **Step 3: Expose event matching and focal-loss mathematics**

In each paper fine-tuning recipe, retain `metric_threshold`, `method`, and
`sigma_s`; add explicit `iou_threshold: 0.0`, `focal_alpha: 0.25`, and
`focal_gamma: 2.0`. Explain that predictions are formed after 100 ms average
pooling, matches require strict `IoU > 0`, and focal loss weights hard examples
with exponent two.

- [ ] **Step 4: Remove inactive controls whose omission is exact**

Remove zero `dropout_input`, `final_dropout`, and `drop_path` entries from paper
fine-tuning recipes. Remove Hyena channel-mask length together with its zero
probability. Remove paper-pretraining fields that equal inactive defaults only
when the surrounding comments still state the active mechanism unambiguously.

- [ ] **Step 5: Rewrite comments for two levels of understanding**

For each retained group, state the mathematical role first and the experimental
meaning second. Keep each explanation adjacent to its value. Correct stale
claims about four-GPU topology where deployment now overrides world size, but
retain `distributed_world_size: 4` as the original paper topology.

- [ ] **Step 6: Explain the compatibility boundary in `configs/README.md`**

Add a concise section saying that old Fairseq fields remain parser-compatible
but are intentionally absent from native recipes. List the major removed
categories and direct readers to the resolved TensorBoard `run/config` when
auditing a run.

- [ ] **Step 7: Run recipe and policy tests**

Run:

```bash
python -m pytest tests/unit/test_documentation.py tests/unit/test_config.py tests/integration/test_recipe_configs.py -q
```

Expected: only YAML hash assertions fail until Task 3 updates the intentional
value inventory; all config-loading and policy assertions pass.

### Task 3: Prove equivalence and finish the documentation contract

**Files:**
- Modify: `tests/unit/test_documentation.py`
- Modify: `docs/reproducing-paper.md`

**Interfaces:**
- Consumes: the cleaned recipes and `config_to_dict(load_config(...))`.
- Produces: updated canonical YAML hashes and a documented minimal-recipe policy.

- [ ] **Step 1: Compare committed and working recipes after native translation**

For every modified YAML, read the committed text with
`git show HEAD:<relative-path>`, load both old and new mappings through the
current `config_from_dict`, and compare `config_to_dict` results after removing
only:

```python
{
    "common.log_format",
    "checkpoint.keep_last_epochs",
    "task.name",
    "distributed.backend",
    "criterion.name",
    "criterion.use_focal_loss",
    "criterion.label_smoothing",
    "criterion.maxfilt_s",
    "criterion.max_duration_s",
    "criterion.low_probability",
    "model.name",
    "model.ema_encoder_only",
    "model.load_pretrain_weights",
    "model.audio.mask_prob_adjust",
    "model.audio.inverse_mask",
    "model.audio.add_masks",
    "model.audio.ema_local_encoder",
}
```

Any other mismatch must be investigated and either restored or explained as an
explicit inactive-default removal.

- [ ] **Step 2: Update canonical YAML hashes**

Recalculate `YAML_HASHES` from `yaml.safe_load`, canonical JSON serialization,
and SHA-256 exactly as the existing documentation test does. Do not hash
comments, so subsequent prose improvements remain safe.

- [ ] **Step 3: Document how to choose fields for future A2V2 recipes**

Add a short checklist to `docs/reproducing-paper.md`: start from the relevant
control, retain all behavior-bearing deviations, avoid compatibility-only
Fairseq keys, and archive the resolved TensorBoard configuration.

- [ ] **Step 4: Run focused configuration tests**

Run:

```bash
python -m pytest tests/unit/test_documentation.py tests/unit/test_config.py tests/integration/test_recipe_configs.py -q
```

Expected: all pass.

- [ ] **Step 5: Run CPU workflow integration tests**

Run:

```bash
python -m pytest tests/integration/test_cli.py tests/integration/test_pretraining_step.py tests/integration/test_finetuning_step.py tests/integration/test_reproduction_driver.py -q
```

Expected: all pass, proving cleaned recipes still construct and run both stages.

- [ ] **Step 6: Run complete verification**

Run:

```bash
python -m pytest
python -m py_compile a2v2/config.py
git diff --check
```

Expected: the complete CPU suite passes, CUDA-only tests skip on a CPU host,
compilation succeeds, and no whitespace errors are reported.

- [ ] **Step 7: Review the final diff**

Confirm that no Python runtime behavior changed, no configuration file was
added or deleted, and the diff contains only tests, YAML, and configuration
documentation.
