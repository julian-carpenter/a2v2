# Task 11 Report: Modern Examples and SLURM Documentation

## Status and scope

Task 11 adds one paired modern pretraining and fine-tuning example, integrates
the new feature surface into the researcher guides, and adds the SLURM
operations guide. The task changes no model, training, launcher, published
recipe, or local reproduction-driver behavior.

The Task 11 commit includes this report. The parent handoff supplies the commit
hash because a commit cannot contain its own hash.

## Files

Task 11 changes:

- `README.md`;
- `configs/README.md`;
- `docs/code-guide.md`;
- `docs/reproducing-paper.md`;
- `tests/unit/test_config.py`; and
- `tests/unit/test_documentation.py`.

Task 11 adds:

- `configs/modern/rope_cls_geglu_pretrain.yaml`;
- `configs/modern/rope_cls_geglu_finetune.yaml`;
- `docs/slurm.md`; and
- this report.

## RED evidence

The resumed worktree already contained the modern recipe compatibility test and
frozen-byte guards in `tests/unit/test_config.py`. After restoring the
declared test extra on the restarted node, the preserved focused test failed
because the pretraining example did not exist:

```text
rtk python -m pytest -q tests/unit/test_config.py::test_modern_example_configs_select_compatible_architecture_and_policies
ConfigError: cannot load config .../configs/modern/rope_cls_geglu_pretrain.yaml
1 failed in 2.44s
```

A new documentation contract test ran before the prose and failed at the
missing SLURM guide:

```text
rtk python -m pytest -q tests/unit/test_documentation.py::test_modern_features_and_slurm_contract_are_documented
FileNotFoundError: .../docs/slurm.md
1 failed in 0.11s
```

The existing recipe-inventory gate then rejected both new YAML files before
their canonical hashes entered the inventory:

```text
rtk python -m pytest -q tests/unit/test_documentation.py::test_yaml_files_open_with_context_and_preserve_values
Extra items: configs/modern/rope_cls_geglu_pretrain.yaml
             configs/modern/rope_cls_geglu_finetune.yaml
1 failed in 0.20s
```

## GREEN evidence

The paired recipe test passed after both YAML files defined one compatible
encoder:

```text
1 passed in 5.17s
```

The documentation contract passed after README, config, code, reproduction,
and SLURM integration:

```text
1 passed in 0.05s
```

The combined Task 11 config and documentation gate passed:

```text
rtk python -m pytest -q tests/unit/test_config.py tests/unit/test_documentation.py
45 passed in 6.37s
```

The canonical modern recipe value hashes are:

```text
rope_cls_geglu_pretrain.yaml
7a1b1225a61f4bc7025c792562f923f752db1dc0050b319910d18e904bff3ee8

rope_cls_geglu_finetune.yaml
5edf20a0d85f401f14c42e07c6388d87a71109fa7665e0f0381cf84d5b937f93
```

## Modern recipe contract

Both recipes set the same sample rate, convolution geometry, encoder depth,
embedding width, head count, prenet depth, position strategy, attention
backend, RoPE theta, CLS selection, FFN type, and initialization. They select:

- RoPE and strict Flash attention;
- CLS regression in pretraining and CLS sequence classification in
  fine-tuning;
- packed GEGLU and DeepScaleLM;
- the existing activation-checkpointing option and partial-graph compilation;
- AdaGC defaults `0.99`, `1.04`, and 100 warm-up updates; and
- AdamW8bit with a 4,096-element threshold and cosine decay from 0.01 to 0.0.

The docs state that CLS, GEGLU, and DeepScaleLM define a new checkpoint family.
They also distinguish strict Flash from SDPA fallback, frame and sequence
heads, architecture fields from execution policy, and checkpoint v1 from v2.

## Command validation

These checks exited 0:

```text
rtk bash -n scripts/a2v2_slurm_node.sh
rtk bash -n scripts/reproduce_meerkat_slurm.sh
rtk a2v2-train --help
rtk bash scripts/reproduce_meerkat_slurm.sh /path/to/manifests /path/to/output --help
rtk bash scripts/a2v2_slurm_node.sh pretrain --help
```

Four bounded dry-runs exited 0:

- standalone published pretraining on two nodes by four GPUs;
- standalone published fine-tuning on two nodes by four GPUs;
- the published `all` graph with final one-node evaluation; and
- the same `all` rendering with both modern config files.

Each rendering produced one `srun` task per node, one torchrun worker per GPU,
world size eight, phase-specific c10d IDs, contract and config fingerprints,
distinct pretraining and fine-tuning output owners, and one-node final
evaluation.

No dry-run touched CUDA, SLURM, manifests, checkpoints, or output directories.
No `sbatch`, training update, paper-scale run, or job over the bounded test
window ran.

## Frozen controls

The frozen-byte test passes. Fresh SHA-256 checks produced:

```text
c5eb23d979cd12704f0dd3977fac1b6031cb4cbf7d5868930eaa6c2816f36a29  configs/MeerKAT/a2v_large_pretrain_best.yaml
384eed9a25da913258e618425c9526cffdaccdacc2a16f912f8e1719fe15a9e0  configs/MeerKAT/finetune_mixup_001.yaml
31daab6409907238af025a480bf236fd2bb82fdd9442a9a1e4d3b0552e0d3f4a  configs/MeerKAT/finetune_mixup_025.yaml
c31c2a3df37d2397ae2cd7cbc502aa849bd819af5ac35f9ff930ab35fb192fd0  configs/MeerKAT/finetune_mixup_100.yaml
6a40999452b56e95834d2487860ddc11974c9840b9814eafec1f21468de09102  configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml
f892d0f9bb52a823a1fe4bcd168aab33a29cbd6bcc395ce0822525b6bc3c07ca  configs/hyenas/finetune_mixup_100.yaml
485b4b36fe1045174a392834bd9ee9848538ab496944631a0b2d514e22d4de18  scripts/reproduce_meerkat_paper.sh
```

## Full verification

The full CPU unit and integration suite passed:

```text
rtk python -m pytest -q tests/unit tests/integration
507 passed, 8 warnings in 303.48s
```

The eight warnings come from the maintained multiworker `fork()`
deprecation cases in `tests/integration/test_cli.py`.

`rtk python -m compileall -q a2v2 tests` exited 0. The final report-inclusive
`git diff --check`, focused tests, frozen controls, and compile check run
before commit.

## Prose review

The stop-slop review scanned 541 added lines in modified prose and test files.
It found zero banned phrases or em dashes. The adverb-shaped scan returned
three `family` nouns in modified files and `supply`, `policy`, and
`family` in new files; manual review confirmed that none serves as an
adverb. The draft avoids canned contrasts, rhetorical setups, vague
performance claims, and passive descriptions of test status.

Scores: directness 10, rhythm 9, trust 10, authenticity 9, density 9,
for 47/50.

## Deployment concern and unrun gate

The modern dry-run exposed a workflow boundary rather than a model defect.
`scripts/reproduce_meerkat_slurm.sh --phase all` renders the
frame-event final evaluator. A modern `classification_head=cls` checkpoint
contains sequence logits, so that evaluator rejects it. The SLURM guide tells
CLS users to launch `pretrain` and `finetune` as separate phases and then
run sequence evaluation. Task 11 changes no Task 10 launcher behavior.

The real-site gate remains unrun. A target cluster must complete one-node
parity and at least two nodes by two GPUs for NCCL data flow, Gloo control,
shared-filesystem visibility and POSIX semantics, exact resume, contention,
proven-dead lock recovery, scheduler preemption, exit-75 requeue policy,
phase handoff, and rendezvous behavior. The guide includes the acceptance
matrix and evidence fields.
