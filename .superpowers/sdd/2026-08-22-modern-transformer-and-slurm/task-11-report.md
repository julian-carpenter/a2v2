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

## Fix Round 1

### Review findings and scope

Fix Round 1 adds the missing checkpoint-backed CLS evaluator and strengthens
the documentation and modern-pair compatibility tests. It also repairs the
Flash sentence in `docs/code-guide.md` and replaces the public `Task 9`
label in `docs/slurm.md` with the distributed checkpoint runtime name.

Files changed or added in this round:

- `a2v2/workflows.py`;
- `pyproject.toml`;
- `tests/integration/test_sequence_evaluation.py`;
- `tests/unit/test_config.py`;
- `tests/unit/test_documentation.py`;
- `README.md`;
- `docs/code-guide.md`;
- `docs/slurm.md`; and
- this report.

No model, optimizer, training-loop, SLURM launcher, published recipe, or local
paper-driver behavior changed.

### RED evidence

The first valid evaluator integration test failed at the missing public
function:

```text
rtk proxy pytest -q tests/integration/test_sequence_evaluation.py -x
AttributeError: module 'a2v2.workflows' has no attribute 'evaluate_sequence_main'
1 failed
```

The modern CLS documentation test failed because the guide named no real
sequence-evaluation section or command:

```text
rtk proxy pytest -q tests/unit/test_documentation.py -k 'task11_relative or documented_slurm or a100_claim or modern_cls_guidance or slurm_site_gate'
test_modern_cls_guidance_uses_sequence_evaluation_and_no_all_stage_training
assert section is not None
2 failed, 3 passed
```

One of those two failures came from an initial dry-run stdout assertion that
did not match the launcher's documented rendered-command output. The corrected
test executes the same dry-run and checks `Launching: srun` plus the node
wrapper path.

A later strict-config test replaced `config.pretrained` with a fine-tuning
config. Before the stage check, evaluation reached model-state mismatch instead
of rejecting the stored config authority:

```text
rtk proxy pytest -q tests/integration/test_sequence_evaluation.py::test_sequence_evaluator_requires_stored_pretraining_config
AssertionError: assert 'stored pretraining config' in stderr
1 failed
```

### Sequence evaluator contract

`a2v2-evaluate-sequence` accepts a native checkpoint, required fine-tuning
config, repeatable strict overrides, and one selected device. It:

1. requires a native fine-tuning checkpoint;
2. requires stored `config.active` and `config.pretrained` mappings with
   fine-tuning and pretraining stages;
3. requires a stored and supplied CLS classification head;
4. checks sample rate, normalization, convolution geometry, ordered labels,
   layer averaging, CLS selection, and focal-loss parameters;
5. rebuilds the encoder from the checkpoint's stored pretraining config;
6. loads the complete model state with `strict=True`;
7. calls the existing `_validate` path, which calls
   `sequence_classification_metrics`; and
8. prints sorted JSON sequence metrics.

The evaluator reads no optimizer, scheduler, scaler, sampler, resume
fingerprint, process group, or saved world-size contract. Tests pass a world
size override of seven against a checkpoint produced at world size one. A
SHA-256 check before and after two evaluations proves that the checkpoint bytes
do not change.

The integration fixture creates two real WAV/HDF5 examples. A zero-logit
classifier produces 0.5 probabilities. At threshold 0.6 the measured sequence
F1 is 0.0 and accuracy is 0.5. At threshold 0.4 the measured F1 is 2/3. The
suite also rejects frame-head configs, frame-head checkpoints, pretraining
checkpoints, reversed label order, incomplete model state, a stored
fine-tuning config in the pretrained slot, and an unknown override.

The local CLI help check exited 0 after refreshing the editable installation:

```text
rtk python -m pip install -e . --no-deps
rtk proxy a2v2-evaluate-sequence --help
```

`README.md` and `docs/slurm.md` show a local invocation.
`docs/slurm.md` also shows:

```text
srun --nodes=1 --ntasks=1 --gpus=1 a2v2-evaluate-sequence ...
```

Modern CLS guidance contains no `--phase all` training command.

### Stronger config and documentation tests

The modern config guard now compares sample rate, normalization, convolution
layers, Transformer depth and width, head count, MLP ratio, norm order,
affinity, epsilon, positional policy, attention backend, RoPE theta, CLS,
FFN, initialization, sinc choices, positional-convolution depth, width, and
groups, prenet depth, and ALiBi choices. It constructs the full pretraining
student and fine-tuning encoder on the meta device. Their ordered state keys,
shapes, and dtypes must match. The test records the permitted fine-tuning
dropout, activation dropout, attention dropout, input dropout, layerdrop,
drop-path, and activation-checkpointing controls. Changing the fine-tuning MLP
ratio to 3.0 fails the guard.

Documentation tests now resolve relative Markdown file and heading links,
execute the README SLURM dry-run, execute launcher and evaluator help, and run
`bash -n` on both SLURM scripts. They keep the `2.7787x` A100 ratio in the
same paragraph as its `B=2`, `T=128`, `D=64`, five-iteration,
two-warm-up, single-A100, and no-general-throughput qualifiers. They also
require the real-site section to retain the `Unrun real-site gate` heading
and `did not run` status.

### GREEN evidence and controls

The final focused suite passed:

```text
rtk pytest -q tests/integration/test_sequence_evaluation.py tests/unit/test_config.py tests/unit/test_documentation.py
57 passed
```

The full CPU suite passed:

```text
rtk pytest -q tests/unit tests/integration
519 passed
```

These checks also exited 0:

```text
rtk python -m compileall -q a2v2 scripts tests
rtk bash -n scripts/reproduce_meerkat_slurm.sh
rtk bash -n scripts/a2v2_slurm_node.sh
rtk git diff --check
```

The frozen published-recipe and local-driver test passed:

```text
rtk pytest -q tests/unit/test_config.py::test_published_recipes_and_local_reproduction_driver_keep_frozen_hashes
1 passed
```

An unconstrained `rtk pytest -q` also ran. It reported 542 passes and three
failures in optional bitsandbytes CUDA tests because this host lacks the native
binary expected by those tests. The requested CPU suite remains green, and
this round changes no bitsandbytes or CUDA behavior.

### Prose, commit, and site status

The stop-slop review covered every added prose line. The new prose contains no
em dash, canned contrast, vague performance claim, or detached benchmark
qualifier. The repository-wide phrase scan found two existing em dashes at
`docs/code-guide.md:623-624`, outside this round's diff.

Commit status: the Fix Round 1 commit includes this report. The parent handoff
supplies its hash because a commit cannot contain its own hash.

The real-site gate remains unrun. This round executes no SLURM allocation,
`srun` GPU evaluation, paper-scale training, or scheduler requeue test. A
target site still must complete the one-node parity gate and the two-node
acceptance matrix in `docs/slurm.md`.
