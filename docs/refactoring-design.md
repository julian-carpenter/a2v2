# Readability refactor design

> **Historical record:** This 2026-07-24 design describes the intermediate
> 28-file `animal2vec/` package. A later intermediate step used `baseline/`;
> the live source now uses the six-file `a2v2/` package described in the
> [A2V2 identity design](superpowers/specs/2026-07-27-a2v2-project-identity-design.md).

Date: 2026-07-24

## Goal

Restructure the standalone Animal2Vec rewrite so a reader can find and follow
the implementation without knowing Python's `src` packaging convention or the
history of the Fairseq port. Preserve the verified native behavior, published
configuration files, native checkpoint schema, model parameter names, and
console commands.

## Constraints

- Keep the 14 corrections recorded in `gpu-verification-20260723.md`.
- Keep the native runtime independent of Fairseq and the analysis stack.
- Remove the archived `nn/` implementation and scripts that can only run with
  that implementation.
- Keep official checkpoint conversion. The converter consumes a plain exported
  tensor/config dictionary when a raw Fairseq checkpoint contains unavailable
  Python classes.
- Do not change model equations, tensor shapes, module attribute names,
  optimizer behavior, distributed settings, or serialized checkpoint fields.
- Preserve the existing YAML recipes without edits.
- Keep `animal2vec-train`, `animal2vec-infer`, and
  `animal2vec-convert-checkpoint`.

This standalone copy has no `.git` directory, so the refactor cannot create
the design and migration commits used in the original development workflow.
The test suite and the final file inventory provide the local audit trail.

## Options considered

### Remove only `nn/`

This removes the largest obsolete directory but leaves the implementation
under `animal2vec/` and five nested package groups. It does not address the
main navigation concern.

### Remove `src/` and retain domain subpackages

This exposes the package at the repository root while retaining
`models/`, `modules/`, `training/`, `data/`, `inference/`, `evaluation/`, and
`cli/`. The domain grouping is conventional, but readers still need to search
several directories to trace one waveform through the system.

### Use one flat package

Place each native implementation module directly under `animal2vec/`. Use
specific filenames and a documented reading order. This gives readers one
directory to scan and makes imports show each dependency directly.

The refactor uses this option because it matches the requested preference for
a shallow, readable structure. It avoids compatibility shim packages because
those shims would recreate the directory hierarchy that this work removes.

## Target layout

```text
animal2vec2.0/
├── animal2vec/
│   ├── __init__.py
│   ├── config.py
│   ├── audio.py
│   ├── dataset.py
│   ├── labels.py
│   ├── batching.py
│   ├── layers.py
│   ├── normalization.py
│   ├── sinc.py
│   ├── masking.py
│   ├── attention.py
│   ├── frontend.py
│   ├── decoder.py
│   ├── ema.py
│   ├── pretraining.py
│   ├── finetuning.py
│   ├── losses.py
│   ├── metrics.py
│   ├── optimizer.py
│   ├── checkpoint.py
│   ├── checkpoint_conversion.py
│   ├── event_detection.py
│   ├── event_evaluation.py
│   ├── inference.py
│   ├── train.py
│   ├── infer.py
│   └── convert_checkpoint.py
├── configs/
├── docs/
├── scripts/
└── tests/
```

`layers.py` replaces the vague `modules/common.py`. `batching.py` names the
token and distributed batch samplers. `event_detection.py` converts frame
probabilities to event intervals, while `event_evaluation.py` scores predicted
intervals. These names remove the previous collision between two files called
`events.py`.

## Public interface

The package root exports the main classes:

- `Animal2VecConfig`
- `Animal2VecPretrainingModel`
- `Animal2VecFineTuningModel`
- `InferenceRunner`
- `InferenceResult`
- `load_config`

The console commands keep their current names. Module execution changes from
`python -m animal2vec.cli.train` to `python -m animal2vec.train`; the README
and reproduction documentation use the new path.

The refactor does not promise compatibility for old internal imports such as
`animal2vec.modules.sinc` or `animal2vec.training.engine`. Those paths described
implementation details rather than a declared package API. A migration table
in the code guide maps each old path to its flat replacement.

## Behavior preservation

Moving Python definitions does not alter PyTorch state-dict names because those
names come from attributes assigned on each model, not from source filenames.
The refactor will leave those attributes and their construction order alone.
Native checkpoints store plain configuration dictionaries and tensor state
dictionaries, so they do not pickle model class module paths.

Tests will protect:

1. configuration parsing and serialization;
2. model construction, parameter names, forward paths, losses, and updates;
3. checkpoint save/load and exact resume;
4. inference, event detection, and event evaluation;
5. CLI smoke workflows;
6. source independence from Fairseq and removed package paths; and
7. the expected flat repository layout.

GPU tests remain in the repository. This CPU host can validate collection and
skip behavior, but the final handoff will list the one-A100 and NCCL commands
that should run again after transfer to a GPU host.

## Documentation and comments

`docs/code-guide.md` will give a reading order and trace pretraining,
fine-tuning, inference, and checkpoint flows across the flat modules. Each
module will start with a purpose and dependency summary. Large files will use
section headings, class docstrings, and tensor-shape notes where shape changes
drive the algorithm. Comments will explain compatibility-sensitive choices,
including mask ordering, ALiBi gathering, AMP skip semantics, DataLoader
resume state, and DDP bucket settings.

The GPU verification report remains as historical evidence. A note at its top
will explain that its `src/` and legacy paths describe the verified pre-refactor
snapshot and link to the path migration table.
