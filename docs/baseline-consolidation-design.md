# Baseline package consolidation design

> **Historical record:** This document specifies the intermediate `baseline`
> package name. The implementation now lives under `a2v2/`; its scientific
> role remains the Animal2Vec 1.0 reproduction baseline. See the
> [A2V2 identity design](superpowers/specs/2026-07-27-a2v2-project-identity-design.md).

Date: 2026-07-27

## Goal

The verified native PyTorch implementation currently spreads its runtime code
across 28 files in a package named `animal2vec`. This change will present the
paper-reproduction implementation as a compact baseline that future research
code can extend without inheriting the old project name.

The new package will be named `baseline`. It will contain six Python files,
including `__init__.py`, and no nested source directories:

```text
baseline/
├── __init__.py
├── config.py
├── data.py
├── model.py
├── training.py
└── workflows.py
```

The name `baseline` describes the role of the code. It also leaves room for
future sibling packages that implement new objectives, architectures, or data
pipelines.

## Compatibility contract

This refactor may change import paths and command names. It must preserve the
properties that determine paper reproduction:

- model classes, submodule attribute names, tensor shapes, and state-dictionary
  insertion order;
- all mathematical operations and their execution order;
- random-number consumption, masking, mixup, batching, and resume behavior;
- checkpoint contents and legacy checkpoint conversion rules;
- YAML values and command-line override semantics;
- single-process, distributed, mixed-precision, and accumulation behavior;
- inference windowing, probability fusion, and event evaluation.

The pretraining state signature must remain 120 entries with SHA-256
`5efbec43fd1c1c392b8b4278cee6a21513f68f3095353bb22524d2ada2ecf6ad`.
The fine-tuning signature must remain 59 entries with SHA-256
`b7b2ea2ad9017ec94d56516fbda49a932866b472a48633804619b7232f3d05bb`.

## File responsibilities and dependency direction

### `baseline/config.py`

This file owns recipe dataclasses, validation, serialization, YAML loading, and
dotted command-line overrides. It imports no other baseline module.

### `baseline/data.py`

This file owns audio loading and resampling, label loading and rasterization,
manifest datasets, collation, and token-based batch samplers. It imports only
`baseline.config`.

### `baseline/model.py`

This file owns mathematical neural-network primitives, numerical normalization,
the Sinc front end, attention and ALiBi, mask generation, the convolutional
decoder, the shared audio encoder, the EMA teacher, both task models, and their
loss functions. It imports `baseline.config` and the shape helpers in
`baseline.data`.

Keeping all model mathematics together gives a researcher one continuous
reading path from waveform samples to pretraining targets or fine-tuning
logits. It also prevents cycles between small files that previously separated
closely coupled pieces such as masking, the decoder, and pretraining.

### `baseline/training.py`

This file owns optimizer construction, the update-based cosine schedule,
checkpoint I/O, random-state capture, metrics, and the reusable training engine.
It imports model types only for annotations or operations that require them.

Checkpoint primitives live in the same file as the engine so that exact resume
state has one owner. Model classes do not import the training engine.

### `baseline/workflows.py`

This file owns training orchestration, inference, event fusion and evaluation,
legacy checkpoint conversion, and the three command-line entry points. It sits
at the top of the dependency graph and may import every lower-level file.

The command entry functions have distinct names:

- `train_main`
- `infer_main`
- `convert_checkpoint_main`

The installed commands will be `baseline-train`, `baseline-infer`, and
`baseline-convert-checkpoint`.

### `baseline/__init__.py`

This file exports the small public surface needed for normal use: configuration
loading, the two model classes, and inference types. Researchers can import
specialized helpers from the five implementation files.

## Documentation standard

Each file begins with a module docstring that explains its scientific role,
data flow, units, shape notation, and dependencies. Section dividers retain the
navigation value that the former small filenames provided.

Comments around significant computations use this order:

```python
# Mathematics: ...
# Interpretation: ...
```

The mathematical line states the equation, domain, shape transformation,
normalization axis, probability rule, or optimizer update. The conceptual line
then explains the purpose in paper-reproduction terms. Comments concentrate on
operations that can change results. They do not restate syntax such as variable
assignment or loop mechanics.

Every named class and function retains a docstring. The repository's
documentation test will continue to enforce this rule.

## Migration and compatibility policy

Tests, GPU utilities, examples, packaging metadata, and active documentation
will move to the `baseline` import paths in one change. The old `animal2vec`
package will be removed rather than kept as a shim. A shim would exceed the
six-file source goal, preserve the misleading namespace, and create two public
ways to reach the same implementation.

Native checkpoints store state dictionaries and serialized configuration
values, not pickled model instances. Moving class definitions therefore does
not change their on-disk format. The state-signature tests and checkpoint tests
will catch attribute-order or naming mistakes introduced during consolidation.

The historical GPU verification report will remain unchanged except for a
short note that maps its recorded command paths to the new command. This keeps
the report as evidence of the version tested on eight A100 GPUs.

## Verification

Verification proceeds from cheap structural checks to the complete CPU suite:

1. Assert that `baseline` contains exactly the six intended Python files and no
   nested source directories.
2. Import every consolidated module and reject removed framework imports.
3. Check both state-dictionary signatures.
4. Run configuration, data, model, training, checkpoint, conversion, inference,
   CLI, and documentation tests.
5. Run the full CPU test suite and confirm GPU tests skip on this host.
6. Build a wheel and inspect it for exactly the new package.
7. Search the active code and documentation for stale `animal2vec` imports.

The local machine has no GPU. The existing GPU report remains the behavioral
reference, and the updated guide will give the GPU agent the new commands for a
future confirmation run.
