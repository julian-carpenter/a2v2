# A2V2 project identity design

Date: 2026-07-27

## Goal

Use `a2v2` as the single live identity for the extensible codebase while
describing the checked-in published recipes as Animal2Vec 1.0 reproduction
baselines.

## Naming contract

The source package will move from `baseline/` to `a2v2/`. Python imports will
use `a2v2.config`, `a2v2.data`, `a2v2.model`, `a2v2.training`, and
`a2v2.workflows`.

Packaging will use:

| Surface | Name |
| --- | --- |
| Python package | `a2v2` |
| Distribution | `a2v2` |
| Training command | `a2v2-train` |
| Inference command | `a2v2-infer` |
| Conversion command | `a2v2-convert-checkpoint` |

The project will not ship `baseline` compatibility imports or command aliases.
One identity gives future modules a clear home and keeps structural tests
unambiguous.

## Recipe terminology

The official recipes under `configs/MeerKAT/` and `configs/hyenas/` reproduce
the architecture and training settings released for Animal2Vec 1.0. Comments
and documentation will call them **Animal2Vec 1.0 reproduction baselines**.

The files under `configs/` will keep their paths and numerical values.
Researchers may therefore reuse published commands, manifests, and recorded
semantic hashes after changing only the executable name.

The two CPU smoke recipes do not reproduce paper metrics. Documentation will
call them reduced workflow checks that exercise the A2V2 implementation of the
Animal2Vec 1.0 flow.

A new `configs/README.md` will define the boundary:

- `configs/MeerKAT/` and `configs/hyenas/`: Animal2Vec 1.0 reproduction
  baselines.
- `configs/cpu_smoke_*.yaml`: fast implementation checks.
- `configs/a2v2/`: reserved naming convention for future A2V2 experiments;
  the rename will not create empty directories.

## Compatibility

The change affects import paths and command names. It must not change:

- YAML values or semantic hashes;
- model class bodies, attribute assignment order, or state dictionaries;
- checkpoint contents and conversion mappings;
- training, inference, event evaluation, or resume behavior;
- runtime dependencies.

Native checkpoints contain plain configuration dictionaries and tensor state,
not pickled package classes. The package rename therefore leaves checkpoint
loading intact. Existing Python programs must update imports from `baseline`
to `a2v2`.

## Documentation

Active documentation will introduce A2V2 as the project and identify the
current implementation as its Animal2Vec 1.0 reproduction baseline. Historical
design and GPU verification documents will preserve recorded paths while adding
a current-name note where readers enter those documents.

Source module docstrings may still use the common noun “baseline” when they
describe scientific role. They will use `a2v2` for package paths and project
identity.

## Verification

Structural tests will require exactly six files under `a2v2/` and reject both
the former `baseline/` and older `animal2vec/` package directories.

Verification will include:

1. A red structural test before the directory move.
2. All unit and integration tests under the new imports.
3. The complete CPU suite, with CUDA tests skipped on this host.
4. Both ordered model state signatures.
5. All YAML semantic hashes.
6. A wheel containing six `a2v2` Python files and three `a2v2-*` commands.
7. A search for stale live `baseline` import paths, commands, or packaging
   metadata.
