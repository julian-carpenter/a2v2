# Repository-wide documentation design

> **Historical record:** This design counted the former 28 runtime modules.
> The documentation principles remain relevant, while
> [code-guide.md](code-guide.md) describes the consolidated source.

Date: 2026-07-24

## Objective

Explain the standalone Animal2Vec implementation to two readers at once:

- a researcher who wants to connect the paper's concepts to tensors,
  parameters, and experimental procedures; and
- a developer who needs exact interfaces, shapes, failure conditions, and
  state-management rules.

The documentation will sit beside the artifact it explains. A reader should
not need to infer a test's purpose from its name, decode a recipe parameter
from Fairseq history, or inspect a function body to learn the expected units.

## Scope

The pass covers the complete standalone source distribution:

- 28 runtime Python modules under `animal2vec/`;
- two dataset and manifest audit programs under `scripts/`;
- 40 CPU, integration, CUDA, and distributed test modules;
- eight example recipes and two test fixture recipes;
- `README.md` and the six documents under `docs/`;
- `.gitignore`, `pyproject.toml`, and `requirements.txt`.

The MIT `LICENSE` text will remain byte-for-byte unchanged. License text is a
legal artifact, and explanatory prose belongs in `README.md`. The dated GPU
report will keep its recorded results, commands, and historical paths. New
navigation notes may point readers to current documentation without rewriting
the audit record.

## Documentation layers

### Runtime source

Each runtime module will provide:

- a module docstring that places the file in the pretraining, fine-tuning,
  inference, or runtime flow;
- class and function docstrings with parameters, units, tensor shapes,
  returns, state changes, and raised errors where those details matter;
- short comments beside mathematical translations and compatibility-sensitive
  ordering;
- plain-language explanations of domain terms such as ALiBi, EMA, Sinc
  filters, focal loss, token budgets, and event IoU.

Comments will explain why the code performs an operation. They will not repeat
assignments or narrate Python syntax.

### Tests

Each test module will start with a statement of the scientific or engineering
contract it protects. Every test will have a docstring that tells the reader:

- which behavior it isolates;
- why the behavior affects reproducibility or numerical compatibility; and
- what type of regression the assertion would catch.

Fixture and helper functions will describe the state they construct. CUDA and
NCCL tests will name their hardware assumptions and distinguish numerical
tolerance from bitwise equality.

### YAML recipes

Comments will identify the experiment, stage, units, and relationship to the
paper. Section comments will explain:

- precision and reproducibility;
- manifests and audio geometry;
- token-budget batching and distributed world size;
- loss, optimizer, and schedule semantics;
- masking, mixup, EMA, and architecture settings.

Comments must not change parsed values. A semantic hash test will canonicalize
each recipe through `yaml.safe_load` and protect the pre-documentation value
tree.

### Project and prose files

`pyproject.toml`, `requirements.txt`, and `.gitignore` will explain their
sections and constraints. The code guide will gain a researcher-oriented
glossary and equations-to-files map. Existing focused documents will keep
their roles instead of duplicating the README.

## Accuracy rules

- Preserve executable statements, imports, values, test assertions, CLI
  behavior, recipes, model attributes, and serialized state.
- Use `[batch, ...]` shape notation and name each dimension at its boundary.
- State whether intervals are closed, open, or half-open.
- State whether losses are summed or averaged and where distributed scaling
  occurs.
- Distinguish random state that belongs to Python, NumPy, PyTorch, DataLoader,
  samplers, and CUDA ranks.
- Mark historical evidence as historical. Do not convert a short GPU burn-in
  into a paper-reproduction claim.

## Verification

The documentation pass will add a source-quality test that requires module and
definition docstrings and verifies YAML comments against the recorded semantic
hashes. Existing behavioral tests, model-state signatures, recipe loading,
CLI help, wheel construction, Markdown links, and forbidden-import checks will
run after the edits.

This standalone directory has no Git metadata. The test results and semantic
hashes provide the audit trail that a commit diff would otherwise supply.
