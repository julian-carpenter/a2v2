# Fairseq checkpoint conversion

This workflow brings official Animal2Vec 1.0 checkpoints into the A2V2
baseline. The native runtime cannot load Fairseq models directly. The
conversion command reads the tensor dictionary, builds the corresponding
native architecture, maps the weights, verifies full coverage, and writes a
versioned native checkpoint. Fairseq is not imported by the converter.

For the implementation details behind the mapping tables, read
`a2v2/workflows.py`. Its module and function docstrings
explain config extraction, key translation, tensor validation, teacher-state
construction, and the native payload written at the end. The
[code guide](code-guide.md) places conversion within the rest of the package.

## Pretraining checkpoints

Use the recipe that created the checkpoint:

```bash
a2v2-convert-checkpoint \
  animal2vec_large_pretrained_MeerKAT.pt \
  animal2vec_large_pretrained_MeerKAT.native.pt \
  --stage pretrain \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml
```

The conversion maps the convolutional frontend, projection, positional encoder,
context prenet, shared transformer, decoder, and EMA state. When a legacy
checkpoint does not contain a separate EMA tensor set, the native teacher is
initialized from the converted student.

## Fine-tuning checkpoints

A fine-tuning checkpoint needs two recipes: the active classifier recipe and
the architecture of its pretrained encoder.

```bash
a2v2-convert-checkpoint \
  animal2vec_large_finetuned_MeerKAT_240507.pt \
  animal2vec_large_finetuned_MeerKAT_240507.native.pt \
  --stage finetune \
  --config configs/MeerKAT/finetune_mixup_100.yaml \
  --pretrained-config configs/MeerKAT/a2v_large_pretrain_best.yaml
```

This maps the wrapped encoder and output projection. Optimizer and scheduler
state are intentionally absent, so a converted checkpoint is suitable for
inference or as initialization, not exact continuation of the old Fairseq run.

## Automatic configuration extraction

`--stage` and the YAML arguments may be omitted when the legacy file contains a
plain, fully resolved `cfg` dictionary. Explicit recipes are recommended because
official checkpoints can contain OmegaConf objects and framework defaults that
plain PyTorch cannot deserialize after Fairseq is removed.

If loading fails because the checkpoint embeds unavailable Python classes, make
a plain export once inside the archived Fairseq environment:

```python
from pathlib import Path

import torch
from omegaconf import OmegaConf

source = Path("animal2vec_large_finetuned_MeerKAT_240507.pt")
destination = source.with_suffix(".plain.pt")
state = torch.load(source, map_location="cpu")
plain_cfg = OmegaConf.to_container(state["cfg"], resolve=True)
torch.save({"model": state["model"], "cfg": plain_cfg,
            "num_updates": state.get("num_updates", 0)}, destination)
```

Move the resulting `.plain.pt` file to the native environment and run the same
conversion command against it. This export step is the only place that needs
OmegaConf or the archived environment.

## Strictness and reports

Successful conversion prints a JSON report with:

- `stage`: inferred or requested training stage
- `mapped`: number of destination tensors populated
- `renamed`: source-to-destination key pairs
- `omitted`: legacy tensors that are not part of the native selected stage
- `missing`: required native tensors, empty on success

A shape mismatch or missing core tensor is an error. Do not use `strict=False`
to bypass it: a plausible-looking partial model can produce invalid predictions.

Inspect the result without Fairseq:

```bash
python - <<'PY'
from a2v2.training import load_checkpoint

checkpoint = load_checkpoint("animal2vec_large_finetuned_MeerKAT_240507.native.pt")
print(checkpoint["format_version"], checkpoint["stage"], checkpoint["update"])
print(len(checkpoint["model"]), checkpoint["config"].keys())
PY
```

Then run inference on a known audio fixture and compare logits or frame
probabilities with a retained reference payload. The integration conversion
tests verify strict tensor mapping and converted fine-tuning logits. The GPU
report records the completed official archived/native layer, gradient, update,
and inference comparisons.

## Native checkpoint contents

The converted file uses only tensors and plain Python containers. It includes
the active config and, for fine-tuning, the pretrained encoder config. Native
inference therefore does not need a sidecar YAML. The schema is validated on
every load and versioned by `format_version`.
