# Configuration families

The checked-in experiment recipes fall into two groups.

## Animal2Vec 1.0 reproduction baselines

The files under `MeerKAT/` and `hyenas/` preserve the architecture, objectives,
optimization settings, masking, mixup, and training schedules released for
Animal2Vec 1.0. Use these recipes when reproducing the original paper or
comparing a new A2V2 method against the published system:

| Dataset | Stage | Recipe |
| --- | --- | --- |
| MeerKAT | large pretraining | `MeerKAT/a2v_large_pretrain_best.yaml` |
| MeerKAT | full-data fine-tuning | `MeerKAT/finetune_mixup_100.yaml` |
| MeerKAT | 25% fine-tuning | `MeerKAT/finetune_mixup_025.yaml` |
| MeerKAT | 1% fine-tuning | `MeerKAT/finetune_mixup_001.yaml` |
| Spotted hyena | base pretraining | `hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml` |
| Spotted hyena | full-data fine-tuning | `hyenas/finetune_mixup_100.yaml` |

These files define the frozen Animal2Vec 1.0 control baseline. New A2V2
experiments should not overwrite them or reuse their filenames for changed
methods.

## What belongs in a native recipe

Each checked-in YAML contains settings that affect data selection, tensor
geometry, model computation, optimization, validation, or saved output. The
comments next to a value explain its mathematical role and experimental
consequence. Recipes use no YAML inheritance, so you can understand and
archive one file without resolving another.

`a2v2.config` accepts a small set of old Fairseq and Hydra fields when it
recovers configurations embedded in archived checkpoints. Native recipes omit
fields that the new runtime does not consume:

- Hydra output directories and the task, criterion, model, and scheduler
  `_name` selectors (`optimizer._name` remains because it chooses the actual
  optimization algorithm);
- Fairseq logging, all-gather, gradient-flattening, and DDP-backend controls;
- criterion log-key, report, and segmentation switches;
- Canny-only event parameters, because the native baseline supports the
  paper's average and maximum pooling methods;
- inactive model flags that never reach the rewritten forward pass; and
- `keep_last_epochs`, because the native trainer never performs destructive
  checkpoint retention.

The pretraining recipes also express Adam and cosine scheduling directly.
Their former `optimizer: composite` wrapper represented one Fairseq parameter
group and did not add a mathematical operation.

TensorBoard stores the typed configuration, including defaults and command-line
overrides, under `run/config`. Archive that event directory with checkpoints.

## Workflow checks

`cpu_smoke_pretraining.yaml` and `cpu_smoke_finetuning.yaml` reduce model width,
depth, data volume, and update count so a developer can test the end-to-end
Animal2Vec 1.0 flow on a CPU. Their outputs do not reproduce paper metrics.

## Future A2V2 configurations

Place new experiment families under `configs/a2v2/` or a more specific
subdirectory below it. Keep each new recipe's method name, dataset, and stage
visible in its path. Document which Animal2Vec 1.0 recipe serves as its control,
and retain that control recipe without value changes.
