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

## Opt-in modern examples

`modern/rope_cls_geglu_pretrain.yaml` and
`modern/rope_cls_geglu_finetune.yaml` form one checkpoint-compatible encoder
pair. Both recipes use the MeerKAT sample rate, convolution geometry, width,
depth, head count, and eight-layer prenet from the published large control.
They start a separate checkpoint family under `checkpoints/modern_*`.

The pair selects RoPE, strict Flash attention, a learned CLS token, packed
GEGLU blocks, DeepScaleLM initialization, activation checkpointing, AdaGC,
AdamW8bit, cosine weight-decay annealing, and `torch.compile`. The pretraining
recipe adds a CLS regression objective while retaining frame reconstruction.
The fine-tuning recipe selects a sequence-level CLS classifier.

These files illustrate the modern path. They do not define a paper baseline
and cannot load the published encoder weights. CLS and GEGLU add parameters
and change tensor shapes. DeepScaleLM also changes residual equations.

### Architecture and checkpoint choices

| Field | Values and contract |
| --- | --- |
| `model.position_encoding` | `legacy` reads the old ALiBi flag. `alibi`, `rope`, and `none` select a strategy. RoPE needs an even head dimension and uses `rope_theta`, which defaults to 10,000. |
| `model.attention_backend` | `legacy` keeps manual attention for legacy position settings and resolves explicit RoPE or no-position models to SDPA. `manual` gives a reference path. `sdpa` permits PyTorch kernel fallback. `flash` forces the Flash SDPA backend and raises an error when the kernel cannot run. ALiBi uses manual attention. |
| `model.use_cls_token` | Adds a learned encoder token and a pretraining CLS predictor. The default `false` keeps legacy state keys. |
| `model.cls_loss_weight` | Scales direct student and EMA-teacher CLS regression. The default 1.0 counts one CLS vector like one masked frame vector. |
| `model.classification_head` | `frame` emits event-localization logits. `cls` emits one multilabel vector per recording and requires a CLS-enabled pretrained encoder. |
| `model.ffn_type` | `mlp` preserves the legacy blocks. `geglu` creates a packed value/gate projection with different state shapes. |
| `model.initialization` | `legacy` preserves the baseline initializer. `deepscale_lm` changes new-model initialization and residual scaling. A loaded checkpoint remains authoritative. |

Keep the architecture fields, `task.sample_rate`,
`task.conv_feature_layers`, and audio prenet depth equal across a pretraining
checkpoint and its fine-tuning recipe. Strict state loading rejects missing
CLS or GEGLU tensors and incompatible shapes.

### Execution and optimization policy

| Field | Values and contract |
| --- | --- |
| `model.checkpoint_activations` | Existing execution option. It uses non-reentrant block recomputation with RNG preservation during gradient-enabled training and adds no model tensors. |
| `common.torch_compile` | Enables in-place module compilation before DDP. The default `false` keeps eager execution. |
| `common.torch_compile_backend`, `torch_compile_mode`, `torch_compile_dynamic` | Pass values to `nn.Module.compile`. The workflow records them in config and summaries. |
| `common.torch_compile_fullgraph` | `false` permits graph breaks around deterministic NumPy masking and dynamic selection. `true` serves as a strict diagnostic and can reject layerdrop or masking control. |
| `optimization.gradient_clip_method` | `global` keeps the legacy clip. `none` disables clipping. `adagc` uses `clip_norm` during warm-up and then clips each parameter tensor against its norm history. |
| `optimization.adagc_beta`, `adagc_relative_clip`, `adagc_warmup_updates` | Defaults: 0.99, 1.04, and 100 successful updates. AdaGC state advances after an optimizer update succeeds. |
| `optimizer._name` | Adds `adamw`, `adam8bit`, and `adamw8bit` beside the Fairseq-compatible `adam` control. |
| `optimizer.min_8bit_size` | Defaults to 4,096. bitsandbytes keeps smaller tensor state in FP32. |
| `optimizer.weight_decay_schedule` | `constant` keeps the legacy fixed value. `cosine` anneals nonzero groups from `weight_decay` to `weight_decay_end`; an omitted endpoint means 0.0. |

Install the optional optimizer extra with:

```bash
python -m pip install -e '.[bnb]'
```

The pinned range is `bitsandbytes>=0.49,<0.50`. Version 0.49.2 does not ship
a CUDA 13.3 binary. The tested CUDA 13.3 host loaded the packaged CUDA 13.0
binary with `BNB_CUDA_VERSION=130`. Use that override when the selected
binary matches the host driver and toolkit contract. A source build provides
the other CUDA 13.3 route. CUDA versions covered by the wheel need no override.

## Adding another A2V2 family

Place each new family below `configs/a2v2/` or another named subdirectory.
Keep the method, dataset, and stage visible in the path. Name the Animal2Vec
1.0 recipe that supplies its control and retain the control file without value
changes.
