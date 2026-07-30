# Minimal configuration recipes

## Goal

Make every checked-in YAML recipe readable as a self-contained experiment
description without changing the resolved Animal2Vec 1.0 baseline behavior.
The recipe should expose parameters that affect data, model mathematics,
optimization, validation, or output. It should not expose Fairseq, Hydra, or
logging controls that the native runtime ignores.

## Scope

The cleanup covers the six paper-baseline recipes, the two CPU smoke recipes,
and the two test fixtures. The typed compatibility parser remains capable of
reading archived Fairseq-style configurations embedded in checkpoints. This
separates two responsibilities:

- checked-in YAML files describe native experiments for researchers;
- `a2v2.config` remains a compatibility boundary for old artifacts.

No model, optimizer, checkpoint, dataset, or metric implementation changes are
part of this work.

## Field classification

A field is retained when changing it can change at least one of:

1. selected audio or labels;
2. convolution or Transformer shape;
3. student, teacher, mask, mixup, or decoder computation;
4. optimizer, scheduler, gradient accumulation, or AMP arithmetic;
5. validation timing, segmentation, or best-checkpoint selection;
6. distributed launch validation, checkpoint cadence, or TensorBoard output.

A field is removed from checked-in recipes when the native runtime accepts it
only for compatibility or never consumes the resulting value. These fields
are:

- `hydra`;
- `common.log_format`, `common.fp16_no_flatten_grads`, and
  `common.all_gather_list_size`;
- `task._name` and `task.verbose_tensorboard_logging`;
- `dataset.skip_invalid_size_inputs_valid_test`;
- `distributed_training.ddp_backend`;
- `criterion._name`, `criterion.report_accuracy`,
  `criterion.segmentation_metrics`, and `criterion.log_keys`;
- `criterion.use_focal_loss` and `criterion.label_smoothing`, because the
  native supervised baseline has one explicit focal-loss implementation;
- `criterion.maxfilt_s`, `criterion.max_duration_s`, and `criterion.lowP`,
  because they belong to the unsupported Canny event detector and do not
  affect `avg` or `max` fusion;
- `model._name`, `model.supported_modality`, `model.ema_encoder_only`, and
  `model.load_pretrain_weights`;
- `model.modalities.audio.mask_prob_adjust`, `inverse_mask`, `add_masks`, and
  `ema_local_encoder`;
- `checkpoint.keep_last_epochs`, because the native trainer does not delete
  historical epoch checkpoints.

An inactive value is also omitted when the native default has exactly the same
effect and the surrounding comment makes the absence clear. Examples include
zero channel masking in the Hyena fine-tuning recipe and zero drop path. This
keeps the recipe focused on active mechanisms.

## Structural simplification

The pretraining recipes currently encode Adam, its learning rate, and the
cosine scheduler inside Fairseq's one-group `composite` optimizer. They will be
rewritten into the same flat sections already used by fine-tuning:

```yaml
optimization:
  lr: [0.0001]

optimizer:
  _name: adam
  adam_betas: [0.9, 0.98]
  adam_eps: 1.0e-6
  weight_decay: 0.01

lr_scheduler:
  warmup_updates: 10000
```

The parser already translates both forms to the same immutable native
dataclasses. `_name: adam` remains because it chooses between the supported
optimizer algorithms. Scheduler `_name` is omitted because the native trainer
constructs the cosine scheduler directly.

No YAML anchors, includes, inheritance, or generated fragments will be added.
Some repetition is preferable because a researcher can understand and archive
one recipe without resolving another file.

## Commenting standard

Comments should explain consequences rather than repeat key names. For
quantitative controls they should state the relevant equation or dependency
where useful, followed by the experimental interpretation. In particular:

- effective global token budget is
  `max_tokens × world_size × update_freq`;
- convolution triples are `(output channels, kernel samples, stride)`;
- `average_top_k_layers` identifies the teacher or classifier
  representation;
- EMA and warmup values are measured in optimizer updates;
- event thresholds distinguish frame thresholding from IoU acceptance;
- masking probabilities may exceed one in the archived span-start
  parameterization and are not Bernoulli probabilities.

Comments should be compact enough that values remain visually prominent.

## Compatibility and verification

Before-and-after recipes must resolve to equivalent native configurations for
all behavior-bearing fields. Differences are allowed only in typed fields that
are themselves unused compatibility metadata, such as stored component names
or `keep_last_epochs`.

Verification consists of:

1. a recipe-policy test that rejects the known irrelevant keys from checked-in
   YAML files;
2. existing exact-value and architecture tests;
3. loading every recipe through the strict parser;
4. CPU smoke and integration tests;
5. a diagnostic comparison between the committed recipes and the cleaned
   recipes after native translation;
6. the complete test suite and whitespace validation.

## Self-review

The design contains no placeholders. It deliberately changes only YAML
representation and documentation, retains old-checkpoint parsing, and does not
introduce a second configuration system. "Minimal" means behavior-focused,
not reliance on undocumented defaults for architecture, training scale, or
evaluation.
