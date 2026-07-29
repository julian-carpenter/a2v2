# TensorBoard and Segmented Validation Design

## Goal

Add durable TensorBoard records for pretraining, fine-tuning, and standalone
checkpoint evaluation. Every labeled validation run must compute the
Animal2Vec 1.0 segmented-event measurements in addition to framewise
measurements.

## Archived behavior and compatibility boundary

`animal2vec/nn/utils.py::FusedSegmentationMixin` performs five operations:

1. Average- or maximum-pool each class probability series with a window of
   `round(feature_rate * sigma_s)` frames.
2. Threshold the pooled series and convert each contiguous run to an interval.
3. Match target and predicted intervals with the archived half-open
   `IntervalTree` interpretation of otherwise inclusive endpoints.
4. Emit one score/target sample per overlap, false negative, or false positive.
5. Record pairwise IoUs and counts for valid splits and mergers.

The native `legacy_segmented_evaluation` function already preserves the
archived pooling shift, endpoint handling, strict `IoU > threshold` decision,
score selection, and fixed-size output tensors. The current
`aggregate_segmented_metrics` function incorrectly removes rows with no
positive target. Those rows include false-positive predictions, so filtering
them can inflate segmented average precision.

The corrected aggregator will retain the complete flattened tensors, including
their zero padding. This matches the arrays passed by the archived logger to
`classification_report` and `average_precision_score`. It also preserves
false-positive samples. New metrics will use the configured fixed threshold:

- segmented precision, recall, F1, and accuracy over all class decisions;
- segmented classwise, macro, and micro average precision;
- focal-label best-threshold measurements retained from the native rewrite;
- IoU, split, and merger distributions for TensorBoard.

The baseline recipes use the `avg` method. Native validation will continue to
support the archived `avg` and `max` paths without SciPy or scikit-learn.

## TensorBoard directory and dependency

The implementation will use `torch.utils.tensorboard.SummaryWriter` and declare
the `tensorboard` package as a runtime dependency. PyTorch supplies the writer
API but imports the event-file implementation from that package.

Training resolves `common.tensorboard_logdir` as follows:

- an absolute value remains absolute;
- a relative value is placed under `checkpoint.save_dir`;
- an omitted value defaults to `tensorboard`.

The published recipes therefore write to `<save_dir>/tb`. Only rank zero opens
a writer. A resumed run uses `purge_step=checkpoint_update + 1` so TensorBoard
discards stale events written after the resume checkpoint.

The standalone evaluator accepts `--tensorboard-dir`. Its default is a
`tensorboard` directory next to the JSON report.

## Logged training measurements

At each configured stdout log interval, rank zero writes:

- `train/loss`
- `train/sample_size`
- `train/gradient_norm`
- `train/learning_rate`
- `train/skipped`
- `train/amp_scale` when AMP supplies a scale
- `pretrain/pred_var` and `pretrain/target_var` during pretraining

The writer also stores the serialized active configuration and model parameter
counts as run metadata. It flushes after each logging interval and closes
before `run_training` returns.

## Logged validation measurements

The shared `_validate` function remains the only implementation used by
scheduled training validation and `evaluate_finetuning_checkpoint.py`.
Fine-tuning validation retains each recording's frame and class axes long
enough to run event matching. Padding frames do not enter the matcher. The
pooling width uses the archived formula based on the model-output frame count,
collated source length, sample rate, and `sigma_s`.

Each validation writes:

- loss;
- framewise precision, recall, F1, accuracy, and average precision;
- segmented precision, recall, F1, accuracy, macro AP, and micro AP;
- classwise frame and segmented AP;
- frame and segmented micro/classwise precision-recall curves;
- nonzero IoU, split, and merger histograms per class;
- focal best-threshold precision, recall, and F1 when the label set contains
  `focal`.

Pretraining validation writes its loss. Training updates supply the
pretraining collapse diagnostics.

## Error handling

Configuration loading rejects an empty TensorBoard path. Validation raises a
clear error for unsupported segmentation methods. TensorBoard initialization
errors identify the requested event directory. Writer closure occurs on normal
completion and exception paths.

## Tests

Tests will establish four contracts:

1. A higher-scoring false-positive segment lowers segmented AP and precision.
2. Random and hand-built interval fixtures match an independent transcription
   of the archived matcher.
3. Native validation and the standalone report contain segmented metrics.
4. Real TensorBoard event files from pretraining, fine-tuning validation, and
   standalone evaluation contain the required scalar, curve, and histogram
   tags.

