# Pretraining student and teacher variance logging

Date: 2026-07-28

## Goal

Add the Animal2Vec 1.0 collapse diagnostics named `pred_var` and
`target_var` to A2V2 pretraining logs. Researchers use their trajectories to
check whether the student predictions or EMA-teacher targets have collapsed
toward a constant representation.

The change must preserve model outputs, gradients, optimizer updates, EMA
updates, checkpoint state, and deterministic resume behavior.

## Archived definition

The archived implementation flattens each masked tensor from any leading
dimensions to `[N, D]`. For feature coordinate \(d\), it computes the unbiased
sample variance from global sufficient statistics:

\[
s_d^2 =
\frac{\sum_{i=1}^{N} z_{i,d}^2}{N-1}
-
\frac{\left(\sum_{i=1}^{N} z_{i,d}\right)^2}{N(N-1)}.
\]

The logged scalar is:

\[
\operatorname{legacy\_var}(z)
=
\frac{1}{D}\sum_{d=1}^{D}\sqrt{s_d^2 + 10^{-6}}.
\]

The historical key says `var`, although the square root makes the value a mean
feature-wise sample standard deviation. A2V2 will keep the historical names
because existing dashboards and reference logs use them.

The prediction input is the decoder output at masked positions. The target
input is the normalized EMA-teacher representation at the same positions.
A2V2 already returns both detached-compatible tensors in
`PretrainingOutput.predictions` and `PretrainingOutput.targets`.

## Placement and data flow

`TrainingEngine.step` will compute diagnostics after each forward call and
before backward:

1. detach the selected prediction and target tensors;
2. cast them to FP32, matching the archived implementation;
3. compute count, sum, and squared sum for each feature coordinate;
4. combine those statistics across distributed ranks with one all-reduce;
5. evaluate the archived formula under `torch.no_grad()`; and
6. average each scalar across the microbatches in one optimizer attempt.

The engine will add optional `pred_var` and `target_var` fields to
`UpdateResult`. Fine-tuning outputs do not carry masked prediction tensors, so
their result fields remain `None`.

The workflow will include the two keys in rank-zero JSON update records when
the values exist. It will also include the most recent pair in each rank's
terminal `training_summary`, which gives short burn-ins a diagnostic even when
their update count does not reach `common.log_interval`.

## Distributed semantics

Each distributed rank processes a different local shard. Averaging local
standard deviations would give the wrong answer because standard deviation is
not linear. The diagnostic must reduce count, sum, and squared sum before it
applies the variance formula.

Predictions and targets share `[N, D]`, so one packed statistics tensor can
carry both sets of moments in one collective. The diagnostic reduction has no
autograd edge. DDP gradient buckets, loss normalization, parameter traversal,
and optimizer arithmetic remain unchanged.

All ranks execute the same number of microbatches in an update. Each rank
therefore enters the diagnostic collective in the same order. The resulting
scalars match on all ranks, while only rank zero prints update records.

## Numerical and error behavior

The archived estimator requires at least two masked vectors. Normal pretraining
batches exceed that bound. The helper will raise a clear `ValueError` if a
synthetic or malformed forward result supplies fewer than two vectors, or if
prediction and target shapes differ.

FP16 and BF16 inputs will use FP32 statistics. The helper will preserve the
archived \(10^{-6}\) stabilizer. It will not clamp negative variance caused by
catastrophic cancellation because that would hide the same instability the
diagnostic should expose.

## Tests

CPU tests will cover:

- the archived formula on hand-constructed prediction and target tensors;
- shape and minimum-count validation;
- one tiny pretraining update returning finite `pred_var` and `target_var`;
- one CPU smoke CLI run printing both JSON keys;
- a fine-tuning update leaving both optional fields absent; and
- no change to loss, gradient, checkpoint, and resume tests.

The existing NCCL resume probe will retain the two result values in its
comparison snapshot. The CPU machine will collect but skip that GPU test.

## Documentation

The README and code guide will define the historical names, formula, and
interpretation. They will advise researchers to plot both series with loss,
gradient norm, EMA decay, and AMP scale during long pretraining runs.
