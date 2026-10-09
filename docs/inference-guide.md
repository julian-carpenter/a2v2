# Inference windows and event fusion

`InferenceRunner` handles resampling, frame timestamps, recording duration,
probabilities, and event fusion for a native checkpoint. Use that API unless
you need to postprocess frame predictions from another source. A pretraining
checkpoint returns embeddings and timestamps without probabilities/events.
A CLS classifier returns window-level CLS predictions rather than frame events.

## Overlapping inference windows

Default inference uses disjoint ten-second windows. To give events near a
window boundary another contextual view, request overlap:

```python
from a2v2 import InferenceRunner
from a2v2.data import load_audio

runner = InferenceRunner("finetune/checkpoint_best.pt")
audio, sample_rate = load_audio("recording.wav")
result = runner.run_tensor(
    audio, sample_rate,
    segment_seconds=10.0,
    overlap_seconds=1.0,
    overlap_method="mean",  # "max" favors the strongest contextual prediction.
)
runner.write_events("events.tsv", result.events)
```

Only load a checkpoint from a source you trust: native `.pt` deserialization
uses pickle. For the command-line event exporter:

```bash
a2v2-infer finetune/checkpoint_best.pt recording.wav events.tsv \
  --segment-seconds 10 --overlap-seconds 1 --overlap-method mean
```

Choose `0 <= overlap_seconds < segment_seconds`. With positive overlap,
the runner rounds the hop down to a whole frontend stride so window frames
share an integer recording-frame grid. The realized overlap can exceed the
requested value by less than one frame hop, plus sample rounding. Each output
frame appears once in chronological order, including the partial last window.
The runner stops once a window reaches the recording end.

`mean` averages each class's sigmoid probabilities over windows covering a
frame. `max` selects the largest probability per class. Both run **before**
optional cross-class `event_detection` voting, temporal smoothing, and
thresholding. These are probability merges, not logit averages or votes on
already fused event intervals. Mean can suppress false positives; max favors
recall and can raise false positives. Tune overlap and merging on validation;
no improvement on a particular corpus is guaranteed.

The runner averages overlapping **frame embeddings** regardless of the
probability merge choice. It keeps CLS embeddings/predictions **per window**,
because each CLS vector describes its whole window. More overlap therefore
produces more CLS rows; those rows are not aligned to the merged frame axis.
An overlapping pretraining run still returns `probabilities=None` and
`events=None`. `overlap_seconds=0` preserves the existing output path exactly.
Overlap costs extra forward passes: the approximate long-recording multiplier
is `segment_seconds / (segment_seconds - overlap_seconds)`.

## Calling fuse_probabilities yourself

Import it from `a2v2.workflows`. It smooths frame scores, thresholds them, and
joins each label's consecutive active frames into half-open event intervals.
It neither resamples audio nor merges inference windows.

| Argument | Source and units |
| --- | --- |
| `probabilities` | CPU tensor `[T,C]` of sigmoid probabilities from a frame classifier, e.g. `result.probabilities`. Independent labels may overlap. Do not supply logits or `[windows,C]` CLS predictions. |
| `timestamps` | Tensor `[T]` of frame-center times in seconds on the recording timeline, e.g. `result.timestamps`. Supply one increasing timestamp per row. Use `feature_timestamps(T, model_sample_rate, conv_layers)` plus the window's start offset if computing them yourself. |
| `threshold` | Probability in `[0,1]`, often `runner.config.criterion.metric_threshold`. Select on validation; lower usually raises recall and false positives. The comparison is inclusive (`>=`). |
| `method` | `"avg"` or `"max"`, often `runner.config.criterion.event_method`. This pools adjacent **time frames**, not classes or inference windows. |
| `window_frames` | Positive integer temporal-pooling width. For desired `window_seconds`, total stride `q`, and model sample rate `f`, use `max(1, round(window_seconds * f / q))`. At 8000 Hz with `q=40`, 0.1 seconds means 20 frames. Use the model rate after resampling. |
| `recording_duration` | Actual recording length in seconds, `audio.shape[-1] / source_sample_rate`, before resampling/padding. It is the whole recording duration, not the window duration or final timestamp. It caps event ends and supplies a hop for a single-frame input. |

The frame rate is `model_sample_rate / product(conv_stride)`. A kernel's
receptive-field width is a different quantity. `sigma_s` in the training
criterion expresses pooling duration in seconds; `fusion_window_seconds`
overrides it in `run_tensor`. The runner performs this conversion for you.

This example runs without a checkpoint or audio file. Three probability frames
have centers at 25, 30, and 35 ms, with one active frame. The result spans
30 to 35 ms, not 30 to 40 ms:

```python
import torch
from a2v2.workflows import fuse_probabilities

events = fuse_probabilities(
    torch.tensor([[0.1], [0.9], [0.1]]),
    torch.tensor([0.025, 0.030, 0.035]),
    threshold=0.5,
    method="avg",
    window_frames=1,  # No temporal smoothing.
    recording_duration=0.040,
)
assert len(events) == 1
assert abs(events[0].start_seconds - 0.030) < 1e-6
assert abs(events[0].end_seconds - 0.035) < 1e-6
```

For a loaded frame-classifier runner, obtain the arguments as follows:

```python
from math import prod
from a2v2.workflows import fuse_probabilities

stride = prod(layer[2] for layer in runner.config.task.conv_feature_layers)
frame_rate = runner.config.task.sample_rate / stride
events = fuse_probabilities(
    result.probabilities, result.timestamps,
    threshold=runner.config.criterion.metric_threshold,
    method=runner.config.criterion.event_method,
    window_frames=max(1, round(0.1 * frame_rate)),
    recording_duration=audio.shape[-1] / sample_rate,
)
```

Trim predictions from padded audio before calling the function. The helper
uses the median timestamp hop to extend the last active frame and caps that
end at `recording_duration`; it cannot infer which score rows came from padding.
Keep the existing center-based timing convention for paper comparisons.
