# Adobe Audition inference CSV design

Date: 2026-07-27

## Goal

Add an `--audition` flag to `a2v2-infer`. Researchers can use the resulting
file with Adobe Audition's marker import workflow. A2V2 will reproduce the
marker-table schema written by the official Animal2Vec 1.0 inference script
while retaining its existing event TSV as the default output.

## Legacy file contract

The official
[`animal2vec_inference.py`](https://github.com/livingingroups/animal2vec/blob/main/animal2vec_inference.py)
writes a tab-delimited file with a `.csv` extension. Its first row contains
these six fields:

```text
Name	Start	Duration	Time Format	Type	Description
```

Each marker row stores:

| Column | A2V2 value |
| --- | --- |
| `Name` | Class label from the checkpoint configuration |
| `Start` | Event onset formatted through `str(datetime.timedelta(...))` |
| `Duration` | `end_seconds - start_seconds`, formatted as a `timedelta` |
| `Time Format` | Literal `decimal` |
| `Type` | Literal `Cue` |
| `Description` | Event score rounded to three digits after the decimal point |

A2V2 will use UTF-8, tab separators, `\n` line endings, and a final newline.
An event-free recording will produce the header and final newline.

## Scope

The flag changes serialization. It does not change frame probabilities,
thresholding, smoothing, event boundaries, class labels, or event scores.
A2V2 will write all events returned by `InferenceRunner.run_tensor`, sorted by
numeric onset, numeric offset, and label index.

The Animal2Vec 1.0 script also applied policy beyond Adobe's file contract: it
treated the final `focal` channel as metadata, suppressed lower-scoring
overlapping classes, added ` nf` to some names, and could write a secondary
prediction file. Those rules select and rename predictions. The `--audition`
flag will not reproduce them because doing so would change A2V2 inference
results under a formatting option. A later feature can add an explicit legacy
event-policy mode if researchers need those selection rules.

## Interfaces

`InferenceRunner.write_events` will accept a keyword-only Boolean:

```python
runner.write_events(path, events, audition=True)
```

The default `audition=False` retains the current columns:

```text
label	start_seconds	end_seconds	score
```

`build_inference_parser` will add:

```text
--audition
```

`infer_main` will pass that value to `write_events`. The output filename remains
the caller's positional argument. Documentation will recommend a `.csv`
extension for Audition mode and a `.tsv` extension for the native format.

## Error behavior

The serializer will reuse the label-index validation implied by the existing
writer: an event with an index outside `task.unique_labels` raises
`IndexError`. `timedelta` will reject non-finite time values. Inference-created
events already guarantee non-negative onsets and offsets bounded by the
recording duration.

## Tests

Tests will cover:

1. the parser's default and `--audition` states;
2. exact header, delimiter, field order, `timedelta` formatting, score
   precision, chronological order, and final newline;
3. an empty marker file;
4. the unchanged native TSV output; and
5. end-to-end routing from `infer_main --audition` to the marker serializer.

The complete CPU suite and wheel-installed `a2v2-infer --help` command will run
after the focused tests pass.
