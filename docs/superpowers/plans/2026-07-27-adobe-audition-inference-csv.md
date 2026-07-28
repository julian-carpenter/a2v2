# Adobe Audition inference CSV implementation plan

**Status:** Completed and verified on 2026-07-27.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `--audition` inference mode that writes A2V2 events in the
marker CSV schema produced by Animal2Vec 1.0.

**Architecture:** Keep event production and the native TSV unchanged. Add one
keyword-controlled serialization branch to `InferenceRunner.write_events`,
then route a `store_true` CLI flag into that branch. Use Python's standard
`datetime.timedelta` formatter to match the legacy time representation.

**Tech Stack:** Python 3.10+, PyTorch, SoundFile, argparse, pytest.

## Global constraints

- Add no runtime dependency.
- Keep exactly six Python files under `a2v2/`.
- Do not change probability fusion, event boundaries, labels, or scores.
- Preserve the native TSV byte format when callers omit `audition=True`.
- Write Audition files as UTF-8 tab-delimited text with a `.csv` filename
  recommended in documentation.
- Follow the contract in
  `docs/superpowers/specs/2026-07-27-adobe-audition-inference-csv-design.md`.
- This standalone directory has no Git metadata, so this plan omits commit
  steps.

---

### Task 1: Add the exact Audition marker serializer

**Files:**

- Modify: `tests/integration/test_inference.py`
- Modify: `a2v2/workflows.py`

**Interfaces:**

- Consumes: `EventInterval` and `InferenceRunner.config.task.unique_labels`.
- Produces:
  `InferenceRunner.write_events(path, events, *, audition: bool = False)`.

- [x] **Step 1: Write failing exact-output tests**

Import `EventInterval` and add tests with hand-written output literals:

```python
def test_audition_csv_matches_legacy_marker_schema(tmp_path: Path) -> None:
    runner = InferenceRunner(_checkpoint(tmp_path), device=torch.device("cpu"))
    events = (
        EventInterval(0, 10, 20, 65.25, 66.5, 0.8751),
        EventInterval(1, 1, 2, 2.5, 3.0, 0.1),
    )
    output = tmp_path / "predictions.csv"

    runner.write_events(output, events, audition=True)

    assert output.read_text(encoding="utf-8") == (
        "Name\tStart\tDuration\tTime Format\tType\tDescription\n"
        "focal\t0:00:02.500000\t0:00:00.500000\tdecimal\tCue\t0.100\n"
        "call\t0:01:05.250000\t0:00:01.250000\tdecimal\tCue\t0.875\n"
    )


def test_audition_csv_with_no_events_contains_header(tmp_path: Path) -> None:
    runner = InferenceRunner(_checkpoint(tmp_path), device=torch.device("cpu"))
    output = tmp_path / "empty.csv"

    runner.write_events(output, (), audition=True)

    assert output.read_text(encoding="utf-8") == (
        "Name\tStart\tDuration\tTime Format\tType\tDescription\n"
    )
```

The first test catches wrong delimiters, columns, onset or duration formatting,
score precision, event ordering, and missing final newline. The second catches
a writer that emits an empty file when the model finds no event.

- [x] **Step 2: Verify the tests fail for the missing API**

Run:

```bash
rtk proxy pytest -q \
  tests/integration/test_inference.py::test_audition_csv_matches_legacy_marker_schema \
  tests/integration/test_inference.py::test_audition_csv_with_no_events_contains_header
```

Expected: both tests fail because `write_events` does not accept `audition`.

- [x] **Step 3: Implement the serialization branch**

Import `timedelta` from `datetime`. Change the writer signature to:

```python
def write_events(
    self,
    path: str | Path,
    events: tuple[EventInterval, ...],
    *,
    audition: bool = False,
) -> None:
```

For Audition mode:

1. start with the exact six-column header;
2. sort events by `(start_seconds, end_seconds, label_index)`;
3. format onset with `str(timedelta(seconds=event.start_seconds))`;
4. format duration with
   `str(timedelta(seconds=event.end_seconds - event.start_seconds))`;
5. write `decimal`, `Cue`, and `f"{event.score:.3f}"`; and
6. join fields with tabs and lines with `\n`.

Retain the existing native branch byte for byte.

- [x] **Step 4: Verify serializer tests and native regression**

Run:

```bash
rtk proxy pytest -q tests/integration/test_inference.py
```

Expected: all inference tests pass, including the existing native TSV test.

---

### Task 2: Route the command-line flag

**Files:**

- Modify: `tests/integration/test_inference.py`
- Modify: `a2v2/workflows.py`

**Interfaces:**

- Consumes: the `audition` keyword added in Task 1.
- Produces: `a2v2-infer ... output.csv --audition`.

- [x] **Step 1: Write failing parser and end-to-end tests**

Import `build_inference_parser`, `infer_main`, and `soundfile`. Add:

```python
def test_inference_parser_exposes_audition_flag() -> None:
    parser = build_inference_parser()
    positional = ["checkpoint.pt", "recording.wav", "predictions.csv"]

    assert parser.parse_args(positional).audition is False
    assert parser.parse_args([*positional, "--audition"]).audition is True


def test_infer_main_writes_audition_csv(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    audio = tmp_path / "recording.wav"
    output = tmp_path / "predictions.csv"
    soundfile.write(audio, torch.zeros(8_000).numpy(), 8_000)

    exit_code = infer_main([
        str(checkpoint),
        str(audio),
        str(output),
        "--audition",
        "--segment-seconds", "0.5",
        "--threshold", "0.5",
        "--method", "max",
        "--fusion-window-seconds", "0.02",
        "--device", "cpu",
    ])

    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert text.startswith(
        "Name\tStart\tDuration\tTime Format\tType\tDescription\n"
    )
    assert "\tdecimal\tCue\t" in text
    assert "call\t" in text
```

The parser test catches a missing or value-taking flag. The end-to-end test
catches failure to pass the parsed value from `infer_main` to the serializer.
It uses a real checkpoint, WAV file, model forward pass, and output file.

- [x] **Step 2: Verify the command tests fail**

Run:

```bash
rtk proxy pytest -q \
  tests/integration/test_inference.py::test_inference_parser_exposes_audition_flag \
  tests/integration/test_inference.py::test_infer_main_writes_audition_csv
```

Expected: the parser rejects `--audition`.

- [x] **Step 3: Implement CLI routing**

Add this parser option:

```python
parser.add_argument(
    "--audition",
    action="store_true",
    help="write Adobe Audition marker CSV instead of the native event TSV",
)
```

Pass `audition=arguments.audition` to `runner.write_events`. Update the
`infer_main` docstring to describe an event file rather than only a TSV.

- [x] **Step 4: Verify the complete inference test module**

Run:

```bash
rtk proxy pytest -q tests/integration/test_inference.py
```

Expected: all inference tests pass.

---

### Task 3: Document and verify the user workflow

**Files:**

- Modify: `README.md`
- Modify: `docs/code-guide.md`
- Modify: `docs/reproducing-paper.md`

**Interfaces:**

- Documents: native TSV and Adobe Audition CSV output modes.

- [x] **Step 1: Add the Adobe Audition example**

Add this command to the inference section:

```bash
a2v2-infer \
  /checkpoints/meerkat-finetune-fold0/checkpoint_best.pt \
  /audio/recording.wav \
  /predictions/recording.csv \
  --audition \
  --segment-seconds 10 \
  --device cuda
```

Show the six-column tab-delimited header and one marker row. State that Adobe
Audition accepts the `.csv` file even though the legacy format uses tabs.
Explain that `Start` stores onset and `Duration` stores offset minus onset.

- [x] **Step 2: Document the API and scope**

Show:

```python
runner.write_events(
    "/predictions/recording.csv",
    result.events,
    audition=True,
)
```

State that the flag changes file serialization and retains the A2V2 event set.
Point readers who need probabilities or custom event policy to
`InferenceResult`.

- [x] **Step 3: Run focused documentation and inference tests**

Run:

```bash
rtk proxy pytest -q \
  tests/unit/test_documentation.py \
  tests/integration/test_inference.py
```

Expected: all selected tests pass.

- [x] **Step 4: Run the complete CPU suite**

Run with multiprocessing permissions and caches disabled:

```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider
```

Expected: 166 tests pass and 14 CUDA tests skip.

- [x] **Step 5: Verify the installed command**

Build a wheel without dependency downloads, install it into a temporary
environment with system site packages, and run:

```bash
a2v2-infer --help
```

Require `--audition` in the rendered options. Inspect the wheel to confirm that
it still contains six `a2v2/*.py` files and the same three command entry points.
