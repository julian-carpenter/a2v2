#!/usr/bin/env python3
"""Exhaustively audit the released MeerKAT audio and annotation archive.

The paper reports counts, duration, sample rate, and label coverage for a large
collection of ten-second recordings. This program reads every WAV header and
HDF5 label file, checks structural invariants, compares sample-index and
second-based annotations, and emits a machine-readable JSON report.

The audit uses worker processes because the archive contains hundreds of
thousands of small files. Workers return bounded error examples while counters
retain the complete number of failures. Exit status zero means all structural
checks and the published WAV count passed; publication discrepancies remain
visible as separate report fields.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import soundfile


EXPECTED_LABELS = (
    "beep",
    "synch",
    "sn",
    "cc",
    "ld",
    "oth",
    "mo",
    "al",
    "soc",
    "agg",
    "eating",
)
REQUIRED_KEYS = ("start_frame_lbl", "end_frame_lbl", "lbl_cat")
ALL_KEYS = (
    "start_frame_lbl",
    "end_frame_lbl",
    "lbl_cat",
    "foc",
    "start_time_lbl",
    "end_time_lbl",
    "lbl",
)
MAX_ERROR_EXAMPLES = 50


def _batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield bounded slices so each worker receives several filesystem paths."""

    for start in range(0, len(values), size):
        yield values[start : start + size]


def _bounded_extend(target: list[Any], values: Iterable[Any]) -> None:
    """Append diagnostic examples without letting reports grow without bound."""

    remaining = MAX_ERROR_EXAMPLES - len(target)
    if remaining > 0:
        target.extend(list(values)[:remaining])


def _decode_label(value: Any) -> str:
    """Convert byte or scalar HDF5 label values into readable text."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _audit_wav_batch(paths: Sequence[str]) -> dict[str, Any]:
    """Inspect one group of WAV headers and summarize formats and durations."""

    formats: Counter[tuple[int, int, int, str, str]] = Counter()
    errors: list[dict[str, str]] = []
    frames: list[tuple[str, int]] = []
    total_frames = 0
    for encoded in paths:
        path = Path(encoded)
        try:
            info = soundfile.info(path)
        except Exception as exc:  # libsndfile exceptions vary by release
            errors.append({"path": str(path), "error": repr(exc)})
            continue
        key = (
            int(info.samplerate),
            int(info.channels),
            int(info.frames),
            str(info.format),
            str(info.subtype),
        )
        formats[key] += 1
        frames.append((path.stem, int(info.frames)))
        total_frames += int(info.frames)
    return {
        "formats": formats,
        "errors": errors[:MAX_ERROR_EXAMPLES],
        "error_count": len(errors),
        "frames": frames,
        "total_frames": total_frames,
    }


def _audit_label_batch(entries: Sequence[tuple[str, int]]) -> dict[str, Any]:
    """Validate one group of HDF5 event files against paired waveform lengths.

    Args:
        entries: Pairs of label paths and waveform lengths in samples.

    Returns:
        Mergeable counters, extrema, time-conversion error, and bounded error
        examples. Event ends must be greater than starts and must not exceed
        the paired waveform length.
    """

    result: dict[str, Any] = {
        "files": 0,
        "empty_files": 0,
        "event_files": 0,
        "events": 0,
        "focal_events": 0,
        "categories": Counter(),
        "category_labels": Counter(),
        "event_count_per_file": Counter(),
        "key_sets": Counter(),
        "dtype_sets": Counter(),
        "errors": Counter(),
        "error_examples": [],
        "max_end": None,
        "min_start": None,
        "maximum_time_error_seconds": 0.0,
    }
    for encoded, waveform_frames in entries:
        path = Path(encoded)
        result["files"] += 1
        try:
            with h5py.File(path, "r") as handle:
                keys = tuple(sorted(handle.keys()))
                result["key_sets"][keys] += 1
                missing = [key for key in REQUIRED_KEYS if key not in handle]
                if missing:
                    result["errors"]["missing_required_keys"] += 1
                    _bounded_extend(
                        result["error_examples"],
                        ({"path": str(path), "error": "missing_required_keys", "keys": missing},),
                    )
                    continue
                arrays = {
                    key: np.asarray(handle[key][:])
                    for key in ALL_KEYS
                    if key in handle
                }
                result["dtype_sets"][tuple(
                    (key, str(arrays[key].dtype)) for key in sorted(arrays)
                )] += 1
        except Exception as exc:
            result["errors"]["unreadable_hdf5"] += 1
            _bounded_extend(
                result["error_examples"],
                ({"path": str(path), "error": "unreadable_hdf5", "detail": repr(exc)},),
            )
            continue

        starts = arrays["start_frame_lbl"].reshape(-1)
        ends = arrays["end_frame_lbl"].reshape(-1)
        categories = arrays["lbl_cat"].reshape(-1)
        focal = arrays.get("foc", np.zeros_like(starts)).reshape(-1)
        lengths = {
            "start_frame_lbl": len(starts),
            "end_frame_lbl": len(ends),
            "lbl_cat": len(categories),
            "foc": len(focal),
        }
        for optional in ("start_time_lbl", "end_time_lbl", "lbl"):
            if optional in arrays:
                lengths[optional] = len(arrays[optional].reshape(-1))
        if len(set(lengths.values())) != 1:
            result["errors"]["unequal_array_lengths"] += 1
            _bounded_extend(
                result["error_examples"],
                ({"path": str(path), "error": "unequal_array_lengths", "lengths": lengths},),
            )
            continue

        event_count = len(starts)
        result["event_count_per_file"][event_count] += 1
        result["events"] += event_count
        if event_count == 0:
            result["empty_files"] += 1
            continue
        result["event_files"] += 1

        integral = all(
            np.all(np.equal(values, np.floor(values)))
            for values in (starts, ends, categories, focal)
        )
        if not integral:
            result["errors"]["non_integral_event_values"] += 1
            _bounded_extend(
                result["error_examples"],
                ({"path": str(path), "error": "non_integral_event_values"},),
            )
        starts_i = starts.astype(np.int64)
        ends_i = ends.astype(np.int64)
        categories_i = categories.astype(np.int64)
        focal_i = focal.astype(np.int64)

        invalid_bounds = (starts_i < 0) | (ends_i <= starts_i) | (ends_i > waveform_frames)
        if invalid_bounds.any():
            result["errors"]["invalid_event_bounds"] += int(invalid_bounds.sum())
            indices = np.flatnonzero(invalid_bounds)[:3]
            _bounded_extend(
                result["error_examples"],
                ({
                    "path": str(path),
                    "error": "invalid_event_bounds",
                    "index": int(index),
                    "start": int(starts_i[index]),
                    "end": int(ends_i[index]),
                    "waveform_frames": waveform_frames,
                } for index in indices),
            )
        invalid_categories = (categories_i < 0) | (categories_i >= len(EXPECTED_LABELS))
        if invalid_categories.any():
            result["errors"]["invalid_categories"] += int(invalid_categories.sum())
        invalid_focal = ~np.isin(focal_i, (0, 1))
        if invalid_focal.any():
            result["errors"]["invalid_focal_values"] += int(invalid_focal.sum())

        result["categories"].update(int(value) for value in categories_i)
        result["focal_events"] += int((focal_i == 1).sum())
        result["max_end"] = max(
            int(ends_i.max()),
            result["max_end"] if result["max_end"] is not None else int(ends_i.max()),
        )
        result["min_start"] = min(
            int(starts_i.min()),
            result["min_start"] if result["min_start"] is not None else int(starts_i.min()),
        )

        if "lbl" in arrays:
            labels = [_decode_label(value) for value in arrays["lbl"].reshape(-1)]
            result["category_labels"].update(
                (int(category), label)
                for category, label in zip(categories_i, labels, strict=True)
            )
        if "start_time_lbl" in arrays and "end_time_lbl" in arrays:
            starts_t = arrays["start_time_lbl"].reshape(-1).astype(np.float64)
            ends_t = arrays["end_time_lbl"].reshape(-1).astype(np.float64)
            error = max(
                float(np.max(np.abs(starts_t - starts_i / 8_000))),
                float(np.max(np.abs(ends_t - ends_i / 8_000))),
            )
            result["maximum_time_error_seconds"] = max(
                result["maximum_time_error_seconds"], error
            )
    return result


def _merge_counter(target: Counter[Any], source: Counter[Any]) -> None:
    """Accumulate one worker's categorical counts into the process total."""

    target.update(source)


def _hash_names(names: Iterable[str]) -> str:
    """Hash a sorted filename set independently of filesystem enumeration order."""

    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _counter_rows(counter: Counter[Any], names: Sequence[str]) -> list[dict[str, Any]]:
    """Convert tuple-key counters into stable JSON table rows."""

    rows = []
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        components = key if len(names) > 1 and isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(names, components, strict=True)), "count": count})
    return rows


def main() -> int:
    """Audit the archive, write its JSON evidence, and return pass/fail status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=512)
    arguments = parser.parse_args()

    wav_dir = arguments.root / "wav" / "08000Hz"
    label_dir = arguments.root / "lbl" / "08000Hz"
    wav_paths = sorted(str(path) for path in wav_dir.iterdir() if path.suffix.lower() == ".wav")
    label_paths = sorted(str(path) for path in label_dir.iterdir() if path.suffix.lower() == ".h5")
    wav_stems = {Path(path).stem for path in wav_paths}
    label_stems = {Path(path).stem for path in label_paths}

    wav_formats: Counter[tuple[int, int, int, str, str]] = Counter()
    wav_errors: list[dict[str, str]] = []
    wav_error_count = 0
    total_frames = 0
    frames_by_stem: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for partial in executor.map(
            _audit_wav_batch,
            _batches(wav_paths, arguments.batch_size),
            chunksize=1,
        ):
            _merge_counter(wav_formats, partial["formats"])
            wav_error_count += partial["error_count"]
            _bounded_extend(wav_errors, partial["errors"])
            total_frames += partial["total_frames"]
            frames_by_stem.update(partial["frames"])

    label_entries = [
        (path, frames_by_stem.get(Path(path).stem, -1)) for path in label_paths
    ]
    label_totals: dict[str, Any] = {
        "files": 0,
        "empty_files": 0,
        "event_files": 0,
        "events": 0,
        "focal_events": 0,
        "categories": Counter(),
        "category_labels": Counter(),
        "event_count_per_file": Counter(),
        "key_sets": Counter(),
        "dtype_sets": Counter(),
        "errors": Counter(),
        "error_examples": [],
        "max_end": None,
        "min_start": None,
        "maximum_time_error_seconds": 0.0,
    }
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for partial in executor.map(
            _audit_label_batch,
            _batches(label_entries, arguments.batch_size),
            chunksize=1,
        ):
            for key in ("files", "empty_files", "event_files", "events", "focal_events"):
                label_totals[key] += partial[key]
            for key in (
                "categories",
                "category_labels",
                "event_count_per_file",
                "key_sets",
                "dtype_sets",
                "errors",
            ):
                _merge_counter(label_totals[key], partial[key])
            _bounded_extend(label_totals["error_examples"], partial["error_examples"])
            if partial["max_end"] is not None:
                label_totals["max_end"] = max(
                    partial["max_end"],
                    label_totals["max_end"]
                    if label_totals["max_end"] is not None
                    else partial["max_end"],
                )
            if partial["min_start"] is not None:
                label_totals["min_start"] = min(
                    partial["min_start"],
                    label_totals["min_start"]
                    if label_totals["min_start"] is not None
                    else partial["min_start"],
                )
            label_totals["maximum_time_error_seconds"] = max(
                label_totals["maximum_time_error_seconds"],
                partial["maximum_time_error_seconds"],
            )

    structural_error_count = wav_error_count + sum(label_totals["errors"].values())
    report = {
        "root": str(arguments.root.resolve()),
        "workers": arguments.workers,
        "batch_size": arguments.batch_size,
        "paper_reference": {
            "samples": 384_592,
            "hours": 1_068,
            "labelled_samples": 66_398,
            "sample_rate": 8_000,
            "nominal_seconds": 10,
            "claimed_quantization_bits": 16,
        },
        "paths": {
            "wav_count": len(wav_paths),
            "label_count": len(label_paths),
            "paired_stem_count": len(wav_stems & label_stems),
            "wav_without_label_count": len(wav_stems - label_stems),
            "label_without_wav_count": len(label_stems - wav_stems),
            "wav_without_label_examples": sorted(wav_stems - label_stems)[:MAX_ERROR_EXAMPLES],
            "label_without_wav_examples": sorted(label_stems - wav_stems)[:MAX_ERROR_EXAMPLES],
            "all_stems_sha256": _hash_names(wav_stems | label_stems),
            "wav_stems_sha256": _hash_names(wav_stems),
            "label_stems_sha256": _hash_names(label_stems),
        },
        "audio": {
            "total_frames": total_frames,
            "total_hours": total_frames / 8_000 / 3_600,
            "format_groups": _counter_rows(
                wav_formats,
                ("sample_rate", "channels", "frames", "format", "subtype"),
            ),
            "unreadable_count": wav_error_count,
            "error_examples": wav_errors,
        },
        "labels": {
            "empty_files": label_totals["empty_files"],
            "event_files": label_totals["event_files"],
            "paper_labelled_sample_difference": label_totals["event_files"] - 66_398,
            "events": label_totals["events"],
            "focal_events": label_totals["focal_events"],
            "min_start": label_totals["min_start"],
            "max_end": label_totals["max_end"],
            "maximum_time_error_seconds": label_totals["maximum_time_error_seconds"],
            "category_counts": _counter_rows(label_totals["categories"], ("category",)),
            "category_label_counts": _counter_rows(
                label_totals["category_labels"], ("category", "label")
            ),
            "event_count_per_file": _counter_rows(
                label_totals["event_count_per_file"], ("events",)
            ),
            "key_sets": _counter_rows(label_totals["key_sets"], ("keys",)),
            "dtype_sets": _counter_rows(label_totals["dtype_sets"], ("dtypes",)),
            "error_counts": dict(sorted(label_totals["errors"].items())),
            "error_examples": label_totals["error_examples"],
        },
        "structural_error_count": structural_error_count,
        "all_structural_checks_pass": (
            structural_error_count == 0
            and wav_stems == label_stems
            and len(wav_paths) == 384_592
        ),
        "publication_discrepancies": {
            "pcm_24_files": sum(
                count for key, count in wav_formats.items() if key[-1] == "PCM_24"
            ),
            "event_bearing_vs_reported_labelled": label_totals["event_files"] - 66_398,
            "released_archive_contains_no_split_manifests": True,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "wav_count": len(wav_paths),
        "label_count": len(label_paths),
        "event_files": label_totals["event_files"],
        "events": label_totals["events"],
        "structural_error_count": structural_error_count,
        "all_structural_checks_pass": report["all_structural_checks_pass"],
        "output": str(arguments.output),
    }, sort_keys=True))
    return 0 if report["all_structural_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
