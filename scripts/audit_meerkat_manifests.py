#!/usr/bin/env python3
"""Audit reconstructed MeerKAT split manifests and their relationships.

The public archive does not contain the exact split manifests used by the
paper. This program checks a reconstructed manifest directory for malformed
rows, missing files, duplicate membership, fold leakage, few-shot containment,
and validation coverage. It records raw file hashes and order-independent
membership hashes.

Passing this audit establishes internal consistency. It does not make a
reconstructed split authoritative, and the JSON provenance section states that
limitation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


FEW_SHOT_FRACTIONS = {0: 0.01, 1: 0.10, 2: 0.25, 3: 0.50, 4: 0.75}


def _sha256_bytes(path: Path) -> str:
    """Hash the exact bytes of one manifest, including row order."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _membership_hash(rows: list[tuple[str, int]]) -> str:
    """Hash sorted path/sample pairs so row order does not affect membership."""

    digest = hashlib.sha256()
    for relative, samples in sorted(rows):
        digest.update(f"{relative}\t{samples}\n".encode("utf-8"))
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    """Parse one root-plus-TSV manifest and audit its referenced files."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty manifest {path}")
    root = Path(lines[0]).resolve()
    rows: list[tuple[str, int]] = []
    malformed: list[dict[str, Any]] = []
    missing_audio: list[str] = []
    missing_labels: list[str] = []
    label_sizes: Counter[int] = Counter()
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != 2:
            malformed.append({"line": line_number, "value": line})
            continue
        relative, encoded_samples = fields
        try:
            samples = int(encoded_samples)
        except ValueError:
            malformed.append({"line": line_number, "value": line})
            continue
        rows.append((relative, samples))
        audio = root / relative
        if not audio.is_file():
            missing_audio.append(relative)
            continue
        parts = list(audio.parts)
        try:
            wav_index = max(
                index for index, part in enumerate(parts) if part.lower() == "wav"
            )
        except ValueError:
            continue
        parts[wav_index] = "lbl"
        label = Path(*parts).with_suffix(".h5")
        if label.is_file():
            label_sizes[label.stat().st_size] += 1
        else:
            missing_labels.append(relative)
    relatives = [relative for relative, _ in rows]
    duplicate_count = len(relatives) - len(set(relatives))
    return {
        "path": str(path),
        "root": str(root),
        "rows": rows,
        "row_count": len(rows),
        "unique_path_count": len(set(relatives)),
        "duplicate_count": duplicate_count,
        "sample_counts": Counter(samples for _, samples in rows),
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:20],
        "missing_audio_count": len(missing_audio),
        "missing_audio_examples": missing_audio[:20],
        "missing_label_count": len(missing_labels),
        "missing_label_examples": missing_labels[:20],
        "label_sizes": label_sizes,
        "raw_sha256": _sha256_bytes(path),
        "membership_sha256": _membership_hash(rows),
    }


def _public(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove full row data and convert counters into JSON-friendly tables."""

    return {
        key: value
        for key, value in manifest.items()
        if key != "rows"
    } | {
        "sample_counts": [
            {"samples": samples, "count": count}
            for samples, count in sorted(manifest["sample_counts"].items())
        ],
        "label_sizes": [
            {"bytes": size, "count": count}
            for size, count in sorted(manifest["label_sizes"].items())
        ],
    }


def main() -> int:
    """Audit all expected folds and write a provenance-rich JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_dir", type=Path)
    parser.add_argument("--dataset-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    arguments = parser.parse_args()

    dataset_audit = json.loads(arguments.dataset_audit.read_text(encoding="utf-8"))
    manifests = {
        path.stem: _read_manifest(path)
        for path in sorted(arguments.manifest_dir.glob("*.tsv"))
    }
    errors: list[str] = []
    expected_names = {"pretrain"}
    for fold in range(arguments.folds):
        expected_names.update({f"train_{fold}", f"valid_{fold}"})
        expected_names.update(f"train_{fold}_few_{index}" for index in FEW_SHOT_FRACTIONS)
    missing_manifests = sorted(expected_names - set(manifests))
    extra_manifests = sorted(set(manifests) - expected_names)
    if missing_manifests:
        errors.append(f"missing manifests: {missing_manifests}")
    if extra_manifests:
        errors.append(f"extra manifests: {extra_manifests}")

    for name, manifest in manifests.items():
        for field in (
            "duplicate_count",
            "malformed_count",
            "missing_audio_count",
        ):
            if manifest[field]:
                errors.append(f"{name}: {field}={manifest[field]}")
        if set(manifest["sample_counts"]) != {80_000}:
            errors.append(f"{name}: unexpected sample counts {dict(manifest['sample_counts'])}")

    pretrain_paths = {
        relative for relative, _ in manifests.get("pretrain", {}).get("rows", [])
    }
    dataset_wav_count = int(dataset_audit["paths"]["wav_count"])
    if len(pretrain_paths) != dataset_wav_count:
        errors.append(
            f"pretrain rows {len(pretrain_paths)} != dataset WAV count {dataset_wav_count}"
        )

    folds: list[dict[str, Any]] = []
    validation_frequency: Counter[str] = Counter()
    labelled_reference: set[str] | None = None
    for fold in range(arguments.folds):
        train_name = f"train_{fold}"
        valid_name = f"valid_{fold}"
        if train_name not in manifests or valid_name not in manifests:
            continue
        train = {relative for relative, _ in manifests[train_name]["rows"]}
        valid = {relative for relative, _ in manifests[valid_name]["rows"]}
        overlap = train & valid
        union = train | valid
        validation_frequency.update(valid)
        if overlap:
            errors.append(f"fold {fold}: {len(overlap)} train/valid overlaps")
        if not union <= pretrain_paths:
            errors.append(f"fold {fold}: labelled rows absent from pretrain")
        if labelled_reference is None:
            labelled_reference = union
        elif union != labelled_reference:
            errors.append(
                f"fold {fold}: labelled universe differs by "
                f"{len(union ^ labelled_reference)} paths"
            )
        few_shot: list[dict[str, Any]] = []
        for index, fraction in FEW_SHOT_FRACTIONS.items():
            name = f"train_{fold}_few_{index}"
            if name not in manifests:
                continue
            subset = {relative for relative, _ in manifests[name]["rows"]}
            if not subset <= train:
                errors.append(f"{name}: {len(subset - train)} paths outside parent train")
            few_shot.append({
                "name": name,
                "declared_fraction": fraction,
                "rows": len(subset),
                "actual_fraction": len(subset) / len(train),
                "contained_in_parent": subset <= train,
            })
        folds.append({
            "fold": fold,
            "train_rows": len(train),
            "valid_rows": len(valid),
            "overlap_count": len(overlap),
            "labelled_union_rows": len(union),
            "few_shot": few_shot,
        })

    valid_sets = [
        {relative for relative, _ in manifests[f"valid_{fold}"]["rows"]}
        for fold in range(arguments.folds)
        if f"valid_{fold}" in manifests
    ]
    overlap_matrix = [
        [len(left & right) for right in valid_sets] for left in valid_sets
    ]
    validation_frequency_counts = Counter(validation_frequency.values())
    never_validated = (
        len((labelled_reference or set()) - set(validation_frequency))
    )
    empty_label_size = 3_208
    min_label_size = 3_032
    report = {
        "manifest_dir": str(arguments.manifest_dir.resolve()),
        "provenance": {
            "authoritative_published_membership": False,
            "generator": "scripts/animal2vec_manifest.py",
            "seed": 1_612,
            "valid_percent": 0.2,
            "folds": arguments.folds,
            "reason_non_authoritative": (
                "The public dataset contains no split TSVs, and the archived "
                "generator depends on unsorted filesystem enumeration order."
            ),
        },
        "missing_manifests": missing_manifests,
        "extra_manifests": extra_manifests,
        "manifests": {
            name: _public(manifest) for name, manifest in sorted(manifests.items())
        },
        "pretrain_unique_rows": len(pretrain_paths),
        "labelled_universe_rows": len(labelled_reference or set()),
        "folds": folds,
        "validation_overlap_matrix": overlap_matrix,
        "validation_frequency_counts": [
            {"fold_appearances": appearances, "files": count}
            for appearances, count in sorted(validation_frequency_counts.items())
        ],
        "never_in_validation_count": never_validated,
        "size_filter_drift": {
            "recipe_min_label_size": min_label_size,
            "released_empty_hdf5_size": empty_label_size,
            "empty_files_pass_recipe_size_filter": empty_label_size > min_label_size,
            "impact_on_generated_manifests": (
                "none: the archived generator reads lbl_cat and emits only event-bearing "
                "files into fine-tuning manifests"
            ),
        },
        "errors": errors,
        "all_manifest_checks_pass": not errors,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest_count": len(manifests),
        "pretrain_unique_rows": len(pretrain_paths),
        "labelled_universe_rows": len(labelled_reference or set()),
        "never_in_validation_count": never_validated,
        "all_manifest_checks_pass": not errors,
        "output": str(arguments.output),
    }, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
