"""Compare uninterrupted and resumed native checkpoints field by field.

The utility descends through tensors, NumPy arrays, mappings, and sequences,
then reports the first mismatch within each top-level checkpoint field. It
requires exact tensor equality. The output directory is normalized because
continuous and resumed experiments store checkpoints in different folders by
construction.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for continuous checkpoint, resumed checkpoint, and report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def _first_difference(left: Any, right: Any, path: str) -> dict[str, Any] | None:
    """Return the first recursive type, metadata, or value difference."""

    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return {
                "path": path,
                "left_shape": list(left.shape),
                "right_shape": list(right.shape),
                "left_dtype": str(left.dtype),
                "right_dtype": str(right.dtype),
            }
        if torch.equal(left, right):
            return None
        difference: dict[str, Any] = {"path": path, "exact": False}
        if left.is_floating_point() or left.is_complex():
            delta = (left - right).abs()
            difference.update({
                "max_abs_error": float(delta.max()),
                "different_values": int(torch.count_nonzero(delta)),
            })
        return difference
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if np.array_equal(left, right):
            return None
        return {
            "path": path,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "exact": False,
        }
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return {
                "path": path,
                "left_only_keys": sorted(str(key) for key in set(left) - set(right)),
                "right_only_keys": sorted(str(key) for key in set(right) - set(left)),
            }
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if type(left) is not type(right) or left != right:
        return {"path": path, "left": repr(left), "right": repr(right)}
    return None


def _normalized_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Copy config and replace the expected experiment-specific save path."""

    config = copy.deepcopy(checkpoint["config"])
    config["active"]["checkpoint"]["save_dir"] = "<normalized-save-dir>"
    return config


def main() -> int:
    """Write the comparison report and fail when any normalized field differs."""

    arguments = build_parser().parse_args()
    continuous = torch.load(
        arguments.continuous, map_location="cpu", mmap=True, weights_only=False
    )
    resumed = torch.load(arguments.resumed, map_location="cpu", mmap=True, weights_only=False)
    fields = sorted(set(continuous) | set(resumed))
    comparisons: dict[str, dict[str, Any] | None] = {}
    for field in fields:
        if field not in continuous or field not in resumed:
            comparisons[field] = {
                "path": field,
                "continuous_present": field in continuous,
                "resumed_present": field in resumed,
            }
            continue
        left = _normalized_config(continuous) if field == "config" else continuous[field]
        right = _normalized_config(resumed) if field == "config" else resumed[field]
        comparisons[field] = _first_difference(left, right, field)

    report = {
        "continuous": str(arguments.continuous),
        "continuous_bytes": arguments.continuous.stat().st_size,
        "resumed": str(arguments.resumed),
        "resumed_bytes": arguments.resumed.stat().st_size,
        "normalized_fields": ["config.active.checkpoint.save_dir"],
        "fields": comparisons,
        "all_exact": all(value is None for value in comparisons.values()),
        "update": continuous.get("update"),
        "epoch": continuous.get("epoch"),
        "batch_in_epoch": continuous.get("batch_in_epoch"),
        "sampler_state": continuous.get("sampler_state"),
        "amp_scale": continuous.get("scaler", {}).get("scale"),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
