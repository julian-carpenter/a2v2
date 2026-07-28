#!/usr/bin/env python3
"""Validate one native A2V2 fine-tuning checkpoint and write an audit report.

This deployment helper deliberately reuses the validation implementation used
during training. It does not introduce a second interpretation of frame
alignment, padding, focal loss, thresholds, or average precision. The only
configuration fields replaced after checkpoint loading are the validation
data location and DataLoader batch controls supplied on the command line.

The report describes a one-fold, eight-rank *approximate* reproduction. It is
not a paper-level statistical claim: a five-fold comparison and the external
Xeno-canto/NIPS4Bplus protocol remain outside this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch

# Running ``python scripts/evaluate_...py`` puts scripts/, rather than the
# repository root, first on sys.path. Adding the parent makes the helper work
# from a source checkout as well as from an editable installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from a2v2.training import CheckpointError  # noqa: E402
from a2v2.workflows import InferenceRunner, _validate  # noqa: E402


SCOPE_STATEMENT = (
    "Approximate Animal2Vec 1.0 MeerKAT reproduction for one fold using an "
    "eight-rank topology and an intentionally adjusted batch profile; this "
    "is not an exact or complete reproduction of every paper result."
)


def _positive_integer(text: str) -> int:
    """Parse a positive integer for token, rank, and accumulation controls."""

    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_integer(text: str) -> int:
    """Parse a non-negative worker or fold count."""

    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Describe every input required to reproduce and audit validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=_non_negative_integer)
    parser.add_argument("--fraction", required=True, choices=("100", "025", "001"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-tokens", required=True, type=_positive_integer)
    parser.add_argument("--num-workers", required=True, type=_non_negative_integer)
    parser.add_argument("--training-world-size", required=True, type=_positive_integer)
    parser.add_argument("--pretrain-max-tokens", required=True, type=_positive_integer)
    parser.add_argument("--pretrain-update-freq", required=True, type=_positive_integer)
    parser.add_argument("--finetune-max-tokens", required=True, type=_positive_integer)
    parser.add_argument("--finetune-update-freq", required=True, type=_positive_integer)
    return parser


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a checkpoint at once."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest_rows(path: Path) -> int:
    """Count recording rows after the manifest's first audio-root line."""

    with path.open("r", encoding="utf-8") as source:
        lines = [line for line in source if line.strip()]
    if not lines:
        raise ValueError(f"manifest is empty: {path}")
    return len(lines) - 1


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    """Capture the framework/runtime boundary that can affect numeric output."""

    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        # ``torch.version.cuda`` is the runtime against which the wheel was
        # compiled. It is intentionally not inferred from nvidia-smi's
        # driver-side maximum CUDA compatibility value.
        "pytorch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        runtime["gpu"] = {
            "logical_index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    else:
        runtime["gpu"] = None
    return runtime


def _atomic_json_write(path: Path, report: dict[str, Any]) -> str:
    """Write complete JSON before replacing a possibly older report."""

    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
    return serialized


def evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    """Reconstruct the checkpoint model, run validation, and assemble metadata."""

    checkpoint_path = arguments.checkpoint.resolve()
    manifest_directory = arguments.manifest_dir.resolve()
    manifest_path = manifest_directory / f"{arguments.subset}.tsv"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"validation manifest does not exist: {manifest_path}")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation was requested but CUDA is unavailable")

    device = torch.device(arguments.device)
    # InferenceRunner enforces the native fine-tuning stage, reconstructs both
    # active and pretrained architecture configs, and uses strict state loading.
    # Conceptually, this proves that the file selected by the shell is a
    # complete deployable model rather than only an optimizer resume artifact.
    runner = InferenceRunner(checkpoint_path, device=device)

    # The checkpoint remains the authority for model/loss/label geometry.
    # Replacing the immutable nested dataclasses makes the narrow evaluation
    # changes explicit. Mathematics: batch partitioning may change, but
    # validation aggregates summed loss/counts and therefore remains invariant
    # to that partition up to floating-point reduction order.
    validation_config = replace(
        runner.config,
        task=replace(runner.config.task, data=manifest_directory),
        dataset=replace(
            runner.config.dataset,
            valid_subset=arguments.subset,
            max_tokens=arguments.max_tokens,
            num_workers=arguments.num_workers,
        ),
    )
    metrics = _validate(
        runner.model,
        validation_config,
        device=device,
        update=runner.update,
    )

    # Effective token batch is the product of per-rank padded-sample budget,
    # rank count, and gradient-accumulation steps. It is a capacity/throughput
    # descriptor, not the exact number of real unpadded samples in every batch.
    pretrain_effective_tokens = (
        arguments.pretrain_max_tokens
        * arguments.training_world_size
        * arguments.pretrain_update_freq
    )
    finetune_effective_tokens = (
        arguments.finetune_max_tokens
        * arguments.training_world_size
        * arguments.finetune_update_freq
    )
    return {
        "schema_version": 1,
        "scope": {
            "exact_paper_reproduction": False,
            "statement": SCOPE_STATEMENT,
            "omitted": [
                "four additional MeerKAT folds",
                "cross-fold uncertainty estimates",
                "Xeno-canto pretraining subset",
                "NIPS4Bplus transfer protocol",
            ],
        },
        "run": {
            "fold": arguments.fold,
            "label_fraction": arguments.fraction,
            "training_world_size": arguments.training_world_size,
            "pretraining": {
                "max_tokens_per_rank": arguments.pretrain_max_tokens,
                "update_frequency": arguments.pretrain_update_freq,
                "effective_tokens_per_update": pretrain_effective_tokens,
                "published_effective_tokens_per_update": 408_000 * 4 * 5,
            },
            "finetuning": {
                "max_tokens_per_rank": arguments.finetune_max_tokens,
                "update_frequency": arguments.finetune_update_freq,
                "effective_tokens_per_update": finetune_effective_tokens,
                "published_effective_tokens_per_update": 426_667 * 4 * 9,
            },
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "update": runner.update,
        },
        "validation_data": {
            "subset": arguments.subset,
            "manifest": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "manifest_rows": _manifest_rows(manifest_path),
            "max_tokens": arguments.max_tokens,
            "num_workers": arguments.num_workers,
        },
        "metrics": {name: float(value) for name, value in metrics.items()},
        "runtime": _runtime_metadata(device),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise errors suitable for long-running jobs."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        report = evaluate(arguments)
        serialized = _atomic_json_write(arguments.output.resolve(), report)
    except (CheckpointError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
