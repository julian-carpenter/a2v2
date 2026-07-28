"""Enforce the repository's researcher-facing documentation contract.

These tests treat documentation as part of the maintained source. They catch
new Python modules or definitions that arrive without an explanation and
protect every YAML recipe from accidental value changes during comment edits.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
PYTHON_ROOTS = (ROOT / "a2v2", ROOT / "scripts", ROOT / "tests")
YAML_HASHES = {
    "configs/MeerKAT/a2v_large_pretrain_best.yaml":
        "c4eaf146a1f9e3319d4e7cbf4f122ff11b26d53ef82d66a75132a05d13b7dcf0",
    "configs/MeerKAT/finetune_mixup_001.yaml":
        "aade7ffe69cea7bee03ed722f14c30fe08a691fbd8f2a2792eae5664f790c199",
    "configs/MeerKAT/finetune_mixup_025.yaml":
        "f8f336a82172ca46441a2ceda494a5cc05c4b562fcf4bdd37f584d49cb6f44e1",
    "configs/MeerKAT/finetune_mixup_100.yaml":
        "8bdf2b3803e15302d76e04a088e2bb09764aca57f5f1a626125afa0cdf10496b",
    "configs/cpu_smoke_finetuning.yaml":
        "fa50853e6f1b14883bb982eae736622f0bd1afb966a4912fbbf443bc4cb3c372",
    "configs/cpu_smoke_pretraining.yaml":
        "77529552a7f50ed170e4de94099befcfab79d6c66b55d2b476afa6118e3bdc08",
    "configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml":
        "d0c70d6fd03d065efacf9ac4f947ef0d362000747dae69554bf55a8a8f6be2be",
    "configs/hyenas/finetune_mixup_100.yaml":
        "d0e556bf3a307b5cba8e4e7ed6dd30d8d7c6cbb1c32c35c6b0e73d361119822b",
    "tests/fixtures/tiny_finetune.yaml":
        "fdea332527289ea5a503ca58ef7fca4602ace01d5dbd92a52016c4529fc3f220",
    "tests/fixtures/tiny_pretrain.yaml":
        "d41fbd7716fc8915a178a3d6406d68aa82f6810447acd3de4587c0a9f998bc54",
}


def _python_files() -> Iterator[Path]:
    """Yield maintained Python files while excluding generated caches."""

    for source_root in PYTHON_ROOTS:
        yield from sorted(source_root.rglob("*.py"))


def _definition_nodes(tree: ast.AST) -> Iterator[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield definitions that need a reader-facing contract.

    Python protocol methods such as ``__len__`` inherit their meaning from the
    documented class. All named research, runtime, helper, fixture, and test
    definitions remain in scope, including nested training closures.
    """

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            yield node


def test_python_files_explain_modules_and_named_definitions() -> None:
    """Require source-level explanations at each Python navigation boundary."""

    missing: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT)
        if ast.get_docstring(tree) is None:
            missing.append(f"{relative}:1 module")
        for node in _definition_nodes(tree):
            if ast.get_docstring(node) is None:
                missing.append(f"{relative}:{node.lineno} {node.name}")
    assert missing == [], "Missing documentation:\n" + "\n".join(missing)


def test_yaml_files_open_with_context_and_preserve_values() -> None:
    """Require recipe introductions and preserve all pre-comment YAML values."""

    discovered = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*.yaml")
    }
    assert discovered == set(YAML_HASHES), "Update the documented recipe inventory"

    for relative, expected_hash in YAML_HASHES.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        first_content_line = next(
            (line for line in text.splitlines() if line.strip()),
            "",
        )
        assert first_content_line.startswith("#"), f"{relative} needs an introductory comment"

        values = yaml.safe_load(text)
        canonical = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == expected_hash
