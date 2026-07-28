"""Structural tests for the reader-facing standalone source layout."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).parents[2]
PACKAGE = ROOT / "a2v2"

EXPECTED_MODULES = (
    "config",
    "data",
    "model",
    "training",
    "workflows",
)
EXPECTED_FILES = {"__init__.py", *(f"{name}.py" for name in EXPECTED_MODULES)}

FORBIDDEN_RUNTIME_IMPORTS = {
    "fairseq",
    "hydra",
    "omegaconf",
    "tensorflow",
    "timm",
    "torchaudio",
    "transformers",
}


def test_a2v2_package_has_exactly_six_flat_source_files() -> None:
    """Require one compact A2V2 package and reject superseded namespaces."""

    assert PACKAGE.is_dir()
    assert not (ROOT / "baseline").exists()
    assert not (ROOT / "animal2vec").exists()
    assert not (ROOT / "src").exists()
    assert not (ROOT / "nn").exists()

    python_files = {path.name for path in PACKAGE.glob("*.py")}
    assert python_files == EXPECTED_FILES

    nested_directories = sorted(
        path.name
        for path in PACKAGE.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )
    assert nested_directories == []


def test_all_a2v2_modules_import_from_the_flat_package() -> None:
    """Import each implementation boundary through its A2V2 package path."""

    for module_name in EXPECTED_MODULES:
        importlib.import_module(f"a2v2.{module_name}")


def test_native_source_does_not_import_removed_frameworks() -> None:
    """Check native source does not import removed frameworks."""
    violations: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.partition(".")[0]
                    if root in FORBIDDEN_RUNTIME_IMPORTS:
                        violations.append(f"{path.name}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module
            if imported and imported.partition(".")[0] in FORBIDDEN_RUNTIME_IMPORTS:
                violations.append(f"{path.name}:{node.lineno}:{imported}")

    assert violations == []
