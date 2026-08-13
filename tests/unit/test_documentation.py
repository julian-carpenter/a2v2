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
        "2582620361833034d075adc81254a05e4d1a608fe00de1e129a3f15564ee68d1",
    "configs/MeerKAT/finetune_mixup_001.yaml":
        "f79fb460836e83b641dcf5defeb8ffaec0e71fe724788980ebe05c0afed4e7ad",
    "configs/MeerKAT/finetune_mixup_025.yaml":
        "9d9e796db0d320f3512adb34ed3cf0c5ffed827eecc61c878ea14d2adf406549",
    "configs/MeerKAT/finetune_mixup_100.yaml":
        "9630fda2d3d9377643acfee4407182ff667c8e5182ffd1dfa02c8c95f4e82052",
    "configs/cpu_smoke_finetuning.yaml":
        "fe6a83820aea9b9c9b0f47e3375ce682cc281d5bcd8e9f29522056cb49adf3d7",
    "configs/cpu_smoke_pretraining.yaml":
        "9f25841c8e6ee848f043a2fcf87d5f2ab0f6bcae3b51895f74e1d5827e4292f9",
    "configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml":
        "5398ae35c32452694f624bbd7df90eadfee74e118de51dcb1d7c74d172f0f59f",
    "configs/hyenas/finetune_mixup_100.yaml":
        "7b657777801b20fdef164a714def35677be579b7cbbd8bca3cbbd16cf96a9681",
    "tests/fixtures/tiny_finetune.yaml":
        "166ebbbc14417430a664b02fdaf2f7c327d9923217cff5f00a6b8487aedff0cc",
    "tests/fixtures/tiny_pretrain.yaml":
        "61e5eb25022f73b63951e5e69b733be57eaed9d2daac4784ab30cc657464c513",
}

IGNORED_RECIPE_PATHS = {
    "hydra",
    "common.log_format",
    "common.fp16_no_flatten_grads",
    "common.all_gather_list_size",
    "checkpoint.keep_last_epochs",
    "task._name",
    "task.verbose_tensorboard_logging",
    "dataset.skip_invalid_size_inputs_valid_test",
    "distributed_training.ddp_backend",
    "criterion._name",
    "criterion.use_focal_loss",
    "criterion.label_smoothing",
    "criterion.maxfilt_s",
    "criterion.max_duration_s",
    "criterion.lowP",
    "criterion.segmentation_metrics",
    "criterion.report_accuracy",
    "criterion.log_keys",
    "optimizer.dynamic_groups",
    "optimizer.groups",
    "lr_scheduler._name",
    "model._name",
    "model.supported_modality",
    "model.ema_encoder_only",
    "model.load_pretrain_weights",
    "model.modalities.audio.mask_prob_adjust",
    "model.modalities.audio.inverse_mask",
    "model.modalities.audio.add_masks",
    "model.modalities.audio.ema_local_encoder",
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


def test_checked_in_recipes_exclude_ignored_legacy_options() -> None:
    """Keep native recipes free of fields consumed only by Fairseq or Hydra."""

    violations: list[str] = []
    for relative in YAML_HASHES:
        values = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        pending: list[tuple[tuple[str, ...], object]] = [((), values)]
        while pending:
            prefix, value = pending.pop()
            if not isinstance(value, dict):
                continue
            for key, child in value.items():
                path = prefix + (str(key),)
                dotted = ".".join(path)
                if dotted in IGNORED_RECIPE_PATHS:
                    violations.append(f"{relative}:{dotted}")
                pending.append((path, child))

    assert violations == [], (
        "Checked-in recipes contain ignored legacy options:\n"
        + "\n".join(sorted(violations))
    )


def test_finetuning_activation_memory_contract_is_documented() -> None:
    """Keep 40 GB fine-tuning guidance tied to the saved sampler topology."""

    reproduction = (ROOT / "docs/reproducing-paper.md").read_text(encoding="utf-8")
    code_guide = (ROOT / "docs/code-guide.md").read_text(encoding="utf-8")

    for required in (
        "Fine-tuning update 10,000 is the first update with a trainable Transformer",
        "`model.checkpoint_activations=true` for fine-tuning",
        "both 960,000 and 800,000 produce\neight-record microbatches",
        "changing `max_tokens` is not an exact resume",
        "batch and activation-memory profile\nprovenance",
        "`--override model.checkpoint_activations=true`",
    ):
        assert required in reproduction

    for required in (
        "`ModelConfig.checkpoint_activations` defaults to false",
        "non-reentrant",
        "RNG preservation",
        "An older pretrained snapshot\ntherefore cannot disable the current execution policy",
        "Fine-tuning activation memory",
    ):
        assert required in code_guide

    assert "A2V2_FINETUNE_MAX_TOKENS=800000 \\\n" not in reproduction
    assert "substantially more measured headroom" not in reproduction
