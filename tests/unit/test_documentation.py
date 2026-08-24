"""Enforce the repository's researcher-facing documentation contract.

These tests treat documentation as part of the maintained source. They catch
new Python modules or definitions that arrive without an explanation and
protect every YAML recipe from accidental value changes during comment edits.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
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
    "configs/modern/rope_cls_geglu_finetune.yaml":
        "5edf20a0d85f401f14c42e07c6388d87a71109fa7665e0f0381cf84d5b937f93",
    "configs/modern/rope_cls_geglu_pretrain.yaml":
        "7a1b1225a61f4bc7025c792562f923f752db1dc0050b319910d18e904bff3ee8",
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


def _shell_fences(markdown: str) -> list[str]:
    """Return the contents of Bash, sh, and shell Markdown fences."""

    return re.findall(
        r"```(?:bash|sh|shell)\n(.*?)```",
        markdown,
        re.DOTALL,
    )


def _heading_slug(heading: str) -> str:
    """Return the GitHub-style anchor used by maintained documentation."""

    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    return re.sub(r" +", "-", slug)


def _markdown_anchors(markdown: str) -> set[str]:
    """Return heading anchors defined in one Markdown document."""

    return {
        _heading_slug(match.group(1))
        for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", markdown)
    }


def _assert_relative_links_resolve(path: Path) -> None:
    """Require every relative Markdown link and fragment to resolve."""

    markdown = path.read_text(encoding="utf-8")
    links = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", markdown)
    for target in links:
        target = target.split(maxsplit=1)[0].strip("<>")
        if re.match(r"^(?:https?://|mailto:)", target):
            continue
        relative, _, fragment = target.partition("#")
        destination = path if not relative else (path.parent / relative).resolve()
        assert destination.exists(), f"{path.relative_to(ROOT)}: missing {target}"
        if fragment and destination.suffix.lower() == ".md":
            anchors = _markdown_anchors(destination.read_text(encoding="utf-8"))
            assert fragment in anchors, (
                f"{path.relative_to(ROOT)}: missing fragment {target}"
            )


def _flatten_shell_command(fence: str) -> str:
    """Collapse one backslash-continued shell fence into a command line."""

    return re.sub(r"\\\r?\n\s*", " ", fence).strip()


def _has_finetune_800000_shell_assignment(markdown: str) -> bool:
    """Flag executable shell-fence assignments of the obsolete budget."""

    assignment = re.compile(
        r"(?m)^[ \t]*(?:export[ \t]+)?"
        r"A2V2_FINETUNE_MAX_TOKENS=[ \t]*"
        r"(?:800000|'800000'|\"800000\")(?=$|[ \t;&|])"
    )
    return any(
        assignment.search(re.sub(r"\\\r?\n[ \t]*", "", fence)) is not None
        for fence in _shell_fences(markdown)
    )


@pytest.mark.parametrize(
    ("assignment",),
    (
        ("A2V2_FINETUNE_MAX_TOKENS=800000 bash reproduce.sh",),
        ("  export A2V2_FINETUNE_MAX_TOKENS=800000",),
        ("A2V2_FINETUNE_MAX_TOKENS='800000' bash reproduce.sh",),
        ('A2V2_FINETUNE_MAX_TOKENS="800000" bash reproduce.sh',),
        ("A2V2_FINETUNE_MAX_TOKENS=\\\n800000 bash reproduce.sh",),
    ),
)
def test_shell_fence_guard_rejects_800000_assignments(assignment: str) -> None:
    """Reject obsolete budgets across executable shell assignment layouts."""

    markdown = f"```bash\n{assignment}\n```"

    assert _has_finetune_800000_shell_assignment(markdown)


@pytest.mark.parametrize(
    "content",
    (
        "A2V2_FINETUNE_MAX_TOKENS=960000 bash reproduce.sh",
        "  export A2V2_FINETUNE_MAX_TOKENS='960000'",
        "# A2V2_FINETUNE_MAX_TOKENS=800000 is obsolete",
        'printf \'%s\\n\' "A2V2_FINETUNE_MAX_TOKENS=800000 is obsolete"',
    ),
)
def test_shell_fence_guard_allows_current_budget_and_harmless_mentions(
    content: str,
) -> None:
    """Allow the current assignment plus comment and prose mentions."""

    markdown = f"```bash\n{content}\n```"

    assert not _has_finetune_800000_shell_assignment(markdown)


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
        "Activation checkpointing is a fixed fine-tuning override in this 40 GB driver",
        "The activation-checkpointing flag changes execution policy and adds no\ncheckpoint tensors",
        "Do not use `A2V2_FINETUNE_MAX_TOKENS=800000` as an OOM recovery setting",
    ):
        assert required in reproduction

    for required in (
        "`ModelConfig.checkpoint_activations` defaults to false",
        "non-reentrant",
        "RNG preservation",
        "An older pretrained snapshot\ntherefore cannot disable the current execution policy",
        "Fine-tuning activation memory",
        "model state dictionaries remain unchanged",
    ):
        assert required in code_guide

    assert _shell_fences(reproduction)
    assert not _has_finetune_800000_shell_assignment(reproduction)
    assert "substantially more measured headroom" not in reproduction


def test_modern_features_and_slurm_contract_are_documented() -> None:
    """Keep opt-in features, launch commands, and site gates discoverable."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    configs = (ROOT / "configs/README.md").read_text(encoding="utf-8")
    code_guide = (ROOT / "docs/code-guide.md").read_text(encoding="utf-8")
    reproduction = (ROOT / "docs/reproducing-paper.md").read_text(
        encoding="utf-8",
    )
    slurm = (ROOT / "docs/slurm.md").read_text(encoding="utf-8")

    for required in (
        "configs/modern/rope_cls_geglu_pretrain.yaml",
        "configs/modern/rope_cls_geglu_finetune.yaml",
        "pip install -e '.[bnb]'",
        "[SLURM guide](docs/slurm.md)",
        "--phase pretrain",
        "--phase finetune",
        "--phase all",
        "--dry-run",
    ):
        assert required in readme

    for required in (
        "Architecture and checkpoint choices",
        "Execution and optimization policy",
        "position_encoding",
        "attention_backend",
        "use_cls_token",
        "classification_head",
        "gradient_clip_method",
        "weight_decay_schedule",
        "torch_compile_fullgraph",
        "checkpoint_activations",
        "BNB_CUDA_VERSION=130",
    ):
        assert required in configs

    for required in (
        "RoPE and ALiBi",
        "Strict Flash and SDPA",
        "CLS pretraining and sequence fine-tuning",
        "Packed GEGLU and DeepScaleLM",
        "AdaGC",
        "Cosine weight-decay clock",
        "Checkpoint format v1 and v2",
        "Graph breaks",
        "A100 microbenchmark",
    ):
        assert required in code_guide

    for required in (
        "scripts/reproduce_meerkat_paper.sh",
        "scripts/reproduce_meerkat_slurm.sh",
        "byte-identical",
        "unrun real-site gate",
    ):
        assert required in reproduction

    for required in (
        "one torchrun agent per node",
        "a2v2.stage-completion.v1",
        "SIGUSR1",
        "300 seconds",
        "exit 75",
        "site policy",
        "A2V2_RUN_ID",
        "c10d",
        "One-node parity",
        "Two-node acceptance matrix",
        "Unrun real-site gate",
    ):
        assert required in slurm


def test_task11_relative_markdown_links_resolve() -> None:
    """Keep Task 11 links tied to existing files and headings."""

    for relative in (
        "README.md",
        "configs/README.md",
        "docs/code-guide.md",
        "docs/reproducing-paper.md",
        "docs/slurm.md",
    ):
        _assert_relative_links_resolve(ROOT / relative)


def test_documented_slurm_help_dry_run_and_shell_syntax_execute() -> None:
    """Execute the documented safe launcher paths and parse both Bash scripts."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    dry_runs = [
        _flatten_shell_command(fence)
        for fence in _shell_fences(readme)
        if "--dry-run" in fence
        and "scripts/reproduce_meerkat_slurm.sh" in fence
    ]
    assert len(dry_runs) == 1
    completed = subprocess.run(
        ["bash", "-c", dry_runs[0]],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Launching: srun" in completed.stdout
    assert "scripts/a2v2_slurm_node.sh" in completed.stdout

    help_result = subprocess.run(
        [
            "bash",
            "scripts/reproduce_meerkat_slurm.sh",
            "/datasets/MeerKAT/manifests",
            "/shared/runs/meerkat",
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--dry-run" in help_result.stdout

    sequence_help = subprocess.run(
        ["a2v2-evaluate-sequence", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert sequence_help.returncode == 0, sequence_help.stderr
    normalized_help = " ".join(sequence_help.stdout.split())
    assert "--override SECTION.KEY=VALUE" in normalized_help
    assert "--trust-checkpoint" in normalized_help
    assert "pickle-backed" in normalized_help
    assert "does not make pickle safe" in normalized_help

    for script in (
        "scripts/reproduce_meerkat_slurm.sh",
        "scripts/a2v2_slurm_node.sh",
    ):
        syntax = subprocess.run(
            ["bash", "-n", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert syntax.returncode == 0, syntax.stderr


def test_a100_claim_keeps_its_scope_and_measurement_qualifiers() -> None:
    """Keep the reported compile ratio attached to its bounded evidence."""

    code_guide = (ROOT / "docs/code-guide.md").read_text(encoding="utf-8")
    paragraph = re.search(
        r"The bounded A100 microbenchmark.*?(?=\n\n)",
        code_guide,
        re.DOTALL,
    )
    assert paragraph is not None
    evidence = paragraph.group(0)
    for qualifier in (
        "B=2",
        "T=128",
        "D=64",
        "five measured iterations",
        "two warm-ups",
        "2.7787x",
        "one A100-SXM4-40GB",
        "no paper-scale or general throughput claim",
    ):
        assert qualifier in evidence


def test_modern_cls_guidance_uses_sequence_evaluation_and_no_all_stage_training() -> None:
    """Keep CLS guidance on its supported two-stage and sequence-evaluation path."""

    slurm = (ROOT / "docs/slurm.md").read_text(encoding="utf-8")
    section = re.search(
        r"(?ms)^### Modern CLS sequence evaluation\n(.*?)(?=^###?\s|\Z)",
        slurm,
    )
    assert section is not None
    guidance = section.group(1)
    assert "a2v2-evaluate-sequence" in guidance
    assert "srun --nodes=1 --ntasks=1 --gpus=1" in guidance
    assert "--phase all" not in guidance
    assert "Task 9" not in slurm


def test_sequence_evaluation_docs_require_trust_and_distinguish_slurm_paths() -> None:
    """Document trusted pickle loading and both evaluator contracts."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    slurm = (ROOT / "docs/slurm.md").read_text(encoding="utf-8")
    code_guide = (ROOT / "docs/code-guide.md").read_text(encoding="utf-8")
    section = re.search(
        r"(?ms)^### Modern CLS sequence evaluation\n(.*?)(?=^###?\s|\Z)",
        slurm,
    )
    assert section is not None
    guidance = section.group(1)

    readme_commands = [
        fence
        for fence in _shell_fences(readme)
        if "a2v2-evaluate-sequence" in fence and "--help" not in fence
    ]
    slurm_commands = [
        fence
        for fence in _shell_fences(guidance)
        if "a2v2-evaluate-sequence" in fence
    ]
    assert len(readme_commands) == 1
    assert len(slurm_commands) == 2
    assert all("--trust-checkpoint" in fence for fence in readme_commands)
    assert all("--trust-checkpoint" in fence for fence in slurm_commands)

    for documented in (readme, guidance):
        assert "pickle-backed" in documented
        assert "source you trust" in documented
        assert "does not make pickle safe" in documented

    command_surface = re.search(
        r"(?ms)^The orchestrator accepts .*?(?=^### Pretrain)",
        slurm,
    )
    assert command_surface is not None
    evaluator_contract = " ".join(command_surface.group(0).split())
    for required in (
        "launcher `evaluate` phase",
        "legacy framewise/event evaluator",
        "frame and event metrics",
        "`a2v2-evaluate-sequence`",
        "CLS sequence evaluator",
        "sequence-level metrics",
        "Both evaluators restore",
        "stored pretraining config",
    ):
        assert required in evaluator_contract
    assert "does not load a pretraining config" not in evaluator_contract
    assert (
        "| `a2v2-evaluate-sequence` | `evaluate_sequence_main` |"
        in code_guide
    )


def test_slurm_site_gate_remains_explicitly_unrun() -> None:
    """Keep the real-site acceptance gate separate from local validation."""

    slurm = (ROOT / "docs/slurm.md").read_text(encoding="utf-8")
    gate = re.search(
        r"(?ms)^## Unrun real-site gate\n(.*?)(?=^##\s|\Z)",
        slurm,
    )
    assert gate is not None
    assert "did not run" in gate.group(1)
