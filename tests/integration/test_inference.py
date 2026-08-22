"""Run the production inference API with native checkpoints and synthetic audio. The tests
cover resampling, chunk timing, partial segments, event limits, and TSV label output."""

from pathlib import Path

import pytest
import soundfile
import torch

from a2v2.data import conv_output_length, feature_timestamps
from a2v2.config import config_to_dict, load_config
from a2v2.workflows import (
    EventInterval,
    InferenceRunner,
    _trim_frames_to_duration,
    build_inference_parser,
    infer_main,
)
from a2v2.model import Animal2VecFineTuningModel
from a2v2.training import CheckpointError, capture_rng_state, save_checkpoint


ROOT = Path(__file__).parents[2]


def test_trims_padded_official_frontend_frame_past_partial_segment() -> None:
    """Check trims padded official frontend frame past partial segment."""
    layers = (
        (127, 63, 1),
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 1),
        (512, 2, 1),
        (512, 2, 1),
    )
    frame_count = conv_output_length(2_000, layers)
    timestamps = feature_timestamps(frame_count, 8_000, layers) + 10.0
    probabilities = torch.zeros(frame_count, 12)
    embeddings = torch.zeros(frame_count, 1_024)

    assert timestamps[-1] > 10.25
    probabilities, embeddings, timestamps = _trim_frames_to_duration(
        probabilities, embeddings, timestamps, end_seconds=10.25
    )

    assert probabilities.shape == (49, 12)
    assert embeddings.shape == (49, 1_024)
    assert timestamps.shape == (49,)
    assert timestamps[-1] < 10.25


def _checkpoint(tmp_path: Path) -> Path:
    """Create the tiny native checkpoint fixture required by this module."""
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([8.0, -8.0]))
    path = tmp_path / "finetuned.pt"
    save_checkpoint(path, {
        "format_version": 1,
        "stage": "finetune",
        "config": {"active": config_to_dict(finetune), "pretrained": config_to_dict(pretrain)},
        "model": model.state_dict(),
        "teacher": None,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "update": 2,
        "epoch": 1,
        "batch_in_epoch": 0,
        "rng_state": capture_rng_state(),
        "sampler_state": None,
        "best_metric": None,
    })
    return path


def _cls_checkpoint(tmp_path: Path, classification_head: str = "cls") -> Path:
    """Create a CLS-encoder checkpoint with the requested classification head."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.use_cls_token=true",),
    )
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=(
            "model.use_cls_token=true",
            f"model.classification_head={classification_head}",
        ),
    )
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )
    path = tmp_path / f"{classification_head}.pt"
    save_checkpoint(path, {
        "format_version": 1,
        "stage": "finetune",
        "config": {
            "active": config_to_dict(finetune),
            "pretrained": config_to_dict(pretrain),
        },
        "model": model.state_dict(),
        "teacher": None,
        "optimizer": None,
        "scheduler": None,
        "scaler": None,
        "update": 2,
        "epoch": 1,
        "batch_in_epoch": 0,
        "rng_state": capture_rng_state(),
        "sampler_state": None,
        "best_metric": None,
    })
    return path


def test_event_inference_rejects_cls_sequence_checkpoint(tmp_path: Path) -> None:
    """Do not reinterpret clip logits as framewise event probabilities."""

    with pytest.raises(
        CheckpointError,
        match=r"event inference.*classification_head=frame.*cls",
    ):
        InferenceRunner(_cls_checkpoint(tmp_path), device=torch.device("cpu"))


def test_event_inference_strips_cls_from_frame_embeddings(tmp_path: Path) -> None:
    """Keep frame probabilities, embeddings, and timestamps aligned with CLS."""

    runner = InferenceRunner(
        _cls_checkpoint(tmp_path, "frame"),
        device=torch.device("cpu"),
    )

    result = runner.run_tensor(
        torch.randn(64),
        8_000,
        segment_seconds=1.0,
    )

    assert len(result.probabilities) == len(result.embeddings)
    assert len(result.probabilities) == len(result.timestamps)


def test_chunked_resampled_inference_preserves_final_segment_timing(tmp_path: Path) -> None:
    """Check chunked resampled inference preserves final segment timing."""
    runner = InferenceRunner(_checkpoint(tmp_path), device=torch.device("cpu"))
    source_rate = 16_000
    duration = 1.1
    time = torch.arange(round(source_rate * duration)) / source_rate
    stereo = torch.stack((torch.sin(2 * torch.pi * 440 * time), torch.zeros_like(time)))
    result = runner.run_tensor(
        stereo,
        source_rate,
        channel=0,
        segment_seconds=0.4,
        threshold=0.5,
        event_method="avg",
        fusion_window_seconds=0.01,
    )
    assert result.probabilities.shape[1] == 2
    assert result.embeddings.shape[1] == 16
    assert len(result.timestamps) == len(result.probabilities)
    assert result.timestamps[-1] < duration
    assert result.events
    assert max(event.end_seconds for event in result.events) <= duration
    assert all(event.label_index == 0 for event in result.events)


def test_inference_tsv_contains_label_names_and_bounded_times(tmp_path: Path) -> None:
    """Check inference TSV contains label names and bounded times."""
    runner = InferenceRunner(_checkpoint(tmp_path), device=torch.device("cpu"))
    result = runner.run_tensor(
        torch.randn(8000), 8000, segment_seconds=0.5, threshold=0.5,
        event_method="max", fusion_window_seconds=0.02,
    )
    output = tmp_path / "predictions.tsv"
    runner.write_events(output, result.events)
    text = output.read_text(encoding="utf-8")
    assert text.startswith("label\tstart_seconds\tend_seconds\tscore\n")
    assert "call\t" in text


def test_audition_csv_matches_legacy_marker_schema(tmp_path: Path) -> None:
    """Match the exact Adobe marker columns, values, order, and delimiters."""

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
    """Give Audition a valid marker table when inference finds no events."""

    runner = InferenceRunner(_checkpoint(tmp_path), device=torch.device("cpu"))
    output = tmp_path / "empty.csv"

    runner.write_events(output, (), audition=True)

    assert output.read_text(encoding="utf-8") == (
        "Name\tStart\tDuration\tTime Format\tType\tDescription\n"
    )


def test_inference_parser_exposes_audition_flag() -> None:
    """Keep native output as default and select marker CSV with one flag."""

    parser = build_inference_parser()
    positional = ["checkpoint.pt", "recording.wav", "predictions.csv"]

    assert parser.parse_args(positional).audition is False
    assert parser.parse_args([*positional, "--audition"]).audition is True


def test_infer_main_writes_audition_csv(tmp_path: Path) -> None:
    """Route the CLI flag through real audio inference to the marker writer."""

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
