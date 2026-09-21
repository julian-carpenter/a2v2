"""Run the production inference API with native checkpoints and synthetic audio. The tests
cover resampling, chunk timing, partial segments, event limits, and TSV label output."""

from pathlib import Path

import pytest
import soundfile
import torch

from a2v2.data import conv_output_length, feature_timestamps, normalize_waveform
from a2v2.config import config_to_dict, load_config
from a2v2.workflows import (
    EventInterval,
    InferenceRunner,
    _trim_frames_to_duration,
    build_inference_parser,
    infer_main,
)
from a2v2.model import Animal2VecFineTuningModel, Animal2VecPretrainingModel
from a2v2.training import capture_rng_state, save_checkpoint


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


def test_cls_sequence_checkpoint_returns_segment_predictions(tmp_path: Path) -> None:
    """Auto-enable CLS without reinterpreting clip logits as frame events."""
    runner = InferenceRunner(_cls_checkpoint(tmp_path), device=torch.device("cpu"))
    result = runner.run_tensor(torch.randn(160), 8000, segment_seconds=0.008)
    assert runner.return_cls is True
    assert result.probabilities is None and result.events is None
    assert result.cls_embeddings.shape == (3, 16)
    assert result.cls_predictions.shape == (3, 2)
    assert result.embeddings.shape == (len(result.timestamps), 16)
    assert result.timestamps[-1] < 0.02
    assert torch.isfinite(result.cls_embeddings).all()
    assert ((result.cls_predictions >= 0) & (result.cls_predictions <= 1)).all()
    empty = runner.run_tensor(torch.empty(0), 8000)
    assert empty.cls_embeddings.shape == (0, 16)
    assert empty.cls_predictions.shape == (0, 2)
    assert empty.probabilities is None and empty.events is None
    with pytest.raises(ValueError, match="no frame events"):
        runner.write_events(tmp_path / "not-events.tsv", result.events)
    assert not (tmp_path / "not-events.tsv").exists()


@pytest.mark.parametrize("use_cls", [False, True])
@pytest.mark.parametrize("return_cls", [False, True])
def test_pretrain_returns_unmasked_student_embeddings(
    tmp_path: Path, use_cls: bool, return_cls: bool,
) -> None:
    """Reject masked/teacher features and preserve the frame and CLS axes."""
    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml", overrides=(
        f"model.use_cls_token={str(use_cls).lower()}",
    ))
    model = Animal2VecPretrainingModel.from_config(config).eval()
    # Distinguish the EMA snapshot so accidentally selecting it cannot pass.
    with torch.no_grad():
        for parameter in model.teacher.parameters():
            parameter.zero_()
    path = tmp_path / "pretrain.pt"
    save_checkpoint(path, {
        "format_version": 1, "stage": "pretrain",
        "config": {"active": config_to_dict(config)}, "model": model.state_dict(),
        "teacher": None, "optimizer": None, "scheduler": None, "scaler": None,
        "update": 2, "epoch": 1, "batch_in_epoch": 0,
        "rng_state": capture_rng_state(), "sampler_state": None, "best_metric": None,
    })
    runner = InferenceRunner(path, device=torch.device("cpu"), return_cls=return_cls)
    waveform = torch.randn(160)
    result = runner.run_tensor(waveform, 8000, segment_seconds=0.008)
    assert result.probabilities is None and result.events is None
    assert result.cls_predictions is None
    expected_frames, expected_cls = [], []
    with torch.inference_mode():
        for start in (0, 64, 128):
            output = model.student(normalize_waveform(waveform[start:start + 64])[None])
            features = torch.stack(output.layer_outputs[-2:]).mean(0)[0]
            if use_cls:
                expected_cls.append(features[0])
                features = features[1:]
            times = feature_timestamps(len(features), 8000, config.task.conv_feature_layers)
            expected_frames.append(features[times < min(64, 160 - start) / 8000])
    torch.testing.assert_close(result.embeddings, torch.cat(expected_frames))
    assert len(result.timestamps) == len(result.embeddings)
    assert (result.timestamps[1:] > result.timestamps[:-1]).all()
    assert result.timestamps[-1] < 0.02
    if return_cls and use_cls:
        torch.testing.assert_close(result.cls_embeddings, torch.stack(expected_cls))
    else:
        assert result.cls_embeddings is None
    empty = runner.run_tensor(torch.empty(0), 8000)
    assert empty.embeddings.shape == (0, 16) and empty.timestamps.shape == (0,)
    assert empty.probabilities is None and empty.events is None
    if return_cls and use_cls:
        assert empty.cls_embeddings.shape == (0, 16)
    detector = InferenceRunner(path, device=torch.device("cpu"), event_detection=True)
    detected = detector.run_tensor(waveform, 8000, segment_seconds=0.008)
    assert detected.probabilities is None and detected.events is None
    torch.testing.assert_close(detected.embeddings, result.embeddings)


@pytest.mark.parametrize("head", ["frame", "cls"])
def test_cls_embeddings_match_segment_encoder_features(tmp_path: Path, head: str) -> None:
    """Return the averaged CLS features used by the trained classifier."""
    runner = InferenceRunner(_cls_checkpoint(tmp_path, head), device=torch.device("cpu"), return_cls=True)
    waveform = torch.randn(64)
    result = runner.run_tensor(waveform, 8000)
    with torch.inference_mode():
        output = runner.model(normalize_waveform(waveform)[None], update=runner.update)
        expected = torch.stack(output.layer_outputs[-2:]).mean(0)[:, 0]
    torch.testing.assert_close(result.cls_embeddings, expected)
    if head == "cls":
        torch.testing.assert_close(result.cls_predictions, output.logits.sigmoid())
    else:
        assert result.cls_predictions is None and result.probabilities.shape[1] == 2


@pytest.mark.parametrize("vote", ["max", "mean", "median"])
@pytest.mark.parametrize("values,expected", [
    ([0.2, 0.3, 0.5, 0.8], {"max": 0.8, "mean": 0.45, "median": 0.4}),
    ([0.1, 0.3, 0.8], {"max": 0.8, "mean": 0.4, "median": 0.3}),
    ([0.7], {"max": 0.7, "mean": 0.7, "median": 0.7}),
])
@pytest.mark.parametrize("head", ["frame", "cls"])
def test_event_votes_reduce_probabilities_before_threshold(
    tmp_path: Path, vote: str, values: list[float], expected: dict[str, float], head: str,
) -> None:
    """Catch logit reduction, binary voting, and lower-middle median errors."""
    path = _cls_checkpoint(tmp_path, head)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["config"]["active"]["task"]["unique_labels"] = ["a", "b", "c", "d"][:len(values)]
    checkpoint["model"]["classifier.weight"] = torch.zeros(len(values), 16)
    checkpoint["model"]["classifier.bias"] = torch.logit(torch.tensor(values))
    save_checkpoint(path, checkpoint)
    runner = InferenceRunner(path, device=torch.device("cpu"), event_detection=True, event_vote=vote)
    result = runner.run_tensor(torch.randn(160), 8000, segment_seconds=0.008,
                               threshold=0.425, event_method="max", fusion_window_seconds=0.0005)
    scores = result.cls_predictions if head == "cls" else result.probabilities
    want = expected[vote]
    assert scores.shape[1] == 1
    torch.testing.assert_close(scores, torch.full_like(scores, want))
    if head == "cls":
        assert result.events is None and result.probabilities is None
    else:
        assert bool(result.events) == (want > 0.425)
        if result.events:
            assert all(event.label_index == 0 for event in result.events)
            for audition in (False, True):
                output = tmp_path / f"events-{audition}.tsv"
                runner.write_events(output, result.events, audition=audition)
                assert all(line.startswith("event\t") for line in output.read_text().splitlines()[1:])


def test_event_detection_defaults_to_max_and_missing_cls_stays_none(tmp_path: Path) -> None:
    """Keep max as the default vote without fabricating a missing CLS token."""
    runner = InferenceRunner(_checkpoint(tmp_path), device=torch.device("cpu"),
                             event_detection=True, return_cls=True)
    result = runner.run_tensor(torch.randn(64), 8000)
    torch.testing.assert_close(result.probabilities, torch.full_like(result.probabilities, 0.99966465))
    assert result.probabilities.shape[1] == 1
    assert result.cls_predictions is None and result.cls_embeddings is None
    empty = runner.run_tensor(torch.empty(0), 8000)
    assert empty.probabilities.shape == (0, 1) and empty.events == ()


def test_inference_rejects_unknown_vote(tmp_path: Path) -> None:
    """Reject invalid voting policies at construction instead of ignoring them."""
    with pytest.raises(ValueError, match="event_vote"):
        InferenceRunner(_checkpoint(tmp_path), event_vote="sum")


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
    assert result.cls_embeddings is None and result.cls_predictions is None


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
