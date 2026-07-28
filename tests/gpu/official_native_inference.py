"""Exercise production inference with a converted official fine-tuned model.

The program synthesizes a stereo recording with two tones, including a partial
final segment, then runs FP32 or autocast FP16 inference on one GPU. It checks
embedded architecture metadata, output finiteness, frame timing, event bounds,
and peak allocated memory before writing a JSON evidence record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from a2v2.workflows import InferenceRunner


def main() -> int:
    """Run the official checkpoint probe and write its hardware/result report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--duration", type=float, default=10.25)
    parser.add_argument("--input-rate", type=int, default=16_000)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    arguments = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runner = InferenceRunner(arguments.checkpoint, device=device)

    time = torch.arange(
        round(arguments.duration * arguments.input_rate), dtype=torch.float32
    ) / arguments.input_rate
    stereo = torch.stack(
        (
            0.6 * torch.sin(2 * torch.pi * 173 * time),
            0.4 * torch.cos(2 * torch.pi * 421 * time),
        )
    )
    amp = arguments.precision == "fp16"
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        result = runner.run_tensor(
            stereo,
            arguments.input_rate,
            segment_seconds=arguments.segment_seconds,
        )
    peak_bytes = torch.cuda.max_memory_allocated(device)

    if result.probabilities.shape[1] != len(runner.config.task.unique_labels):
        raise AssertionError("classifier output count does not match embedded labels")
    if result.embeddings.shape[1] != runner.pretrained_config.model.embed_dim:
        raise AssertionError("embedding width does not match embedded pretraining config")
    if not torch.isfinite(result.probabilities).all():
        raise AssertionError("non-finite official probabilities")
    if not torch.isfinite(result.embeddings).all():
        raise AssertionError("non-finite official embeddings")
    if not torch.isfinite(result.timestamps).all():
        raise AssertionError("non-finite official timestamps")
    if result.timestamps.numel() and float(result.timestamps[-1]) >= arguments.duration:
        raise AssertionError("final timestamp exceeds partial recording duration")
    if any(event.end_seconds > arguments.duration for event in result.events):
        raise AssertionError("event exceeds recording duration")

    report = {
        "checkpoint": str(arguments.checkpoint),
        "precision": arguments.precision,
        "device": torch.cuda.get_device_name(device),
        "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "duration_seconds": arguments.duration,
        "input_sample_rate": arguments.input_rate,
        "model_sample_rate": runner.config.task.sample_rate,
        "segment_seconds": arguments.segment_seconds,
        "probability_shape": list(result.probabilities.shape),
        "embedding_shape": list(result.embeddings.shape),
        "timestamp_count": result.timestamps.numel(),
        "last_timestamp": float(result.timestamps[-1]) if result.timestamps.numel() else None,
        "event_count": len(result.events),
        "probability_min": float(result.probabilities.min()),
        "probability_max": float(result.probabilities.max()),
        "peak_allocated_bytes": peak_bytes,
        "update": runner.update,
        "embedded_fp16_init_scale": runner.config.common.fp16_init_scale,
        "embedded_norm_eps": runner.pretrained_config.model.norm_eps,
        "strict_model_load": True,
        "finite": True,
        "tf32": False,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
