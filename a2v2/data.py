"""Waveform, annotation, dataset, collation, and batching primitives.

Researchers can read this file as the complete path from an audio filename to a
training batch. The sections retain the boundaries of the former small modules:
audio operations establish sample and feature time, label operations map sparse
events onto that feature grid, dataset operations join the two representations,
and samplers form batches under a waveform-token budget.

Shape notation used throughout:

* ``S`` is the number of waveform samples in one recording.
* ``T`` is the number of frames after the convolutional feature encoder.
* ``B`` is the batch size and ``C`` is the number of target classes.

Comments on result-sensitive operations state the mathematics first and then
give a conceptual interpretation. Units and boundary conventions are explicit
because off-by-one frame errors change both training targets and event scores.
"""

from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler
from typing import NotRequired, TypedDict
import h5py
import hashlib
import math
import random
import soundfile as sf
import torch
from .config import (
    ConvLayerSpec,
)


# =============================================================================
# AUDIO GEOMETRY, I/O, NORMALIZATION, AND RESAMPLING
# =============================================================================

def conv_output_length(
    length: Tensor | int,
    layers: Sequence[ConvLayerSpec],
) -> Tensor | int:
    """Return lengths produced by the official padded audio frontend."""

    result = length.clone() if isinstance(length, Tensor) else int(length)
    for index, (_, kernel, stride) in enumerate(layers):
        # Mathematics: layer 0 and all unit-stride layers use total padding
        # P = K - 1; downsampling layers use P = 2 ceil(s / 2), matching the
        # asymmetric effective "same" padding in the archived implementation.
        # Interpretation: these two padding rules make frame counts line up
        # with both the official convolution stack and its label rasterizer.
        if index == 0 or stride == 1:
            padding = (kernel - 1) // 2
            total_padding = kernel - 1
        else:
            padding = math.ceil(stride / 2)
            total_padding = 2 * padding
        # Mathematics: for input length L, PyTorch convolution produces
        # floor((L + P - K) / s) + 1 positions.
        # Interpretation: applying the formula one layer at a time gives the
        # exact number of frames that every later mask and target tensor needs.
        if isinstance(result, Tensor):
            result = torch.div(result + total_padding - kernel, stride, rounding_mode="floor") + 1
        else:
            result = (result + total_padding - kernel) // stride + 1
        result = result.clamp_min(0) if isinstance(result, Tensor) else max(result, 0)
    return result


def feature_timestamps(
    num_frames: int,
    sample_rate: int,
    layers: Sequence[ConvLayerSpec],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return center timestamps for valid-convolution output frames."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    jump = 1
    center = 0.0
    for index, (_, kernel, stride) in enumerate(layers):
        left_padding = (kernel - 1) // 2 if index == 0 or stride == 1 else math.ceil(stride / 2)
        # Mathematics: if the previous receptive-field centers are separated
        # by j samples, this layer shifts the first center by
        # ((K - 1)/2 - p_left) j and increases the separation to j s.
        # Interpretation: carrying the offset and hop through the stack maps
        # model frames back to physical recording time without approximation.
        center += ((kernel - 1) / 2 - left_padding) * jump
        jump *= stride
    # Mathematics: frame t has center (center + t * jump) / sample_rate seconds.
    # Interpretation: event output uses center times rather than treating frame
    # indices as seconds or assuming that the first frame begins at zero.
    return (center + torch.arange(num_frames, device=device, dtype=torch.float32) * jump) / sample_rate


def load_audio(path: str | Path) -> tuple[Tensor, int]:
    """Load audio as contiguous float32 `[channels, samples]`."""

    samples, sample_rate = sf.read(Path(path), dtype="float32", always_2d=True)
    return torch.from_numpy(samples.T.copy()), int(sample_rate)


def normalize_waveform(waveform: Tensor, eps: float = 1e-5) -> Tensor:
    """Apply Fairseq-style layer normalization over each waveform."""

    # Mathematics: y_s = (x_s - mean_s x_s) /
    # sqrt(mean_s (x_s - mean x)^2 + eps), with no learned affine transform.
    # Interpretation: every recording enters the encoder at zero mean and unit
    # variance, reproducing Fairseq's per-waveform normalization convention.
    return F.layer_norm(waveform.float(), waveform.shape[-1:], eps=eps).to(waveform.dtype)


def resample_waveform(
    waveform: Tensor,
    source_rate: int,
    target_rate: int,
    *,
    lowpass_width: int = 16,
    chunk_size: int = 4096,
) -> Tensor:
    """Band-limited sinc resampling without torchaudio.

    Output positions are processed in chunks, so memory grows with channel count
    and filter width rather than recording duration.
    """

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if waveform.ndim < 1:
        raise ValueError("waveform must have a sample dimension")
    if source_rate == target_rate:
        return waveform.clone()
    if waveform.shape[-1] == 0:
        return waveform.clone()

    original_shape = waveform.shape[:-1]
    source_length = waveform.shape[-1]
    # Mathematics: duration preservation requires S_out / f_out ≈ S_in / f_in,
    # hence S_out = round(S_in f_out / f_in).
    # Interpretation: resampling changes the sample grid while retaining the
    # recording duration seen by segmentation and timestamp code.
    target_length = int(round(source_length * target_rate / source_rate))
    work = waveform.reshape(-1, source_length)
    work_dtype = work.dtype
    if not work.is_floating_point():
        work = work.float()
    calculation_dtype = torch.float64 if work.dtype == torch.float64 else torch.float32
    work = work.to(calculation_dtype)

    ratio = target_rate / source_rate
    # Mathematics: the normalized cutoff is min(1, f_out/f_in); 0.99 leaves a
    # narrow transition band below the lower Nyquist frequency. Expanding the
    # radius by 1/cutoff preserves transition sharpness during downsampling.
    # Interpretation: the filter suppresses frequencies that would alias onto
    # the lower-rate grid without importing torchaudio.
    cutoff = min(1.0, ratio) * 0.99
    radius = max(lowpass_width, int(math.ceil(lowpass_width / cutoff)))
    tap_offsets = torch.arange(-radius + 1, radius + 1, device=work.device)
    pieces: list[Tensor] = []

    for start in range(0, target_length, chunk_size):
        stop = min(start + chunk_size, target_length)
        output_positions = torch.arange(start, stop, device=work.device, dtype=calculation_dtype)
        # Mathematics: output index o corresponds to continuous input position
        # u_o = o f_in / f_out. Neighboring integer indices provide its taps.
        # Interpretation: each new sample interpolates the old waveform at the
        # physical time represented by its output index.
        source_positions = output_positions * (source_rate / target_rate)
        left = torch.floor(source_positions).to(torch.long)
        indices = left[:, None] + tap_offsets[None, :]
        distance = indices.to(calculation_dtype) - source_positions[:, None]
        # Mathematics: h_o[n] = cutoff sinc(cutoff(n-u_o)) w(n-u_o), where w
        # is a compact raised-cosine window on |n-u_o| < radius.
        # Interpretation: the ideal low-pass sinc receives a finite smooth
        # window so computation stays bounded and ringing remains controlled.
        window = torch.where(
            distance.abs() < radius,
            0.5 + 0.5 * torch.cos(torch.pi * distance / radius),
            torch.zeros_like(distance),
        )
        weights = cutoff * torch.sinc(cutoff * distance) * window
        valid = (indices >= 0) & (indices < source_length)
        weights = weights * valid
        # Mathematics: boundary truncation changes sum_n h_o[n], so divide by
        # that sum to preserve a constant signal exactly near the recording edge.
        # Interpretation: clips do not fade at their first and final samples.
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        safe_indices = indices.clamp(0, source_length - 1)
        gathered = work[:, safe_indices]
        # Mathematics: y[c,o] = sum_t x[c,index[o,t]] h[o,t].
        # Interpretation: one tensor contraction applies the same resampling
        # filter independently to every flattened leading channel.
        pieces.append(torch.einsum("cot,ot->co", gathered, weights))

    result = torch.cat(pieces, dim=-1).reshape(*original_shape, target_length)
    return result.to(work_dtype)


# =============================================================================
# SPARSE EVENT LABELS AND FRAME RASTERIZATION
# =============================================================================

@dataclass(frozen=True)
class LabelEvents:
    """Sparse sample-index intervals loaded from one published HDF5 label file."""

    starts: tuple[int, ...]
    ends: tuple[int, ...]
    categories: tuple[int, ...]
    focal: tuple[int, ...]


def derive_label_path(audio_path: Path, label_directory: str = "lbl") -> Path:
    """Map an audio-tree path to its sibling HDF5 label-tree path."""

    parts = list(audio_path.parts)
    candidates = {"wav", "audio", "flac"}
    component = next((index for index in range(len(parts) - 1, -1, -1) if parts[index].lower() in candidates), None)
    if component is None:
        raise ValueError(f"cannot derive label path from audio path without wav/audio/flac component: {audio_path}")
    parts[component] = label_directory
    return Path(*parts).with_suffix(".h5")


def load_label_events(path: Path) -> LabelEvents:
    """Load sparse starts, ends, categories, and focal flags from HDF5."""

    try:
        with h5py.File(path, "r") as handle:
            starts = tuple(int(value) for value in handle["start_frame_lbl"][:])
            ends = tuple(int(value) for value in handle["end_frame_lbl"][:])
            categories = tuple(int(value) for value in handle["lbl_cat"][:])
            focal = tuple(int(value) for value in handle["foc"][:]) if "foc" in handle else tuple(0 for _ in starts)
    except (OSError, KeyError) as exc:
        raise ValueError(f"cannot read label file {path}: {exc}") from exc
    if not (len(starts) == len(ends) == len(categories) == len(focal)):
        raise ValueError(f"label arrays have different lengths in {path}")
    return LabelEvents(starts, ends, categories, focal)


def rasterize_labels(
    events: LabelEvents,
    *,
    waveform_length: int,
    sample_rate: int,
    conv_layers: Sequence[ConvLayerSpec],
    num_labels: int,
    focal_label_index: int | None,
) -> Tensor:
    """Rasterize sparse sample intervals onto convolution output frames."""

    # Mathematics: targets have exactly T = ConvLength(S) rows and C columns.
    # Interpretation: labels and model logits share a shape by construction.
    frame_count = int(conv_output_length(waveform_length, conv_layers))
    targets = torch.zeros(frame_count, num_labels, dtype=torch.float32)
    if frame_count == 0:
        return targets
    # The published dataset code interpolates labels onto an evenly spaced
    # zero-origin grid, rather than onto convolution receptive-field centers.
    # Mathematics: archived label frame t samples round(t S / T), t in
    # {0,...,T-1}; this is an evenly spaced zero-origin grid.
    # Interpretation: this intentionally follows the published dataset code,
    # even though inference timestamps use convolution receptive-field centers.
    sample_positions = torch.round(
        torch.arange(frame_count, dtype=torch.float64) * waveform_length / frame_count
    )
    for start, end, category, focal in zip(
        events.starts, events.ends, events.categories, events.focal, strict=True
    ):
        if not 0 <= category < num_labels:
            raise ValueError(f"label category {category} is outside 0..{num_labels - 1}")
        # Mathematics: an annotation [a,b) activates frame t iff
        # a <= round(tS/T) < b. Endpoints therefore follow half-open semantics.
        # Interpretation: adjacent calls can meet at one sample without
        # producing an overlapping positive frame.
        active = (sample_positions >= start) & (sample_positions < end)
        targets[active, category] = 1
        if focal_label_index is not None and focal == 1:
            targets[active, focal_label_index] = 1
    return targets


# =============================================================================
# MANIFEST DATASET AND BATCH COLLATION
# =============================================================================

class ManifestError(ValueError):
    """Raised when a TSV manifest or one of its referenced files is invalid."""


@dataclass(frozen=True)
class ManifestRecord:
    """One validated manifest row and its resolved filesystem paths."""

    index: int
    audio_path: Path
    label_path: Path | None
    num_samples: int
    manifest_line: int


class AudioItem(TypedDict):
    """Single decoded dataset item before length-aware collation."""

    id: int
    source: Tensor
    target: Tensor | None
    path: str
    crop_seed: NotRequired[int]


@dataclass(frozen=True)
class SampleCoordinate:
    """Dataset index plus a crop seed fixed by its ordered sampler occurrence."""

    index: int
    crop_seed: int


def read_manifest(path: str | Path, *, require_labels: bool = False) -> tuple[ManifestRecord, ...]:
    """Read Fairseq's root-plus-TSV audio manifest format.

    Line 1 is an audio root. Every later line contains a relative path and the
    expected number of waveform samples, separated by a tab.
    """

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if not lines or not lines[0].strip():
        raise ManifestError(f"manifest {path} has no audio root on line 1")
    root = Path(lines[0].strip()).expanduser()
    records: list[ManifestRecord] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ManifestError(f"manifest {path} line {line_number}: expected path and sample count")
        relative, encoded_size = fields
        try:
            size = int(encoded_size)
        except ValueError as exc:
            raise ManifestError(f"manifest {path} line {line_number}: invalid sample count {encoded_size!r}") from exc
        if size <= 0:
            raise ManifestError(f"manifest {path} line {line_number}: sample count must be positive")
        audio_path = root / relative
        if not audio_path.is_file():
            raise ManifestError(f"manifest {path} line {line_number}: missing audio {audio_path}")
        try:
            label_path = derive_label_path(audio_path)
        except ValueError:
            label_path = None
        if require_labels and (label_path is None or not label_path.is_file()):
            raise ManifestError(f"manifest {path} line {line_number}: missing label for {audio_path}")
        records.append(ManifestRecord(len(records), audio_path, label_path, size, line_number))
    return tuple(records)


class AudioDataset(Dataset[AudioItem]):
    """Load manifest audio and optional frame-aligned label tensors on demand."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int,
        conv_layers: Sequence[ConvLayerSpec],
        normalize: bool = True,
        labels: Sequence[str] | None = None,
        min_sample_size: int = 1,
        max_sample_size: int | None = None,
        min_label_size: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.conv_layers = tuple(conv_layers)
        self.normalize = normalize
        self.labels = tuple(labels) if labels is not None else None
        records = read_manifest(manifest_path, require_labels=labels is not None)
        # Mathematics: retain record i iff its sample length satisfies the
        # lower bound and, for supervised data, its label file exceeds the
        # configured byte threshold. max_sample_size caps later collation and
        # therefore does not reject a long recording here.
        # Interpretation: the manifest remains the source of ordering while
        # unusable or empty annotation files leave the training population.
        self.records = tuple(
            record for record in records
            if record.num_samples >= min_sample_size
            and (max_sample_size is None or record.num_samples > 0)
            and (
                labels is None and min_label_size <= 0
                or record.label_path is not None
                and record.label_path.is_file()
                and record.label_path.stat().st_size > min_label_size
            )
        )
        self.sizes = tuple(
            min(record.num_samples, max_sample_size) if max_sample_size is not None else record.num_samples
            for record in self.records
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | SampleCoordinate) -> AudioItem:
        coordinate = index if isinstance(index, SampleCoordinate) else None
        index = coordinate.index if coordinate is not None else index
        record = self.records[index]
        channels, actual_rate = load_audio(record.audio_path)
        if actual_rate != self.sample_rate:
            raise ValueError(
                f"sample rate mismatch for {record.audio_path}: expected {self.sample_rate}, received {actual_rate}"
            )
        # Mathematics: mono sample x_s = C^{-1} sum_c x_{c,s}.
        # Interpretation: the baseline gives every input channel equal weight,
        # matching the official dataset's channel reduction.
        source = channels.mean(dim=0)
        if self.normalize:
            source = normalize_waveform(source)
        target = None
        if self.labels is not None:
            if record.label_path is None:
                raise ValueError(f"missing label path for {record.audio_path}")
            events = load_label_events(record.label_path)
            focal_index = len(self.labels) - 1 if self.labels and self.labels[-1].lower() == "focal" else None
            target = rasterize_labels(
                events,
                waveform_length=source.shape[-1],
                sample_rate=self.sample_rate,
                conv_layers=self.conv_layers,
                num_labels=len(self.labels),
                focal_label_index=focal_index,
            )
        item: AudioItem = {
            "id": index,
            "source": source,
            "target": target,
            "path": str(record.audio_path),
        }
        if coordinate is not None:
            item["crop_seed"] = coordinate.crop_seed
        return item


def collate_audio(
    items: Sequence[AudioItem],
    *,
    max_sample_size: int | None,
    pad: bool,
    conv_layers: Sequence[ConvLayerSpec],
    generator: torch.Generator | None = None,
    crop_strategy: str = "legacy",
) -> dict[str, Tensor | list[str]]:
    """Crop or pad variable waveforms and keep frame labels aligned.

    Legacy crop offsets use the supplied or process-local generator. Stateless
    offsets use sampler-provided per-item seeds that survive process restart.
    """

    if crop_strategy not in {"legacy", "stateless"}:
        raise ValueError("crop_strategy must be legacy or stateless")
    if not items:
        raise ValueError("cannot collate an empty batch")
    lengths = [int(item["source"].shape[-1]) for item in items]
    # Mathematics: padded batches use max_i S_i, cropped batches use min_i S_i,
    # and both are then capped by max_sample_size when it is set.
    # Interpretation: one rectangular tensor either retains every sample with
    # padding or removes excess samples with a reproducible random crop.
    uncapped = max(lengths) if pad else min(lengths)
    target_samples = min(uncapped, max_sample_size) if max_sample_size is not None else uncapped
    sources = items[0]["source"].new_zeros(len(items), target_samples)
    padding_mask = torch.zeros(len(items), target_samples, dtype=torch.bool)
    offsets = torch.zeros(len(items), dtype=torch.long)
    # Mathematics: T_batch = ConvLength(S_batch) uses the same recurrence as
    # the encoder, so target[b] and logits[b] share their temporal dimension.
    # Interpretation: the collator fixes label alignment before data reaches
    # model code.
    target_frames = int(conv_output_length(target_samples, conv_layers))
    has_targets = items[0]["target"] is not None
    targets = None
    if has_targets:
        label_count = int(items[0]["target"].shape[-1])  # type: ignore[union-attr]
        targets = torch.zeros(len(items), target_frames, label_count)

    for batch_index, item in enumerate(items):
        source = item["source"]
        difference = source.shape[-1] - target_samples
        # Mathematics: for S_i > S_batch, sample o uniformly from the inclusive
        # integer range [0, S_i-S_batch].
        # Interpretation: random crops cover all legal windows, and a supplied
        # generator makes the choice repeatable after resume.
        if difference > 0 and crop_strategy == "stateless":
            if "crop_seed" not in item:
                raise ValueError(
                    "stateless crop requires sampler-provided crop coordinates"
                )
            item_generator = torch.Generator().manual_seed(item["crop_seed"])
            offset = int(torch.randint(
                difference + 1,
                (1,),
                generator=item_generator,
            ).item())
        else:
            offset = int(torch.randint(
                difference + 1,
                (1,),
                generator=generator,
            ).item()) if difference > 0 else 0
        offsets[batch_index] = offset
        available = min(source.shape[-1], target_samples)
        sources[batch_index, :available] = source[offset: offset + available]
        if available < target_samples:
            padding_mask[batch_index, available:] = True
        if targets is not None:
            item_target = item["target"]
            if item_target is None:
                raise ValueError("a batch cannot mix labeled and unlabeled examples")
            # Mathematics: map sample crop o to label-frame crop
            # round(o T_i / S_i), the inverse scale used during rasterization.
            # Interpretation: waveform and label crops begin at the same
            # relative recording position despite living on different grids.
            frame_offset = round(offset * item_target.shape[0] / source.shape[-1])
            available_frames = min(target_frames, item_target.shape[0] - frame_offset)
            if available_frames > 0:
                targets[batch_index, :available_frames] = item_target[frame_offset: frame_offset + available_frames]

    batch: dict[str, Tensor | list[str]] = {
        "id": torch.tensor([int(item["id"]) for item in items]),
        "source": sources,
        "crop_offsets": offsets,
        "paths": [item["path"] for item in items],
    }
    if pad:
        batch["padding_mask"] = padding_mask
    if targets is not None:
        batch["target"] = targets
    return batch


# =============================================================================
# TOKEN-BUDGET AND DISTRIBUTED BATCH SAMPLERS
# =============================================================================

class TokenBatchSampler(Sampler[list[int]]):
    """Group similar-length examples under a padded-sample token budget.

    ``next_batch`` records batches delivered to the training loop, rather than
    batches merely prefetched by a DataLoader worker. Its state can therefore
    resume at the exact next optimization input.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        *,
        max_tokens: int,
        seed: int = 1,
        shuffle: bool = True,
        required_batch_size_multiple: int = 1,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.sizes = tuple(int(size) for size in sizes)
        self.max_tokens = max_tokens
        self.seed = seed
        self.shuffle = shuffle
        self.required_multiple = required_batch_size_multiple
        self.epoch = 0
        self.next_batch = 0

    def _batches(self) -> list[list[int]]:
        """Build all batches for the current epoch without changing state.

        The method sorts examples by length before packing. It shuffles the
        completed batches, rather than individual examples, with a seed derived
        from the epoch. Repeated calls in one epoch therefore produce the same
        list.
        """

        # Mathematics: sorting S_i in ascending order reduces max(S_i) - min(S_i)
        # inside a batch and therefore reduces padded samples.
        # Interpretation: the sampler spends the token budget on real audio
        # instead of large amounts of length padding.
        ordered = sorted(range(len(self.sizes)), key=self.sizes.__getitem__)
        batches: list[list[int]] = []
        current: list[int] = []
        for index in ordered:
            proposed = current + [index]
            # Mathematics: a proposed batch is valid iff
            # |B| max_{i in B} S_i <= max_tokens.
            # Interpretation: memory cost follows the padded rectangle, so the
            # longest recording determines every row's allocated width.
            if current and max(self.sizes[item] for item in proposed) * len(proposed) > self.max_tokens:
                split = len(current)
                if self.required_multiple > 1:
                    multiple_split = len(current) - len(current) % self.required_multiple
                    if multiple_split:
                        split = multiple_split
                batches.append(current[:split])
                current = current[split:]
                if current and max(self.sizes[item] for item in current + [index]) * (len(current) + 1) > self.max_tokens:
                    batches.append(current)
                    current = []
            current.append(index)
        if current:
            batches.append(current)
        if self.required_multiple > 1 and batches and len(batches[-1]) % self.required_multiple:
            batches.pop()
        if self.shuffle:
            # Mathematics: epoch e uses a private PRNG seeded by seed + e.
            # Interpretation: batch order changes by epoch but can be rebuilt
            # from two small integers stored in the checkpoint.
            random.Random(self.seed + self.epoch).shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()
        if self.sizes and not batches:
            raise ValueError(
                "the required batch-size multiple drops every example; "
                "set dataset.required_batch_size_multiple to 1 or add more data"
            )
        # Mathematics: yield indices in B[next_batch:], then reset the cursor
        # and increment epoch exactly once after exhaustion.
        # Interpretation: next_batch identifies the first undelivered input,
        # which prevents replay or skipping during exact resume.
        start = self.next_batch
        for position in range(start, len(batches)):
            self.next_batch = position + 1
            yield batches[position]
        self.next_batch = 0
        self.epoch += 1

    def __len__(self) -> int:
        return len(self._batches())

    def state_dict(self) -> dict[str, int]:
        """Return the epoch and next undelivered batch index."""

        return {"epoch": self.epoch, "next_batch": self.next_batch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore iteration state and normalize an exhausted epoch.

        A checkpoint may be written after the final batch was delivered but
        before Python requested the iterator's next value. In that case the
        saved cursor points past the batch list, so restoration advances to the
        next epoch.
        """

        self.epoch = int(state["epoch"])
        self.next_batch = int(state["next_batch"])
        if self.next_batch >= len(self._batches()) and self.next_batch > 0:
            self.next_batch = 0
            self.epoch += 1


class DistributedBatchSampler(Sampler[list[int]]):
    """Assign one complete token batch per rank in each synchronized round."""

    def __init__(self, sampler: TokenBatchSampler, *, rank: int, world_size: int) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.sampler = sampler
        self.rank = rank
        self.world_size = world_size
        remaining = len(self.sampler._batches()) - self.sampler.next_batch
        if self.sampler.next_batch > 0 and remaining < world_size:
            self.sampler.next_batch = 0
            self.sampler.epoch += 1
            remaining = len(self.sampler._batches())
        if remaining < world_size:
            raise ValueError(
                "the token sampler produced fewer complete batches than distributed workers; "
                "reduce world size or max_tokens"
            )

    def __iter__(self) -> Iterator[list[int]]:
        batches = self.sampler._batches()
        start = self.sampler.next_batch
        # Mathematics: floor((N-start)/world_size) synchronized rounds discard
        # an incomplete final group; rank r receives global index start+kw+r.
        # Interpretation: every rank executes the same number of optimizer
        # steps and no worker waits for a batch that another worker lacks.
        rounds = max(0, len(batches) - start) // self.world_size
        for round_index in range(rounds):
            round_start = start + round_index * self.world_size
            # Every rank records the same next global batch. A rank-zero
            # checkpoint can therefore resume all ranks without replaying a
            # batch consumed by another rank.
            self.sampler.next_batch = round_start + self.world_size
            yield batches[round_start + self.rank]
        self.sampler.next_batch = 0
        self.sampler.epoch += 1

    def __len__(self) -> int:
        remaining = max(0, len(self.sampler) - self.sampler.next_batch)
        return remaining // self.world_size


class StatelessCropBatchSampler(Sampler[list[SampleCoordinate]]):
    """Attach resume-invariant crop seeds to rank-selected sample occurrences."""

    def __init__(
        self,
        batch_sampler: Sampler[list[int]],
        *,
        token_sampler: TokenBatchSampler,
        seed: int,
    ) -> None:
        self.batch_sampler = batch_sampler
        self.token_sampler = token_sampler
        self.seed = int(seed)

    def __iter__(self) -> Iterator[list[SampleCoordinate]]:
        epoch = self.token_sampler.epoch
        start = self.token_sampler.next_batch
        if isinstance(self.batch_sampler, DistributedBatchSampler):
            rank = self.batch_sampler.rank
            stride = self.batch_sampler.world_size
        else:
            rank = 0
            stride = 1
        for local_occurrence, batch in enumerate(self.batch_sampler):
            global_occurrence = start + local_occurrence * stride + rank
            coordinates: list[SampleCoordinate] = []
            for batch_position, index in enumerate(batch):
                encoded = (
                    f"{self.seed}:{epoch}:{global_occurrence}:"
                    f"{rank}:{batch_position}:{index}"
                ).encode("ascii")
                crop_seed = int.from_bytes(
                    hashlib.sha256(encoded).digest()[:8],
                    byteorder="big",
                ) & ((1 << 63) - 1)
                coordinates.append(SampleCoordinate(index, crop_seed))
            yield coordinates

    def __len__(self) -> int:
        return len(self.batch_sampler)
