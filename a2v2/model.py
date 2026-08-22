"""Complete Animal2Vec 1.0 baseline mathematics in one reading path.

This file starts with small differentiable primitives and proceeds through
normalization, the learnable Sinc filterbank, attention, masking, reconstruction
decoding, objectives, the EMA teacher, the shared audio encoder, and the two
paper tasks. A reader can therefore follow a waveform from the first convolution
to a pretraining regression loss or fine-tuning event logits without crossing
package boundaries.

Shape notation used throughout:

* ``B`` is batch size, ``S`` waveform samples, and ``T`` feature frames.
* ``D`` is the encoder embedding dimension and ``H`` the attention-head count.
* ``M`` is the number of masked frames and ``C`` the target-class count.

Every compatibility-sensitive calculation receives two comments in the order
requested by the repository: ``Mathematics`` records the equation, axes, or
shape mapping; ``Interpretation`` explains why the baseline performs it.
"""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .config import (
    Animal2VecConfig,
    ConvLayerSpec,
    DecoderConfig,
    resolve_attention_backend,
    resolve_position_encoding,
)
from .data import (
    conv_output_length,
)


# =============================================================================
# DIFFERENTIABLE LAYERS AND INITIALIZATION
# =============================================================================

class GradMultiply(torch.autograd.Function):
    """Identity in the forward pass that scales only the backward gradient."""

    @staticmethod
    def forward(ctx: object, value: Tensor, scale: float) -> Tensor:
        """Copy values unchanged and store the gradient multiplier."""

        # Mathematics: forward(x, λ) = x, so predictions and losses remain
        # identical for every multiplier λ.
        # Interpretation: fine-tuning can change how strongly the local
        # encoder learns without changing the features sent to later layers.
        ctx.scale = scale  # type: ignore[attr-defined]
        return value.clone()

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        """Scale the gradient with respect to the input tensor only."""

        # Mathematics: dL/dx = λ dL/dy and dL/dλ is undefined because λ is a
        # fixed hyperparameter rather than a differentiable input.
        # Interpretation: zero freezes upstream features, values below one
        # slow their adaptation, and one leaves ordinary backpropagation.
        return gradient * ctx.scale, None  # type: ignore[attr-defined]


class SamePad(nn.Module):
    """Trim the final position after symmetric padding with an even kernel."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, value: Tensor) -> Tensor:
        """Remove one trailing position only when an even kernel created it."""

        # Mathematics: symmetric padding with even K yields T+1 positions, so
        # selecting [...,:T] restores the input temporal length.
        # Interpretation: residual branches can add tensors without a one-frame
        # mismatch after an even-width positional or decoder convolution.
        return value[..., : -self.remove] if self.remove else value


class TransposeLast(nn.Module):
    """Swap a selected dimension with the last dimension."""

    def __init__(self, transpose_dim: int = -2) -> None:
        super().__init__()
        self.transpose_dim = transpose_dim

    def forward(self, value: Tensor) -> Tensor:
        """Swap ``transpose_dim`` with the final tensor dimension."""

        return value.transpose(self.transpose_dim, -1)


class PSwish(nn.Module):
    """Per-channel parametric Swish used after the Sinc filters."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        shape = (1, num_features, 1)
        self.p_swish_alpha = nn.Parameter(torch.full(shape, 2.0))
        self.p_swish_beta = nn.Parameter(torch.zeros(shape))

    def forward(self, value: Tensor) -> Tensor:
        """Apply channelwise ``alpha * x * sigmoid(beta * x)``."""

        # Mathematics: y_{bct} = α_c x_{bct} σ(β_c x_{bct}), with α and β
        # broadcast across batch and time.
        # Interpretation: each learned acoustic filter controls the magnitude
        # and curvature of its own smooth activation.
        return value * self.p_swish_alpha * torch.sigmoid(self.p_swish_beta * value)


class DropPath(nn.Module):
    """Stochastic depth with one Bernoulli decision per sample."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = probability

    def forward(self, value: Tensor) -> Tensor:
        """Drop or rescale complete sample residual paths during training."""

        if not self.training or self.probability == 0.0:
            return value
        keep_probability = 1.0 - self.probability
        # Mathematics: sample m_b ~ Bernoulli(q) once per residual path and use
        # y_b = x_b m_b / q, so E[y_b] = x_b.
        # Interpretation: stochastic depth removes a whole block path for a
        # sample while keeping the test-time activation scale unchanged.
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        keep = torch.empty(shape, dtype=value.dtype, device=value.device).bernoulli_(keep_probability)
        return value * keep / keep_probability


def init_bert_params(module: nn.Module) -> None:
    """Initialization used by Fairseq's transformer sentence encoder."""

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()


# =============================================================================
# FULL-PRECISION NORMALIZATION
# =============================================================================

class Fp32LayerNorm(nn.LayerNorm):
    """Layer normalization computed in float32, returned in the input dtype."""

    def forward(self, value: Tensor) -> Tensor:
        """Normalize the final configured dimensions in float32."""

        # Mathematics: LayerNorm uses μ and σ² over normalized_shape and
        # computes γ (x-μ)/sqrt(σ²+eps)+β in float32.
        # Interpretation: reductions keep enough precision under AMP, then the
        # result returns to the surrounding model dtype.
        output = F.layer_norm(
            value.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(value.dtype)


class Fp32GroupNorm(nn.GroupNorm):
    """Group normalization computed in float32, returned in the input dtype."""

    def forward(self, value: Tensor) -> Tensor:
        """Normalize channel groups in float32."""

        # Mathematics: GroupNorm computes one mean and variance for every
        # sample/group over that group's channels and remaining spatial axes.
        # Interpretation: channel groups share scale statistics without
        # depending on other recordings in the batch.
        output = F.group_norm(
            value.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(value.dtype)


class Fp32InstanceNorm(nn.InstanceNorm1d):
    """Instance normalization with float32 math and optional time transpose."""

    def __init__(self, *args: object, transpose_last: bool = False, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.transpose_last = transpose_last

    def forward(self, value: Tensor) -> Tensor:
        """Normalize each sample/channel series in float32."""

        # Mathematics: InstanceNorm1d expects [B,C,T]; transpose_last maps
        # [B,T,C] into that convention without changing stored values.
        # Interpretation: callers can use frame-major transformer tensors while
        # the normalization kernel still treats each channel as a time series.
        work = value.transpose(1, 2) if self.transpose_last else value
        output = F.instance_norm(
            work.float(),
            self.running_mean.float() if self.running_mean is not None else None,
            self.running_var.float() if self.running_var is not None else None,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )
        if self.transpose_last:
            output = output.transpose(1, 2)
        return output.to(value.dtype)


# =============================================================================
# LEARNABLE SINC FILTERBANK
# =============================================================================

class SincConv1d(nn.Module):
    """Analytic band-pass convolution used for the first waveform layer.

    Trainable cutoff frequencies generate normalized Sinc filters by default.
    The optional ``learnable_filters`` mode instead stores the full kernel,
    matching the published frontend's alternate initialization path.
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        *,
        in_channels: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: str = "same",
        padding_mode: str = "reflect",
        sample_rate: int = 8000,
        min_low_hz: float = 50.0,
        min_band_hz: float | None = None,
        learnable_filters: bool = False,
        apply_window_to_root: bool = False,
        return_abs: bool = False,
        init_scale: str = "mel",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SincConv1d kernel size must be odd")
        if out_channels % in_channels:
            raise ValueError("out_channels must be divisible by in_channels")
        if padding not in {"same", "valid", "causal"}:
            raise ValueError("padding must be 'same', 'valid', or 'causal'")
        if apply_window_to_root and not learnable_filters:
            raise ValueError("apply_window_to_root requires learnable_filters")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.sample_rate = sample_rate
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(math.ceil(sample_rate / kernel_size) if min_band_hz is None else min_band_hz)
        self.learnable_filters = learnable_filters
        self.apply_window_to_root = apply_window_to_root
        self.return_abs = return_abs

        # Mathematics: admissible passbands obey
        # min_low <= f_low < f_high <= f_s/2 and width >= min_band.
        # Interpretation: initialized filters cannot cross DC or Nyquist and
        # cannot collapse into a zero-width band.
        high_hz = sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        if init_scale == "mel":
            mel = torch.linspace(self._to_mel(self.min_low_hz), self._to_mel(high_hz), out_channels + 1)
            hz = self._to_hz(mel)
        elif init_scale == "linear":
            hz = torch.linspace(self.min_low_hz, high_hz, out_channels + 1)
        else:
            raise ValueError("init_scale must be 'mel' or 'linear'")
        # Mathematics: adjacent grid points define [f_i,f_{i+1}], so lower
        # cutoffs are hz[:-1] and bandwidths are hz[1:]-hz[:-1].
        # Interpretation: the initial bank tiles the usable spectrum in either
        # perceptual Mel spacing or linear Hertz spacing.
        low_hz = hz[:-1].unsqueeze(1)
        band_hz = (hz[1:] - hz[:-1]).unsqueeze(1)

        half_positions = torch.linspace(0, kernel_size / 2 - 1, steps=kernel_size // 2)
        # Mathematics: w[n] = 0.53836 - 0.46164 cos(2πn/K) is a Hamming window.
        # Interpretation: tapering the finite analytic kernel reduces spectral
        # leakage created by truncating an infinite sinc response.
        half_window = 0.53836 - 0.46164 * torch.cos(2 * math.pi * half_positions / kernel_size)
        half_width = (kernel_size - 1) / 2
        time_axis = 2 * math.pi * torch.arange(-half_width, 0).view(1, -1) / sample_rate
        full_positions = torch.linspace(0, kernel_size - 1, steps=kernel_size)
        full_window = 0.53836 - 0.46164 * torch.cos(2 * math.pi * full_positions / kernel_size)
        self.register_buffer("n_", time_axis, persistent=False)
        self.register_buffer("window_", half_window, persistent=False)
        self.register_buffer("window_full", full_window, persistent=False)

        if learnable_filters:
            self.register_buffer("low_hz_", low_hz, persistent=False)
            self.register_buffer("band_hz_", band_hz, persistent=False)
            self.kernel = nn.Parameter(self._sinc_filters())
        else:
            self.low_hz_ = nn.Parameter(low_hz)
            self.band_hz_ = nn.Parameter(band_hz)

    @staticmethod
    def _to_mel(frequency: float | Tensor) -> float | Tensor:
        """Convert hertz to the perceptual Mel frequency scale."""

        if isinstance(frequency, Tensor):
            return 2595 * torch.log10(1 + frequency / 700)
        return 2595 * math.log10(1 + frequency / 700)

    @staticmethod
    def _to_hz(mel: Tensor) -> Tensor:
        """Convert Mel values back to hertz."""

        return 700 * (torch.pow(10.0, mel / 2595) - 1)

    def frequency_bounds(self) -> tuple[Tensor, Tensor]:
        """Return positive, Nyquist-clamped lower and upper filter cutoffs."""

        # Mathematics: absolute offsets parameterize positive lower cutoffs and
        # positive bandwidths; clamp enforces f_high <= Nyquist.
        # Interpretation: unconstrained optimizer parameters still materialize
        # valid physical band-pass filters on every forward pass.
        low = self.min_low_hz + self.low_hz_.abs()
        high = torch.clamp(
            low + self.min_band_hz + self.band_hz_.abs(),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        return low, high

    def _sinc_filters(self) -> Tensor:
        """Construct normalized, symmetric band-pass kernels in float32."""

        low, high = self.frequency_bounds()
        band = (high - low)[:, 0]
        time_axis = self.n_.to(device=low.device, dtype=torch.float32)
        window = self.window_.to(device=low.device, dtype=torch.float32)
        # Mathematics: an ideal band pass equals the difference of two low-pass
        # responses, 2[sin(2πf_h t)-sin(2πf_l t)]/(2πt), with center value
        # 2(f_h-f_l) obtained by the t -> 0 limit.
        # Interpretation: the network learns only band edges while the complete
        # time-domain kernel follows from signal-processing structure.
        low_phase = low.float() @ time_axis
        high_phase = high.float() @ time_axis
        left = 2 * (torch.sin(high_phase) - torch.sin(low_phase)) / time_axis * window
        center = 2 * band[:, None]
        filters = torch.cat((left, center, left.flip(1)), dim=1)
        # Mathematics: dividing by 2(f_h-f_l) normalizes each filter by its
        # analytic center amplitude.
        # Interpretation: wide initial bands do not gain a larger activation
        # scale solely because their passband spans more Hertz.
        filters = filters / (2 * band[:, None])
        return filters.view(self.out_channels, 1, self.kernel_size)

    def filters(self) -> Tensor:
        """Materialize current convolution kernels."""

        return self.kernel if self.learnable_filters else self._sinc_filters()

    def _pad(self, value: Tensor) -> Tensor:
        """Apply valid, causal, or same padding before convolution."""

        if self.padding == "valid":
            return value
        if self.padding == "causal":
            return F.pad(value, ((self.kernel_size - 1) * self.dilation, 0))
        # Mathematics: effective kernel width is d(K-1)+1, so total padding
        # d(K-1) preserves length at unit stride.
        # Interpretation: the filterbank can change dilation without silently
        # shifting downstream frame geometry.
        total = self.dilation * (self.kernel_size - 1)
        left = total // 2
        right = total - left
        if self.padding_mode == "reflect" and left < value.shape[-1] and right < value.shape[-1]:
            # CUDA's reflection_pad1d backward uses atomics and cannot run
            # under deterministic algorithms. These slices are exactly the
            # same reflection, while cat/flip has a deterministic backward.
            return torch.cat((
                value[..., 1 : left + 1].flip(-1),
                value,
                value[..., -(right + 1) : -1].flip(-1),
            ), dim=-1)
        return F.pad(value, (left, right), mode=self.padding_mode)

    def forward(self, value: Tensor) -> Tensor:
        """Filter a waveform batch and optionally return output magnitudes."""

        squeeze_channel = value.ndim == 2
        if squeeze_channel:
            value = value.unsqueeze(1)
        if value.ndim != 3 or value.shape[1] != self.in_channels:
            raise ValueError(f"expected [batch, {self.in_channels}, samples] input")
        padded = self._pad(value)
        kernels = self.filters()
        # Mathematics: y_{b,c,t} = sum_k h_{c,k} x_{b,c_in,ts+kd}; groups makes
        # each input channel use its own subset of filters.
        # Interpretation: the first layer converts raw waveform neighborhoods
        # into learned, frequency-selective feature channels.
        output = F.conv1d(
            padded.float(),
            kernels.float(),
            stride=self.stride,
            dilation=self.dilation,
            groups=self.in_channels,
        ).to(value.dtype)
        if self.learnable_filters and self.apply_window_to_root:
            with torch.no_grad():
                self.kernel.mul_(self.window_full.to(self.kernel))
        return output.abs() if self.return_abs else output


SincConv = SincConv1d


# =============================================================================
# MULTI-HEAD ATTENTION AND ALIBI
# =============================================================================

def alibi_slopes(num_heads: int, *, device: torch.device | str | None = None) -> Tensor:
    """Return the deterministic attention slope assigned to each ALiBi head."""

    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    def power_of_two(count: int) -> list[float]:
        """Construct the geometric slope sequence for a power-of-two head count."""

        # Mathematics: m_h = start^(h+1), where
        # start = 2^(-2^(-(log2(H)-3))).
        # Interpretation: heads span short through long temporal attention
        # ranges without learned positional embeddings.
        start = 2 ** (-(2 ** -(math.log2(count) - 3)))
        return [start * start**index for index in range(count)]

    if math.log2(num_heads).is_integer():
        values = power_of_two(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        values = power_of_two(closest) + power_of_two(2 * closest)[0::2][: num_heads - closest]
    return torch.tensor(values, device=device)


def alibi_bias(
    num_heads: int,
    length: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build ``[heads, query_frames, key_frames]`` linear distance biases."""

    # Mathematics: B_{hij} = -m_h |i-j|, an additive penalty linear in frame
    # distance with head-specific positive slope m_h.
    # Interpretation: nearby audio frames receive less positional penalty, and
    # different heads retain different temporal horizons.
    positions = torch.arange(length, device=device)
    distance = -(positions[None, :] - positions[:, None]).abs()
    return alibi_slopes(num_heads, device=device).to(dtype).view(-1, 1, 1) * distance.to(dtype)


def apply_rotary_position_embedding(
    query: Tensor,
    key: Tensor,
    position_ids: Tensor,
    *,
    theta: float,
) -> tuple[Tensor, Tensor]:
    """Rotate packed-projection Q/K pairs at explicit scalar frame IDs."""

    head_dimension = query.shape[-1]
    if head_dimension % 2:
        raise ValueError("RoPE attention head dimension must be even")
    if query.shape != key.shape:
        raise ValueError("RoPE query and key tensors must have matching shapes")
    if position_ids.shape != (query.shape[0], query.shape[-2]):
        raise ValueError("position_ids must have shape [batch, frames]")

    # Mathematics: each adjacent coordinate pair is rotated by
    # p * theta^(-2i/d), with p taken from the original frame coordinate.
    # Interpretation: shuffled student tokens retain the teacher frame phase
    # instead of being renumbered on their shortened sequence axis.
    frequencies = torch.arange(
        0,
        head_dimension,
        2,
        device=query.device,
        dtype=torch.float32,
    )
    frequencies = theta ** (-frequencies / head_dimension)
    angles = position_ids.to(device=query.device, dtype=torch.float32).unsqueeze(-1) * frequencies
    cosine = angles.cos().unsqueeze(1)
    sine = angles.sin().unsqueeze(1)

    def rotate(value: Tensor) -> Tensor:
        """Apply the FP32 two-coordinate rotations and restore input dtype."""

        pairs = value.float().reshape(*value.shape[:-1], head_dimension // 2, 2)
        first, second = pairs.unbind(-1)
        rotated = torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        )
        return rotated.flatten(-2).to(value.dtype)

    return rotate(query), rotate(key)


class MultiheadAttention(nn.Module):
    """Self-attention with state layout matching Animal2Vec's AltAttention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        cosine_attention: bool = False,
        position_encoding: str = "none",
        rope_theta: float = 10_000.0,
        attention_backend: str = "manual",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        self.cosine_attention = cosine_attention
        if position_encoding not in {"alibi", "rope", "none"}:
            raise ValueError("position_encoding must be alibi, rope, or none")
        if position_encoding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE attention head dimension must be even")
        if attention_backend not in {"manual", "sdpa", "flash"}:
            raise ValueError("attention_backend must be manual, sdpa, or flash")
        if position_encoding == "alibi" and attention_backend != "manual":
            raise ValueError("ALiBi attention requires the manual backend")
        self.position_encoding = position_encoding
        self.rope_theta = rope_theta
        self.attention_backend = attention_backend
        if cosine_attention:
            self.logit_scale = nn.Parameter(torch.log(torch.full((num_heads, 1, 1), 10.0)))

    def forward(
        self,
        value: Tensor,
        padding_mask: Tensor | None = None,
        alibi: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """Apply self-attention to a sequence of audio-frame embeddings.

        Args:
            value: Input with shape ``[batch, frames, embedding]``.
            padding_mask: Optional ``[batch, frames]`` mask. ``True`` marks
                keys that represent padding and must receive no attention.
            alibi: Optional additive bias with shape
                ``[heads, frames, frames]`` or
                ``[batch, heads, frames, frames]``.

        Returns:
            Contextualized embeddings with the same shape and dtype as
            ``value``.

        The softmax runs in float32 even under mixed precision. This prevents
        overflow in long sequences while preserving the surrounding autocast
        dtype for projections and matrix multiplication.
        """

        if self.position_encoding == "rope" and alibi is not None:
            raise ValueError("RoPE and ALiBi position encodings are mutually exclusive")
        batch, length, dimension = value.shape
        # Mathematics: one affine map produces [Q,K,V] in R^{B×T×3×H×d_h},
        # followed by a permutation to R^{3×B×H×T×d_h}.
        # Interpretation: the packed projection preserves the official
        # checkpoint layout while exposing one tensor per attention role.
        qkv = self.qkv(value).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, projected_value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.position_encoding == "rope":
            if position_ids is None:
                raise ValueError("RoPE attention requires explicit position_ids")
            query, key = apply_rotary_position_embedding(
                query,
                key,
                position_ids,
                theta=self.rope_theta,
            )
        if self.attention_backend == "manual":
            input_dtype = query.dtype
            if self.cosine_attention:
                scores = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
                maximum = torch.log(torch.tensor(100.0, device=value.device))
                scores = scores * self.logit_scale.clamp(max=maximum).exp()
            else:
                # Mathematics: s_{bhij} = q_{bhi}·k_{bhj}/sqrt(d_h), unless an
                # explicit qk_scale overrides the standard factor.
                # Interpretation: scaling prevents dot-product variance from
                # growing with head width and saturating the softmax.
                scores = (query * self.scale) @ key.transpose(-2, -1)
            if alibi is not None:
                if alibi.ndim == 3:
                    alibi = alibi.unsqueeze(0)
                # Mathematics: positional log-prior B_{hij} adds to content logits
                # before normalization.
                # Interpretation: attention combines learned acoustic similarity
                # with a fixed or scaled preference for temporal proximity.
                scores = scores.to(alibi.dtype) + alibi
            if padding_mask is not None and padding_mask.any():
                # Mathematics: padded keys receive -∞, hence exp(-∞)=0 in softmax.
                # Interpretation: real frames cannot attend to batch padding.
                scores = scores.masked_fill(padding_mask[:, None, None, :].bool(), float("-inf"))
            # Mathematics: a_{bhij} = exp(s_{bhij})/sum_j exp(s_{bhij}) and
            # z_{bhi} = sum_j a_{bhij} v_{bhj}.
            # Interpretation: float32 normalization protects long sequences from
            # half-precision overflow while weighted values retain the model dtype.
            weights = scores.softmax(dim=-1, dtype=torch.float32).to(input_dtype)
            attended = self.attn_drop(weights) @ projected_value
        else:
            if alibi is not None:
                raise ValueError("ALiBi attention requires the manual backend")
            if self.cosine_attention:
                raise ValueError("cosine attention requires the manual backend")
            allowed_keys = None
            if padding_mask is not None:
                # Mathematics: SDPA boolean masks use True=allowed, the inverse
                # of the encoder's True=padding key mask.
                # Interpretation: only keys are excluded; padded query rows are
                # retained exactly as in the legacy attention path.
                allowed_keys = ~padding_mask[:, None, None, :].bool()
            dropout_probability = self.attn_drop.p if self.training else 0.0

            def fused_attention() -> Tensor:
                """Call SDPA with the configured scale and key-only mask."""

                return F.scaled_dot_product_attention(
                    query,
                    key,
                    projected_value,
                    attn_mask=allowed_keys,
                    dropout_p=dropout_probability,
                    scale=self.scale,
                )

            if self.attention_backend == "flash":
                details = (
                    "strict FlashAttention failed "
                    f"(backend=flash, dtype={query.dtype}, device={query.device}, "
                    f"head_dimension={self.head_dim}, length={length}, "
                    f"padding={padding_mask is not None})"
                )
                if query.device.type != "cuda":
                    raise RuntimeError(f"{details}: a CUDA tensor is required")
                try:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        attended = fused_attention()
                except RuntimeError as error:
                    raise RuntimeError(f"{details}: {error}") from error
            else:
                attended = fused_attention()
        attended = attended.transpose(1, 2)
        attended = attended.reshape(batch, length, dimension)
        return self.proj_drop(self.proj(attended))


# =============================================================================
# TIME-STEP MASK SAMPLING
# =============================================================================

@dataclass(frozen=True)
class MaskInfo:
    """Mask plus gather indices for removing and later restoring frame tokens."""

    x_unmasked: Tensor | None
    mask: Tensor
    ids_restore: Tensor
    ids_keep: Tensor


def compute_mask_indices(
    shape: tuple[int, int],
    padding_mask: Tensor | None,
    mask_prob: float,
    mask_length: int,
    mask_type: Literal["static", "uniform", "normal", "poisson"] = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: int | None = None,
    epoch: int | None = None,
    indices: Tensor | None = None,
    idc_select_ver: int = 1,
    num_mask_ver: int = 2,
) -> Tensor:
    """Compute masks with the algorithm used by Fairseq 0.12 Data2Vec."""

    batch_size, all_size = shape
    if mask_length <= 0:
        raise ValueError("mask_length must be positive")
    if padding_mask is not None and tuple(padding_mask.shape) != shape:
        raise ValueError("padding_mask shape must equal mask shape")
    # Mathematics: mask[b,t] begins as the all-false indicator in
    # {0,1}^{B×T}; later span placement changes selected entries to true.
    # Interpretation: building on CPU NumPy reproduces Fairseq's exact random
    # algorithm before the mask moves to the tensor device.
    mask = np.full((batch_size, all_size), False)

    if num_mask_ver == 1:
        all_num_mask = int(mask_prob * all_size / mask_length + np.random.rand())
        all_num_mask = max(min_masks, all_num_mask)

    mask_indices: list[np.ndarray] = []
    rng = np.random.default_rng()
    for batch_index in range(batch_size):
        if seed is not None and epoch is not None and indices is not None:
            # Fairseq used a float modulus. Keep it because converting the
            # large hash to float changes the rounded seed on some values.
            item_seed = int(hash((seed, epoch, indices[batch_index].item())) % 1e6)
        else:
            item_seed = None
        rng = np.random.default_rng(item_seed)
        size = all_size
        if padding_mask is not None:
            size -= int(padding_mask[batch_index].long().sum().item())
        if size <= 1:
            raise ValueError(f"sequence is too short to mask: {size}")

        if num_mask_ver == 1:
            num_mask = (
                int(mask_prob * size / mask_length + np.random.rand())
                if padding_mask is not None else all_num_mask
            )
        elif num_mask_ver == 2:
            # Mathematics: N = floor(p T / L + U), U~Uniform[0,1), is stochastic
            # rounding of the expected span count pT/L.
            # Interpretation: mean masked coverage follows mask_prob without a
            # systematic rounding bias on short recordings.
            num_mask = int(mask_prob * size / mask_length + rng.random())
        else:
            raise ValueError(f"unknown num_mask_ver: {num_mask_ver}")
        num_mask = max(min_masks, num_mask)

        # Mathematics: the four modes draw span lengths L_n from a point mass,
        # discrete uniform, rounded normal, or rounded Poisson distribution.
        # Interpretation: published recipes use static spans, while keeping the
        # archived alternatives makes converted configurations explicit.
        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.integers(int(mask_other), mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.asarray([max(1, int(round(value))) for value in rng.normal(mask_length, mask_other, size=num_mask)])
        elif mask_type == "poisson":
            lengths = np.asarray([int(round(value)) for value in rng.poisson(mask_length, size=num_mask)])
        else:
            raise ValueError(f"unknown mask selection: {mask_type}")
        if lengths.sum() == 0:
            if mask_type == "static":
                raise ValueError("static masks cannot have zero total length")
            lengths = np.asarray([min(mask_length, size - 1)])

        if no_overlap:
            selected: list[int] = []

            def arrange(start: int, end: int, length: int, keep_length: int) -> list[tuple[int, int]]:
                """Place one non-overlapping span and return usable remainders."""

                span_start = int(rng.integers(start, end - length))
                selected.extend(span_start + offset for offset in range(length))
                new_parts: list[tuple[int, int]] = []
                if span_start - start - min_space >= keep_length:
                    new_parts.append((start, span_start - min_space + 1))
                if end - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, end))
                return new_parts

            parts = [(0, size)]
            min_length = int(lengths.min())
            for length_value in sorted(lengths, reverse=True):
                length = int(length_value)
                available = np.fromiter(
                    (end - start if end - start >= length + min_space else 0 for start, end in parts),
                    dtype=np.int64,
                )
                if available.sum() == 0:
                    break
                # Mathematics: choose free interval r with probability
                # |r| / sum_j |r_j| among intervals that fit this span.
                # Interpretation: longer unused regions receive proportionally
                # more placement opportunities while min_space remains intact.
                chosen_part = int(rng.choice(len(parts), p=available / available.sum()))
                start, end = parts.pop(chosen_part)
                parts.extend(arrange(start, end, length, min_length))
            item_indices = np.asarray(selected, dtype=np.int64)
        else:
            min_length = int(lengths.min())
            if idc_select_ver == 1:
                if size - min_length <= num_mask:
                    min_length = size - num_mask - 1
                starts = rng.choice(size - min_length, num_mask, replace=False)
            elif idc_select_ver == 2:
                starts = rng.choice(size, num_mask, replace=False)
            else:
                raise ValueError(f"unknown idc_select_ver: {idc_select_ver}")
            # Mathematics: each sampled start s_n expands to
            # {s_n,...,s_n+L_n-1}; unique later forms their union.
            # Interpretation: overlapping spans are allowed in the default
            # mode and naturally reduce the realized mask count.
            item_indices = np.asarray([
                starts[span] + offset
                for span in range(len(starts))
                for offset in range(int(lengths[span]))
            ])

        item_indices = np.unique(item_indices[item_indices < size])
        if len(item_indices) >= size:
            raise ValueError(f"the entire sequence is masked: size={size}")
        mask_indices.append(item_indices)

    target_length: int | None = None
    if require_same_masks:
        lengths = [len(item) for item in mask_indices]
        # Mathematics: equalization selects max_b |M_b| when adding masks and
        # min_b |M_b| when removing them.
        # Interpretation: every batch item must expose the same number of
        # unmasked tokens so gathering produces one dense tensor.
        target_length = max(lengths) if add_masks else min(lengths)

    for batch_index, item_indices in enumerate(mask_indices):
        if target_length is not None and len(item_indices) > target_length:
            item_indices = rng.choice(item_indices, target_length, replace=False)
        mask[batch_index, item_indices] = True
        if target_length is not None and len(item_indices) < target_length:
            valid_size = all_size
            if padding_mask is not None:
                valid_size -= int(padding_mask[batch_index].long().sum().item())
            unmasked = np.flatnonzero(~mask[batch_index, :valid_size])
            extra = rng.choice(unmasked, target_length - len(item_indices), replace=False)
            mask[batch_index, extra] = True
        if mask_dropout > 0:
            masked = np.flatnonzero(mask[batch_index])
            holes = int(np.rint(len(masked) * mask_dropout))
            if holes:
                mask[batch_index, rng.choice(masked, holes, replace=False)] = False

    return torch.from_numpy(mask)


def make_mask_info(features: Tensor, mask: Tensor) -> MaskInfo:
    """Gather unmasked tokens and record their inverse temporal permutation."""

    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError("features must be [batch, time, dim] and mask [batch, time]")
    mask = mask.to(device=features.device, dtype=torch.uint8)
    counts = mask.sum(dim=1)
    if not torch.all(counts == counts[0]):
        raise ValueError("each batch item must have the same number of masks")
    _, time, dimension = features.shape
    # Mathematics: sorting the binary mask places zeros (kept frames) before
    # ones (masked frames); argsort(argsort(mask)) is the inverse permutation.
    # Interpretation: the student can process a shorter dense sequence and the
    # decoder can later restore original chronological order.
    ids_shuffle = mask.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, dimension)
    keep_length = time - int(counts[0].item())
    ids_keep_plain = ids_shuffle[:, :keep_length]
    ids_keep = ids_keep_plain.unsqueeze(-1).expand(-1, -1, dimension)
    x_unmasked = torch.gather(features, dim=1, index=ids_keep)
    return MaskInfo(x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep)


def masks_for_cloned_batch(
    *,
    batch_size: int,
    length: int,
    clone_count: int,
    mask_prob: float,
    mask_length: int,
    global_seed: int,
    update: int,
    sample_ids: Tensor,
    padding_mask: Tensor | None = None,
    mask_dropout: float = 0.0,
) -> Tensor:
    """Create independent deterministic masks for each cloned batch item."""

    if sample_ids.numel() != batch_size:
        raise ValueError("sample_ids must contain one ID per batch item")
    # Mathematics: clone c receives stable id i + h(seed,c), so the mask seed
    # hash differs across clones while remaining tied to sample i and update u.
    # Interpretation: multiple masked views of one recording stay reproducible
    # across worker count, rank, and checkpoint resume.
    clone_hashes = [0] + [int(hash((global_seed, index)) % 1e10) for index in range(clone_count - 1)]
    expanded_ids = sample_ids.repeat_interleave(clone_count).view(-1, clone_count)
    expanded_ids = (expanded_ids + torch.tensor(clone_hashes, device=sample_ids.device)).reshape(-1)
    expanded_padding = padding_mask.repeat_interleave(clone_count, dim=0) if padding_mask is not None else None
    return compute_mask_indices(
        (batch_size * clone_count, length),
        expanded_padding,
        mask_prob,
        mask_length,
        min_masks=1,
        require_same_masks=True,
        mask_dropout=mask_dropout,
        seed=global_seed,
        epoch=update,
        indices=expanded_ids,
    )


# =============================================================================
# MASKED-FRAME RECONSTRUCTION DECODER
# =============================================================================

def restore_masked_features(
    unmasked: Tensor,
    mask_info: MaskInfo,
    *,
    noise_std: float,
) -> Tensor:
    """Append mask tokens and restore the original temporal order."""

    # Mathematics: M = T - T_keep mask vectors complete the shuffled sequence
    # [x_keep; z_mask] ∈ R^{B×T×D}; gather by the inverse permutation restores t.
    # Interpretation: the decoder receives placeholders at hidden positions
    # while retaining contextual student values at observed positions.
    masked_count = mask_info.ids_restore.shape[1] - unmasked.shape[1]
    mask_tokens = unmasked.new_empty(unmasked.shape[0], masked_count, unmasked.shape[-1])
    if noise_std:
        mask_tokens.normal_(0, noise_std)
    else:
        mask_tokens.zero_()
    shuffled = torch.cat((unmasked, mask_tokens), dim=1)
    return torch.gather(shuffled, dim=1, index=mask_info.ids_restore)


class DecoderBlock(nn.Module):
    """Grouped convolution block operating on ``[batch, channels, frames]``."""

    def __init__(self, input_dim: int, config: DecoderConfig) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            input_dim,
            config.decoder_dim,
            kernel_size=config.decoder_kernel,
            padding=config.decoder_kernel // 2,
            groups=config.decoder_groups,
        )
        self.same_pad = SamePad(config.decoder_kernel)
        # The archived decoder uses ordinary LayerNorm here. CUDA autocast
        # consequently keeps the normalized block output in float32 before
        # the following convolution/projection returns to float16.
        self.norm = nn.LayerNorm(config.decoder_dim, elementwise_affine=False)
        self.activation = nn.GELU()

    def forward(self, value: Tensor) -> Tensor:
        """Apply grouped temporal convolution, normalization, and GELU."""

        # Mathematics: grouped temporal convolution maps [B,D_in,T] to
        # [B,D_dec,T], then LayerNorm acts over D_dec at each (b,t).
        # Interpretation: the decoder spreads neighboring context into mask
        # slots before applying a channelwise nonlinear representation.
        value = self.same_pad(self.conv(value))
        value = self.norm(value.transpose(1, 2)).transpose(1, 2)
        return self.activation(value)


class ConvDecoder(nn.Module):
    """Restore masked positions and reconstruct teacher feature vectors."""

    def __init__(self, config: DecoderConfig, input_dim: int) -> None:
        super().__init__()
        self.config = config
        self.input_dropout = nn.Dropout(config.input_dropout, inplace=True)
        self.blocks = nn.ModuleList([
            DecoderBlock(input_dim if index == 0 else config.decoder_dim, config)
            for index in range(config.decoder_layers)
        ])
        self.proj = nn.Linear(config.decoder_dim, input_dim)

    def prepare_input(
        self,
        unmasked: Tensor,
        mask_info: MaskInfo,
        *,
        noise_std: float,
    ) -> Tensor:
        """Insert mask tokens and apply the decoder's input dropout."""

        return restore_masked_features(
            self.input_dropout(unmasked),
            mask_info,
            noise_std=noise_std,
        )

    def forward(self, value: Tensor) -> Tensor:
        """Decode a full ``[batch, frames, embedding]`` feature sequence."""

        # Mathematics: blocks operate on [B,D,T]; a residual adds x_l only when
        # D_in = D_out, giving x_{l+1}=f_l(x_l)+x_l.
        # Interpretation: the first block may change width, while later blocks
        # refine reconstructions without discarding their prior estimate.
        value = value.transpose(1, 2)
        residual = value
        for block in self.blocks:
            value = block(value)
            if residual.shape == value.shape:
                value = value + residual
            residual = value
        return self.proj(value.transpose(1, 2))


# =============================================================================
# PRETRAINING AND FINE-TUNING OBJECTIVES
# =============================================================================

@dataclass(frozen=True)
class LossResult:
    """Summed loss and the number of frame tokens contributing to it."""

    loss: Tensor
    sample_size: int


class RegressionLoss(nn.Module):
    """Scaled MSE or smooth-L1 loss for masked teacher targets."""

    def __init__(self, beta: float = 0.0, scale: float | None = None) -> None:
        super().__init__()
        self.beta = beta
        self.scale = scale

    def forward(self, prediction: Tensor, target: Tensor) -> LossResult:
        """Return summed MSE or smooth-L1 regression over flattened frames."""

        # Mathematics: flatten all leading axes into N frame tokens while
        # retaining D target coordinates per token.
        # Interpretation: cloned batches and masked positions contribute under
        # the same per-frame objective regardless of their original layout.
        prediction = prediction.reshape(-1, prediction.shape[-1]).float()
        target = target.reshape_as(prediction).float()
        if self.beta == 0:
            elementwise = F.mse_loss(prediction, target, reduction="none")
        else:
            elementwise = F.smooth_l1_loss(prediction, target, reduction="none", beta=self.beta)
        # Mathematics: L = α sum_{n,d} ell(p_nd,y_nd), with default
        # α = D^{-1/2}; sample_size=N, not N×D.
        # Interpretation: feature width does not cause loss magnitude to grow
        # linearly, and training normalizes the summed loss by frame count.
        scale = self.scale if self.scale is not None else 1 / math.sqrt(prediction.shape[-1])
        return LossResult(loss=(elementwise * scale).sum(), sample_size=prediction.shape[0])


class SigmoidFocalLoss(nn.Module):
    """Pure-PyTorch multi-label sigmoid focal loss."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be none, mean, or sum")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Evaluate independent binary focal loss for each frame and label."""

        logits = logits.float()
        targets = targets.float()
        # Mathematics: p=σ(z), p_t = y p + (1-y)(1-p), and
        # FL = BCE(z,y)(1-p_t)^γ with optional class factor α_t.
        # Interpretation: easy frame-label decisions shrink toward zero so rare
        # or misclassified animal events dominate the supervised gradient.
        probability = torch.sigmoid(logits)
        cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        target_probability = probability * targets + (1 - probability) * (1 - targets)
        loss = cross_entropy * (1 - target_probability).pow(self.gamma)
        if self.alpha >= 0:
            alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_teacher_targets(
    layer_outputs: Sequence[Tensor],
    *,
    top_k: int,
    instance_norm_per_layer: bool,
    layer_norm_per_layer: bool,
    layer_norm_final: bool,
) -> Tensor:
    """Normalize and average the EMA teacher's last ``top_k`` layers."""

    if not layer_outputs or top_k <= 0:
        raise ValueError("teacher targets require at least one layer")
    # Mathematics: select h_{L-K+1},...,h_L, normalize each as configured,
    # and define target y = K^{-1} sum_k h_k.
    # Interpretation: targets combine the most semantic teacher layers
    # instead of binding reconstruction to one arbitrary depth.
    selected = list(layer_outputs[-top_k:])
    normalized: list[Tensor] = []
    with torch.no_grad():
        for layer in selected:
            value = layer.float()
            # Mathematics: InstanceNorm standardizes each (batch,channel)
            # sequence over time; LayerNorm standardizes each frame over D.
            # Interpretation: the recipe chooses whether target scale variation
            # across time, features, or the final average should be removed.
            if instance_norm_per_layer:
                value = F.instance_norm(value.transpose(1, 2)).transpose(1, 2)
            if layer_norm_per_layer:
                value = F.layer_norm(value, value.shape[-1:])
            normalized.append(value)
        target = torch.stack(normalized).mean(dim=0)
        if layer_norm_final:
            target = F.layer_norm(target, target.shape[-1:])
    return target


def make_pretraining_targets(
    layer_outputs: Sequence[Tensor],
    *,
    top_k: int,
    instance_norm_per_layer: bool,
    layer_norm_per_layer: bool,
    layer_norm_final: bool,
    use_cls_token: bool,
) -> tuple[Tensor, Tensor | None]:
    """Build frame targets and an optional feature-normalized CLS target."""

    if not use_cls_token:
        return make_teacher_targets(
            layer_outputs,
            top_k=top_k,
            instance_norm_per_layer=instance_norm_per_layer,
            layer_norm_per_layer=layer_norm_per_layer,
            layer_norm_final=layer_norm_final,
        ), None

    # Mathematics: split h_l=[c_l;f_l] before applying frame normalization, so
    # the time axis of every f_l remains the original T acoustic frames.
    # Interpretation: adding CLS cannot alter the established frame targets.
    frame_layers = [layer[:, 1:] for layer in layer_outputs]
    frame_target = make_teacher_targets(
        frame_layers,
        top_k=top_k,
        instance_norm_per_layer=instance_norm_per_layer,
        layer_norm_per_layer=layer_norm_per_layer,
        layer_norm_final=layer_norm_final,
    )
    selected = list(layer_outputs[-top_k:])
    normalized_cls: list[Tensor] = []
    with torch.no_grad():
        for layer in selected:
            cls = layer[:, 0].float()
            # Mathematics: both per-layer target-normalization modes reduce a
            # CLS vector over D because it has no frame axis to instance-normalize.
            # Interpretation: CLS remains finite and feature-normalized under
            # recipes whose frame targets normalize over time.
            if instance_norm_per_layer or layer_norm_per_layer:
                cls = F.layer_norm(cls, cls.shape[-1:])
            normalized_cls.append(cls)
        cls_target = torch.stack(normalized_cls).mean(dim=0)
        if layer_norm_final:
            cls_target = F.layer_norm(cls_target, cls_target.shape[-1:])
    return frame_target, cls_target


def a_weighted_level(
    waveform: Tensor,
    sample_rate: int,
    *,
    window_seconds: float,
    minimum_db: float = -80.0,
) -> Tensor:
    """Estimate overlapping-window A-weighted levels in decibels."""

    # Mathematics: a window of τ seconds contains round(f_s τ) samples and
    # adjacent windows advance by half that size.
    # Interpretation: gain matching follows short-time acoustic loudness rather
    # than a recording-wide amplitude statistic.
    window_length = round(sample_rate * window_seconds)
    if window_length < 2:
        raise ValueError("A-weighting window must contain at least two samples")
    if waveform.shape[-1] < window_length:
        waveform = F.pad(waveform, (0, window_length - waveform.shape[-1]))
    windows = waveform.unfold(-1, window_length, window_length // 2)
    window = torch.hann_window(window_length, device=waveform.device, dtype=waveform.dtype)
    # Mathematics: P[k] = |RFFT(w[n]x[n])_k|² estimates one-sided spectral
    # power after Hann tapering.
    # Interpretation: tapering reduces edge discontinuities before measuring
    # how strongly each frequency contributes to perceived level.
    spectrum = torch.fft.rfft(windows * window)
    power = spectrum.abs().square()

    frequency = torch.linspace(
        0, sample_rate // 2, power.shape[-1], device=waveform.device, dtype=waveform.dtype
    )
    frequency_squared = frequency.square()
    frequency_squared = torch.where(frequency_squared == 0, torch.ones_like(frequency_squared), frequency_squared)
    # Mathematics: this rational response is the IEC-style A-weighting curve
    # expressed in dB over frequency f; weighted power multiplies by 10^(A/10).
    # Interpretation: mixing compensates for perceptual loudness, so a quiet
    # call is not erased by a louder partner at the same nominal ratio.
    weight_db = 2.0 + 20.0 * (
        2 * math.log10(12194)
        + 2 * torch.log10(frequency_squared)
        - torch.log10(frequency_squared + 12194**2)
        - torch.log10(frequency_squared + 20.6**2)
        - 0.5 * torch.log10(frequency_squared + 107.7**2)
        - 0.5 * torch.log10(frequency_squared + 737.9**2)
    )
    weight_db = weight_db.clamp_min(minimum_db)
    weighted_power = power * torch.pow(10.0, weight_db / 10)
    floor = torch.tensor(10 ** (minimum_db / 10), device=waveform.device, dtype=waveform.dtype)
    return 10 * torch.log10(weighted_power.sum(dim=-1).clamp_min(floor))


@dataclass(frozen=True)
class MixResult:
    """Mixed waveforms and the random choices needed to mix labels equally."""

    waveforms: Tensor
    ratios: Tensor
    permutation: Tensor
    applied: Tensor


@torch.no_grad()
def mix_waveforms(
    waveform: Tensor,
    *,
    strength: float,
    probability: float,
    same_ratio: bool,
    gain_mode: str,
    sample_rate: int,
    window_seconds: float,
    ratios: Tensor | None = None,
    permutation: Tensor | None = None,
) -> MixResult:
    """Apply between-class waveform mixing with optional loudness correction."""

    batch = waveform.shape[0]
    device = waveform.device
    # Mathematics: applied_b ~ Bernoulli(probability) chooses the subset that
    # participates in mixing.
    # Interpretation: the original recording remains untouched for samples
    # whose augmentation coin flip is false.
    applied = torch.ones(batch, dtype=torch.bool, device=device)
    if probability < 1:
        applied = torch.empty(batch, device=device).bernoulli_(probability).bool()
    if ratios is None:
        ratio_count = 1 if same_ratio else int(applied.sum().item())
        # Mathematics: r ~ Uniform(max(10^-6,strength),1); one shared r is used
        # when same_ratio is true, otherwise each applied sample draws its own.
        # Interpretation: strength controls how much of the primary recording
        # must survive in every mixture.
        ratios = torch.empty(ratio_count, device=device, dtype=waveform.dtype).uniform_(max(1e-6, strength), 1)
    else:
        ratios = ratios.to(device=device, dtype=waveform.dtype)
    if permutation is None:
        permutation = torch.randperm(batch, device=device)
    else:
        permutation = permutation.to(device)
    first = waveform[applied]
    second = waveform[permutation][applied]
    ratio = ratios

    if gain_mode == "none":
        coefficient = ratio
    else:
        if gain_mode == "naive_rms":
            levels = waveform.square().mean(dim=-1).sqrt()
        elif gain_mode == "A_weighting":
            levels = a_weighted_level(
                waveform, sample_rate, window_seconds=window_seconds
            ).max(dim=-1).values
        else:
            raise ValueError(f"unknown gain mode: {gain_mode}")
        first_level = levels[applied]
        second_level = levels[permutation][applied]
        # Mathematics: for levels L1,L2 in dB, amplitude ratio is
        # 10^((L1-L2)/20), yielding c = 1/[1 + ratio_level(1-r)/r].
        # Interpretation: c realizes the requested perceptual mixture ratio
        # even when the paired recordings have different loudness.
        coefficient = 1 / (
            1 + torch.pow(10.0, (first_level - second_level) / 20) * (1 - ratio) / ratio
        )
    coefficient = coefficient.unsqueeze(-1)
    mixed = coefficient * first + (1 - coefficient) * second
    # Mathematics: divide c x1 + (1-c)x2 by sqrt(c²+(1-c)²), the RMS gain for
    # uncorrelated unit-variance signals.
    # Interpretation: mixing does not systematically reduce overall waveform
    # energy when coefficients approach one half.
    mixed = mixed / torch.sqrt(coefficient.square() + (1 - coefficient).square())
    result = waveform.clone()
    result[applied] = mixed
    return MixResult(result, ratios, permutation, applied)


# =============================================================================
# EXPONENTIAL-MOVING-AVERAGE TEACHER
# =============================================================================

def ema_decay_at_step(start: float, end: float, step: int, anneal_steps: int) -> float:
    """Linearly anneal the teacher decay and clamp it at the endpoint."""

    if anneal_steps <= 0 or step >= anneal_steps:
        return end
    # Mathematics: d_u = d_0 + (d_end-d_0)u/U for u<U, then d_end.
    # Interpretation: the teacher follows the rapidly changing early student
    # and becomes more stable as optimization progresses.
    return start + (end - start) * step / anneal_steps


class EMATeacher(nn.Module):
    """Non-trainable float32 copy of a student updated by parameter EMA."""

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.model = copy.deepcopy(student).float().eval()
        self.model.requires_grad_(False)
        self.decay = 0.0

    @torch.no_grad()
    def update(self, student: nn.Module, decay: float) -> None:
        """Move teacher parameters toward the student and copy all buffers."""

        student_parameters = dict(student.named_parameters())
        for name, teacher_parameter in self.model.named_parameters():
            source = student_parameters[name].detach().float()
            # Mathematics: θ_teacher <- d θ_teacher + (1-d) θ_student.
            # Interpretation: the target network forms a temporal ensemble of
            # student checkpoints without storing their full history.
            teacher_parameter.mul_(decay).add_(source, alpha=1.0 - decay)
        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in self.model.named_buffers():
            if name in student_buffers:
                teacher_buffer.copy_(student_buffers[name].detach().to(teacher_buffer))
        self.decay = float(decay)

    def train(self, mode: bool = True) -> "EMATeacher":
        """Keep the wrapper and contained teacher in evaluation mode.

        ``nn.Module.train`` normally propagates its mode to children. The EMA
        teacher must never enable dropout or update normalization statistics,
        even when its parent pretraining model enters training mode.
        """

        super().train(False)
        self.model.eval()
        return self


# =============================================================================
# SHARED CONVOLUTIONAL AND TRANSFORMER AUDIO ENCODER
# =============================================================================

# Tensor convention in this module:
#   waveform: [batch, samples]
#   convolution features: [batch, channels, frames]
#   transformer features: [batch, frames, embedding]


# ALiBi selection and transformer blocks

def _select_and_scale_alibi(
    bias: Tensor,
    scale: Tensor,
    keep: Tensor | None,
    num_heads: int,
) -> Tensor:
    """Select unmasked ALiBi rows/columns before materializing its scale."""

    if keep is not None:
        time = bias.shape[-1]
        row_index = keep[:, None, :, None].expand(-1, num_heads, -1, time)
        # Mathematics: B_keep = B[I_keep,I_keep] gathers both query and key
        # coordinates using the same temporal index set.
        # Interpretation: after masked tokens leave the student sequence, ALiBi
        # still measures their original frame distances among retained tokens.
        bias = torch.gather(bias, -2, row_index)
        column_index = keep[:, None, None, :].expand(
            -1, num_heads, bias.shape[-2], -1
        )
        bias = torch.gather(bias, -1, column_index)
    # Mathematics: B'_{l,h,i,j} = max(s_{l,h},0) B_{h,i,j}.
    # Interpretation: each layer or head can weaken or strengthen distance
    # preference without reversing it into a preference for distant frames.
    return bias * scale.clamp_min(0).squeeze(0).to(bias)


def _prepend_cls_alibi(frame_bias: Tensor) -> Tensor:
    """Add a zero-bias CLS row and column around frame-only ALiBi values."""

    # Mathematics: B_cls has B_cls[...,1:,1:]=B_frames and zero entries when
    # either attention coordinate is the special token at index zero.
    # Interpretation: CLS attends globally without a temporal-distance prior,
    # while frame pairs retain their exact original ALiBi values.
    shape = (
        *frame_bias.shape[:-2],
        frame_bias.shape[-2] + 1,
        frame_bias.shape[-1] + 1,
    )
    bias = frame_bias.new_zeros(shape)
    bias[..., 1:, 1:] = frame_bias
    return bias


class MLP(nn.Module):
    """Two-layer GELU feed-forward network inside each transformer block."""

    def __init__(self, dimension: int, hidden_dimension: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden_dimension)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dimension, dimension)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        """Transform ``[..., embedding]`` values through the feed-forward path."""

        # Mathematics: FFN(x)=Drop_2(W_2 Drop_1(GELU(W_1x+b_1))+b_2).
        # Interpretation: attention exchanges information across time, while
        # this path transforms each frame independently in a wider space.
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(value)))))


class TransformerBlock(nn.Module):
    """Animal2Vec AltBlock, including its legacy pre-norm residual order."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        post_mlp_dropout: float = 0.1,
        drop_path: float = 0.0,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        layer_norm_first: bool = False,
        ffn_targets: bool = True,
        position_encoding: str = "none",
        rope_theta: float = 10_000.0,
        attention_backend: str = "manual",
    ) -> None:
        super().__init__()
        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets
        self.norm1 = nn.LayerNorm(dimension, eps=norm_eps, elementwise_affine=norm_affine)
        self.attn = MultiheadAttention(
            dimension,
            num_heads,
            qkv_bias=True,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            position_encoding=position_encoding,
            rope_theta=rope_theta,
            attention_backend=attention_backend,
        )
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.norm2 = nn.LayerNorm(dimension, eps=norm_eps, elementwise_affine=norm_affine)
        self.mlp = MLP(dimension, int(dimension * mlp_ratio), activation_dropout)
        self.post_mlp_dropout = nn.Dropout(post_mlp_dropout)

    def forward(
        self,
        value: Tensor,
        padding_mask: Tensor | None = None,
        alibi: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply attention and feed-forward residual paths.

        Returns the block output plus the representation used as a potential
        EMA teacher target. The target location depends on
        ``ffn_targets`` and the archived residual order.
        """

        if self.layer_norm_first:
            # Mathematics: this branch follows the archived pre-norm ordering;
            # attention receives Norm1(x), then the MLP receives Norm2 of the
            # attention residual. The assignment sequence is kept verbatim.
            # Interpretation: target extraction depends on the intermediate
            # chosen by the official block, so a textbook rewrite would alter
            # teacher regression even if final tensor shapes matched.
            value = value + self.drop_path(
                self.attn(self.norm1(value), padding_mask, alibi, position_ids)
            )
            residual = value = self.mlp(self.norm2(value))
            target = value
            value = residual + self.drop_path(self.post_mlp_dropout(value))
            if not self.ffn_targets:
                target = value
        else:
            # Mathematics: post-norm mode forms x'=Norm1(x+Attn(x)) and
            # x_out=Norm2(x'+Drop(MLP(x'))).
            # Interpretation: this is the paper baseline's alternate residual
            # convention selected by layer_norm_first.
            value = value + self.drop_path(
                self.attn(value, padding_mask, alibi, position_ids)
            )
            residual = value = self.norm1(value)
            value = self.mlp(value)
            target = value
            value = self.norm2(residual + self.drop_path(self.post_mlp_dropout(value)))
            if not self.ffn_targets:
                target = value
        return value, target


class TransformerStack(nn.Module):
    """Ordered transformer blocks with Fairseq-compatible layer/drop behavior.

    The returned list contains the representation selected for teacher targets
    at every block, which is not always identical to the block's final output.
    """

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        depth: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        post_mlp_dropout: float = 0.1,
        drop_path_rates: Sequence[float] | None = None,
        layerdrop: float = 0.0,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        layer_norm_first: bool = False,
        ffn_targets: bool = True,
        input_dropout: float = 0.0,
        norm_before: bool = False,
        norm_after: bool | None = None,
        checkpoint_activations: bool = False,
        position_encoding: str = "none",
        rope_theta: float = 10_000.0,
        attention_backend: str = "manual",
    ) -> None:
        super().__init__()
        self.checkpoint_activations = checkpoint_activations
        rates = list(drop_path_rates or [0.0] * depth)
        if len(rates) != depth:
            raise ValueError("drop_path_rates must match transformer depth")
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dimension,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                post_mlp_dropout=post_mlp_dropout,
                drop_path=rates[index],
                norm_eps=norm_eps,
                norm_affine=norm_affine,
                layer_norm_first=layer_norm_first,
                ffn_targets=ffn_targets,
                position_encoding=position_encoding,
                rope_theta=rope_theta,
                attention_backend=attention_backend,
            )
            for index in range(depth)
        ])
        self.layerdrop = layerdrop
        self.dropout = nn.Dropout(input_dropout, inplace=True)
        self.norm_before = norm_before
        self.norm_after = layer_norm_first if norm_after is None else norm_after
        self.norm = (
            nn.LayerNorm(dimension, eps=norm_eps, elementwise_affine=norm_affine)
            if self.norm_before or self.norm_after else None
        )

    def forward(
        self,
        value: Tensor,
        padding_mask: Tensor | None = None,
        alibi: Tensor | None = None,
        alibi_scale: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Run all non-dropped blocks and collect their teacher-target tensors."""

        if self.norm is not None and self.norm_before:
            value = self.norm(value)
        value = self.dropout(value)
        layer_outputs: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            # Mathematics: block l is skipped when U_l <= p_layerdrop for
            # independent U_l~Uniform[0,1) draws from NumPy's global RNG.
            # Interpretation: using NumPy here is a checkpoint-visible detail;
            # restoring only PyTorch randomness would not reproduce a run.
            if self.training and self.layerdrop and np.random.random() <= self.layerdrop:
                continue
            layer_bias = alibi
            if layer_bias is not None and alibi_scale is not None:
                scale = alibi_scale[index] if alibi_scale.shape[0] > 1 else alibi_scale.squeeze(0)
                # Mathematics: transformer layer l multiplies each ALiBi head
                # by its learned nonnegative scale before adding it to logits.
                # Interpretation: deeper layers may operate at a different
                # temporal range while sharing the same base distance matrix.
                layer_bias = layer_bias * scale.to(layer_bias)
            should_checkpoint = (
                self.checkpoint_activations
                and self.training
                and torch.is_grad_enabled()
            )
            if should_checkpoint:
                value, target = activation_checkpoint(
                    block,
                    value,
                    padding_mask,
                    layer_bias,
                    position_ids,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                value, target = block(value, padding_mask, layer_bias, position_ids)
            layer_outputs.append(target)
        if self.norm is not None and self.norm_after:
            value = self.norm(value)
        return value, layer_outputs


# Local convolutional and positional frontends

class ConvFeatureBlock(nn.Module):
    """One waveform feature block: convolution, normalization, activation."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel: int,
        stride: int,
        *,
        first: bool,
        sample_rate: int,
        sinc_input: bool,
        apply_window_to_root: bool,
        use_pswish: bool,
        sinc_norm: str,
    ) -> None:
        super().__init__()
        # Mathematics: only layer index zero may use analytic Sinc kernels; all
        # later layers use unconstrained learned Conv1d kernels.
        # Interpretation: the baseline injects frequency structure at the raw
        # waveform boundary and lets deeper features learn freely.
        is_sinc = first and sinc_input
        if is_sinc:
            self.conv = SincConv1d(
                output_channels,
                kernel,
                in_channels=input_channels,
                stride=stride,
                sample_rate=sample_rate,
                learnable_filters=apply_window_to_root,
                apply_window_to_root=apply_window_to_root,
                return_abs=sinc_norm in {"pcen", "instance"},
            )
        else:
            padding: str | int = "same" if stride == 1 else math.ceil(stride / 2)
            self.conv = nn.Conv1d(input_channels, output_channels, kernel, stride=stride, padding=padding, bias=False)
            nn.init.kaiming_normal_(self.conv.weight)
        self.dropout = nn.Dropout(0.0)
        self.norm = Fp32LayerNorm(output_channels)
        self.activation = PSwish(output_channels) if is_sinc and use_pswish else nn.GELU()

    def forward(self, value: Tensor) -> Tensor:
        """Convert one channel-first feature tensor through this local block."""

        # Mathematics: Conv maps [B,C_in,S] to [B,C_out,T], LayerNorm acts over
        # C_out after transposition, and activation is pointwise.
        # Interpretation: each local block downsamples or refines time, then
        # standardizes its channel vector before the next block.
        value = self.dropout(self.conv(value))
        value = self.norm(value.transpose(1, 2)).transpose(1, 2)
        return self.activation(value)


class ConvFeatureEncoder(nn.Module):
    """Convert waveforms into the local frame features used by Animal2Vec."""

    def __init__(
        self,
        layers: Sequence[ConvLayerSpec],
        *,
        sample_rate: int,
        sinc_input: bool = True,
        apply_window_to_root: bool = False,
        use_pswish: bool = True,
        sinc_norm: str = "layer_norm",
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("at least one convolution layer is required")
        input_channels = 1
        blocks = []
        for index, (output_channels, kernel, stride) in enumerate(layers):
            blocks.append(ConvFeatureBlock(
                input_channels,
                output_channels,
                kernel,
                stride,
                first=index == 0,
                sample_rate=sample_rate,
                sinc_input=sinc_input,
                apply_window_to_root=apply_window_to_root,
                use_pswish=use_pswish,
                sinc_norm=sinc_norm,
            ))
            input_channels = output_channels
        self.conv_layers = nn.ModuleList(blocks)

    def forward(self, waveform: Tensor) -> Tensor:
        """Encode ``[batch, samples]`` into ``[batch, channels, frames]``."""

        # Mathematics: a mono [B,S] waveform becomes [B,1,S], then composition
        # f_L∘...∘f_1 yields [B,C_L,T].
        # Interpretation: all convolution blocks use PyTorch's channel-first
        # convention while public model inputs remain simple waveform batches.
        value = waveform.unsqueeze(1) if waveform.ndim == 2 else waveform
        for block in self.conv_layers:
            value = block(value)
        return value


class PositionalConvEncoder(nn.Module):
    """Depth-wise stack that adds local relative-position information."""

    def __init__(self, dimension: int, depth: int, width: int, groups: int) -> None:
        super().__init__()
        # Mathematics: each of `depth` convolutions uses K=max(3,floor(width/depth));
        # their stacked receptive field approximates the requested width.
        # Interpretation: several shallow nonlinear position filters replace
        # one very wide convolution while retaining local temporal context.
        kernel = max(3, width // depth)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "conv": nn.Conv1d(dimension, dimension, kernel, padding=kernel // 2, groups=groups),
                "same_pad": SamePad(kernel),
                "norm": Fp32LayerNorm(dimension, elementwise_affine=False),
                "activation": nn.GELU(),
            }))

    def forward(self, value: Tensor) -> Tensor:
        """Add local temporal context while preserving feature shape."""

        # Mathematics: every block applies grouped Conv1d over time, trims any
        # extra even-kernel frame, then normalizes the D-vector at each frame.
        # Interpretation: the encoder adds a learned local position signal
        # without absolute position embeddings or a changed tensor length.
        value = value.transpose(1, 2)
        for block in self.blocks:
            value = block["same_pad"](block["conv"](value))
            value = block["norm"](value.transpose(1, 2)).transpose(1, 2)
            value = block["activation"](value)
        return value.transpose(1, 2)


# Complete shared audio encoder

@dataclass(frozen=True)
class EncoderOutput:
    """Complete encoder result.

    ``local_features`` stays frame-only. ``x`` and ``padding_mask`` use the
    contextual token axis, which is one item longer when CLS is enabled.
    """

    x: Tensor
    padding_mask: Tensor | None
    local_features: Tensor
    prenet_layers: tuple[Tensor, ...]
    layer_outputs: tuple[Tensor, ...]
    alibi: Tensor | None


@dataclass(frozen=True)
class ContextOutput:
    """Transformer-only result, optionally including CLS at token index zero."""

    x: Tensor
    padding_mask: Tensor | None
    prenet_layers: tuple[Tensor, ...]
    layer_outputs: tuple[Tensor, ...]
    alibi: Tensor | None


class AudioEncoder(nn.Module):
    """Shared Animal2Vec waveform encoder.

    Data follows four visible phases:

    1. ``local_encoder`` turns samples into convolution frames.
    2. ``project_features`` maps those frames to the transformer dimension.
    3. ``positional_encoder`` adds local positional context.
    4. Optional CLS is prepended before ``prenet`` and ``transformer``.

    Pretraining can enter at phase 3 through :meth:`encode_projected`, allowing
    the student to remove masked tokens while the EMA teacher sees all tokens.
    Attribute names intentionally match the converted checkpoint schema.
    """

    def __init__(
        self,
        layers: Sequence[ConvLayerSpec],
        sample_rate: int,
        dimension: int,
        num_heads: int,
        depth: int,
        *,
        prenet_depth: int = 0,
        conv_pos_depth: int = 5,
        conv_pos_width: int = 95,
        conv_pos_groups: int = 16,
        sinc_input: bool = True,
        apply_window_to_root: bool = False,
        use_pswish: bool = True,
        sinc_norm: str = "layer_norm",
        mlp_ratio: float = 4.0,
        encoder_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        post_mlp_dropout: float = 0.1,
        layerdrop: float = 0.0,
        prenet_layerdrop: float = 0.0,
        prenet_dropout: float = 0.0,
        dropout_input: float = 0.0,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        layer_norm_first: bool = False,
        end_of_block_targets: bool = False,
        start_drop_path_rate: float = 0.0,
        end_drop_path_rate: float = 0.0,
        use_alibi: bool = True,
        learned_alibi_scale: bool = True,
        learned_alibi_scale_per_head: bool = True,
        checkpoint_activations: bool = False,
        position_encoding: str | None = None,
        rope_theta: float = 10_000.0,
        attention_backend: str = "manual",
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.layers = tuple(layers)
        self.num_heads = num_heads
        self.position_encoding = (
            "alibi" if use_alibi else "none"
        ) if position_encoding is None else position_encoding
        self.use_alibi = self.position_encoding == "alibi"
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.local_encoder = ConvFeatureEncoder(
            layers,
            sample_rate=sample_rate,
            sinc_input=sinc_input,
            apply_window_to_root=apply_window_to_root,
            use_pswish=use_pswish,
            sinc_norm=sinc_norm,
        )
        feature_dimension = layers[-1][0]
        self.project_norm = Fp32LayerNorm(feature_dimension)
        self.project_features = nn.Linear(feature_dimension, dimension)
        self.positional_encoder = PositionalConvEncoder(
            dimension, conv_pos_depth, conv_pos_width, conv_pos_groups
        )
        common = dict(
            dimension=dimension,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            post_mlp_dropout=post_mlp_dropout,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            layer_norm_first=layer_norm_first,
            ffn_targets=not end_of_block_targets,
            position_encoding=self.position_encoding,
            rope_theta=rope_theta,
            attention_backend=attention_backend,
        )
        self.prenet = TransformerStack(
            depth=prenet_depth,
            dropout=encoder_dropout,
            layerdrop=prenet_layerdrop,
            input_dropout=prenet_dropout,
            norm_before=not layer_norm_first,
            norm_after=False,
            checkpoint_activations=checkpoint_activations,
            **common,
        )
        # Mathematics: drop-path probability p_l interpolates linearly from
        # p_start to p_end over transformer depth l=0,...,L-1.
        # Interpretation: deeper residual paths can receive stronger stochastic
        # regularization while preserving the official layerwise schedule.
        rates = torch.linspace(start_drop_path_rate, end_drop_path_rate, depth).tolist()
        self.transformer = TransformerStack(
            depth=depth,
            dropout=encoder_dropout,
            layerdrop=layerdrop,
            drop_path_rates=rates,
            input_dropout=dropout_input,
            checkpoint_activations=checkpoint_activations,
            **common,
        )
        scale_heads = num_heads if learned_alibi_scale_per_head else 1
        self.alibi_scale = nn.Parameter(
            torch.ones(1, 1, scale_heads, 1, 1),
            requires_grad=learned_alibi_scale,
        )
        self.apply(init_bert_params)
        # The official AudioEncoder reset runs after the parent model's BERT
        # initialization and restores this projection to nn.Linear defaults.
        self.project_features.reset_parameters()

    @classmethod
    def from_config(cls, config: Animal2VecConfig) -> "AudioEncoder":
        """Build the encoder from the shared task and model configuration."""

        model = config.model
        audio = model.audio
        return cls(
            config.task.conv_feature_layers,
            config.task.sample_rate,
            model.embed_dim,
            model.num_heads,
            model.depth,
            prenet_depth=audio.prenet_depth,
            conv_pos_depth=audio.conv_pos_depth,
            conv_pos_width=audio.conv_pos_width,
            conv_pos_groups=audio.conv_pos_groups,
            sinc_input=audio.sinc_input,
            apply_window_to_root=audio.apply_window_to_root,
            use_pswish=audio.use_pswish,
            sinc_norm=audio.sinc_norm,
            mlp_ratio=model.mlp_ratio,
            encoder_dropout=model.encoder_dropout,
            attention_dropout=model.attention_dropout,
            activation_dropout=model.activation_dropout,
            post_mlp_dropout=model.post_mlp_drop,
            layerdrop=model.layerdrop,
            prenet_layerdrop=audio.prenet_layerdrop,
            prenet_dropout=audio.prenet_dropout,
            dropout_input=model.dropout_input,
            norm_eps=model.norm_eps,
            norm_affine=model.norm_affine,
            layer_norm_first=model.layer_norm_first,
            end_of_block_targets=model.end_of_block_targets,
            start_drop_path_rate=model.start_drop_path_rate,
            end_drop_path_rate=model.end_drop_path_rate,
            use_alibi=audio.use_alibi_encoder,
            learned_alibi_scale=audio.learned_alibi_scale,
            learned_alibi_scale_per_head=audio.learned_alibi_scale_per_head,
            checkpoint_activations=model.checkpoint_activations,
            position_encoding=resolve_position_encoding(model),
            rope_theta=model.rope_theta,
            attention_backend=resolve_attention_backend(model),
            use_cls_token=model.use_cls_token,
        )

    @staticmethod
    def estimated_parameter_count(config: Animal2VecConfig) -> int:
        """Estimate transformer-side parameters without allocating the model."""

        model = config.model
        dimension = model.embed_dim
        hidden = int(dimension * model.mlp_ratio)
        # Mathematics: count QKV, attention output, two FFN affine maps, and
        # four normalization vectors per transformer block.
        # Interpretation: configuration validation can estimate model scale
        # before allocating base or large networks on a constrained host.
        per_block = (
            3 * dimension * dimension + 3 * dimension
            + dimension * dimension + dimension
            + dimension * hidden + hidden
            + hidden * dimension + dimension
            + 4 * dimension
        )
        transformer = per_block * (model.depth + model.audio.prenet_depth)
        projection = config.task.conv_feature_layers[-1][0] * dimension + dimension
        positional = model.audio.conv_pos_depth * (
            dimension * dimension * max(3, model.audio.conv_pos_width // model.audio.conv_pos_depth)
            // model.audio.conv_pos_groups + dimension
        )
        prenet_norm = 2 * dimension if model.norm_affine else 0
        return transformer + projection + positional + prenet_norm

    def convert_padding_mask(self, value: Tensor, padding_mask: Tensor | None) -> Tensor | None:
        """Map a sample-level padding mask to convolution-frame positions."""

        if padding_mask is None:
            return None
        # Mathematics: L_b = sum_s 1[not padding_{b,s}], followed by the exact
        # convolution length recurrence T_b=ConvLength(L_b).
        # Interpretation: every output frame at index t>=T_b represents padding
        # and must be excluded from attention and metrics.
        input_lengths = (~padding_mask.bool()).sum(dim=-1)
        output_lengths = conv_output_length(input_lengths, self.layers)
        positions = torch.arange(value.shape[1], device=value.device)
        converted = positions.unsqueeze(0) >= output_lengths.unsqueeze(1)
        return converted if converted.any() else None

    def project_waveform(
        self,
        waveform: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the local convolution frontend and embedding projection."""

        # Mathematics: local encoder produces [B,C,T], transpose gives [B,T,C],
        # LayerNorm standardizes C, and the affine projection maps C -> D.
        # Interpretation: acoustic convolution channels become transformer
        # embeddings on the same temporal frame grid.
        local = self.local_encoder(waveform)
        projected = self.project_features(self.project_norm(local.transpose(1, 2)))
        return projected, self.convert_padding_mask(projected, padding_mask)

    def encode_projected(
        self,
        projected: Tensor,
        padding_mask: Tensor | None = None,
        mask_info: MaskInfo | None = None,
    ) -> ContextOutput:
        """Contextualize projected frames, optionally removing masked tokens."""

        positional_input = projected
        if mask_info is not None:
            # The archived encoder applies its zero mask before the relative
            # positional convolution, then gathers the original unmasked
            # features. Neighboring positional features therefore depend on
            # the zeroed mask pattern even after masked tokens are removed.
            # Mathematics: set x_{btd}=0 for t∈M before applying positional
            # convolution P; later gather P(x) only at t∉M.
            # Interpretation: masked neighborhoods affect local positional
            # context exactly as in the archived student path, even though the
            # transformer never receives the masked token itself.
            positional_input = projected.masked_fill(
                mask_info.mask.bool().unsqueeze(-1),
                0,
            )
        positions = self.positional_encoder(positional_input)
        batch, time, _ = projected.shape
        position_ids = None
        if self.position_encoding == "rope":
            position_ids = torch.arange(time, device=projected.device).unsqueeze(0)
            if self.use_cls_token:
                position_ids = position_ids + 1
            position_ids = position_ids.expand(batch, -1)
        bias = None
        if self.use_alibi:
            # Mathematics: expand B∈R^{H×T×T} along batch to B batches; expand
            # creates a shared view rather than independent learned tensors.
            # Interpretation: recordings use identical relative-distance priors
            # while later scaling remains trainable.
            bias = alibi_bias(self.num_heads, time, device=projected.device).unsqueeze(0)
            bias = bias.expand(batch, -1, -1, -1)

        contextual_padding = padding_mask
        keep = None
        if mask_info is None:
            value = projected + positions
        else:
            if mask_info.x_unmasked is None:
                raise ValueError("mask_info must include gathered features")
            keep = mask_info.ids_keep[..., 0]
            if position_ids is not None:
                position_ids = torch.gather(position_ids, 1, keep)
            # Mathematics: x_keep = gather(x,I_keep) and p_keep =
            # gather(P(mask_zero(x)),I_keep), so transformer input is x_keep+p_keep.
            # Interpretation: the student processes only observed frames and
            # saves attention computation proportional to the mask ratio.
            gathered_positions = torch.gather(
                positions, 1, mask_info.ids_keep
            )
            value = mask_info.x_unmasked + gathered_positions
            if padding_mask is not None:
                contextual_padding = torch.gather(padding_mask, 1, keep)
                if not contextual_padding.any():
                    contextual_padding = None
        if bias is not None:
            bias = _select_and_scale_alibi(
                bias,
                self.alibi_scale,
                keep,
                self.num_heads,
            )
        if self.use_cls_token:
            # Mathematics: c∈R^D is broadcast to [B,1,D] and concatenated
            # after positional convolution and any frame gather.
            # Interpretation: CLS is contextual but never participates in the
            # frame mask, restore permutation, or convolutional positions.
            cls = self.cls_token.expand(batch, -1, -1)
            value = torch.cat((cls, value), dim=1)
            if contextual_padding is not None:
                contextual_padding = torch.cat((
                    torch.zeros(
                        batch,
                        1,
                        dtype=contextual_padding.dtype,
                        device=contextual_padding.device,
                    ),
                    contextual_padding,
                ), dim=1)
            if position_ids is not None:
                position_ids = torch.cat((
                    torch.zeros(
                        batch,
                        1,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    ),
                    position_ids,
                ), dim=1)
            if bias is not None:
                bias = _prepend_cls_alibi(bias)

        # Mathematics: contextual representation is Transformer(
        # Prenet(value; padding,bias); padding,bias), retaining every layer target.
        # Interpretation: both optional context blocks and the main stack remain
        # visible to teacher-target selection and fine-tuning.
        value, prenet_layers = self.prenet(
            value,
            contextual_padding,
            bias,
            position_ids=position_ids,
        )
        value, layer_outputs = self.transformer(
            value,
            contextual_padding,
            bias,
            position_ids=position_ids,
        )
        return ContextOutput(
            x=value,
            padding_mask=contextual_padding,
            prenet_layers=tuple(prenet_layers),
            layer_outputs=tuple(layer_outputs),
            alibi=bias,
        )

    def forward(self, waveform: Tensor, padding_mask: Tensor | None = None) -> EncoderOutput:
        """Encode a waveform batch and retain intermediate layer outputs."""

        projected, converted_padding = self.project_waveform(waveform, padding_mask)
        context = self.encode_projected(projected, converted_padding)
        return EncoderOutput(
            x=context.x,
            padding_mask=context.padding_mask,
            local_features=projected,
            prenet_layers=context.prenet_layers,
            layer_outputs=context.layer_outputs,
            alibi=context.alibi,
        )


# =============================================================================
# SELF-SUPERVISED PRETRAINING MODEL
# =============================================================================

@dataclass(frozen=True)
class PretrainingOutput:
    """Loss and regression tensors returned by one pretraining step."""

    loss: Tensor
    sample_size: int
    predictions: Tensor
    targets: Tensor
    mask: Tensor
    ema_decay: float


class Animal2VecPretrainingModel(nn.Module):
    """Mean-teacher masked-prediction model used for Animal2Vec pretraining.

    The student sees only unmasked frames plus optional CLS. A convolutional
    decoder restores the frame axis, and an optional linear head directly
    regresses CLS against normalized EMA-teacher representations.
    """

    def __init__(self, config: Animal2VecConfig) -> None:
        super().__init__()
        if config.stage != "pretrain":
            raise ValueError("pretraining model requires a pretraining config")
        self.config = config
        self.student = AudioEncoder.from_config(config)
        self.teacher = EMATeacher(self.student)
        self.decoder = ConvDecoder(config.model.audio.decoder, config.model.embed_dim)
        self.regression = RegressionLoss(config.model.loss_beta, config.model.loss_scale)
        if config.model.use_cls_token:
            self.cls_predictor = nn.Linear(config.model.embed_dim, config.model.embed_dim)

    @classmethod
    def from_config(cls, config: Animal2VecConfig) -> "Animal2VecPretrainingModel":
        """Build a student, its initial teacher copy, and the decoder."""

        return cls(config)

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_ids: Tensor,
        update: int,
        padding_mask: Tensor | None = None,
    ) -> PretrainingOutput:
        """Compute one masked-prediction batch.

        ``waveform`` is ``[batch, samples]`` and ``sample_ids`` contains stable
        manifest indices. The ids, update number, and configured seed make mask
        placement reproducible across workers and distributed ranks.
        """

        cfg = self.config
        # Mathematics: optional waveform mixup replaces selected x_b with a
        # normalized convex combination before either network observes it.
        # Interpretation: student and teacher receive the same augmented audio,
        # so regression targets remain paired with their inputs.
        if self.training and cfg.model.source_mixup >= 0 and cfg.model.mixup_prob > 0:
            waveform = mix_waveforms(
                waveform,
                strength=cfg.model.source_mixup,
                probability=cfg.model.mixup_prob,
                same_ratio=cfg.model.same_mixup,
                gain_mode=cfg.model.gain_mode,
                sample_rate=cfg.task.sample_rate,
                window_seconds=cfg.model.mixing_window_length,
            ).waveforms

        projected, feature_padding = self.student.project_waveform(waveform, padding_mask)
        batch, length, _ = projected.shape
        clones = cfg.model.clone_batch
        # Mathematics: repeat each projected recording K times to obtain
        # [B K,T,D], then sample an independent deterministic mask for each copy.
        # Interpretation: clone_batch extracts several prediction tasks from one
        # expensive convolutional frontend pass.
        cloned = projected.repeat_interleave(clones, dim=0)
        cloned_padding = feature_padding.repeat_interleave(clones, dim=0) if feature_padding is not None else None
        mask = masks_for_cloned_batch(
            batch_size=batch,
            length=length,
            clone_count=clones,
            mask_prob=cfg.model.audio.mask_prob,
            mask_length=cfg.model.audio.mask_length,
            global_seed=cfg.common.seed,
            update=update,
            sample_ids=sample_ids,
            padding_mask=feature_padding,
            mask_dropout=cfg.model.audio.mask_dropout,
        ).to(projected.device)
        mask_info = make_mask_info(cloned, mask)
        # Mathematics: student maps only T-M retained tokens to contextual
        # features; decoder inserts M placeholders and returns T predictions.
        # Interpretation: reconstruction requires contextual inference from
        # audible surroundings rather than copying hidden frame features.
        student_context = self.student.encode_projected(cloned, cloned_padding, mask_info)
        decoder_input = (
            student_context.x[:, 1:]
            if cfg.model.use_cls_token else student_context.x
        )
        restored = self.decoder.prepare_input(
            decoder_input,
            mask_info,
            noise_std=cfg.model.audio.mask_noise_std,
        )
        decoded = self.decoder(restored)

        with torch.no_grad():
            self.teacher.model.eval()
            # Mathematics: teacher f_{\bar θ} receives all T detached projected
            # frames, and y is the normalized mean of its final K layer outputs.
            # Interpretation: the slowly moving teacher supplies stable targets
            # without gradient flow or masked information loss.
            teacher_context = self.teacher.model.encode_projected(projected.detach(), feature_padding)
            targets, cls_targets = make_pretraining_targets(
                teacher_context.layer_outputs,
                top_k=cfg.model.average_top_k_layers,
                instance_norm_per_layer=cfg.model.instance_norm_target_layer,
                layer_norm_per_layer=cfg.model.layer_norm_target_layer,
                layer_norm_final=cfg.model.layer_norm_targets,
                use_cls_token=cfg.model.use_cls_token,
            )
            targets = targets.repeat_interleave(clones, dim=0)
            if cls_targets is not None:
                cls_targets = cls_targets.repeat_interleave(clones, dim=0)

        # Mathematics: objective inputs are p = decoded[M] and y = targets[M];
        # all unmasked decoder positions are excluded from the regression sum.
        # Interpretation: the model learns to predict hidden content and does
        # not receive credit for reconstructing frames it observed.
        selected_predictions = decoded[mask.bool()]
        selected_targets = targets[mask.bool()]
        loss = self.regression(selected_predictions, selected_targets)
        if cfg.model.use_cls_token:
            assert cls_targets is not None
            cls_predictions = self.cls_predictor(student_context.x[:, 0])
            cls_loss = self.regression(cls_predictions, cls_targets)
            selected_predictions = torch.cat((selected_predictions, cls_predictions), dim=0)
            selected_targets = torch.cat((selected_targets, cls_targets), dim=0)
            loss = LossResult(
                loss=loss.loss + cfg.model.cls_loss_weight * cls_loss.loss,
                sample_size=loss.sample_size + cls_loss.sample_size,
            )
        return PretrainingOutput(
            loss=loss.loss,
            sample_size=loss.sample_size,
            predictions=selected_predictions,
            targets=selected_targets,
            mask=mask,
            ema_decay=self.teacher.decay,
        )

    @torch.no_grad()
    def update_teacher(self, update: int) -> float:
        """Advance the EMA teacher after a successful optimizer update."""

        cfg = self.config.model
        decay = ema_decay_at_step(
            cfg.ema_decay,
            cfg.ema_end_decay,
            update,
            cfg.ema_anneal_end_step,
        )
        self.teacher.update(self.student, decay)
        return decay


# =============================================================================
# SUPERVISED EVENT FINE-TUNING MODEL
# =============================================================================

def mix_targets(
    targets: Tensor,
    *,
    ratios: Tensor,
    permutation: Tensor,
    applied: Tensor,
    same_ratio: bool,
) -> Tensor:
    """Apply the waveform mixup ratios to framewise multi-label targets."""

    result = targets.clone()
    ratio = ratios.reshape(-1, *([1] * (targets.ndim - 1)))
    # Mathematics: y_mix = r y_i + (1-r)y_{π(i)}, with r broadcast across time
    # and classes and applied only to samples selected for waveform mixing.
    # Interpretation: soft multi-label targets describe both recordings in the
    # same proportions as their mixed waveform.
    result[applied] = targets[applied] * ratio + targets[permutation][applied] * (1 - ratio)
    return result


def sequence_targets(targets: Tensor, padding_mask: Tensor | None = None) -> Tensor:
    """Reduce frame labels to padding-aware recording-level occurrences."""

    if targets.ndim != 3:
        raise ValueError("sequence targets require [batch, frames, classes] labels")
    if padding_mask is not None:
        if padding_mask.shape != targets.shape[:2]:
            raise ValueError("target padding mask must match the batch and frame axes")
        targets = targets.masked_fill(padding_mask.bool().unsqueeze(-1), 0)
    # Mathematics: y_bc=max_{t not padded} y_btc.
    # Interpretation: a class is present for the recording if it occurs in any
    # real frame; annotations belonging only to batch padding are ignored.
    return targets.amax(dim=1)


@dataclass(frozen=True)
class FineTuningOutput:
    """Frame or sequence logits, loss, and encoder intermediates for one batch."""

    logits: Tensor
    padding_mask: Tensor | None
    layer_outputs: tuple[Tensor, ...]
    targets: Tensor | None
    loss: Tensor | None
    sample_size: int


class Animal2VecFineTuningModel(nn.Module):
    """Framewise or CLS classifier built on a converted pretrained encoder.

    The pretrained recipe remains the authority for encoder architecture. The
    fine-tuning recipe supplies dropout, masking, mixup, freezing, labels, and
    the focal-loss head.
    """

    def __init__(
        self,
        config: Animal2VecConfig,
        *,
        pretrained_config: Animal2VecConfig,
    ) -> None:
        super().__init__()
        if config.stage != "finetune":
            raise ValueError("fine-tuning model requires a fine-tuning config")
        if config.task.sample_rate != pretrained_config.task.sample_rate:
            raise ValueError("fine-tuning and pretraining sample rates differ")
        if config.task.conv_feature_layers != pretrained_config.task.conv_feature_layers:
            raise ValueError("fine-tuning and pretraining convolution specifications differ")
        if config.model.classification_head == "cls" and not pretrained_config.model.use_cls_token:
            raise ValueError(
                "model.classification_head=cls requires a pretrained encoder "
                "built with model.use_cls_token=true"
            )
        self.config = config
        self.pretrained_config = pretrained_config
        fine_model = config.model
        # Mathematics: architecture coordinates come from the pretrained
        # config; fine-tuning replaces only dropout, layerdrop, and drop-path
        # regularization coordinates listed below.
        # Interpretation: converted encoder weights always load into the exact
        # shape that produced them while the supervised recipe controls noise.
        encoder_model = replace(
            pretrained_config.model,
            encoder_dropout=fine_model.dropout,
            attention_dropout=fine_model.attention_dropout,
            activation_dropout=fine_model.activation_dropout,
            post_mlp_drop=fine_model.dropout,
            dropout_input=fine_model.dropout_input,
            layerdrop=fine_model.layerdrop,
            start_drop_path_rate=fine_model.drop_path,
            end_drop_path_rate=fine_model.drop_path,
            checkpoint_activations=fine_model.checkpoint_activations,
            audio=replace(
                pretrained_config.model.audio,
                prenet_layerdrop=fine_model.layerdrop,
                prenet_dropout=fine_model.dropout,
            ),
        )
        encoder_config = replace(pretrained_config, model=encoder_model)
        self.encoder = AudioEncoder.from_config(encoder_config)
        self.final_dropout = nn.Dropout(config.model.final_dropout)
        self.classifier = nn.Linear(pretrained_config.model.embed_dim, len(config.task.unique_labels))
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.focal_loss = SigmoidFocalLoss(
            config.criterion.focal_alpha,
            config.criterion.focal_gamma,
            reduction="sum",
        )

    @classmethod
    def from_config(
        cls,
        config: Animal2VecConfig,
        *,
        pretrained_config: Animal2VecConfig,
        encoder_state: dict[str, Tensor] | None = None,
    ) -> "Animal2VecFineTuningModel":
        """Build the classifier and optionally load a native encoder state."""

        model = cls(config, pretrained_config=pretrained_config)
        if encoder_state is not None:
            model.load_pretrained_encoder(encoder_state)
        return model

    def load_pretrained_encoder(self, state: dict[str, Tensor]) -> None:
        """Load an exact native encoder state after shape and key validation."""

        expected = self.encoder.state_dict()
        for name, tensor in state.items():
            if name in expected and expected[name].shape != tensor.shape:
                raise ValueError(
                    f"incompatible encoder tensor {name}: expected {tuple(expected[name].shape)}, received {tuple(tensor.shape)}"
                )
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        if missing or unexpected:
            raise ValueError(f"encoder state mismatch: missing={missing}, unexpected={unexpected}")
        self.encoder.load_state_dict(state, strict=True)

    def _project(self, waveform: Tensor, padding_mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Extract and project local features with configured gradient scaling."""

        multiplier = self.config.model.feature_grad_mult
        # Mathematics: m<=0 evaluates the local encoder under no_grad; m>0 uses
        # GradMultiply so its backward derivative is multiplied by m.
        # Interpretation: recipes can freeze expensive acoustic filters or let
        # them adapt more slowly than the transformer and classifier.
        if multiplier <= 0:
            with torch.no_grad():
                local = self.encoder.local_encoder(waveform)
        else:
            local = self.encoder.local_encoder(waveform)
            if multiplier != 1:
                local = GradMultiply.apply(local, multiplier)
        projected = self.encoder.project_features(self.encoder.project_norm(local.transpose(1, 2)))
        return projected, self.encoder.convert_padding_mask(projected, padding_mask)

    def _apply_masks(
        self,
        projected: Tensor,
        padding_mask: Tensor | None,
        sample_ids: Tensor,
        update: int,
    ) -> Tensor:
        """Apply deterministic time and channel masks to projected features."""

        cfg = self.config.model
        value = projected.clone()
        if cfg.mask_prob > 0:
            time_mask = compute_mask_indices(
                value.shape[:2],
                padding_mask,
                cfg.mask_prob,
                cfg.mask_length,
                min_masks=1,
                seed=self.config.common.seed,
                epoch=update,
                indices=sample_ids,
            ).to(value.device)
            # Mathematics: selected time vectors receive iid N(0,0.01²) noise.
            # Interpretation: fine-tuning learns robustness to missing acoustic
            # frames using the same deterministic mask geometry as pretraining.
            value[time_mask] = torch.empty_like(value[time_mask]).normal_(0, 0.01)
        if cfg.mask_channel_prob > 0:
            channel_mask = compute_mask_indices(
                (value.shape[0], value.shape[2]),
                None,
                cfg.mask_channel_prob,
                cfg.mask_channel_length,
                seed=self.config.common.seed + 1,
                epoch=update,
                indices=sample_ids,
            ).to(value.device)
            # Mathematics: selected channel indices c are set to zero for every
            # t, broadcasting mask[b,c] across the time axis.
            # Interpretation: channel masking removes feature detectors across
            # a recording rather than hiding one temporal segment.
            value = value.masked_fill(channel_mask[:, None, :], 0)
        return value

    def forward(
        self,
        waveform: Tensor,
        *,
        target: Tensor | None = None,
        padding_mask: Tensor | None = None,
        sample_ids: Tensor | None = None,
        update: int = 0,
    ) -> FineTuningOutput:
        """Produce configured logits and, when targets are supplied, focal loss."""

        cfg = self.config
        if sample_ids is None:
            sample_ids = torch.arange(waveform.shape[0], device=waveform.device)
        if target is not None and cfg.model.classification_head == "cls":
            target_padding = self.encoder.convert_padding_mask(target, padding_mask)
            target = sequence_targets(target, target_padding)
        if self.training and cfg.model.source_mixup >= 0 and cfg.model.mixup_prob > 0:
            mixed = mix_waveforms(
                waveform,
                strength=cfg.model.source_mixup,
                probability=cfg.model.mixup_prob,
                same_ratio=cfg.model.same_mixup,
                gain_mode=cfg.model.gain_mode,
                sample_rate=cfg.task.sample_rate,
                window_seconds=cfg.model.mixing_window_length,
            )
            waveform = mixed.waveforms
            if target is not None and cfg.model.target_mixup:
                target = mix_targets(
                    target,
                    ratios=mixed.ratios,
                    permutation=mixed.permutation,
                    applied=mixed.applied,
                    same_ratio=cfg.model.same_mixup,
                )

        # Mathematics: encoder gradients are enabled iff update >= U_freeze.
        # Interpretation: the classifier first adapts to fixed pretrained
        # features, then the backbone begins task-specific adaptation.
        backbone_trainable = update >= cfg.model.freeze_finetune_updates
        context = nullcontext() if backbone_trainable else torch.no_grad()
        with context:
            projected, feature_padding = self._project(waveform, padding_mask)
            if self.training and cfg.model.apply_mask:
                projected = self._apply_masks(projected, feature_padding, sample_ids, update)
            encoded = self.encoder.encode_projected(projected, feature_padding)
        layer_count = min(cfg.model.average_top_k_layers, len(encoded.layer_outputs))
        if layer_count == 0:
            raise ValueError("fine-tuning requires transformer layer outputs")
        # Mathematics: z = K^{-1} sum_{l=L-K+1}^L h_l and logits = W Drop(z)+b.
        # Interpretation: the classifier combines several semantic depths,
        # mirroring the top-layer averaging used for pretraining targets.
        features = torch.stack(encoded.layer_outputs[-layer_count:]).mean(dim=0)
        output_padding = encoded.padding_mask
        if cfg.model.classification_head == "cls":
            features = features[:, 0]
            output_padding = None
        elif self.encoder.use_cls_token:
            features = features[:, 1:]
            if output_padding is not None:
                output_padding = output_padding[:, 1:]
        logits = self.classifier(self.final_dropout(features))
        loss = self.focal_loss(logits, target) if target is not None else None
        if cfg.model.classification_head == "cls":
            # Mathematics: sequence loss sums B×C decisions but normalizes by B.
            # Interpretation: each recording contributes one classification
            # example regardless of its duration or label-vocabulary size.
            sample_size = logits.shape[0]
        else:
            # Mathematics: sample_size=B×T counts frame tokens; the class
            # dimension stays inside the summed focal loss and is not counted.
            # Interpretation: the default retains archived loss normalization.
            sample_size = (
                target.shape[0] * target.shape[1]
                if target is not None
                else logits.shape[0] * logits.shape[1]
            )
        return FineTuningOutput(
            logits=logits,
            padding_mask=output_padding,
            layer_outputs=encoded.layer_outputs,
            targets=target,
            loss=loss,
            sample_size=sample_size,
        )
