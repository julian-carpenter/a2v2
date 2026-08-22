"""Test focal loss, target mixup, layer averaging, and pretrained encoder validation. These
are the main equations and state boundaries specific to supervised adaptation."""

import math
from pathlib import Path

import pytest
import torch

from a2v2.config import load_config
import a2v2.model as model_module
from a2v2.model import (
    Animal2VecFineTuningModel,
    AudioEncoder,
    MixResult,
    mix_targets,
)
from a2v2.model import SigmoidFocalLoss


ROOT = Path(__file__).parents[2]


def test_audio_encoder_propagates_activation_checkpointing_to_both_stacks() -> None:
    """Apply one config value to the prenet and main Transformer stacks."""

    config = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.checkpoint_activations=true",),
    )

    encoder = AudioEncoder.from_config(config)

    assert encoder.prenet.checkpoint_activations is True
    assert encoder.transformer.checkpoint_activations is True


def test_active_finetuning_config_controls_encoder_checkpointing() -> None:
    """Do not let an older pretrained config disable the active policy."""

    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=("model.checkpoint_activations=true",),
    )

    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )

    assert pretrain.model.checkpoint_activations is False
    assert model.encoder.prenet.checkpoint_activations is True
    assert model.encoder.transformer.checkpoint_activations is True


def test_geglu_config_selects_packed_ffn_in_both_encoder_stacks() -> None:
    """Use the pretrained architecture choice in prenet and main fine-tuning stacks."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.ffn_type=geglu",),
    )
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")

    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )

    blocks = [*model.encoder.prenet.blocks, *model.encoder.transformer.blocks]
    assert len(blocks) == pretrain.model.audio.prenet_depth + pretrain.model.depth
    assert all(isinstance(block.mlp, model_module.PackedGEGLU) for block in blocks)


def test_geglu_parameter_estimate_counts_second_packed_input_half() -> None:
    """Include the extra hidden-by-input weight and hidden bias in every block."""

    mlp = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    geglu = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.ffn_type=geglu",),
    )
    dimension = geglu.model.embed_dim
    hidden = int(dimension * geglu.model.mlp_ratio)
    depth = geglu.model.depth + geglu.model.audio.prenet_depth

    difference = (
        AudioEncoder.estimated_parameter_count(geglu)
        - AudioEncoder.estimated_parameter_count(mlp)
    )

    assert difference == depth * (dimension * hidden + hidden)


def test_mlp_encoder_checkpoint_has_strict_geglu_shape_mismatch() -> None:
    """Reject an MLP checkpoint instead of reinterpreting it as packed GEGLU."""

    mlp_config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    geglu_config = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.ffn_type=geglu",),
    )
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    mlp_state = AudioEncoder.from_config(mlp_config).state_dict()
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=geglu_config,
    )

    with pytest.raises(
        ValueError,
        match=r"prenet\.blocks\.0\.mlp\.fc1\.weight.*expected.*received",
    ):
        model.load_pretrained_encoder(mlp_state)


def test_deepscale_encoder_checkpoint_loading_overrides_initialization() -> None:
    """Make supplied encoder tensors authoritative over constructor randomness."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.initialization=deepscale_lm",),
    )
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    source = AudioEncoder.from_config(pretrain)
    checkpoint = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }
    checkpoint["transformer.blocks.0.attn.qkv.weight"].fill_(0.125)

    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
        encoder_state=checkpoint,
    )

    for name, expected in checkpoint.items():
        torch.testing.assert_close(
            model.encoder.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )


def test_legacy_initialization_matches_frozen_tensor_and_rng_oracle() -> None:
    """Freeze meaningful tensor values and construction RNG order independently."""

    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    torch.manual_seed(101)
    model = AudioEncoder.from_config(config).eval()
    expected = {
        "project_features.weight": torch.tensor([
            0.011359423398971558,
            -0.04257035255432129,
            0.0513286292552948,
            0.06016454100608826,
            -0.06892189383506775,
            -0.1976364254951477,
            -0.0064213573932647705,
            0.019809722900390625,
        ]),
        "prenet.blocks.0.attn.qkv.weight": torch.tensor([
            0.03903722018003464,
            -0.017712950706481934,
            0.009713570587337017,
            -0.009759217500686646,
            -0.0012960262829437852,
            0.007639708463102579,
            0.01715032383799553,
            -0.02523055486381054,
        ]),
        "prenet.blocks.0.mlp.fc1.weight": torch.tensor([
            -0.01275860145688057,
            0.001614767825230956,
            -0.01917700283229351,
            0.03594064339995384,
            0.0319422148168087,
            0.04713429883122444,
            0.021025190129876137,
            0.00992377009242773,
        ]),
        "transformer.blocks.1.mlp.fc2.weight": torch.tensor([
            -0.011548127979040146,
            0.021507998928427696,
            -0.0016178557416424155,
            0.012524111196398735,
            -0.016345340758562088,
            -0.002257388550788164,
            -0.06215336546301842,
            -0.002851117867976427,
        ]),
    }

    assert config.model.initialization == "legacy"
    for name, frozen in expected.items():
        torch.testing.assert_close(
            model.state_dict()[name].flatten()[:8],
            frozen,
            rtol=0,
            atol=0,
        )
    # CPU initializers are deterministic for the pinned PyTorch runtime, so
    # this exact probe catches added/removed/reordered construction draws.
    torch.testing.assert_close(
        torch.rand(8),
        torch.tensor([
            0.0024824142456054688,
            0.8695276975631714,
            0.6588976383209229,
            0.2709083557128906,
            0.334867000579834,
            0.000815272331237793,
            0.9917974472045898,
            0.5380000472068787,
        ]),
        rtol=0,
        atol=0,
    )


def test_legacy_forward_matches_frozen_output_oracle() -> None:
    """Freeze end-to-end legacy residual and initialization behavior."""

    config = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    torch.manual_seed(101)
    model = AudioEncoder.from_config(config).eval()
    waveform = torch.linspace(-1.0, 1.0, 128).reshape(2, 64)
    with torch.no_grad():
        output = model(waveform)

    # GEMM/convolution reduction order may vary in the final bits across CPU
    # builds, so the forward oracle uses tight numerical tolerances rather than
    # a byte hash. The initialized tensor/RNG oracle above remains bit-exact.
    torch.testing.assert_close(
        output.x[0, 0, :8],
        torch.tensor([
            -0.4695950448513031,
            2.1438100337982178,
            -0.4790281057357788,
            -0.7077038884162903,
            -0.5008716583251953,
            0.9729785919189453,
            -1.0133659839630127,
            -0.2658218741416931,
        ]),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output.x[1, -1, -8:],
        torch.tensor([
            -1.5285754203796387,
            0.3256520926952362,
            1.867682695388794,
            -0.7041085958480835,
            0.18142500519752502,
            0.054556552320718765,
            0.09402453154325485,
            -0.3258442282676697,
        ]),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output.layer_outputs[0][0, 0, :8],
        torch.tensor([
            0.005438052583485842,
            0.0009517626604065299,
            -0.002345379674807191,
            0.009505631402134895,
            0.0051078493706882,
            0.0019196192733943462,
            -0.008520583622157574,
            0.001855779206380248,
        ]),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output.layer_outputs[-1][1, -1, -8:],
        torch.tensor([
            -0.006451415363699198,
            0.0033123784232884645,
            -0.008892624638974667,
            0.0025248508900403976,
            0.007080330513417721,
            0.00374116119928658,
            0.011118483729660511,
            -0.0061199851334095,
        ]),
        rtol=1e-5,
        atol=1e-6,
    )
    assert output.x.square().mean().item() == pytest.approx(
        0.999989926815033,
        rel=1e-5,
        abs=1e-7,
    )
    assert [layer.square().mean().item() for layer in output.layer_outputs] == pytest.approx(
        [3.293444024166092e-05, 4.3511441617738456e-05],
        rel=1e-5,
        abs=1e-8,
    )


def test_deepscale_finetuning_preserves_classifier_xavier_initialization() -> None:
    """Do not reinterpret the supervised classifier as a Transformer FFN role."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=(
            "model.embed_dim=256",
            "model.num_heads=8",
            "model.initialization=deepscale_lm",
        ),
    )
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    torch.manual_seed(111)
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    )

    expected_std = math.sqrt(
        2.0 / (model.classifier.in_features + model.classifier.out_features)
    )
    # 512 Xavier samples have ~3.1% relative sample-std error; 10% is robust.
    assert model.classifier.weight.std().item() == pytest.approx(
        expected_std,
        rel=0.10,
    )
    assert torch.count_nonzero(model.classifier.bias) == 0


def test_sigmoid_focal_loss_matches_hand_calculation() -> None:
    """Check sigmoid focal loss matches hand calculation."""
    logits = torch.zeros(2)
    targets = torch.tensor([1.0, 0.0])
    loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction="sum")(logits, targets)
    expected = -torch.log(torch.tensor(0.5)) * 0.25 * (0.25 + 0.75)
    assert torch.allclose(loss, expected)


def test_target_mixup_uses_unadjusted_sampling_ratio() -> None:
    """Check target mixup uses unadjusted sampling ratio."""
    targets = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    mixed = mix_targets(
        targets,
        ratios=torch.tensor([0.25]),
        permutation=torch.tensor([1, 0]),
        applied=torch.tensor([True, True]),
        same_ratio=True,
    )
    assert torch.allclose(mixed[0], torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(mixed[1], torch.tensor([[0.75, 0.25]]))


def test_target_mixup_accepts_one_ratio_per_selected_example() -> None:
    """Check target mixup accepts one ratio per selected example."""
    targets = torch.eye(4).unsqueeze(1)
    mixed = mix_targets(
        targets,
        ratios=torch.tensor([0.25, 0.75]),
        permutation=torch.tensor([1, 0, 3, 2]),
        applied=torch.tensor([True, False, True, False]),
        same_ratio=False,
    )
    assert torch.allclose(mixed[0, 0], torch.tensor([0.25, 0.75, 0.0, 0.0]))
    assert torch.allclose(mixed[2, 0], torch.tensor([0.0, 0.0, 0.75, 0.25]))
    assert torch.equal(mixed[1], targets[1])
    assert torch.equal(mixed[3], targets[3])


def test_finetuning_averages_requested_encoder_layers() -> None:
    """Check finetuning averages requested encoder layers."""
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain).eval()
    output = model(torch.randn(2, 64), update=2)
    expected_features = torch.stack(output.layer_outputs[-2:]).mean(dim=0)
    expected_logits = model.classifier(model.final_dropout(expected_features))
    assert torch.allclose(output.logits, expected_logits)
    assert output.logits.shape == (2, 16, 2)


def test_frame_head_strips_cls_from_logits_and_padding() -> None:
    """Keep event logits and their mask on the frame-only time axis."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.use_cls_token=true",),
    )
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    ).eval()
    waveform = torch.randn(2, 64)
    padding = torch.zeros_like(waveform, dtype=torch.bool)
    padding[1, 32:] = True

    output = model(waveform, padding_mask=padding, update=2)

    expected_features = torch.stack(output.layer_outputs[-2:]).mean(dim=0)[:, 1:]
    expected_logits = model.classifier(model.final_dropout(expected_features))
    torch.testing.assert_close(output.logits, expected_logits)
    assert output.logits.shape == (2, 16, 2)
    assert output.padding_mask is not None
    assert output.padding_mask.shape == (2, 16)
    assert output.padding_mask[1, 8:].all()


def test_cls_head_averages_top_layer_cls_states() -> None:
    """Classify the averaged special token rather than any frame token."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.use_cls_token=true",),
    )
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=(
            "model.use_cls_token=true",
            "model.classification_head=cls",
        ),
    )
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    ).eval()

    output = model(torch.randn(2, 64), update=2)

    expected_features = torch.stack(output.layer_outputs[-2:]).mean(dim=0)[:, 0]
    expected_logits = model.classifier(model.final_dropout(expected_features))
    torch.testing.assert_close(output.logits, expected_logits)
    assert output.logits.shape == (2, 2)


def test_cls_mixup_shares_permutation_ratio_and_union_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mix waveforms, clip labels, and valid positions with one pairing."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.use_cls_token=true",),
    )
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=(
            "model.use_cls_token=true",
            "model.classification_head=cls",
            "model.source_mixup=0.25",
            "model.mixup_prob=1.0",
            "model.target_mixup=true",
            "model.same_mixup=false",
            "model.gain_mode=none",
            "model.apply_mask=false",
        ),
    )
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    ).train()
    target = torch.zeros(2, 16, 2)
    target[0, 1, 0] = 1
    target[1, 10, 1] = 1
    waveform = torch.zeros(2, 64)
    waveform[0, :32] = 1
    waveform[1, :48] = 2
    padding = torch.zeros_like(waveform, dtype=torch.bool)
    padding[0, 32:] = True
    padding[1, 48:] = True
    ratios = torch.tensor([0.25, 0.6])
    permutation = torch.tensor([1, 0])
    observed: dict[str, torch.Tensor | None] = {}
    mix_waveforms = model_module.mix_waveforms
    project = model._project
    encode_projected = model.encoder.encode_projected

    def fixed_mix(waveform: torch.Tensor, **_: object) -> MixResult:
        """Run real waveform mixup with fixed metadata."""

        return mix_waveforms(
            waveform,
            strength=0.25,
            probability=1.0,
            same_ratio=False,
            gain_mode="none",
            sample_rate=8_000,
            window_seconds=0.1,
            ratios=ratios,
            permutation=permutation,
        )

    def record_project(
        mixed_waveform: torch.Tensor,
        mixed_padding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Capture the mixed waveform and sample mask entering the encoder."""

        observed["waveform"] = mixed_waveform.detach().clone()
        observed["sample_padding"] = (
            mixed_padding.detach().clone() if mixed_padding is not None else None
        )
        return project(mixed_waveform, mixed_padding)

    def record_encode_projected(
        projected: torch.Tensor,
        feature_padding: torch.Tensor | None = None,
        mask_info: object | None = None,
    ) -> object:
        """Capture the feature mask received by CLS attention."""

        observed["feature_padding"] = (
            feature_padding.detach().clone() if feature_padding is not None else None
        )
        return encode_projected(projected, feature_padding, mask_info)  # type: ignore[arg-type]

    monkeypatch.setattr(model_module, "mix_waveforms", fixed_mix)
    monkeypatch.setattr(model, "_project", record_project)
    monkeypatch.setattr(model.encoder, "encode_projected", record_encode_projected)

    output = model(waveform, target=target, padding_mask=padding, update=2)

    assert output.targets is not None
    torch.testing.assert_close(
        output.targets,
        torch.tensor([[0.25, 0.75], [0.4, 0.6]]),
    )
    expected_waveform = torch.empty_like(waveform)
    expected_waveform[0, :32] = 1.75 / (0.25**2 + 0.75**2) ** 0.5
    expected_waveform[0, 32:48] = 1.5 / (0.25**2 + 0.75**2) ** 0.5
    expected_waveform[0, 48:] = 0
    expected_waveform[1, :32] = 1.6 / (0.6**2 + 0.4**2) ** 0.5
    expected_waveform[1, 32:48] = 1.2 / (0.6**2 + 0.4**2) ** 0.5
    expected_waveform[1, 48:] = 0
    torch.testing.assert_close(observed["waveform"], expected_waveform)
    expected_padding = torch.zeros_like(padding)
    expected_padding[:, 48:] = True
    assert observed["sample_padding"] is not None
    assert torch.equal(observed["sample_padding"], expected_padding)
    expected_feature_padding = torch.zeros(2, 16, dtype=torch.bool)
    expected_feature_padding[:, 12:] = True
    assert torch.equal(observed["feature_padding"], expected_feature_padding)


def test_cls_focal_loss_counts_sequence_examples() -> None:
    """Normalize sequence focal loss by recordings, not classes or frames."""

    pretrain = load_config(
        ROOT / "tests/fixtures/tiny_pretrain.yaml",
        overrides=("model.use_cls_token=true",),
    )
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=(
            "model.use_cls_token=true",
            "model.classification_head=cls",
        ),
    )
    model = Animal2VecFineTuningModel.from_config(
        finetune,
        pretrained_config=pretrain,
    ).eval()
    target = torch.zeros(3, 16, 2)
    target[0, 2, 0] = 1
    target[1, 4, 1] = 1

    output = model(torch.randn(3, 64), target=target, update=2)

    assert output.targets is not None
    assert output.loss is not None
    torch.testing.assert_close(output.loss, model.focal_loss(output.logits, output.targets))
    assert output.sample_size == 3


def test_cls_head_rejects_pretrained_encoder_without_cls() -> None:
    """Fail construction before a sequence head can read frame zero as CLS."""

    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(
        ROOT / "tests/fixtures/tiny_finetune.yaml",
        overrides=(
            "model.use_cls_token=true",
            "model.classification_head=cls",
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"classification_head=cls.*pretrained.*use_cls_token=true",
    ):
        Animal2VecFineTuningModel.from_config(
            finetune,
            pretrained_config=pretrain,
        )


def test_sequence_targets_ignore_padded_frame_labels() -> None:
    """Exclude padded positives when deriving recording-level occurrences."""

    target = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    ])
    padding = torch.tensor([
        [False, False, True],
        [False, True, True],
    ])

    result = model_module.sequence_targets(target, padding)

    torch.testing.assert_close(
        result,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )


def test_pretrained_encoder_state_rejects_incompatible_tensors() -> None:
    """Check pretrained encoder state rejects incompatible tensors."""
    pretrain = load_config(ROOT / "tests/fixtures/tiny_pretrain.yaml")
    finetune = load_config(ROOT / "tests/fixtures/tiny_finetune.yaml")
    model = Animal2VecFineTuningModel.from_config(finetune, pretrained_config=pretrain)
    state = model.encoder.state_dict()
    state["project_features.weight"] = torch.zeros(3, 3)
    try:
        model.load_pretrained_encoder(state)
    except ValueError as exc:
        assert "project_features.weight" in str(exc)
    else:
        raise AssertionError("incompatible encoder state was accepted")
