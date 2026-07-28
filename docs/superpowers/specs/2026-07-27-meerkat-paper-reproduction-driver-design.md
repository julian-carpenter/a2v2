# Eight-GPU MeerKAT reproduction driver design

Date: 2026-07-27

## Goal

Add one deployment script that performs a practical, same-order reproduction
of the Animal2Vec 1.0 MeerKAT result on a host with eight NVIDIA A100 40 GB
GPUs. The default workflow is deliberately narrower than the paper:

1. pretrain once on all eight GPUs;
2. fine-tune the resulting student encoder once, using the 100% labels for
   fold 0 and all eight GPUs;
3. validate the best fine-tuning checkpoint; and
4. print and save a machine-readable final report.

The user explicitly accepts that this is not a numerically exact reproduction.
The script must therefore describe the topology and batch changes as an
approximate reproduction instead of silently presenting them as the published
four-rank experiment.

## Scientific scope

The default inputs are:

- `pretrain.tsv`;
- `train_0.tsv`; and
- `valid_0.tsv`.

The fine-tuning fraction and fold remain selectable so the same driver can be
reused for `025`, `001`, or another fold, but only one fine-tuning run is
launched. The workflow does not cover the other four MeerKAT folds or the
paper's Xeno-canto/NIPS4Bplus experiment. The final report records the chosen
manifests, their hashes, and an explicit scope warning.

## Eight-GPU batch profile

The published configurations use four ranks. This driver overrides both
recipes to eight ranks and adjusts accumulation:

| Stage | Per-rank `max_tokens` | Accumulation | Ranks | Effective tokens/update |
| --- | ---: | ---: | ---: | ---: |
| Published pretraining | 408,000 | 5 | 4 | 8,160,000 |
| Driver pretraining | 408,000 | 3 | 8 | 9,792,000 |
| Published fine-tuning | 426,667 | 9 | 4 | 15,360,012 |
| Driver fine-tuning | 960,000 | 2 | 8 | 15,360,000 |

Pretraining retains 408,000 tokens per rank because the verified four-rank
burn-in already reserved approximately 37.1 GiB on a 40 GB A100. Adding ranks
increases throughput but does not create more memory within one rank.

Fine-tuning reserved only approximately 11 GiB during the verified frozen
burn-in. The 960,000-token default is a conservative 2.25-times increase that
uses more of each GPU while leaving headroom for the transformer-unfreeze
boundary at update 10,000. Real memory depends on recording lengths and
packing. Both token budgets and both accumulation factors are environment
variables so an operator can lower them after a genuine out-of-memory report
without editing the script.

The learning rates and update horizons remain unchanged. Fine-tuning retains
almost exactly the published effective token batch. Pretraining is 20% larger,
which is acceptable under the requested same-order target but is recorded as a
material recipe change.

## Files

The implementation adds:

- `scripts/reproduce_meerkat_paper.sh`: preflight checks, provenance capture,
  eight-rank pretraining, eight-rank fine-tuning, resume behavior, validation,
  and final report printing.
- `scripts/evaluate_finetuning_checkpoint.py`: load one native fine-tuning
  checkpoint, run the requested validation manifest, and write JSON metrics.

The runtime package remains at six Python files.

## Interface

```bash
bash scripts/reproduce_meerkat_paper.sh \
  /datasets/MeerKAT/manifests \
  /experiments/a2v2-meerkat-fold0
```

Options:

```text
--fold 0
--fraction 100
--dry-run
```

Important environment controls:

| Variable | Default | Purpose |
| --- | --- | --- |
| `A2V2_PYTHON` | `python` | Prepared environment interpreter |
| `A2V2_TORCHRUN` | `torchrun` | Distributed launcher |
| `A2V2_TRAIN_ENTRY` | installed `a2v2-train` | Training entry point |
| `A2V2_GPUS` | `0,1,2,3,4,5,6,7` | Eight physical devices |
| `A2V2_PRETRAIN_MAX_TOKENS` | `408000` | Safe verified per-rank pretrain budget |
| `A2V2_PRETRAIN_UPDATE_FREQ` | `3` | Eight-rank accumulation |
| `A2V2_FINETUNE_MAX_TOKENS` | `960000` | Higher-memory fine-tuning budget |
| `A2V2_FINETUNE_UPDATE_FREQ` | `2` | Preserves the old global token batch |
| `A2V2_EVAL_MAX_TOKENS` | `320000` | Bounded FP32 validation batch |
| `A2V2_EVAL_WORKERS` | `20` | Validation DataLoader workers |

The real preflight requires exactly eight selected, visible CUDA devices and
rejects devices with less than 39 GiB. Dry-run validates the command graph and
manifest names without requiring CUDA.

## Output and resume

```text
<output>/
├── environment/
│   ├── nvidia-smi.txt
│   ├── python-version.txt
│   ├── pip-freeze.txt
│   ├── manifest-sha256.txt
│   └── recipe-sha256.txt
├── pretrain/
│   ├── checkpoint_last.pt
│   └── train.log
├── finetune/
│   ├── checkpoint_last.pt
│   ├── checkpoint_best.pt
│   └── train.log
└── final-evaluation/
    ├── validation.log
    └── final-evaluation-report.json
```

Each training stage resumes from its native `checkpoint_last.pt` when present.
The configured update horizon is not shortened. Validation prefers
`checkpoint_best.pt` and falls back to `checkpoint_last.pt` only if no best
checkpoint was produced, recording which file was used.

## Final report

The evaluator reconstructs the fine-tuning model from the active and
pretrained configs stored in the checkpoint, strictly loads the state, and
runs the existing native validation path. It reports:

- loss and frame precision, recall, F1, accuracy, and average precision;
- checkpoint update and hash;
- validation manifest hash and number of manifest rows;
- fold, fraction, topology, and effective-batch settings;
- Python, PyTorch, CUDA, and GPU metadata; and
- the approximate-reproduction scope warning.

The shell script prints the saved JSON report after validation.

## Verification

CPU tests cover shell syntax, the one-fold dry-run command graph, eight-rank
and batch overrides, missing-manifest rejection, and a real tiny-checkpoint
validation. The CPU host cannot test memory fit or NCCL. The deployment run is
the authority for those properties.
