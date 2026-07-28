# A100 GPU verification report: 2026-07-23

> **Current source note, 2026-07-27:** This report records the exact paths and
> commands used on the GPU machine and therefore keeps its historical names.
> The verified implementation now lives in six files under `a2v2/`:
> `config.py`, `data.py`, `model.py`, `training.py`, `workflows.py`, and
> `__init__.py`. These files implement the Animal2Vec 1.0 reproduction
> baseline inside the extensible A2V2 project. The old training command maps
> to `a2v2-train`, or to the installed script path passed to `torchrun`. Read
> [code-guide.md](code-guide.md) for the live source map and recheck commands.

> **Intermediate standalone layout note, 2026-07-24:** This report preserves paths and
> commands from the source snapshot that ran on the A100 host. The standalone
> package now lives directly under `animal2vec/`; it no longer contains
> `src/`, `nn/`, or Fairseq-only parity scripts. No model equation, attribute,
> state-dict key, checkpoint field, YAML recipe, or console command name
> changed during that layout refactor. See [the code guide](code-guide.md) for
> the current reading order, old-to-new path table, and GPU recheck commands.

This report records the GPU verification of the Fairseq-free Animal2Vec
rewrite at commit `b1be8d45eaf9307458dafad9902ada0422a8da52`. The complete
artifact bundle is under
`results/gpu_verification/20260723T081428Z/`.

## Report control, scope, and interpretation

This is the human-readable audit record for the verification session captured
from `2026-07-23T08:16:00Z` through `2026-07-23T15:20:40Z`. It is intended to
be read together with the machine-readable decision record
`results/gpu_verification/20260723T081428Z/verification-report.json`, the
resolved configurations, complete rank logs, package locks, checkpoint hashes,
and comparison reports retained beneath the same result root. At the time this
report was finalized, that result root occupied approximately 273 GB.

The operating specification was, in order:

1. `docs/A100_GPU_VERIFICATION_BOOTSTRAP.md`;
2. `docs/GPU_VERIFICATION_HANDOFF.md`;
3. `README.md`;
4. `docs/architecture-map.md`;
5. `docs/checkpoint-conversion.md`; and
6. `docs/reproducing-paper.md`.

All applicable repository `AGENTS.md` instructions were read before work. The
repository was on branch `main` at commit
`b1be8d45eaf9307458dafad9902ada0422a8da52`. The pre-existing staged changes to
`README.md`, `docs/A100_GPU_VERIFICATION_BOOTSTRAP.md`, and
`docs/GPU_VERIFICATION_HANDOFF.md` were treated as user-owned input and were
not reset, unstaged, or rewritten. The initial and final working-tree states,
including staged, tracked, and untracked source hashes, are preserved in
`environment/git-status.txt`, `environment/git-status-final.txt`,
`environment/working-tree-index.patch`,
`environment/working-tree-index-final.patch`, and the corresponding tracked
and untracked captures.

The word **pass** in this report means that the evidence required by the named
handoff gate was produced and its declared comparison criteria were met. It
does not mean that a short burn-in is equivalent to full training. In
particular:

- exact-recipe memory and resume tests retained the published topology and
  hyperparameters but stopped after one or two successful optimizer updates;
- regenerated split manifests were used only where the official split files
  were absent and are explicitly non-authoritative;
- numerical parity means agreement under the recorded exact/tolerance policy,
  not identity of separately serialized checkpoint files; and
- Gate G is the only gate that authorizes the phrase “paper reproduced.” Gate
  G did not pass, so this report makes no such claim.

## Decision

Gates A through E pass. Gate F is partially complete and remains blocked on
publication artifacts. Gate G does not pass and the paper has **not** been
reproduced.

The native implementation passed CPU, single-A100 FP32/FP16, official
checkpoint conversion/parity, training-step parity, two-rank NCCL, four-rank
NCCL, exact distributed resume, and unchanged-recipe memory gates. The
released MeerKAT archive was exhaustively audited and the native event
evaluator agrees exactly with the archived evaluator on controlled fixtures.
The publisher release does not contain the official split manifests, differs
from one paper count, and is insufficient for the paper-level fold and event
metric runs.

## Baseline and environment

The host environment was captured before any setup and was not modified.
Verification used four isolated environments:

| Purpose | Python | PyTorch | CUDA runtime |
| --- | --- | --- | --- |
| Native authority | CPython 3.14.6 | 2.12.1+cu130 | 13.0 |
| Native parity | CPython 3.10.14 | native parity wheel | 12.6 |
| Archived Fairseq parity | CPython 3.10.14 | 2.6.0+cu126 | 12.6 |
| Analysis only | isolated analysis stack | separate from runtime | n/a |

The authority host has eight NVIDIA A100-SXM4-40GB GPUs, each reporting
40,960 MiB, with driver 580.95.05. The native authority runtime reports cuDNN
9.20.0 and NCCL 2.29.7. The reported driver-side CUDA 13.2 value was not treated
as a PyTorch binary runtime: the native wheel uses CUDA 13.0.

Authoritative four-rank jobs used physical GPUs 0–3:

| Rank/GPU | UUID |
| --- | --- |
| 0 | `GPU-47400a00-a1da-909a-242e-54c52ed2bb3d` |
| 1 | `GPU-96ce04fc-7484-3fa5-4eda-3f63b43ebd29` |
| 2 | `GPU-d9d6d09e-df77-352d-f38f-207875f31100` |
| 3 | `GPU-9ec96d03-bd7a-d9ad-dc66-f3579640e788` |

Full environment captures, freezes, UUIDs, topology, driver output, cuDNN,
NCCL, and untouched-host information are in
`results/gpu_verification/20260723T081428Z/environment/`.

### Untouched host capture

The host snapshot was taken before creating or installing into any verification
environment. No command upgraded, uninstalled, or replaced a host package. The
relevant baseline was:

| Property | Captured value |
| --- | --- |
| Operating system | Ubuntu 24.04.4 LTS |
| Kernel | Linux 6.8.0-90-generic |
| CPU | 2 × AMD EPYC 7742 64-Core Processor |
| Logical processors | 256 |
| NUMA nodes | 8 |
| Host memory | approximately 2.0 TiB; no swap |
| Host Python | `/usr/bin/python`, CPython 3.12.3 |
| Host PyTorch | `2.11.0a0+a6c236b9fd.nv26.03.46836102` |
| Host PyTorch CUDA runtime | 13.2 |
| Host cuDNN integer | 92000 |
| Host NCCL | 2.29.7 |
| NVIDIA driver | 580.95.05 |
| GPU model/count | 8 × NVIDIA A100-SXM4-40GB |
| Compute capability | 8.0 |
| MIG / compute mode | disabled / default |

`nvidia-smi` reports 40,960 MiB per device, while
`torch.cuda.get_device_properties(...).total_memory` reports
42,405,855,232 bytes. Both are retained because allocator headroom must be
computed against the capacity visible to the framework, not merely the product
name or nominal 40 GiB label.

The NVLink topology reports `NV12` between every GPU pair. GPUs 0–3 were
reserved for all authoritative recipe and four-rank tests. GPUs 4–7 were not
substituted to create an eight-rank paper topology; they were available only
for independent diagnostics. Their UUIDs were:

| Physical GPU | UUID | Verification role |
| ---: | --- | --- |
| 0 | `GPU-47400a00-a1da-909a-242e-54c52ed2bb3d` | authority rank 0 |
| 1 | `GPU-96ce04fc-7484-3fa5-4eda-3f63b43ebd29` | authority rank 1 |
| 2 | `GPU-d9d6d09e-df77-352d-f38f-207875f31100` | authority rank 2 |
| 3 | `GPU-9ec96d03-bd7a-d9ad-dc66-f3579640e788` | authority rank 3 |
| 4 | `GPU-59774df4-a398-1b55-0728-9489343f075e` | diagnostics only |
| 5 | `GPU-4e51c2c9-f450-74b5-e31d-1422453bb5f6` | diagnostics only |
| 6 | `GPU-2bdd660e-1c52-ee16-dca8-affeff6328c2` | diagnostics only |
| 7 | `GPU-b77ae17c-ca08-cf10-530c-36fd1b0795cb` | diagnostics only |

The raw supporting files are `environment/os-release.txt`,
`environment/uname.txt`, `environment/lscpu.txt`, `environment/memory.txt`,
`environment/host-python-version.txt`, `environment/host-pip-freeze.txt`,
`environment/host-torch.txt`, `environment/nvidia-smi.txt`,
`environment/nvidia-smi-q.txt`, `environment/gpu-inventory.csv`, and
`environment/nvidia-topology.txt`.

### Isolated environment construction

The four-environment boundary was enforced deliberately. Fairseq and its old
configuration/runtime stack never entered the native authority environment,
and analysis packages never became implicit runtime dependencies.

#### Native authority: `a2v-native-cu130`

- Interpreter: CPython 3.14.6, standard-GIL build, executable
  `.pythons/cpython-3.14.6/bin/python3.14`.
- Python source SHA-256:
  `143b1dddefaec3bd2e21e3b839b34a2b7fb9842272883c576420d605e9f30c63`.
- OpenSSL: 3.0.13.
- PyTorch: `2.12.1+cu130`; CUDA runtime 13.0; cuDNN 9.20.0; NCCL 2.29.7.
- Principal packages: NumPy 2.5.0, h5py 3.16, SoundFile 0.14,
  PyYAML 6.0.3, and pytest 9.0.3.
- Final setuptools: 81.0.0. The guide's initial 82.0.1 pin violated the
  PyTorch wheel metadata constraint `setuptools<82`; the narrow downgrade was
  applied only inside this environment, after which `pip check` was clean.
- Complete lock: `environment/a2v-native-cu130.freeze.final.txt`.

The source tree and installed package were scanned for forbidden imports. The
scan found no Fairseq, Hydra, OmegaConf, torchaudio, timm, Transformers,
TensorFlow, pandas, SciPy, scikit-learn, librosa, or numba dependency in the
native runtime. See `environment/native-forbidden-import-scan-final.json` and
`logs/native-cu130-pip-check-final.log`.

#### Native parity: `a2v-native-parity-cu126`

- Interpreter: CPython 3.10.14.
- PyTorch: `2.6.0+cu126`; CUDA runtime 12.6; cuDNN 9.5.1; NCCL 2.21.5.
- Principal packages: NumPy 1.24.4, h5py 3.10, SoundFile 0.12.1,
  PyYAML 6.0.2, and pytest 8.3.5.
- Fairseq and the other forbidden frameworks remained absent.
- Complete lock: `environment/a2v-native-parity-cu126.freeze.txt`.

This environment isolated framework-version effects when native behavior was
compared with archived Fairseq behavior under the same PyTorch/CUDA generation.

#### Archived parity: `a2v-legacy-parity-cu126`

- Interpreter: CPython 3.10.14.
- PyTorch/vision/audio: `torch==2.6.0+cu126`, `torchaudio==2.6.0`, and
  `torchvision==0.21.0`.
- CUDA runtime 12.6; cuDNN 9.5.1; NCCL 2.21.5.
- Fairseq checkout: commit
  `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99` (`0.12.2`).
- Compatibility stack: Hydra 1.0.7, OmegaConf 2.0.6, timm 0.6.13,
  TensorFlow CPU 2.15.1, pandas 1.5.3, and scikit-learn 1.2.2.
- Complete lock: `environment/a2v-legacy-parity-cu126.freeze.txt`.

OmegaConf 2.0.6 shipped metadata containing the modern-pip-invalid requirement
`PyYAML (>=5.1.*)`. Only the installed distribution metadata was normalized to
`PyYAML (>=5.1)`; the original metadata and both hashes are retained in
`environment/omegaconf-2.0.6.METADATA.original` and
`environment/omegaconf-metadata.sha256`. No archived model source, numerical
equation, or Fairseq checkpoint content was changed. `pip check` was clean
after this packaging-only correction.

#### Analysis: `a2v-analysis`

- Interpreter: CPython 3.14.6.
- PyTorch: `2.12.1+cu130`.
- Analysis packages included NumPy 2.4.6, pandas 2.3.3, SciPy 1.18,
  scikit-learn 1.9, scikit-image 0.26, matplotlib 3.11, librosa 0.11,
  numba 0.66, h5py 3.16, SoundFile 0.14, and
  iterative-stratification 0.1.9.
- Complete lock: `environment/a2v-analysis.freeze.txt`.

This environment performed dataset/statistical inspection only. None of these
large analysis frameworks is required to import, train, convert, or infer with
the native package.

### Deterministic execution policy

The durable GPU tests set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed 7 for CPU and
all visible CUDA generators, `torch.use_deterministic_algorithms(True)`, cuDNN
benchmarking off, cuDNN deterministic mode on, and both matmul and cuDNN TF32
off. Recipe runs retained their configured seed 1. Comparisons distinguish
three contracts:

1. masks, sampler positions, rank RNG states, checkpoint counters, and durable
   resume state are required to be exact;
2. FP32 and FP16 archived/native tensor comparisons use declared
   absolute/relative tolerances appropriate to the operation and dtype; and
3. finiteness-only checks are used only where the intent is to prove viable
   mixed-precision forward/backward/update execution rather than legacy
   elementwise equivalence.

The effective settings are serialized in
`environment/determinism-environment.json`, and the test fixture that applies
them is `tests/gpu/conftest.py`.

## Acceptance gates

| Gate | Decision | Evidence |
| --- | --- | --- |
| A — package integrity | **PASS** | 171 CPU tests passed; all 8 checked-in YAML recipes load; native source and installed-package scans found none of Fairseq, Hydra, OmegaConf, torchaudio, timm, Transformers, TensorFlow, or the analysis frameworks; `pip check` passed. |
| B — CUDA correctness | **PASS** | 14 GPU-marked tests passed on one A100, covering FP32 CPU/CUDA forward and backward comparisons, FP16 finite forward/backward/update, pretraining, fine-tuning freeze/unfreeze, checkpoint/scaler/RNG resume, and chunked inference. Official native FP32 and FP16 inference was finite. |
| C — official checkpoint parity | **PASS** | Real published pretraining and fine-tuning checkpoints converted strictly. Missing and omitted key counts are zero. Pretraining passed 2,076 comparisons in FP32 and FP16; fine-tuning passed 1,056 in both modes. Native CUDA 13.0 inference loads the converted fine-tuning checkpoint strictly without Fairseq. |
| D — training-step parity | **PASS** | Official FP32 pretraining compared 342 gradients and 342 post-update parameters; maxima were `2.6822090e-7` and `5.9604645e-8`, and optimizer state was exact. Official FP32 fine-tuning compared 305 gradients and 334 post-update parameters; maxima were `4.4703484e-8` and `2.9243529e-7`, and optimizer state was exact. All FP16 comparisons passed their declared tolerances. |
| E — distributed reliability | **PASS** | NCCL collectives and rank-local RNG restore passed on 2 and 4 ranks. Durable 2- and 4-rank uninterrupted/resume tests are exact on every rank. Published-topology 100% fine-tuning and pretraining continuous/resume checkpoints compare bitwise exactly after normalizing only `checkpoint.save_dir`. The unchanged four-rank pretraining recipe fits 40 GiB. |
| F — data/evaluation parity | **PARTIAL / BLOCKED** | The released MeerKAT archive passes structural audit and has recorded hashes. Native and archived event matching/aggregation are exact on nontrivial synthetic fixtures. Official split membership and full-manifest prediction parity cannot be checked because the release contains no official split TSVs; the released event-bearing count is 66,397 versus 66,398 in the paper. |
| G — paper reproduction | **BLOCKED / NOT RUN** | Full 384,230-update MeerKAT pretraining, all five folds at all three reported label fractions, and the Xeno-canto/NIPS4Bplus protocols were not completed. Required authoritative split/supplement/corpus artifacts are unavailable. No paper-reproduction claim is made. |

## Exact recipe burn-ins and resume

The native pretraining run retained the published four-worker topology and
resolved settings:

- `max_tokens=408000`
- `update_freq=[5]`
- `clone_batch=12`
- 16 transformer layers, width 1024, 16 heads
- FP16, initial scale 1.0, minimum scale `1e-6`
- scheduler horizon `max_update=384230`
- 32 data workers

`--stop-at-update` limited only the verification terminal update; it did not
change the scheduler horizon or any published batch, token, clone, model, or
optimizer setting. The first successful update followed five reproducible
overflow retries, reducing the scale from 1.0 to 0.03125. The resumed job
restored 0.03125 and required no repeated retries.

| Run | Final update | Worst peak allocated | Worst peak reserved | Wall time | GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent stop point | 1 | 36,603,244,032 B (34.09 GiB) | 37,346,082,816 B (34.78 GiB) | 298.97 s | 0.3318 |
| Uninterrupted branch | 2 | 39,130,991,104 B (36.44 GiB) | 39,829,110,784 B (37.09 GiB) | 302.31 s | 0.3359 |
| Resumed branch | 2 | 39,130,991,104 B (36.44 GiB) | 39,724,253,184 B (36.99 GiB) | 275.93 s | 0.3064 |

`nvidia-smi` exposes a nominal 40,960 MiB (42,949,672,960-byte) capacity,
whereas PyTorch reports 42,405,855,232 bytes to the process. Against the more
conservative PyTorch-visible value, the worst reserved value leaves
2,576,744,448 bytes (2.40 GiB) of headroom; against the nominal value it leaves
2.91 GiB. This is an exact-recipe burn-in fit result, not evidence that the
full 384,230-update training completed.

The final recursive comparison reports `all_exact: true` for format version,
stage, config after normalizing only the output path, model, EMA teacher,
optimizer, scheduler, scaler, all four rank RNG states, sampler state, epoch,
batch position, best metric, and update counter. The compared update is 2,
the sampler position is epoch 0 / next batch 140, and the restored AMP scale
is 0.03125.

The 100% fine-tuning burn-in retained `max_tokens=426667`,
`update_freq=[9]`, four ranks, 16 layers at width 1024, 20 workers, and the
30,000-update scheduler horizon. Its update-2 continuous/resume comparison is
also bitwise exact. The worst resumed peak was 9,889,550,336 allocated bytes
and 11,049,893,888 reserved bytes. The durable one-A100 test crosses the
frozen/unfrozen boundary; the production recipe's actual boundary is update
10,000 and was not reached in this short burn-in.

The three final pretraining jobs consumed 0.9740 measured GPU-hours. The three
structured fine-tuning stop/continuous/resume jobs consumed 0.2735 GPU-hours,
for 1.2476 GPU-hours across the final six four-rank authority jobs. This total
does not pretend to reconstruct GPU time for earlier failed diagnostics or
short parity utilities that did not emit structured elapsed-time summaries.

For pretraining DDP, the final native implementation explicitly uses
`find_unused_parameters=True` and one 4096 MiB bucket. A default dynamic bucket
topology produced a first divergence of `8.881784197001252e-16`; preserving
only bucket sizes still diverged by `7.275957614183426e-12`. Static-graph DDP
was rejected after a four-rank probe showed zero reduced buckets and
rank-divergent optimizer state. The final topology makes the durable two- and
four-rank resume tests exact.

The archived Fairseq exact recipe was also measured without changing its
settings. It OOMed on every rank near 40 GB, with rank 0 reaching
40,067,256,320 allocated bytes and 40,481,325,056 reserved bytes before a
436 MiB request at `student.15`. This is retained as a diagnostic result and
is not used to weaken the native recipe.

## Official checkpoint conversion and numerical parity

The official checkpoint hashes and all converted/final checkpoint hashes are
recorded in
`results/gpu_verification/20260723T081428Z/checkpoints/checkpoint-sha256-final.txt`.
The published raw checkpoint hashes are:

- pretraining:
  `c048af7f2fb5c149fcc888a4a982a947e92bba0ea657dee82229fe1f4264d1de`
- fine-tuning:
  `2846cefa9e1d696a95d91f9ae316456895eb3075eaafc5e81b3d20eb9c4bb712`

The archived environment exported plain tensor/config payloads before native
strict conversion. The pretraining source contains 343 model entries and 301
EMA entries at update 408,402. The fine-tuning source contains 334 model
entries at update 30,000. All floating tensors are finite.

Key parity results:

| Check | FP32 | FP16 |
| --- | --- | --- |
| Official pretraining | 2,076/2,076 pass; worst absolute error `7.1525574e-7` in normalized targets | 2,076/2,076 pass; loss delta 0.4453125 at roughly `9.70e-6` relative error |
| Official fine-tuning | 1,056/1,056 pass; worst absolute error `1.9073486e-6` in training logits | 1,056/1,056 pass; loss delta 0.0068359375 at roughly `5.46e-6` relative error |
| Native official inference | finite `[2048, 12]` probabilities and `[2048, 1024]` embeddings | finite `[2048, 12]` probabilities and `[2048, 1024]` embeddings |

Large relative-error maxima in a few FP16/near-zero tensor elements are not
decision changes; every comparison passed its declared absolute/relative
tolerance, and FP32 optimizer state comparisons are exact.

## Dataset and evaluator audit

The released archive is Edmond DOI `10.17617/3.0J0DYB`, archive MD5
`6a5065ddda33d2bfaad118a418d9b965`. Exhaustive audit results:

- 384,592 WAV files and 384,592 matching HDF5 label files
- 30,767,360,000 frames at 8 kHz, or 1,068.311111 hours
- 353,319 mono PCM16 files and 31,273 mono PCM24 files
- no unreadable audio, missing pairs, out-of-bounds labels, or structural
  HDF5 failures
- 66,397 event-bearing files, 254,069 events, and 123,114 focal events
- 318,195 empty-label files
- one fewer event-bearing file than the paper's stated 66,398 labelled samples

The public archive contains no official split manifests. Regenerated
seed-1612 manifests were created only to exercise batching, memory, and resume
paths. They pass structural checks and have recorded raw/membership hashes,
but they are explicitly non-authoritative because the archived generator
depends on input ordering and the paper does not publish the exact membership.

The pure-PyTorch event evaluator was checked against the actual archived
evaluator for average pooling with window sizes 1 and 3 and max pooling with
window size 3. Scores, segmented targets, IoUs, split markers, merger markers,
classwise AP, macro AP, micro AP, focal threshold, precision, recall, and F1
are all exactly equal. The nontrivial fixture produces macro AP
0.8472222222222222, micro AP 0.7807823129251701, and focal F1 2/3.

## Defects found and narrow corrections

Each correction has a focused regression and retained red/green logs:

| Area | First observed divergence or failure | Correction |
| --- | --- | --- |
| Sinc frontend | deterministic CUDA reflection padding failed | deterministic explicit padding path |
| AMP | wrong legacy defaults and overflow advancement | restored scale semantics; skipped updates no longer advance mutable training state |
| Checkpoint RNG | CUDA RNG payloads restored on the wrong device | map-location-aware RNG serialization/restore |
| Positional path | first mismatch after masking | restored archived mask/positional ordering |
| Decoder FP16 | normalization dtype mismatch | archived FP32 normalization under autocast |
| Conversion | official update count read from the wrong legacy location | strict optimizer-history update extraction |
| Configuration | missing `norm_eps` and event IoU setting | explicit parsed fields with regression coverage |
| Inference | final partial segment emitted extra frames/timestamps | convolution-aware final trim |
| DataLoader resume | worker prefetch advanced the saved sampler cursor | track batches delivered to training |
| Epoch resume | DataLoader generator state changed the next epoch | deterministic per-rank loader generator ownership |
| ALiBi memory | full clone-expanded bias consumed about 15.36 GB | gather the needed bias before clone scaling |
| Resume memory | loaded checkpoint retained duplicate CUDA model tensors | restore then release checkpoint payload |
| DDP resume | dynamic bucket rebuild changed reduction order | explicit single pretraining bucket with unused-parameter traversal |
| Event AP | tied scores differed from archived/scikit-learn semantics | group equal thresholds before reverse integration |

The static-graph attempt and intermediate bucket experiments are retained as
failed diagnostics, not reported as passing gates.

## Remaining blockers

The following artifacts or work are required before Gate F/G can pass:

1. Exact publisher/paper split manifests for all five folds and reported label
   fractions, including their authoritative membership hashes.
2. Publisher supplement `mee370218-sup-0001-supinfo.pdf`; the publisher
   endpoint returned HTTP 403 during verification.
3. Resolution of the released 66,397 versus reported 66,398 event-bearing
   sample discrepancy and the release's 31,273 PCM24 files versus the paper's
   16-bit description.
4. The exact Xeno-canto subset, exclusions, hashes, and preprocessing
   provenance.
5. The NIPS4Bplus sequence and framewise protocol artifacts.
6. Completion of the full 384,230-update pretraining run, all five folds at
   1%, 25%, and 100%, and the downstream event metrics.

Until these are supplied and the full jobs complete, the valid conclusion is
GPU, checkpoint, numerical, training-step, and distributed parity—not paper
reproduction.

## Principal artifacts

- CPU suite:
  `logs/native-cu130-pytest-final2.log`
- one-A100 suite:
  `logs/gpu0-final-suite2.log`
- final pretraining memory:
  `metrics/exact-pretrain-ddp4-final-memory.json`
- final pretraining resume comparison:
  `metrics/exact-pretrain-ddp4-final-resume-comparison.json`
- final fine-tuning resume comparison:
  `metrics/exact-finetune100-ddp4-final-resume-comparison.json`
- official pretraining FP32 parity:
  `parity/official-pretrain-fp32/native-after-alibi-fix/comparison-report.json`
- official fine-tuning FP32 parity:
  `parity/official-finetune-fp32/native/comparison-report.json`
- dataset audit:
  `data/official-dataset-audit.json`
- manifest audit and hashes:
  `data/regenerated-manifest-audit.json`,
  `data/regenerated-manifest-sha256.txt`
- event evaluator parity:
  `parity/event-evaluator-synthetic/comparison-report.json`
- machine-readable decision:
  `verification-report.json`

## Detailed execution record

The preceding sections are the decision summary. This section records the
work at a level intended to support independent review and repetition.

### Verification sequence

The work proceeded in the following dependency order so that later GPU claims
were never used to hide an earlier package, import, or CPU regression:

1. Capture the untouched host, repository head/status/diff, GPU inventory,
   driver, topology, filesystems, process list, and host Python/PyTorch state.
2. Build the four isolated interpreters/environments prescribed by the A100
   bootstrap guide and freeze each environment.
3. Run the pre-change 135-test CPU suite on the untouched host runtime and in
   the native CUDA 12.6 parity runtime.
4. Exercise the configuration loader against all eight checked-in YAML files
   and audit the clean native runtime for forbidden framework dependencies.
5. Add deterministic GPU-marked tests, reproduce failures at the smallest
   practical level, fix them narrowly, and retain red/green evidence.
6. Export official checkpoints in the archived Fairseq environment, convert
   them in the native runtime, and perform same-GPU layer, output, gradient,
   optimizer, and post-update comparisons in FP32 and FP16.
7. Prove basic NCCL collectives and rank-local RNG restoration on two ranks and
   then on the required four-rank topology.
8. Establish durable uninterrupted/resumed equivalence on small two- and
   four-rank jobs before measuring the exact published configurations.
9. Run exact-configuration four-rank stop, uninterrupted, and resumed
   pretraining and 100% fine-tuning burn-ins on GPUs 0–3, recording per-rank
   peaks and complete checkpoint state comparisons.
10. Audit the complete released MeerKAT corpus, regenerate clearly marked
    non-authoritative manifests, and compare the native event evaluator with
    the archived implementation.
11. Re-run the entire native CPU suite and the complete one-A100 GPU-marked
    suite after all corrections, validate artifact references, and run
    `git diff --check`.

No published batch, token, clone, layer, width, head, worker, update-frequency,
optimizer, scheduler, or four-rank setting was silently reduced. When a job
needed only a bounded burn-in, `--stop-at-update` changed the terminal update
only; the original scheduler horizon remained in the resolved configuration
and checkpoint.

### Gate A: package and CPU integrity

Three CPU checkpoints were useful during the work:

| Runtime / point | Result | Elapsed | Purpose |
| --- | ---: | ---: | --- |
| untouched host PyTorch 2.11 | 135 passed | 9.99 s | record the starting source baseline without modifying host packages |
| native parity CUDA 12.6 environment | 135 passed | 9.76 s | establish the same source baseline in an isolated native runtime |
| final native CUDA 13.0 environment | 171 passed | 22.97 s | cover all original and newly added regressions after GPU work |

The final fresh log is
`logs/native-cu130-pytest-completion.log`; an earlier complete final rerun is
retained as `logs/native-cu130-pytest-final2.log`. All eight repository YAML
recipes loaded successfully, recorded in
`configs/all-yaml-load-final.json`. The final native `pip check` was clean and
the source/import scan found none of the forbidden runtime dependencies.

The increase from 135 to 171 tests is attributable to focused integration and
unit coverage for the issues found during verification. It is not a change in
how pytest counted the original suite. The added coverage includes:

- stop-at-update behavior without scheduler-horizon mutation;
- consumed-batch checkpointing in the presence of worker prefetch;
- multiworker and next-epoch exact resume;
- inference trimming of a final partial chunk;
- checkpoint conversion update extraction;
- AMP default/overflow/skip semantics;
- map-location-safe rank RNG state;
- positional-mask ordering and memory-bounded ALiBi bias construction;
- event AP ties and event-interval matching semantics; and
- first-diverging-value checkpoint diagnostics.

### Gate B: one-A100 CUDA correctness

The final marker-selectable suite ran on physical GPU 0 in the clean native
authority environment and completed with 14 passed in 4.06 s. The fresh log is
`logs/gpu0-completion-suite.log`; the earlier complete final log is
`logs/gpu0-final-suite2.log`. The cases are durable repository tests, not
one-off notebooks.

#### Component tests: `tests/gpu/test_cuda_components.py`

| Case | Comparison contract | Outcome |
| --- | --- | --- |
| Sinc FP32 | CPU/CUDA forward `atol=1e-5`, `rtol=1e-4`; input and parameter gradients `atol=2e-5`, `rtol=2e-4` | pass |
| Sinc FP16 | finite CUDA forward, backward, and optimizer update | pass |
| frontend FP32 | local CPU/CUDA tensors `1e-5/1e-4`; context tensors `2e-5/2e-4`; finite gradients with padding present | pass |
| ALiBi attention FP32 | output `1e-5/1e-4`; gradient `2e-5/2e-4` | pass |
| ALiBi attention FP16 | FP16 output dtype and finite backward gradient | pass |
| decoder autocast | decoder-block normalization output remains FP32 as archived; decoder output returns FP16 | pass |
| mixup and clones | masks replay exactly under restored RNG; clone masks remain independently sampled | pass |

#### Training tests: `tests/gpu/test_cuda_training.py`

| Case | Assertion | Outcome |
| --- | --- | --- |
| tiny FP32 pretraining CPU/CUDA | identical masks and sample size; predictions/targets within `3e-5/3e-4`; loss within `2e-4/3e-4`; finite teacher updates | pass |
| AMP checkpoint continuation | save on CPU, restore on GPU, execute the next update, and compare the complete mutable state bitwise | pass |
| direct CUDA `map_location` | CPU and all relevant CUDA RNG state restore exactly | pass |
| fine-tuning freeze boundary | frozen transformer state remains unchanged before unfreeze, transformer updates after unfreeze, and local encoder remains frozen according to the recipe | pass |
| overflow skip | non-finite update skips optimizer, scheduler, update counter, and EMA; scale falls from 128 to 64; next finite update advances normally | pass |

#### Inference tests: `tests/gpu/test_cuda_inference.py`

One test feeds stereo audio requiring resampling and chunking, including a
final partial segment. It requires finite bounded output, timestamps no later
than the actual recording, and events bounded by the recording duration. The
second test exercises an empty recording and requires correctly shaped empty
probability, embedding, timestamp, and event outputs. Both passed.

The official converted fine-tuning model was then exercised separately on
GPU 0. A 10.25 s, 16 kHz input was resampled to 8 kHz and processed with a
10 s segment size. Strict checkpoint loading produced 2,048 frames of 12-class
probabilities and 1,024-dimensional embeddings. The last timestamp was
10.245187759 s and all values were finite. FP32 probability extrema were
0.0036694 and 0.2883053 with a 2,299,984,384-byte peak allocation; FP16
extrema were 0.00369263 and 0.2890625 with a 2,297,108,992-byte peak.

### Gates C and D: official checkpoint conversion and same-GPU parity

#### Publisher provenance and export boundary

The official model files were obtained from the publisher record and retained
unaltered under `checkpoints/official/`. Their publisher and locally computed
identifiers are:

| Checkpoint | Bytes | Publisher MD5 | Local SHA-256 |
| --- | ---: | --- | --- |
| `animal2vec_large_pretrained_MeerKAT_240507.pt` | 5,024,620,014 | `c0ae0cb16afd0501f00a5955fb6482ed` | `c048af7f2fb5c149fcc888a4a982a947e92bba0ea657dee82229fe1f4264d1de` |
| `animal2vec_large_finetuned_MeerKAT_240507.pt` | 22,883,417,973 | `b377ea79700f3bbc98b6154f21545158` | `2846cefa9e1d696a95d91f9ae316456895eb3075eaafc5e81b3d20eb9c4bb712` |

Because the source checkpoints require Fairseq-era classes to deserialize,
they were opened only in `a2v-legacy-parity-cu126`. The exporter wrote plain
tensor/config payloads, creating a narrow and inspectable boundary between the
archived framework and the native converter. The pretraining export is
2,497,766,816 bytes and contains 343 model entries plus 301 EMA entries,
643 unique tensors, and legacy update 408,402. The fine-tuning export is
1,255,926,694 bytes and contains 334 model entries at update 30,000. A complete
finiteness scan passed for both payloads.

The native converter then performed strict mapping. Both conversions report
zero missing native keys, zero unmapped legacy keys, and zero silently omitted
keys. The converted SHA-256 values are:

- pretraining native:
  `e5cb00a82b2aac92b98e38e77ba377b36a6a687b0b1f857ccbbb2810d1a41d08`;
- fine-tuning native:
  `b95db2dfa6e13ab9e3f02f593bcc461d67664388d9f3a64ada34b492e65e9795`.

The report distinguishes conversion correctness from numerical correctness.
Conversion proves that the state dictionaries and configuration metadata map
completely. Numerical parity additionally loads archived and native models on
the same physical GPU, feeds equivalent inputs and RNG-controlled masks, and
compares intermediate/output/training state.

#### Comparison policy

All official comparisons used GPU 0, seed 7, deterministic algorithms, and
TF32 disabled. The broad FP32 policy was `atol=1e-5`, `rtol=1e-4`; the broad
FP16 policy was `atol=2e-3`, `rtol=1e-2`, with explicit per-scalar overrides
where a reduction has a known accumulation scale. Masks, state layout,
configuration mappings, and counters were exact. A high relative error beside
a very small absolute error was not treated as a failure if the declared
combined tolerance passed; this matters for near-zero target entries in FP16.

Every comparison report stores the name, shape, dtype, maximum absolute and
relative errors, tolerances, exactness flag, and pass/fail status. Thus the
summary maxima below are not the only retained data.

#### Official pretraining parity

Both FP32 and FP16 reports contain 2,076 passing comparisons:

| Family | Count | FP32 maximum / result | FP16 maximum / result |
| --- | ---: | --- | --- |
| conversion tensors | 643 | exact | exact |
| EMA teacher tensors | 602 | exact | exact |
| masked layout/state | 63 | exact | exact |
| unmasked layout/state | 76 | exact | exact |
| gradients | 342 | max abs `2.6822090e-7` | max abs `7.15315e-4` |
| optimizer values | 3 | exact | max abs `1.04141e-3` |
| post-update parameters | 342 | max abs `5.9604645e-8` | max abs `4.65866e-5` |
| pretraining outputs/reductions | 5 | target max abs `7.1525574e-7` | target max abs `4.51267e-3`; loss delta `0.4453125` at approximately `9.696e-6` relative |

The FP32 gradient and optimizer-update comparison is the decisive Gate D
evidence: 342 gradients were compared before stepping; then the optimizer was
stepped once and 342 model parameters plus optimizer state were compared. This
detects implementations that happen to agree in forward inference but diverge
in backward equations or update semantics.

Evidence directories:

- `parity/official-pretrain-fp32/native-after-alibi-fix/`;
- `parity/official-pretrain-fp16/native-after-alibi-fix/`;
- the archived reference payloads in the adjacent legacy export directories.

#### Official fine-tuning parity

Both FP32 and FP16 reports contain 1,056 passing comparisons:

| Family | Count | FP32 maximum / result | FP16 maximum / result |
| --- | ---: | --- | --- |
| conversion tensors | 334 | exact | exact |
| layer outputs | 59 | max abs `1.9073486e-6` | max abs `3.90625e-3` |
| inference outputs | 18 | max abs `1.9073486e-6` | max abs `3.90625e-3` |
| gradients | 305 | max abs `4.4703484e-8` | max abs `1.24991e-4` |
| optimizer values | 3 | exact | max abs `4.5776e-5` |
| post-update parameters | 334 | max abs `2.9243529e-7` | max abs `5.3925e-5` |
| training outputs/reductions | 3 | max abs `1.9073486e-6` | loss delta `6.8359375e-3` |

The frozen/unfrozen unit-level GPU test complements this official one-update
comparison. The official short parity update proves archived/native gradient
and update equivalence at one state; the durable boundary test proves the
native trainer changes exactly the intended parameter groups when fine-tuning
crosses its freeze boundary.

Evidence directories:

- `parity/official-finetune-fp32/native/`;
- `parity/official-finetune-fp16/native/`.

#### Interpretation of checkpoint file hashes

The uninterrupted and resumed native checkpoint files have different binary
SHA-256 values even when their logical state is exact. The causes are the
intentionally different `checkpoint.save_dir` value and serialization details.
Accordingly, resume acceptance uses the recursive semantic comparator, which
normalizes only that output path and then compares every remaining tensor,
number, sequence, mapping key, RNG payload, and sampler field. Binary hashes
remain useful for artifact identity and are recorded separately; they are not
substituted for logical state comparison.

### Gate E: NCCL, distributed state, exact recipe fit, and resume

#### Collective and RNG probes

The NCCL smoke test ran first with two ranks and then with the prescribed four
ranks. It exercised all-reduce, broadcast, all-gather, synchronization, and
rank-local RNG checkpoint/restore, rather than treating process-group creation
alone as proof of communication.

| Topology | Visible GPUs | All-reduce sum | Broadcast value | Gathered rank values | RNG restore |
| --- | --- | ---: | ---: | --- | --- |
| 2 ranks | 0,1 | 3 | 97 | `[0, 1]` | exact on both ranks |
| 4 ranks | 0,1,2,3 | 10 | 97 | `[0, 1, 2, 3]` | exact on all ranks |

Both probes used PyTorch 2.12.1+cu130 and NCCL 2.29.7. Evidence is in
`parity/nccl-2rank-postfix/` and `parity/nccl-4rank-postfix/`. The durable
trainer-shaped resume probes in
`parity/resume-2rank-single-bucket-find-unused/` and
`parity/resume-4rank-single-bucket-find-unused/` subsequently reported exact
uninterrupted/resumed state on every rank.

#### Resolved pretraining configuration

The authoritative pretraining configuration is serialized in
`configs/official-pretrain-resolved-native-final.json`. Material settings were:

| Group | Resolved value |
| --- | --- |
| topology | 4 ranks; `legacy_ddp`; GPUs 0–3 |
| common | seed 1; FP16; initial scale 1.0; minimum scale `1e-6`; log interval 384 |
| batching | `max_tokens=408000`; `update_freq=[5]`; 32 workers; batch multiple 1 |
| schedule | 384,230 maximum updates; learning rate `1e-4`; cosine; 10,000 warmup updates; minimum LR 0 |
| optimizer | Adam, betas `(0.9, 0.98)`, epsilon `1e-6`, weight decay 0.01, clip norm 1.0 |
| audio/task | 8 kHz, normalization on, no padding, no maximum-sample truncation |
| transformer | 16 layers, width 1,024, 16 heads, MLP ratio 4 |
| clones/targets | clone batch 12; average top 16 target layers |
| frontend/decoder | Sinc frontend; 8-layer prenet; P-Swish; ALiBi; 4 decoder layers, width 768, 16 groups, kernel 7 |
| masking/mixup | time mask probability 1.5, length 2; source mixup 0.5, probability 1.0, same ratio, A-weighting |
| teacher | EMA 0.9997 annealing to 1.0 by update 300,000 |
| normalization | explicit `norm_eps=1e-5` |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was used to reduce allocator
fragmentation. It changes allocator behavior, not model, batch, token, clone,
optimizer, or update semantics.

The short terminal sequence was deliberate:

1. an independent branch stopped immediately after successful update 1 and
   preserved `checkpoint_update1.pt`;
2. an uninterrupted branch trained from initialization through update 2; and
3. a resumed branch loaded the independent update-1 checkpoint and trained
   through update 2.

Five deterministic overflow retries occurred before the first successful
pretraining update, taking the scale through
`1 → 0.5 → 0.25 → 0.125 → 0.0625 → 0.03125`. None of these attempts advanced
the optimizer, scheduler, update counter, EMA teacher, sampler-consumed cursor,
or checkpoint cadence. The resumed branch restored scale 0.03125 and therefore
did not replay those retries.

#### Per-rank pretraining memory and elapsed time

The following values come from the structured rank summaries collected in
`metrics/exact-pretrain-ddp4-final-memory.json`:

| Branch | Rank | Elapsed (s) | Peak allocated (B) | Peak reserved (B) |
| --- | ---: | ---: | ---: | ---: |
| stop after update 1 | 0 | 298.967 | 36,603,244,032 | 37,304,139,776 |
| stop after update 1 | 1 | 298.416 | 36,287,789,568 | 37,199,282,176 |
| stop after update 1 | 2 | 298.538 | 35,977,142,784 | 37,094,424,576 |
| stop after update 1 | 3 | 298.421 | 36,499,162,624 | 37,346,082,816 |
| uninterrupted update 2 | 0 | 302.312 | 38,918,665,728 | 39,346,765,824 |
| uninterrupted update 2 | 1 | 302.280 | 39,130,991,104 | 39,829,110,784 |
| uninterrupted update 2 | 2 | 302.305 | 39,025,803,776 | 39,535,509,504 |
| uninterrupted update 2 | 3 | 302.276 | 38,400,347,648 | 39,451,623,424 |
| resumed update 2 | 0 | 275.642 | 38,918,665,728 | 39,724,253,184 |
| resumed update 2 | 1 | 275.632 | 39,130,991,104 | 39,535,509,504 |
| resumed update 2 | 2 | 275.927 | 39,025,803,776 | 39,537,606,656 |
| resumed update 2 | 3 | 275.845 | 38,400,347,648 | 39,430,651,904 |

The worst allocator reservation, 39,829,110,784 bytes, remains below both the
42,405,855,232-byte PyTorch-visible capacity and the 42,949,672,960-byte
nominal capacity. The conservative framework-visible margin is 2,576,744,448
bytes. The test therefore answers the requested question—whether the exact
recipe can execute on a 40 GB A100—without changing the recipe to make it fit.
It does not promise immunity to future fragmentation, unrelated resident
processes, or memory growth not exercised by the two-update path.

The final recursive report
`metrics/exact-pretrain-ddp4-final-resume-comparison.json` has
`all_exact: true`. After normalizing only `checkpoint.save_dir`, it matched:

- format version, stage, and resolved configuration;
- every student and EMA teacher tensor;
- optimizer tensor/scalar state and parameter-group metadata;
- scheduler and GradScaler state;
- all four rank-local CPU/CUDA RNG payloads;
- sampler epoch and next-consumed position;
- training epoch, delivered batch position, best metric, and update counter.

Both branches ended at update 2, epoch 1, `batch_in_epoch=35`, sampler epoch 0,
sampler next position 140, and AMP scale 0.03125.

#### Resolved 100% fine-tuning configuration

The fine-tuning authority configuration is
`configs/official-finetune100-resolved-native-run.json`. The active settings
retained four ranks, `max_tokens=426667`, `update_freq=[9]`, 20 workers,
batch-size multiple 8, FP16 with scale 128, and a 30,000-update cosine horizon.
It used Adam with betas `(0.9, 0.98)`, epsilon `1e-8`, zero weight decay,
learning rate `3e-5`, 2,000 warmup updates, warmup initial LR `1e-10`, minimum
LR `5e-6`, and no gradient clipping. Audio was 8 kHz with sample bounds
1,000–320,000 frames and `min_label_size=3032`.

The model retained the 16 × 1,024 × 16 transformer, 12 labels, clone batch 1,
feature gradient multiplier 0, and a 10,000-update transformer freeze. Time
masking was probability 0.825/length 4; channel masking was probability
0.5/length 64. Source and target mixup were enabled. The focal criterion used
label smoothing 0.09 and metric threshold 0.175.

Three initial overflows reduced scale 128 to 16 before successful updates. The
structured worst-rank summaries were:

| Branch | Terminal update | Worst elapsed (s) | Peak allocated (B) | Peak reserved (B) | Measured GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| uninterrupted | 2 | 87.039 | 8,631,396,352 | 9,799,991,296 | 0.09606 |
| independent resume source | 1 | 85.577 | 8,631,198,720 | 9,797,894,144 | 0.09496 |
| resumed | 2 | 74.400 | 9,889,550,336 | 11,049,893,888 | 0.08250 |

`metrics/exact-finetune100-ddp4-final-resume-comparison.json` reports all
logical state exact. This two-update burn-in cannot exercise the recipe's
production transformer-unfreeze point at update 10,000; that transition is
covered by the focused one-A100 durable training test.

#### DDP reduction-order investigation

The demand for exact resume exposed a very small but real reduction-order
effect. Under the initial dynamic DDP setup, the first uninterrupted/resumed
model divergence was
`model.student.local_encoder...p_swish_beta` at
`8.881784197001252e-16`; the associated optimizer `exp_avg` differed by about
`5.68e-14`, and a teacher value by about `4.34e-19`. These values were too
small to affect ordinary tolerance tests but violated the explicit bitwise
resume contract.

The investigation followed the first-diverging tensor rather than accepting a
looser threshold:

1. Default DDP began with one 201,326,592-byte bucket and rebuilt into three
   67,108,864-byte buckets after observing the graph.
2. An explicit 25 MiB cap produced three 67,108,864-byte buckets before and
   after rebuild, yet traversal/reduction order still drifted. Every rank first
   diverged in a normalization bias at `7.275957614183426e-12`.
3. A single 4,096 MiB bucket with `find_unused_parameters=False` removed the
   size change but not the state drift.
4. Static-graph DDP was rejected. A small model probe found no ordinary unused
   parameters, but the four-rank production probe reported zero reduced
   buckets, zero backward communication time, rank-divergent optimizer moments,
   and second-update parameter divergence. A rank-0-only checkpoint comparison
   would therefore have produced a misleading pass.
5. The final pretraining setting combines one 4,096 MiB bucket with
   `find_unused_parameters=True`. It keeps traversal/reduction stable, and both
   two- and four-rank durable comparisons plus the exact recipe comparison are
   bitwise exact on every rank. The emitted “no unused parameters” warning is
   expected overhead, not a failed correctness assertion.

Fine-tuning also uses unused-parameter traversal because the backbone is
intentionally frozen during the early phase. The failed/default/static
experiments and their comparison reports remain in `metrics/` and `parity/` so
the successful setting is auditable rather than presented without its
diagnostic history.

#### Archived exact-recipe memory diagnostic

For context only, the archived Fairseq implementation was run with its exact
recipe under PyTorch 2.6.0+cu126. It OOMed on every rank. Rank 0 reached
40,067,256,320 allocated and 40,481,325,056 reserved bytes, then failed a
436 MiB allocation after the last completed `student.15` stage and before the
decoder. This result did not authorize a batch/model reduction and was not
used as a substitute for the native fit gate. Its purpose was to identify the
memory regression boundary and validate the native ALiBi correction.

### Gate F: released data, manifests, and event metrics

#### Dataset acquisition and exhaustive structural audit

The available official corpus is the Edmond release at DOI
`10.17617/3.0J0DYB`. The downloaded
`data/MeerKAT_10s_2024-06-12.zip` is 24,844,330,626 bytes and has publisher
MD5 `6a5065ddda33d2bfaad118a418d9b965`. Publisher metadata, licensing text, and
release/version information are retained in
`environment/official-dataset-metadata.json`; the local MD5 file is
`data/MeerKAT_10s_2024-06-12.zip.md5`.

The audit was exhaustive, not sampled. `scripts/audit_meerkat_dataset.py`
walked every WAV/HDF5 pair, validated decodability, channel count, rate, frame
count, label key set, array lengths/dtypes, frame/time bounds, focal/category
content, and stem pairing. The result
`data/official-dataset-audit.json` reports
`all_structural_checks_pass: true` and zero structural errors.

| Corpus property | Audited result |
| --- | ---: |
| WAV files | 384,592 |
| HDF5 files | 384,592 |
| matched stems | 384,592 |
| sample rate | 8,000 Hz for every file |
| channels | mono for every file |
| frames per file | 80,000 |
| total frames | 30,767,360,000 |
| total duration | 1,068.311111 hours |
| PCM16 files | 353,319 |
| PCM24 files | 31,273 |
| unreadable audio | 0 |
| missing WAV/HDF5 pairs | 0 |
| empty-label files | 318,195 |
| event-bearing files | 66,397 |
| total events | 254,069 |
| focal events | 123,114 |
| minimum start frame / maximum end frame | 0 / 80,000 |
| maximum time-versus-frame error | 0.000125 s |

Every HDF5 file has the same seven keys: `start_frame_lbl`, `end_frame_lbl`,
`start_time_lbl`, `end_time_lbl`, `lbl`, `lbl_cat`, and `foc`. Empty files use
empty float64 arrays. Event-bearing files use integer frame/category/focal
arrays, float64 time arrays, and object label arrays. This dtype distinction is
consistent across all 384,592 files and is recorded rather than normalized
away.

Event counts by published label are:

| Category | Events |
| --- | ---: |
| `beep` | 2,155 |
| `synch` | 9,989 |
| `sn` | 67,387 |
| `cc` | 111,728 |
| `ld` | 832 |
| `oth` | 13,946 |
| `mo` | 5,306 |
| `al` | 8,333 |
| `soc` | 24,219 |
| `agg` | 6,395 |
| `eating` | 3,779 |
| **Total** | **254,069** |

Two publication-level discrepancies remain. The release contains 66,397
event-bearing recordings, one fewer than the paper's 66,398 labelled-sample
statement. It also contains 31,273 PCM24 recordings despite the paper's 16-bit
description. Neither discrepancy is a structural corruption, but both prevent
an unqualified claim that the local corpus is byte-for-byte the paper's exact
experimental input.

#### Regenerated manifests: useful but non-authoritative

The publisher release contains no official pretraining/fold/few-shot TSVs.
The archived manifest generator was therefore run with seed 1612 to unblock
data loading, distributed resume, exact recipe memory, and evaluator work. The
generator depends on filesystem/input ordering, so matching its seed does not
establish official membership. The generated files are explicitly isolated
under `data/manifests/regenerated-seed1612/` and must not be renamed or cited as
publisher splits.

The pretraining manifest contains all 384,592 recordings. The labelled
universe contains all 66,397 event-bearing recordings. Within each fold,
training and validation are disjoint and their union is the complete labelled
universe:

| Fold | Training rows | Validation rows | Within-fold overlap | Union |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 53,057 | 13,340 | 0 | 66,397 |
| 1 | 53,133 | 13,264 | 0 | 66,397 |
| 2 | 53,163 | 13,234 | 0 | 66,397 |
| 3 | 53,074 | 13,323 | 0 | 66,397 |
| 4 | 53,133 | 13,264 | 0 | 66,397 |

The generator also produced declared 1%, 10%, 25%, 50%, and 75% training
subsets. Because multilabel stratification and rounding determine actual row
counts, the reported fractions are approximate:

| Fold | 1% rows | 10% rows | 25% rows | 50% rows | 75% rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 539 | 5,331 | 13,291 | 26,581 | 39,750 |
| 1 | 522 | 5,301 | 13,306 | 26,600 | 39,902 |
| 2 | 526 | 5,293 | 13,249 | 26,572 | 39,899 |
| 3 | 531 | 5,294 | 13,239 | 26,556 | 39,772 |
| 4 | 527 | 5,275 | 13,301 | 26,555 | 39,910 |

Every few-shot subset is contained in its parent training fold. All raw file
and normalized membership hashes are preserved in
`data/regenerated-manifest-sha256.txt` and
`data/regenerated-manifest-audit.json`. Cross-fold validation membership is not
a classical five-way partition: validation sets overlap across folds and
21,826 labelled recordings never appear in a regenerated validation set. This
reflects the archived split-generation procedure and is an additional reason
not to treat these regenerated manifests as authoritative paper folds.

An empty HDF5 file is 3,208 bytes, greater than the fine-tuning
`min_label_size=3032` threshold. That observation does not contaminate the
regenerated fine-tuning manifests: the archived generator reads `lbl_cat` and
emits only event-bearing files. The report records the size edge case because a
different manifest builder that filters solely by file size could behave
incorrectly.

#### Native event-evaluation semantics

The native evaluator in `src/animal2vec/evaluation/events.py` is implemented
with PyTorch and Python standard-library facilities; it does not pull pandas,
SciPy, scikit-learn, or another analysis stack into the runtime. Its archived
behavioral contract is covered explicitly:

- event intervals are handled with the archived inclusive-boundary convention;
- a single-frame label is converted using an exclusive `end + 1` bound where
  required by segmentation;
- candidate intersection must be strictly positive and event acceptance uses
  strict `IoU > threshold`, including strict-boundary test cases;
- average and maximum sliding pooling use stride 1, right padding, and the
  archived `round(window/2)` temporal shift;
- classes are evaluated independently, including recordings containing only
  unmatched predictions or only unmatched targets;
- split and merger markers and interval-level mean scores follow the archived
  implementation;
- average precision groups equal-score thresholds before reverse integration,
  matching the scikit-learn behavior used by the archive; and
- macro/micro metrics exclude the focal channel, while focal precision,
  recall, and F1 use the best unique focal threshold.

Ten focused unit scenarios cover perfect prediction, class aggregation and
focal exclusion, missing-prediction and missing-target recordings, splits,
mergers, strict IoU boundaries, average/maximum pooling with a partial final
window, overlapping classes, and all-negative input.

For the independent archived/native comparison, the legacy environment
exported the actual archived evaluator result and the native environment read
that neutral payload. Three nontrivial modes were compared: average pooling
window 1, average pooling window 3, and maximum pooling window 3. Scores,
segmented targets, IoUs, split flags, merger flags, per-class AP, macro AP,
micro AP, focal threshold, precision, recall, and F1 were exact in all modes.

The controlled fixture produced class AP values 1.0 and
0.6944444444444444, focal AP 0.5, macro AP 0.8472222222222222, micro AP
0.7807823129251701, focal threshold 0.949999988079071, precision 0.5, recall
1.0, and F1 0.6666666666666666. The comparison report is
`parity/event-evaluator-synthetic/comparison-report.json`.

#### Why Gate F is not a pass

Synthetic evaluator exactness proves the code path but not membership or
paper-level metric reproduction. Without the official split manifests, the
same recordings cannot be guaranteed to enter the same folds and few-shot
subsets. Without official full-manifest predictions or completed authoritative
training, event-metric parity cannot be demonstrated on the paper inputs.
Gate F therefore remains **partial / blocked**, even though the independently
feasible corpus-structure and evaluator-algorithm work passed.

## Forensic defect record

The concise defect table earlier in this report identifies the corrected
areas. This section records the symptom, first useful divergence, root cause,
narrow correction, and durable evidence for each one. Failed experiments were
retained so a future rerun can distinguish rediscovery from regression.

### 1. Deterministic Sinc reflection padding

**Symptom.** The one-A100 Sinc backward path failed when deterministic CUDA
algorithms were enabled. The failure occurred at reflection padding, before a
meaningful numerical gradient comparison could complete.

**Root cause.** The generic CUDA reflection-padding backward kernel used by the
original native expression is not deterministic under PyTorch's strict
determinism policy, even though the mathematical padding rule itself is
deterministic.

**Correction.** `src/animal2vec/modules/sinc.py` now builds the required
reflected boundary explicitly from deterministic slice/flip/concatenation
operations. The filter equations, parameterization, output shape, and padding
values are unchanged.

**Evidence.** The focused GPU regression compares CPU/CUDA forward, input
gradient, and parameter gradients and also exercises FP16 update viability.
See `logs/gpu0-sinc-deterministic-regression.log` and the final component/GPU
suite logs.

### 2. Legacy AMP defaults and skipped-update semantics

**Symptom.** Resolved configurations did not preserve the checkpoint/recipe
initial loss scale, and an FP16 overflow could advance mutable training state
despite the optimizer update being skipped. This makes an interrupted branch
and a resumed branch consume different state transitions.

**First divergence.** The focused overflow test observed optimizer/scheduler,
update counter, or EMA progression on a non-finite attempt rather than a pure
scale reduction.

**Root cause.** Scale parameters were not first-class parsed configuration
fields, and the training engine treated a scaler-controlled no-op too much like
a successful step.

**Correction.** `src/animal2vec/config.py` now parses
`common.fp16_init_scale` and `common.min_loss_scale`.
`src/animal2vec/training/engine.py` returns `UpdateResult.skipped`, allows
`GradScaler` to skip and reduce the scale, prevents optimizer/scheduler/update
counter/EMA/checkpoint/validation progression on that attempt, and enforces
the configured minimum-scale guard. `src/animal2vec/cli/train.py` consumes the
explicit skip result.

**Evidence.** `logs/fp16-config-regression-red.log`,
`logs/fp16-config-regression-green.log`,
`logs/amp-overflow-regression-red.log`, and
`logs/amp-overflow-regression-green.log`. The exact pretraining and fine-tuning
burn-ins then demonstrated deterministic scale-retry and resume behavior.

### 3. RNG checkpoint restore under CUDA `map_location`

**Symptom.** A checkpoint loaded directly onto CUDA could leave serialized RNG
byte tensors on the device expected for model tensors. PyTorch CPU/CUDA RNG
setters do not share that placement contract.

**Root cause.** Generic checkpoint `map_location` was allowed to dictate RNG
payload placement, although RNG setters require device-specific normalized
byte tensors.

**Correction.** `src/animal2vec/training/checkpoint.py` validates RNG payloads
and moves them to CPU before passing them to the CPU/CUDA RNG setter APIs.
Model and optimizer tensors still honor the requested map location.

**Evidence.** `logs/rng-map-location-regression-red.log` and
`logs/rng-map-location-regression-green.log`; the multi-rank NCCL probes and
durable checkpoint tests additionally compare rank-local restored sequences
exactly.

### 4. Masking and positional-convolution ordering

**Symptom.** Official pretraining parity agreed before masking but first
diverged immediately after the masked frontend/positional path.

**Root cause.** Masked projected tokens entered the positional convolution in
an order inconsistent with the archived model, allowing masked content to
influence neighboring contextual positions.

**Correction.** `src/animal2vec/models/frontend.py` zeros the masked projected
tokens at the archived point before the positional convolution. No convolution
weights or mask sampling rules changed.

**Evidence.** `logs/masked-positional-regression-red.log` and
`logs/masked-positional-regression-green.log`, followed by
`logs/native-authentic-tiny-fp32-after-masked-positional-fix.log` and the full
official pretraining reports.

### 5. Decoder normalization dtype under autocast

**Symptom.** Deep FP16 parity first diverged at decoder normalization because
the native block returned a different intermediate dtype under CUDA autocast.

**Root cause.** The native normalization wrapper did not reproduce ordinary
`torch.nn.LayerNorm` autocast behavior used by the archived decoder.

**Correction.** `src/animal2vec/modules/decoder.py` uses ordinary
`nn.LayerNorm` at that site, restoring the archived FP32 normalization result
inside autocast while allowing the decoder output path to return FP16.

**Evidence.** `logs/decoder-autocast-dtype-regression-red.log`,
`logs/decoder-autocast-dtype-regression-green.log`, and
`logs/native-authentic-tiny-fp16-after-decoder-fix.log`.

### 6. Official checkpoint update extraction

**Symptom.** A converted official checkpoint could report the wrong update
when the legacy file stored it in optimizer history instead of at the root.

**Root cause.** The converter read only one historical layout.

**Correction.** `src/animal2vec/checkpoint_conversion.py` first honors an
explicit root update and otherwise extracts the latest applicable
`optimizer_history[*].num_updates` value with validation. This is metadata
handling only; model tensors are unchanged.

**Evidence.** `logs/official-num-updates-regression-red.log` and
`logs/official-num-updates-regression-green.log`. The converted checkpoints
now retain official updates 408,402 and 30,000.

### 7. Explicit normalization epsilon and event IoU configuration

**Symptom.** Two behavior-bearing legacy values were implicit or unavailable
through the native typed configuration: normalization epsilon and the event
IoU threshold.

**Root cause.** The rewrite's configuration schema omitted those fields, so
the runtime could fall back to library defaults instead of the checkpoint or
recipe value.

**Correction.** `src/animal2vec/config.py` parses `model.norm_eps` explicitly
as a floating-point value and adds `criterion.iou_threshold`. Call sites use
the resolved typed values.

**Evidence.** `logs/red-config-norm-eps.log`,
`logs/green-config-norm-eps.log`, `logs/event-iou-config-red.log`, and
`logs/event-iou-config-green.log`.

### 8. Final partial-segment inference trimming and empty input

**Symptom.** Chunked inference on a recording whose duration was not a whole
segment emitted convolution frames and timestamps beyond the real end of the
final segment. A zero-length recording also needed a stable shaped result.

**First divergence.** The red integration test observed an official-model
timestamp beyond the 10.25 s input boundary.

**Root cause.** The runner trimmed by padded chunk shape rather than by the
actual final segment duration after frontend subsampling.

**Correction.** `src/animal2vec/inference/runner.py` computes the valid final
frame extent with the frontend's convolution geometry, discards timestamps at
or beyond the actual segment end, bounds derived events, and returns correctly
shaped empty tensors for empty audio.

**Evidence.** `logs/red-partial-frame-trim.log`,
`logs/green-partial-frame-trim.log`,
`logs/red-official-partial-timestamps.log`, and
`logs/green-inference-integration-after-trim.log`, plus the official native
FP32/FP16 inference logs.

### 9. DataLoader prefetch versus consumed sampler position

**Symptom.** With multiple workers, an interrupted run's saved sampler cursor
could be ahead of the last batch actually delivered to the training loop.
Resuming then skipped prefetched-but-unconsumed data.

**Root cause.** Worker prefetch advances the sampler iterator asynchronously;
the iterator's internal cursor is not equivalent to “batches committed by
training.”

**Correction.** `src/animal2vec/cli/train.py` tracks the number of batches
delivered/consumed by the training loop and writes the checkpoint sampler
cursor from that committed count. It does not use the worker-prefetched cursor
as durable state.

**Evidence.** `logs/multiworker-sampler-prefetch-regression-red.log` and
`logs/multiworker-sampler-prefetch-regression-green.log`; the final recipe
checkpoint records sampler next position 140 consistently in both branches.

### 10. Next-epoch DataLoader generator ownership

**Symptom.** A resume ending near an epoch boundary could match the current
epoch but diverge when the next DataLoader was constructed.

**Root cause.** DataLoader construction/worker seeding consumed the global RNG
stream, and uninterrupted versus resumed processes did not necessarily perform
the same construction history.

**Correction.** `src/animal2vec/cli/train.py` supplies a deterministic
per-rank DataLoader generator derived from stable seed/epoch/rank state. Loader
bookkeeping no longer perturbs the model's global RNG stream.

**Evidence.** `logs/epoch-loader-rng-resume-regression-red.log` and
`logs/epoch-loader-rng-resume-regression-green.log`.

### 11. Clone-expanded ALiBi memory explosion

**Symptom.** The exact pretraining model approached/OOMed 40 GB while forming
a clone-expanded attention bias. Instrumentation identified approximately
15.36 GB attributable to scaling the full bias before selecting the actually
needed query/key positions.

**Root cause.** Multiplication by the clone/head scale broadcast first, creating
a very large full `[clone, head, query, key]` intermediate; row/column gathering
occurred afterward.

**Correction.** `src/animal2vec/models/frontend.py` gathers the required ALiBi
rows and columns first, then applies the clone-expanded scale to the much
smaller selected tensor. Indexing and mathematical values are unchanged.

**Evidence.** The regression verifies both forward values and gradients:
`logs/alibi-gather-before-scale-regression-red.log` and
`logs/alibi-gather-before-scale-regression-green.log`. Official numerical
parity remained passing, and the unchanged native recipe subsequently fit with
2.40 GiB of conservative reserved-memory headroom.

### 12. Duplicate CUDA checkpoint residency on resume

**Symptom.** Resume memory was inflated because the loaded checkpoint mapping
continued to own CUDA model tensors after they had been copied into live model,
optimizer, scheduler, scaler, RNG, and sampler state.

**Root cause.** The large deserialized payload stayed referenced for the
remainder of setup.

**Correction.** `src/animal2vec/cli/train.py` restores every required state,
then clears/releases the checkpoint payload before the first measured update.
No state field is discarded before restoration.

**Evidence.** `logs/resume-checkpoint-release-regression-red.log`,
`logs/resume-checkpoint-release-regression-green.log`, and
`logs/checkpoint-release-cli-resume-regression.log`, followed by the exact
resumed recipe memory summaries.

### 13. DDP bucket traversal and exact resume

**Symptom.** Uninterrupted and resumed jobs differed at sub-ULP scale despite
matching data, masks, and saved RNG. The first value was reported rather than
hidden by aggregate tolerance.

**Root cause.** Dynamic bucket rebuild/traversal changed floating-point
reduction order after restart. Static graph was unsafe for this model path and
also allowed ranks to diverge without a rank-0 checkpoint exposing it.

**Correction.** The distributed wrapper uses one explicit 4,096 MiB bucket and
unused-parameter traversal for the relevant stages. The full experiment chain
is described in the Gate E section.

**Evidence.** Focused red/green artifacts include
`logs/pretrain-ddp-explicit-bucket-red.log`,
`logs/pretrain-ddp-explicit-bucket-green.log`,
`logs/pretrain-ddp-single-bucket-red.log`,
`logs/pretrain-ddp-single-bucket-green.log`,
`logs/pretrain-ddp-find-unused-resume-red.log`,
`logs/pretrain-ddp-find-unused-resume-green.log`, and the four bucket-probe
logs. Final authority is the all-exact two-rank, four-rank, pretraining, and
fine-tuning recursive reports.

### 14. Average precision ties and archived event evaluation

**Symptom.** Event average precision differed when multiple predictions shared
the same score, even though untied fixtures agreed.

**Root cause.** Integrating one item at a time makes AP depend on arbitrary
ordering within equal-score groups; the archived scikit-learn path evaluates a
threshold after consuming all items with the same score and integrates the
precision-recall curve in reverse threshold order.

**Correction.** `src/animal2vec/training/metrics.py` groups equal scores before
reverse integration. The new pure-PyTorch evaluator in
`src/animal2vec/evaluation/events.py` composes that AP implementation with the
archived interval, split, merger, pooling, macro/micro, and focal-threshold
rules.

**Evidence.** `logs/average-precision-ties-red.log`,
`logs/average-precision-ties-green.log`, the event-evaluator red/green unit
logs, and the final archived/native synthetic comparison report.

### 15. Diagnostic infrastructure: first-diverging checkpoint state

The recursive comparator was itself regression-tested so that an apparent
“not equal” result identifies the first differing path, kind, shape/dtype, and
maximum error. This capability was essential to the DDP investigation. The
evidence is `logs/checkpoint-comparator-regression-red.log` and
`logs/checkpoint-comparator-regression-green.log`. It is test infrastructure,
not a change to training numerics.

## Source and test change inventory

The final working-tree diff is intentionally narrow relative to the scope of
the architecture. The repository captures the exact tracked and untracked
state under `environment/`; the following describes the purpose of each
material verification change.

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | declare `gpu` and `nccl` pytest markers so accelerator tests are durable and selectable |
| `src/animal2vec/checkpoint_conversion.py` | extract official legacy update from root or latest optimizer history |
| `src/animal2vec/cli/train.py` | add terminal-only `--stop-at-update`; stable DDP settings; exact consumed-sampler and loader RNG state; skipped-update handling; checkpoint payload release; structured per-rank timing/memory summaries |
| `src/animal2vec/config.py` | parse AMP initial/minimum scale, normalization epsilon, and event IoU threshold |
| `src/animal2vec/inference/runner.py` | trim final partial-segment frames/timestamps/events and handle empty recordings |
| `src/animal2vec/models/frontend.py` | restore mask/positional ordering and gather ALiBi before clone scaling |
| `src/animal2vec/modules/decoder.py` | reproduce archived LayerNorm autocast dtype behavior |
| `src/animal2vec/modules/sinc.py` | deterministic explicit reflection padding |
| `src/animal2vec/training/checkpoint.py` | validate and normalize RNG tensor placement on restore |
| `src/animal2vec/training/engine.py` | expose skipped updates and reproduce GradScaler overflow/min-scale semantics |
| `src/animal2vec/training/metrics.py` | make tied-score AP match archived/scikit-learn threshold semantics |
| `src/animal2vec/evaluation/events.py` | dependency-light native implementation of archived event segmentation and scoring |
| `scripts/audit_meerkat_dataset.py` | exhaustive audio/HDF5 pair, structure, dtype, bound, duration, and event audit |
| `scripts/audit_meerkat_manifests.py` | manifest membership, overlap, subset, label-size, and hash audit |

The GPU support directory contains reusable scripts rather than hidden manual
steps:

- `tests/gpu/export_official_checkpoint.py` creates the neutral checkpoint
  export;
- `legacy_official_pretraining_export.py` and
  `legacy_official_finetuning_export.py` execute archived references;
- `native_official_pretraining_compare.py` and
  `native_official_finetuning_compare.py` perform layer/gradient/update
  comparisons;
- `official_native_inference.py` performs strict converted-checkpoint inference;
- `nccl_probe.py` and `nccl_resume.py` cover collectives/RNG/durable resume;
- `compare_training_checkpoints.py` recursively diagnoses full saved state;
- the legacy/native event exporter/comparator pair isolates event evaluation;
- `legacy_official_pretraining_memory.py` localizes archived recipe memory; and
- the three `test_cuda_*.py` files form the marker-selectable one-A100 suite.

Integration and unit changes live beside the code they protect in
`tests/integration/` and `tests/unit/`. No Fairseq import is used in a native
test process; archived reference exporters run only from the legacy
environment.

## Reproduction commands

The canonical authority commands are also preserved verbatim in
`results/gpu_verification/20260723T081428Z/commands/final-authoritative-commands.sh`.
The commands below are normalized for readability but retain the environment,
GPU visibility, topology, configuration, manifest, and result paths used in
the final jobs. They assume the repository working directory is
`/abyss/home/ml/openai_a2v_convert/animal2vec` and the four environments have
already been built exactly as described in
`docs/A100_GPU_VERIFICATION_BOOTSTRAP.md` and verified against the retained
freeze/runtime files.

### Clean CPU and one-A100 suites

```bash
rtk proxy /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/python -m pytest -q

rtk env CUDA_VISIBLE_DEVICES=0 \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/python \
  -m pytest -q -m gpu \
  tests/gpu/test_cuda_components.py \
  tests/gpu/test_cuda_inference.py \
  tests/gpu/test_cuda_training.py
```

Expected final outcomes are 171 CPU tests passing and 14 GPU-marked tests
passing. A rerun may take a different wall time but must not silently deselect
tests.

### Two- and four-rank NCCL probes

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1 \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/torchrun \
  --standalone --nproc-per-node=2 \
  tests/gpu/nccl_probe.py \
  --output-dir results/gpu_verification/20260723T081428Z/parity/nccl-2rank-postfix

rtk env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/torchrun \
  --standalone --nproc-per-node=4 \
  tests/gpu/nccl_probe.py \
  --output-dir results/gpu_verification/20260723T081428Z/parity/nccl-4rank-postfix
```

### Durable distributed resume probes

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1 \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/torchrun \
  --standalone --nproc-per-node=2 \
  tests/gpu/nccl_resume.py \
  --output-dir results/gpu_verification/20260723T081428Z/parity/resume-2rank-single-bucket-find-unused

rtk env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/torchrun \
  --standalone --nproc-per-node=4 \
  tests/gpu/nccl_resume.py \
  --output-dir results/gpu_verification/20260723T081428Z/parity/resume-4rank-single-bucket-find-unused
```

The generated comparison state must be exact for every rank, not only rank 0.

### Exact-recipe four-rank pretraining sequence

Run the stop source first, preserve its update-1 file, then launch independent
continuous and resumed branches. The scheduler horizon remains 384,230 in all
three commands.

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/torchrun \
  --standalone --nproc-per-node=4 \
  -m animal2vec.cli.train \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/data/manifests/regenerated-seed1612 \
  --override checkpoint.save_dir=/abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/checkpoints/exact-pretrain-ddp4-final-stop1 \
  --stop-at-update 1 --device cuda

rtk cp \
  results/gpu_verification/20260723T081428Z/checkpoints/exact-pretrain-ddp4-final-stop1/checkpoint_last.pt \
  results/gpu_verification/20260723T081428Z/checkpoints/exact-pretrain-ddp4-final-stop1/checkpoint_update1.pt

rtk env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/torchrun \
  --standalone --nproc-per-node=4 \
  -m animal2vec.cli.train \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/data/manifests/regenerated-seed1612 \
  --override checkpoint.save_dir=/abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/checkpoints/exact-pretrain-ddp4-final-continuous-update2 \
  --stop-at-update 2 --device cuda

rtk env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/torchrun \
  --standalone --nproc-per-node=4 \
  -m animal2vec.cli.train \
  --config configs/MeerKAT/a2v_large_pretrain_best.yaml \
  --override task.data=/abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/data/manifests/regenerated-seed1612 \
  --override checkpoint.save_dir=/abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/checkpoints/exact-pretrain-ddp4-final-resume-update2 \
  --resume /abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/checkpoints/exact-pretrain-ddp4-final-stop1/checkpoint_update1.pt \
  --stop-at-update 2 --device cuda

rtk proxy /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/python \
  tests/gpu/compare_training_checkpoints.py \
  results/gpu_verification/20260723T081428Z/checkpoints/exact-pretrain-ddp4-final-continuous-update2/checkpoint_last.pt \
  results/gpu_verification/20260723T081428Z/checkpoints/exact-pretrain-ddp4-final-resume-update2/checkpoint_last.pt \
  results/gpu_verification/20260723T081428Z/metrics/exact-pretrain-ddp4-final-resume-comparison.json
```

The historical launches additionally used `torchrun --log-dir`, `--redirects
3`, and `--tee 3` so every rank's stdout/stderr was retained. Those exact log
directory arguments are in the canonical command manifest.

### Exact-recipe 100% fine-tuning sequence

Fine-tuning is initialized from the converted official pretraining checkpoint.
Use the same stopped/continuous/resumed pattern, changing only terminal update
and output directory. The normalized uninterrupted invocation is:

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/torchrun \
  --standalone --nproc-per-node=4 \
  -m animal2vec.cli.train \
  --config configs/MeerKAT/finetune_mixup_100.yaml \
  --pretrained-checkpoint results/gpu_verification/20260723T081428Z/checkpoints/official/animal2vec_large_pretrained_MeerKAT_240507.native.pt \
  --override task.data=/abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/data/manifests/regenerated-seed1612 \
  --override checkpoint.save_dir=/abyss/home/ml/openai_a2v_convert/animal2vec/results/gpu_verification/20260723T081428Z/checkpoints/exact-finetune100-ddp4-final-continuous \
  --stop-at-update 2 --device cuda
```

For the resume branch, first use the same command with save directory
`exact-finetune100-ddp4-final-resume` and `--stop-at-update 1`; preserve the
result as `checkpoint_update1.pt`. Then rerun with the same save directory,
replace `--pretrained-checkpoint ...` with
`--resume .../exact-finetune100-ddp4-final-resume/checkpoint_update1.pt`, and
set `--stop-at-update 2`. Compare continuous `checkpoint_last.pt` with resumed
`checkpoint_last.pt` using `compare_training_checkpoints.py`. The expected
report is `metrics/exact-finetune100-ddp4-final-resume-comparison.json` with
`all_exact: true`.

### Official checkpoint export, conversion, and parity

These operations must remain environment-separated:

1. run `tests/gpu/export_official_checkpoint.py` from
   `a2v-legacy-parity-cu126` to produce the plain checkpoint and export report;
2. run the package converter and native strict loader from the appropriate
   native environment;
3. run `legacy_official_pretraining_export.py` or
   `legacy_official_finetuning_export.py` on GPU 0 in the legacy environment,
   with `--precision fp32` or `fp16`, update 408,402 or 30,000, and seed 7;
4. run the matching `native_official_*_compare.py` from the native environment,
   pointing it at the converted checkpoint, plain legacy checkpoint, conversion
   report, legacy reference payload, resolved config, and a new output
   directory; and
5. require every comparison entry to pass and every strict mapping count to be
   zero.

The scripts expose their complete positional/flag interface through `--help`.
The exact historical arguments and outputs are preserved in each parity
directory and its rank/process logs; the environment boundary is more important
than attempting to import Fairseq and native Animal2Vec in one interpreter.

### Dataset and evaluator reproduction

```bash
rtk proxy /abyss/home/ml/openai_a2v_convert/.venvs/a2v-analysis/bin/python \
  scripts/audit_meerkat_dataset.py \
  results/gpu_verification/20260723T081428Z/data/MeerKAT_10s_2024-06-12 \
  --workers 32 --batch-size 512 \
  --output results/gpu_verification/20260723T081428Z/data/official-dataset-audit.json

rtk proxy /abyss/home/ml/openai_a2v_convert/.venvs/a2v-analysis/bin/python \
  scripts/audit_meerkat_manifests.py \
  results/gpu_verification/20260723T081428Z/data/manifests/regenerated-seed1612 \
  --dataset-audit results/gpu_verification/20260723T081428Z/data/official-dataset-audit.json \
  --output results/gpu_verification/20260723T081428Z/data/regenerated-manifest-audit.json

rtk proxy /abyss/home/ml/openai_a2v_convert/.venvs/a2v-legacy-parity-cu126/bin/python \
  tests/gpu/legacy_event_evaluator_export.py \
  results/gpu_verification/20260723T081428Z/parity/event-evaluator-synthetic/legacy-reference.pt

rtk proxy /abyss/home/ml/openai_a2v_convert/.venvs/a2v-native-cu130/bin/python \
  tests/gpu/native_event_evaluator_compare.py \
  results/gpu_verification/20260723T081428Z/parity/event-evaluator-synthetic/legacy-reference.pt \
  results/gpu_verification/20260723T081428Z/parity/event-evaluator-synthetic/comparison-report.json
```

If a new UTC result root is used, substitute it consistently and regenerate
hashes rather than overwriting this audit bundle.

## Configuration and checkpoint identity

The source recipe SHA-256 values at verification time were:

| Recipe | SHA-256 |
| --- | --- |
| `configs/MeerKAT/a2v_large_pretrain_best.yaml` | `2bdf8333d1c900c984fa32f53f86280d8a5dc09cfb0e7702b63994ee4bfd7702` |
| `configs/MeerKAT/finetune_mixup_100.yaml` | `8244346030fb274d70e64279e9efe7a5547901bd80c0aabfeb84f75469e88eb9` |
| `configs/MeerKAT/finetune_mixup_025.yaml` | `aaa333d3af1f9fe6d9f021faaee6e99c7b70e469981db2695d9ebb5aa245acfe` |
| `configs/MeerKAT/finetune_mixup_001.yaml` | `d67d7831d2a9bb6d0aeb593e4b89a7017726d730ba672b5d200b32cb108480aa` |

The complete file is `configs/source-yaml-sha256.txt`. Resolved JSON, rather
than source YAML alone, is authoritative for a completed job because it records
checkpoint-derived architecture and every applied override.

Retained checkpoint SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| official pretraining raw | `c048af7f2fb5c149fcc888a4a982a947e92bba0ea657dee82229fe1f4264d1de` |
| official pretraining native | `e5cb00a82b2aac92b98e38e77ba377b36a6a687b0b1f857ccbbb2810d1a41d08` |
| official fine-tuning raw | `2846cefa9e1d696a95d91f9ae316456895eb3075eaafc5e81b3d20eb9c4bb712` |
| official fine-tuning native | `b95db2dfa6e13ab9e3f02f593bcc461d67664388d9f3a64ada34b492e65e9795` |
| final fine-tuning continuous update 2 | `f97e5d45bb2b673f0b1a4ca86615a044f050376f620f10b0ba8240018a46790e` |
| final fine-tuning resumed update 2 | `3e37860f6f523a91a83fde4824ab8f373e5e9744a2c0108855568ffe6b8c49b7` |
| final pretraining stop update 1 | `e3e6852e1d44e2ce7a730e2aa0e70975694e7eb194cde62c2f42344c93f39bf6` |
| final pretraining continuous update 2 | `9ac7dfdc8f38190123384a3b1ef358ec513ba202b2771d0baf9f2f219b52e5cd` |
| final pretraining resumed update 2 | `5a0222516713ebccfc545adc2cfe6da365877370740a0f4c82276a37d6e35bdf` |

The source of record is
`checkpoints/checkpoint-sha256-final.txt`. As explained earlier, binary hash
differences between continuous and resumed files are expected and do not
contradict the all-exact normalized logical comparison.

## Artifact bundle layout and retention

The result bundle is deliberately self-describing. At finalization it occupied
approximately 273 GB; the largest categories were checkpoints (approximately
131 GB), released/extracted data and manifests (approximately 85 GB), and
archived/native parity payloads (approximately 49 GB). Complete logs were only
about 68 MB because rank output is text while tensors/checkpoints dominate
storage.

| Relative path below `results/gpu_verification/20260723T081428Z/` | Contents |
| --- | --- |
| `environment/` | untouched host capture, GPU UUIDs/topology, Python builds, four package freezes/runtime inspections, git states/diffs, publisher metadata, deterministic settings |
| `configs/` | resolved pretraining/fine-tuning configurations, YAML load audit, source recipe hashes |
| `commands/` | canonical final authority launch commands |
| `logs/` | complete pytest output, one-A100 runs, every rank's exact-recipe output, red/green regressions, DDP bucket probes, inference/parity logs |
| `checkpoints/` | official raw files, neutral/native conversions, exact stop/continuous/resumed recipe checkpoints, final hash manifest |
| `parity/` | archived references, native comparison reports, NCCL probes, durable resume states, event-evaluator fixtures |
| `metrics/` | structured per-rank memory/timing aggregates and recursive checkpoint comparison reports, including rejected diagnostics |
| `data/` | official archive/checksum/extraction, exhaustive audit, regenerated manifests and hashes |
| `verification-report.json` | machine-readable gate decision and artifact references |
| `verification-report.md` | artifact-bundle copy of this human-readable report |

Large files are retained rather than inlined into Git. Durable source,
regression tests, audit scripts, and this report live in the repository. A
reviewer can therefore validate implementation and small evidence from Git,
then attach or mount the result bundle for checkpoint/data-level audit.

The machine-readable report's artifact references were checked against the
filesystem during finalization. `git diff --check` also passed, preventing
whitespace damage from being hidden in the sizable verification diff.

## Gate G continuation procedure

Gate G remains blocked/not run. Completing it requires both authoritative
publication inputs and long-running experiments; a short correctness burn-in
must never be relabeled as a paper result.

### Artifacts that must be acquired or resolved

1. **Official MeerKAT split manifests.** Obtain exact membership for all five
   folds and the paper's 1%, 25%, and 100% label fractions, with publisher
   hashes or another authoritative provenance chain. Replacing the regenerated
   manifests must be explicit; do not overwrite them.
2. **Publisher supplement.** Acquire
   `mee370218-sup-0001-supinfo.pdf` from an authoritative source. The publisher
   endpoint returned HTTP 403 during this verification, so no procedural
   details were inferred from an inaccessible document.
3. **Corpus discrepancy resolution.** Determine why the released corpus has
   66,397 rather than 66,398 labelled recordings and why 31,273 files are
   PCM24 despite the paper's 16-bit statement. Record any replacement file and
   hash before training.
4. **Xeno-canto protocol input.** Obtain the exact recording subset, exclusions,
   retrieval dates/identifiers, hashes, and preprocessing provenance used by
   the paper.
5. **NIPS4Bplus protocol input.** Obtain the exact sequence-level and
   framewise manifests/labels and evaluation protocol artifacts.

### Experiments still required

1. Run the full 384,230-update four-rank MeerKAT pretraining recipe on GPUs
   0–3, preserving every resolved setting and recording periodic validation,
   checkpoint, memory, and elapsed-time summaries.
2. Fine-tune all five official folds at the paper's 1%, 25%, and 100% fractions
   using the exact 30,000-update recipe. This is 15 authoritative fine-tuning
   jobs, not the additional 10/50/75% regenerated subsets.
3. Cross the real update-10,000 freeze boundary in every relevant fine-tuning
   job and verify that the optimizer/checkpoint state remains resumable there.
4. Generate full-manifest probabilities/events, run the now-verified native
   evaluator, and compare classwise, macro, micro, focal-threshold, precision,
   recall, F1, split, and merger outcomes against archived/published results.
5. Run the exact Xeno-canto and NIPS4Bplus transfer protocols when their inputs
   are available.
6. Record seeds, resolved configs, every checkpoint hash, per-rank memory,
   complete logs, fold membership hashes, event outputs, and final metrics in a
   new UTC result root. Do not mutate this one.

Before accepting those jobs, rerun Gate A/B and the four-rank durable resume
probe on the final training commit. If any source or environment lock changes,
repeat official checkpoint parity in proportion to the affected numerical
surface.

### Gate G acceptance rule

Only after the full pretraining, all required official-fold fine-tuning jobs,
external protocols, and event metrics complete under authoritative inputs may
`docs/GPU_VERIFICATION_HANDOFF.md` Gate G be evaluated. Passing Gates A–F is
necessary but not sufficient. A future report must state the exact paper table
values, deviations, confidence/aggregation convention, and any irreproducible
publication ambiguity before making a paper-reproduction claim.

## Final technical conclusion

The Fairseq-free Animal2Vec rewrite is verified on this A100 host for the
feasible native-runtime gates: clean packaging, CPU behavior, one-A100 FP32 and
FP16 execution, strict official checkpoint conversion, same-GPU layer/output/
gradient/optimizer-update parity, two- and four-rank NCCL, exact distributed
resume, and unchanged-recipe four-rank memory fit. The released MeerKAT corpus
is structurally sound under an exhaustive audit, and the dependency-light
event evaluator exactly reproduces the archived algorithm on controlled
nontrivial fixtures.

The strongest distributed result is not merely “close”: after normalizing only
the output directory, uninterrupted and resumed production-topology
pretraining and 100% fine-tuning burn-in checkpoints are recursively bitwise
exact across model, EMA, optimizer, scheduler, scaler, four rank RNG states,
sampler, epoch, batch, metric, and counter state. The exact pretraining recipe
fits an A100-SXM4-40GB without reducing any published setting, with 2.40 GiB of
headroom against PyTorch-visible capacity at the worst observed reservation.

The boundary of that conclusion is equally important. Official fold
membership, one publication count, the supplement, two external protocol
datasets, full-length training, and paper event metrics remain unavailable or
unfinished. Therefore Gate F is partial, Gate G is blocked/not run, and the
paper has **not** been reproduced.
