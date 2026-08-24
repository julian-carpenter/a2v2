# Modern Transformer and SLURM verification

Date: 2026-08-24 UTC

## Acceptance result

The original bounded gates ran against code revision
`5fe1656e4844e94405bc2e1b9cf87f24ef965624`. Commit
`9f3c7810bb12a2abbf4bd3a907c4c077b8ba915b` records them. They covered the
full CPU suite, one-A100 CUDA components and training, strict Flash attention,
compile, CLS and AdaGC paths, bitsandbytes 8-bit optimizer resume, two-rank
Gloo, and two- and four-rank NCCL state resume. The local and mocked SLURM
checks passed.

A later whole-change review disproved the report's broad multiworker
exact-resume claim. The original integration input used equal lengths and did
not execute random cropping. Fix Round 1 added strict, versioned data
provenance and stateless crop coordinates, then reran variable-length resume in
fresh processes and on two local Gloo ranks. Exact data-path resume requires
`dataset.crop_strategy=stateless` together with
`checkpoint.resume_policy=strict`. Frozen legacy recipes remain compatible,
but their legacy random-crop resume is not bit-exact.

In Fix Round 2, we tightened the strict contract after a scoped review. New
checkpoints carry `a2v2.training-data.v2`: the fine-tuning criterion,
effective AMP/scaler policy, best-metric, retained-record, manifest, sampler,
and task semantics. Legacy crop safety scans the complete repeating epoch
without consulting worker count. Distributed ranks exchange preflight errors
and active fingerprints before any rank constructs a model.

Fix Round 3 moved fresh fine-tuning checkpoint reads and requested-world-size
checks into that coordinated preflight. Each rank hashes and loads the external
checkpoint once, validates its config and student tensor container, then
exchanges the file identity before model construction. The exact encoder
key-and-shape check requires an `AudioEncoder`; the workflow performs it
after the shared preflight and before DDP wrapping. A two-rank NCCL workflow
test at the Round 3 head covers training, resume, and a canonical asymmetric
preflight failure.

Fix Round 4 releases the preflight's deserialized pretrained encoder mapping
after `_make_model` copies it and before `model.to`. The shared checkpoint
loader now translates expected archive and pickle decode failures into a
path-bearing `CheckpointError`; it preserves an existing `CheckpointError`.
This error boundary does not make pickle input safe.

This verification did not run the paper-scale reproduction or a SLURM
allocation. It does not validate a site filesystem, scheduler, requeue policy,
or multi-node transport.

## Git scope

- Base and `origin/main`: `a49d1d18c0adf53c7e33d7df56f47039c144a9e1`
- Verified code head before this evidence commit:
  `5fe1656e4844e94405bc2e1b9cf87f24ef965624`
- Range: 25 commits, 47 tracked files, 19,298 insertions and 251 deletions
- Initial state: `main...origin/main [ahead 25]`, with an empty short status
- Task 12 adds this report and `tests/gpu/benchmark_attention.py`. The
  containing commit records those two files.
- Fix Round 1 starts from
  `9f3c7810bb12a2abbf4bd3a907c4c077b8ba915b`. The final handoff names its
  containing commit because a commit cannot contain its own hash.
- Before the fix commit, `main...origin/main [ahead 26]` had 23 modified
  tracked files and one new Gloo worker file. `HEAD` was `9f3c781`; the
  containing fix commit makes the branch 27 commits ahead of `origin/main`.
- Fix Round 2 starts from
  `ae86f5a4403ea348710699dfae79b682cfb5d946`. The final handoff names its
  containing commit. The frozen paper files remained unchanged.
- Before the Round 2 commit, `main...origin/main [ahead 27]` had nine modified
  tracked files and one new Gloo worker file. The containing commit makes the
  branch 28 commits ahead of `origin/main`.
- Fix Round 3 starts from
  `6031b73d6e96960edc6f7c78293d48a1c448bc57`. Before its containing commit,
  `main...origin/main [ahead 28]` had six modified tracked files, this report,
  and one new NCCL workflow file. The containing commit makes the branch 29
  commits ahead of `origin/main`.
- Fix Round 4 starts from
  `25b99553b2491ffd174552f367c5415cdff541ff`. Before its containing commit,
  `main...origin/main [ahead 29]` has five modified test or production files
  plus this report. The containing commit makes the branch 30 commits ahead
  of `origin/main`.

## Host and package state

The host exposed Python 3.12.3, PyTorch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, NCCL 2.30.7, and eight
NVIDIA A100-SXM4-40GB devices. Every GPU reported 0 MiB used and 0% utilization
before the bounded single-GPU and distributed runs.

The starting host had bitsandbytes 0.50.0, outside the repository's declared
`bitsandbytes>=0.49,<0.50` range. This command restored the declared extra:

```text
rtk python -m pip install -e '.[bnb]'
```

Outcome: exit 0; pip replaced 0.50.0 with bitsandbytes 0.49.2.

Native import on CUDA 13.3 reported the missing
`libbitsandbytes_cuda133.so`. The negative product gate passed and confirmed
that `build_optimizer` raises its targeted setup error. This documented
override loaded the wheel's CUDA 13.0 binary:

```text
rtk env BNB_CUDA_VERSION=130 python -c 'import bitsandbytes as bnb; from bitsandbytes.cextension import lib; print("bitsandbytes", bnb.__version__); print("compiled_with_cuda", bool(lib.compiled_with_cuda)); print("native_library", getattr(lib, "_name", None))'
```

Outcome: bitsandbytes 0.49.2, `compiled_with_cuda=True`, native library
`libbitsandbytes_cuda130.so`. The CUDA 13.3 wheel gap is host compatibility
evidence, not a product failure. A source build with CUDA 13.3 remains unrun.

The first `pip check` found that the preinstalled audiomentations 0.41.0
lacked librosa and soxr. An unpinned repair command selected librosa 1.0.0 and
soxr 1.1.0, which violated audiomentations' version ranges:

```text
rtk python -m pip install librosa soxr
```

The installed metadata identified the required ranges. This correction
restored librosa 0.10.2.post1 and soxr 0.5.0.post1:

```text
rtk python -m pip install 'librosa>=0.8.0,<0.11.0,!=0.10.0' 'soxr>=0.3.2,<1.0.0'
rtk python -m pip check
```

Outcome: both commands exited 0; `pip check` printed
`No broken requirements found.`

## CPU, package, and source checks

| Command | Outcome |
| --- | --- |
| `rtk python -m pytest -q tests/unit tests/integration` | 526 passed, 8 warnings, 316.99 s |
| `rtk python -m compileall -q a2v2 tests` | Exit 0 |
| `rtk python -m pip check` | Exit 0 after the host corrections above |
| `rtk git diff --check` | Exit 0 |
| `rtk git status --short` | Empty before Task 12 files |

The eight CPU warnings came from Python 3.12's warning about `fork()` from a
multithreaded process in two multiworker CLI resume tests. No CPU test failed,
but the equal-length resume fixture could not test crop exactness.

## Fix Round 1: resume, CLS, and RunContract corrections

### RED evidence

The reviewer reproduced a variable-length, `num_workers>0` divergence at
update 2: uninterrupted loss `2.6233898163`, restarted loss `2.5281476974`;
the first state mismatch was `student.alibi_scale`. The root cause was crop
offset selection from process-local DataLoader worker RNG. Sampler position
was checkpointed, but worker RNG and prefetch progress were not.

Focused tests reproduced the other findings before their fixes:

- config tests raised `AttributeError` for the absent crop/resume policies and
  showed that `optimizer.min_8bit_size=0` loaded;
- removing the new divisibility or RoPE even-head config guard made its focused
  rejection test fail; restoring both guards returned the test to GREEN;
- strict provenance tests failed because no manifest/sampler provenance helper
  existed;
- the shipped CLS validation completed, then best-metric selection raised
  because `metrics/finetune/f1` was absent from the sequence metrics;
- changing an unrelated `pretrain.tsv` changed the pretraining RunContract
  even when the selected config named `alternate.tsv`.

### GREEN evidence before the final full-suite rerun

| Command | Outcome |
| --- | --- |
| `rtk python -m pytest -q tests/unit/test_config.py -k 'modern_example_configs or modern_options_default or modern_option_validation or crop_and_resume'` | 4 passed |
| `rtk python -m pytest -q tests/unit/test_sampler.py tests/unit/test_dataset.py -k stateless` | 2 passed |
| `rtk python -m pytest -q tests/integration/test_resume.py -k 'strict_resume or legacy_multiworker'` | 3 passed |
| `rtk python -m pytest -vv tests/integration/test_cli.py -k stateless_multiworker_crop_resume` | 1 passed in 28.39 s |
| `rtk python -m pytest -vv tests/integration/test_cli.py -k two_rank_gloo_stateless_crop` | 1 passed in 38.55 s |
| `rtk python -m pytest -q tests/integration/test_finetuning_step.py -k cls_validation_reports_sequence` | 1 passed in 5.81 s |
| `rtk python -m pytest -q tests/integration/test_slurm_launcher.py -k pretrain_contract_uses_configured` | 1 passed in 45.67 s |
| `rtk python -m pytest -q tests/unit/test_config.py tests/unit/test_dataset.py tests/unit/test_sampler.py tests/unit/test_engine.py tests/unit/test_slurm.py tests/integration/test_resume.py` | 154 passed, 4 expected compatible-policy warnings |

The fresh-process crop test uses twelve unequal waveform lengths from 96 to
184 samples, two DataLoader workers, a 400-sample token budget, strict
provenance, and stateless cropping. It compares model, teacher, optimizer,
scheduler, scaler, update, epoch, batch cursor, RNG, sampler, and best metric
after update 2. The two-rank Gloo test uses the same variable-length population
with one DataLoader worker per rank and compares the same fields after a fresh
torchrun resume. Both comparisons were exact.

Round 1 bound the full task, ordered labels, dataset mathematics and batching,
resolved selected-manifest path and bytes, filtered population, and ordered
sizes. It omitted loss, AMP/scaler, tracked-metric, and retained-record
identity fields. Fix Round 2 closes those gaps below. `num_workers`, logging,
and compile settings remain execution fields.

The required fresh final CPU command was:

```text
rtk python -m pytest -q tests/unit tests/integration
```

Outcome: 535 passed, 12 warnings in 452.17 s. Eight warnings were Python
3.12's existing multi-threaded `fork()` warning. Four were the intentional
compatible-policy warning from tests that exercise checkpoints without
versioned data provenance. No test failed.

Fix Round 1 ran no CUDA command. The corrections affect config loading, CPU
collation and sampler coordinates, checkpoint provenance, metric routing, and
launcher identity. The earlier bounded CUDA, Flash, compile, AdaGC,
bitsandbytes, and NCCL evidence remains recorded above; this fix round makes no
new hardware claim.

Post-fix source and package checks exited 0:

```text
rtk python -m compileall -q a2v2 tests
rtk python -m pip check
rtk git diff --check
rtk bash -n scripts/reproduce_meerkat_slurm.sh
rtk bash -n scripts/a2v2_slurm_node.sh
rtk bash scripts/reproduce_meerkat_slurm.sh /datasets/MeerKAT/manifests /shared/runs/meerkat --phase all --nodes 2 --gpus-per-node 4 --job-id dryrun-4815 --dry-run
```

`pip check` printed `No broken requirements found.` The canonical dry run
selected `pretrain.tsv` from the unchanged frozen default and rendered the same
two-node/eight-worker phase topology. The alternate-subset regression test
proved that a configured `alternate.tsv` changes the pretraining contract,
while an unrelated neighboring `pretrain.tsv` does not.

After Task 12 added the benchmark harness, this focused command exposed one
new documentation failure:

```text
rtk python -m pytest -q tests/unit/test_repository_layout.py tests/unit/test_documentation.py
```

The first run reported 1 failed and 22 passed because the nested benchmark
helper lacked a docstring. I added that docstring without changing executable
logic. The regression test then passed 1/1, and the full focused command passed
23/23 in 19.21 s.

## Fix Round 2: strict resume and distributed preflight

### RED evidence

The scoped review found four ways the Round 1 strict contract could accept a
different resumed computation:

- the fingerprint omitted the criterion, FP16/scaler policy, and saved
  best-metric identity and direction;
- equal-size filtered populations could share manifest and size digests even
  when they retained different rows;
- an exhausted sampler cursor or a changed `num_workers` value could hide a
  legacy crop elsewhere in the repeating epoch;
- one distributed rank could fail during manifest or checkpoint preflight
  while another rank crossed model construction.

This focused RED command produced 10 failures and 2 passing controls:

```text
rtk python -m pytest -q tests/integration/test_resume.py -k 'every_mathematical_state_mismatch or exact_best_metric_direction or filtered_record_identity or strict_resume_rejects_mismatch or legacy_random_crop_resume_policy or old_checkpoint_provenance'
```

The failures named the omitted fingerprint paths, missing retained-record
digest, model-construction sentinel, and absent full-epoch crop warning. The
task-normalization and old-compatible-policy controls passed.

The two-rank RED run failed both cases:

```text
rtk python -m pytest -q tests/integration/test_cli.py -k 'two_rank_gloo_preflight'
```

For the asymmetric manifest case, rank 0 reached the model sentinel while rank
1 raised `ManifestError`; the harness recorded different terminal errors. For
the fingerprint case, both ranks reached the model sentinel. The workflow had
no collective pre-model gate.

### GREEN evidence before the final CPU suite

| Command | Outcome |
| --- | --- |
| `rtk python -m pytest -q tests/integration/test_resume.py tests/integration/test_finetuning_step.py -k 'every_mathematical_state_mismatch or exact_best_metric_direction or strict_resume_rejects_mismatch or tracked_metric or shipped_cls'` | 8 passed |
| `rtk python -m pytest -q tests/integration/test_resume.py -k 'filtered_record_identity or ordered_task_and_manifest or strict_resume_rejects_mismatch or old_checkpoint_provenance'` | 9 passed |
| `rtk python -m pytest -q tests/integration/test_resume.py -k 'legacy_random_crop_resume_policy or old_checkpoint_provenance' tests/unit/test_sampler.py tests/unit/test_dataset.py -k 'legacy_random_crop_resume_policy or old_checkpoint_provenance or stateless'` | 5 passed |
| `rtk python -m pytest -q tests/integration/test_cli.py -k 'two_rank_gloo_preflight'` | 2 passed, 16 deselected, 23.89 s |
| `rtk python -m pytest -q tests/integration/test_resume.py tests/unit/test_engine.py tests/unit/test_sampler.py tests/unit/test_dataset.py` | 77 passed, 4 expected compatible-policy warnings |
| `rtk python -m pytest -q tests/integration/test_cli.py -k 'resume or preflight'` | Exit 0; 6 selected cases, including fresh-process stateless resume and two-rank Gloo resume/preflight |

The first affected 77-test run reported 76 passes and one fixture failure. A
construction-order unit test simulated `world_size=2` without an initialized
process group. `_distributed_device` cannot produce that state. The test
supplies a completed preflight fixture because it targets the later
move/compile/optimizer/DDP order. The rerun passed all 77 cases.

The v2 retained-record digest encodes each filtered row's original manifest
index and line, resolved audio path, and selected size in sampler order. The
regression constructs the same manifest bytes and equal sizes twice, retaining
rows `[0, 1]` and `[1, 2]`; the digests differ. The full criterion enters the
fingerprint for fine-tuning, the stage where focal fields affect loss.
Pretraining fingerprints omit those inactive fields, including when they read
a v2 checkpoint written before this scope correction. Effective
`common.fp16`, `fp16_init_scale`, and `min_loss_scale` fields enter both stage
fingerprints. The tracked metric contributes its configured identity and the
derived `minimize` or `maximize` mode used by best-checkpoint selection. The
allowed `max_update` extension remains excluded.

Strict resume requires v2 provenance. Compatible policy validates the fields
available in older checkpoints and emits a warning. If any batch in the full
sampler epoch can crop, compatible legacy mode warns for any worker count and
strict mode requires stateless coordinates.

For `world_size>1`, each rank completes dataset construction, checkpoint
read, provenance validation, and topology validation inside a local preflight.
Ranks exchange status and full active fingerprints over the Gloo control
group. A local failure produces the lowest-rank canonical error on each rank;
different successful fingerprints produce one shared mismatch error. Both
bounded Gloo cases stopped before the model sentinel and exited without a
hang.

The required fresh Round 2 CPU command was:

```text
rtk python -m pytest -q tests/unit tests/integration
```

Outcome: 546 passed, 12 warnings in 7m37s. Eight warnings came from Python
3.12 for `fork()` in multiworker integration processes. Four tests emitted the
intentional compatible-policy warning for checkpoints without v2 data
provenance. The suite produced no failures.

Round 2 source, package, frozen-control, and launcher checks passed:

```text
rtk python -m compileall -q a2v2 tests
rtk python -m pip check
rtk git diff --check
rtk bash -n scripts/reproduce_meerkat_slurm.sh scripts/a2v2_slurm_node.sh
rtk python -m pytest -q tests/unit/test_config.py tests/unit/test_model_state_contract.py tests/integration/test_recipe_configs.py tests/integration/test_reproduction_driver.py
rtk bash scripts/reproduce_meerkat_slurm.sh /datasets/MeerKAT/manifests /shared/runs/meerkat --phase all --nodes 2 --gpus-per-node 4 --job-id dryrun-4815 --dry-run
```

The first four commands exited 0, and `pip check` printed `No broken
requirements found.` The frozen-control command passed 79 tests in 22.46s;
the recipe, driver, and state-signature hashes remained equal to the values in
the frozen-control section below. The dry run exited 0 and rendered two nodes,
four workers per node, world size 8, separate pretraining and fine-tuning
contracts, checkpoint validation, and one-GPU evaluation. It requested no
SLURM allocation.

Fix Round 2 changed NCCL startup ordering: CUDA training creates the Gloo
control group before distributed preflight. Round 2 ran no CUDA command, so
its report did not verify that ordering. Fix Round 3 supplies the
workflow-level NCCL result below. The earlier Flash, compile, inference, and
bitsandbytes evidence remains scoped to the original Task 12 revision.

## Fix Round 3: pretrained and launch preflight

### RED evidence

The scoped review found three pre-model gaps. Fresh fine-tuning ranks loaded
their external pretraining checkpoints after the shared preflight. A rank with
`distributed_world_size=1` destroyed its default group before another
rank reached the same check. Pretraining fingerprints bound focal-loss
settings that pretraining never reads.

The criterion RED command produced one failure and three passing policy
controls:

```text
rtk python -m pytest -q tests/integration/test_resume.py -k 'fingerprint_binds_criterion_only or strict_finetune_resume_rejects_focal or old_checkpoint_provenance'
```

`criterion.focal_gamma` remained in the pretraining fingerprint. The direct
missing-provenance and `a2v2.training-data.v1` compatible-warning and
strict-rejection controls passed.

Two distributed RED commands reproduced the rank divergence:

```text
rtk python -m pytest -q tests/integration/test_cli.py -k 'two_rank_gloo_preflight_rejects_different_pretrained_identities or two_rank_gloo_preflight_coordinates_divergent_requested_world_size'
rtk python -m pytest -q tests/integration/test_cli.py -k 'two_rank_gloo_preflight_coordinates_fresh_pretrained_failures and pretrained-missing'
```

The first command failed both selected tests. Distinct valid checkpoints
reached the fine-tuning model sentinel. In the world-size case, rank 1
destroyed the default group and rank 0 reported a Gloo connection close. The
second command failed because rank 0 reached model construction while rank 1
reported the missing file, leaving different terminal errors.

### GREEN evidence

Fresh fine-tuning resolves the command-line checkpoint or `model.w2v_path`
inside local preflight. The loader resolves the path, streams a SHA-256 digest,
records byte size, checks device/inode/size/mtime around hashing and loading,
and loads the checkpoint once onto CPU. It validates the pretraining stage,
serialized config, model mapping, student keys, and tensor values. The
preflight returns that bundle to `_make_model`, so model construction performs
no second checkpoint read.

Ranks gather the resume fingerprint and external file identity over the Gloo
control group. The lowest-rank local error wins. Successful ranks with
different identities receive one fingerprint-mismatch error. Tests cover
missing, corrupt, invalid-config, empty-student-state, and distinct valid
checkpoint files. A real single-rank fine-tuning test records one checkpoint
read. Requested-world-size validation runs inside the same collective
preflight. The sequence evaluator keeps its separate checkpoint path and its
integration tests passed.

The affected tests passed:

| Command | Outcome |
| --- | --- |
| `rtk python -m pytest -q tests/integration/test_resume.py -k 'fingerprint_binds_criterion_only or strict_finetune_resume_rejects_focal or old_checkpoint_provenance or resume_compatibility_rejects_every'` | 5 passed, one expected compatible-policy warning |
| `rtk python -m pytest -q tests/integration/test_cli.py -k 'two_rank_gloo_preflight'` | 8 passed |
| `rtk python -m pytest -q tests/integration/test_cli.py -k 'fresh_finetune_consumes_one_preflight_loaded_checkpoint'` | 1 passed |
| `rtk python -m pytest -q tests/integration/test_resume.py tests/integration/test_cli.py tests/integration/test_sequence_evaluation.py` | Exit 0 |

The first full CPU run found two test-only failures. The construction-order
fixture lacked the new `pretrained_bundle` field, and two nested test helpers
lacked docstrings required by the repository documentation test. The focused
rerun passed 24 tests. The final command exited 0 with 556 collected tests and
12 warnings:

```text
rtk python -m pytest -q tests/unit tests/integration
```

Eight warnings came from Python 3.12 `fork()` calls in multiworker integration
processes. Four tests emitted the expected compatible-policy warning for
checkpoints without v2 data provenance.

The Round 3 static, package, frozen-control, and dry-run checks passed:

```text
rtk python -m compileall -q a2v2 tests
rtk python -m pip check
rtk git diff --check
rtk bash -n scripts/reproduce_meerkat_slurm.sh scripts/a2v2_slurm_node.sh
rtk python -m pytest -q tests/unit/test_config.py tests/unit/test_model_state_contract.py tests/integration/test_recipe_configs.py tests/integration/test_reproduction_driver.py
rtk bash scripts/reproduce_meerkat_slurm.sh /datasets/MeerKAT/manifests /shared/runs/meerkat --phase all --nodes 2 --gpus-per-node 4 --job-id dryrun-4815 --dry-run
```

`pip check` printed `No broken requirements found.` The frozen command passed
79 tests in 23.23 s. The recipe, driver, and state-signature hashes remained
equal to the values below. The dry run rendered the unchanged two-node,
eight-worker phase graph and requested no allocation.

All eight A100s reported 0 MiB used and 0% utilization before this bounded
two-rank command:

```text
rtk env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_workflow_resume.py /tmp/a2v2-round3-nccl.3Iga27
```

The command exited 0 in 17.6 s. Both ranks trained to update 1 and resumed to
update 2. The third startup supplied `distributed_world_size=1` on rank 1 and
2 on rank 0. Both ranks received
`distributed training preflight failed on rank 1: ValueError`; no model marker
existed. The two update losses were `2.7812186754666843` and
`2.5518138592059794`, with sample size 52. Each rank reserved 23,068,672 CUDA
bytes. Peak allocated memory was 17,502,720 bytes on rank 0 and 17,511,936 on
rank 1 for update 1, then 17,712,128 and 17,700,864 bytes for update 2.

PyTorch warned that the harness inferred NCCL barrier device IDs and that the
tiny update found no unused DDP parameter despite
`find_unused_parameters=true`. The command used one process per visible GPU,
so the inferred mapping matched this local launch. This test did not exercise
multi-node rank mapping.

## Fix Round 4: pretrained-state lifetime and corrupt-file errors

### RED evidence

The focused RED command ran the new lifetime, native decode, error-preservation,
and direct corrupt-fine-tuning CLI cases:

```text
rtk pytest -q tests/unit/test_engine.py::test_training_releases_pretrained_bundle_before_model_device_move tests/unit/test_checkpoint.py::test_checkpoint_normalizes_native_decode_failures tests/unit/test_checkpoint.py::test_checkpoint_does_not_rewrap_existing_checkpoint_error tests/integration/test_cli.py::test_fresh_finetune_reports_corrupt_pretrained_checkpoint_without_traceback
```

Outcome: 0 passed and 5 failed. At `model.to`, weak references still reached
the preflight bundle and its source tensor. Empty and invalid checkpoint bytes
raised native decode exceptions. The loader also wrapped a pre-existing
`CheckpointError`, and the CLI did not reach its concise parser error.

### GREEN evidence

`_run_training` passes the bundle straight into `_make_model`, then replaces
the immutable preflight with a copy whose `pretrained_bundle` is `None` before
calling `model.to`. The lifetime test lets its coordinator own the only source
mapping references. Its model factory copies one tensor into model-owned
storage. At `model.to`, garbage collection clears weak references to both the
bundle and source tensor while the copied value remains present. The existing
real fine-tuning test still records one checkpoint load.

`load_checkpoint` re-raises an existing `CheckpointError`. It translates
`pickle.UnpicklingError`, `EOFError`, and the existing archive
`OSError`/`RuntimeError`/`ValueError` family into a path-bearing
`CheckpointError`. A corrupt fresh-pretrained CLI run exits with parser status
2, prints no traceback, and never calls `_make_model`. The bounded two-rank
Gloo corrupt-pretrained case still gives both ranks the coordinated pre-model
failure.

The focused rerun passed all 5 RED cases. This affected command passed 26
tests:

```text
rtk pytest -q tests/unit/test_checkpoint.py tests/unit/test_engine.py tests/integration/test_cli.py::test_fresh_finetune_consumes_one_preflight_loaded_checkpoint tests/integration/test_cli.py::test_fresh_finetune_reports_corrupt_pretrained_checkpoint_without_traceback 'tests/integration/test_cli.py::test_two_rank_gloo_preflight_coordinates_fresh_pretrained_failures[pretrained-corrupt]'
```

The first full CPU run found two test-only failures after 559 passes. The
documentation policy named five missing nested-function docstrings. An
existing engine test compared allocator addresses to detect scalar conversion;
the allocator reused a released first-loss address for the final gradient norm,
so the test reported a conversion of the wrong tensor. Isolated and
lifetime-then-engine reproductions passed. The correction compares exact
tensor objects and retains the intended assertion. The five docstrings and
identity correction passed this seven-test command:

```text
rtk pytest -q tests/unit/test_documentation.py::test_python_files_explain_modules_and_named_definitions tests/unit/test_engine.py::test_engine_does_not_materialize_loss_per_microbatch tests/unit/test_engine.py::test_training_releases_pretrained_bundle_before_model_device_move tests/unit/test_checkpoint.py::test_checkpoint_normalizes_native_decode_failures tests/unit/test_checkpoint.py::test_checkpoint_does_not_rewrap_existing_checkpoint_error tests/integration/test_cli.py::test_fresh_finetune_reports_corrupt_pretrained_checkpoint_without_traceback
```

The final explicit CPU command exited 0 with 561 passes:

```text
rtk pytest -q tests/unit tests/integration
```

Round 4 source, package, frozen-control, documentation, and launcher checks
passed:

```text
rtk python -m compileall -q a2v2 tests
rtk python -m pip check
rtk git diff --check
rtk bash -n scripts/reproduce_meerkat_slurm.sh scripts/a2v2_slurm_node.sh
rtk python -m pytest -q tests/unit/test_config.py tests/unit/test_model_state_contract.py tests/integration/test_recipe_configs.py tests/integration/test_reproduction_driver.py
rtk python -m pytest -q tests/unit/test_repository_layout.py tests/unit/test_documentation.py
rtk bash scripts/reproduce_meerkat_slurm.sh /datasets/MeerKAT/manifests /shared/runs/meerkat --phase all --nodes 2 --gpus-per-node 4 --job-id dryrun-4815 --dry-run
```

The frozen-control command passed 79 tests, and the documentation/layout
command passed 23. `pip check` printed `No broken requirements found.` The
other commands exited 0. The dry run rendered the two-node, eight-worker
pretraining and fine-tuning phases plus one-GPU evaluation without requesting
an allocation. The checked-in frozen hashes and state signatures remain equal
to the values below.

Round 4 ran no CUDA command. It changes the lifetime of CPU-deserialized
checkpoint state and the loader's exception boundary. It leaves CUDA
execution, collective ordering, model math, and checkpoint bytes untouched.
The Round 3 bounded NCCL result remains the latest workflow-level CUDA
evidence.

## Frozen reproduction control

This focused command reran config round trips, legacy defaults, state
signatures, published recipes, environment preflight, and local driver dry
runs:

```text
rtk python -m pytest -q tests/unit/test_config.py tests/unit/test_model_state_contract.py tests/integration/test_recipe_configs.py tests/integration/test_reproduction_driver.py
```

Outcome: 78 passed in 22.80 s.

Legacy recipes resolved to eager execution, the legacy position and attention
selectors, no CLS token, MLP, legacy initialization, global clipping, native
Adam, and constant weight decay. Serialization supplied the same defaults when
the new fields were absent. The state signatures remained:

- Tiny pretraining: 120 entries,
  `5efbec43fd1c1c392b8b4278cee6a21513f68f3095353bb22524d2ada2ecf6ad`
- Tiny fine-tuning: 59 entries,
  `b7b2ea2ad9017ec94d56516fbda49a932866b472a48633804619b7232f3d05bb`

The frozen SHA-256 values matched:

| Artifact | SHA-256 |
| --- | --- |
| `configs/MeerKAT/a2v_large_pretrain_best.yaml` | `c5eb23d979cd12704f0dd3977fac1b6031cb4cbf7d5868930eaa6c2816f36a29` |
| `configs/MeerKAT/finetune_mixup_001.yaml` | `384eed9a25da913258e618425c9526cffdaccdacc2a16f912f8e1719fe15a9e0` |
| `configs/MeerKAT/finetune_mixup_025.yaml` | `31daab6409907238af025a480bf236fd2bb82fdd9442a9a1e4d3b0552e0d3f4a` |
| `configs/MeerKAT/finetune_mixup_100.yaml` | `c31c2a3df37d2397ae2cd7cbc502aa849bd819af5ac35f9ff930ab35fb192fd0` |
| `configs/hyenas/animal2vec_base_pretrain_10s-2-1_5_sinc_38ms_mixup_pswish.yaml` | `6a40999452b56e95834d2487860ddc11974c9840b9814eafec1f21468de09102` |
| `configs/hyenas/finetune_mixup_100.yaml` | `f892d0f9bb52a823a1fe4bcd168aab33a29cbd6bcc395ce0822525b6bc3c07ca` |
| `scripts/reproduce_meerkat_paper.sh` | `485b4b36fe1045174a392834bd9ee9848538ab496944631a0b2d514e22d4de18` |

The frozen driver rendered two eight-rank pretraining launches, including the
one-update burn-in and resume, followed by one eight-rank fine-tuning launch.
It retained the published token budgets and update frequencies. The tests also
confirmed the one-GPU final evaluation command.

## Bounded CUDA results

Each pytest command exposed only GPU 0.

```text
rtk env CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s tests/gpu/test_cuda_components.py
```

Outcome: 12 passed in 4.13 s. Strict Flash completed forward and backward for
FP16 and BF16 with both `position_encoding=none` and RoPE. The representative
`B=1, T=2048, D=128, heads=8` memory test reported 472,942,080 incremental
manual allocated bytes and 34,245,120 strict-Flash allocated bytes.

```text
rtk env CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/gpu/test_cuda_training.py -k missing_native_binary
```

Outcome: 1 passed and 11 deselected in 4.18 s. This is the native CUDA 13.3
bitsandbytes failure gate.

```text
rtk env CUDA_VISIBLE_DEVICES=0 BNB_CUDA_VERSION=130 python -m pytest -q -s tests/gpu/test_cuda_training.py
```

Outcome: 11 passed and 1 skipped in 58.22 s. The override caused the native
CUDA 13.3 negative gate to skip. The same run passed Adam8bit and AdamW8bit
state save/resume, global/AdaGC AMP overflow rollback, activation checkpoint
parity, modern CLS/GEGLU updates, and compile coverage.

The pure Transformer compile measurement used strict Flash, RoPE, GEGLU,
activation checkpointing, FP16, `B=2, T=128, D=64`, four heads and two layers.
It used two warm-ups and five measured forward/backward iterations. Inductor
ran with `mode=default`, `fullgraph=true`, and `dynamic=false`:

| Mode | ms/iteration | Peak allocated | Peak reserved | Graphs / breaks |
| --- | ---: | ---: | ---: | ---: |
| Eager | 11.4560 | 70,368,768 | 90,177,536 | n/a |
| Compiled | 3.8630 | 70,651,392 | 90,177,536 | 1 / 0 |

The observed eager/compiled ratio was 2.9656 for this bounded run. It does not
set a training throughput expectation.

The complete compiled pretraining step used `fullgraph=false` and reported
36.261 s including compilation, 67,718,144 peak allocated bytes, 69,206,016
peak reserved bytes, 10 unique graphs, and 7 graph breaks. The breaks came
from sample-ID scalar extraction, NumPy masking, data-dependent masking
branches, and dynamic `nonzero`. The compiled CLS fine-tuning step reported
13.348 s including compilation, 67,876,352 peak allocated bytes, 69,206,016
peak reserved bytes, one graph, and no graph breaks.

```text
rtk env CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s tests/gpu/test_cuda_inference.py
```

Outcome: 2 passed in 2.89 s. The nonempty case produced 2,200 frames and one
event for a 1.1 s stereo input; the empty-input case retained shaped CPU
outputs.

## Manual and strict-Flash attention diagnostic

Task 12 added `tests/gpu/benchmark_attention.py` because the existing tests
did not time manual and strict Flash under one measurement contract. The
harness uses fresh modules with identical weights and input for each backend.
It resets parameter gradients each iteration and checks output, input-gradient,
and parameter-gradient finiteness.

```text
rtk env CUDA_VISIBLE_DEVICES=0 python tests/gpu/benchmark_attention.py --batch 1 --length 2048 --dimension 128 --heads 8 --warmups 2 --iterations 10 --device cuda:0
```

Exact scope: one attention layer, FP16, no position encoding or padding, eager
execution with compilation disabled, forward plus FP32 square-mean loss plus
backward, two warm-ups and ten measured iterations on one A100-SXM4-40GB.

| Backend | Kernel mode | ms/iteration | Peak allocated | Peak reserved |
| --- | --- | ---: | ---: | ---: |
| `manual` | Legacy manual dense body | 2.810616 | 492,210,176 | 568,328,192 |
| `flash` | Strict `SDPBackend.FLASH_ATTENTION` | 1.388581 | 27,200,000 | 31,457,280 |

Both paths produced finite outputs and gradients. Raw differences were
`6.103515625e-05` maximum and `2.9868801902921405e-06` mean for outputs, plus
`5.960464477539063e-08` maximum and `3.142304194625467e-10` mean for input
gradients. These observations describe one small attention module. They do not
predict model or paper-scale throughput.

## Local distributed results

The final test layout has no `tests/gpu/test_distributed_smoke.py`, and
`tests/gpu/nccl_resume.py` has no `--device cpu` option. The integration suite
contains the equivalent two-rank Gloo safe-point tests:

```text
rtk python -m pytest -q -s tests/integration/test_slurm_launcher.py -k two_rank_gloo
```

Outcome: 6 passed and 34 deselected in 73.29 s. The selection covered one-rank
preemption reduced across two ranks, coordinated checkpoint output, and
rank-common prepare, merge, write, and output-visibility failures.

The two-rank NCCL probe used GPUs 0 and 1:

```text
rtk env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_probe.py --output-dir /tmp/a2v2-task12-nccl-probe.9Z4sW3
```

Outcome: exit 0. Both ranks used NCCL for tensors and Gloo for checkpoint
metadata. Each rank restored its RNG stream, loaded its checkpoint tensor onto
its local CUDA device, and validated the rank-local RNG and topology schemas.

The exact-resume script ran at two and four ranks:

```text
rtk env CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task12-nccl-resume-2.TKsmT6
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/gpu/nccl_resume.py --output-dir /tmp/a2v2-task12-nccl-resume-4.kJRh6V
```

Both commands exited 0. At two ranks, expected and resumed digests matched:

- Rank 0: `c5eab56a9820dde2fd09207ac1d682155442c49271c55b49d14a420472e2a2db`
- Rank 1: `1979a422bb1e9392bef6c66dfb8f30efcab66a0c9988f03b09064d074339eeb6`

Both ranks shared AdaGC digest
`6ad554301ec6f72fafa49579394ac4803ae90f438a679f94728102ef79afc012`.
At four ranks, all expected/resumed pairs matched and all ranks shared AdaGC
digest `767a0d1f5344f7cc60ef49ee2bde4747be5e2799168335c9dcb23bc81ba205e0`.
Every report recorded `first_difference=null`, exact local state, exact
rank-wide state, and valid RNG/topology schemas.

PyTorch warned that the standalone scripts inferred an NCCL device for a
barrier and that `find_unused_parameters=true` found no unused parameter in
these tiny updates. The tests passed; the messages identify a performance and
heterogeneous-mapping limitation in the test launcher, not a failed exactness
check.

## SLURM launcher checks

```text
rtk bash -n scripts/a2v2_slurm_node.sh
rtk bash -n scripts/reproduce_meerkat_slurm.sh
rtk bash scripts/reproduce_meerkat_slurm.sh /datasets/MeerKAT/manifests /shared/runs/meerkat --phase all --nodes 2 --gpus-per-node 4 --job-id dryrun-4815 --dry-run
```

All commands exited 0. The dry run rendered one `srun` task per node, four
torchrun workers per node, world size 8, distinct pretrain and fine-tune run
contracts, checkpoint validation between stages, and one-node/one-GPU final
evaluation. The CPU suite ran the mocked launcher, lock, contract,
preemption, and frozen-driver tests.

## Requirements review

| Gate | Evidence | Result |
| --- | --- | --- |
| Gate 0 frozen control | Fresh 561-test CPU suite, focused controls, frozen hashes and state signatures above | Pass |
| Gate 1 component units | Full unit suite plus 12 CUDA component tests | Pass |
| Gate 2 CPU integration | Full integration suite, fresh-process variable-crop resume, and eight two-rank Gloo preflight cases | Pass |
| Gate 3 bounded CUDA | Strict Flash, activation checkpointing, compile, CLS/GEGLU, AdaGC, both 8-bit optimizers, 2/4-rank engine resume, and 2-rank workflow resume/preflight | Pass on this host |
| Gate 4 SLURM acceptance | Real allocation and site-dependent checks | UNRUN |

The legacy contract kept the checked-in recipes and paper driver hashes, the
default resolved math path, model state signatures, and old-checkpoint tests.
The modern component tests covered RoPE token order, SDPA/manual parity, CLS
masking and targets, sequence classification, packed GEGLU, DeepScaleLM,
AdaGC transaction state, decay scheduling, compile wiring, and SLURM contract
validation. The full CPU run supplied this coverage; the CUDA and distributed
commands exercised the hardware branches.

The task prohibited subagents. The initial self-review missed the
variable-length multiworker crop case and the other Fix Round 1 findings. The
follow-up review supplied those counterexamples; the RED and GREEN evidence
above records their disposition. The environment and unrun gates below remain
release constraints.

## Known limits and unrun gates

- Paper-scale pretraining, fine-tuning, resume, and final evaluation: UNRUN.
- Frozen legacy random-crop bit-exact resume: unsupported by design;
  compatible resume warns for any worker count. No paper-scale legacy resume
  was launched.
- The canonical local eight-A100 reproduction command graph ran in dry-run
  tests; no eight-rank training allocation ran.
- Real one-node SLURM launcher parity: UNRUN.
- Real multi-node SLURM NCCL and Gloo transport: UNRUN.
- Site shared-filesystem visibility, output-lock contention, checkpoint
  atomicity, signal delivery, validated requeue, and scheduler policy: UNRUN
  outside mocks and local Gloo processes.
- Native bitsandbytes CUDA 13.3 binary or source build: UNRUN. The supported
  wheel required `BNB_CUDA_VERSION=130` on this host.
- Full pretraining compile retains seven graph breaks in the bounded dynamic
  run. `fullgraph=false` accepts them; fullgraph model training remains outside
  the supported claim.
- The one-A100 attention and compile timings describe their recorded shapes,
  software, and measured scopes. They do not establish paper-scale speed or
  memory requirements.
