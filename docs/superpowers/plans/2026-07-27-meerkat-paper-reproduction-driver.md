# Eight-GPU MeerKAT reproduction driver implementation plan

Date: 2026-07-27

## Task 1: Lock the shell command contract with failing tests

Add `tests/integration/test_reproduction_driver.py`.

- Create temporary `pretrain.tsv`, `train_0.tsv`, and `valid_0.tsv` files.
- Run the future shell script with `--dry-run`.
- Require one pretraining launch and one fine-tuning launch.
- Require both launches to use `--nproc-per-node=8`, all eight devices, and
  `distributed_training.distributed_world_size=8`.
- Require pretraining `max_tokens=408000`, accumulation 3, fine-tuning
  `max_tokens=960000`, and accumulation 2.
- Require the pretraining checkpoint to feed `--pretrained-checkpoint`.
- Require a final evaluator command and saved JSON report path.
- Require a missing selected manifest to fail before launch.

Run the focused tests and confirm they fail because the script does not exist.

## Task 2: Implement and document the deployment shell

Add `scripts/reproduce_meerkat_paper.sh`.

- Parse two positional paths plus `--fold`, `--fraction`, and `--dry-run`.
- Map the three supported fractions to checked-in recipes and subset suffixes.
- Validate paths, numeric batch controls, the eight-device list, and manifests.
- In a real run, verify the Python package, launcher, CUDA device count, model,
  and memory.
- Capture environment, recipe, and manifest provenance.
- Launch full pretraining with eight ranks and resume from
  `pretrain/checkpoint_last.pt`.
- Launch one full fine-tuning run with eight ranks, initialized from the
  pretraining checkpoint, and resume its own last checkpoint when present.
- Invoke validation and print the final JSON report.
- Explain exact versus approximate batch math in the script comments and
  status output.

Run `bash -n` and the focused dry-run tests.

## Task 3: Lock checkpoint validation with a failing test

In the same integration test file:

- construct a tiny native fine-tuning checkpoint;
- create one labeled validation recording and manifest;
- invoke the future evaluator on CPU;
- require JSON output with checkpoint metadata and all native frame metrics.

Run the focused test and confirm it fails because the evaluator does not
exist.

## Task 4: Implement the evaluation helper

Add `scripts/evaluate_finetuning_checkpoint.py`.

- Parse checkpoint, manifest directory, subset, output, fold, fraction,
  device, evaluation token budget, and worker count.
- Load the native checkpoint strictly through the existing inference loader.
- Replace only validation data/batching fields in the stored immutable config.
- Run the existing native validation function.
- Add hashes, row counts, runtime metadata, eight-GPU batch profile, and scope.
- Atomically save readable JSON and print the report.

Run the focused evaluator test.

## Task 5: Update researcher-facing documentation

Add the driver to `README.md` and `docs/reproducing-paper.md`.

- Give the deployment command.
- State clearly that it is one-fold and eight-rank, not an exact paper run.
- Explain effective-batch calculations and environment tuning knobs.
- Explain resume and final-report locations.

Run documentation/search assertions from the focused test.

## Task 6: Verify the complete change

- Run `bash -n scripts/reproduce_meerkat_paper.sh`.
- Run the script's CPU `--dry-run`.
- Run `pytest -q tests/integration/test_reproduction_driver.py`.
- Run the complete CPU suite.
- Check script permissions and inspect the final diff/file list.
- Perform the verification-before-completion checklist before reporting.
