#!/usr/bin/env bash
#
# Run one practical Animal2Vec 1.0 MeerKAT reproduction on eight A100 GPUs.
#
# This is intentionally an approximate reproduction. The published jobs used
# four distributed ranks; this launcher uses all eight GPUs requested by the
# researcher. Every changed batch quantity is printed, saved, and embedded in
# the final report so that the result cannot be confused with an exact paper
# run. The model, loss, schedules, update horizons, and data split conventions
# still come from the checked-in Animal2Vec 1.0 baseline recipes.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"


usage() {
    cat <<'USAGE'
Usage:
  bash scripts/reproduce_meerkat_paper.sh MANIFEST_DIR OUTPUT_DIR [options]

Options:
  --fold N          One MeerKAT fold to fine-tune and validate (default: 0).
  --fraction VALUE  One label fraction: 100, 025, or 001 (default: 100).
  --dry-run         Validate inputs and print commands without using CUDA.
  -h, --help        Show this help.

The manifest directory must contain pretrain.tsv, the selected train manifest,
and valid_<fold>.tsv. See docs/reproducing-paper.md for batch controls.
USAGE
}


die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}


require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
        || die "${name} must be a positive integer; received ${value@Q}"
}


print_command() {
    # ``%q`` makes the dry-run an executable audit record even when a path
    # contains spaces. No command is evaluated by this function.
    printf 'Launching:'
    printf ' %q' "$@"
    printf '\n'
}


run_logged() {
    local phase="$1"
    local log_path="$2"
    shift 2
    {
        printf '\n[%s] phase=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${phase}"
        printf 'Launching:'
        printf ' %q' "$@"
        printf '\n'
    } | tee -a "${log_path}"
    # With ``pipefail``, a failing training process remains a failure even
    # though tee successfully preserved its output.
    "$@" 2>&1 | tee -a "${log_path}"
}


[[ $# -ge 2 ]] || {
    usage >&2
    exit 2
}

MANIFEST_DIR="$1"
OUTPUT_DIR="$2"
shift 2

FOLD=0
FRACTION=100
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fold)
            [[ $# -ge 2 ]] || die "--fold requires a value"
            FOLD="$2"
            shift 2
            ;;
        --fraction)
            [[ $# -ge 2 ]] || die "--fraction requires a value"
            FRACTION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ "${FOLD}" =~ ^[0-9]+$ ]] || die "--fold must be a non-negative integer"

# Map the paper's three label regimes to their official configuration and
# manifest conventions. Exactly one mapping is selected for this run.
case "${FRACTION}" in
    100)
        FINETUNE_CONFIG="${REPOSITORY_ROOT}/configs/MeerKAT/finetune_mixup_100.yaml"
        TRAIN_SUBSET="train_${FOLD}"
        ;;
    025)
        FINETUNE_CONFIG="${REPOSITORY_ROOT}/configs/MeerKAT/finetune_mixup_025.yaml"
        TRAIN_SUBSET="train_${FOLD}_few_2"
        ;;
    001)
        FINETUNE_CONFIG="${REPOSITORY_ROOT}/configs/MeerKAT/finetune_mixup_001.yaml"
        TRAIN_SUBSET="train_${FOLD}_few_0"
        ;;
    *)
        die "--fraction must be one of 100, 025, or 001"
        ;;
esac

readonly PRETRAIN_CONFIG="${REPOSITORY_ROOT}/configs/MeerKAT/a2v_large_pretrain_best.yaml"
readonly VALID_SUBSET="valid_${FOLD}"

[[ -d "${MANIFEST_DIR}" ]] || die "manifest directory does not exist: ${MANIFEST_DIR}"
MANIFEST_DIR="$(cd -- "${MANIFEST_DIR}" && pwd)"
# ``realpath -m`` permits a new output directory while still resolving a
# stable absolute path for checkpoint metadata and dry-run assertions.
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"

readonly PRETRAIN_MANIFEST="${MANIFEST_DIR}/pretrain.tsv"
readonly TRAIN_MANIFEST="${MANIFEST_DIR}/${TRAIN_SUBSET}.tsv"
readonly VALID_MANIFEST="${MANIFEST_DIR}/${VALID_SUBSET}.tsv"
for required_manifest in \
    "${PRETRAIN_MANIFEST}" \
    "${TRAIN_MANIFEST}" \
    "${VALID_MANIFEST}"
do
    [[ -f "${required_manifest}" ]] \
        || die "required manifest is missing: ${required_manifest}"
done

# The physical device list is explicit because an unnoticed scheduler mask is
# a common cause of world-size mismatches. A rank is one independent process,
# and each rank owns one locally remapped CUDA device.
readonly SELECTED_GPUS="${A2V2_GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "${SELECTED_GPUS}"
[[ ${#GPU_IDS[@]} -eq 8 ]] \
    || die "A2V2_GPUS must contain exactly eight comma-separated device IDs"
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] \
        || die "A2V2_GPUS contains a non-numeric device ID: ${gpu_id@Q}"
    [[ -z "${SEEN_GPU_IDS[${gpu_id}]:-}" ]] \
        || die "A2V2_GPUS contains the duplicate device ID ${gpu_id}"
    SEEN_GPU_IDS["${gpu_id}"]=1
done

readonly PYTHON_BIN="${A2V2_PYTHON:-python}"
if [[ -n "${A2V2_TORCHRUN:-}" ]]; then
    TORCHRUN_COMMAND=("${A2V2_TORCHRUN}")
else
    TORCHRUN_COMMAND=("${PYTHON_BIN}" -m torch.distributed.run)
fi
TRAIN_ENTRY="${A2V2_TRAIN_ENTRY:-a2v2-train}"

# Per-device pretraining memory cannot be pooled across ranks. The measured
# published token budget already reserved about 37.1 GiB on each A100, so it
# remains 408,000. Mathematically, the effective token batch is
#   B_pre = max_tokens × ranks × accumulation
#         = 408,000 × 8 × 3 = 9,792,000,
# which is 1.2 times the published 408,000 × 4 × 5 = 8,160,000.
readonly PRETRAIN_MAX_TOKENS="${A2V2_PRETRAIN_MAX_TOKENS:-408000}"
readonly PRETRAIN_UPDATE_FREQ="${A2V2_PRETRAIN_UPDATE_FREQ:-3}"

# Fine-tuning had much more measured headroom. Increasing its per-rank token
# budget and reducing accumulation gives
#   B_ft = 960,000 × 8 × 2 = 15,360,000,
# essentially identical to the published 426,667 × 4 × 9 = 15,360,012.
# Thus GPUs process larger microbatches without changing the learning rate's
# intended global-batch scale. The defaults remain tunable because real
# recording-length distributions affect allocator peaks.
readonly FINETUNE_MAX_TOKENS="${A2V2_FINETUNE_MAX_TOKENS:-960000}"
readonly FINETUNE_UPDATE_FREQ="${A2V2_FINETUNE_UPDATE_FREQ:-2}"
readonly EVAL_MAX_TOKENS="${A2V2_EVAL_MAX_TOKENS:-320000}"
readonly EVAL_WORKERS="${A2V2_EVAL_WORKERS:-20}"
readonly TRAIN_OMP_NUM_THREADS="${A2V2_OMP_NUM_THREADS:-8}"

require_positive_integer A2V2_PRETRAIN_MAX_TOKENS "${PRETRAIN_MAX_TOKENS}"
require_positive_integer A2V2_PRETRAIN_UPDATE_FREQ "${PRETRAIN_UPDATE_FREQ}"
require_positive_integer A2V2_FINETUNE_MAX_TOKENS "${FINETUNE_MAX_TOKENS}"
require_positive_integer A2V2_FINETUNE_UPDATE_FREQ "${FINETUNE_UPDATE_FREQ}"
require_positive_integer A2V2_EVAL_MAX_TOKENS "${EVAL_MAX_TOKENS}"
require_positive_integer A2V2_OMP_NUM_THREADS "${TRAIN_OMP_NUM_THREADS}"
[[ "${EVAL_WORKERS}" =~ ^[0-9]+$ ]] \
    || die "A2V2_EVAL_WORKERS must be a non-negative integer"

readonly PRETRAIN_DIR="${OUTPUT_DIR}/pretrain"
readonly FINETUNE_DIR="${OUTPUT_DIR}/finetune"
readonly EVALUATION_DIR="${OUTPUT_DIR}/final-evaluation"
readonly FINAL_REPORT="${EVALUATION_DIR}/final-evaluation-report.json"
readonly EVALUATION_TENSORBOARD_DIR="${EVALUATION_DIR}/tensorboard"
readonly PRETRAIN_CHECKPOINT="${PRETRAIN_DIR}/checkpoint_last.pt"
readonly FINETUNE_BEST_CHECKPOINT="${FINETUNE_DIR}/checkpoint_best.pt"
readonly FINETUNE_LAST_CHECKPOINT="${FINETUNE_DIR}/checkpoint_last.pt"
FINETUNE_RESUME_CHECKPOINT=""
if [[ -f "${FINETUNE_LAST_CHECKPOINT}" ]]; then
    FINETUNE_RESUME_CHECKPOINT="${FINETUNE_LAST_CHECKPOINT}"
else
    for candidate in \
        "${FINETUNE_BEST_CHECKPOINT}" \
        "${FINETUNE_DIR}"/checkpoint_[0-9]*.pt \
        "${FINETUNE_DIR}"/checkpoint_epoch_*.pt
    do
        [[ -f "${candidate}" ]] || continue
        candidate_name="$(basename -- "${candidate}")"
        [[ "${candidate_name}" == "checkpoint_best.pt"
            || "${candidate_name}" =~ ^checkpoint_[0-9]+\.pt$
            || "${candidate_name}" =~ ^checkpoint_epoch_[0-9]+\.pt$ ]] \
            || continue
        if [[ -z "${FINETUNE_RESUME_CHECKPOINT}" \
            || "${candidate}" -nt "${FINETUNE_RESUME_CHECKPOINT}" ]]
        then
            FINETUNE_RESUME_CHECKPOINT="${candidate}"
        fi
    done
fi

printf '%s\n' \
    "A2V2 approximate Animal2Vec 1.0 MeerKAT reproduction" \
    "  fold=${FOLD}, label_fraction=${FRACTION}" \
    "  CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}, distributed_training.distributed_world_size=8" \
    "  pretrain: dataset.max_tokens=${PRETRAIN_MAX_TOKENS}, optimization.update_freq=[${PRETRAIN_UPDATE_FREQ}]" \
    "  finetune: dataset.max_tokens=${FINETUNE_MAX_TOKENS}, optimization.update_freq=[${FINETUNE_UPDATE_FREQ}]" \
    "  CPU: OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS}" \
    "  scope: one fold on an eight-rank topology; not an exact paper reproduction"

PREFLIGHT_COMMAND=(
    env
    "CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}"
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/check_reproduction_environment.py"
    --output-dir "${OUTPUT_DIR}"
)
PROBE_COMMAND=(
    env
    "CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}"
    "${TORCHRUN_COMMAND[@]}"
    --standalone
    --nproc-per-node=8
    "${REPOSITORY_ROOT}/tests/gpu/nccl_probe.py"
    --output-dir "${OUTPUT_DIR}/environment/nccl-preflight"
)

if [[ "${DRY_RUN}" == false ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
        || die "Python command is unavailable: ${PYTHON_BIN}"
    command -v "${TORCHRUN_COMMAND[0]}" >/dev/null 2>&1 \
        || die "distributed launcher is unavailable: ${TORCHRUN_COMMAND[0]}"
    if [[ "${TRAIN_ENTRY}" == "a2v2-train" ]]; then
        TRAIN_ENTRY="$(command -v a2v2-train)" \
            || die "a2v2-train is unavailable; install this repository first"
    fi
    [[ -x "${TRAIN_ENTRY}" ]] \
        || die "training entry point is not executable: ${TRAIN_ENTRY}"

    mkdir -p \
        "${OUTPUT_DIR}/environment" \
        "${PRETRAIN_DIR}" \
        "${FINETUNE_DIR}" \
        "${EVALUATION_DIR}"

    "${PYTHON_BIN}" --version > "${OUTPUT_DIR}/environment/python-version.txt" 2>&1
    "${PYTHON_BIN}" -m pip freeze > "${OUTPUT_DIR}/environment/pip-freeze.txt"
    nvidia-smi > "${OUTPUT_DIR}/environment/nvidia-smi.txt"
    sha256sum \
        "${PRETRAIN_MANIFEST}" \
        "${TRAIN_MANIFEST}" \
        "${VALID_MANIFEST}" \
        > "${OUTPUT_DIR}/environment/manifest-sha256.txt"
    sha256sum \
        "${PRETRAIN_CONFIG}" \
        "${FINETUNE_CONFIG}" \
        > "${OUTPUT_DIR}/environment/recipe-sha256.txt"
    {
        printf 'scope=approximate-one-fold-eight-rank\n'
        printf 'fold=%s\n' "${FOLD}"
        printf 'fraction=%s\n' "${FRACTION}"
        printf 'gpus=%s\n' "${SELECTED_GPUS}"
        printf 'pretrain_max_tokens=%s\n' "${PRETRAIN_MAX_TOKENS}"
        printf 'pretrain_update_freq=%s\n' "${PRETRAIN_UPDATE_FREQ}"
        printf 'finetune_max_tokens=%s\n' "${FINETUNE_MAX_TOKENS}"
        printf 'finetune_update_freq=%s\n' "${FINETUNE_UPDATE_FREQ}"
        printf 'omp_num_threads=%s\n' "${TRAIN_OMP_NUM_THREADS}"
    } > "${OUTPUT_DIR}/environment/run-profile.txt"
    run_logged \
        environment-preflight \
        "${OUTPUT_DIR}/environment/preflight.log" \
        "${PREFLIGHT_COMMAND[@]}"
    run_logged \
        distributed-probe \
        "${OUTPUT_DIR}/environment/nccl-probe.log" \
        "${PROBE_COMMAND[@]}"
else
    print_command "${PREFLIGHT_COMMAND[@]}"
    print_command "${PROBE_COMMAND[@]}"
fi

PRETRAIN_COMMAND=(
    env
    "CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS}"
    "${TORCHRUN_COMMAND[@]}"
    --standalone
    --nproc-per-node=8
    "${TRAIN_ENTRY}"
    --config "${PRETRAIN_CONFIG}"
    --override "task.data=${MANIFEST_DIR}"
    --override "checkpoint.save_dir=${PRETRAIN_DIR}"
    --override "distributed_training.distributed_world_size=8"
    --override "dataset.max_tokens=${PRETRAIN_MAX_TOKENS}"
    --override "optimization.update_freq=[${PRETRAIN_UPDATE_FREQ}]"
    --device cuda
)
PRETRAIN_BURN_IN_COMMAND=("${PRETRAIN_COMMAND[@]}" --stop-at-update 1)
PRETRAIN_RESUME_COMMAND=("${PRETRAIN_COMMAND[@]}" --resume "${PRETRAIN_CHECKPOINT}")
CHECKPOINT_PREFLIGHT_COMMAND=(
    "${PREFLIGHT_COMMAND[@]}"
    --checkpoint "${PRETRAIN_CHECKPOINT}"
    --expected-stage pretrain
)

if [[ -f "${PRETRAIN_CHECKPOINT}" ]]; then
    if [[ "${DRY_RUN}" == true ]]; then
        print_command "${CHECKPOINT_PREFLIGHT_COMMAND[@]}"
        print_command "${PRETRAIN_RESUME_COMMAND[@]}"
    else
        run_logged \
            pretraining-checkpoint-validation \
            "${PRETRAIN_DIR}/train.log" \
            "${CHECKPOINT_PREFLIGHT_COMMAND[@]}"
        run_logged \
            pretraining-resume \
            "${PRETRAIN_DIR}/train.log" \
            "${PRETRAIN_RESUME_COMMAND[@]}"
    fi
else
    if [[ "${DRY_RUN}" == true ]]; then
        print_command "${PRETRAIN_BURN_IN_COMMAND[@]}"
        print_command "${CHECKPOINT_PREFLIGHT_COMMAND[@]}"
        print_command "${PRETRAIN_RESUME_COMMAND[@]}"
    else
        run_logged \
            pretraining-burn-in \
            "${PRETRAIN_DIR}/train.log" \
            "${PRETRAIN_BURN_IN_COMMAND[@]}"
        [[ -f "${PRETRAIN_CHECKPOINT}" ]] \
            || die "pretraining burn-in completed without ${PRETRAIN_CHECKPOINT}"
        run_logged \
            pretraining-checkpoint-validation \
            "${PRETRAIN_DIR}/train.log" \
            "${CHECKPOINT_PREFLIGHT_COMMAND[@]}"
        run_logged \
            pretraining-resume \
            "${PRETRAIN_DIR}/train.log" \
            "${PRETRAIN_RESUME_COMMAND[@]}"
    fi
fi

FINETUNE_COMMAND=(
    env
    "CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS}"
    "${TORCHRUN_COMMAND[@]}"
    --standalone
    --nproc-per-node=8
    "${TRAIN_ENTRY}"
    --config "${FINETUNE_CONFIG}"
    --pretrained-checkpoint "${PRETRAIN_CHECKPOINT}"
    --override "task.data=${MANIFEST_DIR}"
    --override "dataset.train_subset=${TRAIN_SUBSET}"
    --override "dataset.valid_subset=${VALID_SUBSET}"
    --override "checkpoint.save_dir=${FINETUNE_DIR}"
    --override "distributed_training.distributed_world_size=8"
    --override "dataset.max_tokens=${FINETUNE_MAX_TOKENS}"
    --override "optimization.update_freq=[${FINETUNE_UPDATE_FREQ}]"
    --device cuda
)
if [[ -n "${FINETUNE_RESUME_CHECKPOINT}" ]]; then
    FINETUNE_COMMAND+=(--resume "${FINETUNE_RESUME_CHECKPOINT}")
    FINETUNE_CHECKPOINT_PREFLIGHT_COMMAND=(
        "${PREFLIGHT_COMMAND[@]}"
        --checkpoint "${FINETUNE_RESUME_CHECKPOINT}"
        --expected-stage finetune
    )
fi

if [[ "${DRY_RUN}" == true ]]; then
    if [[ -n "${FINETUNE_RESUME_CHECKPOINT}" ]]; then
        print_command "${FINETUNE_CHECKPOINT_PREFLIGHT_COMMAND[@]}"
    fi
    print_command "${FINETUNE_COMMAND[@]}"
    EVALUATION_CHECKPOINT="${FINETUNE_BEST_CHECKPOINT}"
else
    if [[ -n "${FINETUNE_RESUME_CHECKPOINT}" ]]; then
        run_logged \
            finetuning-checkpoint-validation \
            "${FINETUNE_DIR}/train.log" \
            "${FINETUNE_CHECKPOINT_PREFLIGHT_COMMAND[@]}"
    fi
    run_logged finetuning "${FINETUNE_DIR}/train.log" "${FINETUNE_COMMAND[@]}"
    if [[ -f "${FINETUNE_BEST_CHECKPOINT}" ]]; then
        EVALUATION_CHECKPOINT="${FINETUNE_BEST_CHECKPOINT}"
    elif [[ -f "${FINETUNE_LAST_CHECKPOINT}" ]]; then
        printf 'WARNING: checkpoint_best.pt is absent; validating checkpoint_last.pt\n' >&2
        EVALUATION_CHECKPOINT="${FINETUNE_LAST_CHECKPOINT}"
    else
        die "fine-tuning completed without a native checkpoint"
    fi
fi

EVALUATION_COMMAND=(
    env
    "CUDA_VISIBLE_DEVICES=${GPU_IDS[0]}"
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/evaluate_finetuning_checkpoint.py"
    --checkpoint "${EVALUATION_CHECKPOINT}"
    --manifest-dir "${MANIFEST_DIR}"
    --subset "${VALID_SUBSET}"
    --output "${FINAL_REPORT}"
    --tensorboard-dir "${EVALUATION_TENSORBOARD_DIR}"
    --fold "${FOLD}"
    --fraction "${FRACTION}"
    --device cuda
    --max-tokens "${EVAL_MAX_TOKENS}"
    --num-workers "${EVAL_WORKERS}"
    --training-world-size 8
    --pretrain-max-tokens "${PRETRAIN_MAX_TOKENS}"
    --pretrain-update-freq "${PRETRAIN_UPDATE_FREQ}"
    --finetune-max-tokens "${FINETUNE_MAX_TOKENS}"
    --finetune-update-freq "${FINETUNE_UPDATE_FREQ}"
)

if [[ "${DRY_RUN}" == true ]]; then
    print_command "${EVALUATION_COMMAND[@]}"
    printf 'Dry-run complete. Final report would be written to %s\n' "${FINAL_REPORT}"
else
    run_logged evaluation "${EVALUATION_DIR}/validation.log" "${EVALUATION_COMMAND[@]}"
    [[ -f "${FINAL_REPORT}" ]] || die "validation did not write ${FINAL_REPORT}"
    printf '\nFinal evaluation report (%s):\n' "${FINAL_REPORT}"
    cat "${FINAL_REPORT}"
fi
