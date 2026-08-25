#!/usr/bin/env bash
#
# Run the full-label modern Animal2Vec MeerKAT benchmark on eight A100 GPUs.
#
# This keeps the proven reproduction driver's data exposure, update horizons,
# checkpoint checks, and final validation flow while selecting the opt-in
# RoPE, FlashAttention, CLS, GEGLU, DeepScaleLM, AdaGC, AdamW8bit, compile,
# and weight-decay-annealing paths.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"


usage() {
    cat <<'USAGE'
Usage:
  bash scripts/animal2vec2_benchmark.sh MANIFEST_DIR OUTPUT_DIR [options]

Options:
  --fold N    One MeerKAT fold to fine-tune and validate (default: 0).
  --dry-run   Validate inputs and print commands without using CUDA.
  -h, --help  Show this help.

This benchmark uses only the full 100% training split. The manifest directory
must contain pretrain.tsv, train_<fold>.tsv, and valid_<fold>.tsv.
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
    "$@" 2>&1 | tee -a "${log_path}"
}


run_json_report() {
    local phase="$1"
    local log_path="$2"
    local report_path="$3"
    local checksum_path="$4"
    local temporary_report="${report_path}.tmp.$$"
    shift 4
    {
        printf '\n[%s] phase=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${phase}"
        printf 'Launching:'
        printf ' %q' "$@"
        printf '\n'
    } | tee -a "${log_path}"
    if "$@" > "${temporary_report}" 2> >(tee -a "${log_path}" >&2); then
        [[ -s "${temporary_report}" ]] \
            || {
                rm -f -- "${temporary_report}"
                die "JSON command returned an empty report"
            }
        if ! env "${BNB_ENV[@]}" \
            "${PYTHON_BIN}" -m json.tool "${temporary_report}" >/dev/null
        then
            rm -f -- "${temporary_report}"
            die "JSON command returned invalid JSON"
        fi
        if [[ -n "${checksum_path}" ]] \
            && ! sha256sum --check --status "${checksum_path}"
        then
            rm -f -- "${temporary_report}"
            die "evaluated checkpoint changed while validation was running"
        fi
        tee -a "${log_path}" < "${temporary_report}"
        mv -f -- "${temporary_report}" "${report_path}"
    else
        local status=$?
        rm -f -- "${temporary_report}"
        return "${status}"
    fi
}


[[ $# -ge 2 ]] || {
    usage >&2
    exit 2
}

MANIFEST_DIR="$1"
OUTPUT_DIR="$2"
shift 2

FOLD=0
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fold)
            [[ $# -ge 2 ]] || die "--fold requires a value"
            FOLD="$2"
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

readonly PRETRAIN_CONFIG="${REPOSITORY_ROOT}/configs/modern/rope_cls_geglu_pretrain.yaml"
readonly FINETUNE_CONFIG="${REPOSITORY_ROOT}/configs/modern/rope_cls_geglu_finetune.yaml"
readonly TRAIN_SUBSET="train_${FOLD}"
readonly VALID_SUBSET="valid_${FOLD}"

[[ -d "${MANIFEST_DIR}" ]] || die "manifest directory does not exist: ${MANIFEST_DIR}"
MANIFEST_DIR="$(cd -- "${MANIFEST_DIR}" && pwd)"
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
SEQUENCE_EVAL_ENTRY="${A2V2_SEQUENCE_EVAL_ENTRY:-a2v2-evaluate-sequence}"

# These values preserve the successful eight-GPU reproduction's effective
# token batches: 9,792,000 for pretraining and 15,360,000 for fine-tuning.
# The modern pretraining partition uses the measured faster 612000 x 2 profile.
readonly PRETRAIN_MAX_TOKENS="${A2V2_PRETRAIN_MAX_TOKENS:-612000}"
readonly PRETRAIN_UPDATE_FREQ="${A2V2_PRETRAIN_UPDATE_FREQ:-2}"
readonly FINETUNE_MAX_TOKENS="${A2V2_FINETUNE_MAX_TOKENS:-960000}"
readonly FINETUNE_UPDATE_FREQ="${A2V2_FINETUNE_UPDATE_FREQ:-2}"
readonly PRETRAIN_MAX_UPDATE=384230
readonly FINETUNE_MAX_UPDATE=30000
readonly CHECKPOINT_ACTIVATIONS=false
readonly EVAL_MAX_TOKENS="${A2V2_EVAL_MAX_TOKENS:-320000}"
readonly EVAL_WORKERS="${A2V2_EVAL_WORKERS:-20}"
readonly TRAIN_OMP_NUM_THREADS="${A2V2_OMP_NUM_THREADS:-8}"
readonly BNB_CUDA_SELECTION="${A2V2_BNB_CUDA_VERSION:-auto}"
readonly TORCHINDUCTOR_CACHE_DIR="${A2V2_TORCHINDUCTOR_CACHE_DIR:-${OUTPUT_DIR}/environment/torchinductor-cache}"
BNB_ENV=()
if [[ "${BNB_CUDA_SELECTION}" == "auto" ]]; then
    BNB_ENV=(-u BNB_CUDA_VERSION)
else
    [[ "${BNB_CUDA_SELECTION}" =~ ^[0-9]+$ ]] \
        || die "A2V2_BNB_CUDA_VERSION must contain only digits or be auto"
    BNB_ENV=("BNB_CUDA_VERSION=${BNB_CUDA_SELECTION}")
fi

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
readonly EVALUATION_CHECKPOINT_METADATA="${EVALUATION_DIR}/evaluation-checkpoint-preflight.json"
readonly EVALUATION_CHECKPOINT_SHA256="${EVALUATION_DIR}/evaluation-checkpoint-sha256.txt"
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
        if [[ -z "${FINETUNE_RESUME_CHECKPOINT}"
            || "${candidate}" -nt "${FINETUNE_RESUME_CHECKPOINT}" ]]
        then
            FINETUNE_RESUME_CHECKPOINT="${candidate}"
        fi
    done
fi

printf '%s\n' \
    "Animal2Vec 2 modern MeerKAT benchmark" \
    "  fold=${FOLD}, label_fraction=100" \
    "  CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}, distributed_training.distributed_world_size=8" \
    "  architecture: RoPE, strict FlashAttention, CLS, packed GEGLU, DeepScaleLM" \
    "  optimization: AdaGC, AdamW8bit, cosine weight-decay annealing" \
    "  compile: torch.compile with persistent cache=${TORCHINDUCTOR_CACHE_DIR}" \
    "  bitsandbytes CUDA selection=${BNB_CUDA_SELECTION}" \
    "  pretrain: dataset.max_tokens=${PRETRAIN_MAX_TOKENS}, optimization.update_freq=[${PRETRAIN_UPDATE_FREQ}], pretrain_max_update=${PRETRAIN_MAX_UPDATE}" \
    "  finetune: dataset.max_tokens=${FINETUNE_MAX_TOKENS}, optimization.update_freq=[${FINETUNE_UPDATE_FREQ}], finetune_max_update=${FINETUNE_MAX_UPDATE}" \
    "  checkpoint_activations=${CHECKPOINT_ACTIVATIONS}, OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS}"

PREFLIGHT_COMMAND=(
    env
    "${BNB_ENV[@]}"
    "CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}"
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/check_reproduction_environment.py"
    --output-dir "${OUTPUT_DIR}"
    --require-bitsandbytes
)
PROBE_COMMAND=(
    env
    "${BNB_ENV[@]}"
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
    if [[ "${SEQUENCE_EVAL_ENTRY}" == "a2v2-evaluate-sequence" ]]; then
        SEQUENCE_EVAL_ENTRY="$(command -v a2v2-evaluate-sequence)" \
            || die "a2v2-evaluate-sequence is unavailable; install this repository first"
    fi
    [[ -x "${SEQUENCE_EVAL_ENTRY}" ]] \
        || die "sequence evaluation entry point is not executable: ${SEQUENCE_EVAL_ENTRY}"

    mkdir -p \
        "${OUTPUT_DIR}/environment" \
        "${TORCHINDUCTOR_CACHE_DIR}" \
        "${PRETRAIN_DIR}" \
        "${FINETUNE_DIR}" \
        "${EVALUATION_DIR}"

    env "${BNB_ENV[@]}" \
        "${PYTHON_BIN}" --version \
        > "${OUTPUT_DIR}/environment/python-version.txt" 2>&1
    env "${BNB_ENV[@]}" \
        "${PYTHON_BIN}" -m pip freeze \
        > "${OUTPUT_DIR}/environment/pip-freeze.txt"
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
        printf 'variant=animal2vec2-modern\n'
        printf 'label_fraction=100\n'
        printf 'fold=%s\n' "${FOLD}"
        printf 'gpus=%s\n' "${SELECTED_GPUS}"
        printf 'position_encoding=rope\n'
        printf 'attention_backend=flash\n'
        printf 'use_cls_token=true\n'
        printf 'classification_head=cls\n'
        printf 'ffn_type=geglu\n'
        printf 'initialization=deepscale_lm\n'
        printf 'gradient_clip_method=adagc\n'
        printf 'optimizer=adamw8bit\n'
        printf 'weight_decay_schedule=cosine\n'
        printf 'torch_compile=true\n'
        printf 'checkpoint_activations=%s\n' "${CHECKPOINT_ACTIVATIONS}"
        printf 'pretrain_max_tokens=%s\n' "${PRETRAIN_MAX_TOKENS}"
        printf 'pretrain_update_freq=%s\n' "${PRETRAIN_UPDATE_FREQ}"
        printf 'pretrain_max_update=%s\n' "${PRETRAIN_MAX_UPDATE}"
        printf 'finetune_max_tokens=%s\n' "${FINETUNE_MAX_TOKENS}"
        printf 'finetune_update_freq=%s\n' "${FINETUNE_UPDATE_FREQ}"
        printf 'finetune_max_update=%s\n' "${FINETUNE_MAX_UPDATE}"
        printf 'bnb_cuda_selection=%s\n' "${BNB_CUDA_SELECTION}"
        printf 'torchinductor_cache_dir=%s\n' "${TORCHINDUCTOR_CACHE_DIR}"
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

MODERN_OVERRIDES=(
    --override "common.torch_compile=true"
    --override "model.position_encoding=rope"
    --override "model.attention_backend=flash"
    --override "model.use_cls_token=true"
    --override "model.ffn_type=geglu"
    --override "model.initialization=deepscale_lm"
    --override "model.modalities.audio.use_alibi_encoder=false"
    --override "model.checkpoint_activations=${CHECKPOINT_ACTIVATIONS}"
    --override "optimization.gradient_clip_method=adagc"
    --override "optimizer._name=adamw8bit"
    --override "optimizer.weight_decay=0.01"
    --override "optimizer.weight_decay_schedule=cosine"
    --override "optimizer.weight_decay_end=0.0"
)
TRAIN_ENV=(
    env
    "${BNB_ENV[@]}"
    "CUDA_VISIBLE_DEVICES=${SELECTED_GPUS}"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
    "OMP_NUM_THREADS=${TRAIN_OMP_NUM_THREADS}"
)
PRETRAIN_COMMAND=(
    "${TRAIN_ENV[@]}"
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
    --override "optimization.max_update=${PRETRAIN_MAX_UPDATE}"
    "${MODERN_OVERRIDES[@]}"
    --override "model.classification_head=frame"
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
    "${TRAIN_ENV[@]}"
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
    --override "optimization.max_update=${FINETUNE_MAX_UPDATE}"
    "${MODERN_OVERRIDES[@]}"
    --override "model.classification_head=cls"
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
    EVALUATION_REFERENCE_CHECKPOINT=""
    if [[ -f "${FINETUNE_LAST_CHECKPOINT}" ]]; then
        EVALUATION_REFERENCE_CHECKPOINT="${FINETUNE_LAST_CHECKPOINT}"
    elif [[ -n "${FINETUNE_RESUME_CHECKPOINT}" \
        && -f "${FINETUNE_RESUME_CHECKPOINT}" ]]
    then
        EVALUATION_REFERENCE_CHECKPOINT="${FINETUNE_RESUME_CHECKPOINT}"
    fi
    if [[ -f "${FINETUNE_BEST_CHECKPOINT}" ]]; then
        EVALUATION_CHECKPOINT="${FINETUNE_BEST_CHECKPOINT}"
        if [[ -n "${EVALUATION_REFERENCE_CHECKPOINT}" \
            && "${EVALUATION_REFERENCE_CHECKPOINT}" != "${FINETUNE_BEST_CHECKPOINT}" ]]
        then
            BEST_CHECKPOINT_COMPATIBILITY_COMMAND=(
                "${PREFLIGHT_COMMAND[@]}"
                --checkpoint "${FINETUNE_BEST_CHECKPOINT}"
                --reference-checkpoint "${EVALUATION_REFERENCE_CHECKPOINT}"
                --expected-stage finetune
            )
            if ! run_logged \
                best-checkpoint-compatibility \
                "${FINETUNE_DIR}/train.log" \
                "${BEST_CHECKPOINT_COMPATIBILITY_COMMAND[@]}"
            then
                printf '%s\n' \
                    "WARNING: checkpoint_best.pt does not match the current run; evaluating ${EVALUATION_REFERENCE_CHECKPOINT}" \
                    >&2
                EVALUATION_CHECKPOINT="${EVALUATION_REFERENCE_CHECKPOINT}"
            fi
        fi
    elif [[ -n "${EVALUATION_REFERENCE_CHECKPOINT}" ]]; then
        printf 'WARNING: checkpoint_best.pt is absent; validating checkpoint_last.pt\n' >&2
        EVALUATION_CHECKPOINT="${EVALUATION_REFERENCE_CHECKPOINT}"
    else
        die "fine-tuning completed without a native checkpoint"
    fi
fi

EVALUATION_COMMAND=(
    env
    "${BNB_ENV[@]}"
    "CUDA_VISIBLE_DEVICES=${GPU_IDS[0]}"
    "${SEQUENCE_EVAL_ENTRY}"
    "${EVALUATION_CHECKPOINT}"
    --trust-checkpoint
    --config "${FINETUNE_CONFIG}"
    --override "task.data=${MANIFEST_DIR}"
    --override "dataset.valid_subset=${VALID_SUBSET}"
    --override "dataset.max_tokens=${EVAL_MAX_TOKENS}"
    --override "dataset.num_workers=${EVAL_WORKERS}"
    --device cuda
)
EVALUATION_CHECKPOINT_PREFLIGHT_COMMAND=(
    "${PREFLIGHT_COMMAND[@]}"
    --checkpoint "${EVALUATION_CHECKPOINT}"
    --expected-stage finetune
)

if [[ "${DRY_RUN}" == true ]]; then
    print_command "${EVALUATION_COMMAND[@]}"
    printf 'Dry-run complete. Final report would be written to %s\n' "${FINAL_REPORT}"
else
    sha256sum "${EVALUATION_CHECKPOINT}" > "${EVALUATION_CHECKPOINT_SHA256}"
    run_json_report \
        evaluation-checkpoint-validation \
        "${EVALUATION_DIR}/validation.log" \
        "${EVALUATION_CHECKPOINT_METADATA}" \
        "" \
        "${EVALUATION_CHECKPOINT_PREFLIGHT_COMMAND[@]}"
    run_json_report \
        evaluation \
        "${EVALUATION_DIR}/validation.log" \
        "${FINAL_REPORT}" \
        "${EVALUATION_CHECKPOINT_SHA256}" \
        "${EVALUATION_COMMAND[@]}"
    [[ -f "${FINAL_REPORT}" ]] || die "validation did not write ${FINAL_REPORT}"
    printf '\nFinal evaluation report (%s):\n' "${FINAL_REPORT}"
    cat "${FINAL_REPORT}"
fi
