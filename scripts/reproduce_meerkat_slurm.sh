#!/usr/bin/env bash
#SBATCH --signal=B:USR1@300
#
# Run the MeerKAT pretraining, fine-tuning, and final evaluation stages under
# an existing homogeneous SLURM allocation. Site partition/account/QoS policy
# intentionally stays outside this reusable script.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly NODE_LAUNCHER="${SCRIPT_DIR}/a2v2_slurm_node.sh"
readonly EVALUATOR="${SCRIPT_DIR}/evaluate_finetuning_checkpoint.py"
readonly CHECKPOINT_VALIDATOR_CODE='import sys
from pathlib import Path
from collections.abc import Mapping
from a2v2.training import load_checkpoint
path = Path(sys.argv[1]).resolve()
expected_stage = sys.argv[2]
expected_world = int(sys.argv[3])
checkpoint = load_checkpoint(path, map_location="cpu")
stage = checkpoint.get("stage")
if stage != expected_stage:
    raise SystemExit(f"checkpoint {path} has stage {stage!r}; expected {expected_stage!r}")
rng = checkpoint.get("rng_state")
if expected_world > 1:
    if not isinstance(rng, Mapping) or int(rng.get("world_size", -1)) != expected_world:
        raise SystemExit(f"checkpoint {path} RNG world size does not equal {expected_world}")
    topology = checkpoint.get("topology")
    if not isinstance(topology, Mapping) or int(topology.get("world_size", -1)) != expected_world:
        raise SystemExit(f"checkpoint {path} topology world size does not equal {expected_world}")
print(f"validated {expected_stage} checkpoint: {path}")'
readonly SLURM_HELPER_CODE='from a2v2.slurm import main; raise SystemExit(main())'
readonly LOCK_RECOVERY_CODE='import sys
from a2v2.slurm import recover_output_lock
owner = recover_output_lock(sys.argv[1], expected_fingerprint=sys.argv[2])
print(f"recovered output lock for run {owner.run_id!r} (job {owner.job_id!r})")'


usage() {
    cat <<'USAGE'
Usage:
  bash scripts/reproduce_meerkat_slurm.sh MANIFEST_DIR OUTPUT_DIR [options]

Stages:
  --phase VALUE             all, pretrain, finetune, or evaluate (default: all)
  --fold N                  MeerKAT fold (default: 0)
  --fraction VALUE          100, 025, or 001 (default: 100)
  --pretrain-config PATH    Override the checked-in pretraining recipe
  --finetune-config PATH    Override the selected fine-tuning recipe
  --pretrain-checkpoint P   Intended pretraining handoff checkpoint
  --pretrain-resume P       Explicit pretraining-stage resume checkpoint
  --finetune-resume P       Explicit fine-tuning-stage resume checkpoint

Topology/rendezvous:
  --nodes N
  --gpus-per-node N
  --job-id ID
  --master-addr HOST
  --master-port PORT
  --rdzv-endpoint HOST:PORT
  --run-id ID

Recovery and inspection:
  --recover-lock PHASE:FINGERPRINT
                            Explicitly recover one proven-dead Task 9 lock
  --dry-run                 Render exact commands without SLURM, CUDA, or data
  -h, --help

The #SBATCH header requests SIGUSR1 with five minutes of lead time. This script
does not call scontrol requeue. It returns exit 75 only after validating the
stage checkpoint, leaving retry/requeue policy to the site or submitter.
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


run_or_render() {
    if [[ "${DRY_RUN}" == true ]]; then
        print_command "$@"
        return 0
    fi
    "$@"
}


validate_checkpoint() {
    local checkpoint="$1"
    local stage="$2"
    local expected_world="$3"
    local command=(
        env
        "PYTHONPATH=${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        "${PYTHON_BIN}"
        -c "${CHECKPOINT_VALIDATOR_CODE}"
        "${checkpoint}"
        "${stage}"
        "${expected_world}"
    )
    if [[ "${DRY_RUN}" == true ]]; then
        print_command "${command[@]}"
        return 0
    fi
    [[ -f "${checkpoint}" ]] \
        || die "required ${stage} checkpoint is missing: ${checkpoint}"
    "${command[@]}"
}


recover_lock() {
    local phase="$1"
    local fingerprint="$2"
    local directory
    case "${phase}" in
        pretrain) directory="${PRETRAIN_DIR}" ;;
        finetune) directory="${FINETUNE_DIR}" ;;
        *) die "--recover-lock phase must be pretrain or finetune" ;;
    esac
    local command=(
        env
        "PYTHONPATH=${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        "${PYTHON_BIN}"
        -c "${LOCK_RECOVERY_CODE}"
        "${directory}"
        "${fingerprint}"
    )
    run_or_render "${command[@]}"
}


ACTIVE_STEP_PID=""
PREEMPTION_REQUESTED=false
LAST_VALIDATED_MARKER=""
LAST_VALIDATED_CHECKPOINT=""


forward_usr1() {
    PREEMPTION_REQUESTED=true
    if [[ -n "${ACTIVE_STEP_PID}" ]] && kill -0 "${ACTIVE_STEP_PID}" 2>/dev/null; then
        kill -USR1 "${ACTIVE_STEP_PID}"
    fi
}


forward_term() {
    if [[ -n "${ACTIVE_STEP_PID}" ]] && kill -0 "${ACTIVE_STEP_PID}" 2>/dev/null; then
        kill -TERM "${ACTIVE_STEP_PID}"
    fi
}


wait_for_step() {
    local status
    "$@" &
    ACTIVE_STEP_PID=$!
    while true; do
        if wait "${ACTIVE_STEP_PID}"; then
            status=0
        else
            status=$?
        fi
        if kill -0 "${ACTIVE_STEP_PID}" 2>/dev/null; then
            continue
        fi
        break
    done
    ACTIVE_STEP_PID=""
    return "${status}"
}


checkpoint_signature() {
    local checkpoint="$1"
    if [[ -e "${checkpoint}" ]]; then
        local digest
        digest="$(sha256sum -- "${checkpoint}")" \
            || die "cannot fingerprint checkpoint before launch: ${checkpoint}"
        digest="${digest%% *}"
        printf 'sha256:%s\n' "${digest}"
    else
        printf 'missing\n'
    fi
}


trap forward_usr1 USR1
trap forward_term TERM

[[ $# -ge 2 ]] || {
    usage >&2
    exit 2
}

MANIFEST_DIR="$(realpath -m -- "$1")"
OUTPUT_DIR="$(realpath -m -- "$2")"
shift 2

PHASE=all
FOLD=0
FRACTION=100
DRY_RUN=false
NODES_OVERRIDE=""
GPUS_PER_NODE_OVERRIDE=""
JOB_ID_OVERRIDE=""
MASTER_ADDR=""
MASTER_PORT=""
RDZV_ENDPOINT=""
RUN_ID_OVERRIDE=""
PRETRAIN_CONFIG_OVERRIDE=""
FINETUNE_CONFIG_OVERRIDE=""
PRETRAIN_CHECKPOINT_OVERRIDE=""
PRETRAIN_RESUME_OVERRIDE=""
FINETUNE_RESUME_OVERRIDE=""
RECOVERY_REQUESTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase|--fold|--fraction|--nodes|--gpus-per-node|--job-id|--master-addr|--master-port|--rdzv-endpoint|--run-id|--pretrain-config|--finetune-config|--pretrain-checkpoint|--pretrain-resume|--finetune-resume|--recover-lock)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            option="$1"
            value="$2"
            case "${option}" in
                --phase) PHASE="${value}" ;;
                --fold) FOLD="${value}" ;;
                --fraction) FRACTION="${value}" ;;
                --nodes) NODES_OVERRIDE="${value}" ;;
                --gpus-per-node) GPUS_PER_NODE_OVERRIDE="${value}" ;;
                --job-id) JOB_ID_OVERRIDE="${value}" ;;
                --master-addr) MASTER_ADDR="${value}" ;;
                --master-port) MASTER_PORT="${value}" ;;
                --rdzv-endpoint) RDZV_ENDPOINT="${value}" ;;
                --run-id) RUN_ID_OVERRIDE="${value}" ;;
                --pretrain-config) PRETRAIN_CONFIG_OVERRIDE="${value}" ;;
                --finetune-config) FINETUNE_CONFIG_OVERRIDE="${value}" ;;
                --pretrain-checkpoint) PRETRAIN_CHECKPOINT_OVERRIDE="${value}" ;;
                --pretrain-resume) PRETRAIN_RESUME_OVERRIDE="${value}" ;;
                --finetune-resume) FINETUNE_RESUME_OVERRIDE="${value}" ;;
                --recover-lock) RECOVERY_REQUESTS+=("${value}") ;;
            esac
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

case "${PHASE}" in
    all|pretrain|finetune|evaluate) ;;
    *) die "--phase must be all, pretrain, finetune, or evaluate" ;;
esac
[[ "${FOLD}" =~ ^[0-9]+$ ]] || die "--fold must be a non-negative integer"
case "${FRACTION}" in
    100)
        DEFAULT_FINETUNE_CONFIG="${REPOSITORY_ROOT}/configs/MeerKAT/finetune_mixup_100.yaml"
        TRAIN_SUBSET="train_${FOLD}"
        ;;
    025)
        DEFAULT_FINETUNE_CONFIG="${REPOSITORY_ROOT}/configs/MeerKAT/finetune_mixup_025.yaml"
        TRAIN_SUBSET="train_${FOLD}_few_2"
        ;;
    001)
        DEFAULT_FINETUNE_CONFIG="${REPOSITORY_ROOT}/configs/MeerKAT/finetune_mixup_001.yaml"
        TRAIN_SUBSET="train_${FOLD}_few_0"
        ;;
    *) die "--fraction must be one of 100, 025, or 001" ;;
esac

PRETRAIN_CONFIG="${PRETRAIN_CONFIG_OVERRIDE:-${REPOSITORY_ROOT}/configs/MeerKAT/a2v_large_pretrain_best.yaml}"
FINETUNE_CONFIG="${FINETUNE_CONFIG_OVERRIDE:-${DEFAULT_FINETUNE_CONFIG}}"
case "${PHASE}" in
    pretrain)
        [[ -f "${PRETRAIN_CONFIG}" ]] \
            || die "pretraining config does not exist: ${PRETRAIN_CONFIG}"
        ;;
    finetune|evaluate)
        [[ -f "${FINETUNE_CONFIG}" ]] \
            || die "fine-tuning config does not exist: ${FINETUNE_CONFIG}"
        ;;
    all)
        [[ -f "${PRETRAIN_CONFIG}" ]] \
            || die "pretraining config does not exist: ${PRETRAIN_CONFIG}"
        [[ -f "${FINETUNE_CONFIG}" ]] \
            || die "fine-tuning config does not exist: ${FINETUNE_CONFIG}"
        ;;
esac

if [[ "${DRY_RUN}" == false ]]; then
    for variable in SLURM_JOB_ID SLURM_JOB_NUM_NODES SLURM_GPUS_ON_NODE SLURM_JOB_NODELIST
    do
        [[ -n "${!variable:-}" ]] \
            || die "missing required environment variable ${variable}"
    done
fi

JOB_ID="${JOB_ID_OVERRIDE:-${SLURM_JOB_ID:-dry-run}}"
NODES="${NODES_OVERRIDE:-${SLURM_JOB_NUM_NODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE_OVERRIDE:-${SLURM_GPUS_ON_NODE:-1}}"
[[ -n "${JOB_ID}" ]] || die "job ID must be nonempty"
require_positive_integer "node count" "${NODES}"
require_positive_integer "GPUs per node" "${GPUS_PER_NODE}"

if [[ "${DRY_RUN}" == false ]]; then
    [[ "${SLURM_JOB_ID}" == "${JOB_ID}" ]] \
        || die "--job-id ${JOB_ID@Q} conflicts with SLURM_JOB_ID=${SLURM_JOB_ID@Q}"
    [[ "${SLURM_JOB_NUM_NODES}" == "${NODES}" ]] \
        || die "--nodes ${NODES} conflicts with SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
    [[ "${SLURM_GPUS_ON_NODE}" == "${GPUS_PER_NODE}" ]] \
        || die "--gpus-per-node ${GPUS_PER_NODE} conflicts with SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
    if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
        gpu_spec="${SLURM_GPUS_PER_NODE}"
        if [[ "${gpu_spec}" != "${GPUS_PER_NODE}" \
            && "${gpu_spec}" != "${GPUS_PER_NODE}(x${NODES})" \
            && "${gpu_spec}" != "gpu:${GPUS_PER_NODE}" \
            && "${gpu_spec}" != "gpu:${GPUS_PER_NODE}(x${NODES})" \
            && ! "${gpu_spec}" =~ ^gpu:[^,:]+:${GPUS_PER_NODE}(\\(x${NODES}\\))?$ ]]
        then
            die "SLURM launch requires homogeneous GPU counts; SLURM_GPUS_PER_NODE=${gpu_spec@Q} does not describe ${GPUS_PER_NODE} GPU(s) on each of ${NODES} nodes"
        fi
    fi
fi

WORLD_SIZE=$((NODES * GPUS_PER_NODE))
if [[ -n "${RDZV_ENDPOINT}" ]]; then
    [[ -z "${MASTER_ADDR}" && -z "${MASTER_PORT}" ]] \
        || die "--rdzv-endpoint cannot be combined with --master-addr or --master-port"
    if [[ "${RDZV_ENDPOINT}" =~ ^\[[^]]+\]:([0-9]+)$ ]]; then
        MASTER_ADDR="${RDZV_ENDPOINT%:*}"
        MASTER_PORT="${BASH_REMATCH[1]}"
    elif [[ "${RDZV_ENDPOINT}" =~ ^[^:[:space:]]+:([0-9]+)$ ]]; then
        MASTER_ADDR="${RDZV_ENDPOINT%:*}"
        MASTER_PORT="${BASH_REMATCH[1]}"
    else
        die "--rdzv-endpoint must use HOST:PORT syntax (bracket IPv6 literals)"
    fi
else
    if [[ -z "${MASTER_ADDR}" ]]; then
        if [[ "${DRY_RUN}" == true ]]; then
            MASTER_ADDR="dry-run-master"
        else
            command -v scontrol >/dev/null 2>&1 \
                || die "scontrol is unavailable; supply --master-addr explicitly"
            mapfile -t ALLOCATED_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
            [[ ${#ALLOCATED_HOSTS[@]} -eq "${NODES}" ]] \
                || die "scontrol resolved ${#ALLOCATED_HOSTS[@]} host(s) from SLURM_JOB_NODELIST; expected ${NODES}"
            [[ -n "${ALLOCATED_HOSTS[0]}" ]] \
                || die "scontrol did not resolve a first rendezvous host"
            MASTER_ADDR="${ALLOCATED_HOSTS[0]}"
        fi
    fi
    if [[ -z "${MASTER_PORT}" ]]; then
        job_checksum="$(printf '%s' "${JOB_ID}" | cksum)"
        job_checksum="${job_checksum%% *}"
        MASTER_PORT=$((15000 + job_checksum % 20000))
    fi
    RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
fi
require_positive_integer "rendezvous port" "${MASTER_PORT}"
(( MASTER_PORT <= 65535 )) \
    || die "rendezvous port must be at most 65535; received ${MASTER_PORT}"

RUN_ID="${RUN_ID_OVERRIDE:-${JOB_ID}-meerkat-f${FOLD}-${FRACTION}}"
[[ -n "${RUN_ID}" ]] || die "--run-id must be nonempty"
PYTHON_BIN="${A2V2_PYTHON:-python}"
CONTRACT_PYTHON="${A2V2_CONTRACT_PYTHON:-${PYTHON_BIN}}"
SRUN_BIN="${A2V2_SRUN:-srun}"
TRAIN_ENTRY="${A2V2_TRAIN_ENTRY:-a2v2-train}"
PRETRAIN_MAX_TOKENS="${A2V2_PRETRAIN_MAX_TOKENS:-408000}"
PRETRAIN_UPDATE_FREQ="${A2V2_PRETRAIN_UPDATE_FREQ:-3}"
FINETUNE_MAX_TOKENS="${A2V2_FINETUNE_MAX_TOKENS:-960000}"
FINETUNE_UPDATE_FREQ="${A2V2_FINETUNE_UPDATE_FREQ:-2}"
EVAL_MAX_TOKENS="${A2V2_EVAL_MAX_TOKENS:-320000}"
EVAL_WORKERS="${A2V2_EVAL_WORKERS:-20}"
OMP_NUM_THREADS="${A2V2_OMP_NUM_THREADS:-8}"
for control in \
    PRETRAIN_MAX_TOKENS \
    PRETRAIN_UPDATE_FREQ \
    FINETUNE_MAX_TOKENS \
    FINETUNE_UPDATE_FREQ \
    EVAL_MAX_TOKENS \
    OMP_NUM_THREADS
do
    require_positive_integer "${control}" "${!control}"
done
[[ "${EVAL_WORKERS}" =~ ^[0-9]+$ ]] \
    || die "A2V2_EVAL_WORKERS must be a non-negative integer"

VALID_SUBSET="valid_${FOLD}"
TRAIN_MANIFEST="${MANIFEST_DIR}/${TRAIN_SUBSET}.tsv"
VALID_MANIFEST="${MANIFEST_DIR}/${VALID_SUBSET}.tsv"
PRETRAIN_OVERRIDES=(
    "task.data=${MANIFEST_DIR}"
    "checkpoint.save_dir=${OUTPUT_DIR}/pretrain"
    "distributed_training.distributed_world_size=${WORLD_SIZE}"
    "dataset.max_tokens=${PRETRAIN_MAX_TOKENS}"
    "optimization.update_freq=[${PRETRAIN_UPDATE_FREQ}]"
)
PRETRAIN_SUBSET="pretrain"
if [[ "${PHASE}" == pretrain || "${PHASE}" == all ]]; then
    PRETRAIN_SUBSET_COMMAND=(
        env
        "PYTHONPATH=${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        "${CONTRACT_PYTHON}"
        -c "${SLURM_HELPER_CODE}"
        train-subset
        --config "${PRETRAIN_CONFIG}"
    )
    for override in "${PRETRAIN_OVERRIDES[@]}"; do
        PRETRAIN_SUBSET_COMMAND+=(--override "${override}")
    done
    PRETRAIN_SUBSET="$("${PRETRAIN_SUBSET_COMMAND[@]}")" \
        || die "could not resolve dataset.train_subset from the pretraining config"
    [[ -n "${PRETRAIN_SUBSET}" && "${PRETRAIN_SUBSET}" != */* ]] \
        || die "pretraining dataset.train_subset must be one manifest stem"
fi
PRETRAIN_MANIFEST="${MANIFEST_DIR}/${PRETRAIN_SUBSET}.tsv"
FINETUNE_OVERRIDES=(
    "task.data=${MANIFEST_DIR}"
    "dataset.train_subset=${TRAIN_SUBSET}"
    "dataset.valid_subset=${VALID_SUBSET}"
    "checkpoint.save_dir=${OUTPUT_DIR}/finetune"
    "distributed_training.distributed_world_size=${WORLD_SIZE}"
    "dataset.max_tokens=${FINETUNE_MAX_TOKENS}"
    "optimization.update_freq=[${FINETUNE_UPDATE_FREQ}]"
    "model.checkpoint_activations=true"
)
REQUIRED_MANIFESTS=()
case "${PHASE}" in
    all) REQUIRED_MANIFESTS=("${PRETRAIN_MANIFEST}" "${TRAIN_MANIFEST}" "${VALID_MANIFEST}") ;;
    pretrain) REQUIRED_MANIFESTS=("${PRETRAIN_MANIFEST}") ;;
    finetune) REQUIRED_MANIFESTS=("${TRAIN_MANIFEST}" "${VALID_MANIFEST}") ;;
    evaluate) REQUIRED_MANIFESTS=("${TRAIN_MANIFEST}" "${VALID_MANIFEST}") ;;
esac

if [[ -n "${PRETRAIN_CHECKPOINT_OVERRIDE}" ]]; then
    PRETRAIN_CHECKPOINT="$(realpath -m -- "${PRETRAIN_CHECKPOINT_OVERRIDE}")"
    [[ "$(basename -- "${PRETRAIN_CHECKPOINT}")" == checkpoint_last.pt ]] \
        || die "--pretrain-checkpoint must name checkpoint_last.pt so pretraining and fine-tuning share one unambiguous handoff"
    PRETRAIN_DIR="$(dirname -- "${PRETRAIN_CHECKPOINT}")"
else
    PRETRAIN_DIR="${OUTPUT_DIR}/pretrain"
    PRETRAIN_CHECKPOINT="${PRETRAIN_DIR}/checkpoint_last.pt"
fi
PRETRAIN_OVERRIDES[1]="checkpoint.save_dir=${PRETRAIN_DIR}"
FINETUNE_DIR="${OUTPUT_DIR}/finetune"
EVALUATION_DIR="${OUTPUT_DIR}/final-evaluation"
[[ "${PRETRAIN_DIR}" != "${FINETUNE_DIR}" ]] \
    || die "pretraining and fine-tuning output directories must be distinct: ${PRETRAIN_DIR}"
FINETUNE_LAST_CHECKPOINT="${FINETUNE_DIR}/checkpoint_last.pt"
FINAL_REPORT="${EVALUATION_DIR}/final-evaluation-report.json"
EVALUATION_TENSORBOARD_DIR="${EVALUATION_DIR}/tensorboard"

if [[ "${DRY_RUN}" == false ]]; then
    [[ -d "${MANIFEST_DIR}" ]] || die "manifest directory does not exist: ${MANIFEST_DIR}"
    for manifest in "${REQUIRED_MANIFESTS[@]}"; do
        [[ -f "${manifest}" ]] || die "required manifest is missing: ${manifest}"
    done
    command -v "${SRUN_BIN}" >/dev/null 2>&1 \
        || die "srun command is unavailable: ${SRUN_BIN}"
    mkdir -p "${PRETRAIN_DIR}" "${FINETUNE_DIR}" "${EVALUATION_DIR}"
fi

LAUNCH_CONTRACT_COMMAND=(
    env
    "PYTHONPATH=${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    "${CONTRACT_PYTHON}"
    -c "${SLURM_HELPER_CODE}"
    launcher-contracts
    --job-id "${JOB_ID}"
    --nodes "${NODES}"
    --processes-per-node "${GPUS_PER_NODE}"
    --rendezvous-endpoint "${RDZV_ENDPOINT}"
)
if [[ "${PHASE}" == pretrain || "${PHASE}" == all ]]; then
    LAUNCH_CONTRACT_COMMAND+=(
        --phase pretrain
        --pretrain-output-directory "${PRETRAIN_DIR}"
        --pretrain-config "${PRETRAIN_CONFIG}"
        --pretrain-manifest-entry "${PRETRAIN_SUBSET}.tsv=${PRETRAIN_MANIFEST}"
    )
    for override in "${PRETRAIN_OVERRIDES[@]}"; do
        LAUNCH_CONTRACT_COMMAND+=(--pretrain-override "${override}")
    done
fi
if [[ "${PHASE}" == finetune || "${PHASE}" == evaluate || "${PHASE}" == all ]]; then
    LAUNCH_CONTRACT_COMMAND+=(
        --phase finetune
        --finetune-output-directory "${FINETUNE_DIR}"
        --finetune-config "${FINETUNE_CONFIG}"
        --finetune-manifest-entry "${TRAIN_SUBSET}.tsv=${TRAIN_MANIFEST}"
        --finetune-manifest-entry "${VALID_SUBSET}.tsv=${VALID_MANIFEST}"
    )
    for override in "${FINETUNE_OVERRIDES[@]}"; do
        LAUNCH_CONTRACT_COMMAND+=(--finetune-override "${override}")
    done
fi
if [[ "${DRY_RUN}" == true ]]; then
    LAUNCH_CONTRACT_COMMAND+=(--allow-missing-manifests)
fi
LAUNCH_CONTRACT_OUTPUT="$("${LAUNCH_CONTRACT_COMMAND[@]}")" \
    || die "could not construct canonical phase RunContracts"
mapfile -t LAUNCH_CONTRACT_FIELDS <<< "${LAUNCH_CONTRACT_OUTPUT}"
case "${PHASE}" in
    pretrain)
        [[ ${#LAUNCH_CONTRACT_FIELDS[@]} -eq 4 ]] \
            || die "canonical pretraining RunContract helper returned an incomplete record"
        PRETRAIN_MANIFEST_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[0]}"
        PRETRAIN_CONTRACT_JSON="${LAUNCH_CONTRACT_FIELDS[1]}"
        PRETRAIN_CONTRACT_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[2]}"
        PRETRAIN_CONFIG_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[3]}"
        ;;
    finetune|evaluate)
        [[ ${#LAUNCH_CONTRACT_FIELDS[@]} -eq 4 ]] \
            || die "canonical fine-tuning RunContract helper returned an incomplete record"
        FINETUNE_MANIFEST_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[0]}"
        FINETUNE_CONTRACT_JSON="${LAUNCH_CONTRACT_FIELDS[1]}"
        FINETUNE_CONTRACT_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[2]}"
        FINETUNE_CONFIG_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[3]}"
        ;;
    all)
        [[ ${#LAUNCH_CONTRACT_FIELDS[@]} -eq 8 ]] \
            || die "canonical phase RunContract helper returned an incomplete record"
        PRETRAIN_MANIFEST_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[0]}"
        PRETRAIN_CONTRACT_JSON="${LAUNCH_CONTRACT_FIELDS[1]}"
        PRETRAIN_CONTRACT_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[2]}"
        PRETRAIN_CONFIG_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[3]}"
        FINETUNE_MANIFEST_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[4]}"
        FINETUNE_CONTRACT_JSON="${LAUNCH_CONTRACT_FIELDS[5]}"
        FINETUNE_CONTRACT_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[6]}"
        FINETUNE_CONFIG_FINGERPRINT="${LAUNCH_CONTRACT_FIELDS[7]}"
        ;;
esac

for recovery in "${RECOVERY_REQUESTS[@]}"; do
    recovery_phase="${recovery%%:*}"
    recovery_fingerprint="${recovery#*:}"
    [[ "${recovery_phase}" != "${recovery}" ]] \
        || die "--recover-lock requires PHASE:FINGERPRINT"
    [[ "${recovery_fingerprint}" =~ ^[0-9a-f]{64}$ ]] \
        || die "--recover-lock fingerprint must contain exactly 64 lowercase hexadecimal characters"
    recover_lock "${recovery_phase}" "${recovery_fingerprint}"
done

PRETRAIN_RESUME="${PRETRAIN_RESUME_OVERRIDE}"
if [[ -z "${PRETRAIN_RESUME}" && -f "${PRETRAIN_CHECKPOINT}" ]]; then
    PRETRAIN_RESUME="${PRETRAIN_CHECKPOINT}"
fi
FINETUNE_RESUME="${FINETUNE_RESUME_OVERRIDE}"
if [[ -z "${FINETUNE_RESUME}" && -f "${FINETUNE_LAST_CHECKPOINT}" ]]; then
    FINETUNE_RESUME="${FINETUNE_LAST_CHECKPOINT}"
fi


training_srun_command() {
    local phase="$1"
    local output_directory="$2"
    local config="$3"
    local run_id="$4"
    local resume_checkpoint="$5"
    local pretrained_checkpoint="$6"
    shift 6
    local training_arguments=("$@")
    local manifest_fingerprint
    local run_contract_json
    local contract_fingerprint
    local config_fingerprint
    case "${phase}" in
        pretrain)
            manifest_fingerprint="${PRETRAIN_MANIFEST_FINGERPRINT}"
            run_contract_json="${PRETRAIN_CONTRACT_JSON}"
            contract_fingerprint="${PRETRAIN_CONTRACT_FINGERPRINT}"
            config_fingerprint="${PRETRAIN_CONFIG_FINGERPRINT}"
            ;;
        finetune)
            manifest_fingerprint="${FINETUNE_MANIFEST_FINGERPRINT}"
            run_contract_json="${FINETUNE_CONTRACT_JSON}"
            contract_fingerprint="${FINETUNE_CONTRACT_FINGERPRINT}"
            config_fingerprint="${FINETUNE_CONFIG_FINGERPRINT}"
            ;;
        *) die "training phase has no canonical contract: ${phase}" ;;
    esac
    local command=(
        "${SRUN_BIN}"
        "--nodes=${NODES}"
        "--ntasks=${NODES}"
        --ntasks-per-node=1
        "--gpus-per-node=${GPUS_PER_NODE}"
        --kill-on-bad-exit=1
        --signal=USR1@300
        "--export=ALL,OMP_NUM_THREADS=${OMP_NUM_THREADS},PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        "${NODE_LAUNCHER}"
        "${phase}"
        --job-id "${JOB_ID}"
        --nodes "${NODES}"
        --gpus-per-node "${GPUS_PER_NODE}"
        --master-addr "${MASTER_ADDR}"
        --master-port "${MASTER_PORT}"
        --run-id "${run_id}"
        --manifest-dir "${MANIFEST_DIR}"
        --manifest-fingerprint "${manifest_fingerprint}"
        --run-contract-json "${run_contract_json}"
        --contract-fingerprint "${contract_fingerprint}"
        --config-fingerprint "${config_fingerprint}"
        --output-dir "${output_directory}"
        --config "${config}"
    )
    if [[ -n "${resume_checkpoint}" ]]; then
        command+=(--resume-checkpoint "${resume_checkpoint}")
    fi
    if [[ -n "${pretrained_checkpoint}" ]]; then
        command+=(--pretrained-checkpoint "${pretrained_checkpoint}")
    fi
    command+=(-- "${training_arguments[@]}")
    printf '%s\0' "${command[@]}"
}


completion_record_command() {
    local action="$1"
    local phase="$2"
    local checkpoint="$3"
    local contract_json
    local contract_fingerprint
    local config_fingerprint
    case "${phase}" in
        pretrain)
            contract_json="${PRETRAIN_CONTRACT_JSON}"
            contract_fingerprint="${PRETRAIN_CONTRACT_FINGERPRINT}"
            config_fingerprint="${PRETRAIN_CONFIG_FINGERPRINT}"
            ;;
        finetune)
            contract_json="${FINETUNE_CONTRACT_JSON}"
            contract_fingerprint="${FINETUNE_CONTRACT_FINGERPRINT}"
            config_fingerprint="${FINETUNE_CONFIG_FINGERPRINT}"
            ;;
        *) die "completion record has no canonical contract: ${phase}" ;;
    esac
    local command=(
        env
        "PYTHONPATH=${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
        "${CONTRACT_PYTHON}"
        -c "${SLURM_HELPER_CODE}"
        "${action}-completion"
        --marker "${checkpoint}.stage-complete"
        --run-contract "${contract_json}"
        --contract-fingerprint "${contract_fingerprint}"
        --config-fingerprint "${config_fingerprint}"
        --checkpoint "${checkpoint}"
    )
    printf '%s\0' "${command[@]}"
}


write_completion_record() {
    local phase="$1"
    local checkpoint="$2"
    local command=()
    while IFS= read -r -d '' argument; do
        command+=("${argument}")
    done < <(completion_record_command write "${phase}" "${checkpoint}")
    "${command[@]}" \
        || die "could not atomically publish ${phase} completion marker"
    LAST_VALIDATED_MARKER="${checkpoint}.stage-complete"
    LAST_VALIDATED_CHECKPOINT="${checkpoint}"
    LAST_VALIDATED_PHASE="${phase}"
}


validate_completion_record() {
    local phase="$1"
    local checkpoint="$2"
    local command=()
    while IFS= read -r -d '' argument; do
        command+=("${argument}")
    done < <(completion_record_command validate "${phase}" "${checkpoint}")
    if [[ "${DRY_RUN}" == true ]]; then
        print_command "${command[@]}"
    else
        [[ -f "${checkpoint}.stage-complete" ]] \
            || die "required ${phase} completion marker is missing: ${checkpoint}.stage-complete"
        validate_checkpoint "${checkpoint}" "${phase}" "${WORLD_SIZE}"
        VALIDATED_COMPLETION_OUTPUT="$("${command[@]}")" \
            || die "${phase} completion marker is malformed, mismatched, or stale; repair or remove it explicitly"
        [[ "${VALIDATED_COMPLETION_OUTPUT}" == "$(realpath -m -- "${checkpoint}")" ]] \
            || die "${phase} completion marker selected an unexpected checkpoint"
    fi
    LAST_VALIDATED_MARKER="${checkpoint}.stage-complete"
    LAST_VALIDATED_CHECKPOINT="${checkpoint}"
    LAST_VALIDATED_PHASE="${phase}"
}


preemption_boundary() {
    [[ "${PREEMPTION_REQUESTED}" == true ]] || return 0
    if [[ -n "${LAST_VALIDATED_MARKER}" ]]; then
        validate_completion_record \
            "${LAST_VALIDATED_PHASE}" \
            "${LAST_VALIDATED_CHECKPOINT}"
    fi
    printf 'SIGUSR1 is latched at a safe phase boundary; exiting 75 without launching further work\n' >&2
    return 75
}


run_training_phase() {
    local phase="$1"
    local checkpoint="$2"
    local initial_checkpoint_signature
    initial_checkpoint_signature="$(checkpoint_signature "${checkpoint}")"
    local command=()
    while IFS= read -r -d '' argument; do
        command+=("${argument}")
    done < <(training_srun_command "${phase}" "${@:3}")

    if [[ "${DRY_RUN}" == true ]]; then
        print_command "${command[@]}"
        return 0
    fi
    preemption_boundary || return $?

    local status
    if wait_for_step "${command[@]}"; then
        status=0
    else
        status=$?
    fi
    if (( status == 75 )) || [[ "${PREEMPTION_REQUESTED}" == true ]]
    then
        local current_checkpoint_signature
        current_checkpoint_signature="$(checkpoint_signature "${checkpoint}")"
        [[ "${current_checkpoint_signature}" != "${initial_checkpoint_signature}" ]] \
            || die "${phase} did not publish a new checkpoint after SIGUSR1; refusing requeue-friendly exit 75"
        if ! validate_checkpoint "${checkpoint}" "${phase}" "${WORLD_SIZE}"; then
            die "${phase} checkpoint validation failed; refusing requeue-friendly exit 75"
        fi
        if (( status == 0 )); then
            write_completion_record "${phase}" "${checkpoint}"
        fi
        return 75
    fi
    (( status == 0 )) || return "${status}"
    validate_checkpoint "${checkpoint}" "${phase}" "${WORLD_SIZE}"
    write_completion_record "${phase}" "${checkpoint}"
}


run_pretraining() {
    if [[ -n "${PRETRAIN_RESUME}" ]]; then
        validate_checkpoint "${PRETRAIN_RESUME}" pretrain "${WORLD_SIZE}"
    fi
    local arguments=(
        "${TRAIN_ENTRY}"
        --config "${PRETRAIN_CONFIG}"
        --override "task.data=${MANIFEST_DIR}"
        --override "checkpoint.save_dir=${PRETRAIN_DIR}"
        --override "distributed_training.distributed_world_size=${WORLD_SIZE}"
        --override "dataset.max_tokens=${PRETRAIN_MAX_TOKENS}"
        --override "optimization.update_freq=[${PRETRAIN_UPDATE_FREQ}]"
        --device cuda
    )
    if [[ -n "${PRETRAIN_RESUME}" ]]; then
        arguments+=(--resume "${PRETRAIN_RESUME}")
    fi
    run_training_phase \
        pretrain \
        "${PRETRAIN_CHECKPOINT}" \
        "${PRETRAIN_DIR}" \
        "${PRETRAIN_CONFIG}" \
        "${RUN_ID}/pretrain" \
        "${PRETRAIN_RESUME}" \
        "" \
        "${arguments[@]}"
}


run_finetuning() {
    validate_checkpoint "${PRETRAIN_CHECKPOINT}" pretrain "${WORLD_SIZE}"
    if [[ -n "${FINETUNE_RESUME}" ]]; then
        validate_checkpoint "${FINETUNE_RESUME}" finetune "${WORLD_SIZE}"
    fi
    local arguments=(
        "${TRAIN_ENTRY}"
        --config "${FINETUNE_CONFIG}"
        --pretrained-checkpoint "${PRETRAIN_CHECKPOINT}"
        --override "task.data=${MANIFEST_DIR}"
        --override "dataset.train_subset=${TRAIN_SUBSET}"
        --override "dataset.valid_subset=${VALID_SUBSET}"
        --override "checkpoint.save_dir=${FINETUNE_DIR}"
        --override "distributed_training.distributed_world_size=${WORLD_SIZE}"
        --override "dataset.max_tokens=${FINETUNE_MAX_TOKENS}"
        --override "optimization.update_freq=[${FINETUNE_UPDATE_FREQ}]"
        --override model.checkpoint_activations=true
        --device cuda
    )
    if [[ -n "${FINETUNE_RESUME}" ]]; then
        arguments+=(--resume "${FINETUNE_RESUME}")
    fi
    run_training_phase \
        finetune \
        "${FINETUNE_LAST_CHECKPOINT}" \
        "${FINETUNE_DIR}" \
        "${FINETUNE_CONFIG}" \
        "${RUN_ID}/finetune" \
        "${FINETUNE_RESUME}" \
        "${PRETRAIN_CHECKPOINT}" \
        "${arguments[@]}"
}


run_evaluation() {
    local checkpoint
    checkpoint="${FINETUNE_LAST_CHECKPOINT}"
    validate_completion_record finetune "${checkpoint}"
    preemption_boundary || return $?
    local command=(
        "${SRUN_BIN}"
        --nodes=1
        --ntasks=1
        --ntasks-per-node=1
        --gpus-per-node=1
        --kill-on-bad-exit=1
        "${PYTHON_BIN}"
        "${EVALUATOR}"
        --checkpoint "${checkpoint}"
        --manifest-dir "${MANIFEST_DIR}"
        --subset "${VALID_SUBSET}"
        --output "${FINAL_REPORT}"
        --tensorboard-dir "${EVALUATION_TENSORBOARD_DIR}"
        --fold "${FOLD}"
        --fraction "${FRACTION}"
        --device cuda
        --max-tokens "${EVAL_MAX_TOKENS}"
        --num-workers "${EVAL_WORKERS}"
        --training-world-size "${WORLD_SIZE}"
        --pretrain-max-tokens "${PRETRAIN_MAX_TOKENS}"
        --pretrain-update-freq "${PRETRAIN_UPDATE_FREQ}"
        --finetune-max-tokens "${FINETUNE_MAX_TOKENS}"
        --finetune-update-freq "${FINETUNE_UPDATE_FREQ}"
    )
    if [[ "${DRY_RUN}" == true ]]; then
        print_command "${command[@]}"
        return 0
    fi
    wait_for_step "${command[@]}"
}


run_or_propagate() {
    local status
    if "$@"; then
        return 0
    else
        status=$?
    fi
    exit "${status}"
}


reuse_or_run_stage() {
    local phase="$1"
    local checkpoint="$2"
    local runner="$3"
    if [[ "${DRY_RUN}" == false ]] \
        && { [[ -e "${checkpoint}.stage-complete" ]] \
            || [[ -L "${checkpoint}.stage-complete" ]]; }
    then
        validate_completion_record "${phase}" "${checkpoint}"
    else
        run_or_propagate "${runner}"
    fi
}


case "${PHASE}" in
    pretrain)
        preemption_boundary || exit $?
        reuse_or_run_stage pretrain "${PRETRAIN_CHECKPOINT}" run_pretraining
        preemption_boundary || exit $?
        ;;
    finetune)
        preemption_boundary || exit $?
        reuse_or_run_stage finetune "${FINETUNE_LAST_CHECKPOINT}" run_finetuning
        preemption_boundary || exit $?
        ;;
    evaluate)
        preemption_boundary || exit $?
        run_or_propagate run_evaluation
        preemption_boundary || exit $?
        ;;
    all)
        preemption_boundary || exit $?
        reuse_or_run_stage pretrain "${PRETRAIN_CHECKPOINT}" run_pretraining
        preemption_boundary || exit $?
        reuse_or_run_stage finetune "${FINETUNE_LAST_CHECKPOINT}" run_finetuning
        preemption_boundary || exit $?
        run_or_propagate run_evaluation
        preemption_boundary || exit $?
        ;;
esac
