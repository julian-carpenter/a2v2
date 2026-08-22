#!/usr/bin/env bash
#
# Run one torchrun agent on one homogeneous SLURM node. The outer launcher
# starts exactly one copy per node with srun.

set -Eeuo pipefail


usage() {
    cat <<'USAGE'
Usage:
  bash scripts/a2v2_slurm_node.sh PHASE [options] -- TRAIN_COMMAND [args...]

Required options:
  --run-id ID
  --manifest-dir PATH
  --output-dir PATH
  --config PATH

Rendezvous/topology options:
  --job-id ID
  --nodes N
  --node-rank N
  --gpus-per-node N
  --master-addr HOST
  --master-port PORT
  --rdzv-endpoint HOST:PORT
  --manifest-fingerprint VALUE

Checkpoint preflight options:
  --resume-checkpoint PATH
  --pretrained-checkpoint PATH

Other:
  --dry-run
  -h, --help

Real launches require the matching SLURM_JOB_ID, SLURM_JOB_NUM_NODES,
SLURM_NODEID, SLURM_GPUS_ON_NODE, SLURM_JOB_NODELIST, and
CUDA_VISIBLE_DEVICES values. This wrapper never rewrites CUDA visibility.
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


require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] \
        || die "${name} must be a non-negative integer; received ${value@Q}"
}


print_command() {
    printf 'Launching:'
    printf ' %q' "$@"
    printf '\n'
}


[[ $# -ge 1 ]] || {
    usage >&2
    exit 2
}

PHASE="$1"
shift
[[ "${PHASE}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || die "phase must contain only letters, digits, dots, underscores, and dashes"

DRY_RUN=false
JOB_ID_OVERRIDE=""
NODES_OVERRIDE=""
NODE_RANK_OVERRIDE=""
GPUS_PER_NODE_OVERRIDE=""
MASTER_ADDR=""
MASTER_PORT=""
RDZV_ENDPOINT=""
RUN_ID=""
MANIFEST_DIR=""
MANIFEST_FINGERPRINT="unverified"
OUTPUT_DIR=""
CONFIG_PATH=""
RESUME_CHECKPOINT=""
PRETRAINED_CHECKPOINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id|--nodes|--node-rank|--gpus-per-node|--master-addr|--master-port|--rdzv-endpoint|--run-id|--manifest-dir|--manifest-fingerprint|--output-dir|--config|--resume-checkpoint|--pretrained-checkpoint)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            option="$1"
            value="$2"
            case "${option}" in
                --job-id) JOB_ID_OVERRIDE="${value}" ;;
                --nodes) NODES_OVERRIDE="${value}" ;;
                --node-rank) NODE_RANK_OVERRIDE="${value}" ;;
                --gpus-per-node) GPUS_PER_NODE_OVERRIDE="${value}" ;;
                --master-addr) MASTER_ADDR="${value}" ;;
                --master-port) MASTER_PORT="${value}" ;;
                --rdzv-endpoint) RDZV_ENDPOINT="${value}" ;;
                --run-id) RUN_ID="${value}" ;;
                --manifest-dir) MANIFEST_DIR="${value}" ;;
                --manifest-fingerprint) MANIFEST_FINGERPRINT="${value}" ;;
                --output-dir) OUTPUT_DIR="${value}" ;;
                --config) CONFIG_PATH="${value}" ;;
                --resume-checkpoint) RESUME_CHECKPOINT="${value}" ;;
                --pretrained-checkpoint) PRETRAINED_CHECKPOINT="${value}" ;;
            esac
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --)
            shift
            break
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

[[ $# -gt 0 ]] || die "a training command is required after --"
TRAIN_COMMAND=("$@")
[[ -n "${RUN_ID}" ]] || die "--run-id is required"
[[ -n "${MANIFEST_DIR}" ]] || die "--manifest-dir is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
[[ -n "${CONFIG_PATH}" ]] || die "--config is required"

if [[ "${DRY_RUN}" == false ]]; then
    for variable in \
        SLURM_JOB_ID \
        SLURM_JOB_NUM_NODES \
        SLURM_NODEID \
        SLURM_GPUS_ON_NODE \
        SLURM_JOB_NODELIST \
        CUDA_VISIBLE_DEVICES
    do
        [[ -n "${!variable:-}" ]] \
            || die "missing required environment variable ${variable}"
    done
fi

JOB_ID="${JOB_ID_OVERRIDE:-${SLURM_JOB_ID:-dry-run}}"
NODES="${NODES_OVERRIDE:-${SLURM_JOB_NUM_NODES:-1}}"
NODE_RANK="${NODE_RANK_OVERRIDE:-${SLURM_NODEID:-0}}"
GPUS_PER_NODE="${GPUS_PER_NODE_OVERRIDE:-${SLURM_GPUS_ON_NODE:-1}}"
[[ -n "${JOB_ID}" ]] || die "job ID must be nonempty"
require_positive_integer "node count" "${NODES}"
require_nonnegative_integer "node rank" "${NODE_RANK}"
require_positive_integer "GPUs per node" "${GPUS_PER_NODE}"
(( NODE_RANK < NODES )) \
    || die "node rank ${NODE_RANK} must be smaller than node count ${NODES}"

if [[ "${DRY_RUN}" == false ]]; then
    [[ "${SLURM_JOB_ID}" == "${JOB_ID}" ]] \
        || die "SLURM_JOB_ID=${SLURM_JOB_ID@Q}; expected job ID ${JOB_ID@Q}"
    [[ "${SLURM_JOB_NUM_NODES}" == "${NODES}" ]] \
        || die "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES@Q}; expected ${NODES} nodes"
    [[ "${SLURM_NODEID}" == "${NODE_RANK}" ]] \
        || die "SLURM_NODEID=${SLURM_NODEID@Q}; expected node rank ${NODE_RANK}"
    [[ "${SLURM_GPUS_ON_NODE}" == "${GPUS_PER_NODE}" ]] \
        || die "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE@Q}; expected ${GPUS_PER_NODE} GPUs per node"

    IFS=',' read -r -a VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
    [[ ${#VISIBLE_DEVICES[@]} -eq "${GPUS_PER_NODE}" ]] \
        || die "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_DEVICES[@]} device(s); expected ${GPUS_PER_NODE}"
    for device in "${VISIBLE_DEVICES[@]}"; do
        [[ -n "${device}" ]] || die "CUDA_VISIBLE_DEVICES contains an empty device entry"
    done

    [[ -d "${MANIFEST_DIR}" ]] \
        || die "shared manifest directory is not visible on node ${NODE_RANK}: ${MANIFEST_DIR}"
    [[ -d "${OUTPUT_DIR}" ]] \
        || die "shared output directory is not visible on node ${NODE_RANK}: ${OUTPUT_DIR}"
    [[ -f "${CONFIG_PATH}" ]] \
        || die "config file is not visible on node ${NODE_RANK}: ${CONFIG_PATH}"
    if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
        die "resume checkpoint is not visible on node ${NODE_RANK}: ${RESUME_CHECKPOINT}"
    fi
    if [[ -n "${PRETRAINED_CHECKPOINT}" && ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
        die "pretrained checkpoint is not visible on node ${NODE_RANK}: ${PRETRAINED_CHECKPOINT}"
    fi
    if (( NODE_RANK == 0 )) && [[ -e "${OUTPUT_DIR}/.a2v2-output.lock" ]]; then
        die "output lock already exists at ${OUTPUT_DIR}/.a2v2-output.lock; do not steal it automatically; use reproduce_meerkat_slurm.sh --recover-lock ${PHASE}:<64-hex-owner-fingerprint>"
    fi
fi

if [[ -n "${RDZV_ENDPOINT}" ]]; then
    [[ -z "${MASTER_ADDR}" && -z "${MASTER_PORT}" ]] \
        || die "--rdzv-endpoint cannot be combined with --master-addr or --master-port"
    [[ "${RDZV_ENDPOINT}" =~ ^(\[[^]]+\]|[^:[:space:]]+):([0-9]+)$ ]] \
        || die "--rdzv-endpoint must use HOST:PORT syntax"
    MASTER_PORT="${BASH_REMATCH[2]}"
else
    [[ -n "${MASTER_ADDR}" ]] || die "--master-addr is required"
    [[ -n "${MASTER_PORT}" ]] || die "--master-port is required"
    RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
fi
require_positive_integer "rendezvous port" "${MASTER_PORT}"
(( MASTER_PORT <= 65535 )) \
    || die "rendezvous port must be at most 65535; received ${MASTER_PORT}"

WORLD_SIZE=$((NODES * GPUS_PER_NODE))
RDZV_ID="${JOB_ID}-${PHASE}"
CONTRACT_INPUT="$(
    printf '%s\n' \
        "${JOB_ID}" \
        "${PHASE}" \
        "${NODES}" \
        "${GPUS_PER_NODE}" \
        "${WORLD_SIZE}" \
        "${RDZV_ENDPOINT}" \
        "${RDZV_ID}" \
        "${OUTPUT_DIR}" \
        "${MANIFEST_FINGERPRINT}"
)"
CONTRACT_FINGERPRINT="$(printf '%s' "${CONTRACT_INPUT}" | sha256sum)"
CONTRACT_FINGERPRINT="${CONTRACT_FINGERPRINT%% *}"

PYTHON_BIN="${A2V2_PYTHON:-python}"
TORCHRUN_COMMAND=(
    env
    "A2V2_RUN_ID=${RUN_ID}"
    "A2V2_SLURM_PHASE=${PHASE}"
    "A2V2_SLURM_WORLD_SIZE=${WORLD_SIZE}"
    "A2V2_SLURM_RDZV_ENDPOINT=${RDZV_ENDPOINT}"
    "A2V2_SLURM_RDZV_ID=${RDZV_ID}"
    "A2V2_SLURM_MANIFEST_FINGERPRINT=${MANIFEST_FINGERPRINT}"
    "A2V2_SLURM_CONTRACT_FINGERPRINT=${CONTRACT_FINGERPRINT}"
    "${PYTHON_BIN}"
    -m torch.distributed.run
    "--nnodes=${NODES}"
    "--node-rank=${NODE_RANK}"
    "--nproc-per-node=${GPUS_PER_NODE}"
    --rdzv-backend=c10d
    "--rdzv-endpoint=${RDZV_ENDPOINT}"
    "--rdzv-id=${RDZV_ID}"
    "${TRAIN_COMMAND[@]}"
)

if [[ "${DRY_RUN}" == true ]]; then
    print_command "${TORCHRUN_COMMAND[@]}"
else
    exec "${TORCHRUN_COMMAND[@]}"
fi
