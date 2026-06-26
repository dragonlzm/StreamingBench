#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HF_HOME="${HF_HOME:-/mnt/scratch/group/li968/zliu2346/huggingface}"
STREAMINGBENCH_ROOT="${STREAMINGBENCH_ROOT:-${HF_HOME}/streamingbench}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${STREAMINGBENCH_ROOT}/annotations}"
MODEL_PATH="${MODEL_PATH:-${HF_HOME}/models/Qwen2-VL-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/src/data/platformr_qwen2vl_outputs}"
CONTEXT_TIME="${CONTEXT_TIME:--1}"
DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TASKS="${TASKS:-real omni sqa proactive}"
OFFLINE="${OFFLINE:-1}"
QWEN2VL_ATTN_IMPLEMENTATION="${QWEN2VL_ATTN_IMPLEMENTATION:-flash_attention_2}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_platformr_qwen2vl_offline.sh [options]

Options:
  --repo-root PATH          StreamingBench checkout. Defaults to this script's repo.
  --python-bin PATH         Python executable from the active conda environment.
  --hf-home PATH            Platformr HF/cache asset root.
  --annotation-root PATH    Directory with prepared questions_*.json files.
  --model-path PATH         Local Qwen2-VL-7B-Instruct checkpoint directory.
  --output-root PATH        Directory for result JSON and logs.
  --context-time N          -1 for all prior context, or seconds before query.
  --devices IDS             CUDA_VISIBLE_DEVICES value.
  --tasks LIST              Space-separated tasks: real omni sqa proactive.
  --attn-implementation X   flash_attention_2 or sdpa.
  --online                  Do not set HF offline environment variables.
  --run-id ID               Output suffix.
  --dry-run                 Print commands without running them.
  -h, --help                Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="${2:?$1 requires a value}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:?$1 requires a value}"
      shift 2
      ;;
    --hf-home)
      HF_HOME="${2:?$1 requires a value}"
      STREAMINGBENCH_ROOT="${HF_HOME}/streamingbench"
      ANNOTATION_ROOT="${STREAMINGBENCH_ROOT}/annotations"
      MODEL_PATH="${HF_HOME}/models/Qwen2-VL-7B-Instruct"
      shift 2
      ;;
    --annotation-root)
      ANNOTATION_ROOT="${2:?$1 requires a value}"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="${2:?$1 requires a value}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?$1 requires a value}"
      shift 2
      ;;
    --context-time)
      CONTEXT_TIME="${2:?$1 requires a value}"
      shift 2
      ;;
    --devices)
      DEVICES="${2:?$1 requires a value}"
      shift 2
      ;;
    --tasks)
      TASKS="${2:?$1 requires a value}"
      shift 2
      ;;
    --attn-implementation)
      QWEN2VL_ATTN_IMPLEMENTATION="${2:?$1 requires a value}"
      shift 2
      ;;
    --online)
      OFFLINE=0
      shift
      ;;
    --offline)
      OFFLINE=1
      shift
      ;;
    --run-id)
      RUN_ID="${2:?$1 requires a value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}/src"

export HF_HOME
export QWEN2VL_MODEL_PATH="${MODEL_PATH}"
export QWEN2VL_ATTN_IMPLEMENTATION

if [ "${OFFLINE}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

if [[ "${PYTHON_BIN}" == */* ]]; then
  if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found on PATH: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [ ! -d "${ANNOTATION_ROOT}" ]; then
  echo "Annotation directory not found: ${ANNOTATION_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

benchmark_for_task() {
  case "$1" in
    real|omni)
      echo "Streaming"
      ;;
    sqa)
      echo "StreamingSQA"
      ;;
    proactive)
      echo "StreamingProactive"
      ;;
    *)
      echo "Unknown task: $1" >&2
      return 1
      ;;
  esac
}

run_task() {
  local task="$1"
  local benchmark
  benchmark="$(benchmark_for_task "${task}")"
  local data_file="${ANNOTATION_ROOT}/questions_${task}.json"
  local output_file="${OUTPUT_ROOT}/${task}_output_Qwen2-VL_${RUN_ID}.json"
  local log_file="${OUTPUT_ROOT}/${task}_Qwen2-VL_${RUN_ID}.log"

  if [ ! -f "${data_file}" ]; then
    echo "Data file not found: ${data_file}" >&2
    return 1
  fi

  local cmd=(
    "${PYTHON_BIN}" eval.py
    --model_name Qwen2-VL
    --benchmark_name "${benchmark}"
    --data_file "${data_file}"
    --output_file "${output_file}"
    --context_time "${CONTEXT_TIME}"
  )

  {
    echo "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "repo_root=${REPO_ROOT}"
    echo "hf_home=${HF_HOME}"
    echo "offline=${OFFLINE}"
    echo "model_path=${MODEL_PATH}"
    echo "annotation_root=${ANNOTATION_ROOT}"
    echo "task=${task}"
    echo "benchmark=${benchmark}"
    echo "context_time=${CONTEXT_TIME}"
    echo "cuda_visible_devices=${DEVICES}"
    echo "attn_implementation=${QWEN2VL_ATTN_IMPLEMENTATION}"
    printf 'command=CUDA_VISIBLE_DEVICES=%q' "${DEVICES}"
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
  } | tee "${log_file}"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${DEVICES}" "${cmd[@]}" 2>&1 | tee -a "${log_file}"
}

echo "Starting offline Qwen2-VL StreamingBench evaluation. RUN_ID=${RUN_ID}"
echo "tasks=${TASKS}"
echo "model_path=${MODEL_PATH}"
echo "annotation_root=${ANNOTATION_ROOT}"
echo "output_root=${OUTPUT_ROOT}"

for task in ${TASKS}; do
  run_task "${task}"
done

echo "Finished offline Qwen2-VL StreamingBench evaluation. RUN_ID=${RUN_ID}"
