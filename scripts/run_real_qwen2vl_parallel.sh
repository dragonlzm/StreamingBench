#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SRC_DIR="${REPO_ROOT}/src"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SRC_DIR}/data/parallel_real_qwen2vl}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
SHARD_DIR="${RUN_DIR}/shards"
RESULT_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"
HF_REPO_ID="${HF_REPO_ID:-zhuomingliu/testDataset}"
HF_REPO_PATH="${HF_REPO_PATH:-StreamingBench/real_qwen2vl_parallel/${RUN_ID}}"
CONTEXT_TIME="${CONTEXT_TIME:--1}"
ATTN_IMPLEMENTATION="${QWEN2VL_ATTN_IMPLEMENTATION:-sdpa}"
MAX_SHARD_ATTEMPTS="${MAX_SHARD_ATTEMPTS:-3}"

mkdir -p "${SHARD_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

cd "${SRC_DIR}"

RUNNER_LOG="${LOG_DIR}/runner.log"
exec > >(tee -a "${RUNNER_LOG}") 2>&1

echo "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "repo_root=${REPO_ROOT}"
echo "src_dir=${SRC_DIR}"
echo "run_id=${RUN_ID}"
echo "gpu_ids=${GPU_IDS}"
echo "run_dir=${RUN_DIR}"
echo "hf_repo_id=${HF_REPO_ID}"
echo "hf_repo_path=${HF_REPO_PATH}"
echo "context_time=${CONTEXT_TIME}"
echo "attn_implementation=${ATTN_IMPLEMENTATION}"
echo "max_shard_attempts=${MAX_SHARD_ATTEMPTS}"

BASE_DATA="${BASE_DATA:-${SRC_DIR}/data/real_output_Qwen2-VL.json}"
if ! "${PYTHON_BIN}" - <<'PY' "${BASE_DATA}" >/dev/null 2>&1
import json, sys
json.load(open(sys.argv[1]))
PY
then
  BASE_DATA="${SRC_DIR}/data/questions_real.json"
fi
echo "base_data=${BASE_DATA}"

export SHARD_DIR RESULT_DIR BASE_DATA GPU_IDS
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

base_path = Path(os.environ["BASE_DATA"])
shard_dir = Path(os.environ["SHARD_DIR"])
result_dir = Path(os.environ["RESULT_DIR"])
gpu_ids = os.environ["GPU_IDS"].split()
data = json.load(base_path.open())

manifest = {}
for shard_idx, gpu_id in enumerate(gpu_ids):
    indices = [idx for idx in range(len(data)) if idx % len(gpu_ids) == shard_idx]
    shard = [data[idx] for idx in indices]
    shard_path = shard_dir / f"questions_real_shard_{shard_idx:02d}_gpu{gpu_id}.json"
    output_path = result_dir / f"real_output_Qwen2-VL_shard_{shard_idx:02d}_gpu{gpu_id}.json"
    with shard_path.open("w", encoding="utf-8") as f:
        json.dump(shard, f, indent=4, ensure_ascii=False)
    manifest[str(shard_idx)] = {
        "gpu_id": gpu_id,
        "indices": indices,
        "shard_path": str(shard_path),
        "output_path": str(output_path),
    }

answered = 0
total = 0
for subset in data:
    for question in subset.get("questions", []):
        total += 1
        if question.get("Qwen2-VL"):
            answered += 1

with (shard_dir / "manifest.json").open("w", encoding="utf-8") as f:
    json.dump(
        {
            "base_data": str(base_path),
            "num_shards": len(gpu_ids),
            "num_subsets": len(data),
            "total_questions": total,
            "answered_questions_in_base": answered,
            "shards": manifest,
        },
        f,
        indent=4,
    )

print(f"created_shards={len(gpu_ids)}")
print(f"num_subsets={len(data)}")
print(f"total_questions={total}")
print(f"answered_questions_in_base={answered}")
PY

declare -a PIDS=()
declare -a LABELS=()
shard_idx=0
for gpu_id in ${GPU_IDS}; do
  shard_label="$(printf 'shard_%02d_gpu%s' "${shard_idx}" "${gpu_id}")"
  shard_file="${SHARD_DIR}/questions_real_${shard_label}.json"
  if [ ! -f "${shard_file}" ]; then
    shard_file="${SHARD_DIR}/questions_real_shard_$(printf '%02d' "${shard_idx}")_gpu${gpu_id}.json"
  fi
  output_file="${RESULT_DIR}/real_output_Qwen2-VL_shard_$(printf '%02d' "${shard_idx}")_gpu${gpu_id}.json"
  log_file="${LOG_DIR}/${shard_label}.log"

  (
    set -o pipefail
    echo "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "gpu_id=${gpu_id}"
    echo "shard_file=${shard_file}"
    echo "output_file=${output_file}"
    status=1
    for attempt in $(seq 1 "${MAX_SHARD_ATTEMPTS}"); do
      echo "attempt=${attempt}/${MAX_SHARD_ATTEMPTS}"
      FORCE_QWENVL_VIDEO_READER=decord \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      QWEN2VL_ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
      "${PYTHON_BIN}" eval.py \
        --model_name Qwen2-VL \
        --benchmark_name Streaming \
        --data_file "${shard_file}" \
        --output_file "${output_file}" \
        --context_time "${CONTEXT_TIME}"
      status=$?
      if [ "${status}" = "0" ]; then
        break
      fi
      echo "attempt_status=${status}"
      if [ "${attempt}" != "${MAX_SHARD_ATTEMPTS}" ]; then
        sleep 15
      fi
    done
    echo "finished_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "exit_status=${status}"
    exit "${status}"
  ) > "${log_file}" 2>&1 &

  pid=$!
  PIDS+=("${pid}")
  LABELS+=("${shard_label}")
  echo "launched ${shard_label} pid=${pid} log=${log_file}"
  shard_idx=$((shard_idx + 1))
done

overall_status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "${LABELS[$i]} completed"
  else
    status=$?
    echo "${LABELS[$i]} failed status=${status}"
    overall_status=1
  fi
done

export COMBINED_OUTPUT="${RUN_DIR}/real_output_Qwen2-VL_merged_${RUN_ID}.json"
export CANONICAL_OUTPUT="${SRC_DIR}/data/real_output_Qwen2-VL_parallel_${RUN_ID}.json"
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

shard_dir = Path(os.environ["SHARD_DIR"])
base_path = Path(os.environ["BASE_DATA"])
combined_output = Path(os.environ["COMBINED_OUTPUT"])
canonical_output = Path(os.environ["CANONICAL_OUTPUT"])
manifest = json.load((shard_dir / "manifest.json").open())
merged = json.load(base_path.open())

for shard_id, meta in manifest["shards"].items():
    output_path = Path(meta["output_path"])
    shard_path = Path(meta["shard_path"])
    source_path = output_path if output_path.exists() else shard_path
    shard_data = json.load(source_path.open())
    for original_idx, subset in zip(meta["indices"], shard_data):
        merged[original_idx] = subset

answered = 0
total = 0
for subset in merged:
    for question in subset.get("questions", []):
        total += 1
        if question.get("Qwen2-VL"):
            answered += 1

for path in (combined_output, canonical_output):
    with path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

summary = {
    "total_questions": total,
    "answered_questions": answered,
    "combined_output": str(combined_output),
    "canonical_output": str(canonical_output),
}
with (combined_output.parent / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)
print(json.dumps(summary, indent=2))
PY

if command -v hf >/dev/null 2>&1; then
  if hf auth whoami >/dev/null 2>&1; then
    echo "Uploading run directory to hf://${HF_REPO_ID}/${HF_REPO_PATH}"
    hf upload "${HF_REPO_ID}" "${RUN_DIR}" "${HF_REPO_PATH}" --repo-type dataset
    echo "hf_upload_status=$?"
  else
    echo "HF upload skipped: hf auth is not logged in."
  fi
else
  echo "HF upload skipped: hf CLI not found."
fi

echo "overall_eval_status=${overall_status}"
if [ "${overall_status}" != "0" ]; then
  echo "Skipping gpu_burn because one or more evaluation shards failed."
  exit "${overall_status}"
fi

echo "starting_gpu_burn_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
cd "${HOME}/env/gpu-burn" && ./gpu_burn -d 100000000
