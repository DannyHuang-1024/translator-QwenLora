#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ADAPTER="${ADAPTER:-/content/drive/MyDrive/qwen3-translation/checkpoints/qwen3-0.6b-en-zh-lora/final}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/qwen3-translation/evaluation/qwen3-0.6b-en-zh-lora}"
LIMIT="${LIMIT:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
METRICS="${METRICS:-/content/drive/MyDrive/qwen3-translation/checkpoints/qwen3-0.6b-en-zh-lora/metrics.jsonl}"

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive is not mounted. Run drive.mount('/content/drive') in a Colab Python cell first." >&2
  exit 3
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m pip install "sacrebleu>=2.4" "matplotlib>=3.8" "pandas>=2.0"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" evaluation/evaluate.py \
  --adapter "${ADAPTER}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE}"
"${PYTHON_BIN}" evaluation/plot_report.py \
  --evaluation-dir "${OUTPUT_DIR}" \
  --training-metrics "${METRICS}"
echo "Evaluation artifacts saved to ${OUTPUT_DIR}"

