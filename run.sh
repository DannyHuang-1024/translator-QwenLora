#!/usr/bin/env bash
set -euo pipefail

# This launcher is intended to run after Google Drive has been mounted by a
# Colab Notebook cell. Drive authentication requires the Notebook kernel and
# cannot be completed by a shell subprocess.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/train.yaml}"
cd "${ROOT_DIR}"

if ! "${PYTHON_BIN}" -c 'import google.colab' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
This launcher must run inside a Google Colab runtime.

In Colab, upload/clone this project and run:
  cd /content/translator-QwenLora
  bash run.sh

The script will install dependencies, mount Google Drive, and resume the last
checkpoint under MyDrive/qwen3-translation when one exists.
EOF
  exit 2
fi

if [[ ! -d /content/drive/MyDrive ]]; then
  cat >&2 <<'EOF'
Google Drive is not mounted in this runtime.

Run this in a Colab Notebook Python cell (not in the terminal):
  from google.colab import drive
  drive.mount('/content/drive')

Then run this launcher again:
  bash run.sh
EOF
  exit 3
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements-colab.txt"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_env.py"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m qwen_translation.train --config "${CONFIG}" "$@"
