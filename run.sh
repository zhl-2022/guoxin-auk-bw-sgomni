#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
if [[ " $* " != *" --dry-run "* ]]; then
  "${SCNET_BASE_PYTHON:-python3}" -B "$TASK_DIR/scripts/prepare_runtime032.py" ensure
fi
exec "${SCNET_BASE_PYTHON:-python3}" -B "$TASK_DIR/entry031.py" launch --python "$PYTHON_BIN" "$@"
