#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec "${SCNET_BASE_PYTHON:-python3}" -B "$TASK_DIR/entry031.py" smoke --python "$PYTHON_BIN" "$@"
