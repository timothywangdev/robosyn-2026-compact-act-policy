#!/usr/bin/env bash
set -euo pipefail
HERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON311="${PYTHON311:-${PYTHON_BIN:-python3.11}}"
exec "$PYTHON311" -B "$HERE/setup_entry.py" "$@"
