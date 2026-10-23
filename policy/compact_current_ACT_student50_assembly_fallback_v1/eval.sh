#!/usr/bin/env bash
set -euo pipefail
HERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$HERE/runtime"
ARTIFACT="${ROBOSYN_STUDENT50_PACKAGE_ROOT:-$RUNTIME/artifact}"
PYTHON="${PYTHON_BIN:-${PYTHON311:-python3.11}}"
if [[ -z "${PYTHON_BIN:-}" && -z "${PYTHON311:-}" ]] && ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python3
fi
if [[ ! -x "$PYTHON" ]] && ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Run bash $HERE/setup.sh --root /tmp/rscp-policy, then set ROBOSYN_STUDENT50_PACKAGE_ROOT=/tmp/rscp-policy (or set PYTHON_BIN)." >&2
    exit 1
fi
exec "$PYTHON" -B "$HERE/entry_cli.py" --root "$ARTIFACT" "$@"
