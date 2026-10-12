#!/usr/bin/env bash
set -euo pipefail
HERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(CDPATH= cd -- "$HERE/../.." && pwd)"
exec "${PYTHON_BIN:-$ROOT/artifact/.venvs/sim/bin/python}" -B "$ROOT/experiments/coupled_act_student50_assembly_fallback_release/entry_cli.py" --root "${ROBOSYN_STUDENT50_PACKAGE_ROOT:-$ROOT/artifact}" "$@"
