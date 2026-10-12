#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON311:-python3.11}" -B "$ROOT/experiments/coupled_act_student50_release/setup_release_v2.py" "$@"
