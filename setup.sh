#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/policy/compact_current_ACT_student50_assembly_fallback_v1/setup.sh" "$@"
