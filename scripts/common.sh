#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
source "$ROOT/config.env"
set +a
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
command -v python3 >/dev/null || { echo 'Нужен Python 3.' >&2; exit 1; }
