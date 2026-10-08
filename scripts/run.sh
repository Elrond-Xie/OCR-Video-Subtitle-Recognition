#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/.ocr-environment"
if [[ ! -f "$STATE" ]]; then
  echo "OCR environment is not configured. Run: bash scripts/setup.sh" >&2
  exit 2
fi
source "$ROOT/environment/ocr-versions.env"
source "$STATE"
if command -v micromamba >/dev/null 2>&1; then
  MAMBA="$(command -v micromamba)"
else
  MAMBA="$ROOT/.tools/micromamba"
fi
if [[ ! -x "$MAMBA" ]]; then
  echo "micromamba is missing; rerun bash scripts/setup.sh" >&2
  exit 2
fi
export MAMBA_ROOT_PREFIX="$ROOT/.tools/mamba-root"
exec "$MAMBA" run -p "$prefix" subtitles "$@"
