#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/.ocr-environment"
usage() {
  echo 'Usage: scripts/run.sh --input PATH_OR_URL [--ocr] [--output FILE] [other subtitles options]' >&2
}
has_input=0
for argument in "$@"; do
  case "$argument" in
    --input|--input=*) has_input=1 ;;
    --help|-h) has_input=1 ;;
  esac
done
if (( ! has_input )); then
  usage
  exit 2
fi
if [[ ! -f "$STATE" ]]; then
  echo "OCR environment is not configured. Run: ./configure --setup" >&2
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
  echo "micromamba is missing; rerun ./configure --setup" >&2
  exit 2
fi
export MAMBA_ROOT_PREFIX="$ROOT/.tools/mamba-root"
exec "$MAMBA" run -p "$prefix" subtitles "$@"
