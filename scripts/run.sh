#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/.ocr-environment"
usage() {
  echo 'Usage: scripts/run.sh --input PATH_OR_URL [--ocr] [--output FILE] [other subtitles options]' >&2
}
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    cat <<'USAGE'
Usage: scripts/run.sh --input PATH_OR_URL [OPTIONS]

Examples:
  scripts/run.sh --input ./video.mp4 --ocr
  scripts/run.sh --input "https://example.com/video" --cookies ./cookies.txt --ocr

Options are forwarded to the subtitles CLI, including --output, --cookies,
--model, --force, --ocr, --ocr-backend, --ocr-server-url, and --ocr-concurrency.
USAGE
    exit 0
  fi
done
has_input=0
for argument in "$@"; do
  case "$argument" in
    --input|--input=*) has_input=1 ;;
  esac
done
if (( ! has_input )); then
  usage
  exit 2
fi
if [[ ! -f "$STATE" ]]; then
  echo "OCR environment is not configured. Run: ./scripts/configure --setup" >&2
  exit 2
fi
source "$ROOT/environment/ocr-versions.env"
source "$STATE"
DOWNLOAD_ROOT="$DOWNLOAD_DIR"
if [[ "$DOWNLOAD_ROOT" != /* ]]; then
  DOWNLOAD_ROOT="$ROOT/$DOWNLOAD_ROOT"
fi
LOG_ROOT="$LOG_DIR"
if [[ "$LOG_ROOT" != /* ]]; then
  LOG_ROOT="$ROOT/$LOG_ROOT"
fi
mkdir -p "$DOWNLOAD_ROOT/pip" "$DOWNLOAD_ROOT/conda" "$DOWNLOAD_ROOT/xdg" "$DOWNLOAD_ROOT/huggingface" \
  "$DOWNLOAD_ROOT/paddlex" "$DOWNLOAD_ROOT/torch"
export PIP_CACHE_DIR="$DOWNLOAD_ROOT/pip"
export XDG_CACHE_HOME="$DOWNLOAD_ROOT/xdg"
export DOWNLOAD_DIR="$DOWNLOAD_ROOT"
export LOG_DIR="$LOG_ROOT"
export SAVE_LOGS
export HF_HOME="$DOWNLOAD_ROOT/huggingface"
export HF_HUB_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
export PADDLE_PDX_CACHE_HOME="$DOWNLOAD_ROOT/paddlex"
export TORCH_HOME="$DOWNLOAD_ROOT/torch"
# WSL exposes the NVIDIA driver library here; Paddle's dynamic loader needs
# this directory to resolve libcuda.so (not only libcuda.so.1 from ldconfig).
if [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
CONDA="$ROOT/.tools/miniforge3/bin/conda"
if [[ ! -x "$CONDA" ]]; then
  echo "Project-local Miniforge Conda is missing; rerun ./scripts/configure --setup" >&2
  exit 2
fi
export CONDA_PKGS_DIRS="$DOWNLOAD_ROOT/conda"
exec "$CONDA" run --no-capture-output -p "$prefix" subtitles "$@"
