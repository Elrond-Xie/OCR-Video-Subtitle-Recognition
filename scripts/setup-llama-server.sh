#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/environment/ocr-versions.env"
assume_yes=0
if [[ "${1:-}" == "--yes" ]]; then assume_yes=1; elif [[ $# -gt 0 ]]; then
  echo 'Usage: scripts/setup-llama-server.sh [--yes]' >&2
  exit 2
fi

resolve_path() { [[ "$1" == /* ]] && printf '%s\n' "$1" || printf '%s/%s\n' "$ROOT" "$1"; }
DOWNLOAD_ROOT="$(resolve_path "$DOWNLOAD_DIR")"
MODEL_DIR="$DOWNLOAD_ROOT/paddleocr-vl-gguf"
PREFIX="$ROOT/.envs/ocr-llama-server"
CONDA="$ROOT/.tools/miniforge3/bin/conda"
mkdir -p "$MODEL_DIR" "$DOWNLOAD_ROOT/conda"
if [[ ! -x "$CONDA" ]]; then echo '[ERROR] Project Miniforge missing. Run ./scripts/configure --setup first.' >&2; exit 2; fi

if [[ -x "$PREFIX/bin/llama-server" ]] && "$PREFIX/bin/llama-server" --version >/dev/null 2>&1; then
  echo "[INFO] Project-local llama.cpp is already installed: $PREFIX"
else
  if [[ -e "$PREFIX" ]]; then
    if (( ! assume_yes )); then
      printf 'Replace the incomplete llama.cpp environment? [y/N] '
      read -r answer || answer=''
      case "$answer" in y|Y|yes|YES) ;; *) echo '[INFO] Setup cancelled.'; exit 0 ;; esac
    fi
    rm -rf -- "$PREFIX"
  fi
  export CONDA_PKGS_DIRS="$DOWNLOAD_ROOT/conda"
  echo "[INFO] Installing CPU llama.cpp $LLAMA_CPP_VERSION into $PREFIX"
  if ! "$CONDA" create -y -p "$PREFIX" -c conda-forge "llama.cpp=$LLAMA_CPP_VERSION=$LLAMA_CPP_BUILD"; then
    rm -rf -- "$PREFIX"
    echo '[ERROR] Could not install the pinned CPU-only llama.cpp build; removed the incomplete environment.' >&2
    exit 1
  fi
  if [[ ! -x "$PREFIX/bin/llama-server" ]]; then
    rm -rf -- "$PREFIX"
    echo '[ERROR] Conda package did not provide llama-server; removed the incomplete environment.' >&2
    exit 1
  fi
fi

download_model() {
  local filename="$1"
  local expected="$2"
  local url="https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/$filename"
  local destination="$MODEL_DIR/$filename"
  if [[ -f "$destination" ]] && echo "$expected  $destination" | sha256sum -c --status; then
    echo "[INFO] Verified model file: $destination"
    return
  fi
  echo "[INFO] Downloading $filename (resumable; roughly 1 GB)."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -C - "$url" -o "$destination" || {
      echo "[ERROR] Download failed: $url" >&2
      exit 1
    }
  elif command -v wget >/dev/null 2>&1; then
    wget -c "$url" -O "$destination" || {
      echo "[ERROR] Download failed: $url" >&2
      exit 1
    }
  else
    echo "[ERROR] Download failed: $url" >&2
    echo '[ERROR] Install curl or wget in the project setup environment first.' >&2
    exit 1
  fi
  if ! echo "$expected  $destination" | sha256sum -c; then
    rm -f -- "$destination"
    echo "[ERROR] SHA-256 verification failed; removed $destination." >&2
    exit 1
  fi
}

download_model PaddleOCR-VL-1.6-GGUF.gguf "$OCR_GGUF_MODEL_SHA256"
download_model PaddleOCR-VL-1.6-GGUF-mmproj.gguf "$OCR_GGUF_MMPROJ_SHA256"
echo '[INFO] llama.cpp CPU service and PaddleOCR-VL GGUF files are ready.'
echo '[INFO] Start it with ./scripts/llama-server.sh start.'
