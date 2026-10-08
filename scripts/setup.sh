#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/.ocr-environment"
TOOLS="$ROOT/.tools"
ENVS="$ROOT/.envs"
source "$ROOT/environment/ocr-versions.env"
mkdir -p "$TOOLS" "$ENVS"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Linux setup currently supports x86_64; this platform can use the existing Python installation." >&2
  exit 1
fi

if command -v micromamba >/dev/null 2>&1; then
  MAMBA="$(command -v micromamba)"
else
  MAMBA="$TOOLS/micromamba"
  if [[ ! -x "$MAMBA" ]]; then
    release="https://github.com/mamba-org/micromamba-releases/releases/latest/download"
    curl -fsSL "$release/micromamba-linux-64" -o "$TOOLS/micromamba.new"
    curl -fsSL "$release/micromamba-linux-64.sha256" -o "$TOOLS/micromamba.sha256"
    expected="$(tr -d '[:space:]' < "$TOOLS/micromamba.sha256")"
    actual="$(sha256sum "$TOOLS/micromamba.new" | cut -d ' ' -f1)"
    [[ "$expected" == "$actual" ]] || { rm -f "$TOOLS/micromamba.new"; echo "micromamba checksum verification failed." >&2; exit 1; }
    mv "$TOOLS/micromamba.new" "$MAMBA"
    chmod 755 "$MAMBA"
  fi
fi
export MAMBA_ROOT_PREFIX="$TOOLS/mamba-root"

create_base() {
  local prefix="$1"
  "$MAMBA" create -y -p "$prefix" -c conda-forge "python=$PYTHON_VERSION" pip ffmpeg || return 1
  "$MAMBA" run -p "$prefix" python -m pip install --upgrade pip || return 1
  "$MAMBA" run -p "$prefix" python -m pip install -e "$ROOT" || return 1
}

gpu_ok=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_info="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true)"
  if [[ -n "$gpu_info" ]]; then
    echo "NVIDIA GPU detected: $gpu_info"
    gpu_prefix="$ENVS/ocr-gpu"
    rm -rf "$gpu_prefix"
    if create_base "$gpu_prefix" && \
      "$MAMBA" run -p "$gpu_prefix" python -m pip install "paddlepaddle-gpu==$PADDLE_VERSION" -i "$CUDA_INDEX_URL" && \
      "$MAMBA" run -p "$gpu_prefix" python -m pip install "paddleocr[doc-parser]==$PADDLEOCR_VERSION" && \
      "$MAMBA" run -p "$gpu_prefix" python -c 'import paddle; assert paddle.device.cuda.device_count() > 0' && \
      "$MAMBA" run -p "$gpu_prefix" python -c 'from paddleocr import PaddleOCRVL; PaddleOCRVL(device="gpu:0")'; then
      printf 'mode=gpu\nprefix=%q\n' "$gpu_prefix" > "$STATE"
      gpu_ok=1
      echo "Selected OCR environment: GPU ($gpu_prefix)"
    else
      echo "GPU OCR setup or initialization failed; removing the GPU environment and falling back to CPU." >&2
      rm -rf "$gpu_prefix"
    fi
  fi
fi

if [[ "$gpu_ok" -eq 0 ]]; then
  cpu_prefix="$ENVS/ocr-cpu"
  rm -rf "$cpu_prefix"
  if ! create_base "$cpu_prefix" || \
    ! "$MAMBA" run -p "$cpu_prefix" python -m pip install "paddlepaddle==$PADDLE_VERSION" -i "$CPU_INDEX_URL" || \
    ! "$MAMBA" run -p "$cpu_prefix" python -m pip install "paddleocr[doc-parser]==$PADDLEOCR_VERSION" || \
    ! "$MAMBA" run -p "$cpu_prefix" python -c 'import paddle, rapidocr; import video_subtitles.cli'; then
    rm -rf "$cpu_prefix"
    echo "CPU OCR environment setup failed. See the package manager output above." >&2
    exit 1
  fi
  printf 'mode=cpu\nprefix=%q\n' "$cpu_prefix" > "$STATE"
  echo "Selected OCR environment: CPU ($cpu_prefix)"
fi

echo "Environment selection saved to $STATE"
