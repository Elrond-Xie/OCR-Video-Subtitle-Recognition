#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/environment/ocr-versions.env"
export VLLM_VERSION TORCH_C_DLPACK_EXT_VERSION

assume_yes=0
if [[ "${1:-}" == "--yes" ]]; then
  assume_yes=1
elif [[ $# -gt 0 ]]; then
  echo 'Usage: scripts/setup-vlm-server.sh [--yes]' >&2
  exit 2
fi

resolve_path() {
  local path="$1"
  if [[ "$path" != /* ]]; then path="$ROOT/$path"; fi
  printf '%s\n' "$path"
}

DOWNLOAD_ROOT="$(resolve_path "$DOWNLOAD_DIR")"
LOG_ROOT="$(resolve_path "$LOG_DIR")"
PREFIX="$ROOT/.envs/ocr-vlm-server"
CONDA="$ROOT/.tools/miniforge3/bin/conda"
TMP_DIR="$ROOT/output/.tmp-vlm-setup"
export PATH="$PREFIX/bin:$PATH"
mkdir -p "$TMP_DIR"
export TMPDIR="$TMP_DIR"
mkdir -p "$DOWNLOAD_ROOT/pip" "$DOWNLOAD_ROOT/conda" "$DOWNLOAD_ROOT/xdg" \
  "$DOWNLOAD_ROOT/huggingface" "$DOWNLOAD_ROOT/paddlex" "$DOWNLOAD_ROOT/torch" \
  "$LOG_ROOT/configure"
log="$LOG_ROOT/configure/vlm-server-setup-$(date -u +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$log") 2>&1

if [[ ! -x "$CONDA" ]]; then
  rm -rf -- "$TMP_DIR"
  echo '[ERROR] Project-local Miniforge is missing. Run ./scripts/configure --setup first.' >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  rm -rf -- "$TMP_DIR"
  echo '[ERROR] This setup configures the NVIDIA vLLM route only. For a local CPU service, run ./scripts/setup-llama-server.sh.' >&2
  exit 2
fi

export CONDA_PKGS_DIRS="$DOWNLOAD_ROOT/conda"
export PIP_CACHE_DIR="$DOWNLOAD_ROOT/pip"
export XDG_CACHE_HOME="$DOWNLOAD_ROOT/xdg"
export HF_HOME="$DOWNLOAD_ROOT/huggingface"
export HF_HUB_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
export PADDLE_PDX_CACHE_HOME="$DOWNLOAD_ROOT/paddlex"
export TORCH_HOME="$DOWNLOAD_ROOT/torch"
export CCACHE_DIR="$DOWNLOAD_ROOT/ccache"
export CC="ccache gcc"
export CXX="ccache g++"
export PIP_PROGRESS_BAR=on
if [[ -d /usr/lib/wsl/lib ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -x "$PREFIX/bin/python" && -x "$PREFIX/bin/ccache" ]] && "$PREFIX/bin/python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
import vllm
import os
assert vllm.__version__ == os.environ["VLLM_VERSION"]
assert version("torch-c-dlpack-ext") == os.environ["TORCH_C_DLPACK_EXT_VERSION"]
import paddleocr, paddle
assert not paddle.is_compiled_with_cuda()
PY
then
  rm -rf -- "$TMP_DIR"
  echo "[INFO] The project-local vLLM server environment is already configured: $PREFIX"
  echo "[INFO] Setup log: $log"
  exit 0
fi

if [[ -e "$PREFIX" ]]; then
  if (( ! assume_yes )); then
    printf 'Replace the incomplete or incompatible vLLM server environment? [y/N] '
    read -r answer || answer=''
    case "$answer" in y|Y|yes|YES) ;; *) rm -rf -- "$TMP_DIR"; echo '[INFO] Setup cancelled.'; exit 0 ;; esac
  fi
  rm -rf -- "$PREFIX"
fi

setup_complete=0
cleanup_incomplete() {
  status=$?
  rm -rf -- "$TMP_DIR"
  if (( ! setup_complete )); then
    rm -rf -- "$PREFIX"
    echo '[INFO] Incomplete vLLM server environment removed.' >&2
  fi
  exit "$status"
}
trap cleanup_incomplete EXIT INT TERM

echo "[INFO] Creating isolated vLLM server environment: $PREFIX"
"$CONDA" create -y -p "$PREFIX" -c conda-forge "python=$PYTHON_VERSION" pip ccache
echo '[INFO] Installing CPU PaddleOCR host dependencies; vLLM alone will use the NVIDIA GPU.'
"$PREFIX/bin/python" -m pip install --upgrade pip
"$PREFIX/bin/python" -m pip install "paddlepaddle==$PADDLE_VERSION" "paddleocr[doc-parser]==$PADDLEOCR_VERSION"

echo '[INFO] Installing PaddleX vLLM server dependencies.'
PATH="$PREFIX/bin:$PATH" "$PREFIX/bin/paddleocr" install_genai_server_deps vllm
echo "[INFO] Installing vLLM $VLLM_VERSION, required by PaddleOCR-VL-1.6."
"$PREFIX/bin/python" -m pip install "vllm==$VLLM_VERSION"
echo '[INFO] Installing the optional vLLM DLPack extension and removing stale external FlashAttention builds.'
"$PREFIX/bin/python" -m pip install "torch-c-dlpack-ext==$TORCH_C_DLPACK_EXT_VERSION"
"$PREFIX/bin/python" -m pip uninstall -y flash-attn || true
"$PREFIX/bin/python" -m pip check

echo '[INFO] Verifying vLLM, PaddleOCR-VL integration, and the NVIDIA GPU.'
"$PREFIX/bin/python" - <<'PY'
import paddle
import torch
import vllm
import os
from paddlex.inference.genai.models import is_integrated_model_available
from paddlex.utils.deps import is_genai_engine_plugin_available

assert vllm.__version__ == os.environ["VLLM_VERSION"], vllm.__version__
assert not paddle.is_compiled_with_cuda(), "Server Paddle must stay CPU-only to avoid CUDA library conflicts."
assert torch.cuda.is_available(), "PyTorch cannot access the NVIDIA GPU."
assert torch.cuda.get_device_capability(0) >= (12, 0), "Expected an RTX 50-series Blackwell GPU."
assert is_genai_engine_plugin_available("vllm-server")
assert is_integrated_model_available("PaddleOCR-VL-1.6-0.9B", "vllm")
print(f"vLLM {vllm.__version__}; PyTorch {torch.__version__}; GPU {torch.cuda.get_device_name(0)}")
PY

setup_complete=1
rm -rf -- "$TMP_DIR"
trap - EXIT INT TERM
echo "[INFO] vLLM server setup complete. Log: $log"
echo '[INFO] Start it with ./scripts/vlm-server.sh start.'
