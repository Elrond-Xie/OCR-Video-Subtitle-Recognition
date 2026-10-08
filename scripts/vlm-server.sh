#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/environment/ocr-versions.env"
PREFIX="$ROOT/.envs/ocr-vlm-server"
CLIENT_PY="$ROOT/.envs/ocr-client/bin/python"
PID_DIR="$ROOT/.runtime"
PID_FILE="$PID_DIR/vlm-server.pid"
LOG_DIR_PATH="$ROOT/logs/vlm-server"
LOG_FILE="$LOG_DIR_PATH/server.log"
SERVER_CONFIG="$PID_DIR/vllm-server.yaml"
TMP_DIR="$ROOT/output/.tmp-vlm-server"
export PATH="$PREFIX/bin:$PATH"
HOST="${OCR_VLLM_HOST:-127.0.0.1}"
PORT="${OCR_VLLM_PORT:-8118}"

resolve_path() {
  local path="$1"
  if [[ "$path" != /* ]]; then path="$ROOT/$path"; fi
  printf '%s\n' "$path"
}

DOWNLOAD_ROOT="$(resolve_path "$DOWNLOAD_DIR")"
mkdir -p "$PID_DIR" "$LOG_DIR_PATH" "$DOWNLOAD_ROOT/xdg" "$DOWNLOAD_ROOT/huggingface" \
  "$DOWNLOAD_ROOT/paddlex" "$DOWNLOAD_ROOT/torch" "$TMP_DIR"

pid_is_running() {
  [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

health_check() {
  "$PREFIX/bin/python" - "$HOST" "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
url = f"http://{sys.argv[1]}:{sys.argv[2]}/health"
with urllib.request.urlopen(url, timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
}

start_server() {
  if [[ ! -x "$PREFIX/bin/paddleocr" ]]; then
    echo '[ERROR] vLLM server is not installed. Run ./scripts/setup-vlm-server.sh.' >&2
    exit 2
  fi
  if pid_is_running; then
    if health_check; then
      echo "[INFO] vLLM server is already healthy at http://$HOST:$PORT/v1"
      return
    fi
    echo "[INFO] vLLM server process $(cat "$PID_FILE") is still starting."
    return
  fi
  if [[ ! -x "$CLIENT_PY" ]]; then
    echo '[ERROR] OCR client environment is missing. Run ./scripts/configure --setup first.' >&2
    exit 2
  fi
  rm -f -- "$PID_FILE"
  detected_max_seqs="$(PYTHONPATH="$ROOT/src" "$CLIENT_PY" - <<'PY'
from video_subtitles.resources import detect_system_resources, recommend_ocr_concurrency
print(recommend_ocr_concurrency(detect_system_resources(), "vllm"))
PY
  )"
  max_seqs="${OCR_VLLM_MAX_SEQS:-$detected_max_seqs}"
  if [[ ! "$max_seqs" =~ ^[1-9][0-9]*$ ]]; then
    echo '[ERROR] OCR_VLLM_MAX_SEQS must be a positive integer.' >&2
    exit 2
  fi
  sed -E "s/^max-num-seqs:.*/max-num-seqs: $max_seqs/" \
    "$ROOT/environment/vllm-server.yaml" > "$SERVER_CONFIG"
  export PIP_CACHE_DIR="$DOWNLOAD_ROOT/pip"
  export XDG_CACHE_HOME="$DOWNLOAD_ROOT/xdg"
  export HF_HOME="$DOWNLOAD_ROOT/huggingface"
  export HF_HUB_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
  export PADDLE_PDX_CACHE_HOME="$DOWNLOAD_ROOT/paddlex"
  export TORCH_HOME="$DOWNLOAD_ROOT/torch"
  export TMPDIR="$TMP_DIR"
  export PYTHONUNBUFFERED=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  setsid "$PREFIX/bin/python" "$PREFIX/bin/paddleocr" genai_server \
    --model_name PaddleOCR-VL-1.6-0.9B \
    --backend vllm \
    --backend_config "$SERVER_CONFIG" \
    --host "$HOST" --port "$PORT" \
    >>"$LOG_FILE" 2>&1 </dev/null &
  echo "$!" > "$PID_FILE"
  echo "[INFO] Starting PaddleOCR-VL-1.6 with vLLM at http://$HOST:$PORT/v1 ($max_seqs concurrent sequences from detected system capacity)"
  echo "[INFO] Server log: $LOG_FILE"
  for _ in $(seq 1 600); do
    if health_check; then
      echo '[INFO] vLLM server is healthy.'
      return
    fi
    if ! pid_is_running; then
      rm -f -- "$PID_FILE"
      echo '[ERROR] vLLM server stopped during startup. Recent log output:' >&2
      tail -n 80 "$LOG_FILE" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "[WARN] vLLM is still loading. Check $LOG_FILE; the process remains running."
}

stop_server() {
  if ! pid_is_running; then
    rm -f -- "$PID_FILE"
    echo '[INFO] vLLM server is not running.'
    return
  fi
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f -- "$PID_FILE"
      rm -rf -- "$TMP_DIR"
      echo '[INFO] vLLM server stopped.'
      return
    fi
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  rm -f -- "$PID_FILE"
  rm -rf -- "$TMP_DIR"
  echo '[INFO] vLLM server stopped.'
}

show_status() {
  if pid_is_running && health_check; then
    echo "[INFO] vLLM server is healthy at http://$HOST:$PORT/v1 (PID $(cat "$PID_FILE"))."
  elif pid_is_running; then
    echo "[INFO] vLLM server is starting (PID $(cat "$PID_FILE")). Log: $LOG_FILE"
  else
    rm -f -- "$PID_FILE"
    echo '[INFO] vLLM server is not running.'
  fi
}

case "${1:-}" in
  start) start_server ;;
  stop) stop_server ;;
  status) show_status ;;
  logs) tail -n 100 -f "$LOG_FILE" ;;
  *)
    echo 'Usage: scripts/vlm-server.sh {start|stop|status|logs}' >&2
    exit 2
    ;;
esac
