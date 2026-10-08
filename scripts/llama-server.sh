#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/environment/ocr-versions.env"
resolve_path() { [[ "$1" == /* ]] && printf '%s\n' "$1" || printf '%s/%s\n' "$ROOT" "$1"; }
DOWNLOAD_ROOT="$(resolve_path "$DOWNLOAD_DIR")"
PREFIX="$ROOT/.envs/ocr-llama-server"
CLIENT_PY="$ROOT/.envs/ocr-client/bin/python"
MODEL_DIR="$DOWNLOAD_ROOT/paddleocr-vl-gguf"
PID_DIR="$ROOT/.runtime"
PID_FILE="$PID_DIR/llama-server.pid"
LOG_DIR_PATH="$ROOT/logs/llama-server"
LOG_FILE="$LOG_DIR_PATH/server.log"
TMP_DIR="$ROOT/output/.tmp-llama-server"
HOST="${OCR_LLAMA_HOST:-127.0.0.1}"
PORT="${OCR_LLAMA_PORT:-8111}"
MODEL="$MODEL_DIR/PaddleOCR-VL-1.6-GGUF.gguf"
MMPROJ="$MODEL_DIR/PaddleOCR-VL-1.6-GGUF-mmproj.gguf"

mkdir -p "$PID_DIR" "$LOG_DIR_PATH" "$TMP_DIR" "$DOWNLOAD_ROOT/llama.cpp"
pid_is_running() { [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
health_check() { curl -fsS --max-time 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; }

start_server() {
  if [[ ! -x "$PREFIX/bin/llama-server" || ! -s "$MODEL" || ! -s "$MMPROJ" ]]; then
    echo '[ERROR] CPU OCR service is incomplete. Run ./scripts/setup-llama-server.sh.' >&2
    exit 2
  fi
  if [[ ! -x "$CLIENT_PY" ]]; then
    echo '[ERROR] OCR client environment is missing. Run ./scripts/configure --setup first.' >&2
    exit 2
  fi
  if pid_is_running; then
    if health_check; then echo "[INFO] llama.cpp server is healthy at http://$HOST:$PORT/v1"; else echo "[INFO] llama.cpp server is starting (PID $(cat "$PID_FILE"))."; fi
    return
  fi
  rm -f -- "$PID_FILE"
  export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TMPDIR="$TMP_DIR" LLAMA_CACHE="$DOWNLOAD_ROOT/llama.cpp"
  export HF_HOME="$DOWNLOAD_ROOT/huggingface" HF_HUB_CACHE="$DOWNLOAD_ROOT/huggingface/hub"
  read -r detected_parallel detected_threads detected_batch_threads < <(
    PYTHONPATH="$ROOT/src" "$CLIENT_PY" - <<'PY'
from video_subtitles.resources import detect_system_resources, recommend_ocr_concurrency
resources = detect_system_resources()
parallel = recommend_ocr_concurrency(resources, "llama-cpp")
print(parallel, resources.effective_cpu_threads, resources.effective_cpu_threads)
PY
  )
  parallel="${OCR_LLAMA_PARALLEL:-$detected_parallel}"
  if [[ ! "$parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo '[ERROR] OCR_LLAMA_PARALLEL must be a positive integer.' >&2
    exit 2
  fi
  threads="${OCR_LLAMA_THREADS:-$detected_threads}"
  batch_threads="${OCR_LLAMA_THREADS_BATCH:-$detected_batch_threads}"
  if [[ ! "$threads" =~ ^[1-9][0-9]*$ || ! "$batch_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo '[ERROR] OCR_LLAMA_THREADS and OCR_LLAMA_THREADS_BATCH must be positive integers.' >&2
    exit 2
  fi
  context_size=$((4096 * parallel))
  export OMP_NUM_THREADS="$batch_threads"
  setsid "$PREFIX/bin/llama-server" \
    --model "$MODEL" --mmproj "$MMPROJ" \
    --host "$HOST" --port "$PORT" --ctx-size "$context_size" --parallel "$parallel" \
    --n-gpu-layers 0 --temp 0 --threads "$threads" --threads-batch "$batch_threads" \
    >>"$LOG_FILE" 2>&1 </dev/null &
  echo "$!" > "$PID_FILE"
  echo "[INFO] Starting PaddleOCR-VL-1.6 GGUF with llama.cpp on CPU at http://$HOST:$PORT/v1 ($parallel request slots, $threads generation threads, $batch_threads batch/prompt threads and 4096 context tokens per request)"
  echo "[INFO] Server log: $LOG_FILE"
  for _ in $(seq 1 600); do
    if health_check; then echo '[INFO] llama.cpp server is healthy.'; return; fi
    if ! pid_is_running; then
      rm -f -- "$PID_FILE"
      echo '[ERROR] llama.cpp server stopped during startup. Recent log output:' >&2
      tail -n 80 "$LOG_FILE" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "[WARN] llama.cpp is still loading. Check $LOG_FILE; the process remains running."
}

stop_server() {
  if ! pid_is_running; then rm -f -- "$PID_FILE"; echo '[INFO] llama.cpp server is not running.'; return; fi
  local pid; pid="$(cat "$PID_FILE")"; kill "$pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then rm -f -- "$PID_FILE"; rm -rf -- "$TMP_DIR"; echo '[INFO] llama.cpp server stopped.'; return; fi
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  rm -f -- "$PID_FILE"; rm -rf -- "$TMP_DIR"
  echo '[INFO] llama.cpp server stopped.'
}

show_status() {
  if pid_is_running && health_check; then echo "[INFO] llama.cpp server is healthy at http://$HOST:$PORT/v1 (PID $(cat "$PID_FILE"))."
  elif pid_is_running; then echo "[INFO] llama.cpp server is starting (PID $(cat "$PID_FILE")). Log: $LOG_FILE"
  else rm -f -- "$PID_FILE"; echo '[INFO] llama.cpp server is not running.'; fi
}

case "${1:-}" in
  start) start_server ;;
  stop) stop_server ;;
  status) show_status ;;
  logs) tail -n 100 -f "$LOG_FILE" ;;
  *) echo 'Usage: scripts/llama-server.sh {start|stop|status|logs}' >&2; exit 2 ;;
esac
