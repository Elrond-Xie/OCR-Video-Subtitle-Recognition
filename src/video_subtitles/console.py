"""Shared progress rendering and screen-plus-file logging for the CLI."""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configured_log_dir() -> Path:
    root = _project_root()
    configured = os.environ.get("LOG_DIR")
    if not configured:
        manifest = root / "environment" / "ocr-versions.env"
        try:
            for line in manifest.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "LOG_DIR":
                    configured = value.strip().strip("\"'")
                    break
        except OSError:
            pass
    path = Path(configured or "logs").expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def local_log_saving_enabled() -> bool:
    configured = os.environ.get("SAVE_LOGS")
    if configured is None:
        manifest = _project_root() / "environment" / "ocr-versions.env"
        try:
            for line in manifest.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "SAVE_LOGS":
                    configured = value.strip().strip("\"'")
                    break
        except OSError:
            pass
    return str(configured or "true").strip().lower() in {"1", "true", "yes", "on"}


class _TeeStream:
    """Write to the original console and a log while preserving TTY behavior."""

    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log
        self._lock = threading.Lock()

    def write(self, value: str) -> int:
        with self._lock:
            written = self.console.write(value)
            # Progress renderers use carriage returns. Expand those in the file so
            # each update remains readable in editors and non-interactive logs.
            self.log.write(value.replace("\r", "\n"))
            return written

    def flush(self) -> None:
        with self._lock:
            self.console.flush()
            self.log.flush()

    def isatty(self) -> bool:
        return self.console.isatty()

    def fileno(self) -> int:
        return self.console.fileno()

    def __getattr__(self, name: str):
        return getattr(self.console, name)

    @property
    def encoding(self) -> str | None:
        return self.console.encoding

    @property
    def errors(self) -> str | None:
        return self.console.errors


def start_runtime_logging() -> Path | None:
    if not local_log_saving_enabled():
        return None
    log_dir = configured_log_dir() / "run"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"run-{stamp}-{os.getpid()}.log"
    log = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, log)  # type: ignore[assignment]
    print(f"Runtime log: {log_path}", file=sys.stderr, flush=True)
    return log_path


def _format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}" if hours else f"{minutes:02d}:{seconds_part:02d}"


class ProgressDisplay:
    """Render determinate progress in a tqdm-like style, with plain log lines."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr
        self.started_at: float | None = None
        self._last_log_at = 0.0
        self._last_percent = -1
        self._active = False

    def update(self, label: str, current: float, total: float | None) -> None:
        if self.started_at is None:
            self.started_at = time.monotonic()
        elapsed = time.monotonic() - self.started_at
        if total is not None and total > 0:
            fraction = min(1.0, max(0.0, current / total))
            percent = int(fraction * 100)
            width = 24
            units = int(fraction * width)
            bar = "█" * units + ("▏" if units < width and fraction * width - units >= 0.5 else "")
            bar = bar.ljust(width, " ")
            remaining = (elapsed / fraction * (1 - fraction)) if fraction > 0 else 0
            message = (
                f"{label}: {percent:3d}%|{bar}| {current:6.1f}/{total:.1f}s "
                f"[{_format_time(elapsed)}<{_format_time(remaining)}]"
            )
            should_log = percent != self._last_percent and (
                percent == 100 or percent - self._last_percent >= 5 or elapsed - self._last_log_at >= 5
            )
            self._last_percent = percent
        else:
            message = f"{label}: processed {_format_time(current)} [elapsed {_format_time(elapsed)}; duration unknown]"
            should_log = elapsed - self._last_log_at >= 5

        if self.stream.isatty():
            print("\r" + message, end="", file=self.stream, flush=True)
            self._active = True
        elif should_log:
            print(message, file=self.stream, flush=True)
        if should_log:
            self._last_log_at = elapsed

    def close(self) -> None:
        if self._active:
            print(file=self.stream, flush=True)
            self._active = False


@contextmanager
def long_operation(label: str, interval: float = 10.0) -> Iterator[None]:
    """Periodically report elapsed time for work that has no percentage."""
    started = time.monotonic()
    stopped = threading.Event()
    stream = sys.stderr

    def report() -> None:
        while not stopped.wait(interval):
            elapsed = _format_time(time.monotonic() - started)
            if stream.isatty():
                print(f"\r{label} … elapsed {elapsed}", end="", file=stream, flush=True)
            else:
                print(f"{label} is still running (elapsed {elapsed})", file=stream, flush=True)

    print(f"{label}…", file=stream, flush=True)
    worker = threading.Thread(target=report, name="progress-notice", daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        stopped.set()
        worker.join(timeout=1)
        if stream.isatty():
            print(file=stream, flush=True)
        elapsed = _format_time(time.monotonic() - started)
        print(f"{label} stopped after {elapsed}", file=stream, flush=True)
        raise
    finally:
        stopped.set()
        worker.join(timeout=1)
    if stream.isatty():
        print(file=stream, flush=True)
    elapsed = _format_time(time.monotonic() - started)
    print(f"{label} finished (elapsed {elapsed})", file=stream, flush=True)
