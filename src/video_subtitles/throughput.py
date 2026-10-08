"""Online OCR batch sizing from measured throughput and runtime headroom."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import psutil

_MEMORY_PRESSURE_RATIO = 0.05
_THROUGHPUT_REGRESSION_RATIO = 0.90
_REQUIRED_REGRESSIONS = 2


@dataclass(frozen=True)
class RuntimeMetrics:
    cpu_percent: float | None
    gpu_percent: float | None
    available_ram_bytes: int
    total_ram_bytes: int
    free_gpu_memory_bytes: int | None
    total_gpu_memory_bytes: int | None

    @property
    def memory_pressure(self) -> bool:
        ram_low = self.available_ram_bytes < self.total_ram_bytes * _MEMORY_PRESSURE_RATIO
        gpu_low = (
            self.free_gpu_memory_bytes is not None
            and self.total_gpu_memory_bytes is not None
            and self.free_gpu_memory_bytes < self.total_gpu_memory_bytes * _MEMORY_PRESSURE_RATIO
        )
        return ram_low or gpu_low


@dataclass(frozen=True)
class BatchObservation:
    processed: int
    elapsed_seconds: float
    frames_per_second: float
    cpu_percent: float | None
    gpu_percent: float | None
    next_batch_size: int
    best_batch_size: int
    memory_pressure: bool
    best_changed: bool


class OCRBatchController:
    """Ramp request concurrency against a safe ceiling and retain measured wins."""

    def __init__(self, maximum_batch_size: int) -> None:
        self.maximum_batch_size = max(1, int(maximum_batch_size))
        self.batch_size = 1
        self.best_batch_size = 1
        self.best_frames_per_second = 0.0
        self._regressions = 0
        self._frames_since_probe = 0
        self._frozen = self.maximum_batch_size == 1
        psutil.cpu_percent(interval=None)

    def observe(self, processed: int, elapsed_seconds: float) -> BatchObservation:
        metrics = sample_runtime_metrics()
        elapsed = max(float(elapsed_seconds), 1e-6)
        throughput = max(0, int(processed)) / elapsed
        previous_best = self.best_frames_per_second
        previous_best_size = self.best_batch_size
        if throughput > self.best_frames_per_second:
            self.best_frames_per_second = throughput
            self.best_batch_size = max(1, int(processed))

        if metrics.memory_pressure:
            self.batch_size = max(1, min(self.best_batch_size, max(1, int(processed) // 2)))
            self.maximum_batch_size = min(self.maximum_batch_size, self.batch_size)
            self._frozen = True
        elif not self._frozen and int(processed) < self.maximum_batch_size:
            if (
                previous_best > 0
                and int(processed) > self.best_batch_size
                and throughput < previous_best * _THROUGHPUT_REGRESSION_RATIO
            ):
                self._regressions += 1
            else:
                self._regressions = 0
            if self._regressions >= _REQUIRED_REGRESSIONS:
                self.batch_size = self.best_batch_size
                self._frozen = True
            else:
                self.batch_size = min(self.maximum_batch_size, max(int(processed) + 1, int(processed) * 2))
        elif self._frozen:
            self._frames_since_probe += int(processed)
            if (
                self.best_batch_size < self.maximum_batch_size
                and self._frames_since_probe >= max(8, self.best_batch_size * 4)
            ):
                self.batch_size = min(self.maximum_batch_size, self.best_batch_size * 2)
                self._frames_since_probe = 0
                self._regressions = 0
                self._frozen = False
            else:
                self.batch_size = self.best_batch_size
        else:
            self.batch_size = self.best_batch_size
            self._frozen = True

        return BatchObservation(
            processed=int(processed),
            elapsed_seconds=elapsed,
            frames_per_second=throughput,
            cpu_percent=metrics.cpu_percent,
            gpu_percent=metrics.gpu_percent,
            next_batch_size=self.batch_size,
            best_batch_size=self.best_batch_size,
            memory_pressure=metrics.memory_pressure,
            best_changed=self.best_batch_size != previous_best_size,
        )

    def back_off_after_memory_error(self, failed_batch_size: int) -> int:
        """Reduce the active batch after an explicit memory-related inference error."""
        self.batch_size = max(1, int(failed_batch_size) // 2)
        self.maximum_batch_size = min(self.maximum_batch_size, self.batch_size)
        self.best_batch_size = min(self.best_batch_size, self.batch_size)
        self._frozen = self.batch_size == 1
        return self.batch_size


def sample_runtime_metrics() -> RuntimeMetrics:
    from .resources import detect_system_resources

    resources = detect_system_resources()
    gpu_percent = None
    gpu_free = resources.free_gpu_memory_bytes
    gpu_total = resources.total_gpu_memory_bytes
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
            gpu_percent = float(fields[0])
            gpu_free = int(fields[1]) * 1024**2
            gpu_total = int(fields[2]) * 1024**2
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    return RuntimeMetrics(
        cpu_percent=psutil.cpu_percent(interval=None),
        gpu_percent=gpu_percent,
        available_ram_bytes=resources.available_ram_bytes,
        total_ram_bytes=resources.total_ram_bytes,
        free_gpu_memory_bytes=gpu_free,
        total_gpu_memory_bytes=gpu_total,
    )


def log_batch_observation(observation: BatchObservation, logger=print) -> None:
    if observation.next_batch_size == observation.processed and not observation.best_changed and not observation.memory_pressure:
        return
    cpu = "n/a" if observation.cpu_percent is None else f"{observation.cpu_percent:.0f}%"
    gpu = "n/a" if observation.gpu_percent is None else f"{observation.gpu_percent:.0f}%"
    pressure = ", memory pressure detected" if observation.memory_pressure else ""
    logger(
        f"[INFO] OCR tuning: batch {observation.processed}, "
        f"{observation.frames_per_second:.2f} frames/s, CPU {cpu}, GPU {gpu}; "
        f"next {observation.next_batch_size}, best {observation.best_batch_size}{pressure}"
    )
