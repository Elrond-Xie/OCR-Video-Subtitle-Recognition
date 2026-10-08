"""System resource detection and bounded OCR execution planning."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Literal

import psutil

_GIB = 1024**3
_MIB = 1024**2
_MAX_CPU_THREADS = 32
_MAX_FRAME_QUEUE = 8
_MAX_GPU_BATCH = 4
_VL_CPU_MIN_RAM = 12 * _GIB
_VL_GPU_MIN_RAM = 5 * _GIB
_VL_GPU_MIN_VRAM = 8 * _GIB


@dataclass(frozen=True)
class SystemResources:
    cpu_cores: int
    available_ram_bytes: int
    total_ram_bytes: int
    free_vram_bytes: int | None
    nvidia_device_visible: bool
    nvidia_device_id: int | None


@dataclass(frozen=True)
class OCRResourcePlan:
    engine: Literal["paddle-vl", "rapidocr"]
    device: Literal["cuda", "cpu"]
    device_id: int | None
    cpu_threads: int
    frame_queue_size: int
    batch_size: int
    memory_budget_bytes: int


def detect_system_resources() -> SystemResources:
    """Return effective CPU/RAM limits and the NVIDIA device with most free VRAM."""
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    cpu_cores = _effective_cpu_count(max(1, physical))

    memory = psutil.virtual_memory()
    total_ram = int(memory.total)
    available_ram = int(memory.available)
    memory_limit, memory_usage = _cgroup_memory()
    if memory_limit is not None:
        total_ram = min(total_ram, memory_limit)
        if memory_usage is not None:
            available_ram = min(available_ram, max(0, memory_limit - memory_usage))
        else:
            available_ram = min(available_ram, memory_limit)

    gpu_id, free_vram = _nvidia_free_memory()
    return SystemResources(
        cpu_cores=cpu_cores,
        available_ram_bytes=max(0, available_ram),
        total_ram_bytes=max(0, total_ram),
        free_vram_bytes=free_vram,
        nvidia_device_visible=gpu_id is not None,
        nvidia_device_id=gpu_id,
    )


def plan_ocr_resources(
    resources: SystemResources,
    cuda_runtime_ready: bool,
) -> OCRResourcePlan:
    """Choose a model/device tier and bound threads, batches, and queued frames."""
    total = max(resources.total_ram_bytes, _GIB)
    available = resources.available_ram_bytes
    reserve = max(_GIB, int(total * 0.15))
    memory_budget = max(0, available - reserve)

    cores = max(1, resources.cpu_cores)
    cpu_threads = min(_MAX_CPU_THREADS, max(1, cores - 1 if cores > 2 else cores))
    frame_queue_size = min(
        _MAX_FRAME_QUEUE,
        max(0, memory_budget // (128 * _MIB)),
    )

    gpu_is_usable = (
        cuda_runtime_ready
        and resources.nvidia_device_id is not None
        and resources.free_vram_bytes is not None
        and resources.free_vram_bytes >= _VL_GPU_MIN_VRAM
        and available >= _VL_GPU_MIN_RAM
    )
    if gpu_is_usable:
        free_vram = resources.free_vram_bytes or 0
        batch_size = 4 if free_vram >= 20 * _GIB else 2 if free_vram >= 12 * _GIB else 1
        return OCRResourcePlan(
            engine="paddle-vl",
            device="cuda",
            device_id=resources.nvidia_device_id,
            cpu_threads=min(cpu_threads, 8),
            frame_queue_size=frame_queue_size,
            batch_size=batch_size,
            memory_budget_bytes=memory_budget,
        )

    if available >= _VL_CPU_MIN_RAM and memory_budget >= 8 * _GIB:
        batch_size = min(4, max(1, memory_budget // (3 * _GIB)))
        return OCRResourcePlan(
            engine="paddle-vl",
            device="cpu",
            device_id=None,
            cpu_threads=cpu_threads,
            frame_queue_size=frame_queue_size,
            batch_size=batch_size,
            memory_budget_bytes=memory_budget,
        )

    return OCRResourcePlan(
        engine="rapidocr",
        device="cpu",
        device_id=None,
        cpu_threads=cpu_threads,
        frame_queue_size=frame_queue_size,
        batch_size=1,
        memory_budget_bytes=memory_budget,
    )


def _effective_cpu_count(physical_cores: int) -> int:
    effective = physical_cores
    try:
        affinity = len(os.sched_getaffinity(0))
        effective = min(effective, max(1, affinity))
    except (AttributeError, OSError):
        pass

    quota, period = _cgroup_cpu_quota()
    if quota is not None and period and quota > 0:
        quota_cores = max(1, (quota + period - 1) // period)
        effective = min(effective, quota_cores)
    return max(1, effective)


def _cgroup_cpu_quota() -> tuple[int | None, int | None]:
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        fields = cpu_max.split()
        if len(fields) == 2 and fields[0] != "max":
            try:
                return int(fields[0]), int(fields[1])
            except ValueError:
                return None, None

    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    return quota, period


def _cgroup_memory() -> tuple[int | None, int | None]:
    limit_text = _read_text("/sys/fs/cgroup/memory.max")
    usage_text = _read_text("/sys/fs/cgroup/memory.current")
    if limit_text and limit_text != "max":
        try:
            return int(limit_text), int(usage_text) if usage_text else None
        except ValueError:
            return None, None

    limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    usage = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit is not None and limit >= 2**60:
        return None, None
    return limit, usage


def _nvidia_free_memory() -> tuple[int | None, int | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None

    candidates: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            candidates.append((int(fields[0]), int(float(fields[1]) * _MIB)))
        except ValueError:
            continue
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1])


def _read_text(path: str) -> str | None:
    try:
        return open(path, encoding="ascii").read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    value = _read_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
