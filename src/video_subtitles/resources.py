"""System resource detection and bounded OCR execution planning."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil

_GIB = 1024**3
_MIB = 1024**2
_FRAME_MEMORY_ESTIMATE = 128 * _MIB
_MIN_MEMORY_RESERVE = _GIB
_MEMORY_RESERVE_RATIO = 0.15
_GPU_MEMORY_UTILIZATION = 0.78
_GPU_ENGINE_RESERVE_RATIO = 0.10


@dataclass(frozen=True)
class SystemResources:
    cpu_cores: int
    available_ram_bytes: int
    total_ram_bytes: int
    total_gpu_memory_bytes: int | None = None
    logical_cpu_threads: int | None = None
    free_gpu_memory_bytes: int | None = None

    @property
    def effective_cpu_threads(self) -> int:
        return max(1, self.logical_cpu_threads or self.cpu_cores)


@dataclass(frozen=True)
class OCRResourcePlan:
    cpu_threads: int
    frame_queue_size: int
    batch_size: int
    memory_budget_bytes: int


def detect_system_resources() -> SystemResources:
    """Return effective CPU/RAM limits for the local OCR client and decoder."""
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    effective_threads = _effective_cpu_count(max(1, logical))
    effective_cores = min(max(1, physical), effective_threads)

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

    gpu_total, gpu_free = _nvidia_gpu_memory()
    return SystemResources(
        cpu_cores=effective_cores,
        available_ram_bytes=max(0, available_ram),
        total_ram_bytes=max(0, total_ram),
        total_gpu_memory_bytes=gpu_total,
        logical_cpu_threads=effective_threads,
        free_gpu_memory_bytes=gpu_free,
    )


def plan_ocr_resources(
    resources: SystemResources,
    concurrency: int | None = None,
    backend: str = "vllm",
) -> OCRResourcePlan:
    """Size client queues and service request concurrency for available hardware."""
    buffer_budget = _host_memory_budget(resources)
    cpu_threads = resources.effective_cpu_threads
    queue_size = max(1, _frame_queue_size(buffer_budget))
    recommended = recommend_ocr_concurrency(resources, backend)
    requested = recommended if concurrency is None else max(1, concurrency)
    batch_size = max(1, min(requested, queue_size))
    return OCRResourcePlan(
        cpu_threads=cpu_threads,
        frame_queue_size=queue_size,
        batch_size=batch_size,
        memory_budget_bytes=buffer_budget,
    )


def recommend_ocr_concurrency(resources: SystemResources, backend: str) -> int:
    """Estimate safe parallel requests from effective CPU, RAM, and GPU capacity."""
    cpu_slots = resources.effective_cpu_threads
    memory_budget = _host_memory_budget(resources)
    context_limit = 4096 if backend == "llama-cpp" else None
    memory_per_request = _FRAME_MEMORY_ESTIMATE + _kv_bytes_per_sequence(context_limit)
    memory_slots = max(1, memory_budget // max(1, memory_per_request))
    if backend == "llama-cpp":
        return min(cpu_slots, memory_slots)
    if backend == "vllm":
        return min(cpu_slots, memory_slots, _vllm_sequence_capacity(resources))
    raise ValueError(f"Unsupported OCR service backend: {backend}.")


def _nvidia_gpu_memory() -> tuple[int | None, int | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None, None
    try:
        total, free = (int(value.strip()) * _MIB for value in result.stdout.splitlines()[0].split(","))
        return total, free
    except (IndexError, ValueError):
        return None, None


def _effective_cpu_count(logical_threads: int) -> int:
    effective = logical_threads
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


def _frame_queue_size(memory_budget: int) -> int:
    return max(0, memory_budget // _FRAME_MEMORY_ESTIMATE)


def _host_memory_budget(resources: SystemResources) -> int:
    reserve = max(
        _MIN_MEMORY_RESERVE,
        int(max(resources.total_ram_bytes, _GIB) * _MEMORY_RESERVE_RATIO),
    )
    return max(0, resources.available_ram_bytes - reserve)


def _model_metadata() -> tuple[int, int, int]:
    """Return KV bytes/token, model weight bytes, and context length for PaddleOCR-VL-1.6."""
    root = Path(__file__).resolve().parents[2]
    cache_root = os.environ.get("PADDLE_PDX_CACHE_HOME")
    model_dir = (
        Path(cache_root) / "official_models" / "PaddleOCR-VL-1.6"
        if cache_root
        else root / "downloads" / "paddlex" / "official_models" / "PaddleOCR-VL-1.6"
    )
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = {}
    layers = int(config.get("num_hidden_layers", 18))
    kv_heads = int(config.get("num_key_value_heads", 2))
    head_dim = int(config.get("head_dim", 128))
    dtype = str(config.get("torch_dtype", "bfloat16"))
    bytes_per_value = 4 if dtype in {"float32", "fp32"} else 2
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_value
    weight_bytes = sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
    if weight_bytes <= 0:
        weight_bytes = 2 * _GIB
    context_length = 8192
    try:
        text = (root / "environment" / "vllm-server.yaml").read_text(encoding="utf-8")
        match = re.search(r"^max-model-len:\s*(\d+)", text, re.MULTILINE)
        if match:
            context_length = int(match.group(1))
    except OSError:
        pass
    return kv_bytes_per_token, weight_bytes, context_length


def _kv_bytes_per_sequence(context_limit: int | None = None) -> int:
    kv_bytes_per_token, _, context_length = _model_metadata()
    if context_limit is not None:
        context_length = min(context_length, context_limit)
    return kv_bytes_per_token * context_length


def _vllm_sequence_capacity(resources: SystemResources) -> int:
    total = resources.total_gpu_memory_bytes
    if total is None:
        return 1
    kv_bytes_per_token, weight_bytes, context_length = _model_metadata()
    model_budget = int(total * _GPU_MEMORY_UTILIZATION)
    engine_reserve = max(_MIN_MEMORY_RESERVE, int(model_budget * _GPU_ENGINE_RESERVE_RATIO))
    kv_budget = max(0, model_budget - weight_bytes - engine_reserve)
    return max(1, kv_budget // max(1, kv_bytes_per_token * context_length))


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


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as source:
            return source.read().strip()
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
