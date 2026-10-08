"""Local OCR for subtitle text burned into video frames."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Protocol, Sequence

from .errors import DependencyError, OCRProcessingError
from .subtitles import segments_to_srt

_SAMPLE_INTERVAL = 0.5
_MIN_CONFIDENCE = 0.50
_LOWER_FRAME_RATIO = 0.55
_MERGE_SIMILARITY = 0.90
_MAX_MERGE_GAP = 1.0
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class OCRText:
    text: str
    box: tuple[tuple[float, float], ...] | None
    confidence: float | None


class FrameOCR(Protocol):
    engine_name: str
    device_name: str

    def recognize_batch(self, images: Sequence[Any]) -> list[list[OCRText]]: ...

    def close(self) -> None: ...


class RapidOCREngine:
    engine_name = "RapidOCR"
    device_name = "CPU"

    def __init__(self, cpu_threads: int) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise DependencyError("RapidOCR is missing. Run the platform setup script to install OCR dependencies.") from exc
        try:
            self._engine = RapidOCR(
                params={
                    "EngineConfig.onnxruntime.intra_op_num_threads": max(1, cpu_threads),
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                }
            )
        except Exception as exc:
            raise OCRProcessingError(f"Could not initialize RapidOCR: {exc}") from exc

    def recognize_batch(self, images: Sequence[Any]) -> list[list[OCRText]]:
        results = []
        for image in images:
            try:
                result = self._engine(image)
            except Exception as exc:
                raise OCRProcessingError(f"RapidOCR failed while processing a video frame: {exc}") from exc
            if isinstance(result, tuple):
                result = result[0]
            results.append(_rapid_result(result))
        return results

    def close(self) -> None:
        self._engine = None


class PaddleOCRVLEngine:
    engine_name = "PaddleOCR-VL"

    def __init__(self, device: str, cpu_threads: int, batch_size: int) -> None:
        self.device_name = "CUDA" if device.startswith("gpu:") or device == "cuda" else "CPU"
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise DependencyError("PaddleOCR-VL is missing. Run the platform setup script to install the VL runtime.") from exc
        kwargs: dict[str, Any] = {"device": device if device.startswith("gpu:") else "cpu"}
        if device == "cpu":
            kwargs["cpu_threads"] = max(1, cpu_threads)
        try:
            self._engine = PaddleOCRVL(**kwargs)
        except Exception as exc:
            raise OCRProcessingError(f"Could not initialize PaddleOCR-VL: {exc}") from exc
        self.batch_size = max(1, batch_size)

    def recognize_batch(self, images: Sequence[Any]) -> list[list[OCRText]]:
        try:
            crops = [_subtitle_crop(image) for image in images]
            output = self._predict(crops)
            normalized = [_paddle_result(result) for result in output]
            for index, entries in enumerate(normalized):
                if entries:
                    continue
                # The crop may omit unusually positioned captions. Use the full frame once.
                normalized[index] = _paddle_result(self._predict([images[index]])[0])
            return normalized
        except Exception as exc:
            if isinstance(exc, OCRProcessingError):
                raise
            raise OCRProcessingError(f"PaddleOCR-VL failed while processing video frames: {exc}") from exc

    def _predict(self, images: Sequence[Any]) -> list[Any]:
        output: list[Any] = []
        # PaddleOCR-VL exposes predict_iter for streamed, memory-bounded inference.
        iterator = self._engine.predict_iter(list(images), prompt_label="ocr")
        for result in iterator:
            output.append(result)
        if len(output) != len(images):
            raise OCRProcessingError("PaddleOCR-VL returned an unexpected number of frame results.")
        return output

    def close(self) -> None:
        self._engine = None


class FallbackOCREngine:
    """Use RapidOCR for empty VL results and disable VL after any runtime failure."""

    def __init__(self, primary: FrameOCR, fallback: FrameOCR, on_fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.on_fallback = on_fallback
        self._primary_enabled = True
        self.engine_name = primary.engine_name
        self.device_name = primary.device_name

    def recognize_batch(self, images: Sequence[Any]) -> list[list[OCRText]]:
        if not self._primary_enabled:
            return self.fallback.recognize_batch(images)
        try:
            primary_results = self.primary.recognize_batch(images)
        except Exception as exc:
            self._disable_primary(str(exc))
            return self.fallback.recognize_batch(images)
        output = []
        for image, entries in zip(images, primary_results):
            if _usable(entries, image):
                output.append(entries)
            else:
                output.append(self.fallback.recognize_batch([image])[0])
        return output

    def _disable_primary(self, reason: str) -> None:
        if not self._primary_enabled:
            return
        self._primary_enabled = False
        self.primary.close()
        self.engine_name = self.fallback.engine_name
        self.device_name = self.fallback.device_name
        if self.on_fallback is not None:
            self.on_fallback(reason)

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()


def create_rapid_engine(cpu_threads: int | None = None) -> FrameOCR:
    threads = cpu_threads if cpu_threads is not None else max(1, min(8, os.cpu_count() or 1))
    return RapidOCREngine(threads)


def create_paddle_vl_engine(device: str, cpu_threads: int, batch_size: int) -> FrameOCR:
    return PaddleOCRVLEngine(device, cpu_threads, batch_size)


def create_ocr_engine(on_model_load: Callable[[], None] | None = None) -> FrameOCR:
    """Compatibility constructor for callers that need a RapidOCR CPU engine."""
    if on_model_load is not None:
        on_model_load()
    return create_rapid_engine()


def create_ocr_runtime(resources=None, on_model_load=None, on_fallback=None):
    """Detect available resources, initialize the selected engine, and warm it up."""
    from .resources import detect_system_resources, plan_ocr_resources

    system = resources or detect_system_resources()
    cuda_ready = False
    if system.nvidia_device_visible:
        try:
            import paddle

            cuda_ready = paddle.device.cuda.device_count() > 0
        except Exception:
            cuda_ready = False
    plan = plan_ocr_resources(system, cuda_ready)
    if on_model_load is not None:
        on_model_load()
    if plan.engine == "rapidocr":
        return create_rapid_engine(plan.cpu_threads), plan
    if sys.platform == "darwin":
        plan = _rapid_plan(plan)
        return create_rapid_engine(plan.cpu_threads), plan
    try:
        import importlib.util

        if importlib.util.find_spec("paddleocr") is None:
            if on_fallback is not None:
                on_fallback("PaddleOCR is not installed in this Python environment")
            plan = _rapid_plan(plan)
            return create_rapid_engine(plan.cpu_threads), plan
    except (ImportError, ValueError):
        if on_fallback is not None:
            on_fallback("PaddleOCR is not installed in this Python environment")
        plan = _rapid_plan(plan)
        return create_rapid_engine(plan.cpu_threads), plan
    try:
        vl_device = f"gpu:{plan.device_id or 0}" if plan.device == "cuda" else "cpu"
        vl = create_paddle_vl_engine(vl_device, plan.cpu_threads, plan.batch_size)
        # Small in-memory warm-up catches missing model files and unusable CUDA contexts.
        import numpy as np

        vl.recognize_batch([np.zeros((160, 640, 3), dtype=np.uint8)])
    except Exception as exc:
        try:
            vl.close()
        except (UnboundLocalError, AttributeError):
            pass
        if on_fallback is not None:
            on_fallback(str(exc))
        plan = _rapid_plan(plan)
        return create_rapid_engine(plan.cpu_threads), plan
    rapid = create_rapid_engine(plan.cpu_threads)
    return FallbackOCREngine(vl, rapid, on_fallback), plan


def _rapid_plan(plan):
    from dataclasses import replace

    return replace(plan, engine="rapidocr", device="cpu", device_id=None, batch_size=1)


def subtitle_from_frames(
    frames: Iterable[tuple[float, Any]],
    engine: FrameOCR,
    duration: float | None = None,
    on_progress: Callable[[float, float | None], None] | None = None,
    batch_size: int = 1,
) -> str | None:
    """Recognize caption-like text in bounded batches and convert stable detections into SRT."""
    detections: list[tuple[float, str]] = []
    next_progress = 0.0
    if on_progress is not None:
        on_progress(0.0, duration)
    size = max(1, batch_size)
    batch: list[tuple[float, Any]] = []

    def consume(items: list[tuple[float, Any]]) -> None:
        if not items:
            return
        results = engine.recognize_batch([image for _, image in items])
        if len(results) != len(items):
            raise OCRProcessingError("OCR returned an unexpected number of frame results.")
        for (timestamp, image), entries in zip(items, results):
            if on_progress is not None and timestamp >= next_progress:
                on_progress(float(timestamp), duration)
            caption = _caption_from_entries(entries, image)
            if caption:
                detections.append((float(timestamp), caption))

    try:
        for timestamp, image in frames:
            batch.append((timestamp, image))
            if len(batch) >= size:
                consume(batch)
                batch = []
                next_progress = float(timestamp) + 1.0
        consume(batch)
    except OCRProcessingError:
        raise
    except Exception as exc:
        raise OCRProcessingError(f"OCR failed while processing video frames: {exc}") from exc

    if on_progress is not None and duration is not None and next_progress < duration:
        on_progress(duration, duration)
    groups = _merge_detections(detections)
    if not groups:
        return None
    segments = [
        (
            max(0.0, group[0][0] - _SAMPLE_INTERVAL / 2),
            max(group[-1][0] + _SAMPLE_INTERVAL / 2, group[0][0] + _SAMPLE_INTERVAL),
            group[0][1],
        )
        for group in groups
    ]
    try:
        return segments_to_srt(segments)
    except Exception as exc:
        raise OCRProcessingError(f"Could not turn OCR captions into SRT: {exc}") from exc


def _rapid_result(result: Any) -> list[OCRText]:
    if result is None:
        return []
    texts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None)
    if texts is None or boxes is None:
        return []
    output = []
    for index, (box, text) in enumerate(zip(boxes, texts)):
        score = None
        try:
            score = float(scores[index]) if scores is not None else None
        except (IndexError, TypeError, ValueError):
            pass
        output.append(OCRText(str(text), _normalize_box(box), score))
    return output


def _paddle_result(result: Any) -> list[OCRText]:
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if value is None and isinstance(result, dict):
        value = result
    if not isinstance(value, dict):
        return []
    # PaddleOCR-VL 1.6 returns a JSON dictionary with parsing_res_list blocks.
    if isinstance(value.get("res"), dict):
        value = value["res"]
    blocks = value.get("parsing_res_list") or []
    output = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("block_content") or block.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        label = str(block.get("block_label") or "").lower()
        if label and label not in {"text", "title", "paragraph", "caption", "list"}:
            continue
        box = _normalize_box(block.get("block_bbox") or block.get("bbox"))
        output.append(OCRText(text, box, None))
    return output


def _normalize_box(value: Any) -> tuple[tuple[float, float], ...] | None:
    try:
        points = tuple((float(point[0]), float(point[1])) for point in value)
        return points if len(points) >= 4 else None
    except (TypeError, ValueError, IndexError):
        try:
            left, top, right, bottom = (float(part) for part in value)
            return ((left, top), (right, top), (right, bottom), (left, bottom))
        except (TypeError, ValueError):
            return None


def _subtitle_crop(image: Any) -> Any:
    height = int(image.shape[0])
    top = int(height * _LOWER_FRAME_RATIO)
    return image[top:, :]


def _usable(entries: list[OCRText], image: Any) -> bool:
    return bool(_caption_from_entries(entries, image))


def _caption_from_entries(entries: list[OCRText], image: Any) -> str | None:
    height = int(image.shape[0])
    lines: list[tuple[float, float, str]] = []
    for entry in entries:
        cleaned = _clean_text(entry.text)
        if not cleaned or (entry.confidence is not None and entry.confidence < _MIN_CONFIDENCE):
            continue
        if entry.box is None:
            continue
        try:
            top = min(point[1] for point in entry.box)
            bottom = max(point[1] for point in entry.box)
            left = min(point[0] for point in entry.box)
        except (TypeError, ValueError, IndexError):
            continue
        # Paddle boxes are relative to the lower crop; RapidOCR boxes are full-frame.
        if top < height * _LOWER_FRAME_RATIO and bottom <= height * (1 - _LOWER_FRAME_RATIO) + 1:
            top += height * _LOWER_FRAME_RATIO
            bottom += height * _LOWER_FRAME_RATIO
        if (top + bottom) / 2 < height * _LOWER_FRAME_RATIO:
            continue
        lines.append((top, left, cleaned))
    if not lines:
        return None
    lines.sort(key=lambda line: (line[0], line[1]))
    return _join_lines([line[2] for line in lines])


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _join_lines(lines: list[str]) -> str:
    if all(
        _CJK.search(line) and len(_CJK.findall(line)) >= len(line.replace(" ", "")) * 0.6
        for line in lines
    ):
        return "".join(line.replace(" ", "") for line in lines)
    return " ".join(lines)


def _similarity(left: str, right: str) -> float:
    normalized_left = re.sub(r"\W", "", left).casefold()
    normalized_right = re.sub(r"\W", "", right).casefold()
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _merge_detections(detections: list[tuple[float, str]]) -> list[list[tuple[float, str]]]:
    groups: list[list[tuple[float, str]]] = []
    for detection in detections:
        if (
            groups
            and detection[0] - groups[-1][-1][0] <= _MAX_MERGE_GAP
            and _similarity(groups[-1][-1][1], detection[1]) >= _MERGE_SIMILARITY
        ):
            groups[-1].append(detection)
        else:
            groups.append([detection])
    return [group for group in groups if len(group) >= 2]
