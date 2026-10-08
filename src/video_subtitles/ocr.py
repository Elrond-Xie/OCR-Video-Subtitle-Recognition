"""Local OCR for subtitle text burned into video frames."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Protocol, Sequence

from .errors import DependencyError, MediaProcessingError, OCRProcessingError
from .subtitles import segments_to_srt
from .throughput import OCRBatchController, log_batch_observation

_SAMPLE_INTERVAL = 0.5
_LOWER_FRAME_RATIO = 0.55
_PADDLE_SUBTITLE_CROP_RATIO = 0.75
_PADDLE_MAX_NEW_TOKENS = 128
_PADDLE_MAX_PIXELS = 131072
# Keep CPU and GPU VLM requests at a comparable input resolution. The CPU
# llama.cpp OCR prompt does not use PaddleX's spotting-only 2x image upscale.
_PADDLE_CPU_MAX_PIXELS = _PADDLE_MAX_PIXELS
_PADDLE_TENSOR_COPY_WARNING = (
    r"To copy construct from a tensor, it is recommended to use "
    r"sourceTensor\.clone\(\)\.detach\(\), rather than "
    r"paddle\.to_tensor\(sourceTensor\)\."
)
_MERGE_SIMILARITY = 0.90
_MAX_MERGE_GAP = 1.0
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class OCRText:
    text: str
    box: tuple[tuple[float, float], ...] | None
    confidence: float | None


class CPUFrameSampler:
    """Reduce repeated CPU VLM calls while confirming newly visible text."""

    def __init__(
        self,
        confirmation_interval: float = 0.2,
        active_interval: float = 1.0,
        fallback_interval: float = 0.5,
    ) -> None:
        self.confirmation_interval = confirmation_interval
        self.active_interval = active_interval
        self.fallback_interval = fallback_interval
        self.last_processed: float | None = None
        self.candidate_started: float | None = None
        self.candidate_confirmed = False
        self.previous_candidate = False
        self.last_signature: tuple[tuple[int, int, int], ...] | None = None

    def should_process(
        self,
        timestamp: float,
        has_text: bool,
        signature: tuple[tuple[int, int, int], ...] | None = None,
        use_fallback: bool = True,
    ) -> bool:
        if has_text:
            if not self.previous_candidate:
                self.candidate_started = timestamp
                self.candidate_confirmed = False
                self.previous_candidate = True
                return self._selected(timestamp, signature)
            if (
                signature is not None
                and self.last_signature is not None
                and _caption_layout_changed(self.last_signature, signature)
                and self.last_processed is not None
                and timestamp - self.last_processed >= self.confirmation_interval
            ):
                self.candidate_confirmed = True
                return self._selected(timestamp, signature)
            if (
                not self.candidate_confirmed
                and self.candidate_started is not None
                and timestamp - self.candidate_started >= self.confirmation_interval
            ):
                self.candidate_confirmed = True
                return self._selected(timestamp, signature)
            if self.last_processed is None or timestamp - self.last_processed >= self.active_interval:
                self.candidate_confirmed = True
                return self._selected(timestamp, signature)
            return False

        self.previous_candidate = False
        self.candidate_started = None
        self.candidate_confirmed = False
        self.last_signature = None
        if use_fallback and (
            self.last_processed is None
            or timestamp - self.last_processed >= self.fallback_interval
        ):
            return self._selected(timestamp, None)
        return False

    def _selected(
        self,
        timestamp: float,
        signature: tuple[tuple[int, int, int], ...] | None,
    ) -> bool:
        self.last_processed = timestamp
        self.last_signature = signature
        return True


class FrameOCR(Protocol):
    engine_name: str
    device_name: str

    def recognize_batch(self, images: Sequence[Any]) -> list[list[OCRText]]: ...

    def close(self) -> None: ...


class PaddleOCRVLEngine:
    engine_name = "PaddleOCR-VL"

    def __init__(
        self,
        cpu_threads: int,
        batch_size: int,
        backend: str = "vllm",
        server_url: str | None = None,
        concurrency: int = 1,
    ) -> None:
        self.backend = backend
        self.device_name = f"{backend} service"
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise DependencyError("PaddleOCR-VL is missing. Run the platform setup script to install the VL runtime.") from exc
        kwargs: dict[str, Any] = {
            "device": "cpu",
            # Subtitle crops need OCR only; document layout detection adds an
            # unnecessary model pass and can split one caption into many calls.
            "use_layout_detection": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_queues": True,
        }
        kwargs["cpu_threads"] = max(1, cpu_threads)
        paddle_backend = {"vllm": "vllm-server", "llama-cpp": "llama-cpp-server"}[backend]
        kwargs.update(vl_rec_backend=paddle_backend, vl_rec_max_concurrency=max(1, concurrency))
        if server_url:
            kwargs["vl_rec_server_url"] = server_url
        try:
            self._engine = PaddleOCRVL(**kwargs)
        except Exception as exc:
            raise OCRProcessingError(f"Could not initialize PaddleOCR-VL: {exc}") from exc
        self._candidate_detector = None
        self._frame_sampler = CPUFrameSampler()
        self.frame_scan_interval = 0.5
        if backend == "llama-cpp":
            self.frame_scan_interval = 0.2
            try:
                from paddleocr import TextDetection

                self._candidate_detector = TextDetection(
                    model_name="PP-OCRv5_mobile_det",
                    device="cpu",
                    limit_side_len=640,
                    limit_type="max",
                    thresh=0.3,
                    box_thresh=0.5,
                    enable_mkldnn=False,
                )
            except Exception as exc:
                self.frame_scan_interval = 0.5
                print(
                    f"[WARN] Lightweight CPU text detection is unavailable; using the safe 0.5s OCR cadence: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        self.batch_size = max(1, batch_size)

    def should_process_frame(self, timestamp: float, image: Any) -> bool:
        detector = self._candidate_detector
        if detector is None:
            return self._frame_sampler.should_process(
                timestamp, has_text=False, use_fallback=True
            )
        try:
            crop = _subtitle_crop(image)
            result = next(detector.predict_iter([crop]))
            value = getattr(result, "json", None)
            if callable(value):
                value = value()
            payload = value.get("res", value) if isinstance(value, dict) else {}
            polygons = payload.get("dt_polys", []) if isinstance(payload, dict) else []
            signature = _caption_layout_signature(polygons, crop.shape[1], crop.shape[0])
            has_text = bool(signature)
        except Exception as exc:
            print(
                f"[WARN] Lightweight CPU text detection failed; retaining the frame for OCR: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self._candidate_detector = None
            has_text = False
            signature = None
            use_fallback = True
        else:
            use_fallback = False
        return self._frame_sampler.should_process(
            timestamp, has_text, signature, use_fallback=use_fallback
        )

    def recognize_batch(self, images: Sequence[Any]) -> list[list[OCRText]]:
        try:
            crops = [_subtitle_crop(image) for image in images]
            prepared_crops = crops
            crop_scales: list[tuple[float, float]] | None = None
            if self.backend == "llama-cpp":
                prepared_crops, crop_scales = _resize_images_to_pixel_limit(
                    crops, _PADDLE_CPU_MAX_PIXELS
                )
            output = self._predict(prepared_crops)
            normalized = [
                _paddle_result(result, allow_full_crop=self.backend == "llama-cpp")
                for result in output
            ]
            for index, entries in enumerate(normalized):
                if entries:
                    if crop_scales is not None:
                        entries = _scale_entries(entries, *crop_scales[index])
                    normalized[index] = _offset_entries(
                        entries, int(images[index].shape[0] * _PADDLE_SUBTITLE_CROP_RATIO)
                    )
            return normalized
        except Exception as exc:
            if isinstance(exc, OCRProcessingError):
                raise
            raise OCRProcessingError(f"PaddleOCR-VL failed while processing video frames: {exc}") from exc

    def _predict(self, images: Sequence[Any]) -> list[Any]:
        output: list[Any] = []
        # PaddleOCR-VL exposes predict_iter for streamed, memory-bounded inference.
        with warnings.catch_warnings():
            # PaddleX 3.7 wraps an existing Paddle tensor with paddle.to_tensor;
            # its copy-construction warning is harmless and originates upstream.
            warnings.filterwarnings(
                "ignore",
                message=_PADDLE_TENSOR_COPY_WARNING,
                category=UserWarning,
                module=r"paddle\.tensor\.creation",
            )
            if self.backend == "llama-cpp":
                warnings.filterwarnings(
                    "ignore",
                    message=r"'llama-cpp-server' does not support `(?:min|max)_pixels`\.",
                    category=UserWarning,
                    module=r"paddlex\.inference\.models\.doc_vlm\.predictor",
                )
            prediction_options = {
                # Spotting returns text with localized polygons. Plain OCR returns
                # one full-crop box, which cannot distinguish subtitles from
                # hallucinated/background text.
                "prompt_label": "ocr" if self.backend == "llama-cpp" else "spotting",
                "max_new_tokens": _PADDLE_MAX_NEW_TOKENS,
                "use_queues": True,
            }
            if self.backend == "vllm":
                prediction_options["max_pixels"] = _PADDLE_MAX_PIXELS
            iterator = self._engine.predict_iter(list(images), **prediction_options)
            for result in iterator:
                output.append(result)
        if len(output) != len(images):
            raise OCRProcessingError("PaddleOCR-VL returned an unexpected number of frame results.")
        return output

    def close(self) -> None:
        self._engine = None


def create_ocr_runtime(
    resources=None,
    on_model_load=None,
    server_url: str | None = None,
    concurrency: int | None = None,
    backend: str = "auto",
):
    """Connect the PaddleOCR client to the hardware-appropriate local inference service."""
    from .resources import detect_system_resources, plan_ocr_resources

    system = resources or detect_system_resources()
    if backend == "auto":
        backend = "vllm" if _nvidia_gpu_available() else "llama-cpp"
    if backend not in {"vllm", "llama-cpp"}:
        raise OCRProcessingError(f"Unsupported OCR service backend: {backend}.")
    if concurrency is not None and concurrency < 1:
        raise OCRProcessingError("OCR service concurrency must be a positive integer.")
    if server_url is None:
        port = 8118 if backend == "vllm" else 8111
        server_url = f"http://127.0.0.1:{port}/v1"
    plan = plan_ocr_resources(system, concurrency, backend)
    if on_model_load is not None:
        on_model_load()
    import importlib.util

    if importlib.util.find_spec("paddleocr") is None:
        raise DependencyError(
            "PaddleOCR is missing from the client environment. Run ./scripts/configure --setup."
        )
    try:
        vl = PaddleOCRVLEngine(
            plan.cpu_threads, plan.batch_size, backend=backend,
            server_url=server_url, concurrency=plan.batch_size
        )
        # In-memory warm-up confirms the local pipeline and configured service cooperate.
        import numpy as np

        vl.recognize_batch([np.zeros((160, 640, 3), dtype=np.uint8)])
    except Exception as exc:
        try:
            vl.close()
        except (UnboundLocalError, AttributeError):
            pass
        hint = (
            f"Start it with ./scripts/{'vlm-server.sh' if backend == 'vllm' else 'llama-server.sh'} start."
            if server_url.startswith("http://127.0.0.1")
            else "Check the configured service URL."
        )
        raise OCRProcessingError(
            f"Could not initialize the PaddleOCR-VL client or reach the {backend} service at {server_url}. {hint} {exc}"
        ) from exc
    return vl, plan


def _nvidia_gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def subtitle_from_frames(
    frames: Iterable[tuple[float, Any]],
    engine: FrameOCR,
    duration: float | None = None,
    on_progress: Callable[[float, float | None], None] | None = None,
    batch_size: int = 1,
) -> str | None:
    """Recognize caption-like text in bounded batches and convert stable detections into SRT."""
    detections: list[tuple[float, str]] = []
    next_progress = 1.0
    if on_progress is not None:
        on_progress(0.0, duration)
    controller = OCRBatchController(maximum_batch_size=max(1, batch_size))
    size = controller.batch_size
    batch: list[tuple[float, Any]] = []

    def consume(items: list[tuple[float, Any]], adapt: bool = True) -> None:
        if not items:
            return
        started = time.monotonic()
        recovered_from_memory = False

        def recognize(frame_items: list[tuple[float, Any]]) -> list[list[OCRText]]:
            nonlocal recovered_from_memory
            try:
                return engine.recognize_batch([image for _, image in frame_items])
            except OCRProcessingError as exc:
                if len(frame_items) <= 1 or not _is_memory_pressure_error(exc):
                    raise
                reduced = controller.back_off_after_memory_error(len(frame_items))
                recovered_from_memory = True
                print(
                    f"[WARN] OCR batch hit a memory limit; retrying at concurrency {reduced}.",
                    file=sys.stderr,
                    flush=True,
                )
                recovered: list[list[OCRText]] = []
                for start in range(0, len(frame_items), reduced):
                    recovered.extend(recognize(frame_items[start : start + reduced]))
                return recovered

        results = recognize(items)
        elapsed = time.monotonic() - started
        if len(results) != len(items):
            raise OCRProcessingError("OCR returned an unexpected number of frame results.")
        if adapt and not recovered_from_memory and len(items) == size and controller.maximum_batch_size > 1:
            observation = controller.observe(len(items), elapsed)
            log_batch_observation(
                observation,
                logger=lambda message: print(message, file=sys.stderr, flush=True),
            )
        for (timestamp, image), entries in zip(items, results):
            caption = _caption_from_entries(entries, image)
            if caption:
                detections.append((float(timestamp), caption))

    try:
        frame_selector = getattr(engine, "should_process_frame", None)
        for timestamp, image in frames:
            if on_progress is not None and timestamp >= next_progress:
                on_progress(float(timestamp), duration)
                next_progress = float(timestamp) + 1.0
            if frame_selector is not None and not frame_selector(timestamp, image):
                continue
            batch.append((timestamp, image))
            if len(batch) >= size:
                consume(batch)
                batch = []
                size = controller.batch_size
        consume(batch)
    except OCRProcessingError:
        raise
    except MediaProcessingError:
        raise
    except Exception as exc:
        raise OCRProcessingError(f"OCR failed while processing video frames: {exc}") from exc
    finally:
        close_frames = getattr(frames, "close", None)
        if close_frames is not None:
            close_frames()

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


def _is_memory_pressure_error(error: Exception) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in ("out of memory", "cuda oom", "memoryerror", "kv cache is full", "failed to allocate")
    )


def _paddle_result(result: Any, allow_full_crop: bool = False) -> list[OCRText]:
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if value is None and isinstance(result, dict):
        value = result
    if not isinstance(value, dict):
        return []
    # Spotting mode returns localized polygons, while the llama.cpp route's
    # ordinary OCR prompt currently returns text blocks over the full crop.
    if isinstance(value.get("res"), dict):
        value = value["res"]
    spotting = value.get("spotting_res") or {}
    if not isinstance(spotting, dict):
        spotting = {}
    texts = spotting.get("rec_texts") or []
    polygons = spotting.get("rec_polys") or []
    if not isinstance(texts, (list, tuple)) or not isinstance(polygons, (list, tuple)):
        return []
    output: list[OCRText] = []
    for text, polygon in zip(texts, polygons):
        if not isinstance(text, str) or not text.strip():
            continue
        box = _normalize_box(polygon)
        if box is not None:
            output.append(OCRText(text, box, None))
    if output or not allow_full_crop:
        return output
    for block in value.get("parsing_res_list", []) or []:
        if not isinstance(block, dict):
            continue
        text = block.get("block_content")
        if not isinstance(text, str) or not text.strip():
            continue
        box = _normalize_box(block.get("block_bbox"))
        if box is not None:
            output.append(OCRText(text, box, None))
    return output


def _has_caption_region_text(polygons: Any, width: int, height: int) -> bool:
    return bool(_caption_layout_signature(polygons, width, height))


def _caption_layout_signature(
    polygons: Any, width: int, height: int
) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(polygons, (list, tuple)):
        return ()
    boxes: list[tuple[float, float, float]] = []
    for polygon in polygons:
        try:
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
        except (TypeError, ValueError, IndexError):
            continue
        if not xs or not ys:
            continue
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)
        center_y = (min(ys) + max(ys)) / 2
        if (
            center_y >= height * 0.2
            and center_y <= height * 0.96
            and box_width >= width * 0.01
            and box_height >= height * 0.015
            and box_height <= max(box_width / 0.55, height * 0.08)
        ):
            boxes.append((min(xs) / max(1, width), max(xs) / max(1, width), center_y / max(1, height)))
    if not boxes:
        return ()

    boxes.sort(key=lambda box: box[2])
    lines: list[list[tuple[float, float, float]]] = []
    for box in boxes:
        matching = next(
            (line for line in lines if abs(box[2] - sum(item[2] for item in line) / len(line)) <= 0.10),
            None,
        )
        if matching is None:
            lines.append([box])
        else:
            matching.append(box)
    return tuple(
        (
            round(min(box[0] for box in line) * 20),
            round(max(box[1] for box in line) * 20),
            round(sum(box[2] for box in line) / len(line) * 20),
        )
        for line in lines
    )


def _caption_layout_changed(
    previous: tuple[tuple[int, int, int], ...],
    current: tuple[tuple[int, int, int], ...],
) -> bool:
    if len(previous) != len(current):
        return True
    return any(
        abs(left[0] - right[0]) >= 1
        or abs(left[1] - right[1]) >= 1
        or abs(left[2] - right[2]) >= 1
        for left, right in zip(previous, current)
    )


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
    top = int(height * _PADDLE_SUBTITLE_CROP_RATIO)
    return image[top:, :]


def _resize_images_to_pixel_limit(
    images: Sequence[Any], max_pixels: int
) -> tuple[list[Any], list[tuple[float, float]]]:
    import cv2

    resized_images = []
    scales = []
    for image in images:
        height, width = image.shape[:2]
        scale = min(1.0, (max_pixels / max(1, width * height)) ** 0.5)
        if scale < 1.0:
            resized_width = max(1, int(width * scale))
            resized_height = max(1, int(height * scale))
            resized = cv2.resize(
                image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
            )
        else:
            resized_width, resized_height = width, height
            resized = image
        resized_images.append(resized)
        scales.append((width / resized_width, height / resized_height))
    return resized_images, scales


def _scale_entries(
    entries: list[OCRText], scale_x: float, scale_y: float
) -> list[OCRText]:
    return [
        OCRText(
            entry.text,
            tuple((x * scale_x, y * scale_y) for x, y in entry.box)
            if entry.box is not None
            else None,
            entry.confidence,
        )
        for entry in entries
    ]


def _offset_entries(entries: list[OCRText], vertical_offset: int) -> list[OCRText]:
    return [
        OCRText(
            entry.text,
            tuple((x, y + vertical_offset) for x, y in entry.box) if entry.box is not None else None,
            entry.confidence,
        )
        for entry in entries
    ]


def _caption_from_entries(entries: list[OCRText], image: Any) -> str | None:
    height = int(image.shape[0])
    lines: list[tuple[float, float, str]] = []
    for entry in entries:
        cleaned = _clean_text(entry.text)
        if not cleaned:
            continue
        if entry.box is None:
            continue
        try:
            top = min(point[1] for point in entry.box)
            bottom = max(point[1] for point in entry.box)
            left = min(point[0] for point in entry.box)
        except (TypeError, ValueError, IndexError):
            continue
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
