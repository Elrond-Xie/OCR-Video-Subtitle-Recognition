"""FFmpeg helpers for local media files."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from .errors import DependencyError, MediaProcessingError

_IMAGE_SUBTITLE_CODECS = {"dvd_subtitle", "dvb_subtitle", "hdmv_pgs_subtitle", "xsub"}


def _require_program(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise DependencyError(
            f"{name} is required. Install FFmpeg (which includes {name}) and make sure it is on PATH."
        )
    return path


def _run(command: list[str], purpose: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        message = detail[-1] if detail else "unknown FFmpeg error"
        raise MediaProcessingError(f"Could not {purpose}: {message}") from exc


def subtitle_streams(path: Path) -> list[dict[str, object]]:
    ffprobe = _require_program("ffprobe")
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=index,codec_name:stream_tags=language,title",
            "-of",
            "json",
            str(path),
        ],
        f"inspect media file {path}",
    )
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise MediaProcessingError(f"FFprobe returned invalid media information for {path}.") from exc
    return [
        stream
        for stream in streams
        if stream.get("codec_name") not in _IMAGE_SUBTITLE_CODECS
    ]


def extract_subtitle(path: Path, workdir: Path) -> Path | None:
    ffmpeg = _require_program("ffmpeg")
    streams = subtitle_streams(path)
    for stream in streams:
        stream_index = stream.get("index")
        if stream_index is None:
            continue
        target = workdir / f"local-subtitle-{stream_index}.srt"
        try:
            _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-map",
                    f"0:{stream_index}",
                    "-c:s",
                    "srt",
                    str(target),
                ],
                "convert an embedded subtitle stream",
            )
        except MediaProcessingError:
            continue
        if target.exists() and target.stat().st_size:
            return target
    return None


def extract_audio(path: Path, workdir: Path) -> Path:
    ffmpeg = _require_program("ffmpeg")
    target = workdir / "audio.m4a"
    _run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(target),
        ],
        f"extract audio from {path}",
    )
    if not target.exists() or not target.stat().st_size:
        raise MediaProcessingError(f"No audio stream was found in {path}.")
    return target


def iter_video_frames(
    source: Path | BinaryIO,
    sample_interval: float = 0.5,
    prefetch: int = 0,
) -> Iterator[tuple[float, object]]:
    """Decode timestamped RGB frames at a fixed interval from a file or byte stream."""
    if prefetch > 0:
        yield from _prefetched_video_frames(source, sample_interval, prefetch)
        return
    yield from _decode_video_frames(source, sample_interval)


def _decode_video_frames(source: Path | BinaryIO, sample_interval: float) -> Iterator[tuple[float, object]]:
    try:
        import av
    except ImportError as exc:
        raise DependencyError("PyAV is missing. Reinstall the project dependencies with `pip install -e .`.") from exc

    container = None
    try:
        container = av.open(str(source) if isinstance(source, Path) else source, mode="r")
        video_stream = next(iter(container.streams.video), None)
        if video_stream is None:
            raise MediaProcessingError("The input does not contain a video stream to OCR.")
        next_sample = 0.0
        sampled = False
        for frame in container.decode(video_stream):
            timestamp = frame.time
            if timestamp is None and frame.pts is not None and frame.time_base is not None:
                timestamp = float(frame.pts * frame.time_base)
            if timestamp is None or timestamp + 0.001 < next_sample:
                continue
            yield float(timestamp), frame.to_ndarray(format="rgb24")
            sampled = True
            next_sample = float(timestamp) + sample_interval
        if not sampled:
            raise MediaProcessingError("No video frames could be decoded for OCR.")
    except DependencyError:
        raise
    except Exception as exc:
        raise MediaProcessingError(f"Could not decode video frames for OCR: {exc}") from exc
    finally:
        if container is not None:
            container.close()


def _prefetched_video_frames(
    source: Path | BinaryIO,
    sample_interval: float,
    prefetch: int,
) -> Iterator[tuple[float, object]]:
    items: queue.Queue[object] = queue.Queue(maxsize=max(1, prefetch))
    stopped = threading.Event()
    sentinel = object()

    def produce() -> None:
        try:
            for item in _decode_video_frames(source, sample_interval):
                while not stopped.is_set():
                    try:
                        items.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stopped.is_set():
                    return
        except BaseException as exc:
            while not stopped.is_set():
                try:
                    items.put(exc, timeout=0.1)
                    break
                except queue.Full:
                    continue
        finally:
            while not stopped.is_set():
                try:
                    items.put(sentinel, timeout=0.1)
                    break
                except queue.Full:
                    continue

    worker = threading.Thread(target=produce, name="video-frame-decoder", daemon=True)
    worker.start()
    try:
        while True:
            item = items.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stopped.set()
        worker.join(timeout=2)


def video_duration(path: Path) -> float | None:
    """Return container duration in seconds when available."""
    try:
        import av
    except ImportError as exc:
        raise DependencyError("PyAV is missing. Reinstall the project dependencies with `pip install -e .`.") from exc

    try:
        with av.open(str(path), mode="r") as container:
            if container.duration is not None:
                return max(0.0, float(container.duration) / av.time_base)
            stream = next(iter(container.streams.video), None)
            if stream is not None and stream.duration is not None and stream.time_base is not None:
                return max(0.0, float(stream.duration * stream.time_base))
    except Exception:
        return None
    return None
