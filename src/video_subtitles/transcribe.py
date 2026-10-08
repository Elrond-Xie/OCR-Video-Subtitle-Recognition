"""Local speech recognition backed by faster-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .errors import DependencyError, TranscriptionError
from .subtitles import segments_to_srt

_MODELS = {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}


def transcribe_file(audio_path: Path, model_size: str, on_model_load: Callable[[], None] | None = None) -> str:
    if model_size not in _MODELS:
        raise TranscriptionError(
            f"Unsupported model '{model_size}'. Choose one of: {', '.join(sorted(_MODELS))}."
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise DependencyError("faster-whisper is missing. Install the project dependencies with `pip install -e .`.") from exc

    if on_model_load is not None:
        on_model_load()
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(audio_path), language=None)
        return segments_to_srt((segment.start, segment.end, segment.text) for segment in segments)
    except Exception as exc:
        raise TranscriptionError(
            f"Local transcription failed. Check the audio and available memory; first use also downloads the model: {exc}"
        ) from exc
