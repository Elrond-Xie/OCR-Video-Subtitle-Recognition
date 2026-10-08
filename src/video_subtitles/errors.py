"""Errors intended to be shown directly to CLI users."""


class SubtitleCliError(Exception):
    """Base class for expected processing errors."""


class DependencyError(SubtitleCliError):
    """A required external program or Python package is unavailable."""


class MediaProcessingError(SubtitleCliError):
    """A local media file could not be inspected or converted."""


class OnlineExtractionError(SubtitleCliError):
    """An online video could not be inspected or downloaded."""


class TranscriptionError(SubtitleCliError):
    """Local speech recognition failed."""


class OCRProcessingError(SubtitleCliError):
    """Local OCR model or frame recognition failed."""
