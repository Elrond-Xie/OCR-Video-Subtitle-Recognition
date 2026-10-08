"""yt-dlp integration for supported online video pages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .errors import DependencyError, MediaProcessingError, OnlineExtractionError
from .media import iter_video_frames
from .ocr import FrameOCR, create_ocr_engine, subtitle_from_frames
from .subtitles import subtitle_file_to_srt

_SUBTITLE_SUFFIXES = {".srt", ".vtt", ".json", ".json3", ".bcc", ".ttml", ".dfxp", ".xml"}


def _youtube_dl():
    try:
        import yt_dlp
    except ImportError as exc:
        raise DependencyError("yt-dlp is missing. Install the project dependencies with `pip install -e .`.") from exc
    return yt_dlp


def _options(cookies: Path | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {"quiet": True, "no_warnings": True, "noplaylist": True}
    if cookies is not None:
        options["cookiefile"] = str(cookies)
    return options


def _extract_info(url: str, cookies: Path | None) -> dict[str, Any]:
    yt_dlp = _youtube_dl()
    try:
        with yt_dlp.YoutubeDL(_options(cookies)) as downloader:
            info = downloader.extract_info(url, download=False)
    except Exception as exc:
        raise OnlineExtractionError(
            f"Could not read this URL. Check that the site is supported and the video is accessible: {exc}"
        ) from exc
    if not isinstance(info, dict):
        raise OnlineExtractionError("The URL did not return a video entry.")
    return info


def _subtitle_choice(info: dict[str, Any]) -> tuple[str, bool] | None:
    preferred = str(info.get("language") or "").lower()
    for key, automatic in (("subtitles", False), ("automatic_captions", True)):
        tracks = info.get(key) or {}
        candidates = [
            language
            for language, formats in tracks.items()
            if language.lower() != "danmaku" and formats
        ]
        if candidates:
            if preferred:
                matching = [
                    language
                    for language in candidates
                    if language.lower() == preferred
                    or language.lower().split("-")[0] == preferred.split("-")[0]
                ]
                if matching:
                    return matching[0], automatic
            return candidates[0], automatic
    return None


def available_subtitle(url: str, workdir: Path, cookies: Path | None) -> tuple[str, str] | None:
    """Return (title, normalized SRT) when a non-danmaku subtitle track exists."""
    info = _extract_info(url, cookies)
    choice = _subtitle_choice(info)
    if choice is None:
        # Some extractors (including Bilibili) expose subtitles only during the
        # download pass, even though `--list-subs` can discover them.
        language = "all"
        automatic: bool | None = None
    else:
        language, automatic = choice
    yt_dlp = _youtube_dl()
    options = _options(cookies)
    options.update(
        {
            "skip_download": True,
            "writesubtitles": automatic is not True,
            "writeautomaticsub": automatic is not False,
            "subtitleslangs": [language],
            "subtitlesformat": "srt/best",
            "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        }
    )
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([url])
    except Exception:
        # The advertised track may disappear or fail to download; the pipeline can still try ASR.
        return None

    files = sorted(
        (
            path
            for path in workdir.iterdir()
            if path.is_file()
            and path.suffix.lower() in _SUBTITLE_SUFFIXES
            and ".danmaku." not in path.name.lower()
        ),
        key=lambda path: (path.suffix.lower() != ".srt", path.name),
    )
    for path in files:
        try:
            return str(info.get("title") or info.get("id") or "video"), subtitle_file_to_srt(path)
        except Exception:
            continue
    # Some extractors provide subtitle content directly and yt-dlp can report it without a file.
    track_map = info.get("automatic_captions" if automatic else "subtitles") or {}
    for entry in track_map.get(language, []):
        if entry.get("data"):
            try:
                from .subtitles import subtitle_text_to_srt

                return (
                    str(info.get("title") or info.get("id") or "video"),
                    subtitle_text_to_srt(str(entry["data"]), str(entry.get("ext", ""))),
                )
            except Exception:
                continue
    return None


def download_audio(url: str, workdir: Path, cookies: Path | None) -> tuple[str, Path]:
    yt_dlp = _youtube_dl()
    info = _extract_info(url, cookies)
    options = _options(cookies)
    options.update(
        {
            "format": "bestaudio",
            "outtmpl": str(workdir / "online-audio.%(ext)s"),
        }
    )
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([url])
    except Exception as exc:
        raise OnlineExtractionError(f"Could not retrieve an audio stream for local transcription: {exc}") from exc
    files = sorted(
        path
        for path in workdir.iterdir()
        if path.is_file() and path.name.startswith("online-audio.") and not path.name.endswith(".part")
    )
    if not files:
        raise OnlineExtractionError("The site did not provide an audio-only stream that yt-dlp could download.")
    return str(info.get("title") or info.get("id") or "video"), files[0]


def extract_online_ocr(
    url: str,
    workdir: Path,
    cookies: Path | None,
    on_model_load=None,
    on_progress=None,
    engine: FrameOCR | None = None,
    batch_size: int = 1,
    prefetch: int = 0,
) -> tuple[str, str | None]:
    """Try decoding a yt-dlp stdout stream, then retry from a temporary video file."""
    try:
        info = _extract_info(url, cookies)
    except OnlineExtractionError as exc:
        raise OnlineExtractionError(
            f"Could not retrieve video frames for OCR. If the page plays in your browser, save the video locally and run subtitles on that file: {exc}"
        ) from exc
    title = str(info.get("title") or info.get("id") or "video")
    duration = info.get("duration")
    duration = float(duration) if isinstance(duration, (int, float)) and duration > 0 else None
    engine = engine or create_ocr_engine(on_model_load=on_model_load)

    try:
        process = subprocess.Popen(
            _stream_command(url, cookies),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        video_path = _download_video_for_ocr(url, workdir, cookies)
        return title, _ocr_downloaded_video(video_path, engine, duration, on_progress, batch_size, prefetch)
    stream_decoded = False
    stream_result: str | None = None
    try:
        if process.stdout is not None:
            stream_result = subtitle_from_frames(
                iter_video_frames(process.stdout, prefetch=prefetch),
                engine,
                duration=duration,
                on_progress=on_progress,
                batch_size=batch_size,
            )
            stream_decoded = process.wait() == 0
    except MediaProcessingError:
        stream_decoded = False
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()

    if stream_decoded:
        return title, stream_result

    video_path = _download_video_for_ocr(url, workdir, cookies)
    return title, _ocr_downloaded_video(video_path, engine, duration, on_progress, batch_size, prefetch)


def _ocr_downloaded_video(
    video_path: Path,
    engine: Any,
    duration: float | None = None,
    on_progress=None,
    batch_size: int = 1,
    prefetch: int = 0,
) -> str | None:
    try:
        return subtitle_from_frames(
            iter_video_frames(video_path, prefetch=prefetch),
            engine,
            duration=duration,
            on_progress=on_progress,
            batch_size=batch_size,
        )
    except MediaProcessingError as exc:
        raise OnlineExtractionError(
            f"Could not decode video frames for OCR. If the page plays in your browser, save the video locally and run subtitles on that file: {exc}"
        ) from exc


def _stream_command(url: str, cookies: Path | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--quiet",
        "--no-warnings",
        "--no-progress",
        "--no-playlist",
        "--format",
        "best",
        "--output",
        "-",
    ]
    if cookies is not None:
        command.extend(("--cookies", str(cookies)))
    command.append(url)
    return command


def _download_video_for_ocr(url: str, workdir: Path, cookies: Path | None) -> Path:
    yt_dlp = _youtube_dl()
    options = _options(cookies)
    options.update(
        {
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mkv",
            "outtmpl": str(workdir / "ocr-video.%(ext)s"),
        }
    )
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            result = downloader.download([url])
    except Exception as exc:
        raise OnlineExtractionError(
            "Could not retrieve video frames for OCR. If the page plays in your browser, save the video locally and run subtitles on that file. "
            f"yt-dlp reported: {exc}"
        ) from exc
    files = sorted(
        path
        for path in workdir.iterdir()
        if path.is_file()
        and path.name.startswith("ocr-video.")
        and not path.name.endswith(".part")
    )
    if result not in (None, 0) or not files:
        raise OnlineExtractionError(
            "Could not retrieve video frames for OCR. If the page plays in your browser, save the video locally and run subtitles on that file."
        )
    return files[0]
