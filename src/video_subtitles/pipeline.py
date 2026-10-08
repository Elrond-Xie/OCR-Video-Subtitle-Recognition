"""Coordinate subtitle-first extraction and local ASR fallback."""

from __future__ import annotations

import re
import tempfile
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlparse

from .errors import MediaProcessingError, OnlineExtractionError, SubtitleCliError
from .media import extract_audio, extract_subtitle, iter_video_frames, video_duration
from .ocr import create_ocr_runtime, subtitle_from_frames
from .online import available_subtitle, download_audio, extract_online_ocr
from .subtitles import subtitle_file_to_srt
from .transcribe import transcribe_file


def _is_url(value: str) -> bool:
    return urlparse(value).scheme.lower() in {"http", "https"}


def _safe_title(value: str) -> str:
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return (title[:120] or "video")


def _default_output(input_value: str, title: str | None) -> Path:
    output_dir = Path.cwd() / "output"
    if _is_url(input_value):
        filename = _safe_title(title or "video")
    else:
        filename = _safe_title(Path(input_value).expanduser().stem)
    return output_dir / f"{filename}.srt"


def _write_output(output: Path, content: str, force: bool) -> Path:
    output = output.expanduser()
    if output.suffix.lower() != ".srt":
        raise SubtitleCliError("The output filename must use the .srt extension.")
    if not output.parent.exists():
        raise SubtitleCliError(f"Output directory does not exist: {output.parent}")
    if output.exists() and not force:
        raise SubtitleCliError(f"Output already exists: {output}. Pass --force to replace it.")
    try:
        output.write_text(content, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise SubtitleCliError(f"Could not write subtitle file: {output}") from exc
    return output


def create_subtitles(
    input_value: str,
    output: Path | None = None,
    cookies: Path | None = None,
    model_size: str = "small",
    force: bool = False,
    use_ocr: bool = False,
    ocr_server_url: str | None = None,
    ocr_concurrency: int | None = None,
    ocr_backend: str = "auto",
    on_model_load=None,
    on_ocr_load=None,
    on_ocr_progress=None,
    on_ocr_route=None,
    on_long_operation=None,
) -> Path:
    is_online = _is_url(input_value)
    local_path = Path(input_value).expanduser().resolve() if not is_online else None
    if local_path is not None and not local_path.is_file():
        raise MediaProcessingError(f"Input file does not exist: {local_path}")
    if cookies is not None:
        cookies = cookies.expanduser().resolve()
        if not cookies.is_file():
            raise SubtitleCliError(f"Cookie file does not exist: {cookies}")

    title: str | None = local_path.stem if local_path is not None else None
    if output is not None:
        output = output.expanduser()
        if output.suffix.lower() != ".srt":
            raise SubtitleCliError("The output filename must use the .srt extension.")
        if not output.parent.exists():
            raise SubtitleCliError(f"Output directory does not exist: {output.parent}")
        if output.exists() and not force:
            raise SubtitleCliError(f"Output already exists: {output}. Pass --force to replace it.")
    elif local_path is not None:
        local_target = _default_output(input_value, title)
        if local_target.exists() and not force:
            raise SubtitleCliError(f"Output already exists: {local_target}. Pass --force to replace it.")

    workdir_parent = Path.cwd() / "output"
    workdir_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".video-subtitles-", dir=workdir_parent
    ) as temporary:
        workdir = Path(temporary)
        content: str | None = None
        audio: Path | None = None

        if use_ocr:
            wait = on_long_operation("Loading OCR models") if on_long_operation else nullcontext()
            with wait:
                engine, resource_plan = create_ocr_runtime(
                    on_model_load=on_ocr_load,
                    server_url=ocr_server_url,
                    concurrency=ocr_concurrency,
                    backend=ocr_backend,
                )
            if on_ocr_route is not None:
                on_ocr_route(engine.engine_name, engine.device_name, resource_plan)
            try:
                if is_online:
                    title, content = extract_online_ocr(
                        input_value,
                        workdir,
                        cookies,
                        on_model_load=on_ocr_load,
                        on_progress=on_ocr_progress,
                        engine=engine,
                        batch_size=resource_plan.batch_size,
                        prefetch=resource_plan.frame_queue_size,
                        on_long_operation=on_long_operation,
                    )
                else:
                    assert local_path is not None
                    try:
                        content = subtitle_from_frames(
                            iter_video_frames(
                                local_path,
                                sample_interval=getattr(engine, "frame_scan_interval", 0.5),
                                prefetch=resource_plan.frame_queue_size,
                            ),
                            engine,
                            duration=video_duration(local_path),
                            on_progress=on_ocr_progress,
                            batch_size=resource_plan.batch_size,
                        )
                    except MediaProcessingError:
                        content = None
            finally:
                engine.close()
            if is_online and content is None:
                title, audio = download_audio(input_value, workdir, cookies, on_long_operation=on_long_operation)
            elif not is_online and content is None:
                assert local_path is not None
                audio = extract_audio(local_path, workdir)
        elif is_online:
            found = available_subtitle(input_value, workdir, cookies, on_long_operation=on_long_operation)
            if found is not None:
                title, content = found
            else:
                title, audio = download_audio(input_value, workdir, cookies, on_long_operation=on_long_operation)
        else:
            assert local_path is not None
            subtitle = extract_subtitle(local_path, workdir)
            if subtitle is not None:
                try:
                    content = subtitle_file_to_srt(subtitle)
                except SubtitleCliError:
                    content = None
            if content is None:
                audio = extract_audio(local_path, workdir)

        if content is None:
            assert audio is not None
            wait = on_long_operation("Loading ASR model and transcribing audio") if on_long_operation else nullcontext()
            with wait:
                content = transcribe_file(audio, model_size, on_model_load=on_model_load)

        target = output.expanduser() if output is not None else _default_output(input_value, title)
        return _write_output(target, content, force)
