"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .errors import SubtitleCliError
from .pipeline import create_subtitles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitles",
        description="Extract existing subtitles, read on-screen captions with OCR, or transcribe audio locally.",
    )
    parser.add_argument("input", help="local media path or supported online video URL")
    parser.add_argument("-o", "--output", type=Path, help="output SRT path (default: same base name)")
    parser.add_argument("--cookies", type=Path, help="optional yt-dlp cookies file for authenticated pages")
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="extract captions burned into the video image before falling back to local ASR",
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"),
        help="local multilingual ASR model used when no usable subtitle or OCR captions exist (default: small)",
    )
    parser.add_argument("--force", action="store_true", help="replace the output file if it already exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    progress_started = False
    progress_started_at = 0.0

    def close_progress() -> None:
        nonlocal progress_started
        if progress_started:
            print(file=sys.stderr, flush=True)
            progress_started = False

    def format_time(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, seconds_part = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}" if hours else f"{minutes:02d}:{seconds_part:02d}"

    def announce_model_load() -> None:
        close_progress()
        print(
            f"No usable subtitle or OCR captions found. Loading local ASR model '{args.model}' (first use may download it)...",
            file=sys.stderr,
        )

    def announce_ocr_load() -> None:
        close_progress()
        print(
            "Loading local OCR model (first use may download model files)...",
            file=sys.stderr,
        )

    def announce_ocr_route(engine: str, device: str, plan) -> None:
        print(
            f"OCR runtime: {engine} on {device}; {plan.cpu_threads} CPU threads, "
            f"frame queue {plan.frame_queue_size}, batch {plan.batch_size}.",
            file=sys.stderr,
        )

    def announce_ocr_fallback(reason: str) -> None:
        summary = " ".join(str(reason).split())[:240]
        print(f"OCR fallback: PaddleOCR-VL unavailable; using RapidOCR/CPU ({summary})", file=sys.stderr)

    def announce_ocr_progress(video_seconds: float, total_seconds: float | None) -> None:
        nonlocal progress_started, progress_started_at
        if not progress_started:
            progress_started = True
            progress_started_at = time.monotonic()
        elapsed = format_time(time.monotonic() - progress_started_at)
        if total_seconds is not None and total_seconds > 0:
            fraction = min(1.0, max(0.0, video_seconds / total_seconds))
            percent = int(fraction * 100)
            width = 24
            filled = int(fraction * width)
            bar = "#" * filled + "." * (width - filled)
            message = (
                f"\rOCR [{bar}] {percent:3d}% "
                f"{format_time(video_seconds)}/{format_time(total_seconds)} elapsed {elapsed}"
            )
        else:
            message = f"\rOCR processed {format_time(video_seconds)} elapsed {elapsed} (duration unknown)"
        print(message, end="", file=sys.stderr, flush=True)

    try:
        result = create_subtitles(
            input_value=args.input,
            output=args.output,
            cookies=args.cookies,
            model_size=args.model,
            force=args.force,
            use_ocr=args.ocr,
            on_model_load=announce_model_load,
            on_ocr_load=announce_ocr_load,
            on_ocr_progress=announce_ocr_progress,
            on_ocr_route=announce_ocr_route,
            on_ocr_fallback=announce_ocr_fallback,
        )
    except SubtitleCliError as exc:
        close_progress()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        close_progress()
        print("Interrupted.", file=sys.stderr)
        return 130

    close_progress()
    print(f"Subtitle saved to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
