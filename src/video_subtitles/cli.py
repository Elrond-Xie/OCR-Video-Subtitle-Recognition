"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .console import ProgressDisplay, long_operation, start_runtime_logging
from .downloads import configure_download_locations
from .errors import SubtitleCliError
from .pipeline import create_subtitles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitles",
        description="Extract existing subtitles, read on-screen captions with OCR, or transcribe audio locally.",
    )
    parser.add_argument("input", nargs="?", help="local media path or supported online video URL")
    parser.add_argument(
        "--input",
        dest="input_option",
        help="local media path or supported online video URL (named form)",
    )
    parser.add_argument("-o", "--output", type=Path, help="output SRT path (default: same base name)")
    parser.add_argument("--cookies", type=Path, help="optional yt-dlp cookies file for authenticated pages")
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="extract captions burned into the video image before falling back to local ASR",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=("auto", "vllm", "llama-cpp"),
        default="auto",
        help="OCR service backend (auto selects vLLM with NVIDIA GPU, llama.cpp on CPU)",
    )
    parser.add_argument(
        "--ocr-server-url",
        help="OpenAI-compatible OCR service URL (default: local backend port)",
    )
    parser.add_argument(
        "--ocr-concurrency",
        type=int,
        help="cap adaptive OCR concurrency at this request count (positive integer)",
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
    if args.input is not None and args.input_option is not None:
        parser.error("provide the input once, either positionally or with --input")
    input_value = args.input_option or args.input
    if not input_value:
        parser.error("an input path or URL is required; pass it as --input PATH_OR_URL")
    start_runtime_logging()
    configure_download_locations()
    progress = ProgressDisplay()

    def announce_model_load() -> None:
        print(f"No usable subtitle or OCR captions found; preparing ASR model '{args.model}'.", file=sys.stderr)

    def announce_ocr_load() -> None:
        print("Connecting to the PaddleOCR-VL inference service.", file=sys.stderr)

    def announce_ocr_route(engine: str, device: str, plan) -> None:
        print(
            f"OCR runtime: {engine} via {device}; {plan.cpu_threads} CPU preprocessing threads, "
            f"frame queue {plan.frame_queue_size}, adaptive batch ceiling {plan.batch_size}.",
            file=sys.stderr,
        )

    def announce_ocr_progress(video_seconds: float, total_seconds: float | None) -> None:
        progress.update("OCR", video_seconds, total_seconds)

    try:
        result = create_subtitles(
            input_value=input_value,
            output=args.output,
            cookies=args.cookies,
            model_size=args.model,
            force=args.force,
            use_ocr=args.ocr,
            ocr_server_url=args.ocr_server_url,
            ocr_concurrency=args.ocr_concurrency,
            ocr_backend=args.ocr_backend,
            on_model_load=announce_model_load,
            on_ocr_load=announce_ocr_load,
            on_ocr_progress=announce_ocr_progress,
            on_ocr_route=announce_ocr_route,
            on_long_operation=long_operation,
        )
    except SubtitleCliError as exc:
        progress.close()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        progress.close()
        print("Interrupted.", file=sys.stderr)
        return 130

    progress.close()
    print(f"Subtitle saved to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
