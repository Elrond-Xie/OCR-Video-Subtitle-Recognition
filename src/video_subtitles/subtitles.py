"""Subtitle format normalization and SRT serialization."""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

from .errors import SubtitleCliError

_SRT_CUE = re.compile(
    r"(?m)^\s*\d+\s*\n\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+"
    r"\d{2}:\d{2}:\d{2}[,.]\d{3}"
)
_TIMESTAMP = re.compile(r"(?:(\d{2}):)?(\d{2}):(\d{2})[,.](\d{3})")


def _format_time(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    whole_seconds, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{millis:03}"


def segments_to_srt(segments: Iterable[tuple[float, float, str]]) -> str:
    cues: list[str] = []
    for index, (start, end, text) in enumerate(segments, start=1):
        content = " ".join(text.replace("\r", " ").split())
        if not content:
            continue
        cues.append(
            f"{len(cues) + 1}\n{_format_time(start)} --> {_format_time(end)}\n{content}"
        )
    if not cues:
        raise SubtitleCliError("No subtitle text was produced.")
    return "\n\n".join(cues) + "\n"


def _normalize_srt(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"(?m)(\d{2}:\d{2}:\d{2})\.(\d{3})", r"\1,\2", text)
    if not _SRT_CUE.search(text):
        raise SubtitleCliError("The subtitle file does not contain valid SRT cues.")
    return text + "\n"


def _parse_bilibili_json(text: str) -> str:
    data = json.loads(text)
    body = data.get("body", [])
    return segments_to_srt(
        (float(cue["from"]), float(cue["to"]), str(cue.get("content", "")))
        for cue in body
    )


def _parse_json3(text: str) -> str:
    data = json.loads(text)
    segments: list[tuple[float, float, str]] = []
    for event in data.get("events", []):
        start = float(event.get("tStartMs", 0)) / 1000
        end = start + float(event.get("dDurationMs", 0)) / 1000
        content = "".join(str(part.get("utf8", "")) for part in event.get("segs", []))
        if content.strip() and end > start:
            segments.append((start, end, content))
    return segments_to_srt(segments)


def _parse_vtt(text: str) -> str:
    lines = text.replace("\ufeff", "").replace("\r", "").splitlines()
    cues: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if "-->" not in line:
            index += 1
            continue
        start_raw, end_raw = (part.strip().split()[0] for part in line.split("-->", 1))
        start = _vtt_timestamp(start_raw)
        end = _vtt_timestamp(end_raw)
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index].strip():
            content.append(lines[index].strip())
            index += 1
        body = unescape(re.sub(r"<[^>]*>", "", " ".join(content))).strip()
        if body and end > start:
            cues.append(f"{len(cues) + 1}\n{_format_time(start)} --> {_format_time(end)}\n{body}")
        index += 1
    if not cues:
        raise SubtitleCliError("The WebVTT subtitle file contains no usable cues.")
    return "\n\n".join(cues) + "\n"


def _vtt_timestamp(value: str) -> float:
    match = _TIMESTAMP.fullmatch(value.replace(".", ","))
    if not match:
        raise SubtitleCliError(f"Unsupported WebVTT timestamp: {value}")
    hours, minutes, seconds, millis = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def _parse_ttml(text: str) -> str:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise SubtitleCliError("The TTML subtitle file could not be parsed.") from exc

    segments: list[tuple[float, float, str]] = []
    for cue in root.iter():
        if cue.tag.rsplit("}", 1)[-1] != "p":
            continue
        begin = cue.attrib.get("begin")
        end = cue.attrib.get("end")
        if not begin or not end:
            continue
        try:
            start_seconds = _ttml_time(begin)
            end_seconds = _ttml_time(end)
        except ValueError:
            continue
        body = " ".join("".join(cue.itertext()).split())
        if body and end_seconds > start_seconds:
            segments.append((start_seconds, end_seconds, body))
    return segments_to_srt(segments)


def _ttml_time(value: str) -> float:
    if value.endswith("s"):
        return float(value[:-1])
    match = _TIMESTAMP.fullmatch(value.replace(".", ","))
    if not match:
        raise ValueError("unsupported TTML time")
    hours, minutes, seconds, millis = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def subtitle_text_to_srt(text: str, suffix: str) -> str:
    suffix = suffix.lower().lstrip(".")
    if suffix in {"srt", "srv1"}:
        return _normalize_srt(text)
    if suffix in {"vtt", "webvtt"}:
        return _parse_vtt(text)
    if suffix in {"json", "bcc"}:
        return _parse_bilibili_json(text)
    if suffix in {"json3"}:
        return _parse_json3(text)
    if suffix in {"ttml", "dfxp", "xml"}:
        stripped = text.lstrip()
        if stripped.startswith("{"):
            return _parse_bilibili_json(text)
        return _parse_ttml(text)
    raise SubtitleCliError(f"Unsupported subtitle format: .{suffix or '(unknown)'}")


def subtitle_file_to_srt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise SubtitleCliError(f"Could not read subtitle file: {path}") from exc
    return subtitle_text_to_srt(text, path.suffix)
