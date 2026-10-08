# Video Subtitle CLI Design

## Goal

Build a cross-platform Python command-line tool that creates an SRT subtitle file from either a local video file or a supported online video page.

## User decisions

- Delivery: local command-line tool.
- Platforms: Windows, macOS, and Linux.
- Online inputs: use a generic site extractor (yt-dlp) for supported sites, including Bilibili.
- Subtitle strategy: use available dialogue subtitles first; if none are available, transcribe the audio locally.
- Transcription: local multilingual model, automatic language detection, output in the spoken language.
- Authentication: support a user-supplied cookies file through an explicit CLI option.
- Implementation language: Python.

## Proposed flow

1. Classify the input as a local file or URL.
2. For a URL, inspect available subtitle tracks and prefer a dialogue subtitle track. Convert the selected track to SRT. Do not treat danmaku as dialogue subtitles.
3. If no suitable subtitle track is available, obtain only the audio needed for transcription.
4. For a local file, inspect embedded subtitle tracks first; if none are available, extract audio for transcription.
5. Run a multilingual local speech recognition model with automatic language detection and timestamps.
6. Write a same-base-name `.srt` beside the local input, or beside the current working directory for a URL unless an output path is provided.
7. Remove temporary downloads and extracted audio when processing finishes or fails.

## Components

- CLI argument parsing and user-facing progress/errors.
- Input resolver for local paths and URLs.
- Media adapter based on yt-dlp for supported online sites and subtitle/audio retrieval.
- FFmpeg-backed local media inspection and audio extraction.
- Subtitle normalization to SRT.
- Local ASR adapter, with a CPU-capable default model configuration.

## CLI shape

```text
subtitles <input> [--output PATH] [--cookies FILE] [--model MODEL]
```

The output path and model are optional. `--cookies` is only used when explicitly passed. The initial version does not read cookies directly from browser profiles.

## Boundaries and errors

- Supported online pages are limited to sites currently handled by yt-dlp; site support can change independently of this tool.
- Login-required pages need a user-supplied cookie file. The tool does not bypass access controls or DRM.
- Report clear errors for unsupported URLs, unavailable media/subtitles, missing FFmpeg, invalid cookie files, model download/load failures, and transcription failures.
- Burned-in subtitle OCR, subtitle translation, cloud transcription, GUI, and batch processing are out of scope for the initial version.

## Acceptance criteria

- A local video with an embedded subtitle track produces an SRT without ASR.
- A local video without subtitles produces an SRT using local ASR.
- A supported online URL with an available dialogue subtitle produces an SRT without ASR.
- A supported online URL without usable subtitles falls back to local ASR using audio only.
- The user can explicitly pass a cookies file for sites that need an authenticated session.
- Outputs use the requested path or the documented default path; temporary media is cleaned up after success or failure.
- The command and its options are documented for Windows, macOS, and Linux.
