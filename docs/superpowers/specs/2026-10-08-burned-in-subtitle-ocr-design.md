# Burned-in Subtitle OCR Design

## Goal

Add an opt-in OCR mode to the `subtitles` CLI for extracting text that is visibly burned into video frames. This mode addresses videos where the downloadable subtitle track does not match the captions shown in the picture.

## User decisions

- Keep the existing subtitle-track-first and ASR behavior as the default.
- Add an explicit `--ocr` option to prefer on-screen text for a selected video.
- OCR must run locally; video frames and audio must not be uploaded to a service.
- Keep support for local videos and yt-dlp-supported online video pages, including an explicitly supplied cookies file.
- If OCR finds no usable visible captions, fall back to local Whisper transcription.

## Proposed flow

1. When `--ocr` is absent, preserve the existing extraction pipeline.
2. When `--ocr` is present, prepare the video frames:
   - For a local input, read frames from the local video.
   - For an online URL, use yt-dlp to download the video into the existing temporary work directory, passing cookies only if the user supplied `--cookies`.
3. Sample approximately two frames per second and run local OCR on each sampled frame. Process the full frame, then prefer stable text detections in the lower portion of the image, where dialogue captions are normally placed.
4. Normalize recognized lines, suppress low-confidence or isolated detections, merge consecutive repeated text, and infer cue boundaries from the sample times.
5. Convert the resulting cues to SRT and write them using the existing output and `--force` behavior.
6. If OCR produces no usable cues, use the existing local ASR path. Do not substitute a known-mismatching downloaded subtitle track in `--ocr` mode.
7. Clean up downloaded video and intermediate frames with the temporary work directory.

## Components

- `src/video_subtitles/cli.py`: add the `--ocr` switch and report OCR/model setup progress.
- `src/video_subtitles/pipeline.py`: route `--ocr` inputs through OCR before any subtitle-track or ASR processing.
- `src/video_subtitles/ocr.py`: sample frames, run OCR, filter and merge detections, and serialize cues through the existing SRT helpers.
- `src/video_subtitles/media.py`: expose a frame-reading helper for local inputs using the existing FFmpeg media dependency.
- `src/video_subtitles/online.py`: download video to the temporary directory for OCR mode, preserving yt-dlp support and explicit cookie handling.
- `pyproject.toml`: add RapidOCR with CPU ONNX Runtime support.
- `README.md`: document `--ocr`, local model behavior, online video download/storage cost, and estimated timestamp precision.

RapidOCR is the proposed engine because it accepts image arrays and returns text, bounding boxes, and confidence scores. Its documentation recommends ONNX Runtime CPU as a starting point and provides a model-download command for offline deployment. Model initialization and caching must be handled with visible progress and a clear error if model files cannot be obtained. See [RapidOCR usage](https://rapidai.github.io/RapidOCRDocs/latest/install_usage/rapidocr/usage) and [RapidOCR quickstart](https://rapidai.github.io/RapidOCRDocs/latest/quickstart).

## CLI shape

```text
subtitles <input> [--output PATH] [--cookies FILE] [--model MODEL] [--ocr] [--force]
```

- `--ocr` is opt-in and means “use the captions visible in the video image.”
- Without `--ocr`, existing subtitle-track extraction remains the fast default.
- `--model` continues to configure the Whisper fallback only.

## Timing and recognition behavior

- Sample at approximately 2 frames per second, giving estimated cue boundaries to about 0.5 seconds.
- Use OCR text, position, and confidence to favor persistent dialogue captions near the lower image area.
- Normalize whitespace and punctuation conservatively; do not translate or rewrite recognized text.
- Merge consecutive detections of the same caption and ignore transient detections that do not persist across samples.
- The first version does not expose frame rate, subtitle region, or OCR confidence threshold as CLI flags.

## Boundaries and errors

- Online OCR requires downloading video rather than audio only. Temporary video files are removed after processing; bandwidth and disk use can be substantial.
- OCR is CPU-capable and local, but may be slower than subtitle-track extraction. The first use may need to obtain OCR model files.
- Accuracy depends on caption font, contrast, animation, screen position, and frame sampling. OCR does not guarantee exact text or frame-perfect timing.
- If the video has no visible dialogue captions or OCR cannot produce usable cues, the command falls back to the existing local ASR behavior.
- The feature does not capture browser-player overlays, translate subtitles, recognize speakers, or bypass DRM or site access restrictions.

## Acceptance criteria

- `subtitles INPUT --ocr` attempts OCR even when an online subtitle track exists.
- Local video OCR produces SRT cues from captions burned into video frames without uploading media.
- Online OCR uses yt-dlp to retrieve video, honors an explicitly supplied cookie file, and does not download a video when `--ocr` is absent.
- Repeated visible captions are merged into timed cues and transient non-caption text is filtered where possible.
- If OCR finds no usable captions, local ASR is used; if OCR succeeds, ASR is not loaded.
- Output paths, overwrite protection, and temporary-file cleanup follow existing CLI behavior.

## Validation constraints

- Do not add or run tests unless the user explicitly asks for testing or verification.
- Before completion, permitted non-test checks are limited to static inspection and non-networking CLI/package metadata checks, consistent with the existing plan.
