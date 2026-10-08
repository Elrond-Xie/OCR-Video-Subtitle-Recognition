# Adaptive Video Subtitle OCR Design

## Goal

Improve burned-in subtitle recognition while keeping the existing video-subtitle CLI usable on both high-performance and low-resource Windows/Linux machines. Prefer PaddleOCR-VL when the host can run it reliably; keep RapidOCR as the lightweight fallback. Detect and use a supported NVIDIA GPU when its Python runtime works, and otherwise use CPU.

## User decisions

- PaddleOCR-VL is the preferred OCR engine on machines with enough memory and a working inference backend.
- RapidOCR remains available as a fallback for constrained hosts and failed PaddleOCR-VL initialization or inference. Its current PP-OCRv6 small defaults remain the lightweight model path.
- Engine and device selection are automatic by default. Logs identify the selected engine, device, and fallback reason.
- Environment preparation is explicit and isolated from the user's global Python environment.
- Linux setup uses micromamba by default. Windows receives a dedicated PowerShell setup script. New PaddleOCR-VL setup and GPU configuration are deferred on macOS; its existing RapidOCR CPU path remains in scope for preservation.
- Initial automatic GPU environment installation targets NVIDIA CUDA. Unsupported vendors or combinations use CPU.
- Resource use is detected at runtime and bounded: high-end machines should use their available CPU/GPU efficiently, while lower-resource machines should select a smaller path and remain usable.
- Existing subtitle extraction, online retrieval, local-only media handling, SRT output, and ASR fallback behavior remain intact.

## Approaches considered

1. **PaddleOCR-VL on every host, then RapidOCR only after failure.** This follows the preferred engine literally, but may initialize a large model on low-memory systems and can be very slow on CPU.
2. **Resource-aware hybrid routing (selected).** Use PaddleOCR-VL when preflight resource checks and a runtime warm-up indicate adequate headroom. Use RapidOCR directly on constrained machines and fall back to it if PaddleOCR-VL cannot initialize or repeatedly fails. This preserves quality on capable systems without making the larger model a minimum requirement.
3. **RapidOCR as the primary engine with PaddleOCR-VL only for uncertain frames.** This is more efficient for ordinary videos, but conflicts with the requested preference for PaddleOCR-VL on capable machines.

PaddleOCR-VL is a document-parsing pipeline, not a video-subtitle-specific benchmark winner. Its official guide supports local inference on x64 CPU and multiple accelerators and recommends using the complete pipeline rather than the VLM component alone; it also warns that using only the VLM can produce hallucinated text. The implementation must therefore filter results for subtitle-like position and persistence, and must not claim that VL is universally more accurate on every subtitle style. See [PaddleOCR-VL usage and hardware matrix](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html).

## Proposed processing flow

1. Keep existing subtitle-track extraction as the default when `--ocr` is absent.
2. In `--ocr` mode, read machine resources and the configured inference environment. Account for process/container CPU and memory limits as well as host resources.
3. Probe for a supported NVIDIA runtime. A visible NVIDIA device alone is insufficient: the installed Paddle runtime must initialize CUDA and report a usable device.
4. If the GPU runtime is usable and free VRAM plus host RAM meet the configured headroom policy, initialize PaddleOCR-VL on GPU. If GPU setup is unavailable but host memory is sufficient, try PaddleOCR-VL on CPU. If the host is constrained, use RapidOCR directly.
5. Run a small warm-up inference before processing the video. If model initialization or warm-up fails, record the reason and switch to the next viable engine/device route.
6. Decode sampled frames through a bounded producer/consumer queue. Run OCR on frames, favor subtitle-like text near the lower image region, and merge persistent detections into SRT cues using the existing subtitle utilities.
7. If PaddleOCR-VL returns no usable text or an output fails consistency checks, run RapidOCR on that frame as a second opinion. If PaddleOCR-VL raises a runtime error, disable that engine for the remainder of the task and continue with RapidOCR.
8. If OCR produces no usable cues, preserve the existing local ASR fallback. Online video retrieval errors continue to report an actionable error rather than silently switching to ASR when frames could not be retrieved.
9. Log the selected engine/device and fallbacks to stderr. Do not log media contents, cookies, or recognized text by default.

## Components

- `src/video_subtitles/ocr.py`: define a common frame-recognition adapter, PaddleOCR-VL and RapidOCR adapters, result normalization, consistency checks, and engine fallback.
- `src/video_subtitles/pipeline.py`: use automatic OCR engine/device routing while preserving existing subtitle, online media, ASR, and output behavior.
- `src/video_subtitles/media.py`: retain timestamped frame decoding and add bounded prefetch only where it improves throughput without unbounded frame buffering.
- `src/video_subtitles/online.py`: keep the existing yt-dlp stream and temporary-video behavior; feed frames through the same bounded OCR pipeline.
- `src/video_subtitles/resources.py` (new): inspect effective CPU count, available RAM, container/cgroup limits, available NVIDIA VRAM, and provide a resource plan with conservative reserves.
- `scripts/setup.sh` (new): Linux environment bootstrap and installation using micromamba.
- `scripts/setup.ps1` (new): Windows environment bootstrap and installation using a Python virtual environment and PowerShell.
- `environment/` (new): pinned CPU/GPU dependency manifests or lock inputs used by setup scripts. CPU and GPU Paddle packages must not be installed together in one environment.
- `README.md`: setup instructions, supported hardware scope, GPU fallback behavior, resource-selection logs, and the deferred macOS status.
- `pyproject.toml`: separate base/app dependencies from OCR runtime choices so CPU and GPU runtime packages do not conflict.

## Model and result policy

- PaddleOCR-VL is the preferred route only after resource preflight and warm-up. Use its supported full local pipeline and structured result format; parse output conservatively.
- For subtitle frames, select text in or near the lower caption region and require temporal persistence before emitting a cue. Use a focused lower-region pass when appropriate, with a full-frame pass available when the focused pass finds no text.
- Treat VL output as a candidate, not ground truth. A cue that is empty, structurally invalid, or inconsistent across adjacent frames is eligible for RapidOCR review.
- RapidOCR remains the lightweight fallback and uses a pinned RapidOCR release with PP-OCRv6 small ONNX models. The current project requirement already permits RapidOCR 3.9+, whose official defaults use PP-OCRv6 small; the implementation should pin a known-compatible release rather than rely on a moving dependency range. See [RapidOCR model list](https://rapidai.github.io/RapidOCRDocs/main/model_list/).
- Do not load two copies of PaddleOCR-VL to parallelize frames. A failed VL runtime should be shut down before selecting the fallback where possible.

## Device detection and fallback

- Initial automatic GPU installation and verification target NVIDIA CUDA on Linux and Windows.
- Detection must verify framework runtime capability and device initialization, not rely only on `nvidia-smi` or a hardware name.
- Supported GPU wheel/version combinations are selected from a maintained compatibility manifest. The setup script must not scrape the vendor documentation or install arbitrary development wheels at runtime.
- An unsupported GPU, missing driver, incompatible wheel, failed CUDA initialization, insufficient free VRAM, or model-level GPU failure selects CPU or RapidOCR according to the resource plan.
- The CPU and GPU Paddle packages are mutually exclusive. The installer creates a separate isolated environment for each installation attempt; on GPU installation or smoke-check failure, it creates a fresh CPU environment and verifies that environment instead of trying to repair a partially installed GPU environment in place.
- Other GPU vendors are detected only for diagnostics in this release and use CPU. Their runtime packages can be added later through separate compatibility entries.

## Environment setup behavior

### Linux

- `scripts/setup.sh` checks for micromamba and bootstraps it into a project-local tools directory from the official distribution if it is absent.
- It creates an isolated project environment, installs the pinned application and media dependencies, and tries the maintained NVIDIA GPU package combination only when NVIDIA hardware and a compatible driver are detected.
- It imports Paddle, checks the actual CUDA device count, and runs a small local model initialization/warm-up. Any failure causes a clean CPU environment to be created and verified.
- The script reports the chosen environment prefix and command for launching the CLI. It never modifies system Python packages.

### Windows

- `scripts/setup.ps1` uses the Python Launcher (`py`) to create a project-local `venv`, then uses the same pinned dependency manifests and install verification flow.
- It attempts GPU setup only for a stable official Windows wheel combination in the compatibility manifest. If none matches the detected driver/GPU/Python version or verification fails, it installs the CPU environment.
- The PowerShell script provides actionable messages when Python, the Windows runtime, or required FFmpeg tools are unavailable. It does not install unverified preview/development wheels automatically.

### macOS

- New PaddleOCR-VL setup and automatic GPU configuration are out of scope for this change. Preserve the currently documented RapidOCR CPU path on macOS where its existing dependencies are installable; do not imply that the new PaddleOCR-VL environment setup is supported there.

## Resource scheduling

- Determine effective CPU availability from OS/container limits; reserve capacity for the OS and media work instead of multiplying per-frame workers by the OCR runtime's own thread pool.
- Determine available RAM and, on supported NVIDIA systems, free VRAM. Model selection uses available headroom, not total installed memory alone.
- Maintain per-engine/device memory eligibility thresholds in the compatibility configuration. If resource detection is unavailable or a host falls below the threshold, choose the smaller route without initializing PaddleOCR-VL.
- Preserve a configurable safety reserve for the OS and other applications. Bound frame queue depth by the remaining memory budget; never enqueue an entire video.
- CPU inference uses a thread count derived from effective cores and leaves system capacity free, especially on small machines. Decoder prefetch and inference concurrency share one total resource budget.
- GPU inference uses one model instance per process. Select batch size from free VRAM and reduce it on allocation failure; do not run many independent GPU workers that duplicate model weights.
- Use PaddleOCR-VL's incremental prediction interface where it can process the frame stream, avoiding retention of all results in memory. RapidOCR remains one bounded frame at a time unless measurements justify a small batch.
- If a warm-up or allocation fails, lower the model/device tier and retry once before reporting an actionable low-memory error.

## CLI and user-visible behavior

The existing `subtitles INPUT --ocr` remains the automatic entry point. No extra required flags are introduced. Startup output should identify the selected OCR engine and device (for example, PaddleOCR-VL on CUDA, PaddleOCR-VL on CPU, or RapidOCR on CPU) and explain a fallback in one concise line. Progress reporting continues to use the existing OCR progress callback.

## Errors and cleanup

- Setup failures report which GPU attempt failed and confirm whether a CPU environment was successfully created.
- If both PaddleOCR-VL and RapidOCR are unavailable, report the missing package/model or memory issue clearly; do not silently switch to an unrelated source.
- On runtime model failure, release model resources before fallback where possible and continue from the current video frame. If fallback also fails, raise the existing domain error.
- Keep temporary media cleanup and downloader process cleanup behavior unchanged.
- Never expose cookie contents or upload video frames/audio.

## Acceptance criteria

- The default non-OCR path is unchanged.
- With `--ocr`, supported high-resource machines select PaddleOCR-VL; constrained machines select RapidOCR before exhausting resources.
- A functioning supported NVIDIA CUDA environment is used automatically. Missing/incompatible GPU runtime, initialization failure, or warm-up failure selects CPU without requiring a manual reinstall.
- PaddleOCR-VL runtime failure switches to RapidOCR and processing continues from the current frame; empty or inconsistent VL detections can be checked by RapidOCR.
- CPU thread count, frame queue depth, and GPU batch size respond to effective system resources and have explicit upper bounds.
- Linux setup uses micromamba and Windows setup creates an isolated virtual environment; GPU install failure results in a separately verified CPU environment.
- macOS keeps its existing RapidOCR CPU path; PaddleOCR-VL setup and GPU configuration are explicitly deferred.
- Existing local/online video handling, subtitle timing/merging, ASR fallback, output naming, cookie policy, and temporary cleanup are preserved.
- Setup scripts do not modify global Python packages or install unverified development wheels.

## Risks and assumptions

- PaddleOCR-VL is designed for document parsing; its accuracy on video subtitles must not be assumed from document benchmarks. Lower-region selection, conservative parsing, persistence checks, and RapidOCR review reduce false cues but cannot guarantee exact transcription.
- PaddleOCR-VL requires a substantially heavier dependency/model stack than RapidOCR. CPU support does not imply acceptable speed on every machine; adaptive routing is required.
- GPU Python wheels are sensitive to Python version, platform, driver, CUDA, and device architecture. A maintained compatibility manifest is required; unknown combinations should fall back to CPU.
- Model and package installation may require internet access and substantial disk space. Environment preparation should report download progress and model-cache paths.
- Runtime memory availability may change during processing. Resource limits are estimates and must retain headroom instead of attempting to consume all reported free memory.
- First-release GPU installation targets NVIDIA CUDA only. Other hardware vendors use CPU until their installation and runtime support can be maintained reliably.

## References

- [PaddleOCR-VL usage, full-pipeline guidance, and hardware support](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleOCR framework installation and CPU/GPU package selection](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/paddlepaddle_installation.en.md)
- [RapidOCR model list and PP-OCRv6 support](https://rapidai.github.io/RapidOCRDocs/main/model_list/)
