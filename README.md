# Video Subtitles

跨平台命令行工具：从本地视频或受支持的在线视频中提取现有字幕；也可以识别画面中的烧录字幕，或在本机将语音转写为 SRT。

## 功能

- 使用 FFmpeg/FFprobe 检查媒体并提取字幕、音轨或视频帧。
- 在线媒体由 yt-dlp 读取；Cookie 仅在显式传入 `--cookies` 时使用。
- `--ocr` 识别烧录字幕。OCR 使用 PaddleOCR-VL：NVIDIA GPU 通过 vLLM 推理，CPU 通过 llama.cpp 推理。
- 没有可用字幕或 OCR 结果时，使用本地 faster-whisper 转写。默认 OCR 服务也在本机运行；如果通过 `--ocr-server-url` 指定远程服务，视频帧会发送到该服务。
- 默认不覆盖已有 SRT；需要覆盖时使用 `--force`。

## Linux / WSL 安装

推荐在 Linux x86_64 或 WSL 工程目录中运行配置器。Miniforge、Python、FFmpeg 和 Python 包安装在项目目录，不需要 Docker；系统只需提供下载工具和 SHA-256 校验工具。

```bash
./scripts/configure                 # 检查依赖
./scripts/configure --setup         # 安装客户端环境
```

安装会创建 `.envs/ocr-client`，其中包括 ASR/媒体处理包、CPU PaddleOCR-VL 客户端和轻量字幕候选检测器。Miniforge 位于 `.tools/`，包与模型缓存位于 `downloads/`。固定版本和缓存路径见 [`environment/ocr-versions.env`](environment/ocr-versions.env)。

### NVIDIA GPU 推理

在可通过 `nvidia-smi` 访问 NVIDIA GPU 的 Linux/WSL 环境中安装并启动项目内 vLLM 服务：

```bash
./scripts/setup-vlm-server.sh
./scripts/vlm-server.sh start
./scripts/vlm-server.sh status
```

服务环境在 `.envs/ocr-vlm-server`，日志在 `logs/vlm-server/`。首次启动会下载 PaddleOCR-VL 模型。处理完成后可运行 `./scripts/vlm-server.sh stop`。

### CPU 推理

CPU 服务使用项目内独立的 llama.cpp 环境和 PaddleOCR-VL GGUF 模型：

```bash
./scripts/setup-llama-server.sh
./scripts/llama-server.sh start
./scripts/llama-server.sh status
```

模型保存在 `downloads/paddleocr-vl-gguf/`，下载后会校验 SHA-256。服务日志在 `logs/llama-server/`；处理完成后运行 `./scripts/llama-server.sh stop`。

GPU 和 CPU 服务安装脚本目前面向 Linux x86_64/WSL。检测到 NVIDIA GPU 时，`--ocr-backend auto` 选择 vLLM；否则选择 llama.cpp。也可以用 `--ocr-backend vllm` 或 `--ocr-backend llama-cpp` 显式选择服务。服务不可用时程序会报错，不会静默切换推理后端。

## Windows 与 macOS

Windows PowerShell 客户端需要 Python 3.11 x64（通过 `py.exe`）以及在 `PATH` 中可用的 FFmpeg/FFprobe：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/configure.ps1
powershell -ExecutionPolicy Bypass -File scripts/configure.ps1 --setup
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --input .\video.mp4 --ocr --ocr-server-url http://127.0.0.1:8111/v1
```

OCR 推理服务仍需在 Linux/WSL 中单独安装和启动：CPU llama.cpp 默认端口是 `8111`，GPU vLLM 默认端口是 `8118`。macOS 可安装 Python 环境并使用 CLI；本项目没有提供 macOS 本地 OCR 服务安装脚本，可用 `--ocr-server-url` 连接兼容的服务。

## 使用

提取已有字幕或在需要时进行本地语音转写：

```bash
subtitles ./video.mp4
subtitles "https://www.bilibili.com/video/BV..."
```

识别画面中的烧录字幕：

```bash
./scripts/run.sh --input ./video.mp4 --ocr
./scripts/run.sh --input "https://www.bilibili.com/video/BV..." --cookies ./cookies.txt --ocr
```

指定输出位置、覆盖已有文件或限制 OCR 并发：

```bash
subtitles ./video.mp4 --output ./captions.srt --force
subtitles ./video.mp4 --ocr --ocr-concurrency 2
```

完整选项可通过 `subtitles --help` 查看。ASR 默认使用 `small` 模型，也可以通过 `--model` 选择 `tiny`、`base`、`medium`、`large-v3` 或 `large-v3-turbo`。模型首次使用时会下载到本地缓存。

## OCR 采样与资源使用

GPU 路径每 0.5 秒采样一次。CPU 路径使用 PP-OCRv5 mobile 文本检测器每 0.2 秒扫描字幕区域，并在发现候选字幕、字幕版式变化或稳定字幕复查时调用 llama.cpp；候选检测器不可用时回退到 0.5 秒识别间隔。客户端根据当前机器资源和实测帧吞吐调整并发，`--ocr-concurrency` 可限制并发上限。CPU VLM 单帧推理仍可能较慢；CPU 全片识别的速度和准确率尚未完成充分验证。

## 输出、日志和清理

- 默认 SRT 输出到本地视频旁边，在线视频输出到当前目录；使用 `--output` 可指定路径。
- 中间媒体和服务临时文件放在项目 `output/` 中，任务结束后会清理。
- 配置和运行日志放在 `logs/`；包、模型和下载缓存放在 `downloads/`。
- `./scripts/configure --clean` 清理运行缓存和生成字幕；`./scripts/configure --distclean` 还会删除项目管理的下载缓存和本地环境。清理前会要求确认，可加 `--yes` 跳过确认。

## 依赖概览

项目客户端依赖 PyAV、faster-whisper、ONNX Runtime、psutil、yt-dlp、CPU PaddlePaddle 和 PaddleOCR-VL。Linux 配置器还在项目 Conda 环境中提供 FFmpeg/FFprobe。GPU 推理服务使用独立的 vLLM/PyTorch 环境；CPU 推理服务使用独立的 Conda llama.cpp 环境。依赖检查命令为 `./scripts/configure`；修改依赖版本时请同步更新 `environment/ocr-versions.env` 和安装脚本。

在线网站的适配、登录要求和地区限制由 yt-dlp 支持情况决定。工具不会绕过 DRM、认证或访问限制，也不会自动读取浏览器 Cookie。
