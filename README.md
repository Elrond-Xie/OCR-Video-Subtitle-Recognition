# Video Subtitles

命令行工具：从本地视频或受支持的在线视频中提取现有字幕；也可以识别画面中的烧录字幕，或在本机将语音转写为 SRT。项目提供 Linux/WSL 配置器和 Windows 客户端配置器；OCR 推理服务目前通过 Linux/WSL 脚本安装。

## 功能

- 使用 FFmpeg/FFprobe 检查媒体并提取字幕、音轨或视频帧。
- 在线媒体由 yt-dlp 读取；Cookie 仅在显式传入 `--cookies` 时使用。
- `--ocr` 识别烧录字幕。OCR 使用 PaddleOCR-VL：NVIDIA GPU 通过 vLLM 推理，CPU 通过 llama.cpp 推理。
- 没有可用字幕或 OCR 结果时，使用本地 faster-whisper 转写。默认 OCR 服务地址是本机；如果通过 `--ocr-server-url` 指定其他服务，视频帧会发送到该服务。
- 默认不覆盖已有 SRT；需要覆盖时使用 `--force`。

## Linux / WSL 安装

在 Linux x86_64 或 WSL 工程目录中运行配置器。Miniforge、Python、FFmpeg 和 Python 包安装在项目目录，不需要 Docker。系统需要 `curl` 或 `wget`、`sha256sum` 或 `shasum`；缺少时，配置器会尝试通过支持的系统包管理器安装 `curl` 和 `coreutils`，这一步可能需要 `sudo`。系统 FFmpeg/FFprobe 是可选的，`--setup` 会在项目环境中安装它们。

```bash
./scripts/configure                 # 检查依赖
./scripts/configure --setup         # 安装客户端环境
```

安装会创建 `.envs/ocr-client`，其中包括 ASR/媒体处理包、CPU PaddleOCR-VL 客户端和轻量字幕候选检测器。Miniforge 位于 `.tools/miniforge3/`，包与模型缓存位于 `downloads/`。固定版本和缓存路径见 [`environment/ocr-versions.env`](environment/ocr-versions.env)。安装完成后通过 `./scripts/run.sh` 启动客户端。

### NVIDIA GPU 推理

GPU 服务当前要求可通过 `nvidia-smi` 使用的 NVIDIA RTX 50 系列（Blackwell，计算能力 12.0）GPU，并在 Linux/WSL 中运行。安装并启动项目内 vLLM 服务：

```bash
./scripts/setup-vlm-server.sh
./scripts/vlm-server.sh start
./scripts/vlm-server.sh status
```

服务环境在 `.envs/ocr-vlm-server`，日志在 `logs/vlm-server/`。首次启动会下载 PaddleOCR-VL 模型；服务按检测到的系统容量配置最大并发。处理完成后可运行 `./scripts/vlm-server.sh stop`。

### CPU 推理

CPU 服务使用项目内独立的 Conda llama.cpp 环境和 PaddleOCR-VL GGUF 模型：

```bash
./scripts/setup-llama-server.sh
./scripts/llama-server.sh start
./scripts/llama-server.sh status
```

服务环境在 `.envs/ocr-llama-server`。模型保存在 `downloads/paddleocr-vl-gguf/`，下载后会校验 SHA-256。服务日志在 `logs/llama-server/`；处理完成后运行 `./scripts/llama-server.sh stop`。

GPU 和 CPU 服务安装脚本面向 Linux x86_64/WSL。`--ocr-backend auto` 在检测到可用 NVIDIA GPU 时选择 vLLM，否则选择 llama.cpp。GPU 机器需要先单独安装、启动 vLLM 服务；CPU 机器需要先安装、启动 llama.cpp 服务。也可以用 `--ocr-backend vllm` 或 `--ocr-backend llama-cpp` 显式选择。服务不可用时程序会报错，不会静默切换推理后端。

## Windows 与 macOS

Windows PowerShell 客户端需要 Python 3.11 x64（通过 `py.exe`）以及在 `PATH` 中可用的 FFmpeg/FFprobe。配置器会在工程内创建客户端环境：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/configure.ps1
powershell -ExecutionPolicy Bypass -File scripts/configure.ps1 --setup
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --input .\video.mp4 --ocr --ocr-backend vllm --ocr-server-url http://127.0.0.1:8118/v1
```

Windows 的 OCR 推理服务仍需在 Linux/WSL 中单独安装和启动：CPU llama.cpp 默认端口是 `8111`，GPU vLLM 默认端口是 `8118`。示例使用 GPU 服务；使用 CPU 服务时同时改为 `--ocr-backend llama-cpp --ocr-server-url http://127.0.0.1:8111/v1`。客户端的 backend 必须与服务匹配。macOS 没有项目配置器或本地 OCR 服务安装脚本；可手动安装项目 Python 依赖并配置 FFmpeg/FFprobe 运行基础 CLI，或用 `--ocr-server-url` 连接兼容服务。

## 使用

提取已有字幕或在需要时进行本地语音转写：

```bash
./scripts/run.sh --input ./video.mp4
./scripts/run.sh --input "https://www.bilibili.com/video/BV..."
```

识别画面中的烧录字幕：

```bash
./scripts/run.sh --input ./video.mp4 --ocr
./scripts/run.sh --input "https://www.bilibili.com/video/BV..." --cookies ./cookies.txt --ocr
```

指定输出位置、覆盖已有文件或限制 OCR 并发：

```bash
./scripts/run.sh --input ./video.mp4 --output ./captions.srt --force
./scripts/run.sh --input ./video.mp4 --ocr --ocr-concurrency 2
```

完整选项可通过 `./scripts/run.sh --help` 查看。ASR 默认使用 `small` 模型，也可以通过 `--model` 选择 `tiny`、`base`、`medium`、`large-v3` 或 `large-v3-turbo`。模型首次使用时会下载到本地缓存。

## OCR 采样与资源使用

GPU 路径每 0.5 秒采样一次。CPU 路径使用 PP-OCRv5 mobile 文本检测器每 0.2 秒扫描字幕区域，并在发现候选字幕、字幕版式变化或稳定字幕复查时调用 llama.cpp；检测器初始化或运行失败时会回退到 0.5 秒 OCR 采样。客户端根据当前机器可用资源和实测帧吞吐调整并发，`--ocr-concurrency` 可限制客户端并发上限。CPU VLM 单帧推理仍可能较慢；CPU 全片识别的速度和准确率尚未完成充分验证。

## 输出、日志和清理

- 默认 SRT 输出到当前工作目录的 `output/`，文件名取本地视频文件名或在线视频标题；使用 `--output` 可指定其他路径。自定义输出目录必须预先存在。
- 中间媒体和服务临时文件放在项目 `output/` 下的临时目录，任务结束后会清理。
- 配置和运行日志放在 `logs/`；包、模型和下载缓存放在 `downloads/`。
- `./scripts/configure --clean` 会停止项目服务并清理运行临时文件、Python 缓存及工程内的 `.srt` 文件，保留下载缓存和环境；`./scripts/configure --distclean` 还会删除项目管理的下载缓存、Miniforge 和 OCR 环境。两种清理都需要确认，可加 `--yes` 跳过确认。深度清理会删除 `logs/configure/` 和 `logs/run/`，但保留服务日志目录。

## 依赖概览

项目客户端使用 PyAV、faster-whisper（含 ONNX Runtime）、psutil、yt-dlp、CPU PaddlePaddle 和 PaddleOCR-VL。Linux 配置器还在项目 Conda 环境中提供 FFmpeg/FFprobe。GPU 推理服务使用独立的 vLLM/PyTorch 环境；CPU 推理服务使用独立的 Conda llama.cpp 环境。Linux 依赖检查命令为 `./scripts/configure`，Windows 检查命令为 `powershell -ExecutionPolicy Bypass -File scripts/configure.ps1`；修改固定依赖版本时请同步更新 [`environment/ocr-versions.env`](environment/ocr-versions.env) 和相应安装脚本。

在线网站的适配、登录要求和地区限制由 yt-dlp 支持情况决定。工具不会绕过 DRM、认证或访问限制，也不会自动读取浏览器 Cookie。

## 许可证

本项目采用 [Apache License 2.0](LICENSE)。
