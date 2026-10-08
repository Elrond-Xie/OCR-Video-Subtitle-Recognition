# Video Subtitles

跨平台命令行工具：输入本地视频或 yt-dlp 支持的在线视频页面，优先提取已有对白字幕；没有可用字幕时，在本机运行语音识别并生成 SRT。

## 功能

- 支持 Windows、macOS、Linux。
- 支持本地视频文件和 yt-dlp 支持的在线页面，包括哔哩哔哩。
- 优先提取已有字幕；忽略 B 站弹幕。
- 使用 `--ocr` 时，优先识别烧录在画面中的字幕；识别不到时回退到本地 Whisper。
- 找不到可用字幕时，使用本机 faster-whisper 多语言模型转写，并自动识别原语言。
- 在线视频可通过 `--cookies` 显式提供登录 Cookie 文件。
- 生成 `.srt` 文件。已有输出文件默认不会被覆盖，需显式使用 `--force`。

## 安装

需要 Python 3.10 或更新版本，以及 FFmpeg（包含 `ffmpeg` 和 `ffprobe`），并确保两个程序都能从终端的 `PATH` 找到。FFmpeg 下载方式见[官方页面](https://ffmpeg.org/download.html)。

### 自适应 OCR 环境（推荐）

Linux 使用项目本地的 micromamba 环境。脚本会检查 NVIDIA GPU 并尝试配置 CUDA；GPU 环境安装、运行时检查或 PaddleOCR-VL 初始化失败时，会清理该环境并新建 CPU 环境：

```bash
bash scripts/setup.sh
bash scripts/run.sh ./video.mp4 --ocr
```

Windows PowerShell 使用独立 Python 虚拟环境，GPU 安装失败时同样创建全新的 CPU 环境：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 .\video.mp4 --ocr
```

脚本不会修改系统 Python 或 shell 启动文件。Linux 环境把 FFmpeg 一并装入 micromamba；Windows 需要自行安装 FFmpeg 并加入 `PATH`。首次 setup 会检查并预热 PaddleOCR-VL，可能下载模型文件。两套脚本都只尝试清单中固定的 Paddle/PaddleOCR 版本和 NVIDIA CUDA 12.6 运行时；CUDA 依赖由 Paddle wheel 提供，仍需兼容的 NVIDIA 驱动。若 GPU 不可用，就使用 CPU PaddleOCR 与 RapidOCR。

程序根据有效 CPU 数、可用内存、容器限制和 NVIDIA 显存选择 PaddleOCR-VL 或 RapidOCR，并限制 CPU 线程、预取帧和推理批量。运行时会报告所选模型和设备；VL 无法初始化或推理时会回退到 RapidOCR。PaddleOCR-VL 的文档解析能力不代表它在所有视频字幕、字体或画面条件下都更准确；当前没有项目字幕样本基准，最终效果仍受视频质量和语言影响。

macOS 本轮暂不安装 PaddleOCR-VL，继续使用现有 Python 安装流程中的 RapidOCR CPU 路径。

### 手动 Python 环境

如需手动管理环境，在项目目录运行：

```bash
python -m venv .venv
```

Linux/macOS 激活虚拟环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

安装命令行工具：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## 使用

提取本地视频字幕；如果没有可用字幕轨，会自动在本机转写：

```bash
subtitles ./video.mp4
```

处理在线视频页面：

```bash
subtitles "https://www.bilibili.com/video/BV..."
```

指定输出位置：

```bash
subtitles ./video.mp4 --output ./captions.srt
```

覆盖已存在的输出文件：

```bash
subtitles ./video.mp4 --output ./captions.srt --force
```

为需要登录的页面显式传入 Cookie 文件：

```bash
subtitles "https://www.bilibili.com/video/BV..." --cookies ./cookies.txt
```

识别视频画面中烧录的字幕（用于画面字幕与可下载字幕轨不一致的情况）：

```bash
subtitles ./video.mp4 --ocr
subtitles "https://www.bilibili.com/video/BV..." --cookies ./cookies.txt --ocr
```

OCR 大约每 0.5 秒识别一帧并估算字幕时间，终端会显示实时进度条；能获取视频总时长时显示百分比和当前时间，时长未知时显示已处理时间。自适应环境会优先在高资源机器上运行 PaddleOCR-VL；弱机器、VL 初始化或推理失败时使用 RapidOCR CPU。手动安装环境仍可通过 `pip install -e .` 使用 RapidOCR CPU。OCR 会解码采样点之间的视频帧，因此处理速度取决于视频分辨率和编解码器，可能慢于视频播放速度。时间精度和文字准确度会受视频帧率、字幕样式、位置与对比度影响。在线视频 OCR 需要 yt-dlp 获取视频画面；程序优先尝试流式读取，不兼容时会使用并清理临时视频文件。如果 yt-dlp 仍无法获取视频，先通过你可用的方式保存视频，再把本地文件交给命令处理。`--ocr` 模式不会退回可能不匹配的下载字幕轨。

选择 ASR 模型（默认 `small`；仅无可用字幕或 OCR 未识别到画面字幕时才会加载）：

```bash
subtitles ./video.mp4 --model base
```

支持的模型名：`tiny`、`base`、`small`、`medium`、`large-v3`、`large-v3-turbo`。首次执行 ASR 时需要下载模型文件并缓存在本机；视频音频不会上传。CPU 速度和可用内存会影响转写耗时，可用较小模型降低资源需求。

## 输出位置

- 本地文件：同目录、同主文件名的 `.srt`。
- 在线页面：当前工作目录下以视频标题命名的 `.srt`；无法获取标题时使用稳定的回退名称。
- `--output` 可以指定完整输出路径。
- 工具使用临时目录处理媒体，并在任务结束后清理临时文件。

## 限制

- 在线页面支持情况取决于 yt-dlp 的站点适配；平台变化、登录要求、地区限制和视频权限可能影响提取。
- 登录页面必须由用户显式传入 Cookie 文件。工具不绕过 DRM 或平台访问限制，也不会直接读取浏览器配置。
- 当前 OCR 只识别画面下方持续出现的字幕文本，不包含字幕翻译、批量队列或图形界面。
- 部分图像字幕轨（例如 PGS）不能直接转成文本；这类文件会尝试回退到音频转写。
