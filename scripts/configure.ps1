param()
$ErrorActionPreference = 'Stop'
$Operation = 'check'
$AssumeYes = $false
foreach ($Argument in $args) {
    switch ($Argument) {
        '--setup' { if ($Operation -ne 'check') { throw 'Choose only one operation.' }; $Operation = 'setup' }
        '--yes' { $AssumeYes = $true }
        '-y' { $AssumeYes = $true }
        '--help' { Write-Host 'Usage: configure.ps1 [--setup [--yes]]'; exit 0 }
        default { throw "Unknown option: $Argument" }
    }
}
if ($AssumeYes -and $Operation -ne 'setup') { throw '--yes must be used with --setup.' }
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Manifest = ConvertFrom-StringData (Get-Content (Join-Path $Root 'environment/ocr-versions.env') -Raw)
$State = Join-Path $Root '.ocr-environment'
$ClientEnv = Join-Path $Root '.venv-ocr-client'
$ConfiguredDownloadDir = $Manifest.DOWNLOAD_DIR.Trim().Trim('"').Trim("'")
$ConfiguredLogDir = $Manifest.LOG_DIR.Trim().Trim('"').Trim("'")
$SaveLogs = $true
$script:TranscriptActive = $false
if ($Manifest.ContainsKey('SAVE_LOGS')) {
    $SaveLogs = [System.Convert]::ToBoolean($Manifest.SAVE_LOGS.Trim().Trim('"').Trim("'"))
}
if ([System.IO.Path]::IsPathRooted($ConfiguredDownloadDir)) {
    $DownloadRoot = [System.IO.Path]::GetFullPath($ConfiguredDownloadDir)
} else {
    $DownloadRoot = [System.IO.Path]::GetFullPath((Join-Path $Root $ConfiguredDownloadDir))
}
$LogRoot = if ([System.IO.Path]::IsPathRooted($ConfiguredLogDir)) {
    [System.IO.Path]::GetFullPath($ConfiguredLogDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $Root $ConfiguredLogDir))
}
if ($SaveLogs) {
    $LogDirectory = Join-Path $LogRoot 'configure'
    New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
    $LogPath = Join-Path $LogDirectory ("configure-{0}-{1}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $PID)
    Start-Transcript -Path $LogPath -Append | Out-Null
    $script:TranscriptActive = $true
    Write-Host "[INFO] Configuration output is being saved to: $LogPath"
}

function Stop-ConfigureTranscript {
    if (-not $SaveLogs -or -not $script:TranscriptActive) { return }
    Stop-Transcript | Out-Null
    $script:TranscriptActive = $false
    if (Test-Path $LogPath) {
        $Contents = [System.IO.File]::ReadAllText($LogPath)
        $Contents = [System.Text.RegularExpressions.Regex]::Replace($Contents, '\x1B\[[0-?]*[ -/]*[@-~]', '')
        [System.IO.File]::WriteAllText($LogPath, $Contents, [System.Text.UTF8Encoding]::new($false))
    }
}
trap { Stop-ConfigureTranscript; throw }
$env:PIP_PROGRESS_BAR = 'on'
$env:LOG_DIR = $LogRoot
$env:SAVE_LOGS = $SaveLogs.ToString().ToLowerInvariant()
$env:DOWNLOAD_DIR = $DownloadRoot
$env:PIP_CACHE_DIR = Join-Path $DownloadRoot 'pip'
$env:XDG_CACHE_HOME = Join-Path $DownloadRoot 'xdg'
$env:HF_HOME = Join-Path $DownloadRoot 'huggingface'
$env:HF_HUB_CACHE = Join-Path $DownloadRoot 'huggingface/hub'
$env:PADDLE_PDX_CACHE_HOME = Join-Path $DownloadRoot 'paddlex'
$env:TORCH_HOME = Join-Path $DownloadRoot 'torch'

function Write-DependencyStatus([bool]$Available, [string]$Name, [string]$Detail = '') {
    $Status = if ($Available) { 'YES' } else { 'NO' }
    Write-Host ('{0,-3} | {1,-46} {2}' -f $Status, $Name, $Detail)
}

function Test-EnvironmentImport([string]$Python, [string]$Name, [string]$Expression) {
    & $Python -c $Expression *> $null
    Write-DependencyStatus ($LASTEXITCODE -eq 0) $Name $(if ($LASTEXITCODE -ne 0) { 'Run configure.ps1 --setup.' })
}

function Write-InstallStage([string]$Message) {
    Write-Host "[INFO] $Message"
}

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
Write-Host 'Checking required software and the selected OCR environment.'
$Ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$Ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
$FfmpegReady = $null -ne $Ffmpeg
$FfprobeReady = $null -ne $Ffprobe
if ($FfmpegReady) { & ffmpeg -version *> $null; $FfmpegReady = ($LASTEXITCODE -eq 0) }
if ($FfprobeReady) { & ffprobe -version *> $null; $FfprobeReady = ($LASTEXITCODE -eq 0) }
Write-DependencyStatus $FfmpegReady 'FFmpeg executable' $(if ($FfmpegReady) { $Ffmpeg.Source } else { 'Install FFmpeg and add it to PATH.' })
Write-DependencyStatus $FfprobeReady 'FFprobe executable' $(if ($FfprobeReady) { $Ffprobe.Source } else { 'Install FFmpeg and add it to PATH.' })
Write-DependencyStatus ($null -ne $PythonLauncher) 'Python Launcher (py.exe)' $(if (-not $PythonLauncher) { 'Install 64-bit Python 3.11.' })
$SystemPythonReady = $false
if ($PythonLauncher) {
    & py "-$($Manifest.PYTHON_VERSION)" -c 'import sys; assert sys.maxsize > 2**32' *> $null
    $SystemPythonReady = ($LASTEXITCODE -eq 0)
}
Write-DependencyStatus $SystemPythonReady "Python $($Manifest.PYTHON_VERSION) x64 for setup" $(if (-not $SystemPythonReady) { 'Install 64-bit Python 3.11.' })

$StateReady = $false
$Values = $null
if (Test-Path $State) {
    $Values = ConvertFrom-StringData (Get-Content $State -Raw)
    $StateReady = Test-Path $Values.prefix
}
Write-DependencyStatus $StateReady 'Project OCR Python environment' $(if (-not $StateReady) { 'Run configure.ps1 --setup.' })
if ($StateReady) {
    $Python = Join-Path $Values.prefix 'Scripts/python.exe'
    $PythonReady = Test-Path $Python
    Write-DependencyStatus $PythonReady 'Project Python runtime (3.10 or newer)' $(if (-not $PythonReady) { 'Run configure.ps1 --setup.' })
    if ($PythonReady) {
        & $Python -c 'import sys; assert sys.version_info >= (3, 10)' *> $null
        Write-DependencyStatus ($LASTEXITCODE -eq 0) 'Python version requirement' $(if ($LASTEXITCODE -ne 0) { 'Requires Python 3.10 or newer.' })
        & $Python -m pip --version *> $null
        Write-DependencyStatus ($LASTEXITCODE -eq 0) 'Python pip installer' $(if ($LASTEXITCODE -ne 0) { 'Run configure.ps1 --setup.' })
        Test-EnvironmentImport $Python 'Python package: PyAV (av)' 'import av'
        Test-EnvironmentImport $Python 'Python package: faster-whisper (ASR)' 'import faster_whisper'
        Test-EnvironmentImport $Python 'ASR dependency: ONNX Runtime (required by faster-whisper)' 'import onnxruntime'
        Test-EnvironmentImport $Python 'Python package: psutil (resource detection)' 'import psutil'
        Test-EnvironmentImport $Python 'Python package: yt-dlp (online video)' 'import yt_dlp'
        Test-EnvironmentImport $Python 'Python package: PaddlePaddle CPU client' 'import paddle; assert not paddle.is_compiled_with_cuda()'
        Test-EnvironmentImport $Python 'Python package: PaddleOCR-VL client' 'from paddleocr import PaddleOCRVL'
        $DetectorModel = Join-Path $env:PADDLE_PDX_CACHE_HOME 'official_models/PP-OCRv5_mobile_det/inference.pdiparams'
        Write-DependencyStatus (Test-Path $DetectorModel) 'CPU subtitle detector model' $(if (-not (Test-Path $DetectorModel)) { 'Run configure.ps1 --setup to download it.' })
        Test-EnvironmentImport $Python 'Project CLI package' 'import video_subtitles.cli'
    } else {
        foreach ($Name in @('Python pip installer', 'Python package: PyAV (av)', 'Python package: faster-whisper (ASR)', 'Python package: ONNX Runtime (faster-whisper dependency)', 'Python package: psutil', 'Python package: yt-dlp', 'Python package: PaddlePaddle CPU client', 'Python package: PaddleOCR-VL client', 'CPU subtitle detector model', 'Project CLI package')) {
            Write-DependencyStatus $false $Name 'Project Python is unavailable.'
        }
    }
} else {
    foreach ($Name in @('Python pip installer', 'Python package: PyAV (av)', 'Python package: faster-whisper (ASR)', 'Python package: ONNX Runtime (faster-whisper dependency)', 'Python package: psutil (resource detection)', 'Python package: yt-dlp (online video)', 'Python package: PaddlePaddle CPU client', 'Python package: PaddleOCR-VL client', 'CPU subtitle detector model', 'Project CLI package')) {
        Write-DependencyStatus $false $Name 'Environment is not configured.'
    }
}
$GpuTool = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$GpuDetected = $false
if ($GpuTool) {
    & nvidia-smi -L *> $null
    $GpuDetected = ($LASTEXITCODE -eq 0)
}
if ($GpuDetected) {
    Write-DependencyStatus $true 'NVIDIA GPU visible' 'Run the project-local vLLM server in Linux/WSL.'
} else {
    Write-DependencyStatus $true 'CPU OCR route available' 'Run the project-local llama.cpp server from Linux/WSL.'
}
if ($Operation -eq 'check') {
    Stop-ConfigureTranscript
    exit 0
}

if (-not $FfmpegReady -or -not $FfprobeReady) {
    throw 'FFmpeg and FFprobe are required on Windows. Install FFmpeg, add its bin directory to PATH, then rerun configure.ps1.'
}
if (-not $PythonLauncher) { throw 'Python Launcher (py.exe) is required. Install 64-bit Python 3.11 first.' }
& py "-$($Manifest.PYTHON_VERSION)" -c 'import sys; assert sys.maxsize > 2**32'
if ($LASTEXITCODE -ne 0) { throw "Python $($Manifest.PYTHON_VERSION) x64 is not available through py.exe." }
if (-not $AssumeYes) {
    if ([Console]::IsInputRedirected) { throw 'Setup needs interactive confirmation; pass --yes to confirm automatically.' }
    $Answer = Read-Host 'Install or replace the project-local OCR environment? [y/N]'
    if ($Answer -notin @('y', 'Y', 'yes', 'YES')) {
        Write-Host 'Installation cancelled.'
        Stop-ConfigureTranscript
        exit 0
    }
}
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:XDG_CACHE_HOME, $env:HF_HOME, $env:PADDLE_PDX_CACHE_HOME, $env:TORCH_HOME | Out-Null

function New-AppEnvironment([string]$Path, [string]$Name) {
    if (Test-Path $Path) { Remove-Item -Recurse -Force $Path }
    Write-InstallStage "Creating $Name Python environment at $Path (Python $($Manifest.PYTHON_VERSION))."
    & py "-$($Manifest.PYTHON_VERSION)" -m venv $Path
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Failed: creating $Name Python environment."; return $false }
    $Python = Join-Path $Path 'Scripts/python.exe'
    Write-InstallStage "Upgrading pip in the $Name environment."
    & $Python -m pip install --cache-dir $env:PIP_CACHE_DIR --upgrade pip
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Failed: upgrading pip in the $Name environment."; return $false }
    Write-InstallStage "Installing application requirements in the $Name environment (ASR, media and online-video packages)."
    & $Python -m pip install --cache-dir $env:PIP_CACHE_DIR -e $Root
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Failed: installing application requirements in the $Name environment."; return $false }
    Write-Host "[INFO] Completed: preparing the $Name Python environment."
    return $true
}

$OldGpuEnv = Join-Path $Root '.venv-ocr-gpu'
$OldCpuEnv = Join-Path $Root '.venv-ocr-cpu'
if (Test-Path $ClientEnv) { Remove-Item -Recurse -Force $ClientEnv }
if (-not (New-AppEnvironment $ClientEnv 'OCR client')) {
    if (Test-Path $ClientEnv) { Remove-Item -Recurse -Force $ClientEnv }
    throw 'Could not create the OCR client Python environment.'
}
$Python = Join-Path $ClientEnv 'Scripts/python.exe'
Write-InstallStage "Installing CPU PaddlePaddle $($Manifest.PADDLE_VERSION) for the client preprocessing stage."
& $Python -m pip install --cache-dir $env:PIP_CACHE_DIR "paddlepaddle==$($Manifest.PADDLE_VERSION)" -i $Manifest.CPU_INDEX_URL
if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $ClientEnv; throw 'Could not install CPU PaddlePaddle.' }
Write-InstallStage "Installing PaddleOCR-VL $($Manifest.PADDLEOCR_VERSION) client packages."
& $Python -m pip install --cache-dir $env:PIP_CACHE_DIR "paddleocr[doc-parser]==$($Manifest.PADDLEOCR_VERSION)"
if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $ClientEnv; throw 'Could not install PaddleOCR client packages.' }
Write-InstallStage 'Checking PaddleOCR-VL client dependencies.'
& $Python -c 'import paddle, video_subtitles.cli; assert not paddle.is_compiled_with_cuda(); from paddleocr import PaddleOCRVL'
if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $ClientEnv; throw 'OCR client environment verification failed.' }
Write-InstallStage 'Downloading the lightweight CPU subtitle candidate detector.'
& $Python -c 'from paddleocr import TextDetection; TextDetection(model_name="PP-OCRv5_mobile_det", device="cpu", limit_side_len=640, limit_type="max", enable_mkldnn=False)'
if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $ClientEnv; throw 'Could not prepare the CPU subtitle detector model.' }
@("mode=client", "prefix=$ClientEnv") | Set-Content -Encoding ascii $State
if (Test-Path $OldGpuEnv) { Remove-Item -Recurse -Force $OldGpuEnv }
if (Test-Path $OldCpuEnv) { Remove-Item -Recurse -Force $OldCpuEnv }
Write-Host "Selected PaddleOCR-VL client environment: CPU preprocessing ($ClientEnv)"
if (-not $GpuDetected) { Write-Host '[INFO] For local CPU inference, configure and start scripts/setup-llama-server.sh in Linux/WSL.' }
Stop-ConfigureTranscript
