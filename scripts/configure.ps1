param()
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Manifest = ConvertFrom-StringData (Get-Content (Join-Path $Root 'environment/ocr-versions.env') -Raw)
$State = Join-Path $Root '.ocr-environment'
$GpuEnv = Join-Path $Root '.venv-ocr-gpu'
$CpuEnv = Join-Path $Root '.venv-ocr-cpu'

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'FFmpeg is not on PATH. Subtitle extraction and video decoding will require FFmpeg.'
}
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Warning 'FFprobe is not on PATH. Media inspection requires FFprobe.'
}

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PythonLauncher) { throw 'Python Launcher (py.exe) is required. Install 64-bit Python 3.11 first.' }
& py "-$($Manifest.PYTHON_VERSION)" -c 'import sys; assert sys.maxsize > 2**32'
if ($LASTEXITCODE -ne 0) { throw "Python $($Manifest.PYTHON_VERSION) x64 is not available through py.exe." }

function New-AppEnvironment([string]$Path) {
    if (Test-Path $Path) { Remove-Item -Recurse -Force $Path }
    & py "-$($Manifest.PYTHON_VERSION)" -m venv $Path
    if ($LASTEXITCODE -ne 0) { return $false }
    $Python = Join-Path $Path 'Scripts/python.exe'
    & $Python -m pip install --upgrade pip | Out-Host
    if ($LASTEXITCODE -ne 0) { return $false }
    & $Python -m pip install -e $Root | Out-Host
    return ($LASTEXITCODE -eq 0)
}

$GpuReady = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $GpuInfo = & nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>$null | Select-Object -First 1
    if ($GpuInfo) {
        Write-Host "NVIDIA GPU detected: $GpuInfo"
        if (New-AppEnvironment $GpuEnv) {
            $Python = Join-Path $GpuEnv 'Scripts/python.exe'
            & $Python -m pip install "paddlepaddle-gpu==$($Manifest.PADDLE_VERSION)" -i $Manifest.CUDA_INDEX_URL
            if ($LASTEXITCODE -eq 0) {
                & $Python -m pip install "paddleocr[doc-parser]==$($Manifest.PADDLEOCR_VERSION)"
                if ($LASTEXITCODE -eq 0) {
                    & $Python -c 'import paddle; assert paddle.device.cuda.device_count() > 0'
                    if ($LASTEXITCODE -eq 0) {
                        & $Python -c 'from paddleocr import PaddleOCRVL; PaddleOCRVL(device="gpu:0")'
                        $GpuReady = ($LASTEXITCODE -eq 0)
                    }
                }
            }
        }
    }
}

if ($GpuReady) {
    @("mode=gpu", "prefix=$GpuEnv") | Set-Content -Encoding ascii $State
    Write-Host "Selected OCR environment: GPU ($GpuEnv)"
} else {
    if (Test-Path $GpuEnv) { Remove-Item -Recurse -Force $GpuEnv }
    Write-Host 'GPU setup or initialization was unavailable; creating a clean CPU environment.'
    if (-not (New-AppEnvironment $CpuEnv)) {
        if (Test-Path $CpuEnv) { Remove-Item -Recurse -Force $CpuEnv }
        throw 'Could not create the CPU Python environment.'
    }
    $Python = Join-Path $CpuEnv 'Scripts/python.exe'
    & $Python -m pip install "paddlepaddle==$($Manifest.PADDLE_VERSION)" -i $Manifest.CPU_INDEX_URL
    if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $CpuEnv; throw 'Could not install CPU PaddlePaddle.' }
    & $Python -m pip install "paddleocr[doc-parser]==$($Manifest.PADDLEOCR_VERSION)"
    if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $CpuEnv; throw 'Could not install PaddleOCR.' }
    & $Python -c 'import paddle, rapidocr; import video_subtitles.cli'
    if ($LASTEXITCODE -ne 0) { Remove-Item -Recurse -Force $CpuEnv; throw 'CPU OCR environment verification failed.' }
    @("mode=cpu", "prefix=$CpuEnv") | Set-Content -Encoding ascii $State
    Write-Host "Selected OCR environment: CPU ($CpuEnv)"
}
