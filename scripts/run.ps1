$ErrorActionPreference = 'Stop'
$Arguments = $args
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Manifest = ConvertFrom-StringData (Get-Content (Join-Path $Root 'environment/ocr-versions.env') -Raw)
$ConfiguredDownloadDir = $Manifest.DOWNLOAD_DIR.Trim().Trim('"').Trim("'")
$ConfiguredLogDir = $Manifest.LOG_DIR.Trim().Trim('"').Trim("'")
if ([System.IO.Path]::IsPathRooted($ConfiguredDownloadDir)) {
    $DownloadRoot = [System.IO.Path]::GetFullPath($ConfiguredDownloadDir)
} else {
    $DownloadRoot = [System.IO.Path]::GetFullPath((Join-Path $Root $ConfiguredDownloadDir))
}
$env:DOWNLOAD_DIR = $DownloadRoot
$env:LOG_DIR = if ([System.IO.Path]::IsPathRooted($ConfiguredLogDir)) {
    [System.IO.Path]::GetFullPath($ConfiguredLogDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $Root $ConfiguredLogDir))
}
$env:SAVE_LOGS = if ($Manifest.ContainsKey('SAVE_LOGS')) { $Manifest.SAVE_LOGS.Trim().Trim('"').Trim("'") } else { 'true' }
$env:PIP_CACHE_DIR = Join-Path $DownloadRoot 'pip'
$env:XDG_CACHE_HOME = Join-Path $DownloadRoot 'xdg'
$env:HF_HOME = Join-Path $DownloadRoot 'huggingface'
$env:HF_HUB_CACHE = Join-Path $DownloadRoot 'huggingface/hub'
$env:PADDLE_PDX_CACHE_HOME = Join-Path $DownloadRoot 'paddlex'
$env:TORCH_HOME = Join-Path $DownloadRoot 'torch'
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:XDG_CACHE_HOME, $env:HF_HOME, $env:PADDLE_PDX_CACHE_HOME, $env:TORCH_HOME | Out-Null
$State = Join-Path $Root '.ocr-environment'
if (-not (Test-Path $State)) { throw 'OCR environment is not configured. Run scripts/configure.ps1 first.' }
$Values = ConvertFrom-StringData (Get-Content $State -Raw)
$Executable = Join-Path $Values.prefix 'Scripts/subtitles.exe'
if (-not (Test-Path $Executable)) { throw 'The selected OCR environment is missing. Rerun scripts/configure.ps1.' }
& $Executable @Arguments
exit $LASTEXITCODE
