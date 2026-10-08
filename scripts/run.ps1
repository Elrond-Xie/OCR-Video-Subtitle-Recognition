$ErrorActionPreference = 'Stop'
$Arguments = $args
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$State = Join-Path $Root '.ocr-environment'
if (-not (Test-Path $State)) { throw 'OCR environment is not configured. Run scripts/setup.ps1 first.' }
$Values = ConvertFrom-StringData (Get-Content $State -Raw)
$Executable = Join-Path $Values.prefix 'Scripts/subtitles.exe'
if (-not (Test-Path $Executable)) { throw 'The selected OCR environment is missing. Rerun scripts/setup.ps1.' }
& $Executable @Arguments
exit $LASTEXITCODE
