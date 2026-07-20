<#
.SYNOPSIS
    Stage everything the Windows installer needs into installer/staging/.

.DESCRIPTION
    Builds the WPF studio, gathers the already-built C++ pipeline (exe + DLLs),
    the models, the example config, docs and the mock server into a single
    self-contained tree that Inno Setup (anpr.iss) then packages.

    Build the C++ pipeline first:
        cd cmake/anpr; cmake --preset x64-release; cmake --build out/build/x64-release
    And fetch models:
        cd cmake/anpr/models; python prepare_models.py --vehicle

.EXAMPLE
    cd installer; ./package.ps1
#>
[CmdletBinding()]
param(
    [string]$Configuration = "x64-release"
)
$ErrorActionPreference = "Stop"

$repo    = Split-Path -Parent $PSScriptRoot
$anpr    = Join-Path $repo "cmake/anpr"
$control = Join-Path $anpr "out/build/$Configuration/control"
$mock    = Join-Path $anpr "out/build/$Configuration/tools/mock_server/mock_server.exe"
$models  = Join-Path $anpr "models"
$studio  = Join-Path $repo "proj/AnprStudio"
$staging = Join-Path $PSScriptRoot "staging"

if (-not (Test-Path (Join-Path $control "anpr.exe"))) {
    throw "anpr.exe not found. Build first: cmake --build out/build/$Configuration"
}

Write-Host "Cleaning staging..." -ForegroundColor Cyan
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
$pipeline  = New-Item -ItemType Directory -Force -Path (Join-Path $staging "pipeline")
$studioOut = New-Item -ItemType Directory -Force -Path (Join-Path $staging "studio")

# 1) Pipeline: anpr.exe + all runtime DLLs (vcpkg already placed them here).
#    Only executables and libraries -- skip CMake build artifacts.
Write-Host "Staging pipeline (exe + DLLs)..." -ForegroundColor Cyan
Get-ChildItem $control -File | Where-Object { $_.Extension -in ".exe", ".dll" } |
    Copy-Item -Destination $pipeline -Force
if (Test-Path $mock) { Copy-Item $mock -Destination $pipeline -Force }

# 2) Config: ship the sanitized example AS anpr.json so it runs out of the box.
$cfgDir = New-Item -ItemType Directory -Force -Path (Join-Path $pipeline "config")
$example = Join-Path $anpr "config/anpr.example.json"
Copy-Item $example -Destination (Join-Path $cfgDir "anpr.json") -Force
Copy-Item $example -Destination $cfgDir -Force

# 3) Models: ONNX + charset files.
$mdlDir = New-Item -ItemType Directory -Force -Path (Join-Path $pipeline "models")
$modelFiles = Get-ChildItem $models -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -eq ".onnx" -or $_.Name -like "charset*.txt" }
if ($null -eq $modelFiles) {
    Write-Warning "No models found in $models. Run models/prepare_models.py first."
} else {
    $modelFiles | Copy-Item -Destination $mdlDir -Force
}

# 4) ANPR Studio: self-contained publish so end users need no .NET install.
Write-Host "Publishing ANPR Studio (self-contained)..." -ForegroundColor Cyan
dotnet publish $studio -c Release -r win-x64 --self-contained true -p:PublishSingleFile=false -o $studioOut | Out-Null

# 5) Docs.
Copy-Item (Join-Path $repo "readme.md") -Destination (Join-Path $staging "README.md") -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $repo "LICENSE") -Destination $staging -Force
Copy-Item (Join-Path $repo "THIRD_PARTY_NOTICES.md") -Destination $staging -Force

$mb = "{0:N1}" -f ((Get-ChildItem $staging -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "Staging complete: $staging ($mb MB)" -ForegroundColor Green
Write-Host 'Next: iscc anpr.iss' -ForegroundColor Green
