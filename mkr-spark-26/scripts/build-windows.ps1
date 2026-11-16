<#
.SYNOPSIS
  Builds MKR-SPARK-26.exe (a standalone Windows executable) from source.

.DESCRIPTION
  Requirements (one-time):
    * Rust toolchain:  https://rustup.rs
    * For -Toolchain msvc (default): "Visual Studio Build Tools" with the "Desktop development with C++" workload
      (SQLite is compiled from bundled C source, so a C compiler is required).
    * For -Toolchain gnu: MinGW-w64 (e.g. MSYS2 `pacman -S mingw-w64-x86_64-gcc`) on PATH.
  The produced exe needs no Python, Node.js, Java, Ollama or llama.cpp. HTTPS uses the Windows TLS stack.

.EXAMPLE
  .\scripts\build-windows.ps1
  .\scripts\build-windows.ps1 -Toolchain gnu -SkipTests
#>
param(
    [ValidateSet("msvc", "gnu")] [string]$Toolchain = "msvc",
    [switch]$SkipTests,
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { Fail "cargo not found. Install Rust from https://rustup.rs and reopen the terminal." }
$target = if ($Toolchain -eq "msvc") { "x86_64-pc-windows-msvc" } else { "x86_64-pc-windows-gnu" }
Write-Host "== MKR - SPARK 26 :: building for $target ==" -ForegroundColor Cyan

& rustup target add $target | Out-Null
if ($Toolchain -eq "gnu" -and -not (Get-Command gcc -ErrorAction SilentlyContinue)) { Fail "gcc (MinGW-w64) not found on PATH." }

if (-not $SkipTests) {
    Write-Host "== Running the automated test suite ==" -ForegroundColor Cyan
    & cargo test --release --workspace --target $target
    if ($LASTEXITCODE -ne 0) { Fail "tests failed; refusing to package a build that does not pass its tests (use -SkipTests to override)." }
}

Write-Host "== Building release executable ==" -ForegroundColor Cyan
& cargo build --release -p mkr-cli --target $target
if ($LASTEXITCODE -ne 0) { Fail "build failed." }

$built = Join-Path "target\$target\release" "MKR-SPARK-26.exe"
if (-not (Test-Path $built)) { Fail "expected output $built not found." }

New-Item -ItemType Directory -Force -Path $OutDir, "$OutDir\config", "$OutDir\models", "$OutDir\datasets\seed", "$OutDir\data", "$OutDir\checkpoints" | Out-Null
Copy-Item $built (Join-Path $OutDir "MKR-SPARK-26.exe") -Force
Copy-Item $built (Join-Path $OutDir "mkr.exe") -Force          # short alias: `mkr train`, `mkr chat`, ...
Copy-Item "datasets\seed\*" "$OutDir\datasets\seed" -Force
Copy-Item "README.md", "LICENSE" $OutDir -Force
Copy-Item "docs" "$OutDir\docs" -Recurse -Force

# default configuration written by the program itself, so it always matches the binary
Push-Location $OutDir
& .\MKR-SPARK-26.exe config init --config config\mkr.conf | Out-Null
$ver = & .\MKR-SPARK-26.exe version
Pop-Location

$hash = (Get-FileHash (Join-Path $OutDir "MKR-SPARK-26.exe") -Algorithm SHA256).Hash
Set-Content -Path (Join-Path $OutDir "MKR-SPARK-26.exe.sha256") -Value "$hash  MKR-SPARK-26.exe"
$size = [math]::Round((Get-Item (Join-Path $OutDir "MKR-SPARK-26.exe")).Length / 1MB, 2)

Write-Host ""
Write-Host "OK: $ver  ($size MB)" -ForegroundColor Green
Write-Host "Output : $((Resolve-Path $OutDir).Path)"
Write-Host "SHA-256: $hash"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  cd $OutDir"
Write-Host "  .\mkr.exe train --data datasets\seed --preset micro --steps 300   # demo training on the tiny seed corpus"
Write-Host "  .\mkr.exe chat"
