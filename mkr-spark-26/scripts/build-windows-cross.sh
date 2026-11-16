#!/usr/bin/env bash
# Cross-compile MKR-SPARK-26.exe from Linux/macOS with MinGW-w64.
#   sudo apt install gcc-mingw-w64-x86-64      (Debian/Ubuntu)
#   rustup target add x86_64-pc-windows-gnu
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
cargo build --release -p mkr-cli --target x86_64-pc-windows-gnu
mkdir -p dist
cp target/x86_64-pc-windows-gnu/release/MKR-SPARK-26.exe dist/MKR-SPARK-26.exe
cp dist/MKR-SPARK-26.exe dist/mkr.exe
sha256sum dist/MKR-SPARK-26.exe | tee dist/MKR-SPARK-26.exe.sha256
echo "NOTE: this exe was cross-compiled; test it on a real Windows machine before relying on it."
