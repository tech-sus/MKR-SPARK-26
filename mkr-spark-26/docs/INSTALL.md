# Installation

## Option A – use the prebuilt executable
`dist/MKR-SPARK-26.exe` (also `mkr.exe`, same program) is a single native Windows x86-64 file (≈3.7 MB). It imports only Windows system DLLs
(kernel32, ws2_32, crypt32, secur32, …): no Python, Node.js, Java, Ollama, llama.cpp or C++ runtime is needed. SQLite is compiled in;
HTTPS uses the Windows TLS stack.

```
cd dist
mkr.exe help
mkr.exe data seed
mkr.exe train --preset micro --steps 300      # demo run on the 6 KB seed corpus
mkr.exe chat
```
For Arabic output use **Windows Terminal** (or a console font with Arabic glyphs). The program switches the console code page to UTF-8,
but the legacy console host does not shape/right-to-left Arabic text; the *data* is correct even when the display is not.

> **Honesty note on testing:** the exe was cross-compiled on Linux (MinGW-w64) and exercised under Wine (version, plan, training with
> checkpoints and Windows memory measurement, model verification, SQLite memory with Arabic text, offline chat). It has **not** been run on
> a physical Windows machine. Please verify it on yours; the SHA-256 is in `MKR-SPARK-26.exe.sha256`.

## Option B – build on Windows with the PowerShell script
1. Install Rust (https://rustup.rs) and *Visual Studio Build Tools → Desktop development with C++* (needed to compile bundled SQLite).
2. Open PowerShell in the project root:
   ```
   .\scripts\build-windows.ps1              # runs the whole test-suite, then builds, then packages .\dist
   .\scripts\build-windows.ps1 -SkipTests   # build only
   .\scripts\build-windows.ps1 -Toolchain gnu   # MinGW-w64 instead of MSVC
   ```
   The script refuses to package a build whose tests fail. Output: `dist\MKR-SPARK-26.exe`, `dist\mkr.exe`, default `config\mkr.conf`, docs, SHA-256.

## Option C – build/cross-compile on Linux/macOS
```
cargo build --release -p mkr-cli                 # native binary: target/release/MKR-SPARK-26
./scripts/build-windows-cross.sh                 # needs gcc-mingw-w64-x86-64 and `rustup target add x86_64-pc-windows-gnu`
cargo test --release --workspace                 # full automated test-suite
```
Requires Rust ≥ 1.80 (developed with 1.100). Release profile: `opt-level 3`, fat LTO, one codegen unit, stripped.

## Model file
Weights live in a separate file, `models/MKR-SPARK-26.model` (path configurable). `mkr train` creates it. `mkr model quantize` writes an int8 variant
(`*.int8.model`, ~3× smaller, ~1 % logit error) that `chat`/`generate` load transparently. Embedding weights into the exe is *not* implemented
(the requirement marks it optional/later).

## Configuration
`mkr config` lists all keys; `mkr config set <key> <value>` persists to `config/mkr.conf`. Notable keys: `online`, `providers`, `searxng_url`,
`brave_api_key`, `timeout_secs`, `retries`, `threads`, `knowledge_ttl_days`, `trust_model_answers`, `model_confidence_threshold`.
