<div align="center">
  <h1>MKR · SPARK 26</h1>
  <h3>Small Core. Infinite Knowledge.</h3>
  <p>
    <strong>An independent, CPU-first AI research prototype built from scratch with Rust.</strong>
  </p>

  <p>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Language-Rust-orange?style=for-the-badge&logo=rust&logoColor=white" alt="Rust"></a>
    <img src="https://img.shields.io/badge/Inference-CPU--first-555?style=for-the-badge" alt="CPU first">
    <img src="https://img.shields.io/badge/Pretrained%20weights-None-555?style=for-the-badge" alt="No pretrained weights">
    <img src="https://img.shields.io/badge/Status-Research%20prototype-yellow?style=for-the-badge" alt="Research prototype">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License"></a>
  </p>

  <p>
    <a href="#-overview">Overview</a> ·
    <a href="#-features">Features</a> ·
    <a href="#-quick-start">Quick start</a> ·
    <a href="#-training">Training</a> ·
    <a href="#-architecture">Architecture</a> ·
    <a href="#-documentation">Documentation</a>
  </p>
</div>

---

## ✦ Overview

**MKR-SPARK 26** is an experimental AI system designed around a simple idea: build a compact neural model and its supporting infrastructure independently, using Rust and CPU execution.

The repository includes a byte-level neural model, a tensor/autograd engine, a training pipeline, CPU inference, local memory, a knowledge store, and a web retrieval subsystem with source checking.

> [!IMPORTANT]
> **This is a research prototype, not a ChatGPT replacement.** The model starts with randomly initialized weights and has no pretrained knowledge. A small model trained on a CPU learns limited text patterns; it should not be expected to produce GPT-class reasoning or factual answers. Factual responses are designed to rely on retrieval and verification, not on the model weights alone.

<div align="center">
  <table>
    <tr>
      <td align="center"><strong>Rust</strong><br>Core implementation</td>
      <td align="center"><strong>CPU</strong><br>Local training & inference</td>
      <td align="center"><strong>SQLite</strong><br>Persistent local memory</td>
      <td align="center"><strong>Byte-level</strong><br>Text processing</td>
    </tr>
  </table>
</div>

## ✦ Features

| Component | What it does |
|---|---|
| **Tensor & Autograd Engine** | Tensor operations and automatic differentiation implemented in Rust. |
| **SPARK Neural Architecture** | A compact byte-level neural architecture with configurable model presets. |
| **Training Engine** | Training from random initialization, validation, checkpointing, and resume support. |
| **CPU Inference** | Local text generation without requiring a GPU runtime. |
| **Model Serialization** | Model files and checkpoints with integrity checks. |
| **Local Memory** | SQLite-backed memory with full-text search support. |
| **Knowledge Store** | Store, retrieve, and manage sourced knowledge locally. |
| **Web Search & Verification** | Retrieve information from configured providers and compare sources. |
| **Controlled Learning** | A guarded learning workflow with approval and rollback mechanisms. |
| **Windows Executable** | A standalone Windows x86-64 executable is included in `dist/`. |

## ✦ What makes it different?

- **Built in Rust:** the core model and supporting systems are implemented in a Rust workspace.
- **No pretrained weights:** the neural model is initialized locally and trained from text supplied by the user.
- **CPU-first design:** local training and inference do not require a dedicated GPU.
- **Local by default:** memory and model files are stored on the machine.
- **Retrieval-aware answers:** the assistant can use stored knowledge and configured web sources instead of pretending that a tiny model knows everything.
- **Arabic and English groundwork:** the byte-level pipeline includes Unicode handling and bilingual seed text.

## ✦ Quick start

### Requirements

- Rust toolchain (Rust 1.80 or newer; the project was developed with a newer toolchain).
- A C/C++ build toolchain on Windows for bundled native dependencies.
- Windows Terminal is recommended for Arabic console output on Windows.

### Build from source

```bash
cargo build --release -p mkr-cli
```

Run the command-line help:

```bash
# Linux / macOS
./target/release/MKR-SPARK-26 help

# Windows PowerShell
.\target\release\MKR-SPARK-26.exe help
```

### Try the included demo corpus

```bash
mkr data seed
mkr train --preset micro --steps 300
mkr model info
mkr chat
```

If the `mkr` command is not installed globally, replace `mkr` with the executable path shown above.

### Use the prebuilt Windows executable

The repository includes:

```text
dist/
├── MKR-SPARK-26.exe
├── mkr.exe
├── MKR-SPARK-26.exe.sha256
├── config/
├── datasets/
└── docs/
```

From PowerShell:

```powershell
cd dist
.\MKR-SPARK-26.exe help
.\MKR-SPARK-26.exe data seed
.\MKR-SPARK-26.exe train --preset micro --steps 300
.\MKR-SPARK-26.exe model info
```

> [!NOTE]
> The included executable was cross-compiled on Linux and exercised under Wine. It has **not** yet been verified on a physical Windows installation. Please test it on your own system and report any issues.

## ✦ Training

MKR-SPARK 26 trains from randomly initialized weights. No model weights are downloaded automatically.

### 1. Prepare your text corpus

Put `.txt` files into a folder, for example:

```text
mytexts/
├── arabic/
│   ├── text-01.txt
│   └── text-02.txt
└── english/
    └── text-01.txt
```

Prepare the corpus:

```bash
mkr data prepare mytexts/ --out datasets/corpus.bin
```

### 2. Start training

```bash
mkr train --data datasets/corpus.bin --ram 8 --steps 20000
```

### 3. Resume from a checkpoint

```bash
mkr train --resume --steps 40000
```

### Training notes

- The bundled bilingual seed corpus is only a tiny demonstration corpus (approximately 6 KB); it is not sufficient for meaningful language-model training.
- Training data is streamed from disk, so the full corpus does not need to fit in RAM.
- Validation metrics include loss, perplexity, and bits per byte.
- Checkpoints preserve model and optimizer state and support resuming.
- The trainer includes checks for overfitting and numerical divergence.
- A useful corpus should be substantially larger, carefully cleaned, legally usable, and free of duplicated or low-quality material.

### Memory-based presets

The training guide provides these suggested configurations:

| Available RAM | Suggested preset | Estimated training memory |
|---:|---|---:|
| 2 GB | `micro` | 72 MB |
| 4 GB | `mini` | 424 MB |
| 8 GB | `small` | 1.3 GB |
| 16 GB | `base` | 6.0 GB |

These are estimates, not guarantees. Actual usage depends on the operating system, other running applications, corpus settings, and training configuration. Start with a small run and monitor memory and temperature.

> [!WARNING]
> CPU training can take a long time. In the project's recorded benchmark environment, the `small` preset was measured for only one training step; meaningful training at that size was not completed. Larger presets are implemented and shape-tested, but should not be treated as trained or quality-validated models.

## ✦ Architecture

The project is organized as a Cargo workspace with separate crates for the model, training, inference, storage, retrieval, and command-line interface.

```text
mkr-spark-26/
├── crates/
│   ├── spark-tensor/       # Tensor operations and autograd
│   ├── spark-neural/       # SPARK model architecture
│   ├── spark-training/    # Training loop and checkpoints
│   ├── spark-inference/   # CPU inference and generation
│   ├── spark-byte-engine/ # Byte-level corpus and streaming
│   ├── spark-optimizer/   # Optimization algorithms
│   ├── mkr-core/          # Shared types, config and utilities
│   ├── mkr-memory/        # Local memory
│   ├── mkr-knowledge/     # Knowledge storage
│   ├── mkr-search/        # Search and retrieval
│   ├── mkr-verification/  # Source checking and sanitization
│   ├── mkr-learning/      # Controlled learning
│   └── mkr-cli/           # Command-line application
├── config/                # Runtime configuration
├── datasets/              # Seed and user-prepared datasets
├── models/                # Generated model files
├── checkpoints/           # Training checkpoints
├── docs/                  # Technical documentation
├── benchmarks/            # Recorded benchmark notes
├── scripts/               # Build scripts
└── dist/                  # Packaged Windows build
```

### High-level data flow

```text
                 ┌─────────────────────┐
                 │     User prompt    │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │    Local memory     │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Stored knowledge   │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   Model confidence  │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Verified web search │
                 │   (when enabled)    │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Answer + sources   │
                 └─────────────────────┘
```

The exact path depends on configuration and available sources. Web providers may rate-limit, change their HTML, or be unavailable.

## ✦ Project status

The repository contains implementation and automated tests for the following areas. The status below reflects the project's included documentation and test notes; it is not a claim that the system is production-ready.

| Area | Recorded status |
|---|---|
| Workspace and tensor engine | Implemented; numerical gradient checks documented |
| SPARK neural architecture | Implemented; shape and behavior tests included |
| Byte-level text processing | Implemented; Unicode and streaming tests included |
| Training from scratch | Implemented; seed-corpus training tests included |
| Model serialization | Implemented; integrity and corruption checks included |
| CPU inference | Implemented; inference tests included |
| Windows executable | Packaged; physical Windows testing still needed |
| Local memory | Implemented; SQLite tests included |
| Web search | Implemented; provider behavior tested with fixtures/mock server |
| Knowledge verification | Implemented; heuristic/lexical verification |
| Controlled learning | Implemented; approval and rollback tests included |
| Performance optimization | Partial; no GPU backend or SIMD intrinsics |

Run the complete test suite:

```bash
cargo test --release --workspace
```

## ✦ Benchmarks

The project's benchmark notes were recorded on a Linux build environment with **one CPU core and 4 GB RAM**. They are reference measurements, not expected performance on every machine.

| Metric | `nano` (0.17M parameters) | `micro` (1.1M parameters) |
|---|---:|---:|
| Training throughput | ~12,000 tokens/s | ~2,000 tokens/s |
| Inference, f32 | ~29,000 tokens/s | ~3,200 tokens/s |
| Inference, int8 | ~24,000 tokens/s | ~4,900 tokens/s |
| f32 model weights | 670 KB | 4.5 MB |
| int8 model weights | 228 KB | 1.3 MB |
| Peak process RAM | 13 MB | 61 MB |

Benchmark methodology and additional notes are available in [`benchmarks/`](benchmarks/).

## ✦ Configuration

The default configuration is located at:

```text
config/mkr.conf
```

View available settings:

```bash
mkr config
```

Examples:

```bash
mkr config get online
mkr config set online false
mkr config path
```

Some settings control online retrieval, providers, timeouts, retries, thread count, knowledge freshness, and whether model-only answers are trusted. Model-only answers are disabled by default for safety and quality reasons.

## ✦ Documentation

| Document | Description |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Internal architecture and model design |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Dataset preparation, training, validation, and checkpoints |
| [`docs/KNOWLEDGE.md`](docs/KNOWLEDGE.md) | Knowledge retrieval and verification behavior |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Windows, Linux, and macOS build instructions |
| [`benchmarks/`](benchmarks/) | Recorded performance measurements |

## ✦ Known limitations

- The model has no pretrained weights and must be trained on a suitable corpus.
- A small CPU-trained model is not expected to match large language models.
- The included seed dataset is a demonstration dataset, not a production training corpus.
- Knowledge verification uses lexical and heuristic checks; agreement between sources does not guarantee truth.
- Web search depends on external providers and network availability.
- The Windows executable has not yet been tested on physical Windows hardware.
- Performance work is incomplete: there is no dedicated GPU backend, SIMD implementation, or gradient checkpointing.

## ✦ Roadmap

- [ ] Validate the packaged executable on physical Windows 10/11 systems.
- [ ] Expand and document the Arabic/English training corpus.
- [ ] Run reproducible training experiments on consumer CPUs.
- [ ] Publish model-quality evaluation results.
- [ ] Improve performance and memory efficiency.
- [ ] Expand retrieval-provider reliability and verification tests.
- [ ] Document reproducible benchmark methodology.

## ✦ Contributing

Contributions, bug reports, testing feedback, and documentation improvements are welcome.

1. Fork the repository.
2. Create a focused branch.
3. Make your changes and add or update tests.
4. Run `cargo test --release --workspace`.
5. Open a pull request describing the change and how it was tested.

Please avoid submitting copyrighted or private training data without the right to redistribute it.

## ✦ License

This project is distributed under the license in [`LICENSE`](LICENSE).

---

<div align="center">
  <h3>MKR · SPARK 26</h3>
  <p><em>Small Core. Infinite Knowledge.</em></p>
  <sub>Built with Rust · Designed for local experimentation</sub>
</div>
