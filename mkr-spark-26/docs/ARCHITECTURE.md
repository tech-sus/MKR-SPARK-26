# Architecture

## Workspace

| crate | role |
|---|---|
| `spark-tensor` | dense f32 matrices, multithreaded matmul (3 kernels), tape-based reverse-mode autograd, RoPE, PRNG |
| `spark-byte-engine` | byte vocabulary, UTF-8-safe streaming decode, corpus preparation, streaming file-backed batching |
| `spark-optimizer` | AdamW, warmup+cosine schedule, global-norm clipping, serialisable optimiser state |
| `spark-neural` | SPARK Neural Architecture (SNA), presets, `.model` format with CRC-32 |
| `spark-training` | training loop, validation, checkpoints/resume, overfitting detection, RAM planning, memory measurement |
| `spark-inference` | O(1)-state incremental decoder, int8 quantisation, sampler, confidence statistics |
| `mkr-core` | config, errors, Arabic/English text normalisation, dates, trust priors, SQLite open/recover |
| `mkr-memory` | conversations, user-approved memories (FTS5), preferences, search history |
| `mkr-knowledge` | verified facts + sources, confidence, expiry, supersession, user correction |
| `mkr-search` | SSRF-safe HTTP client, HTML extraction, providers, fusion ranking, cache, citations |
| `mkr-verification` | cross-source agreement, contradiction/freshness checks, prompt-injection filter |
| `mkr-learning` | verified-knowledge ingestion, memory proposals, review queue, guarded fine-tuning + rollback |
| `mkr-cli` | the `MKR-SPARK-26` / `mkr` executable and the answer pipeline |

Only third-party crates: `rayon` (threads), `rusqlite` (bundled SQLite), `ureq` (+ rustls on Linux/macOS,
Windows TLS on Windows), `serde`/`serde_json`, `url`. No ML framework, no pretrained weights, no inference runtime.

## SPARK Neural Architecture (SNA)

Input: bytes (vocab 264 = 256 bytes + PAD/BOS/EOS/USER/ASSISTANT/SYSTEM + padding to a multiple of 8).

Each of `n_layers` pre-norm residual blocks:

1. **SPARK-Scan** – `a = σ(h·Wa + ba)`, `v = h·Wv`, `s_t = a_t ⊙ s_{t-1} + (1−a_t) ⊙ v_t`, output `(s ⊙ silu(h·Wg))·Wo`.
   Diagonal gated linear recurrence; the decay bias `ba` is initialised multi-timescale
   (memory from ~2 to ~1000 tokens). Hand-written forward *and* backward (reverse scan).
2. **Sliding-window causal attention** with RoPE, window `W` (0 disables it → recurrence-only ablation).
3. **SwiGLU** feed-forward (hidden ≈ 8/3·d).

Final RMSNorm, linear head to 264 logits. Loss: next-byte cross-entropy (padding masked).

### Established vs. original

*Established techniques used as-is:* RMSNorm, RoPE, SwiGLU, sliding-window attention, diagonal linear recurrences
(S4 / LRU / Hawk / minGRU family), pre-norm residual blocks, AdamW, cosine schedule, int8 per-column quantisation,
byte-level modelling.

*Our contribution (modest, and not claimed as a breakthrough):* the specific composition above — the `(1−a)`-coupled gated scan
with multi-timescale initialisation feeding a SiLU output gate, placed *before* attention in each block so the window attention
sees context-aware features — plus a single definition shared by training and inference, verified numerically identical
(`max |engine − graph| logit difference = 1.3e-7`, including when the attention ring buffer wraps). No claim is made that SNA
outperforms existing architectures; no such comparison has been run.

### Why O(1) state at inference
Per layer the engine stores one `d`-vector (scan state) and a `W×d` key/value ring buffer. Memory does not grow with
generated length (`mkr model info` prints it; 17 KB for `nano`, 133 KB for `micro`).

### Byte-level trade-offs (measured/known, not assumed)
* No OOV, no tokenizer, trivial Unicode safety. Arabic letters are 2 bytes in UTF-8, so a 512-byte context holds
  roughly 256 Arabic characters.
* Sequences are ~3–5× longer than with a subword tokenizer, so per-character compute is higher. Byte-level modelling does **not**
  automatically give a smaller or faster model; the embedding table is tiny (264×d) but compute per character is larger.

## Presets

| preset | d | layers | heads | window | params | intended RAM |
|---|---|---|---|---|---|---|
| nano | 64 | 2 | 2 | 16 | 0.17 M | tests |
| micro | 128 | 4 | 4 | 32 | 1.1 M | 2 GB (experimental) |
| mini | 256 | 6 | 4 | 64 | 6.5 M | 4 GB |
| small | 384 | 8 | 6 | 64 | 19 M | 8 GB |
| base | 512 | 12 | 8 | 96 | 51 M | 8–16 GB |
| large | 768 | 12 | 12 | 128 | 114 M | 16 GB |

All dimensions are configurable through `spark_neural::Config`.

## Not implemented (stated plainly)
* **Gradient checkpointing** is *not* implemented. Memory is controlled by gradient accumulation (`--accum`), lazy
  gradient allocation in the autograd tape, and smaller batch/context. `mkr plan` reports the resulting estimates.
* No GPU path. CPU only, by design.
* Matmul is a portable blocked-free kernel compiled for baseline x86-64 (no AVX-specific intrinsics), measured at ~16–18 GFLOP/s
  on one core of the build machine. Faster kernels would help `base`/`large` considerably.
