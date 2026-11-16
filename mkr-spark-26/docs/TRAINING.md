# Training

```
mkr data seed                       # writes the tiny bilingual demo corpus (≈6 KB) to datasets/seed
mkr data prepare mytexts/ --out datasets/corpus.bin
mkr train --data datasets/corpus.bin --ram 8 --steps 20000
mkr train --resume --steps 40000    # continue from the newest valid checkpoint
```

* Weights start random (seeded). Nothing is downloaded.
* Corpus files are streamed from disk with random seeks, so RAM does not depend on corpus size.
* Validation is the final `--val` fraction (default 10 %) of the corpus, evaluated deterministically on non-overlapping windows.
  Reported as loss, perplexity and **bits per byte** (tokenizer-independent).
* Checkpoints (`checkpoints/step_N.ckpt`) contain model + AdamW state + step + history, CRC-checked, written atomically,
  newest 3 kept. A corrupt newest checkpoint is skipped automatically on resume.
  Resume is bit-exact: a run interrupted at step 20 and resumed produces identical parameters to an uninterrupted run
  (`max |Δparam| = 0`, automated test).
* `best.model` always holds the lowest-validation-loss weights and is what `mkr train` installs as `models/MKR-SPARK-26.model`.
* **Overfitting detection:** if validation loss has not improved for `patience` evaluations while training loss keeps
  falling, training reports it and (by default) stops. Demonstrated on the seed corpus: validation loss best 2.58 at step 100,
  training loss 0.15 and validation 5.1 at step 400 → detected.
* Divergence (NaN/inf) aborts with a message instead of silently continuing.

## Memory tiers (`mkr plan`)

| RAM | preset | batch × ctx × accum | estimated training memory |
|---|---|---|---|
| 2 GB | micro | 4 × 128 × 4 | 72 MB |
| 4 GB | mini | 4 × 256 × 4 | 424 MB |
| 8 GB | small | 4 × 384 × 4 | 1.3 GB |
| 16 GB | base | 8 × 512 × 2 | 6.0 GB |

The estimator was checked against measured tape memory for `micro` (72 MB estimated vs 58 MB measured). Estimates are
deliberately conservative; large-preset figures were not measured on the build machine.

## What to expect
A from-scratch model of 10–100 M parameters trained on a CPU, even on a good corpus, learns spelling, local syntax and
short-range patterns; it will **not** answer factual questions like a large language model. That is why factual answers
come from verified retrieval (see `KNOWLEDGE.md`) and why the model's own answers are labelled "unverified" and disabled
by default (`trust_model_answers = false`).

Throughput measured on the build machine (1 CPU core): `nano` ≈ 12 000 tokens/s, `micro` ≈ 2 000 tokens/s training
(batch 4 × ctx 128). Training `small` or larger to a useful loss needs many CPU-hours/days and a real corpus (hundreds of MB);
that has **not** been done here and no quality claim is made for those sizes.
