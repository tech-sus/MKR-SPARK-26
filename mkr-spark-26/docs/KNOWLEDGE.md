# Knowledge, memory, search and learning

## Four layers, never mixed silently
| layer | what | stored in | label in answers |
|---|---|---|---|
| model weights | what the network generates | `.model` file | "model weights (unverified)" |
| retrieved knowledge | verified facts with sources | `data/knowledge.db` | "retrieved knowledge" |
| user memory | things *you* approved | `data/memory.db` | "user memory" |
| conversation context | the current session | in memory / `messages` table | "conversation context" |

**Searching the internet never trains the network.** Weights change only through `mkr learn finetune` (below).

## Answer pipeline (`mkr chat`, `mkr search`)
1. Message handling: `remember …` commands, first-person questions → user memory.
2. "I will check my internal knowledge first": stored knowledge (FTS5, Arabic/English normalisation). Time-sensitive questions
   with a stored answer older than 7 days are refreshed.
3. The internal model's confidence is *evaluated* (mean log-probability / entropy of its greedy answer). It answers only if
   `trust_model_answers = true` **and** confidence ≥ `model_confidence_threshold`, and the answer is labelled unverified.
4. "Searching trusted sources…": providers → fusion ranking (reciprocal-rank fusion + trust prior + query overlap) →
   near-duplicate merge → fetch top pages (cached) → extraction.
5. Verification → cited answer → optional storage.

## Search providers (all optional/configurable: `mkr config set providers wikipedia,duckduckgo,searxng,brave`)
* **wikipedia** – MediaWiki API, free, Arabic and English (language chosen from the query).
* **duckduckgo** – HTML endpoint, free, no key; can rate-limit/bot-check (reported as such, never silently empty).
* **searxng** – your own instance (`searxng_url`), may be on a private network (only that host is exempted from the SSRF filter).
* **brave** – needs `brave_api_key`; unavailable without it. No provider is assumed unlimited; a failing provider never blocks others.

Timeouts, bounded retries with backoff (5xx/429/transport only), 2 MB body cap (after decompression), content-type allow-list,
≤5 redirects each re-validated, windows-1256/Latin-1/UTF-8 decoding, SQLite cache with TTL.

**Verification status of the providers:** the HTTP stack is tested end to end against a local mock server, and the four provider
parsers are tested against fixtures shaped like the documented responses. The sandbox this was developed in cannot reach the live
services, so live behaviour (especially DuckDuckGo's HTML markup, which changes) has **not** been confirmed. If a provider's
format changes, it reports "markup may have changed" and other providers keep working.

## Verification (`mkr-verification`)
* A claim is **Verified** only with ≥ 2 *independent origins*: different registrable domains whose sentences are not near-verbatim
  copies. Same-domain pages and copied text count once.
* **Conflicting**: sources that disagree on figures (query numbers excluded) or polarity (negation, English/Arabic) → both sides
  shown, nothing stored.
* **Freshness**: for time-sensitive questions (latest, current, price, version, 2025+ years, Arabic equivalents) old or undated
  sources lower confidence and are noted; a newer disagreeing source is pointed out.
* **Prompt-injection defence**: sentences that instruct an AI/agent or the shell (EN/AR patterns, blobs, link spam) are removed,
  the source's trust is halved and the user is told. The pipeline is purely extractive — web text is data, is never executed and
  is never fed to the model as a prompt.
* Heuristic limits (be aware): agreement is lexical (token overlap, light Arabic/English stemming), not semantic; paraphrases with
  little shared vocabulary can be missed (→ SingleSource), and two sentences that share the question's words plus one entity can
  occasionally be merged though they differ in meaning. Treat "verified" as "two independent sources said the same thing", not as truth.

## Learning
* **A. Knowledge**: only *Verified* claims (confidence ≥ 0.6) are stored automatically, with per-source excerpts, an expiry
  (`knowledge_ttl_days`) and merged duplicates. Single-source claims need your approval and are capped at 0.45 confidence;
  conflicting/insufficient evidence is never stored. Inspect/correct/delete: `mkr knowledge …` (a corrected fact becomes
  user-verified and never auto-expires).
* **B. Memory**: nothing is stored without your say-so (`remember …`, `mkr memory add`, or answering "y" to a proposal).
* **C. Neural**: `mkr learn stage|approve|reject` builds a reviewed dataset; `mkr learn finetune --anchor <original text>`
  trains a *candidate* on approved text mixed with replayed anchor data and compares it with the live model on two held-out sets.
  It replaces the live model only if new-data loss improves ≥ 3 % **and** anchor loss rises ≤ max(2 %, 0.02 nats); the previous
  weights are backed up and `mkr learn rollback` restores them bit for bit. Tested: a gentle update accepted (new-data loss −38 %,
  anchor +0.015 nats); a destructive update (lr 3e-2, no replay) rejected (anchor 0.18 → 12.07) with the live file untouched.
  A too-aggressive *good* update was also rejected by the guard (anchor +45 %) — on a 0.17 M-parameter model, forgetting is a real
  trade-off, which is exactly what the guard is for.

## Security summary
URL validation (http/https only, no credentials, private/loopback/link-local/CGNAT/metadata ranges blocked for literal IPs *and*
every resolved address, redirects re-validated), request timeouts, size caps, FTS query sanitising, SQLite `quick_check` on open with
automatic quarantine-and-recreate of corrupt databases, CRC-32 on model and checkpoint files, atomic writes, secrets masked in
`mkr config`. Nothing obtained from a website is ever executed.
