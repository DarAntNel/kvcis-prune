# KVCIS-3 — two probes: original KVCIS tiers + deferred eviction

A standalone variant of the KVCIS method (see `../kvcis/`). The original KVCIS
probe is kept **operating exactly as designed**, and a **second probe** is
trained alongside it to decide — later, as the cache grows — which already-stored
tokens can be evicted.

| Probe | What it is | When it acts | What it decides |
|-------|-----------|--------------|-----------------|
| **A — importance** | the *original* KVCIS Ridge regression, recreated unchanged | at insertion | importance ≥ threshold → **fp16**, else **int8**. Every token is stored; nothing is dropped up front. |
| **B — evictability** | new binary logistic probe | **deferred** — only after a token has aged past a grace window | flags int8-tier tokens whose attention it predicts will go quiet later; those are then physically removed from the cache |

Hard rules, enforced at runtime regardless of what probe B says:

1. **fp16-tier tokens are never evicted.** High-attention tokens always remain.
2. The first `--force-sink` tokens are never evicted (attention sinks).
3. The most recent `--grace-recent` tokens are never evicted — they are too
   young for the "goes quiet later" judgement to apply. A token like *"over"*
   in *"The cow jumped over…"* must survive long enough to serve a follow-up
   question ("where did the cow jump?").
4. Only evict when `P(evictable) ≥ --evict-margin` (raise it for safety;
   precision is the safety metric — a false positive evicts a needed token).

The motivating example: after the model finishes "The cow jumped over the
moon" and the user asks "where did the cow jump?", the second *"the"*
(position 5) has served its brief syntactic duty and adds no semantic value —
it is int8-tier, aged past grace, and quiet: evict it. *"over"* stays.

This folder is self-contained: it copies the cache/quantization plumbing it
needs into `code/kv_utils.py` and shares nothing at runtime with `../kvcis/`
(same venv, same model, separate `data/` and `results/`).

> **History:** this folder previously implemented a 3-class insertion-time
> probe (fp16 / int8 / never-store); those results remain in
> `results/kvcis3_results.json` and `results/margin_*/`. The two-probe design
> replaces it: instead of refusing to store tokens up front, everything is
> stored and eviction is a *deferred, revocable-by-design* decision.

---

## How the labels are made (step 2)

At each generation step *t*, the attention row over the current sequence sums
to 1, so a token's *fair share* is `1/L_t`. We use the share **ratio**
`r_t(j) = a_t(j) · L_t` (1.0 = exactly average attention), comparable across
steps and context lengths. For each prompt token, two targets:

- **`importance`** (probe A): cumulative attention received, max-normalized
  per sequence — byte-for-byte the original repo's step2 target.
- **`evictable`** (probe B): 1 iff `max over steps t ≥ W of r_t < --evict-max-ratio`
  (defaults: W = `--evict-window` = 10, ratio 0.5). Being consulted *early* is
  fine and expected — local syntax duty; what matters is that the token goes
  quiet **after** the window. This is strictly more permissive than the old
  "never consulted at any step" discard rule, and is exactly the
  position-5-"the" case.

"Quiet after W" is only as trustworthy as the horizon it was measured over:
prompts whose generation ends before W+3 steps are skipped, and
`--generation-steps` (default 30, `-Full` 40) is the label horizon.

Step 2 also saves `ratio_mean.npy` / `ratio_max.npy` / `ratio_max_late.npy`
so labels can be re-thresholded offline without re-running the model.

## The probes (step 3)

- **Probe A** — `Ridge` on layer-10 activations → importance scalar. Saved
  exactly like the parent repo (`probe2/regression/weights.npy [H]`, `bias [1]`),
  verified against manual `X @ w + b`.
- **Probe B** — binary `LogisticRegression` (balanced) → evictability logit
  (`probe2/evict/`). Trained on all tokens; the fp16 exemption is a hard
  runtime rule, not learned.

Both read the token's **own activation the moment it is computed**, so both
decisions are *predicted* at insertion — but probe B's action is **deferred**:
the token is stored normally and only removed after it has aged past the grace
window. Prediction at insertion, action after grace.

## Evaluation (step 4)

Byte-accurate memory (fp16 = 2 B/elem; int8 = 1 B/elem + 8 B quant metadata per
token/layer/tensor; evicted = 0 B once removed) and perplexity on held-out
wikitext continuations, scored at the targets' **true absolute positions** so
RoPE stays correct after tokens are removed (verified in the parent repo:
keep-all == baseline exactly).

Methods: `Baseline`, `Uniform-INT8`, `KVCIS` (probe A only — the original
scheme, nothing evicted), `KVCIS+Evict` (probe A tiers + probe B deferred
eviction).

**Honest memory accounting:** deferred eviction stores everything first, so its
*peak* bytes equal the plain-KVCIS row; the table reports *steady-state* bytes
(after eviction has caught up). Both figures are in the JSON. If you need the
peak never to occur, that is the old insertion-discard design — the trade is
recorded in this repo's history.

---

## Run

```powershell
cd kvcis3\scripts
.\run_3080.ps1          # quick pass: step1 -> step2 (200 prompts) -> step3 -> step4
.\run_3080.ps1 -Full    # 500 prompts, longer label horizon
```

Or step by step (from `kvcis3\code`, with `$env:PYTHONUTF8=1`):

```powershell
python step1_single_prompt.py    --model Qwen/Qwen2.5-1.5B-Instruct --extraction-layer 10   # sanity check on one prompt
python step2_collect_data.py     --model Qwen/Qwen2.5-1.5B-Instruct --extraction-layer 10 --n-prompts 200 --generation-steps 30 --evict-window 10 --output-dir ../data
python step3_train_probe.py      --data-dir ../data --output-dir ../data/probe2
python step4_compression_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --extraction-layer 10 --probe-path ../data/probe2 --n-texts 20
```

Step 1 mirrors the parent repo's `step1_single_prompt.py`: it verifies
activation extraction, records one prompt's per-step share-ratio trajectory,
prints each token's trajectory-derived evictability next to probe A's tier and
probe B's P(evictable), applies the hard rules, and shows the storage summary
(all-fp16 → peak → steady). Its default prompt is the cow example above.

Knobs worth sweeping on step 4:

- `--evict-margin` (default 0.5) — the quality/memory dial. Raise toward 0.7–0.9
  to evict only what probe B is confident about.
- `--grace-recent` (default 16) — how long a token must age before eviction.
- `--high-threshold` (default 0.9) — probe A's fp16 bar (original KVCIS default).
