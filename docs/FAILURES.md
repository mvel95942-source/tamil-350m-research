# Failures and postmortems

This file exists because a project that only publishes its wins is not evidence. Each entry
has the symptom, the cause, the fix, and where the proof lives.

---

## F1. First training attempt died on its first backward pass

- **Symptom:** out-of-memory on 2×T4 at 350M, before any step completed.
- **Cause:** two independent memory spikes. The loss was computed over a 250 MB logits chunk,
  and Hugging Face's SDPA path took the math kernel for grouped-query attention, storing a
  full `seq × seq` score grid per head.
- **Fix:** compute logits in 512-token chunks (64 MB), and register a custom
  `sdpa_expand_kv` attention that expands KV heads before calling SDPA.
- **Result:** training ran at 5,260 tokens/s with activation checkpointing, then 6,486
  tokens/s after removing it.
- **Evidence:** `tamil/model.py`, `docs/TRAINING.md`.

## F2. The reliability holdout scored 0 / 15

- **Symptom:** an early SFT preview answered every one of the 15 held-out cases incorrectly.
- **Cause:** too little instruction data, no cleaned mixture, synthetic reliability rows not
  yet at full weight, and a base model with only 0.93B pretraining tokens.
- **Fix:** built a cleaned SFT mixture and a mechanically verified reliability set, repeated
  three times; kept the holdout frozen.
- **Result:** pending. The cleaned SFT run had not executed at snapshot time.
- **Evidence:** `docs/EVALUATION.md`, `reports/tamil_reliability_preview_aligned.json`.

## F3. Epoch 1 stopped 72 steps short

- **Symptom:** v5 exited cleanly at about step 3,464 of 3,536.
- **Cause:** 11.8 h of wall time is what the weekly quota allows after v4; the remaining 72
  steps needed roughly 50 more minutes.
- **Fix:** a dedicated short continuation run (v6, retried as v7) capped at 1.6 h.
- **Result:** in flight at snapshot time.
- **Evidence:** `logs/v5_final.log`, `docs/TRAINING.md`.

## F4. A DNS outage broke the unattended chain

- **Symptom:** from 2026-11-08 19:53 to 2026-11-09 04:33 the chain logged
  `Failed to resolve 'api.kaggle.com'` every five minutes.
- **Cause:** local name resolution failure, not a Kaggle outage.
- **Fix:** the chain retries and resumes from its state file; no push was lost.
- **Result:** recovered without manual intervention, but SFT was delayed by about a day.
- **Evidence:** `artifacts/auto_chain.log` (in the private working tree), `docs/FAILURES.md`.

## F5. Pretraining v6 failed after 23 seconds

- **Symptom:** `could not patch LOSS_CHUNK in the mounted tamil/train.py`.
- **Cause:** the kernel selected its code root from its own previous output. That copy already
  contained the environment-variable form of `LOSS_CHUNK`, so the regex that expected a
  literal integer no longer matched and the script exited by design.
- **Fix:** select the code root deterministically (the project dataset snapshot), accept
  either form of the line, and only fail if neither is present. Also select the checkpoint by
  step number instead of file modification time.
- **Result:** retried as kernel version 7.
- **Evidence:** `logs/v6_error.log`, `kaggle/tamil-pretrain-resume.py`.

## F6. The Kaggle code dataset was stale

- **Symptom:** discovered while preparing SFT. The mounted dataset had an older `tamil/sft.py`,
  an older `sources.py`, and **no `eval/tamil_reliability_holdout.jsonl`**.
- **Cause:** the dataset was uploaded once early in the project and never refreshed while the
  code kept changing. The SFT kernel would have trained for 4.5 h and then crashed when it
  tried to open the holdout.
- **Fix:** uploaded dataset version 2 with the current package, the frozen holdout, tokenizers
  and blocklist; made the SFT kernel fail fast if the holdout is missing instead of failing
  after training.
- **Result:** fixed before the SFT push.
- **Evidence:** `kaggle_datasets/tamil-tok-code/dataset-metadata.json`, `docs/TRAINING.md`.

## F7. The unattended chain depends on the laptop being on at handoff time

- **Symptom:** training runs in the cloud and survives shutdown, but the *next push* only
  happens when a Windows scheduled task fires on the local machine.
- **Cause:** the orchestrator is a local Python process, by design, because it uses local
  Kaggle credentials.
- **Fix:** scheduled task every 30 minutes plus a startup shortcut; state is idempotent.
- **Result:** the handoff can be delayed by hours if the laptop is off when a run ends. This
  is a known limitation, not a solved problem.
- **Evidence:** `scripts/auto_chain.py`, `scripts/install_auto_chain.ps1`.

## F8. The Updesh pilot was rejected after inspection

- **Symptom:** the Microsoft Updesh Tamil pilot data inverted semantic roles (agent and
  patient swapped) in a large fraction of examples.
- **Cause:** machine translation of an English source without role-preserving validation.
- **Fix:** rejected the dataset entirely; it was never used for training.
- **Result:** the SFT set is smaller, but it is not poisoned.
- **Evidence:** `reports/sft_data_search.md`.

## F9. Benchmark and test splits were refused

- **Symptom:** several attractive Indic datasets are evaluation sets
  (`sarvamai/gsm8k-indic`, `sarvamai/mmlu-indic`, `sarvamai/arc-challenge-indic`,
  `ai4bharat/indic-math-500`, `ai4bharat/IndicQA`, `google/IndicGenBench_*`).
- **Cause:** they are exactly the data that would make the model look better and the
  evaluation meaningless.
- **Fix:** rejected outright and logged; the rule is enforced in the source review.
- **Result:** no contamination by construction.
- **Evidence:** `reports/sft_data_search.md`.

## F10. A 1.2B from-scratch model is not launchable on this quota

- **Symptom:** only the 350M, 30M and tiny configurations exist; the 13B-token corpus for a
  larger model is unprepared.
- **Cause:** a 1.2B model needs roughly 70 H100-hours at a sensible token budget, against a
  free T4 quota of about 30 hours per week.
- **Fix:** not attempted. Staying at 350M was a deliberate scope decision.
- **Result:** no 1.2B claim is made anywhere.
- **Evidence:** `docs/TRAINING.md`.

## F11. IndicAlign initially produced zero Tamil rows

- **Symptom:** the first cleaned SFT build found no Tamil conversations in IndicAlign.
- **Cause:** the subsets have different schemas (`tam_Taml` versus `interactions`), and a
  whole-folder load let the first file's columns win.
- **Fix:** load each subset from its own parquet file and read the correct field; recorded in
  `tamil/sources.py`.
- **Result:** about 55k Tamil conversations before cleaning.
- **Evidence:** commit history in the private tree, `tamil/sources.py`.

## F12. Synthetic data has not been audited by a fluent speaker

- **Symptom:** arithmetic rows are mechanically correct, but fluency and register are
  template-written.
- **Cause:** no Tamil reviewer was available.
- **Fix:** none yet. The generator is deterministic and every arithmetic answer is computed,
  not typed.
- **Result:** an open gap, stated here rather than hidden.
- **Evidence:** `scripts/build_sft_reliability_ta.py`, `data/sft_reliability_ta.jsonl`.

---

## What we would do differently

1. Pin the Kaggle container image and library versions at the start.
2. Treat the code dataset as a build artifact: version it in the same commit as the kernel.
3. Fail fast on every input the kernel needs (holdout, tokenizer, synthetic set) before
   starting a multi-hour job.
4. Keep one source of truth for "current step count" instead of deriving it from throughput.
5. Get a fluent Tamil reviewer involved before writing synthetic SFT data, not after.
