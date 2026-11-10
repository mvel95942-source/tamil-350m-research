# Tamil 350M — a from-scratch Tamil language model

This repository is the research log of an attempt to train a **350M-parameter Tamil-first
chat model from scratch**, on free Kaggle T4 GPUs, without starting from a pretrained
multilingual model. It contains the code, the data rules, the exact training runs, the raw
logs, and the evaluation results — including the parts that did not work.

It is **not a product**, **not a released model**, and **not a claim of state-of-the-art**.
It is a reproducible record of what was built, what it cost, what it achieved, and where it
currently fails.

Status date: **2026-11-10**.

---

## Current status

| Stage | State | Evidence |
|---|---|---|
| Akshara-safe Tamil tokenizer (32k) | **done** | `reports/fertility_v2.md`, `reports/tokenizer_bench.md` |
| Packed pretraining corpus | **done** — 927,436,918 tokens | `docs/TRAINING.md` |
| Pretraining, epoch 1 | **3,464 / 3,536 steps** in the last completed run; a short continuation to 3,536 is in flight | `logs/v4_final.log`, `logs/v5_final.log` |
| Supervised fine-tuning | **not run yet** — kernel written, never pushed | `kaggle_sft_clean/` |
| Reliability holdout (15 cases) | **0 / 15** on an early SFT preview checkpoint | `docs/EVALUATION.md` |
| Local chat UI | built; tested with the optional facts layer | `scripts/chat_tamil_web.py` |
| Optional dated facts layer | built and tested | `scripts/facts_ta.py` |

The honest one-line summary: **the tokenizer and the training pipeline work; the model does
not chat reliably yet, and the final SFT run has not happened.**

---

## What this project achieved

1. **A Tamil tokenizer that never splits a letter.**
   Every akshara (grapheme cluster) is mapped to a single Private Use Area code point before
   training, so no token can cut a Tamil letter in half. Measured on 2,000 held-out documents
   per language set:

   | tokenizer | Tamil tokens/word | Tamil cut letters | round-trip |
   |---|---:|---:|---:|
   | **ours, 32k** | **2.26** | **0.01%** | **100.0%** |
   | ours, 48k | 2.13 | 0.01% | 100.0% |
   | Gemma-3 / Gemma-4 (262k) | 2.67 | 11.4% | 100.0% |
   | Sarvam-1 | 2.70 | 11.5% | 100.0% |
   | Gemma-2 | 4.33 | 21.8% | 100.0% |
   | GPT-4o (o200k) | 7.61 | 40.0% | 100.0% |
   | Qwen-2.5 | 9.74 | 31.3% | 100.0% |
   | Llama-3 | 11.94 | 25.5% | 100.0% |
   | IndicBERT-v2 | 1.88 | 7.6% | 0.1% |
   | Agaram-2.5B | 1.91 | 15.9% | 100.0% |

   We are **not** the lowest-token tokenizer: IndicBERT-v2 and Agaram-2.5B are lower. But
   IndicBERT-v2 is lossy (0.1% round-trip), and both cut 7.6–15.9% of Tamil letters. The
   defensible claim is: **near-best Tamil fertility at a 32k vocabulary, with no broken
   letters and lossless round-trip.** English fertility is worse than the competition
   (1.73 vs 1.37–1.44), which is a deliberate Tamil-first trade-off and a real limitation.

2. **A 927.4M-token, contamination-controlled pretraining corpus** assembled only from
   reputable, licence-clear releases (Sangraha, IndicCorpV2, Tamil Wikipedia, FineWeb-Edu,
   Dakshina). Held-out documents are blocklisted from training.

3. **A working from-scratch training pipeline** at 6,426–6,486 tokens/s on 2×T4 for a 380.2M
   parameter model, with a resumable checkpoint chain that survived restarts, quota limits
   and a DNS outage.

4. **A discipline of refusing convenient data.** No benchmark or test split was ever trained
   on. The Microsoft Updesh pilot was rejected after inspection showed semantic role
   inversion. Community datasets with unknown provenance were rejected. The reasons are
   recorded per dataset in `reports/sft_data_search.md`.

5. **A frozen 15-case reliability holdout** written before the final SFT run, so the result
   cannot be quietly redefined after the fact.

---

## What does not work (and we are not hiding it)

- **The model is not a usable chat assistant yet.** The last reliability probe, on an early
  SFT preview checkpoint, scored **0 of 15**. Answers were short, wrong, or unrelated
  (for example, `45 ÷ 5` produced `32`). Full transcript in `docs/EVALUATION.md`.
- **World knowledge is thin.** The model saw 0.93B tokens. A compute-optimal run for this
  size would be roughly 7B tokens, so it has about 13% of the training usually needed.
  Fine-tuning cannot add facts that pretraining never saw.
- **Epoch 1 is not finished inside one weekly quota.** It reached step 3,464 of 3,536 in
  23.2 GPU-hours; the remaining 72 steps needed a short extra run.
- **No weights are published yet.** Checkpoints live in private Kaggle kernel outputs and
  are multi-gigabyte. This repository ships code, data rules, and evidence, not tensors.
- **No fluent Tamil speaker has audited the SFT data or the outputs.** Arithmetic templates
  are mechanically verified; naturalness is not.
- **The optional facts layer is a lookup, not model knowledge.** It is documented and
  switchable, and its results are always reported separately from the raw model.

---

## Repository map

```
tamil/                  model, tokenizer, corpus, packing, SFT, training loop, tests
kaggle/                 Kaggle kernel scripts (tokenizer build, pretrain, eval, SFT)
kaggle_sft_clean/       the cleaned SFT kernel (Aya + IndicAlign + synthetic reliability)
kaggle_datasets/        dataset metadata for the two private Kaggle datasets we used
scripts/                evaluation, facts layer, chat UI, synthetic data generator, auto-chain
eval/                   the frozen 15-case reliability holdout
data/                   the 3,889-row synthetic reliability SFT set (no holdout prompts)
logs/                   raw Kaggle logs for the completed pretraining runs and the failed one
reports/                tokenizer benchmarks, fertility, SFT source search, reliability probes
docs/                   data rules, training record, evaluation, failures, reproduction
docs/history/           the internal handoff note kept for provenance
```

---

## Availability

- **Code:** this repository, Apache-2.0.
- **Tokenizer:** benchmarked here; publication planned.
- **Model weights:** not released. When they are, they will be a separate download with the
  model card in `MODEL_CARD.md`.
- **Upstream corpora:** not redistributed. Every source is identified by its Hugging Face
  ID and licence in `docs/DATA.md` so you can rebuild the corpus yourself.

---

## Reproduce

See `docs/REPRODUCE.md`. The short version: create the two Kaggle datasets, push
`kaggle/tamil-pretrain-resume.py` on 2×T4, then push `kaggle_sft_clean/`. Expect roughly
11.5 GPU-hours per pretraining session and 4.5 hours for SFT, against a free quota of about
30 GPU-hours per week.

---

## Provenance and drift

Every number here comes from a logged run on the date shown. The runs used the **Kaggle
container image of that day**; `torch`, `transformers`, `tokenizers` and CUDA were not
pinned. Re-running this code later will use newer library versions and may not reproduce the
same throughput or the same tokenizer internals. Where a plan changed mid-project, the
as-run configuration is recorded in `docs/TRAINING.md`, not the original plan.

Design decisions and their dates are recorded in `docs/history/`. This is a snapshot of a
moving project, not a polished release.

---

## Licence and research use

Code: Apache-2.0 (`LICENSE`). Third-party data keeps its own licence (`NOTICE`,
`docs/DATA.md`). This work is published for research and reproducibility. It is not
affiliated with Kaggle, AI4Bharat, Google, Cohere or any dataset provider.

If you use this work, cite it as described in `CITATION.cff`.
