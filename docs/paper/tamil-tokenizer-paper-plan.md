# Paper plan: Tamil tokenizer (decided 2026-11-03)

Working title: "Akshara-aware tokenization for Tamil, Tanglish and English:
fewer tokens, no broken letters, and what it does for a small LM"

Rule: every number comes from a logged run (Kaggle notebook + version, or a commit + command). No borrowed or estimated numbers in the paper.

## Claims (only if the data supports them)
1. Fewer tokens per word than public tokenizers on held-out Tamil, at 0% letters cut mid-akshara.
   Evidence: fertility benchmark (`tamil.fertility`, 2,000 held-out docs per set) vs GPT-4o, Llama-3, Gemma-2, Qwen-2.5, mT5, IndicBERT-v2, Sarvam-1.
2. Tanglish (romanized Tamil) and English stay competitive in the same 32k/48k vocab.
3. Same model, same data bytes, different tokenizer: ours vs Llama-3 tokenizer gives lower bits-per-byte on Tamil (ablation, `kaggle/03_ablation.py`).
4. A 350M model trained from scratch on it: bits-per-byte plus IndicXNLI / IndicSentiment / Belebele-ta / Aksharantar scores (labeled sets never trained on).

## Sections
1. Intro: why Tamil splits badly in multilingual tokenizers (3-byte UTF-8 code points, grapheme clusters cut apart).
2. Method: akshara to Private Use Area mapping (312 units), Unigram with byte fallback, synthetic Tanglish romanization.
3. Data: Sangraha, IndicCorp v2, Tamil Wikipedia (2,000 articles held out), FineWeb-Edu; contamination blocklist.
4. Tokenizer results: fertility table; 32k vs 48k.
5. Model results: ablation + 350M evals.
6. Limitations: decoded leading space is lost; offsets point at an akshara's last code point; synthetic Tanglish is not real chat Tanglish.

## Where to publish (check each venue's current dates and rules before submitting)
- arXiv technical report (cs.CL). A first-time submitter may need an endorsement.
- A Dravidian-language or low-resource NLP workshop, if a deadline fits.
- Always: tokenizer + model on Hugging Face with a model card, and the code repo.

## Results log
| Result | Run | Number |
|---|---|---|
| Smoke fertility, 36 lines of one Tamil Wikipedia article (NOT the benchmark) | local, commit 532e160 | tok/word: ours-48k 2.66, ours-32k 2.86, Gemma-2 4.52, GPT-4o 7.91; cut letters: ours 0%, Gemma-2 19%, GPT-4o 39% |
| GPU speed, 350M, 2x T4, random tokens | Kaggle tamil-gpu-throughput v5 | 7,336 tok/s at micro_bs 4 |
| Smoke fertility, 42 prose lines of ta.wikipedia "தமிழ் எழுத்து முறை" (NOT the benchmark) | local, commit 15275de | tok/word: ours-48k 2.10, ours-32k 2.23, Gemma-4 (E2B/12B/31B, 262k) 2.64, Gemma-3 2.64; cut letters: ours 0%, Gemma-4 10.9% |
| Gemma-4 tokenizer vs Gemma-3 | same run | identical numbers on all 42 lines, same 262,144 vocab: treat as the same tokenizer |
| Gemini 3.8 Flash, GPT-6/5.6 "Luna" | n/a | API-only, no public tokenizer; measurable only via each API's token-count endpoint with an owner key |
| **Fertility benchmark, 2,000 held-out docs per set** (ta/en Wikipedia first 2,000, Dakshina test) | local, commit 5ea5a2d, parquet reader (datasets blocked by Windows App Control); reports/fertility.md | Tamil tok/word: ours-32k 2.26, ours-48k 2.13, Gemma-3/4 2.67, Sarvam-1 2.70, Gemma-2 4.33, GPT-4o 7.61, Qwen-2.5 9.74, Llama-3 11.94, IndicBERT-v2 1.88 (lossy: 0.1% round-trip). Cut letters: ours 0%, others 7.6-40%. English: ours 1.73/1.64 vs 1.37-1.44 (worse). mT5 skipped (sentencepiece not installed). Vocab pick: 32k (48k only 5.7% lower, rule needs 8%) |
| **Agaram-2.5B tokenizer** (murai-labs/agaram-2.5b, 64k vocab, arXiv 2610.18420), same 2,000-doc sets | local; reports/fertility_agaram.md | Tamil tok/word: Agaram 1.91 vs ours-32k 2.26 / ours-48k 2.13 (Agaram wins). Tanglish real: ours-32k 2.86 vs Agaram 3.04 (ours wins). English: Agaram 1.55 vs ours 1.73. Cut letters: Agaram 15.9%, ours 0%. Headline "fewest Tamil tokens" is FALSE vs Agaram; the claim must be "no broken letters + best Tanglish at half the vocab". |
