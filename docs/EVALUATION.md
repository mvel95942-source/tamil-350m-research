# Evaluation: protocol, results, and what the numbers mean

There are three separate things here, and they must not be mixed:

1. **Tokenizer quality** — measured, reproducible, strong.
2. **Model reliability** — measured on 15 frozen cases, currently **0/15**.
3. **Optional facts layer** — a dated lookup, *not* the model. Reported separately, always.

---

## 1. Tokenizer

Benchmark: `tamil.fertility`, 2,000 held-out documents per set. Tamil and English Wikipedia
are held out by taking the first 2,000 articles (the rest is training data, and those first
2,000 articles are blocklisted). Tanglish uses the Dakshina test split.

| tokenizer | Tamil tok/word | Tanglish (real) | English | Tamil bytes/token | cut letters | round-trip |
|---|---:|---:|---:|---:|---:|---:|
| **ours 32k** | **2.26** | **2.86** | 1.73 | 10.73 | **0.01%** | **100.0%** |
| ours 48k | 2.13 | 2.69 | 1.66 | 12.27 | 0.01% | 100.0% |
| Gemma-3 / Gemma-4 | 2.67 | 3.53 | 1.41 | 9.09 | 11.4% | 100.0% |
| Sarvam-1 | 2.70 | 4.67 | 1.72 | 8.97 | 11.5% | 100.0% |
| Gemma-2 | 4.33 | 3.55 | 1.40 | 5.60 | 21.8% | 100.0% |
| GPT-4o (o200k) | 7.61 | 3.54 | 1.42 | 3.19 | 40.0% | 100.0% |
| Qwen-2.5 | 9.74 | 3.99 | 1.44 | 2.49 | 31.3% | 100.0% |
| Llama-3 | 11.94 | 3.93 | 1.40 | 2.03 | 25.5% | 100.0% |
| IndicBERT-v2 | 1.88 | 2.86 | 1.37 | 12.92 | 7.6% | 0.1% |
| Agaram-2.5B | 1.91 | 3.04 | 1.55 | 12.67 | 15.9% | 100.0% |

Reading this honestly:

- We are **not** the best on Tamil tokens/word. IndicBERT-v2 (1.88) and Agaram-2.5B (1.91)
  are lower.
- IndicBERT-v2 is **lossy**: 0.1% round-trip means it cannot reconstruct the input, which
  makes it unusable as a generation tokenizer.
- Agaram-2.5B round-trips correctly but **cuts 15.9% of Tamil letters**, and its vocabulary is
  64k, twice ours.
- Our defensible claim: **2.26 tokens/word at 32k, with 0.01% cut letters and 100%
  round-trip.** No public tokenizer in this comparison combines those three.
- **English is worse** (1.73 vs 1.37–1.44). That is the cost of a Tamil-first vocabulary.

Speed and memory: `reports/tokenizer_bench.md` (tok64k artifact, Windows, Python 3.14,
`tokenizers` 0.22.2). Tamil 1 MB encode at 67,723 tokens/s, 200k-character document in
776 ms, round-trip verified.

Artifacts: `reports/fertility_v2.md`, `reports/fertility_v2.json`,
`reports/tokenizer_bench.md`, `reports/tokenizer_bench.json`.

---

## 2. Model reliability

### Protocol

- 15 cases in `eval/tamil_reliability_holdout.jsonl`, frozen before this evaluation.
- Each case has the prompt, a category, and human-written pass criteria in Tamil.
- Decoding: greedy (`do_sample=False`), 96 new tokens, `repetition_penalty=1.15`,
  `no_repeat_ngram_size=3`.
- Prompt format matches training exactly: `<s><|user|>\n{prompt}</s><|assistant|>\n`.
- Scoring: a human checks each answer against its criteria. The harness records text; it does
  not pretend to score correctness automatically.

### Result

Checkpoint: an **early SFT preview**, trained before the current data cleanup. It is not the
final SFT model, which has not been trained yet. Evidence:
`reports/tamil_reliability_preview_aligned.json`.

**Score: 0 / 15.**

| id | prompt (gloss) | expected | produced | verdict |
|---|---|---|---|---|
| arith_add_01 | 17 + 26, show the work | 43 | `30 + 45 = 20 + + 12 + 32 + + +` | fail: wrong, malformed |
| arith_mult_01 | 14 × 6 | 84 | `20 × (3) ×8 ×9` | fail: no answer |
| arith_div_01 | 45 ÷ 5 | 9 | `32` | fail: wrong |
| arith_word_01 | 8 pencils in a box, 3 boxes | 24 | `10 pencils, 2 pencils` | fail: wrong, no reasoning |
| geometry_01 | rectangle 9 m × 4 m, perimeter | 26 m | `2 metres` | fail: wrong formula |
| science_01 | chemical symbol for CO₂ | CO₂ | `I don't know` | fail: abstained on a known fact |
| science_02 | gas in air used for breathing | oxygen | `hydrogen` | fail: wrong |
| language_01 | plural of *puththakam* (book) | *puththakangal* | long unrelated paragraph about books | fail: no answer |
| language_02 | verb in *aval paadinaal* | *paadinaal* | `smart people sang` | fail: no analysis |
| instruction_01 | list the first five weekdays, one sentence | five days in order | `yes, writing only the last three days of the month helps` | fail: ignored instruction |
| correction_01 | correct 7 × 9 = 56 | it is wrong, 63 | `10 + 24 → 33 ×8 = 22 ×22` | fail: nonsense |
| uncertainty_weather | is it raining in Chennai now? | admit it cannot know | `Chennai` | fail: no answer, no abstention |
| uncertainty_current | exact India population this minute | admit it cannot know | invented `49,528` and a `10%` claim | fail: fabricated precision |
| ambiguity_01 | "how much is it?" | ask what "it" refers to | `post box` | fail: guessed |
| brevity_01 | half of 100, briefly | 50 | `24 hours` | fail: wrong |

### Why it failed

- The preview checkpoint had very little instruction data and had not been trained with the
  cleaned mixture or the synthetic reliability rows at full weight.
- A model at 0.93B pretraining tokens has weak arithmetic and factual grounding to begin with.
  Instruction tuning can teach the *shape* of an answer; it cannot supply the knowledge.
- The failure mode is consistent: the model has learned the answer register (`விடை:` /
  "answer:") but not the content. That is exactly what an under-trained SFT looks like.

This is a failure, reported as a failure. It is the main reason the cleaned SFT run exists.

### What will be measured after the cleaned SFT

The same 15 cases, same protocol, same file. Results will be written to
`reports/tamil_reliability_sft_clean.json`. Success is not "15/15"; it is a clear improvement
with the remaining failures named.

---

## 3. Optional facts layer (not the model)

`scripts/facts_ta.py` answers a narrow set of questions from a dated snapshot
(`data/facts_ta.json`) built from Wikipedia, Wikidata and the World Bank, plus live weather
from Open-Meteo. It abstains on prices and news.

Verified by manual test on **2026-11-07**:

| Question | Answer | Source date |
|---|---|---|
| Prime Minister of India | Narendra Modi (since 26 May 2014) | Wikidata |
| President of India | Droupadi Murmu (since 25 Jul 2022) | Wikidata |
| Chief Minister of Tamil Nadu | C. Joseph Vijay (since 10 May 2026) | Wikipedia |
| Chief of Army Staff | Dhiraj Seth (since 30 Jun 2026) | Wikipedia |
| India population | 1,463,865,525 | World Bank 2025 |
| Chennai weather | live | Open-Meteo |

This layer is **off by default** and is never counted as model ability. The model should learn
facts from training data; the lookup exists for volatile facts that a 350M model cannot know,
and its use is always visible in the response path.

---

## Evaluation gaps

- No bits-per-byte on the final checkpoint yet.
- No IndicXNLI / IndicSentiment / Belebele / Aksharantar scores yet.
- No human evaluation by a fluent Tamil speaker.
- 15 cases cannot distinguish "close" from "lucky". They are a gate, not a benchmark.

Each gap is listed because the absence of a number is itself a result.
