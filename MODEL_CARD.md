# Model card — Tamil 350M (research checkpoint)

Status: **work in progress**. This card describes the checkpoint as it exists on
2026-11-10. It is not a release announcement.

---

## Model details

| Field | Value |
|---|---|
| Model class | `Qwen3ForCausalLM` (dense decoder, QK-norm, RoPE) |
| Parameters | **380,155,904** measured from the checkpoint (project name: 350M class) |
| Layers | 20 |
| Hidden size | 1,024 |
| Feed-forward size | 3,072 |
| Attention | 16 query heads, 8 KV heads, head dim 128 |
| Context length | 4,096 tokens |
| Vocabulary | 32,000 (custom akshara-safe Tamil tokenizer) |
| Embeddings | tied input/output |
| Positional | RoPE, theta 1,000,000 |
| Precision | fp16 autocast, fp32 master weights |
| Tokenizer | Unigram with byte fallback, akshara→PUA preprocessing |
| Chat format | `<s><|user|>\n...\n</s><|assistant|>\n...\n</s>` |

The tokenizer maps each Tamil akshara to one Private Use Area code point before training.
Decoded text is mapped back, so generation stays in normal Unicode. Offsets point at the last
code point of an akshara, not the first.

---

## Training

| Field | Value |
|---|---|
| Objective | next-token prediction, cross-entropy |
| Corpus | 927,436,918 tokens, packed into a flat uint16 token file |
| Steps per epoch | 3,536 at 262,144 tokens/step |
| Epochs completed | **0.98** (step 3,464 of 3,536 in the last completed run) |
| Hardware | Kaggle free tier, 2 × NVIDIA T4 (16 GB each) |
| Throughput | 6,426–6,486 tokens/s |
| GPU time on pretraining | 9.1 + 11.4 + 11.8 = **32.3 h** across three sessions |
| Optimizer | AdamW, decoupled weight decay 0.0, betas (0.9, 0.95) |
| Batch | micro-batch 2, gradient accumulation 64, 2 data-parallel ranks = 256 sequences/step |
| Precision | fp16 autocast with loss scaling |
| Memory measures | loss computed in 512-token chunks; `expandable_segments:True`; no activation checkpointing |

Corpus sources (details and licences in `docs/DATA.md`):

- AI4Bharat **Sangraha** (verified Tamil)
- AI4Bharat **IndicCorpV2** (Tamil)
- **Tamil Wikipedia** (first 2,000 articles held out and blocklisted)
- **FineWeb-Edu** sample-10BT (English)
- Google **Dakshina** (romanized Tamil, dev split)

No benchmark or test split was included. Held-out documents are blocklisted by sentence.

---

## Evaluation

### Tokenizer

2,000 held-out documents per set, measured with `tamil.fertility`:

- Tamil: **2.26 tokens/word**, **0.01%** of letters cut, **100%** round-trip.
- Tanglish (real, Dakshina): 2.86 tokens/word.
- English: 1.73 tokens/word.

Full comparison in `reports/fertility_v2.md`.

### Model reliability

Fifteen frozen cases covering arithmetic, a word problem, geometry, two science facts,
two grammar questions, instruction following, error correction, live weather, live
population, ambiguity and brevity. Protocol: greedy decoding, 96 new tokens, scored against
written criteria.

Result on an **early SFT preview checkpoint**: **0 / 15**. Every case failed, including
`17 + 26` (returned a malformed expression) and `45 ÷ 5` (returned `32`). The full
transcript is in `docs/EVALUATION.md`.

This score is not the final model. It is the last measured checkpoint, reported because
hiding it would make every other number in this repository untrustworthy.

### Intrinsic language quality

Not yet measured on the final checkpoint. The plan is bits-per-byte on held-out Tamil and
English, plus IndicXNLI, IndicSentiment, Belebele-Tamil and Aksharantar, all of which are
blocklisted from training. None of these numbers exist yet.

---

## Intended use

- Research on Tamil tokenization, small-model pretraining and evaluation methodology.
- A testbed for reliability evaluation on a model small enough to run on consumer hardware.
- Reproducing the training pipeline on free-tier GPUs.

## Out of scope

- Factual question answering about current events, people, prices or news.
- Any medical, legal, financial or safety-critical advice.
- Production chat, customer support, or anything where a wrong answer has a cost.
- Languages other than Tamil and English. Tanglish support is partial and synthetic.
- Long-context reasoning. The context window is 4,096 tokens but the model has not been
  evaluated beyond short prompts.

---

## Limitations and risks

1. **Undertrained for knowledge.** 0.93B tokens is about 13% of a compute-optimal budget for
   this parameter count. Expect confident errors on facts.
2. **Not instruction-tuned in the recorded results.** The only SFT checkpoint measured was an
   early preview; the cleaned SFT run is pending.
3. **Data naturalness is not human-audited.** The synthetic reliability set is mechanically
   correct but written by templates.
4. **Tokenizer trade-off.** English is tokenized worse than multilingual competitors, by
   design.
5. **Evaluation is small.** Fifteen cases detect gross failure, not fine-grained ability.
   They are a gate, not a benchmark.
6. **Bias.** The corpus is web text and Wikipedia. It carries the usual demographic,
   geographic and political skews, plus English dominance in the mixed corpus.
7. **Reproducibility drift.** The Kaggle image, CUDA, torch and transformers versions were
   not pinned. Later runs may differ.

---

## Ethical and legal notes

- Only licence-clear sources from established organisations were used. Community datasets
  with unknown provenance were rejected, even when convenient.
- No benchmark test sets were trained on.
- The synthetic SFT data contains no prompts from the frozen holdout.
- Upstream corpora are not redistributed here; their licences are listed in `NOTICE`.
- The model has not been red-teamed. Do not expose it to users without a filter and a
  human in the loop.

---

## Citation

See `CITATION.cff`. If you use the tokenizer or the training record, cite this repository
and the upstream datasets listed in `docs/DATA.md`.
