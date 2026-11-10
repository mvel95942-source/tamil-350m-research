# Tokenizer validation on an independent held-out Tamil corpus

- corpus: `HuggingFaceFW/fineweb-2` / `tam_Taml` / `test` (downloaded parquet)
- parquet sha256: `bba0eb727be01a334ae38f1cee3275ce55fb7c9c7d1e4a20b0b8bad5509c96d1`
- docs kept: 20,000 (rows seen 20,131)
- chars: 40,207,389 · words: 4,491,695 · bytes: 108,603,006
- max chars/doc: 4000 · min Tamil share: 0.5

## Tamil tokens per word (lower is better), bootstrap 95% CI

| tokenizer | tokens/word | 95% CI | chars/token | bytes/token | cut letters | round-trip |
|---|---|---|---|---|---|---|
| ours-tok64k | 1.693 | [1.690, 1.696] | 5.29 | 14.28 | 0.04% | 100.0% |
| ours-tok32k | 1.853 | [1.849, 1.857] | 4.83 | 13.05 | 0.05% | 100.0% |
| gpt-4o (o200k) | 7.502 | [7.489, 7.515] | 1.19 | 3.22 | 41.5% | 100.0% |
| llama-3 | 11.855 | [11.833, 11.875] | 0.76 | 2.04 | 33.0% | 100.0% |
| gemma-4 (262k) | 2.357 | [2.353, 2.361] | 3.80 | 10.26 | 12.0% | 100.0% |
| qwen-2.5 | 9.711 | [9.694, 9.727] | 0.92 | 2.49 | 40.3% | 100.0% |
| indicbert-v2 | 1.640 | [1.637, 1.643] | 5.46 | 14.74 | 6.6% | 0.0% |
| agaram-2.5b | 1.591 | [1.588, 1.594] | 5.63 | 15.20 | 13.7% | 100.0% |

## Tamil-dominant subset (>=95% Tamil script chars)

| tokenizer | docs | tokens/word | 95% CI | cut letters | round-trip |
|---|---|---|---|---|---|
| ours-tok64k | 15,700 | 1.670 | [1.667, 1.673] | 0.04% | 100.0% |
| ours-tok32k | 15,700 | 1.833 | [1.830, 1.837] | 0.05% | 100.0% |
| gpt-4o (o200k) | 15,700 | 7.663 | [7.652, 7.673] | 41.8% | 100.0% |
| llama-3 | 15,700 | 12.148 | [12.131, 12.164] | 33.1% | 100.0% |
| gemma-4 (262k) | 15,700 | 2.365 | [2.361, 2.369] | 12.3% | 100.0% |
| qwen-2.5 | 15,700 | 9.931 | [9.918, 9.943] | 40.5% | 100.0% |
| indicbert-v2 | 15,700 | 1.641 | [1.638, 1.644] | 6.9% | 0.0% |
| agaram-2.5b | 15,700 | 1.573 | [1.570, 1.576] | 14.3% | 100.0% |

## Byte fallback / unknown tokens (ours)

| tokenizer | byte tokens | <unk> |
|---|---|---|
| ours-tok64k | 0.01% | 0.0000% |
| ours-tok32k | 0.01% | 0.0000% |

`ours-tok32k` vs `ours-tok64k`: +9.5% tokens per Tamil word.
