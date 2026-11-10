# Tokenizer speed and memory benchmark

- artifact: `artifacts\tok64k` · vocab 63,995
- sha256: `861ebf3409db01b3964686bbbf3a71f0b721cd88b571bcbdb3d67196397a25d8`
- machine: Windows-11-10.0.26300-SP0 · 6C/12T · 16.4 GB RAM
- python 3.14.4 · tokenizers 0.22.2 · numpy 2.4.2
- wall time: 735.5s

## Load cost

| tokenizer | load ms | RSS before (MB) | RSS after (MB) | delta (MB) |
|---|---|---|---|---|
| tok64k | 107.7 | 106.4 | 176.4 | 70.0 |

## Encode / decode throughput

Median wall time on a warm tokenizer; MB = 10^6 UTF-8 bytes of input. `tok/s` counts output tokens.

| domain | size | MB/s encode | tok/s encode | MB/s decode | ms encode | round-trip |
|---|---|---|---|---|---|---|
| tok64k · tamil | 1 MB | 0.98 | 67,723 | 1.89 | 1020.1 | yes |
| tok64k · tanglish | 1 MB | 0.47 | 161,502 | 1.44 | 2145.5 | yes |
| tok64k · english | 1 MB | 0.47 | 96,881 | 1.60 | 2143.7 | yes |
| tok64k · code | 1 MB | 0.13 | 52,710 | 1.39 | 7479.0 | yes |
| tok64k · digits | 1 MB | 0.33 | 296,900 | 1.06 | 3047.4 | yes |
| tok64k · emoji | 1 MB | 0.52 | 240,877 | 1.29 | 1930.8 | yes |
| tok64k · tamil | 10MB | 0.65 | 44,716 | 1.77 | 15451.0 | yes |
| tok64k · tanglish | 10MB | 0.45 | 157,210 | 1.31 | 22042.7 | yes |
| tok64k · english | 10MB | 0.46 | 96,341 | 1.44 | 21558.1 | yes |
| tok64k · code | 10MB | 0.13 | 50,196 | 1.26 | 78538.3 | yes |
| tok64k · digits | 10MB | 0.28 | 249,799 | 0.93 | 36219.6 | yes |
| tok64k · emoji | 10MB | 0.52 | 239,796 | 1.29 | 19396.3 | yes |

## Long document and batch

| tokenizer | 200k-char Tamil doc | chars | tokens | encode ms | decode ms | round-trip |
|---|---|---|---|---|---|---|
| tok64k | 200k chars | 200,000 | 36,893 | 776.0 | 293.9 | yes |

| tokenizer | batch docs | MB | tok/s | docs/s | MB/s |
|---|---|---|---|---|---|
| tok64k | 512 | 0.5 | 584,645 | 2,506.5 | 2.24 |

## Streaming 100 MB (1 MB chunks, ids discarded)

| tokenizer | chunks | MB/s | peak RSS (MB) | baseline RSS (MB) | peak delta (MB) |
|---|---|---|---|---|---|
| tok64k | 100 | 0.64 | 569.5 | 566.7 | 2.8 |

## Memory per 1M token ids

| tokenizer | tokens measured | Python list (MB) | numpy uint16 (MB) | numpy int32 (MB) | bytes/token (list) |
|---|---|---|---|---|---|
| tok64k | 310,898 | 27.16 | 2.00 | 4.00 | 27.2 |

## Notes

- Timings are single-process, CPU only; `tokenizers` encodes in Rust and `encode_batch` uses its internal thread pool.
- RSS is process resident memory from `psutil`; the delta includes the loaded model, not the Python interpreter.
- The 100 MB stream keeps only the per-chunk token count, so peak RSS shows the encoder's own working set rather than the corpus.
- `tokenizer.json` is read from the OS page cache after the first pass, so load time is a warm-start figure.
