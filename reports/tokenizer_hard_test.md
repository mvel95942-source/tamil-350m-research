# Tokenizer hard test

- artifact: `artifacts\tok64k`
- vocab: 63,995
- sha256: `861ebf3409db01b3964686bbbf3a71f0b721cd88b571bcbdb3d67196397a25d8`
- checks: 12,156 · errors: 0 · warnings: 5
- wall time: 48.8s
- verdict: **PASS (with warnings)**

| suite | checks | errors | warnings | status |
|---|---|---|---|---|
| structure | 15 | 0 | 0 | ok |
| vocab_hygiene | 0 | 0 | 2 | WARN |
| akshara | 1,292 | 0 | 0 | ok |
| tamil_block | 768 | 0 | 0 | ok |
| malformed_tamil | 141 | 0 | 0 | ok |
| unicode | 326 | 0 | 0 | ok |
| fuzz | 9,001 | 0 | 0 | ok |
| whitespace_code | 192 | 0 | 0 | ok |
| specials_injection | 3 | 0 | 1 | WARN |
| known_collisions | 3 | 0 | 0 | ok |
| integration | 7 | 0 | 0 | ok |
| perf | 6 | 0 | 0 | ok |
| compare_tok32k | 402 | 0 | 2 | WARN |

## structure

vocab, id map, decoder/normalizer wiring

```json
{
  "vocab_size": 63995,
  "tokenizer_json_sha256": "861ebf3409db01b3964686bbbf3a71f0b721cd88b571bcbdb3d67196397a25d8",
  "vocab_composition": {
    "latin": 30173,
    "tamil": 27905,
    "other": 5521,
    "byte": 256,
    "special/tool/reserved": 32,
    "space-only": 44,
    "mixed_script": 54,
    "whitespace": 10
  }
}
```


## vocab_hygiene

orphan-sign and mixed-script pieces

```json
{
  "orphan_dependent_pieces": 18,
  "mixed_script_pieces": 54,
  "orphan_examples": [
    "\\u0bc2",
    "\\u0bcb",
    "\\u0bbe\\u0bcd",
    "\\u0bc7",
    "\\u0bc0",
    "\\u0bbf",
    "\\u0bbe",
    "\\u2581\\u0bc6",
    "\\u0bc8",
    "\\u0bd7"
  ]
}
```

**2 warning(s)**
- orphan-sign pieces: can only match malformed text (wasted vocab slots): `18` {"examples": ["\\u0bc2", "\\u0bcb", "\\u0bbe\\u0bcd", "\\u0bc7", "\\u0bc0", "\\u0bbf"]}
- mixed Tamil+Latin pieces (usually date/suffix artefacts): `54` {"examples": ["vaa\\u0bcd", "\\\\n\\ue001\\ue09b\\ue099\\ue06c", "naa\\u0bcd.", "\\u2581cho\\u0bcd", "\\\\n\\ue003\\ue06c", "a\\u0bcd"]}


## akshara

every table unit must be exactly one token and round-trip

```json
{
  "akshara_units": 312,
  "atomic_failures": 0
}
```


## tamil_block

U+0B80-U+0BFF standalone, after a consonant, after a space

```json
{
  "roundtrip_failures": 0
}
```


## malformed_tamil

degenerate Tamil, numbers, punctuation, invisible chars


## unicode

other scripts, emoji ZWJ, combining marks, bidi, controls

- note: lone surrogate raises TypeError (input must be sanitized upstream)


## fuzz

3000 seeded random mixed-script strings

```json
{
  "strings": 3000,
  "tokens": 66765,
  "chars": 65026,
  "chars_per_token": 0.9739534187074066,
  "non_idempotent": 0
}
```


## whitespace_code

indent runs, tabs, CRLF, code exactness


## specials_injection

can plain text reach special/tool/reserved ids?

```json
{
  "specials_reachable_from_text": 32,
  "specials_of": 32,
  "examples": [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<tool_call>"
  ]
}
```

**1 warning(s)**
- special tokens are matched inside plain text (injectable; decode drops them): `<pad>` {"count": 32, "examples": ["<pad>", "<s>", "</s>", "<unk>", "<|user|>", "<|assistant|>"]}


## known_collisions

raw PUA and U+2581 collide with the akshara/space encoding

```json
{
  "lossy_cases": 6,
  "pua_alphabet_size": 312
}
```

**Known lossy inputs (documented, must be sanitized upstream)**

- collision: `\ue000` -> `\u0b83` (expected `\ue000`)
- collision: `\ue001` -> `\u0b85` (expected `\ue001`)
- collision: `\u2581` -> ` ` (expected `\u2581`)
- collision: `a\u2581b` -> `a b` (expected `a\u2581b`)
- collision: `\u2581\u2581` -> `  ` (expected `\u2581\u2581`)
- collision: `\ue000\u0ba4\u0bae\u0bbf\u0bb4\u0bcd` -> `\u0b83\u0ba4\u0bae\u0bbf\u0bb4\u0bcd` (expected `\ue000\u0ba4\u0bae\u0bbf\u0bb4\u0bcd`)


## integration

transformers load, batch, save/reload, threads


## perf

encode/decode throughput and long-document latency

```json
{
  "tamil": {
    "input_mb": 1.0,
    "tokens": 60001,
    "encode_mb_s": 0.53,
    "encode_tok_s": 31766.0,
    "decode_mb_s": 1.74,
    "tokens_per_char": 0.162
  },
  "tanglish": {
    "input_mb": 1.0,
    "tokens": 370361,
    "encode_mb_s": 0.32,
    "encode_tok_s": 119718.0,
    "decode_mb_s": 1.26,
    "tokens_per_char": 0.37
  },
  "english": {
    "input_mb": 1.0,
    "tokens": 237501,
    "encode_mb_s": 0.26,
    "encode_tok_s": 61479.0,
    "decode_mb_s": 1.2,
    "tokens_per_char": 0.238
  },
  "code": {
    "input_mb": 1.0,
    "tokens": 296875,
    "encode_mb_s": 0.12,
    "encode_tok_s": 35699.0,
    "decode_mb_s": 1.35,
    "tokens_per_char": 0.352
  },
  "numbers": {
    "input_mb": 1.0,
    "tokens": 920000,
    "encode_mb_s": 0.29,
    "encode_tok_s": 266700.0,
    "decode_mb_s": 0.91,
    "tokens_per_char": 0.92
  },
  "emoji": {
    "input_mb": 1.0,
    "tokens": 446425,
    "encode_mb_s": 0.41,
    "encode_tok_s": 183428.0,
    "decode_mb_s": 1.22,
    "tokens_per_char": 1.562
  },
  "long_doc_200k_chars_ms": 384.5,
  "long_doc_tokens": 12001
}
```


## compare_tok32k

tok64k vs tok32k

```json
{
  "vocab_main": 63995,
  "vocab_other": 32000,
  "tokens_main": 15386,
  "tokens_other": 16073,
  "main_vs_other_token_ratio": 0.9573,
  "other_roundtrip_failures": 1,
  "main_leading_space_lossy": [],
  "other_leading_space_lossy": [
    " x",
    "  ",
    " a b",
    " 1",
    " \\u0ba4\\u0bae\\u0bbf\\u0bb4\\u0bcd",
    "    x"
  ]
}
```

**2 warning(s)**
- tok32k is lossy on fuzz input: `1` {"examples": [{"input": " \\u0be9.[)\\U0001f44d\\u0bd7\\u0628)\\u0300\\xe7\\u0b8f\\u0b8e\\uae00]<\\u0be7+8(/+\\u0bc6@\\u05d1-\\xe7(\\U0001f525}\\u0c17\\u0bd7;\\u0629\\u0bd0#\\u0bec?\\u0bd7\\U0001f600\\u0b83\\", "decoded": "\\u0be9.[)\\U0001f44d\\u0bd7\\u0628)\\u0300\\xe7\\u0b8f\\u0b8e\\uae00]<\\u0be7+8(/+\\u0bc6@\\u05d1-\\xe7(\\U0001f525}\\u0c17\\u0bd7;\\u0629\\u0bd0#\\u0bec?\\u0bd7\\U0001f600\\u0b83\\u"}]}
- tok32k loses leading whitespace (Metaspace prepend=always): `6` {"examples": [" x", "  ", " a b", " 1", " \\u0ba4\\u0bae\\u0bbf\\u0bb4\\u0bcd", "    x"]}

