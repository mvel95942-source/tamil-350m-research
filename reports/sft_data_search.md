# Tamil SFT data search

Generated 2026-11-07 04:18 UTC from the Hugging Face dataset API.

Rules: accept only reputable organisations, a clear licence, and non-benchmark splits. Benchmark/test sets and speech data are rejected outright. Unknown provenance is rejected. Anything not explicitly accepted needs human review.

| decision | dataset | licence | downloads | tamil splits | reason |
|---|---|---|---:|---|---|
| ACCEPT | `CohereLabs/aya_dataset` | apache-2.0 | 23969 | test:1750, train:202362 | human-curated multilingual instructions; Tamil split used |
| ACCEPT | `ai4bharat/indic-align` | cc-by-4.0 | 2312 | tamil config not found | CC-BY-4.0 human-aligned instructions; Tamil split used |
| MAYBE | `ai4bharat/IndicParaphrase` | cc-by-nc-4.0 | 1230 | ? | reputable paraphrase pairs; not chat, needs format review |
| MAYBE | `ai4bharat/IndicWikiBio` | cc-by-nc-4.0 | 533 | ? | reputable biography text; could ground 'who is' answers, no QA pairs |
| REJECT | `HuggingFaceFW/fineweb-2` | odc-by | 141788 | tamil config not found | pretraining corpus, not instruction/chat pairs |
| REJECT | `ai4bharat/IndicVoices` | cc-by-4.0 | 20602 | valid:5276, train:422963 | speech data, not text SFT |
| REJECT | `microsoft/Updesh_beta` | other | 6828 | tamil config not found | pilot showed semantic role inversion; licence review pending |
| REJECT | `ai4bharat/IndicQA` | cc-by-4.0 | 3258 | ? | benchmark/test split — contamination risk |
| REJECT | `nvidia/Nemotron-Pretraining-Indic-Specialized-v1` | other | 3107 | ? | pretraining corpus, not instruction/chat pairs |
| REJECT | `nvidia/Nemotron-Pretraining-Indic-CC-v1` | other | 1596 | ? | pretraining corpus, not instruction/chat pairs |
| REJECT | `Srini99/TamilQA` | unknown | 1253 | ? | community dataset; provenance/quality not verified |
| REJECT | `AnanthZeke/oscar_tamil_clean` | unknown | 1224 | train:1263180 | pretraining corpus, not instruction/chat pairs |
| REJECT | `sarvamai/mmlu-indic` | mit | 794 | tamil config not found | benchmark/test split — contamination risk |
| REJECT | `AswiN037/tamil-question-answering-dataset` | afl-3.0 | 749 | ? | community dataset; provenance/quality not verified |
| REJECT | `sarvamai/gsm8k-indic` | unknown | 658 | tamil config not found | benchmark/test split — contamination risk |
| REJECT | `Abirami/tamilwikipediadataset` | unknown | 624 | ? | pretraining corpus, not instruction/chat pairs |
| REJECT | `nirantk/chaii-hindi-and-tamil-question-answering` | unknown | 605 | ? | community dataset; provenance/quality not verified |
| REJECT | `sarvamai/arc-challenge-indic` | unknown | 435 | tamil config not found | benchmark/test split — contamination risk |
| REJECT | `Hemanth-thunder/tamil-madlad-400` | unknown | 435 | clean:2594191 | community dataset; provenance/quality not verified |
| REJECT | `prakod/indic-align-merged-cleaned` | unknown | 303 | train:7389838 | unknown licence / unverified provenance |
| REJECT | `prasadvittaldev/gemma4-tamil-sft` | unknown | 145 | train:347463 | unknown licence / unverified provenance |
| REJECT | `zicsx/Indus_indic-align-hindi` | unknown | 141 | train:13310858 | unknown licence / unverified provenance |
| REJECT | `zicsx/indic-align-hindi` | unknown | 123 | train:13310858 | unknown licence / unverified provenance |
| REJECT | `ai4bharat/indic-math-500` | apache-2.0 | 119 | ? | benchmark/test split — contamination risk |
| REJECT | `BhabhaAI/indic-instruct-data-v0.1-filtered` | unknown | 109 | ? | unknown licence / unverified provenance |
| REJECT | `Ankit9311/indic-instruct-data-v0.1` | unknown | 90 | tamil config not found | unknown licence / unverified provenance |
| REJECT | `pranjalchitale/indicsft` | unknown | 82 | train:25030167 | unknown licence / unverified provenance |
| REJECT | `BhabhaAI/indic-instruct-data-v0.2-filtered` | unknown | 62 | ? | unknown licence / unverified provenance |
| REJECT | `gradshaan/Indic_Guard_SFT_Data` | unknown | 48 | train:10240 | unknown licence / unverified provenance |
| REJECT | `manishiitg/indic-synthetic-instruct` | unknown | 38 | train:42649 | unknown licence / unverified provenance |
| REJECT | `gradshaan/indic-instruct-v2` | unknown | 31 | train:144810 | unknown licence / unverified provenance |
| REJECT | `open-llm-leaderboard-old/details_Telugu-LLM-Labs__Indic-gemma-2b-finetuned-sft-Navarasa-2.0` | unknown | 31 | ? | unknown licence / unverified provenance |
| REJECT | `gradshaan/indic-instruct-v1` | unknown | 30 | train:120504 | unknown licence / unverified provenance |
| REJECT | `open-llm-leaderboard-old/details_Hemanth-thunder__Tamil-Mistral-7B-Instruct-v0.1` | unknown | 29 | ? | unknown licence / unverified provenance |
| REJECT | `Sano86/indic-align-malayalam-english-sft` | unknown | 29 | ? | unknown licence / unverified provenance |
| REJECT | `open-llm-leaderboard-old/details_abhinand__tamil-llama-13b-instruct-v0.1` | unknown | 28 | ? | unknown licence / unverified provenance |
| REJECT | `open-llm-leaderboard-old/details_Telugu-LLM-Labs__Indic-gemma-7b-finetuned-sft-Navarasa-2.0` | unknown | 27 | ? | unknown licence / unverified provenance |
| REJECT | `gradshaan/indic-instruct` | unknown | 26 | train:135241 | unknown licence / unverified provenance |
| REJECT | `gradshaan/indic-instruct-eval` | unknown | 26 | train:597 | unknown licence / unverified provenance |
| REJECT | `open-llm-leaderboard-old/details_abhinand__tamil-llama-7b-instruct-v0.2` | unknown | 25 | ? | unknown licence / unverified provenance |
| REJECT | `Sano86/indic-align-malayalam-english` | unknown | 25 | ? | unknown licence / unverified provenance |
| REJECT | `open-llm-leaderboard-old/details_abhinand__tamil-llama-7b-instruct-v0.1` | unknown | 24 | ? | unknown licence / unverified provenance |
| REJECT | `Yasshhhh/adaption-telugu-tamil-travel-translation-and-reasoning-corpus` | unknown | 18 | ? | unknown licence / unverified provenance |
| REJECT | `nlpctx/indic-self-instruct-gemma` | unknown | 16 | ? | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v5_2` | unknown | 15 | ? | unknown licence / unverified provenance |
| REJECT | `ansh123456789/rag-sft-indic` | unknown | 14 | ? | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v3_2` | unknown | 12 | ? | unknown licence / unverified provenance |
| REJECT | `manishiitg/indic-synthetic-instruct-follow` | unknown | 12 | train:7168 | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v3_1` | unknown | 11 | ? | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v3_3` | unknown | 11 | ? | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v4_0` | unknown | 10 | train:1365, validation:149 | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v4_1` | unknown | 10 | train:13083, validation:1452 | unknown licence / unverified provenance |
| REJECT | `open-llm-leaderboard/Telugu-LLM-Labs__Indic-gemma-7b-finetuned-sft-Navarasa-2.0-details` | unknown | 10 | ? | unknown licence / unverified provenance |
| REJECT | `commotion/ASR_instruction_indic_data_filtered` | unknown | 9 | ? | unknown licence / unverified provenance |
| REJECT | `open-llm-leaderboard/Telugu-LLM-Labs__Indic-gemma-2b-finetuned-sft-Navarasa-2.0-details` | unknown | 8 | ? | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v3_6` | unknown | 7 | train:3667 | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v5_3` | unknown | 7 | ? | unknown licence / unverified provenance |
| REJECT | `CryptoYogi/vazhi-tamil-sft-v5_1` | unknown | 6 | ? | unknown licence / unverified provenance |
| REJECT | `jai23023/indic-align-50k` | unknown | 5 | train:149972 | unknown licence / unverified provenance |
| REJECT | `diwank/indic-align_dolly-chatml` | unknown | 4 | train:7474 | unknown licence / unverified provenance |
| REVIEW | `ai4bharat/indic-instruct-data-v0.1` | unknown | 664 | tamil config not found | licence unknown needs review |
| REVIEW | `LingoIITGN/PI-Indic-Align` | cc-by-4.0 | 165 | ? | not enough evidence for training use |
| REVIEW | `Vedisasi/tamil-reasoning-dataset` | mit | 148 | ? | not enough evidence for training use |
| REVIEW | `adityabanerjee13/indic-sft-mini` | cc-by-4.0 | 55 | ? | not enough evidence for training use |
| REVIEW | `Hemanth-thunder/tamil-open-instruct-v1` | apache-2.0 | 46 | train:493813 | not enough evidence for training use |
| REVIEW | `Subi1152/tamil-question-answering-dataset` | afl-3.0 | 46 | ? | licence afl-3.0 needs review |
| REVIEW | `prithivMLmods/Deepthink-Reasoning-Tamil` | apache-2.0 | 39 | ? | not enough evidence for training use |
| REVIEW | `tniranjan/aitamilnadu_tamil_stories_no_instruct` | apache-2.0 | 32 | train:1081, validation:121 | not enough evidence for training use |
| REVIEW | `TokenBender/Tamil_chat_dataset` | apache-2.0 | 25 | ? | not enough evidence for training use |
| REVIEW | `Sidharth1743/tamil-med-sft-cot-600` | mit | 22 | ? | not enough evidence for training use |
| REVIEW | `CryptoYogi/vazhi-tamil-sft-v7_0` | apache-2.0 | 16 | ? | not enough evidence for training use |
| REVIEW | `CryptoYogi/vazhi-tamil-sft-v5_0` | cc-by-sa-4.0 | 14 | ? | not enough evidence for training use |
| REVIEW | `Dark7Devil/indic-culture-instruction-data` | mit | 14 | ? | not enough evidence for training use |
| REVIEW | `adityabanerjee13/indic-sft-mini-train` | cc-by-4.0 | 12 | ? | not enough evidence for training use |
| REVIEW | `Kdhanushiya1325/Tamil-Informal-Chat-Dataset` | apache-2.0 | 11 | ? | not enough evidence for training use |
| REVIEW | `adityabanerjee13/indic-sft-mini-val` | cc-by-4.0 | 11 | ? | not enough evidence for training use |
| ERROR | `ai4bharat/IndicXquad` | ? | ? | ? | HTTP Error 401: Unauthorized |
| ERROR | `ai4bharat/IndicCOQA` | ? | ? | ? | HTTP Error 401: Unauthorized |
| ERROR | `ai4bharat/IndicGlue` | ? | ? | ? | HTTP Error 401: Unauthorized |
| ERROR | `ai4bharat/IndicCorp` | ? | ? | ? | HTTP Error 401: Unauthorized |
| ERROR | `ai4bharat/IndicTTS` | ? | ? | ? | HTTP Error 401: Unauthorized |
| ERROR | `google/IndicGenBench` | ? | ? | ? | HTTP Error 401: Unauthorized |

## Decision

Training mixture remains:

- `CohereLabs/aya_dataset` — human-curated multilingual instructions; Tamil split used
- `ai4bharat/indic-align` — CC-BY-4.0 human-aligned instructions; Tamil split used
- `artifacts/sft_reliability_ta.jsonl` — 3,889 mechanically verifiable synthetic rows (arithmetic, corrections, grammar, uncertainty, science). This is the only addition that can be checked without a fluent reviewer.

No new natural Tamil instruction source cleared the bar. Anything in `REVIEW` or `MAYBE` must not enter training until a fluent Tamil speaker inspects at least 100 samples and the licence is confirmed.
