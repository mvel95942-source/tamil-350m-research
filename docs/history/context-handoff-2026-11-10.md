# Context handoff — Tamil 350M (2026-11-07)

> Historical working note, kept for provenance. It reflects the state on 2026-11-07 through
> 2026-11-10 and contains internal paths. It is superseded by `README.md` and `docs/`.

Compact state for resuming after a context reset. This was the source of truth for the
current run at the time, not the chat history.

## Objective

Finish pretraining and supervised training of the from-scratch 350M Tamil chat model on the
weekly Kaggle quota (~30 GPU h, T4 x2), then judge it only on the frozen 15-case reliability
holdout. A separate track selects an already fine-tuned Gemma derivative; it is not part of
these GPU runs.

## Verified state

- `tamil-pretrain` v3 completed with exit 0 after 9.1 h. It resumed at step 890 and ended at
  **step 1484**, loss ~3.64, 5,260 tok/s.
- `tamil-pretrain` v4 completed **2026-11-08** with `train exit=0` after 11.39 h,
  **6,486 tok/s**, ~1,015 steps → **step ~2,499**. Log: `artifacts/logs/v4_final.log`.
- Checkpoint downloaded to
  `%LOCALAPPDATA%\Temp\opencode\kaggle-pretrain-v3\repo\artifacts\run\ckpt.pt`
  (4,168,927,311 bytes) and verified: `torch.load(..., weights_only=True)` succeeds; keys are
  `model`, `optim`, `scaler`, `step`; 380,155,904 parameters; fp32 tensors.
- Packed corpus: `artifacts/train_tok32k.bin`, 927,436,918 tokens; tokenizer
  `artifacts/tok32k/tokenizer.json`. One epoch = 3,536 steps at 262,144 tokens/step.
- The 350M SFT preview failed all 15 held-out probes. Evidence:
  `reports/tamil_reliability_preview_aligned.json`.

## Current GPU plan

1. **v4 pretrain — COMPLETE** (`train exit=0`, 6,486 tok/s, step ~2,499).
2. **v5 pretrain — QUEUED/RUNNING** (pushed 2026-11-08 17:35, kernel version 5).
   Same T4 x2 config; wall budget raised to 11.7 h to finish the 3,536-step epoch.
   At 6,486 tok/s, 11.7 h adds ~1,040 steps → ~step 3,540.
3. **`tamil-sft-clean`** — pushed automatically by `scripts/auto_chain.py` when v5 completes.
   Single GPU, ~5.5 h cap. Mixture: Aya (Tamil) + IndicAlign (Tamil), blocklist-filtered,
   plus `artifacts/sft_reliability_ta.jsonl` repeated **3x**. lr 3e-5, warmup 100, 3 epochs,
   `micro_bs=2`, `grad_accum=16`, ctx 1024. Scores the 15 cases in-kernel and exports an
   HF-format model to `/kaggle/working/sft_clean/`.

Total ≈ 11.39 + 11.7 + 5.5 = 28.6 GPU h, inside the ~30 h weekly quota.

## Unattended chain

`scripts/auto_chain.py` polls Kaggle and pushes the next stage. It is idempotent and
resumable after a reboot via `artifacts/auto_chain_state.json`.

- Windows scheduled task **MerzalTamilAutoChain**: every 30 min (interactive session).
- Startup shortcut: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\MerzalTamilAutoChain.lnk`.
- Log: `artifacts/auto_chain.log`; one-line status: `artifacts/auto_chain_status.txt`.
- Power: lid close on AC = do nothing; AC sleep/hibernate/disk timeouts = never.
- When v5 completes: pushes SFT, downloads the model to `~/models/tamil-350m-sft`, runs
  `scripts/eval_tamil_reliability.py`, writes `reports/tamil_reliability_sft_clean.json`.
- Honest limit: pretraining continues on Kaggle while the laptop is off; the *next push*
  waits until the laptop is back on and a task tick fires.

## SFT reliability data

`scripts/build_sft_reliability_ta.py` → `artifacts/sft_reliability_ta.jsonl` (3,889 rows):

| group | rows | source of truth |
|---|---:|---|
| arithmetic | 3,013 | answers computed in Python |
| correction | 727 | false equation + computed fix |
| instruction | 62 | fixed lists/facts |
| grammar | 50 | reviewed plural/verb list |
| uncertainty | 23 | refusal for live/ambiguous questions |
| science | 14 | small fixed fact list |

Contamination guard: excludes exact holdout prompts, the holdout operand pairs in both
orders, `'புத்தகம்'`, `'அவள் பாடினாள்'`, the Chennai-weather probe, and the
India-population probe. Verified 0 exact and 0 normalized prompt overlap.

## Commands

```powershell
# status / logs
kaggle kernels status velmurugan04/tamil-pretrain
kaggle kernels logs velmurugan04/tamil-pretrain
kaggle kernels output velmurugan04/tamil-pretrain -p <new-dir>

# push next pretrain session (reuses kaggle/kernel-metadata.json)
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200

# build/refresh synthetic dataset
python scripts\build_sft_reliability_ta.py
kaggle datasets version -p kaggle_datasets\tamil-sft-reliability -m "update"

# push SFT (folder has its own kernel-metadata.json)
kaggle kernels push -p kaggle_sft_clean --accelerator NvidiaTeslaT4 --timeout 43200

# local reliability probe after downloading sft_clean
python scripts\eval_tamil_reliability.py --model <path> --out reports\<name>.json

# local chat server
python scripts\chat_tamil_web.py     # http://127.0.0.1:8765/
```

## Hard gates

- Do **not** train on the Updesh pilot (semantic role inversion) or `tamil/reliability_seed.jsonl`
  until a fluent Tamil reviewer approves it.
- The 15-case holdout is never trained on; keep `eval/tamil_reliability_holdout.jsonl` frozen.
- Do not claim reliability from training loss. Only the fixed probe plus human scoring counts.
- A 350M model with <1B tokens cannot be a dependable source of live facts. State limitations;
  use retrieval for weather/news/population/prices.
- 1.2B-from-scratch is **not** launchable on free Kaggle: no config in `tamil/model.py`
  beyond 350M/30M/tiny, 13B-token corpus unprepared, ~70 rented H100-hours estimated.

## Key files

- `kaggle/tamil-pretrain-resume.py`, `kaggle/kernel-metadata.json` — pretrain continuation.
- `kaggle_sft_clean/tamil-sft-clean.py`, `kernel-metadata.json` — cleaned SFT run.
- `tamil/train.py` — `LOSS_CHUNK` now reads `TAMIL_LOSS_CHUNK` (default 2048).
- `tamil/sft.py`, `tamil/sources.py`, `tamil/model.py` — trainer, datasets, 350M config.
- `scripts/build_sft_reliability_ta.py` — synthetic reliability data.
- `scripts/eval_tamil_reliability.py` — local holdout runner.
- `eval/tamil_reliability_holdout.jsonl` — frozen gate.
- `reports/tamil_reliability_preview_aligned.json` — current failure evidence.
- `TOMORROW_TRAINING_READINESS.md`, `tamil/RELIABILITY_WORK.md`, `tamil/SFT_DATA_RESEARCH.md`.
