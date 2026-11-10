# Training record (as run, not as planned)

Every number below is copied from a Kaggle log stored in `logs/`, or derived from the run's
wall time and tokens/s. Where a plan changed during the project, the change is recorded here.

Hardware for every run: **Kaggle free tier, 2 × NVIDIA T4 (16 GB each)**, 12-hour kernel
limit, weekly GPU quota of roughly 30 hours.

---

## Model

| Field | Value |
|---|---|
| Class | Qwen3-style dense decoder |
| Measured parameters | 380,155,904 |
| Layers / hidden / FFN | 20 / 1,024 / 3,072 |
| Heads / KV heads / head dim | 16 / 8 / 128 |
| Context | 4,096 |
| Vocabulary | 32,000 |
| Embeddings | tied |
| Attention implementation | custom `sdpa_expand_kv` (see below) |

Full description in `MODEL_CARD.md`.

---

## Pretraining runs

| Run | Kernel version | Wall budget | Actual wall | Throughput | Start step | End step | Result |
|---|---|---:|---:|---:|---:|---:|---|
| v3 | earlier | 9.1 h | 9.1 h | 5,260 tok/s | 890 | 1,484 | exit 0 |
| v4 | 4 | 11.3 h | 11.39 h | 6,486 tok/s | 1,484 | 2,423 | exit 0 |
| v5 | 5 | 11.7 h | 11.8 h | 6,426 tok/s | 2,423 | 3,361 | exit 0 |
| v6 | 6 | 1.6 h | 23 s | — | — | — | **failed at startup** |
| v7 | 7 | 1.6 h | 1.81 h | 6,311 tok/s | 3,361 | **3,490** | exit 0 |

One epoch is **3,536 steps** at 262,144 tokens per step, for 927,401,984 tokens. The final
run stopped 46 steps short (98.7%) and the remaining steps were deliberately not run: they
would have consumed the quota margin reserved for SFT. Loss at step 3,490 was 3.29, and the
curve had flattened.

The exact end step is read from the run's own `log.jsonl` (`logs/v7_run_log.jsonl`), not
inferred from throughput. Earlier estimates in this project were wrong by about 100 steps for
v5, which is why the checkpoint step is now the only accepted source.

Logs: `logs/v4_final.log`, `logs/v5_final.log`, `logs/v6_error.log`.

### Exact command

```bash
torchrun --nproc_per_node=2 -m tamil.train \
  --bin_path artifacts/train_tok32k.bin \
  --tokenizer_path artifacts/tok32k/tokenizer.json \
  --out_dir artifacts/run \
  --total_tokens 927000000 \
  --wall_budget_h 11.3 \
  --micro_bs 2 --grad_accum 64 --grad_ckpt false \
  --resume_from /kaggle/input/tamil-pretrain/repo/artifacts/run/ckpt.pt
```

Environment:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAMIL_LOSS_CHUNK=512
```

Effective batch: 2 micro-batch × 64 accumulation × 2 ranks = **256 sequences per optimizer
step**, 262,144 tokens.

### Why the run is shaped this way

- **T4 memory is the binding constraint, not compute.** The first attempt died on its first
  backward pass. Two changes fixed it without disabling activation checkpointing:
  computing logits in 512-token chunks (64 MB instead of 250 MB), and using a custom SDPA
  wrapper that expands KV heads up front. The wrapper matters because Hugging Face's default
  SDPA path passes `enable_gqa=True` when there is no mask, which on this stack fell back to
  the math kernel and stored a full score grid per head — an avoidable out-of-memory.
- **Removing activation checkpointing recovered roughly 30% throughput**, taking the run from
  5,260 to about 6,450 tokens/s once the memory fix was in place.
- **Checkpoints are resumable.** Each session resumes from the newest checkpoint by step
  number, not by filename, and writes to the same output directory.

---

## Supervised fine-tuning

Kernel: `kaggle_sft_clean/tamil-sft-clean.py`. Pushed as kernel version 1 on
**2026-11-10 14:46**, after the pretraining checkpoint reached step 3,490.

| Setting | Value |
|---|---|
| Base checkpoint | latest pretraining checkpoint (highest step) |
| Data | Aya Tamil + 4 IndicAlign Tamil subsets, cleaned, plus synthetic ×3 |
| Max conversations | 40,000 after filtering and deduplication; synthetic rows guaranteed to survive the cap |
| Epochs | 2 (reduced from 3 to fit the remaining quota) |
| Learning rate | 3e-5, cosine to 3e-6, 100 warmup steps |
| Micro-batch / accumulation | 2 / 16 |
| Context | 1,024 |
| Wall budget | 3.5 h |
| Precision | fp16 autocast |
| Output | HF-format model plus in-kernel 15-case generations |

Command inside the kernel:

```bash
python -m tamil.sft \
  --base_ckpt <checkpoint> \
  --tokenizer_path artifacts/tok32k/tokenizer.json \
  --data_path artifacts/sft_ta_clean.jsonl \
  --out_dir /kaggle/working/sft_run \
  --lr 3e-5 --min_lr 3e-6 --warmup 100 \
  --epochs 3 --micro_bs 2 --grad_accum 16 --ctx 1024 \
  --wall_budget_h 4.5 --ckpt_every_min 20
```

The wall budget was cut from 5.5 h to 4.5 h, then to 3.5 h with 2 epochs when the remaining
quota was measured against the SFT cost.

---

## Compute accounting

| Item | GPU hours |
|---|---:|
| v3 (previous quota window) | 9.1 |
| v4 | 11.39 |
| v5 | 11.8 |
| v6 (failed after 23 s) | 0.006 |
| v7 (continuation) | 1.81 |
| SFT (running, capped) | 3.5 |

Pretraining total across all sessions: **34.1 h**. Inside the current quota window,
v4 + v5 + v7 + SFT is about **28.5 h**, against roughly 30 h available. That margin is why the
last 46 pretraining steps were skipped and why SFT runs 2 epochs instead of 3.

---

## Deviations from the original plan

| Planned | Actual | Why |
|---|---|---|
| One 11.5 h run to finish pretraining | Four sessions (v3–v7) | quota ceiling and the 12 h kernel limit |
| Finish all 3,536 steps | stopped at 3,490 (98.7%) | 46 steps were not worth losing the SFT margin |
| SFT wall 5.5 h, 3 epochs | 3.5 h, 2 epochs | protect the remaining quota |
| v6 finishes the epoch | failed in 23 s, retried as v7 | code-root change broke a regex patch |
| Dataset code was current | it was stale | fixed by uploading dataset version 2 |
| Unattended chain runs to completion | paused during two DNS outages and for the v6→v7 fix | see `docs/FAILURES.md` |

The project used the **Kaggle container image available on each run date**. Library versions
were not pinned, so a future rerun may behave differently. That is a reproducibility gap, and
it is stated rather than hidden.
