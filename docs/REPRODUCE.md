# Reproducing the work

The pipeline is built for **Kaggle free-tier notebooks** (2 × T4, 12-hour kernel limit,
roughly 30 GPU-hours per week). It also runs on any two-GPU box with enough VRAM, but the
memory settings below assume 16 GB per GPU.

Nothing here downloads model weights from a third party. Everything is trained from scratch.

---

## 0. Prerequisites

- Python 3.11 or newer.
- `torch`, `transformers`, `tokenizers`, `numpy`, `datasets`, `huggingface_hub`.
- The Kaggle CLI (`pip install kaggle`) and credentials for your own account.
- About 30 GPU-hours of quota for pretraining plus SFT, and disk for a 927M-token corpus.

> Library versions were not pinned during the original runs. If you need bit-level
> reproducibility, pin the Kaggle image from the run dates listed in `docs/TRAINING.md`.

---

## 1. Build the tokenizer

```bash
python kaggle/01_build_tokenizer.py
```

This produces `artifacts/tok32k/tokenizer.json` and `tokenizer_config.json`. The akshara
mapping is in `tamil/akshara.py`; the Unigram training recipe is in
`tamil/tokenizer_train.py`.

Verify it before spending GPU time:

```bash
python scripts/bench_tokenizer.py
python scripts/validate_tokenizer_holdout.py
```

Expected: Tamil at about 2.26 tokens/word, round-trip 100%, cut letters near 0.01%.

---

## 2. Build the packed corpus

```bash
python -m tamil.pack --vocab tok32k
```

This reads the sources registered in `tamil/sources.py`, applies the blocklist built from the
held-out documents, tokenizes, and writes:

- `artifacts/train_tok32k.bin` — flat uint16 tokens
- `artifacts/train_tok32k.idx.npy` — document starts
- `artifacts/train_tok32k.lang.npy` — language id per document
- `artifacts/train_tok32k.meta.json` — dtype metadata

Some upstream datasets (for example Sangraha) require accepting terms on Hugging Face first.
The packed corpus is **not** redistributed in this repository.

Target: 927,401,984 tokens = 3,536 steps × 262,144.

---

## 3. Create your Kaggle datasets

Three private datasets, all owned by you:

| Dataset | Contents |
|---|---|
| `<you>/tamil-tok-code` | `tamil/`, `kaggle/`, `artifacts/blocklist.npy`, `artifacts/tok32k/*`, `artifacts/tok48k/*`, `eval/tamil_reliability_holdout.jsonl` |
| `<you>/tamil-pack` | `artifacts/train_tok32k.bin` and its index files |
| `<you>/tamil-sft-reliability` | `sft_reliability_ta.jsonl` |

Then edit every `kernel-metadata.json` and replace `velmurugan04/...` with your own slugs.
The kernel source list must include the pretraining kernel itself, because each session mounts
the previous session's checkpoint.

---

## 4. Pretrain

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200
```

The resume script:

1. picks the code root from the dataset snapshot,
2. finds the checkpoint with the highest step number under `/kaggle/input`,
3. runs `torchrun --nproc_per_node=2 -m tamil.train ...`,
4. stops itself at the wall budget and leaves a new checkpoint in the kernel output.

Defaults: `micro_bs=2`, `grad_accum=64`, `grad_ckpt=false`, `TAMIL_LOSS_CHUNK=512`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Expect about **6,400–6,500 tokens/s**, roughly 1,040 steps per 11.7-hour session. Push again
after each session until step 3,536.

Check progress:

```bash
kaggle kernels status <you>/tamil-pretrain
kaggle kernels logs <you>/tamil-pretrain
```

Kaggle hides logs until a run finishes. The startup line prints the resume step.

---

## 5. Supervised fine-tuning

```bash
kaggle kernels push -p kaggle_sft_clean --accelerator NvidiaTeslaT4 --timeout 43200
```

The kernel:

1. loads Aya Tamil and the four IndicAlign Tamil subsets, filters and deduplicates,
2. adds the synthetic reliability set three times,
3. runs 2 epochs at lr 3e-5, context 1,024, micro-batch 2, accumulation 16,
4. generates the 15 holdout cases with greedy decoding,
5. exports an HF-format model plus `reliability.json`, `sft_log.jsonl`, `sft_data_stats.json`.

It should fail immediately, not after hours, if the holdout or synthetic file is missing. Do
not remove those checks.

---

## 6. Download and evaluate

```bash
kaggle kernels output <you>/tamil-sft-clean -p ./sft-out
python scripts/eval_tamil_reliability.py --model ./sft-out/sft_clean \
  --out reports/tamil_reliability_sft_clean.json
```

The local evaluator uses `transformers.AutoModelForCausalLM`, which is why the model is
exported in Qwen3 format. It matches the training prompt exactly and uses greedy decoding.

Read the generations yourself. The script does not claim to score them.

---

## 7. Run the chat UI locally

```bash
python scripts/chat_tamil_web.py --model ./sft-out/sft_clean
```

Default port 8765. On Windows, set `PYTHONIOENCODING=utf-8` before printing Tamil to a
console.

The optional facts layer starts disabled. If you enable it, keep its answers labelled as a
lookup; do not blend them into model-ability claims.

---

## Time and quota budget

| Stage | GPU hours |
|---|---:|
| Pretraining, epoch 1 (3 sessions) | ~32 |
| SFT | ~4.5 |
| Evaluation | minutes on CPU/GPU |

Pretraining does not fit in one weekly quota. Plan for three sessions and checkpoint between
them.

---

## Known pitfalls

- **Do not select checkpoints by file modification time.** Mount times are not run times.
  Compare `step`.
- **Do not mount the kernel's own output as code.** Select the dataset copy explicitly, or
  the trainer version becomes whatever the last run left behind.
- **Do not train without the blocklist.** Held-out documents must be removed by sentence.
- **Do not add benchmark/test splits.** They are the evaluation.
- **T4 memory:** keep the logits chunk at 512 and the custom SDPA wrapper enabled, or the run
  will OOM on its first backward pass.
