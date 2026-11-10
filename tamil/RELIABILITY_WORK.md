# 350M Tamil checkpoint: reliability work

## Current truth

The available local runnable checkpoint is the 350M **SFT preview** at
`%LOCALAPPDATA%\Temp\opencode\kaggle-sft\sft_preview`. It was fine-tuned on the existing
Aya + IndicAlign mixture. The original pre-SFT checkpoint is not available locally, and the
Kaggle pretraining kernel is reported as `ERROR`; therefore, a before/after SFT claim or a
reliability improvement claim cannot yet be made. The Updesh sample is explicitly excluded:
manual inspection found semantic role inversion.

## First reliability pass

- `eval/tamil_reliability_holdout.jsonl`: fixed 15-case probe spanning arithmetic, basic
  science, Tamil grammar, instruction following, correction, ambiguity, and requests for
  live/current information. Cases are not included in the proposed seed training data.
- `scripts/eval_tamil_reliability.py`: deterministic baseline runner; saves raw model
  generations with scoring criteria for human review. Criteria are not an automatic metric.
- The first run used an untrained synthetic system-message prefix and is superseded. The
  aligned run now uses the exact training/chat template (user turn followed by assistant
  generation marker). It still failed all 15 held-out probes: arithmetic, science, grammar,
  following constraints, correcting a false premise, and calibrated uncertainty. Raw
  generations are saved at `reports/tamil_reliability_preview_aligned.json`. This is a
  decisive no-go for presenting the current SFT preview as a usable chat model; it is not
  evidence that any new training has improved it.
- `tamil/reliability_seed.jsonl`: small, hand-authored candidate set with answer correctness,
  concise instruction following, correction rather than sycophancy, and explicit uncertainty.
  It is **not approved for training yet**: a fluent Tamil reviewer must check naturalness,
  facts, and wording. It intentionally contains no Updesh rows.

## Gate before any fine-tuning

1. Run the baseline script and review every held-out answer; score each criterion pass/fail
   and note repetition, language quality, and unsupported claims.
2. Have a fluent Tamil reviewer audit every seed example; fix/remove any unnatural or wrong
   item. Check train/holdout prompt and answer overlap.
3. Recover the exact pre-SFT checkpoint and compare it to the SFT preview on the same frozen
   holdout. Do not train if SFT is already worse on reliability dimensions.
4. Only then run a small, resumable pilot from a copy of the checkpoint, evaluate the exact
   same holdout, and keep it only if factuality, answer accuracy, and repetition improve
   without damaging language or instruction following.

Training cannot make a 350M model a dependable source of changing facts. For weather, news,
current population, and other time-sensitive questions, use retrieval/live sources and cite
them; the desired model behavior without retrieval is to state the limitation rather than
invent a number.

## Kaggle quota action (2026-11-06)

The previous pretraining resume failed at the first backward pass: CUDA OOM while allocating
250 MiB for chunked logits on both T4s. This was a memory configuration failure, not evidence
that the checkpoint or corpus is invalid. A new run has been submitted to the existing private
`velmurugan04/tamil-pretrain` kernel (version 3), with the existing packed tok32k corpus and
the previous checkpoint as input. It halves micro-batch from 4 to 2, doubles accumulation
from 32 to 64 (preserving the effective batch), enables activation checkpointing, enables
expandable allocator segments, and caps this pre-reset run at 9 hours. No noisy Updesh data or
unreviewed SFT seed is used in this run. Checkpoint must be fetched/attached before tomorrow's
larger training session.

Tomorrow's larger run is **not cleared to start blindly**: check this job's final step/loss,
confirm output checkpoint integrity, then continue from its saved checkpoint with the same
memory-safe settings and the pre-registered 15-case reliability probe. SFT resumes only after
the training examples pass a fluent Tamil review; larger model scaling is a separate decision.
