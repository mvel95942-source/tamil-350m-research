# Tamil SFT data: web research and pilot decision

**Decision:** test a small, filtered sample of the Tamil `dialog_gen` subset of
[`microsoft/Updesh_beta`](https://huggingface.co/datasets/microsoft/Updesh_beta) as a
research-only pilot. Do not replace the existing Aya + IndicAlign data wholesale, and do
not publish a model trained on this pilot until the dataset license has been reviewed.

## Why this candidate

The dataset card describes Updesh as a synthetic multilingual instruction-tuning dataset
with Tamil (`tam_Taml`) splits, including `dialog_gen` conversations. It reports automatic
language identification, repetition filtering, deduplication, and sampled quality checks.
The dataset is much larger than needed for the first experiment; this pilot caps at 3,000
examples and applies additional format, Tamil-script, deduplication, and repetition checks.

The card also identifies important limitations: it is a beta release, synthetic, partly
translated, and has a **Microsoft Research License**. That makes it a candidate for
research evaluation, not a cleared source for a public/commercial Merzal release. Check
the license itself before use.

## Existing data is not automatically better

The current SFT recipe already used Aya Tamil and IndicAlign. IndicAlign is CC-BY-4.0, but
its card documents translated/generated components; the preview includes visibly bad or
factually wrong Tamil examples. More training on the full existing mixture is not the
default recommendation. First compare a sample of old and proposed data with a fluent Tamil
reviewer.

## Pilot preparation

`prepare_updesh_chat.py` downloads the published Tamil dialogue subset and emits JSONL in
the existing `tamil.sft` format. It removes system messages (the current trainer has no
system-role token), rejects malformed turn order, requires at least 55% Tamil-script
letters in assistant text, removes highly repetitive answers, deduplicates conversations,
and caps the pilot at 3,000 examples. These are mechanical filters, not a quality guarantee.

## Initial sample outcome: do not train on this pilot

The 3,000-row, 52 MB pilot was downloaded to `artifacts/updesh_ta_pilot.jsonl`. Manual spot
checks found a serious dialogue-role problem: in many examples the user presents themselves
as a Tamil expert/teacher, while the assistant answers as a student, praising the user's
explanation and asking for more. That teaches the wrong assistant behavior for ordinary
chat. It appears in multiple inspected rows, not only the first sample. The artifact is kept
for audit, but **must not be used for training** unless a robust transformation is designed
and a new sample receives human review. The current filters do not detect semantic role
inversion.

This is why script-ratio, deduplication, and repetition checks alone are insufficient to
approve SFT data. Candidate sources need human review of roles and answer quality before
they enter the training mixture.

Before training:

1. Inspect at least 100 randomly selected conversations with a fluent Tamil speaker.
2. Reject mistranslations, unnatural Tamil, repeated templates, unsafe or unsupported facts.
3. Keep a fixed held-out chat set that is not included in SFT.
4. Compare the current checkpoint and pilot SFT on the same prompts; retain the new model
   only if human review shows an improvement without worse repetition or factuality.

This dataset alone cannot make current factual knowledge reliable. For current facts, later
add retrieval/web search with visible sources and evaluate source-faithful answers.

## Sources checked

- Microsoft Updesh beta: <https://huggingface.co/datasets/microsoft/Updesh_beta>
- AI4Bharat IndicAlign: <https://huggingface.co/datasets/ai4bharat/indic-align>
- Cohere Aya dataset: <https://huggingface.co/datasets/CohereLabs/aya_dataset>
