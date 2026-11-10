# Paper draft: Akshara-aware Tamil tokenizer

Files: `main.tex` (ACL review format), `references.bib` (27 verified entries), `paper.md` (readable copy with Mermaid), `figures/*.svg`, `WRITING_NOTES.md` (rules + citation status).

## Build the PDF

1. Get `acl.sty` and `acl_natbib.bst` from https://github.com/acl-org/acl-style-files and put them next to `main.tex`. They are not vendored here.
2. Tamil needs a Unicode engine. Use LuaLaTeX or XeLaTeX, not pdfLaTeX. Install a Tamil font (Noto Sans Tamil) or change `\newfontfamily\tamilfont` in `main.tex`.
3. Overleaf: upload the folder, set the compiler to LuaLaTeX (Menu > Compiler), upload the two ACL files.
4. Local: `latexmk -lualatex main.tex`.
5. Status: `pdflatex` and `latexmk` are not installed on this machine, so the draft has NOT been compiled. Expect small TikZ or font fixes on first build. The `\usepackage{times}` line may clash with LuaLaTeX; delete it if so (see `acl_lualatex.tex`).

## PENDING cells and what fills them

| Cell group | Where | Script |
|---|---|---|
| Fertility, all baselines, 4 languages, bytes/token, cut letters, round-trip, vocab sizes | Table 2 | `python -m tamil.fertility --ours artifacts/tok32k artifacts/tok48k --n 2000`. Add Gemma 3 (`unsloth/gemma-3-1b-it`) to `BASELINES` in `tamil/fertility.py` first; Qwen3 failed to load in the smoke run. |
| Ablation BPB (Tamil, Tanglish, English), model size, shards, tokens | Table 3 | `kaggle/03_ablation.py` then `kaggle/04_eval.py` |
| 350M pretrained BPB and 4 task scores | Table 4 row 1 | `kaggle/02_train.py` then `kaggle/04_eval.py` |
| 350M + SFT BPB and 4 task scores | Table 4 row 2 | `kaggle/05_sft.py` then `kaggle/04_eval.py` |
| Section 7 analysis | text | write only after Tables 2 to 4 are filled |

Open decisions: the ablation model size (30M fallback or 350M); a same-size baseline model for Table 4; IndicQA has no scoring code; the tokenizer sample size (brief says about 1 GB, the build script defaults to 2 GB with an override, so confirm which run produced the shipped tokenizers).

## Submission checklist

- [ ] All PENDING cells filled from logged runs (Kaggle notebook + version, or commit + command).
- [ ] Re-run the smoke numbers on the full benchmark; delete or keep Table 1 as a labelled smoke test.
- [ ] Main text within 8 pages (long) or 4 (short) in review mode. Limitations and Ethical Considerations sit after the Conclusion, outside the limit.
- [ ] Anonymous: no names, no repo URLs, no "we previously"; supplementary code anonymized.
- [ ] Responsible NLP checklist: licences, intended use, data statistics, parameter count, compute (2xT4 hours), hyperparameters, single run, package versions, AI-assistant disclosure.
- [ ] Re-check each dataset licence on its page (strings come from `tamil/sources.py`).
- [ ] Verify the unverified citations listed in `WRITING_NOTES.md` (mT5, Qwen2.5) or leave them out.
- [ ] Confirm the "57 hours" estimate: it assumes the 7,336 tokens/s is the total across both GPUs.
- [ ] arXiv (cs.CL): a first-time submitter may need an endorsement; ask a cs.CL author for it early.
- [ ] ARR or a workshop: check the current deadlines and the workshop's own page limits on the venue site. No dates are stated here because none were verified.
- [ ] Release tokenizer, model and code with a model card that states the share-alike terms.
