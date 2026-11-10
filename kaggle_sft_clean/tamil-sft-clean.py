"""Kaggle GPU job: supervised fine-tuning of the 350M Tamil checkpoint on a cleaned mixture.

Mixture:
  * Aya (Tamil) + IndicAlign (Tamil), blocklist-filtered (as before), and
  * a small, mechanically verifiable reliability set (arithmetic/instruction/correction/
    uncertainty/grammar) mounted as a dataset and repeated to carry real weight.

The 15-case reliability holdout is generated here only for scoring; it is never trained on.
Exports an HF-format model plus a reliability report to /kaggle/working/sft_clean/.
"""
import glob
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time

T0 = time.time()
MAX_CONVS = int(os.environ.get("SFT_MAX_CONVS", "40000"))
INDIC_SCAN = int(os.environ.get("SFT_INDIC_SCAN", "30000"))
EPOCHS = os.environ.get("SFT_EPOCHS", "3")
SYNTH_REPEAT = int(os.environ.get("SFT_SYNTH_REPEAT", "3"))
VOCAB = "tok32k"

code = glob.glob("/kaggle/input/**/tamil/akshara.py", recursive=True)
if not code:
    raise SystemExit("code dataset not mounted")
repo = "/kaggle/working/repo"
shutil.copytree(os.path.dirname(os.path.dirname(code[0])), repo, dirs_exist_ok=True)
cks = sorted(glob.glob("/kaggle/input/**/artifacts/run/ckpt.pt", recursive=True), key=os.path.getmtime)
print("checkpoint candidates:", cks, flush=True)
if not cks:
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) <= 6:
            print(root, files[:6])
    raise SystemExit("no ckpt.pt under /kaggle/input")
ckpt = cks[-1]
os.chdir(repo)
sys.path.insert(0, repo)
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

step_base = torch.load(ckpt, map_location="cpu", weights_only=True)["step"]
print(f"base ckpt {ckpt} step={step_base}", flush=True)

# ---- Aya + IndicAlign, per-source counts kept for the report
from tamil import sources
from tamil.corpus import load_rows
from tamil.sft import build_sft_rows

counts = {}


def capped_rows(src):
    it = load_rows(src)
    if "indic" in src.hf_id:
        it = itertools.islice(it, INDIC_SCAN)
    return it


t = time.time()
convs = []
import tamil.sft as S

for src in sources.SFT:
    old = S.sources.SFT
    S.sources.SFT = [src]
    try:
        rows = build_sft_rows(lambda s, _src=src: capped_rows(_src), set(np.load("artifacts/blocklist.npy").tolist()))
    except ValueError as e:
        print("SOURCE FAILED:", e, flush=True)
        rows = []
    finally:
        S.sources.SFT = old
    counts[src.hf_id] = len(rows)
    print(f"{src.hf_id}: {len(rows):,} Tamil conversations ({time.time()-t:.0f}s)", flush=True)
    convs += rows
if not convs:
    raise SystemExit("no SFT data")


def keep(c: list[dict]) -> bool:
    """Cheap quality gate: non-trivial Tamil/English answer, no URL spam, deduped prompt."""
    ans = " ".join(t["content"] for t in c if t["role"] == "assistant")
    if not (8 <= len(ans) <= 2000):
        return False
    if len(ans.split()) < 3:
        return False
    low = ans.lower()
    if "http://" in low or "https://" in low or "www." in low:
        return False
    return True


before = len(convs)
convs = [c for c in convs if keep(c)]
seen_prompts: set[str] = set()
deduped = []
for c in convs:
    key = c[0]["content"].strip()
    if key in seen_prompts:
        continue
    seen_prompts.add(key)
    deduped.append(c)
convs = deduped
print(f"quality/dedup kept {len(convs):,} of {before:,}", flush=True)

# ---- synthetic reliability set (mounted dataset); never contains a holdout prompt
synth_files = glob.glob("/kaggle/input/**/sft_reliability_ta.jsonl", recursive=True)
if not synth_files:
    raise SystemExit("synthetic reliability dataset not mounted")
synth = [json.loads(l) for l in open(sorted(synth_files)[-1], encoding="utf-8") if l.strip()]
print(f"synthetic reliability conversations: {len(synth):,}", flush=True)
convs += synth * SYNTH_REPEAT
random.Random(0).shuffle(convs)
convs = convs[:MAX_CONVS]
print(f"SFT conversations used: {len(convs):,} (synthetic repeated {SYNTH_REPEAT}x)", flush=True)
with open("artifacts/sft_ta_clean.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(json.dumps(c, ensure_ascii=False) for c in convs))

# ---- SFT
tok_path = f"artifacts/{VOCAB}/tokenizer.json"
out_dir = "/kaggle/working/sft_run"
cmd = [sys.executable, "-m", "tamil.sft", "--base_ckpt", ckpt, "--tokenizer_path", tok_path,
       "--data_path", "artifacts/sft_ta_clean.jsonl", "--out_dir", out_dir,
       "--lr", "3e-5", "--min_lr", "3e-6", "--warmup", "100",
       "--epochs", EPOCHS, "--micro_bs", "2", "--grad_accum", "16", "--ctx", "1024",
       "--wall_budget_h", "4.5", "--ckpt_every_min", "20"]
print("=====", " ".join(cmd), flush=True)
rc = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=repo,
                                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")).returncode
print(f"===== sft exit {rc} after {(time.time()-T0)/60:.0f} min", flush=True)
if rc != 0:
    sys.exit(rc)

# ---- reliability probe + export
from tokenizers import Tokenizer
from tamil.model import make_model, EOS_ID

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer.from_file(tok_path)
model = make_model(tok.get_vocab_size(), "350m")
ck = torch.load(f"{out_dir}/ckpt.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ck["model"])
model = model.to(dev).eval().half() if dev == "cuda" else model.eval()

holdout = [json.loads(l) for l in open("eval/tamil_reliability_holdout.jsonl", encoding="utf-8")]


def chat_ids(prompt: str) -> list[int]:
    return ([1, tok.token_to_id("<|user|>")]
            + tok.encode("\n" + prompt, add_special_tokens=False).ids
            + [EOS_ID, tok.token_to_id("<|assistant|>")]
            + tok.encode("\n", add_special_tokens=False).ids)


cases = []
for case in holdout:
    ids = chat_ids(case["prompt"])
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        y = model.generate(x, attention_mask=torch.ones_like(x), do_sample=False,
                           repetition_penalty=1.15, no_repeat_ngram_size=3,
                           max_new_tokens=96, eos_token_id=EOS_ID, pad_token_id=0)
    answer = tok.decode(y[0, len(ids):].tolist(), skip_special_tokens=True).strip()
    print(f"\n[{case['id']}] {case['prompt']}\n=> {answer}", flush=True)
    cases.append({**case, "answer": answer})

os.makedirs("/kaggle/working/sft_clean", exist_ok=True)
with open("/kaggle/working/sft_clean/reliability.json", "w", encoding="utf-8") as f:
    json.dump({"base_step": step_base, "sft_step": ck["step"], "cases": cases}, f,
              ensure_ascii=False, indent=2)
model.save_pretrained("/kaggle/working/sft_clean")
shutil.copy(tok_path, "/kaggle/working/sft_clean/tokenizer.json")
shutil.copy(f"artifacts/{VOCAB}/tokenizer_config.json", "/kaggle/working/sft_clean/tokenizer_config.json")
shutil.copy(f"{out_dir}/log.jsonl", "/kaggle/working/sft_clean/sft_log.jsonl")
json.dump({"source_counts": counts, "synthetic": len(synth), "synthetic_repeat": SYNTH_REPEAT,
           "used": len(convs), "base_step": step_base, "sft_step": ck["step"]},
          open("/kaggle/working/sft_clean/sft_data_stats.json", "w", encoding="utf-8"), indent=2)
shutil.rmtree(out_dir, ignore_errors=True)  # drop the multi-GB optimizer checkpoint
shutil.rmtree(repo, ignore_errors=True)
print(f"===== done in {(time.time()-T0)/60:.0f} min", flush=True)
