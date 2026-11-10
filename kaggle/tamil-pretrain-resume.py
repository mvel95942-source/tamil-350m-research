"""Quota-bounded continuation for the 350M Tamil pretraining run.

Reuses the existing packed corpus and latest checkpoint. Code is taken from the
project dataset snapshot (/kaggle/input/tamil-tok-code) rather than from this
kernel's own previous output, so the trainer is always the current version.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

VOCAB = os.environ.get("TAMIL_VOCAB", "tok32k")
TOTAL_TOKENS = int(os.environ.get("TAMIL_TOTAL_TOKENS", "927000000"))
WALL_HOURS = float(os.environ.get("TAMIL_WALL_HOURS", "1.6"))

code = sorted({os.path.dirname(os.path.dirname(p))
               for p in glob.glob("/kaggle/input/**/tamil/akshara.py", recursive=True)})
bins = glob.glob(f"/kaggle/input/**/artifacts/train_{VOCAB}.bin", recursive=True)
if not (code and bins):
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) <= 6:
            print(root, files[:6])
    raise SystemExit("code dataset or packed corpus not mounted")


def root_rank(root: str) -> tuple[bool, bool, float]:
    """Prefer the dataset snapshot (it carries the frozen eval set) over kernel-output copies."""
    has_eval = os.path.exists(f"{root}/eval/tamil_reliability_holdout.jsonl")
    train_py = f"{root}/tamil/train.py"
    return (has_eval, os.path.exists(f"{root}/kaggle/02_train.py"),
            os.path.getmtime(train_py) if os.path.exists(train_py) else 0.0)


print("code candidates:", code, flush=True)
src_root = max(code, key=root_rank)
print("using code root:", src_root, flush=True)

repo = "/kaggle/working/repo"
shutil.copytree(src_root, repo, dirs_exist_ok=True)

# Keep the logits chunk small (64 MB instead of 250 MB) so T4 memory stays safe.
# Current train.py reads TAMIL_LOSS_CHUNK from the environment already; older copies need the patch.
train_py = Path(repo) / "tamil" / "train.py"
src = train_py.read_text(encoding="utf-8")
if "TAMIL_LOSS_CHUNK" in src:
    print("LOSS_CHUNK: already env-configurable", flush=True)
else:
    patched = re.sub(
        r"^LOSS_CHUNK = \d+.*$",
        'LOSS_CHUNK = int(os.environ.get("TAMIL_LOSS_CHUNK", "512"))',
        src, count=1, flags=re.M,
    )
    if patched == src:
        raise SystemExit("could not patch LOSS_CHUNK in the mounted tamil/train.py")
    train_py.write_text(patched, encoding="utf-8")
    print("LOSS_CHUNK: patched to env setting", flush=True)

os.makedirs(f"{repo}/artifacts", exist_ok=True)
stem = bins[0][:-len(".bin")]
for ext in (".bin", ".lang.npy", ".idx.npy", ".meta.json"):
    dst = f"{repo}/artifacts/train_{VOCAB}{ext}"
    if not os.path.exists(dst):
        os.symlink(stem + ext, dst)

import torch


def ckpt_step(path: str) -> tuple[int, str]:
    """Step number of a checkpoint without materialising its 4 GB of tensors when possible."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    step = int(ck.get("step", -1))
    del ck
    return step, path


prev = sorted(glob.glob("/kaggle/input/**/artifacts/run/ckpt.pt", recursive=True))
if not prev:
    raise SystemExit("No previous pretraining checkpoint found; refusing random initialization")
ranked = sorted((ckpt_step(p) for p in prev), key=lambda item: item[0])
print("checkpoint candidates (step, path):", ranked, flush=True)
resume_step, resume_from = ranked[-1]

os.chdir(repo)
env = dict(os.environ, PYTHONPATH=repo, PYTHONUNBUFFERED="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           TAMIL_LOSS_CHUNK="512")
# micro-batch 2 keeps the activation footprint small; dropping activation checkpointing
# recovers the ~30% recompute cost the previous session paid. The earlier T4 OOM was the
# 250 MB logits chunk, now 64 MB at chunk 512.
cmd = [
    sys.executable, "kaggle/02_train.py",
    "--tokenizer", f"artifacts/{VOCAB}/tokenizer.json",
    "--total_tokens", str(TOTAL_TOKENS), "--wall_budget_h", str(WALL_HOURS),
    "--micro_bs", "2", "--grad_accum", "64", "--grad_ckpt", "false",
    "--resume_from", resume_from,
]
print("resume checkpoint:", resume_from, "step:", resume_step, flush=True)
print("=====", " ".join(cmd), flush=True)
rc = subprocess.run(cmd, env=env, cwd=repo).returncode
print(f"===== train exit={rc}", flush=True)

for ext in (".bin", ".lang.npy", ".idx.npy", ".meta.json"):
    p = f"{repo}/artifacts/train_{VOCAB}{ext}"
    if os.path.islink(p):
        os.unlink(p)
sys.exit(rc)
