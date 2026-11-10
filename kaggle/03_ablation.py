"""Kaggle GPU job: tokenizer ablation. Same model size, same shards (same UTF-8 bytes), one epoch each:
our tokenizer vs the Llama-3 tokenizer. Re-run to resume either arm.

    python kaggle/03_ablation.py --ours artifacts/tok32k/tokenizer.json --n_shards 20 --model_size 30m
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamil.evaluate import doc_sep_id  # noqa: E402
from tamil.pack import pack  # noqa: E402

LLAMA = "unsloth/Llama-3.2-1B"


def export_llama(out: Path) -> Path:
    if not out.exists():
        from transformers import AutoTokenizer
        out.parent.mkdir(parents=True, exist_ok=True)
        AutoTokenizer.from_pretrained(LLAMA).backend_tokenizer.save(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--n_shards", type=int, default=20, help="first N train shards (same text for both arms)")
    ap.add_argument("--model_size", default="30m")
    ap.add_argument("--micro_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    art = Path("artifacts")
    from tamil.corpus import SCRATCH, shards_done
    if not shards_done(SCRATCH / "train_shards"):
        sys.exit("train shards missing/incomplete: run kaggle/02_train.py --pack_only first")
    shards = sorted((SCRATCH / "train_shards").glob("shard_*.jsonl.gz"))[: args.n_shards]
    arms = {"ours": args.ours, "llama3": export_llama(art / "tok_llama3" / "tokenizer.json")}
    for arm, tok_path in arms.items():
        bin_path = art / f"ablation_{arm}.bin"
        if not bin_path.exists():
            pack(shards, tok_path, bin_path, eos_id=doc_sep_id(Tokenizer.from_file(str(tok_path))))
        meta = json.loads(bin_path.with_suffix(".meta.json").read_text())
        n_tokens = np.memmap(bin_path, dtype=meta["dtype"], mode="r").size
        print(f"{arm}: {n_tokens:,} tokens for the same {len(shards)} shards")
        cmd = ["torchrun", "--nproc_per_node=2", "-m", "tamil.train", "--bin_path", str(bin_path),
               "--tokenizer_path", str(tok_path), "--out_dir", str(art / f"ablation_{arm}"),
               "--model_size", args.model_size, "--total_tokens", str(n_tokens),
               "--micro_bs", str(args.micro_bs), "--grad_accum", str(args.grad_accum)]
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
