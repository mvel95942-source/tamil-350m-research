"""Kaggle GPU job: build the Tamil SFT set (Aya + IndicAlign, blocklist-filtered), then instruction-tune.

    python kaggle/05_sft.py --base_ckpt artifacts/run/ckpt.pt --tokenizer artifacts/tok32k/tokenizer.json
Re-run to resume. Extra flags pass through to tamil.sft (see SFTConfig).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamil.sft import prepare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts/sft"))
    args, passthrough = ap.parse_known_args()
    data = Path("artifacts/sft_ta.jsonl")
    if not data.exists():
        n = prepare(data, set(np.load("artifacts/blocklist.npy").tolist()))
        print(f"SFT conversations: {n:,}")
    cmd = [sys.executable, "-m", "tamil.sft", "--base_ckpt", str(args.base_ckpt), "--tokenizer_path",
           str(args.tokenizer), "--data_path", str(data), "--out_dir", str(args.out_dir), *passthrough]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
