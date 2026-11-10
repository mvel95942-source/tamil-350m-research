"""Kaggle: build + pack the training corpus (CPU job), then pretrain with torchrun (GPU T4 x2 job).

    # 1) CPU session, once (hours; must not eat the GPU session's 12 h):
    TAMIL_SCRATCH=/tmp python kaggle/02_train.py --tokenizer artifacts/tok32k/tokenizer.json --pack_only
    # 2) GPU session: throughput test, then the full run. Kaggle wipes /kaggle/working between sessions,
    #    so attach the previous session's output as a dataset and pass its checkpoint:
    python kaggle/02_train.py --tokenizer artifacts/tok32k/tokenizer.json --total_tokens 20000000 \
        --out_dir artifacts/throughput_test
    python kaggle/02_train.py --tokenizer artifacts/tok32k/tokenizer.json \
        --resume_from /kaggle/input/<prev-session-output>/artifacts/run/ckpt.pt

Extra flags pass through to tamil.train (see TrainConfig), e.g. --grad_ckpt true if T4 memory runs out.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamil import sources  # noqa: E402
from tamil.corpus import SCRATCH, iter_mixed, shards_done, write_shards  # noqa: E402
from tamil.pack import pack  # noqa: E402

MODEL_MIX = {"tamil": 0.55, "tanglish": 0.20, "english": 0.25}
GPU_HOURS_BUDGET = 60  # plan: cut total_tokens if the projection exceeds ~60 GPU-hours
SHARD_DIR = SCRATCH / "train_shards"


def build_and_pack(tokenizer: Path, bin_path: Path, corpus_bytes: int) -> None:
    import numpy as np
    blocklist = set(np.load(Path("artifacts") / "blocklist.npy").tolist())
    if not shards_done(SHARD_DIR):
        used = 0

        def capped():
            nonlocal used
            for lang, text in iter_mixed(sources.PRETRAIN, MODEL_MIX, blocklist, seed=2):
                used += len(text.encode("utf-8"))
                if used > corpus_bytes:
                    return
                yield lang, text
        write_shards(capped(), SHARD_DIR, docs_per_shard=50_000)
    n = pack(sorted(SHARD_DIR.glob("shard_*.jsonl.gz")), tokenizer, bin_path, eos_id=2)
    print(f"packed {n:,} tokens -> {bin_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--corpus_bytes", type=int, default=8_000_000_000)
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts/run"))
    ap.add_argument("--pack_only", action="store_true", help="CPU job: build shards + pack, then exit")
    args, passthrough = ap.parse_known_args()

    bin_path = Path("artifacts") / f"train_{args.tokenizer.parent.name}.bin"
    if args.pack_only:
        if not bin_path.exists():
            build_and_pack(args.tokenizer, bin_path, args.corpus_bytes)
        return
    if not bin_path.exists():
        sys.exit(f"{bin_path} missing: run with --pack_only in a CPU session first (or attach it as a dataset)")

    cmd = ["torchrun", "--nproc_per_node=2", "-m", "tamil.train", "--bin_path", str(bin_path),
           "--tokenizer_path", str(args.tokenizer), "--out_dir", str(args.out_dir), *passthrough]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)

    log = [json.loads(l) for l in (args.out_dir / "log.jsonl").read_text().splitlines()]
    tail = log[len(log) // 2:]
    tok_s = sum(r["tok_s"] for r in tail) / len(tail)
    print(f"throughput {tok_s:,.0f} tok/s -> 1.5B tokens in {1.5e9 / tok_s / 3600:.1f} h "
          f"(budget ~{GPU_HOURS_BUDGET} GPU-h per plan; lower total_tokens if over)")


if __name__ == "__main__":
    main()
