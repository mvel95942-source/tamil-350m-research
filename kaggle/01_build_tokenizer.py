"""Kaggle CPU job: eval blocklist -> balanced ~2 GB tokenizer sample -> tokenizer(s), default v2 tok64k.

v1 (tok32k/tok48k, commit 52bafc2) used mix 45/25/30 and no code. v2 is Tamil-heavier (to beat Agaram's
1.91 Tamil tokens/word at the same 64k) with ~5% code from codeparrot GitHub Code.

Run from the repo root on Kaggle (CPU):  TAMIL_SCRATCH=/tmp python kaggle/01_build_tokenizer.py
Outputs: artifacts/blocklist.npy, $TAMIL_SCRATCH/tok_sample_v2/, artifacts/tok64k/
"""
import os
import sys
from itertools import islice
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tamil import sources  # noqa: E402
from tamil.blocklist import build_blocklist, build_blocklist_from_texts  # noqa: E402
from tamil.corpus import SCRATCH, iter_mixed, load_rows, shards_done, write_shards  # noqa: E402
from tamil.fertility import load_heldout  # noqa: E402
from tamil.tokenizer_train import read_shards, train_tokenizer  # noqa: E402

TOKENIZER_MIX = {"tamil": 0.60, "tanglish": 0.15, "english": 0.20, "code": 0.05}  # share of characters
VOCABS = [int(v) for v in os.environ.get("TOK_VOCABS", "64000").split(",")]
SAMPLE_BYTES = int(os.environ.get("TOK_SAMPLE_BYTES", 2_000_000_000))  # Unigram trainer RAM grows with this
N_HELDOUT = sources.HELDOUT_DOCS  # WIKI_TA_TRAIN skips exactly these docs
ART = Path("artifacts")


def capped(it, max_bytes):
    used = 0
    for lang, text in it:
        used += len(text.encode("utf-8"))
        if used > max_bytes:
            return
        yield lang, text


def main() -> None:
    bl_path = ART / "blocklist.npy"
    if bl_path.exists():
        blocklist = set(np.load(bl_path).tolist())
    else:
        held = load_heldout(N_HELDOUT)  # not whole Wikipedias: ~150M hashes and needless data loss
        blocklist = build_blocklist(sources.EVAL, load_rows) | build_blocklist_from_texts(
            t for texts in held.values() for t in texts)
        ART.mkdir(exist_ok=True)
        np.save(bl_path, np.array(sorted(blocklist), dtype=np.uint64))
    print(f"blocklist: {len(blocklist):,} sentence hashes")

    sample_dir = SCRATCH / "tok_sample_v2"
    if not shards_done(sample_dir):
        mixed = iter_mixed(sources.PRETRAIN + sources.CODE, TOKENIZER_MIX, blocklist, seed=1)
        write_shards(capped(mixed, SAMPLE_BYTES), sample_dir, docs_per_shard=50_000)
    shards = sorted(sample_dir.glob("shard_*.jsonl.gz"))
    print(f"sample: {len(shards)} shards, first doc: {next(islice(read_shards(shards), 1))[:80]!r}")

    for v in VOCABS:
        tok = train_tokenizer(read_shards(shards), v, ART / f"tok{v // 1000}k" / "tokenizer.json")
        print(f"tok{v // 1000}k: vocab {tok.get_vocab_size():,} (target {v:,})")


if __name__ == "__main__":
    main()
