"""Tokenize shards into one uint16 token stream (docs separated by EOS) and serve fixed windows.

Batch contents are a pure function of (step, rank), so a resumed run sees exactly the data it
would have seen without the interruption.
"""
import argparse
import gzip
import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

LANG_CODES = {"tamil": 0, "tanglish": 1, "english": 2}
ENCODE_BATCH = 1000


def _rows(shards: list[Path]) -> Iterator[tuple[str, str]]:
    for p in shards:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                yield row["lang"], row["text"]


def _dtype_of(bin_path: Path) -> np.dtype:
    meta = bin_path.with_suffix(".meta.json")
    return np.dtype(json.loads(meta.read_text())["dtype"]) if meta.exists() else np.dtype(np.uint16)


def pack(shards: list[Path], tokenizer_path: Path, out_bin: Path, eos_id: int, dtype: str | None = None) -> int:
    """Write all docs as ids + EOS (uint16, or uint32 for vocabs >= 65536 like Llama-3's 128k);
    also <out>.lang.npy (lang per doc), <out>.idx.npy (doc starts) and <out>.meta.json (dtype)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    dtype = dtype or ("uint16" if tok.get_vocab_size() < 2**16 else "uint32")
    assert dtype == "uint32" or tok.get_vocab_size() < 2**16, "uint16 storage needs vocab < 65536"
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = out_bin.with_suffix(".bin.tmp")  # renamed last: an existing .bin is always complete
    langs, starts, total = [], [], 0

    def flush(batch, f):
        nonlocal total
        for enc in tok.encode_batch([t for _, t in batch], add_special_tokens=False):
            ids = np.array(enc.ids + [eos_id], dtype=dtype)
            starts.append(total)
            ids.tofile(f)
            total += ids.size
        langs.extend(LANG_CODES[l] for l, _ in batch)

    with open(tmp_bin, "wb") as f:
        batch = []
        for row in _rows(shards):
            batch.append(row)
            if len(batch) == ENCODE_BATCH:
                flush(batch, f)
                batch = []
        if batch:
            flush(batch, f)
    np.save(out_bin.with_suffix(".lang.npy"), np.array(langs, dtype=np.uint8))
    np.save(out_bin.with_suffix(".idx.npy"), np.array(starts, dtype=np.uint64))
    out_bin.with_suffix(".meta.json").write_text(json.dumps({"dtype": dtype, "eos_id": eos_id}))
    os.replace(tmp_bin, out_bin)
    return total


class PackedDataset:
    """Shuffled fixed-size windows over the packed stream; window order fixed by `seed`."""

    def __init__(self, bin_path: Path, ctx: int, rank: int, world: int, seed: int = 0):
        self.tokens = np.memmap(bin_path, dtype=_dtype_of(Path(bin_path)), mode="r")
        self.ctx, self.rank, self.world = ctx, rank, world
        self.n_windows = (self.tokens.size - 1) // ctx
        assert self.n_windows > 0, "token stream shorter than one window"
        self.order = np.random.default_rng(seed).permutation(self.n_windows)

    def batch(self, step: int, micro_bs: int) -> torch.LongTensor:
        """Rows of ctx+1 tokens (inputs + next-token labels) for global micro-step `step` on this rank."""
        first = (step * self.world + self.rank) * micro_bs
        rows = []
        for i in range(micro_bs):
            w = int(self.order[(first + i) % self.n_windows])
            rows.append(self.tokens[w * self.ctx: w * self.ctx + self.ctx + 1].astype(np.int64))
        return torch.from_numpy(np.stack(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json")
    ap.add_argument("--out", type=Path, required=True, help="output .bin")
    ap.add_argument("--eos-id", type=int, default=2)
    args = ap.parse_args()
    n = pack(sorted(args.shards.glob("shard_*.jsonl.gz")), args.tokenizer, args.out, args.eos_id)
    print(f"packed {n:,} tokens -> {args.out}")


if __name__ == "__main__":
    main()
