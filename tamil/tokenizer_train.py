"""Train the akshara-safe Unigram tokenizer (Tamil + Tanglish + English) into one tokenizer.json.

The normalizer maps each Tamil akshara to one PUA char (see tamil.akshara), so no learned piece
can cut a letter; the decoder maps them back. Unknown chars fall back to UTF-8 byte tokens.
"""
import argparse
import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from tamil.akshara import AKSHARA_TABLE

# ids 0-5 are fixed (sft.py, pack.py eos_id=2); chat/tool/code tokens follow, plus reserved slots for later
CHAT_TOOL_TOKENS = ["<|system|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
                    "<think>", "</think>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
RESERVED_TOKENS = [f"<|reserved_{i}|>" for i in range(16)]
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<|user|>", "<|assistant|>"] + CHAT_TOOL_TOKENS + RESERVED_TOKENS
# matched on raw text before normalization: one token per indent run instead of one per space
WHITESPACE_TOKENS = ["\n", "\n\n", "  ", "    ", "        ", "            ", "                ",
                     "\t", "\t\t", "\t\t\t"]
BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]
_LONGEST_FIRST = sorted(AKSHARA_TABLE.items(), key=lambda kv: -len(kv[0]))


def _pipeline(tok: Tokenizer) -> None:
    tok.normalizer = normalizers.Sequence(
        [normalizers.NFC()] + [normalizers.Replace(unit, pua) for unit, pua in _LONGEST_FIRST])
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Metaspace(prepend_scheme="never"), pre_tokenizers.Digits(individual_digits=True)])
    # nothing is prepended, so nothing is stripped on decode: leading spaces and indents survive exactly
    tok.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
        + [decoders.Replace(pua, unit) for unit, pua in AKSHARA_TABLE.items()])


def _add_byte_pieces(tok: Tokenizer) -> Tokenizer:
    """The Unigram trainer doesn't emit <0xNN> pieces; append them so byte_fallback can use them."""
    spec = json.loads(tok.to_str())
    vocab = spec["model"]["vocab"]
    floor = min(score for _, score in vocab) - 10.0
    vocab += [[b, floor] for b in BYTE_TOKENS if b not in {p for p, _ in vocab}]
    spec["model"]["byte_fallback"] = True
    return Tokenizer.from_str(json.dumps(spec))


def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: Path) -> Tokenizer:
    tok = Tokenizer(models.Unigram())
    _pipeline(tok)
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size - len(BYTE_TOKENS) - len(WHITESPACE_TOKENS),
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
        initial_alphabet=list(AKSHARA_TABLE.values()),
        max_piece_length=16,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok = _add_byte_pieces(tok)  # bytes first: added tokens take ids after the model vocab
    tok.add_tokens([AddedToken(ws, normalized=False, special=False) for ws in WHITESPACE_TOKENS])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    from transformers import PreTrainedTokenizerFast
    PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                            pad_token="<pad>").save_pretrained(str(out_path.parent))
    return tok


def read_shards(paths: Iterable[Path]) -> Iterator[str]:
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True, help="dir of shard_*.jsonl.gz from write_shards")
    ap.add_argument("--vocab", type=int, nargs="+", default=[32000, 48000])
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    args = ap.parse_args()
    shards = sorted(args.shards.glob("shard_*.jsonl.gz"))
    for v in args.vocab:
        path = args.out / f"tok{v // 1000}k" / "tokenizer.json"
        train_tokenizer(read_shards(shards), v, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
