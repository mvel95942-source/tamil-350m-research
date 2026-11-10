"""Fertility benchmark: our tokenizers vs the big ones, on held-out Tamil / Tanglish / English.

Metrics per (tokenizer, language): tokens per word, bytes per token, cut-letter rate (share of
token starts that land on a Tamil vowel sign or pulli, i.e. mid-letter) and exact round-trip.
"""
import argparse
import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from random import Random

from tamil.akshara import PULLI, VOWEL_SIGNS

DEPENDENT = set(VOWEL_SIGNS) | {PULLI, "ௗ"}

# open, ungated mirrors so the benchmark runs without HF tokens
BASELINES: dict[str, str] = {
    "gpt-4o (o200k)": "Xenova/gpt-4o",
    "llama-3": "unsloth/Llama-3.2-1B",
    "gemma-2": "unsloth/gemma-2-2b",
    "gemma-3 (262k)": "unsloth/gemma-3-1b-it",
    "gemma-4 (262k)": "google/gemma-4-E2B-it",  # newest public Gemini-lineage tokenizer (Apache-2.0, ungated)
    "qwen-2.5": "Qwen/Qwen2.5-0.5B",
    "mt5": "google/mt5-small",
    "indicbert-v2": "ai4bharat/IndicBERTv2-MLM-only",  # strong Indic fertility, but not lossless
    "sarvam-1": "sarvamai/sarvam-1",  # Indic generative LLM
    "agaram-2.5b": "murai-labs/agaram-2.5b",  # Tamil-first 64k: the one v2 must beat (arXiv 2610.18420)
}
LANGS = ("tamil", "tanglish_synth", "tanglish_real", "english")
MAX_DOC_CHARS = 2000

Encode = Callable[[str], tuple[list[int], list[tuple[int, int]]]]


@dataclass
class Tok:
    name: str
    encode: Encode  # text -> (ids, char offsets)
    decode: Callable[[list[int]], str] | None
    # text -> each token's own text; preferred for the cut-letter count when offsets are unreliable
    pieces: Callable[[str], list[str]] | None = None


def from_tokenizers(name: str, path: Path) -> Tok:
    from tokenizers import Tokenizer
    from tamil.akshara import from_pua
    tok = Tokenizer.from_file(str(path))

    def enc(s):
        e = tok.encode(s, add_special_tokens=False)
        return e.ids, e.offsets

    # the Replace normalizer aligns each akshara to its last code point, so offsets are wrong here
    def pieces(s):
        return [from_pua(p) for p in tok.encode(s, add_special_tokens=False).tokens]
    return Tok(name, enc, tok.decode, pieces)


def from_hf(name: str, hf_id: str) -> Tok:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)

    def enc(s):
        out = tok(s, add_special_tokens=False, return_offsets_mapping=True)
        return out["input_ids"], out["offset_mapping"]
    return Tok(name, enc, lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False))


def fertility(tok: Tok, texts: list[str]) -> dict:
    n_tok = n_words = n_bytes = n_chars = cuts = exact = 0
    for t in texts:
        ids, offsets = tok.encode(t)
        n_tok += len(ids)
        n_words += len(t.split())
        n_bytes += len(t.encode("utf-8"))
        n_chars += len(t)
        if tok.pieces is not None:
            cuts += sum(1 for p in tok.pieces(t) if p.lstrip(" ▁")[:1] in DEPENDENT)
        else:
            starts = {s for s, e in offsets if e > s}
            cuts += sum(1 for s in starts if s > 0 and t[s] in DEPENDENT)
        if tok.decode is not None:
            exact += tok.decode(ids) == unicodedata.normalize("NFC", t)
    return {
        "tokens_per_word": n_tok / max(n_words, 1),
        "bytes_per_token": n_bytes / max(n_tok, 1),
        "chars_per_token": n_chars / max(n_tok, 1),
        "cut_letter_rate": cuts / max(n_tok, 1),
        "roundtrip_exact": exact / len(texts) if tok.decode is not None and texts else None,
    }


def _fmt_rate(x: float) -> str:
    """Percent with enough precision to distinguish 0.01% from 0.0%."""
    return f"{x:.2%}" if x < 0.01 else f"{x:.1%}"


def run_benchmark(toks: list[Tok], heldout: dict[str, list[str]], out_md: Path, out_json: Path) -> dict:
    results = {tok.name: {lang: fertility(tok, texts) for lang, texts in heldout.items()} for tok in toks}
    langs = list(heldout)
    lines = ["## Tokens per word (lower is better)", "",
             "| tokenizer | " + " | ".join(langs) + " | tamil bytes/token | tamil cut letters | round-trip |",
             "|---" * (len(langs) + 4) + "|"]
    for name, r in results.items():
        ta = r.get("tamil") or next(iter(r.values()))
        rt = ta["roundtrip_exact"]
        lines.append(f"| {name} | " + " | ".join(f"{r[l]['tokens_per_word']:.2f}" for l in langs)
                     + f" | {ta['bytes_per_token']:.2f} | {_fmt_rate(ta['cut_letter_rate'])} | "
                     + ("n/a" if rt is None else f"{rt:.1%}") + " |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def load_heldout(n: int, seed: int = 0) -> dict[str, list[str]]:
    from tamil import sources
    from tamil.corpus import load_texts
    from tamil.tanglish import romanize
    rng = Random(seed)

    def take(src):
        return [t[:MAX_DOC_CHARS] for t in islice((t for t in load_texts(src) if t.strip()), n)]

    tamil = take(sources.WIKI_TA)
    return {
        "tamil": tamil,
        "tanglish_synth": [romanize(t, rng.choice(["iso", "casual", "chat"]), rng, 0.3) for t in tamil],
        "tanglish_real": take(sources.DAKSHINA_TA_TEST),
        "english": take(sources.WIKI_EN),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, nargs="+", required=True, help="dirs containing tokenizer.json")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("reports/fertility"))
    args = ap.parse_args()
    toks = [from_tokenizers(f"ours-{d.name}", d / "tokenizer.json") for d in args.ours]
    for name, hf_id in BASELINES.items():
        try:
            toks.append(from_hf(name, hf_id))
        except Exception as e:  # one missing/gated baseline must not kill the benchmark; it is reported
            print(f"SKIPPED baseline {name} ({hf_id}): {type(e).__name__}: {e}")
    res = run_benchmark(toks, load_heldout(args.n), args.out.with_suffix(".md"), args.out.with_suffix(".json"))
    ours = sorted((n for n in res if n.startswith("ours-")), key=lambda n: int("".join(filter(str.isdigit, n))))
    if len(ours) == 2:
        small, big = (res[n]["tamil"]["tokens_per_word"] for n in ours)
        pick = ours[1] if big <= small * 0.92 else ours[0]
        print(f"vocab pick: {pick} (rule: larger vocab only if Tamil tokens/word >= 8% lower; {small:.3f} vs {big:.3f})")
    print(args.out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
