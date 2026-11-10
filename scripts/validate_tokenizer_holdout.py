"""Validate the trained Tamil tokenizer on a large, independent held-out corpus.

Default corpus: FineWeb-2 `tam_Taml` TEST split (HuggingFaceFW/fineweb-2) — a different
dataset and organisation from anything the tokenizer trained on (Sangraha / IndicCorpV2 /
Tamil Wikipedia / Dakshina), and a different split from the 2k-doc fertility benchmark.

Reports tokens/word with a bootstrap 95% CI, chars & bytes per token, cut-letter rate,
byte-fallback share, unknown share and exact round-trip, plus baseline tokenizers.

    python scripts/validate_tokenizer_holdout.py --tok artifacts/tok64k artifacts/tok32k
    python scripts/validate_tokenizer_holdout.py --parquet PATH.parquet --limit 20000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tamil.akshara import AKSHARA_TABLE, PULLI, VOWEL_SIGNS
from tamil.tokenizer_train import BYTE_TOKENS

DEPENDENT = set(VOWEL_SIGNS) | {PULLI, "ௗ"}
BYTE_SET = frozenset(BYTE_TOKENS)
# A token "starts mid-letter" if its first non-space code point is a dependent sign,
# either as the raw Tamil char or as the PUA char the normalizer maps it to.
ORPHAN_HEADS = frozenset(DEPENDENT) | frozenset(
    pua for unit, pua in AKSHARA_TABLE.items() if unit[:1] in DEPENDENT)
TAMIL_LO, TAMIL_HI = 0x0B80, 0x0BFF

CORE_BASELINES = {
    "gpt-4o (o200k)": "Xenova/gpt-4o",
    "llama-3": "unsloth/Llama-3.2-1B",
    "gemma-4 (262k)": "google/gemma-4-E2B-it",
    "qwen-2.5": "Qwen/Qwen2.5-0.5B",
    "indicbert-v2": "ai4bharat/IndicBERTv2-MLM-only",
    "agaram-2.5b": "murai-labs/agaram-2.5b",
}
ALL_BASELINES = {
    **CORE_BASELINES,
    "gemma-2": "unsloth/gemma-2-2b",
    "gemma-3 (262k)": "unsloth/gemma-3-1b-it",
    "mt5": "google/mt5-small",
    "sarvam-1": "sarvamai/sarvam-1",
}


def tamil_share(text: str) -> float:
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if TAMIL_LO <= ord(c) <= TAMIL_HI) / len(letters)


def load_corpus(path: Path, limit: int, min_chars: int = 60, max_chars: int = 4000,
                min_tamil: float = 0.5):
    import pyarrow.parquet as pq
    stats = {"rows": 0, "kept": 0, "short": 0, "not_tamil": 0, "duplicate": 0}
    seen: set[str] = set()
    docs: list[str] = []
    scores: list[float] = []
    pf = pq.ParquetFile(path)
    cols = [c for c in ("text", "language_score") if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=4096, columns=cols):
        texts = batch.column("text").to_pylist()
        sc = batch.column("language_score").to_pylist() if "language_score" in cols else [None] * len(texts)
        for text, score in zip(texts, sc):
            stats["rows"] += 1
            if not text:
                stats["short"] += 1
                continue
            text = text.strip()
            if len(text) < min_chars or tamil_share(text) < min_tamil:
                stats["short" if len(text) < min_chars else "not_tamil"] += 1
                continue
            text = text[:max_chars]
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen:
                stats["duplicate"] += 1
                continue
            seen.add(h)
            docs.append(text)
            stats["kept"] += 1
            if score is not None:
                scores.append(float(score))
            if len(docs) >= limit:
                break
        if len(docs) >= limit:
            break
    stats["language_score_mean"] = round(float(np.mean(scores)), 4) if scores else None
    return docs, stats


class Ours:
    def __init__(self, name: str, path: Path):
        from tokenizers import Tokenizer
        self.name = name
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.dirs = [path.name]
        self.unk = self.tok.token_to_id("<unk>")

    def run(self, docs: list[str], purity: list[bool]) -> list[dict]:
        rows = []
        enc = self.tok.encode
        dec = self.tok.decode
        unk = self.unk
        for text, dominant in zip(docs, purity):
            e = enc(text, add_special_tokens=False)
            ids = e.ids
            cuts = byte = 0
            for t in e.tokens:
                if t in BYTE_SET:
                    byte += 1
                elif t.lstrip(" ▁")[:1] in ORPHAN_HEADS:
                    cuts += 1
            rows.append({
                "tokens": len(ids), "words": len(text.split()),
                "chars": len(text), "bytes": len(text.encode("utf-8")),
                "cuts": cuts, "byte": byte,
                "unk": ids.count(unk) if unk is not None else 0,
                "exact": dec(ids) == unicodedata.normalize("NFC", text),
                "dominant": dominant,
            })
        return rows


class Baseline:
    def __init__(self, name: str, hf_id: str):
        from transformers import AutoTokenizer
        self.name = name
        self.hf_id = hf_id
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.dirs = []
        self.unk = self.tok.unk_token_id
        self.fast = bool(getattr(self.tok, "is_fast", False))

    def run(self, docs: list[str], purity: list[bool]) -> list[dict]:
        rows = []
        for text, dominant in zip(docs, purity):
            try:
                if self.fast:
                    out = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
                    ids, offs = out["input_ids"], out["offset_mapping"]
                    cuts = sum(1 for s, e in offs if e > s and s > 0 and text[s] in DEPENDENT)
                else:
                    ids = self.tok(text, add_special_tokens=False)["input_ids"]
                    toks = self.tok.convert_ids_to_tokens(ids)
                    cuts = sum(1 for t in toks if isinstance(t, str)
                               and t.lstrip(" ▁▁")[:1] in DEPENDENT)
                exact = self.tok.decode(ids) == unicodedata.normalize("NFC", text)
                unk = ids.count(self.unk) if self.unk is not None else 0
            except Exception:  # noqa: BLE001 - a baseline that chokes on a doc is counted as failed
                ids, cuts, exact, unk = [], 0, False, 0
            rows.append({
                "tokens": len(ids), "words": len(text.split()),
                "chars": len(text), "bytes": len(text.encode("utf-8")),
                "cuts": cuts, "byte": 0, "unk": unk, "exact": exact, "dominant": dominant,
            })
        return rows


def bootstrap_ci(tokens: np.ndarray, words: np.ndarray, n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float]:
    """CI for the corpus-level ratio sum(tokens)/sum(words), resampling documents."""
    rng = np.random.default_rng(seed)
    n = len(tokens)
    idx = rng.integers(0, n, size=(n_boot, n))
    tok = tokens[idx].sum(axis=1)
    wrd = np.maximum(words[idx].sum(axis=1), 1)
    means = tok / wrd
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict], dominant_only: bool = False) -> dict:
    r = [x for x in rows if x["dominant"]] if dominant_only else rows
    tok = np.array([x["tokens"] for x in r], dtype=np.int64)
    words = np.array([x["words"] for x in r], dtype=np.int64)
    chars = np.array([x["chars"] for x in r], dtype=np.int64)
    byts = np.array([x["bytes"] for x in r], dtype=np.int64)
    lo, hi = bootstrap_ci(tok, words)
    return {
        "docs": len(r),
        "words": int(words.sum()),
        "chars": int(chars.sum()),
        "bytes": int(byts.sum()),
        "tokens": int(tok.sum()),
        "tokens_per_word": round(float(tok.sum() / max(words.sum(), 1)), 4),
        "tokens_per_word_ci95": [round(lo, 4), round(hi, 4)],
        "chars_per_token": round(float(chars.sum() / max(tok.sum(), 1)), 4),
        "bytes_per_token": round(float(byts.sum() / max(tok.sum(), 1)), 4),
        "cut_letter_rate": round(float(sum(x["cuts"] for x in r) / max(tok.sum(), 1)), 6),
        "byte_fallback_rate": round(float(sum(x["byte"] for x in r) / max(tok.sum(), 1)), 6),
        "unk_rate": round(float(sum(x["unk"] for x in r) / max(tok.sum(), 1)), 6),
        "roundtrip_exact": round(float(sum(x["exact"] for x in r) / max(len(r), 1)), 6),
    }


def fmt_rate(x: float) -> str:
    """Percent with enough precision to distinguish 0.04% from 0.0%."""
    return f"{x:.2%}" if x < 0.01 else f"{x:.1%}"


def render(results: dict, meta: dict, order: list[str]) -> str:
    lines = [
        "# Tokenizer validation on an independent held-out Tamil corpus",
        "",
        f"- corpus: `{meta['dataset']}` / `{meta['config']}` / `{meta['split']}` (downloaded parquet)",
        f"- parquet sha256: `{meta['parquet_sha256']}`",
        f"- docs kept: {meta['stats']['kept']:,} (rows seen {meta['stats']['rows']:,})",
        f"- chars: {meta['totals']['chars']:,} · words: {meta['totals']['words']:,} · bytes: {meta['totals']['bytes']:,}",
        f"- max chars/doc: {meta['max_chars']} · min Tamil share: {meta['min_tamil']}",
        "",
        "## Tamil tokens per word (lower is better), bootstrap 95% CI",
        "",
        "| tokenizer | tokens/word | 95% CI | chars/token | bytes/token | cut letters | round-trip |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in order:
        r = results[name]["all"]
        lines.append(
            f"| {name} | {r['tokens_per_word']:.3f} | "
            f"[{r['tokens_per_word_ci95'][0]:.3f}, {r['tokens_per_word_ci95'][1]:.3f}] | "
            f"{r['chars_per_token']:.2f} | {r['bytes_per_token']:.2f} | "
            f"{fmt_rate(r['cut_letter_rate'])} | {r['roundtrip_exact']:.1%} |")
    lines += ["", "## Tamil-dominant subset (>=95% Tamil script chars)", "",
              "| tokenizer | docs | tokens/word | 95% CI | cut letters | round-trip |",
              "|---|---|---|---|---|---|"]
    for name in order:
        r = results[name]["dominant"]
        lines.append(f"| {name} | {r['docs']:,} | {r['tokens_per_word']:.3f} | "
                     f"[{r['tokens_per_word_ci95'][0]:.3f}, {r['tokens_per_word_ci95'][1]:.3f}] | "
                     f"{fmt_rate(r['cut_letter_rate'])} | {r['roundtrip_exact']:.1%} |")
    lines += ["", "## Byte fallback / unknown tokens (ours)", "",
              "| tokenizer | byte tokens | <unk> |", "|---|---|---|"]
    for name in order:
        if name.startswith("ours"):
            r = results[name]["all"]
            lines.append(f"| {name} | {r['byte_fallback_rate']:.2%} | {r['unk_rate']:.4%} |")
    ours = [n for n in order if n.startswith("ours")]
    if len(ours) >= 2:
        a, b = ours[0], ours[1]
        d = results[b]["all"]["tokens_per_word"] / results[a]["all"]["tokens_per_word"] - 1
        lines += ["", f"`{b}` vs `{a}`: {d:+.1%} tokens per Tamil word."]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", type=Path, nargs="+", default=[Path("artifacts/tok64k"), Path("artifacts/tok32k")])
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--config", default="tam_Taml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--min-tamil", type=float, default=0.5)
    ap.add_argument("--baselines", choices=["none", "core", "all"], default="core")
    ap.add_argument("--out", type=Path, default=Path("reports/tokenizer_holdout"))
    args = ap.parse_args()

    path = args.parquet
    if path is None:
        path = Path(".cache/holdout") / "data" / args.config / args.split / "000_00000.parquet"
        if not path.is_file():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(args.dataset, f"data/{args.config}/{args.split}/000_00000.parquet",
                                        repo_type="dataset", local_dir=".cache/holdout"))
    t0 = time.perf_counter()
    docs, stats = load_corpus(path, args.limit, max_chars=args.max_chars, min_tamil=args.min_tamil)
    print(f"corpus: {len(docs):,} docs from {path} ({time.perf_counter() - t0:.1f}s) {stats}")
    if not docs:
        raise SystemExit("no documents loaded; check --parquet/--limit/--min-tamil")

    toks = [Ours(f"ours-{p.name}", p) for p in args.tok]
    if args.baselines != "none":
        table = CORE_BASELINES if args.baselines == "core" else ALL_BASELINES
        for name, hf_id in table.items():
            try:
                toks.append(Baseline(name, hf_id))
            except Exception as e:  # noqa: BLE001 - a gated/missing baseline must not kill the run
                print(f"SKIPPED baseline {name}: {type(e).__name__}: {e}")

    results, order = {}, []
    purity = [tamil_share(d) >= 0.95 for d in docs]
    for t in toks:
        t1 = time.perf_counter()
        rows = t.run(docs, purity)
        results[t.name] = {"all": summarize(rows), "dominant": summarize(rows, dominant_only=True),
                           "dirs": t.dirs}
        order.append(t.name)
        print(f"{t.name}: {results[t.name]['all']['tokens_per_word']:.3f} tok/word "
              f"({time.perf_counter() - t1:.1f}s)", flush=True)

    totals = {
        "chars": sum(len(d) for d in docs),
        "words": sum(len(d.split()) for d in docs),
        "bytes": sum(len(d.encode()) for d in docs),
    }
    meta = {
        "dataset": args.dataset, "config": args.config, "split": args.split,
        "parquet": str(path), "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "docs": len(docs), "max_chars": args.max_chars, "min_tamil": args.min_tamil,
        "stats": stats, "totals": totals,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    payload = {"meta": meta, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(render(results, meta, order), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(render(results, meta, order))
    print(f"wrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
