"""Finish the legacy 32K package: fill the README hash, write metadata + SHA256SUMS.

    python scripts/package_tokenizer_32k_release.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release" / "merzal-tamil-tokenizer-32k"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    tok = REL / "tokenizer.json"
    digest = sha256(tok)
    readme = REL / "README.md"
    text = readme.read_text(encoding="utf-8").replace("PLACEHOLDER_SHA", digest)
    readme.write_text(text, encoding="utf-8")

    hold = json.loads((ROOT / "reports" / "tokenizer_holdout.json").read_text(encoding="utf-8"))
    r = hold["results"]["ours-tok32k"]["all"]
    meta = {
        "name": "merzal-tamil-tokenizer-32k",
        "version": "0.1.0",
        "status": "legacy",
        "superseded_by": "merzal-tamil-tokenizer (64K, v1.0.0)",
        "release_date": "2026-11-06",
        "artifact": "tokenizer.json",
        "sha256": digest,
        "vocab_size": 32000,
        "special_tokens": 6,
        "byte_fallback": True,
        "model_type": "Unigram",
        "max_length": 32768,
        "license": "Apache-2.0",
        "known_defect": "Metaspace(prepend_scheme='always') drops leading whitespace",
        "heldout_validation": {
            "dataset": hold["meta"]["dataset"],
            "config": hold["meta"]["config"],
            "split": hold["meta"]["split"],
            "docs": r["docs"],
            "tokens_per_word": r["tokens_per_word"],
            "tokens_per_word_ci95": r["tokens_per_word_ci95"],
            "cut_letter_rate": r["cut_letter_rate"],
            "roundtrip_exact": r["roundtrip_exact"],
        },
    }
    (REL / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    files = sorted(p for p in REL.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS" and p.suffix != ".pyc")
    lines = [f"{sha256(p)}  {p.relative_to(REL).as_posix()}" for p in files]
    (REL / "SHA256SUMS").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"sha256       {digest}")
    print(f"SHA256SUMS   {len(lines)} files")
    print(f"package      {sum(p.stat().st_size for p in files) / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
