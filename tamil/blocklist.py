"""Sentence fingerprints of eval/held-out data, so none of it leaks into training."""
import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator

from tamil.sources import Source

_SENT_SPLIT = re.compile(r"[.?!।\n]")
_MIN_SENT_CHARS = 12  # shorter fragments ("ஆம்", "Yes") would block innocent docs


def norm_hash(s: str) -> int:
    s = " ".join(unicodedata.normalize("NFC", s).lower().split())
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def sentences(text: str) -> Iterator[str]:
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if len(s) >= _MIN_SENT_CHARS:
            yield s


def build_blocklist_from_texts(texts: Iterable[str]) -> set[int]:
    return {norm_hash(s) for t in texts for s in sentences(t)}


def build_blocklist(sources: list[Source], load_rows: Callable[[Source], Iterable[dict]]) -> set[int]:
    """Raises if any source yields no text: a wrong field name must not silently disable leak protection."""
    out: set[int] = set()
    for src in sources:
        texts = [str(row[f]) for row in load_rows(src) for f in src.fields if row.get(f)]
        if not texts:
            raise ValueError(f"{src.hf_id}: no text in fields {src.fields}; check the field names")
        out |= build_blocklist_from_texts(texts)
    return out


def is_blocked(text: str, blocklist: set[int]) -> bool:
    return any(norm_hash(s) in blocklist for s in sentences(text))
