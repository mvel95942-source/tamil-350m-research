"""Tamil akshara (grapheme cluster) segmentation and a 1-char-per-akshara PUA encoding.

A Tamil letter like "மி" is two code points (consonant + vowel sign). Mapping every
letter to a single Private Use Area char lets a tokenizer learn pieces that can never
cut a letter in half; `from_pua` maps them back.
"""
import re
import unicodedata

PUA_START = 0xE000
PULLI = "்"
VOWEL_SIGNS = [chr(c) for c in range(0x0BBE, 0x0BCD) if unicodedata.category(chr(c)) in ("Mc", "Mn")]
CONSONANTS = [chr(c) for c in range(0x0B95, 0x0BBA) if unicodedata.category(chr(c)) == "Lo"]
VOWELS = [chr(c) for c in range(0x0B85, 0x0B95) if unicodedata.category(chr(c)) == "Lo"]
AYTHAM = "ஃ"

# consonant + optional (pulli | vowel sign | au length mark), or a vowel, or aytham
_CLUSTER = re.compile("[க-ஹ][ா-்ௗ]?|[அ-ஔ]|ஃ")


def _build_table() -> dict[str, str]:
    units = list(VOWELS) + [AYTHAM]
    for c in CONSONANTS:
        units += [c, c + PULLI] + [c + s for s in VOWEL_SIGNS]
    return {u: chr(PUA_START + i) for i, u in enumerate(sorted(set(units)))}


AKSHARA_TABLE: dict[str, str] = _build_table()
_REVERSE = {v: k for k, v in AKSHARA_TABLE.items()}
_PUA_RE = re.compile("[" + "".join(_REVERSE) + "]")


def split_aksharas(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    out, i = [], 0
    for m in _CLUSTER.finditer(text):
        out.extend(text[i:m.start()])
        out.append(m.group())
        i = m.end()
    out.extend(text[i:])
    return out


def to_pua(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return _CLUSTER.sub(lambda m: AKSHARA_TABLE.get(m.group(), m.group()), text)


def from_pua(text: str) -> str:
    return _PUA_RE.sub(lambda m: _REVERSE[m.group()], text)
