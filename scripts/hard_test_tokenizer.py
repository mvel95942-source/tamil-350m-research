"""Hard test suite for a trained Tamil tokenizer (tok32k / tok64k).

Unlike tamil/tests/test_tokenizer.py (tiny 2k-vocab fixture), this runs against the real
Kaggle artifact: structural invariants, akshara coverage, every Tamil code point, malformed
Tamil, Unicode/emoji/control fuzz, whitespace + code exactness, special-token collisions,
reload/thread determinism, throughput, and tok32k vs tok64k.

Usage:
    python scripts/hard_test_tokenizer.py --tok artifacts/tok64k --compare artifacts/tok32k
Exit code 1 if any error is found (warnings do not fail).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer

from tamil.akshara import AKSHARA_TABLE, PULLI, VOWEL_SIGNS, from_pua
from tamil.tokenizer_train import (BYTE_TOKENS, CHAT_TOOL_TOKENS, RESERVED_TOKENS,
                                   SPECIAL_TOKENS, WHITESPACE_TOKENS)

DEPENDENT = set(VOWEL_SIGNS) | {PULLI, "ௗ"}
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_RE = re.compile(r"[A-Za-z]")
PUA = {chr(0xE000 + i) for i in range(len(AKSHARA_TABLE))}


def esc(s: str, n: int = 160) -> str:
    return ascii(s)[1:-1][:n]


@dataclass
class Suite:
    name: str
    description: str
    checks: int = 0
    counts: Counter = field(default_factory=Counter)
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    known: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return sum(self.counts.values())

    def check(self, ok: bool, kind: str, value: str, got: str | None = None,
              expected: str | None = None, extra: dict | None = None) -> bool:
        self.checks += 1
        if not ok:
            self.counts[kind] += 1
            rec = {"kind": kind, "input": esc(value)}
            if got is not None:
                rec["got"] = esc(got)
            if expected is not None:
                rec["expected"] = esc(expected)
            if extra:
                rec.update(extra)
            if len(self.errors) < 25:
                self.errors.append(rec)
        return ok

    def warn(self, kind: str, value: str, **extra) -> None:
        if len(self.warnings) < 25:
            self.warnings.append({"kind": kind, "input": esc(value), **extra})

    def note_known(self, kind: str, value: str, got: str, expected: str) -> None:
        if len(self.known) < 10:
            self.known.append({"kind": kind, "input": esc(value),
                               "decoded": esc(got), "expected": esc(expected)})


class Ctx:
    def __init__(self, path: Path):
        self.path = path
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.vocab = self.tok.get_vocab()
        self.specials = set(SPECIAL_TOKENS) | set(CHAT_TOOL_TOKENS) | set(RESERVED_TOKENS)
        self.bytes = set(BYTE_TOKENS)
        self.ws = set(WHITESPACE_TOKENS)

    def enc(self, s: str):
        return self.tok.encode(s, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids)

    def rt(self, s: str) -> tuple[object, str, bool]:
        e = self.enc(s)
        out = self.decode(e.ids)
        exp = unicodedata.normalize("NFC", s)
        return e, out, out == exp

    def token_texts(self, s: str) -> list[str]:
        return [from_pua(t) for t in self.enc(s).tokens if t not in self.bytes]


def suite_structure(ctx: Ctx) -> Suite:
    s = Suite("structure", "vocab, id map, decoder/normalizer wiring")
    tok, vocab = ctx.tok, ctx.vocab
    s.metrics["vocab_size"] = len(vocab)
    s.metrics["tokenizer_json_sha256"] = hashlib.sha256(
        (ctx.path / "tokenizer.json").read_bytes()).hexdigest()

    s.check(len(vocab) == 63995, "vocab size != 63995", str(len(vocab)), str(len(vocab)), "63995")
    ids = sorted(vocab.values())
    s.check(ids == list(range(len(vocab))), "ids not contiguous 0..n-1", "", str(ids[:5]), "0..n-1")

    expect = ["<pad>", "<s>", "</s>", "<unk>", "<|user|>", "<|assistant|>"]
    got = [tok.token_to_id(t) for t in expect]
    s.check(got == list(range(6)), "core special ids changed (sft.py/pack.py assume 0-5)",
            str(expect), str(got), "[0, 1, 2, 3, 4, 5]")

    missing_special = sorted(t for t in ctx.specials if tok.token_to_id(t) is None)
    s.check(not missing_special, "missing special/tool tokens", "", str(missing_special[:8]), "[]")
    missing_bytes = sorted(b for b in ctx.bytes if tok.token_to_id(b) is None)
    s.check(not missing_bytes, "missing byte pieces", "", str(missing_bytes[:8]), "[]")
    missing_ws = sorted(t for t in ctx.ws if tok.token_to_id(t) is None)
    s.check(not missing_ws, "missing whitespace pieces", "", str(missing_ws), "[]")

    spec = json.loads((ctx.path / "tokenizer.json").read_text(encoding="utf-8"))
    model = spec["model"]
    s.check(model.get("byte_fallback") is True, "byte_fallback disabled", "", str(model.get("byte_fallback")), "True")
    s.check(model.get("unk_id") is not None, "unk_id missing", "", str(model.get("unk_id")), "an id")
    # get_vocab() = model pieces + added tokens that the model did not already learn
    model_pieces = {p for p, _ in model["vocab"]}
    added = spec.get("added_tokens") or []
    added_new = [a for a in added if a["content"] not in model_pieces]
    s.check(len(model_pieces) + len(added_new) == len(vocab), "vocab/model/added accounting mismatch",
            "", f"{len(model_pieces)}+{len(added_new)}", str(len(vocab)))
    s.check(all(vocab.get(a["content"]) == a["id"] for a in added), "added-token id mismatch",
            "", "added_tokens ids disagree with get_vocab()")

    norm = spec.get("normalizer") or {}
    n_repl = len(norm.get("normalizers", [])) if norm.get("type") == "Sequence" else 0
    s.check(n_repl == len(AKSHARA_TABLE) + 1, "normalizer not NFC + one Replace per akshara",
            "", str(n_repl), str(len(AKSHARA_TABLE) + 1))

    dec = spec.get("decoder") or {}
    d_repl = [d for d in dec.get("decoders", []) if d.get("type") == "Replace"]
    s.check(len(d_repl) == len(AKSHARA_TABLE) + 1, "decoder Replace count wrong",
            "", str(len(d_repl)), str(len(AKSHARA_TABLE) + 1))

    no_post = not (spec.get("post_processor") or {}).get("special_tokens")
    s.check(no_post, "post_processor adds tokens (would duplicate BOS/EOS)", "", str(no_post), "True")
    a, b = ctx.enc("hi").ids, ctx.enc("hi").ids
    s.check(a == b, "encoding not deterministic", "hi", str(a), str(b))

    empty_ids = [p for p in vocab if p == ""]
    s.check(not empty_ids, "empty vocab piece", "", str(empty_ids), "[]")

    # composition
    comp = Counter()
    for piece in vocab:
        if piece in ctx.specials:
            comp["special/tool/reserved"] += 1
        elif piece in ctx.bytes:
            comp["byte"] += 1
        elif piece in ctx.ws:
            comp["whitespace"] += 1
        else:
            text = from_pua(piece).replace("▁", " ").strip()
            if not text:
                comp["space-only"] += 1
            elif TAMIL_RE.search(text) and LATIN_RE.search(text):
                comp["mixed_script"] += 1
            elif TAMIL_RE.search(text):
                comp["tamil"] += 1
            elif LATIN_RE.search(text):
                comp["latin"] += 1
            else:
                comp["other"] += 1
    s.metrics["vocab_composition"] = dict(comp)
    return s


def suite_vocab_hygiene(ctx: Ctx) -> Suite:
    s = Suite("vocab_hygiene", "orphan-sign and mixed-script pieces")
    orphan, mixed = [], []
    for piece in ctx.vocab:
        if piece in ctx.specials or piece in ctx.bytes:
            continue
        text = from_pua(piece)
        if text.lstrip(" ▁")[:1] in DEPENDENT:
            orphan.append(piece)
        if TAMIL_RE.search(text) and LATIN_RE.search(text):
            mixed.append(piece)
    s.metrics["orphan_dependent_pieces"] = len(orphan)
    s.metrics["mixed_script_pieces"] = len(mixed)
    s.metrics["orphan_examples"] = [esc(p) for p in orphan[:10]]
    if orphan:
        s.warn("orphan-sign pieces: can only match malformed text (wasted vocab slots)",
               str(len(orphan)), examples=[esc(p) for p in orphan[:6]])
    if mixed:
        s.warn("mixed Tamil+Latin pieces (usually date/suffix artefacts)", str(len(mixed)),
               examples=[esc(p) for p in mixed[:6]])
    return s


def suite_akshara(ctx: Ctx) -> Suite:
    s = Suite("akshara", "every table unit must be exactly one token and round-trip")
    units = sorted(AKSHARA_TABLE, key=len, reverse=True)
    s.metrics["akshara_units"] = len(units)
    bad = 0
    for u in units:
        e, out, ok = ctx.rt(u)
        if not s.check(ok and len(e.ids) == 1, "akshara not atomic", u, f"{out!r} ({len(e.ids)} ids)",
                       f"{u!r} (1 id)"):
            bad += 1
        nfd = unicodedata.normalize("NFD", u)
        if nfd != u:
            e2, out2, ok2 = ctx.rt(nfd)
            s.check(ok2 and len(e2.ids) == 1, "NFD akshara not atomic", nfd,
                    f"{out2!r} ({len(e2.ids)} ids)", f"{u!r} (1 id)")
        e3, out3, ok3 = ctx.rt(" " + u)
        # 1 id is good (the "▁akshara" piece exists); 2 ids (space + akshara) is also fine
        s.check(ok3 and len(e3.ids) in (1, 2), "space+akshara not 1-2 tokens", " " + u,
                f"{out3!r} ({len(e3.ids)} ids)", "1-2 ids")
    s.metrics["atomic_failures"] = bad

    # consonant x vowel-sign matrix: must never produce a token that starts mid-letter.
    # U+0BD7 is excluded: it is only valid inside NFC(0BC6 + 0BD7) = 0BCC, i.e. it is
    # already covered by the au sign; a bare 0BD7 after a consonant is malformed input.
    cons = [u for u in units if len(u) == 1 and "\u0B95" <= u <= "\u0BBA"]
    signs = ["", PULLI] + list(VOWEL_SIGNS)
    for c in cons:
        for v in signs:
            text = c + v
            e, out, ok = ctx.rt(text)
            s.check(ok, "consonant+sign round-trip", text, out, unicodedata.normalize("NFC", text))
            for t in ctx.token_texts(text):
                head = t.lstrip(" ▁")
                s.check(head[:1] not in DEPENDENT, "token starts mid-letter", text, t, "whole akshara")
    return s


def suite_tamil_block(ctx: Ctx) -> Suite:
    s = Suite("tamil_block", "U+0B80-U+0BFF standalone, after a consonant, after a space")
    fails = 0
    for cp in range(0x0B80, 0x0C00):
        ch = chr(cp)
        for prefix in ("", "க", " "):
            text = prefix + ch
            e, out, ok = ctx.rt(text)
            if not s.check(ok, f"block U+{cp:04X} round-trip", text, out,
                           unicodedata.normalize("NFC", text)):
                fails += 1
            s.check(3 not in e.ids, f"block U+{cp:04X} hit <unk>", text,
                    str(e.tokens[:6]), "no <unk>")
    s.metrics["roundtrip_failures"] = fails
    return s


MALFORMED = [
    "்", "ா", "ௗ", "ஃ", "ஂ", "ௐ", "௰", "௱", "௲",
    "க்க", "க்ஷ", "க்ி", "காா", "கொௌ", "அ்", "ஃ்", "க\u200dஷ", "க\u200cஷ",
    "க்ஷேத்திரம்", "ஸ்ரீ", "ஜ்ஞ", "தமிழ்123", "தமிழ்_123", "aதமிழ்b",
    "\u0b95\u0bc6\u0bbe", "\u0b95\u0bc7\u0bbe", "\u0b95\u0bcd\u0b95\u0bcd",
    "123", "1234.56", "1,00,000", "3.14e-9", "2026-11-06", "0x1F600",
    "  ", "\n", "\r\n", " \t \n ", "\u00a0", "\u3000", "\u200b", "\ufeff",
    "நன்றி!", "நன்றி?", "நன்றி...", "«தமிழ்»", "தமிழ்—மொழி", "தமிழ்/ஆங்கிலம்",
]


def suite_malformed(ctx: Ctx) -> Suite:
    s = Suite("malformed_tamil", "degenerate Tamil, numbers, punctuation, invisible chars")
    for text in MALFORMED:
        e, out, ok = ctx.rt(text)
        s.check(ok, "malformed round-trip", text, out, unicodedata.normalize("NFC", text))
        s.check(3 not in e.ids, "malformed hit <unk>", text, str(e.tokens[:8]), "no <unk>")
        s.check(len(e.ids) > 0 or text == "", "non-empty text encoded to zero ids", text, "0", ">0")
    return s


EMOJI = [
    "😀", "🔥", "👍🏽", "👨‍👩‍👧‍👦", "🏳️‍🌈", "1️⃣", "🇮🇳", "🇱🇰", "🧑🏾‍💻",
    "❤️", "☺️", "🫠", "🪿", "👩‍❤️‍💋‍👨",
]


def suite_unicode(ctx: Ctx) -> Suite:
    s = Suite("unicode", "other scripts, emoji ZWJ, combining marks, bidi, controls")
    samples = list("éüñçßÅΩπ→√∞≈≠≤≥") + list("中文日本語한글") + \
        list("العربيةעברית") + list("กขค") + list("देवनागरी") + list("తెలుగు") + \
        list("ಕನ್ನಡ") + list("മലയാളം") + list("සිංහල") + list("ខ្មែរ") + EMOJI
    samples += [chr(c) for c in range(0x00, 0x20)] + ["\x7f"]
    samples += [chr(c) for c in range(0x0300, 0x0310)]        # combining diacritics
    samples += ["\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2069"]
    samples += ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad", "\u034f"]
    samples += ["\u00a0", "\u1680", "\u2000", "\u2007", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000"]
    samples += ["\U0001F1EE\U0001F1F3", "\U0001F3F3\ufe0f\u200d\U0001F308", "\U0001F469\U0001F3FD\u200d\U0001F4BB"]
    samples += ["\U000E0001", "\U000F0000", "\U0010FFFD"]   # tag / supplementary PUA / plane 16
    for text in samples:
        e, out, ok = ctx.rt(text)
        s.check(ok, "unicode round-trip", text, out, unicodedata.normalize("NFC", text))
        s.check(3 not in e.ids, "unicode hit <unk>", text, str(e.tokens[:8]), "no <unk>")

    # lone surrogate is not valid UTF-8 and cannot be encoded: must fail loudly, not corrupt
    lone = "\ud800"
    try:
        ctx.enc(lone)
        s.warnings.append({"kind": "lone surrogate accepted silently", "input": "\\ud800"})
    except Exception as exc:  # noqa: BLE001 - any loud failure is acceptable here
        s.notes.append(f"lone surrogate raises {type(exc).__name__} (input must be sanitized upstream)")
    return s


FUZZ_CHARS = (
    list("abcXYZ 0123456789.,!?-_/\\'\"()[]{}<>=\n\t\r:;%$#@&*+~`|")
    + list("கஙசஞடணதநபமயரலவழளறனஅஆஇஈஉஊஎஏஐஒஓஔஃ")
    + list("்ாிீுூெேைொோௌௗஂ")
    + list("௦௧௨௩௪௫௬௭௮௯௰௱௲ௐ")
    + list("éüñçßΩπ→√∞😀🔥👍🏽")
    + list("中文日本語한글العربيةעבריתกขคదేవతెలుగు")
    + list("\u0300\u0301\u0308\u200b\u200c\u200d\ufeff\u00a0\u2028\u3000")
)


def suite_fuzz(ctx: Ctx, n: int) -> Suite:
    s = Suite("fuzz", f"{n} seeded random mixed-script strings")
    rng = random.Random(20261106)
    total_ids = total_chars = 0
    idempotent_fail = 0
    for _ in range(n):
        k = rng.choice([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
        text = "".join(rng.choice(FUZZ_CHARS) for _ in range(k))
        e, out, ok = ctx.rt(text)
        s.check(ok, "fuzz round-trip", text, out, unicodedata.normalize("NFC", text))
        s.check(3 not in e.ids, "fuzz hit <unk>", text, str(e.tokens[:8]), "no <unk>")
        s.check(all(0 <= i < len(ctx.vocab) for i in e.ids), "id out of range", text)
        if ok:
            re_ids = ctx.enc(out).ids
            if re_ids != e.ids:
                idempotent_fail += 1
        total_ids += len(e.ids)
        total_chars += len(text)
    s.metrics.update({"strings": n, "tokens": total_ids, "chars": total_chars,
                      "chars_per_token": total_chars / max(total_ids, 1),
                      "non_idempotent": idempotent_fail})
    s.check(idempotent_fail == 0, "encode(decode(ids)) != ids", "", str(idempotent_fail), "0")
    return s


CODE_SNIPPETS = [
    'def add(a, b):\n    if a > 0:\n        return {"x": [a, b]}\n\treturn a + b\n',
    "int main() {\n    return 0;\n}\n",
    '<div class="a">\n  <p>hi</p>\n</div>',
    '{\n  "a": 1,\n  "b": [true, null]\n}',
    "key:\n  nested:\n    - one\n    - two\n",
    "SELECT a, b\nFROM t\nWHERE a = 1;\n",
    "```python\nprint('hi')\n```\n",
    "    leading indent\n        deeper\n",
    " one leading space",
    "trailing spaces   \n\n",
    "x\r\ny\r\n",
]


def suite_whitespace_code(ctx: Ctx) -> Suite:
    s = Suite("whitespace_code", "indent runs, tabs, CRLF, code exactness")
    texts = [f"{' ' * n}" for n in range(0, 41)]
    texts += [f"{'\t' * n}" for n in range(0, 9)]
    texts += [f"{'\n' * n}" for n in range(0, 6)]
    texts += [f"x{' ' * n}y" for n in (1, 2, 3, 4, 8, 9, 16, 17)]
    texts += [f"{' ' * a}x{' ' * b}" for a in (0, 1, 4) for b in (0, 1, 4)]
    texts += [" \t \n x", "\n\n\n    x\n\n", "x\t\t\ty", "x  \n  y", "\r\n\r\n", "a\rb"]
    texts += CODE_SNIPPETS
    for text in texts:
        e, out, ok = ctx.rt(text)
        s.check(ok, "whitespace/code round-trip", text, out, text)
        s.check(3 not in e.ids, "whitespace/code hit <unk>", text, str(e.tokens[:8]), "no <unk>")

    for ws in WHITESPACE_TOKENS:
        wid = ctx.tok.token_to_id(ws)
        s.check(wid is not None and [wid] == ctx.enc(ws).ids, "whitespace run not one token",
                ws, str(ctx.enc(ws).ids), str(wid))
    s.check("    " in ctx.enc("x\n    y").tokens, "4-space indent not a single token", "x\n    y")
    s.check("        " in ctx.enc("x\n        y").tokens, "8-space indent not a single token", "x\n        y")
    return s


def suite_specials(ctx: Ctx) -> Suite:
    s = Suite("specials_injection", "can plain text reach special/tool/reserved ids?")
    matched, dropped = [], []
    for t in SPECIAL_TOKENS:
        tid = ctx.tok.token_to_id(t)
        text = f"a {t} b"
        e = ctx.enc(text)
        if tid in e.ids:
            matched.append(t)
            if ctx.decode(e.ids) != unicodedata.normalize("NFC", text):
                dropped.append(t)
    s.metrics["specials_reachable_from_text"] = len(matched)
    s.metrics["specials_of"] = len(SPECIAL_TOKENS)
    s.metrics["examples"] = matched[:8]
    if matched:
        s.warn("special tokens are matched inside plain text (injectable; decode drops them)", matched[0],
               count=len(matched), examples=matched[:6])
    s.check(True, "informational", "")
    # byte-token literals must NOT be matched
    for lit in ("<0x41>", "<0xFF>"):
        s.check(lit not in ctx.enc(f"a {lit} b").tokens, "byte-token literal matched in text", lit)
    return s


def suite_collisions(ctx: Ctx) -> Suite:
    s = Suite("known_collisions", "raw PUA and U+2581 collide with the akshara/space encoding")
    cases = [chr(0xE000), chr(0xE001), "▁", "a▁b", "▁▁", chr(0xE000) + "தமிழ்"]
    lossy = 0
    for text in cases:
        e, out, ok = ctx.rt(text)
        if not ok:
            lossy += 1
            s.note_known("collision", text, out, unicodedata.normalize("NFC", text))
    s.metrics["lossy_cases"] = lossy
    s.metrics["pua_alphabet_size"] = len(PUA)
    # out-of-table PUA (plane 15) must still round-trip
    for cp in (0xF0000, 0xFFFFD, 0x10FFFD):
        text = chr(cp)
        _, out, ok = ctx.rt(text)
        s.check(ok, "out-of-table PUA round-trip", text, out, text)
    return s


def suite_integration(ctx: Ctx, tmp: Path) -> Suite:
    s = Suite("integration", "transformers load, batch, save/reload, threads")
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(ctx.path), local_files_only=True)
    s.check(hf.pad_token_id == 0 and hf.bos_token_id == 1 and hf.eos_token_id == 2
            and hf.unk_token_id == 3, "transformers special id mismatch", "",
            str([hf.pad_token_id, hf.bos_token_id, hf.eos_token_id, hf.unk_token_id]),
            "[0, 1, 2, 3]")
    text = "வணக்கம் நண்பா, how are you? 123"
    s.check(hf(text)["input_ids"] == ctx.enc(text).ids, "AutoTokenizer ids != raw tokenizers ids", text)
    s.check(hf.decode(ctx.enc(text).ids) == unicodedata.normalize("NFC", text),
            "AutoTokenizer decode mismatch", text)

    batch = ["தமிழ்", "hello", "123", "", "😀", "கா"]
    singles = [ctx.enc(t).ids for t in batch]
    s.check(ctx.tok.encode_batch(batch) and [e.ids for e in ctx.tok.encode_batch(batch)] == singles,
            "encode_batch != encode", str(batch))

    save = tmp / "reload" / "tokenizer.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    ctx.tok.save(str(save))
    again = Tokenizer.from_file(str(save))
    rng = random.Random(7)
    corpus = ["".join(rng.choice(FUZZ_CHARS) for _ in range(rng.randint(0, 60))) for _ in range(200)]
    same = all(again.encode(t, add_special_tokens=False).ids == ctx.enc(t).ids for t in corpus)
    s.check(same, "save/reload changes ids", "", str(same), "True")
    s.check(again.decode(again.encode("தமிழ் மொழி", add_special_tokens=False).ids) == "தமிழ் மொழி",
            "reloaded tokenizer decode", "தமிழ் மொழி")

    expected = {t: ctx.enc(t).ids for t in corpus}
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda t: (t, ctx.enc(t).ids), corpus))
    bad = [t for t, ids in results if ids != expected[t]]
    s.check(not bad, "encode not thread-safe/deterministic", str(bad[:3]), str(len(bad)), "0")
    return s


def _mb_text(unit: str, target: int = 1_000_000) -> str:
    reps = max(1, target // max(len(unit.encode()), 1))
    return unit * reps


PERF_TEXTS = {
    "tamil": "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்றாகும். இலக்கியம் செழித்தது. ",
    "tanglish": "vanakkam da, epdi iruke? naan nalla iruken, nee epdi? ",
    "english": "The quick brown fox jumps over the lazy dog while the scientist writes a paper. ",
    "code": "def add(a, b):\n    return a + b  # comment with நன்றி\n",
    "numbers": "1234567890 3.14159 1,00,000 2026-11-06 0xDEADBEEF ",
    "emoji": "😀🔥👍🏽👨‍👩‍👧‍👦🏳️‍🌈 ",
}


def suite_perf(ctx: Ctx) -> Suite:
    s = Suite("perf", "encode/decode throughput and long-document latency")
    ctx.enc("warmup")
    for name, unit in PERF_TEXTS.items():
        text = _mb_text(unit)
        nbytes = len(text.encode("utf-8"))
        t0 = time.perf_counter()
        e = ctx.enc(text)
        enc_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = ctx.decode(e.ids)
        dec_s = time.perf_counter() - t0
        s.check(out == text, f"{name} 1MB round-trip", name, "mismatch")
        s.metrics[name] = {
            "input_mb": round(nbytes / 1e6, 3),
            "tokens": len(e.ids),
            "encode_mb_s": round(nbytes / 1e6 / enc_s, 2),
            "encode_tok_s": round(len(e.ids) / enc_s, 0),
            "decode_mb_s": round(nbytes / 1e6 / dec_s, 2),
            "tokens_per_char": round(len(e.ids) / len(text), 3),
        }
    long_text = _mb_text(PERF_TEXTS["tamil"], 200_000)
    t0 = time.perf_counter()
    e = ctx.enc(long_text)
    s.metrics["long_doc_200k_chars_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    s.metrics["long_doc_tokens"] = len(e.ids)
    return s


def suite_compare(main: Ctx, other: Ctx) -> Suite:
    s = Suite(f"compare_{other.path.name}", f"{main.path.name} vs {other.path.name}")
    s.metrics["vocab_main"] = len(main.vocab)
    s.metrics["vocab_other"] = len(other.vocab)
    same_ids = all(main.tok.token_to_id(t) == other.tok.token_to_id(t) for t in SPECIAL_TOKENS[:6])
    s.check(same_ids, "core special ids differ between tokenizers", "", str(same_ids), "True")
    rng = random.Random(11)
    corpus = ["".join(rng.choice(FUZZ_CHARS) for _ in range(rng.randint(0, 80))) for _ in range(300)]
    corpus += [" ".join(rng.choice(list("கஙசஞடணதநபமயரலவழளறன")) for _ in range(20)) for _ in range(100)]
    n_main = n_other = other_bad = decode_diff = 0
    other_examples = []
    for text in corpus:
        em, om, okm = main.rt(text)
        eo, oo, oko = other.rt(text)
        n_main += len(em.ids)
        n_other += len(eo.ids)
        s.check(okm, "main tokenizer round-trip", text, om, unicodedata.normalize("NFC", text))
        if not oko:
            other_bad += 1
            if len(other_examples) < 5:
                other_examples.append({"input": esc(text), "decoded": esc(oo)})
        elif om != oo:
            decode_diff += 1
    if other_bad:
        s.warn(f"{other.path.name} is lossy on fuzz input", str(other_bad), examples=other_examples)
    if decode_diff:
        s.warn("tokenizers decode differently", str(decode_diff))
    s.metrics["tokens_main"] = n_main
    s.metrics["tokens_other"] = n_other
    s.metrics["main_vs_other_token_ratio"] = round(n_main / max(n_other, 1), 4)
    s.metrics["other_roundtrip_failures"] = other_bad

    # the v1 tokenizer was built with Metaspace prepend="always": leading spaces vanish
    lead_cases = [" x", "  ", " a b", " 1", " தமிழ்", "    x"]
    main_loss = [t for t in lead_cases if not main.rt(t)[2]]
    other_loss = [t for t in lead_cases if not other.rt(t)[2]]
    s.metrics["main_leading_space_lossy"] = [esc(t) for t in main_loss]
    s.metrics["other_leading_space_lossy"] = [esc(t) for t in other_loss]
    s.check(not main_loss, "main tokenizer loses leading space", str([esc(t) for t in main_loss]))
    if other_loss:
        s.warn(f"{other.path.name} loses leading whitespace (Metaspace prepend=always)", str(len(other_loss)),
               examples=[esc(t) for t in other_loss])
    return s


def render(ctx: Ctx, suites: list[Suite], elapsed: float) -> tuple[str, dict]:
    errors = sum(s.n_errors for s in suites)
    warnings = sum(len(s.warnings) for s in suites)
    checks = sum(s.checks for s in suites)
    verdict = "FAIL" if errors else ("PASS (with warnings)" if warnings else "PASS")
    lines = [
        "# Tokenizer hard test",
        "",
        f"- artifact: `{ctx.path}`",
        f"- vocab: {len(ctx.vocab):,}",
        f"- sha256: `{hashlib.sha256((ctx.path / 'tokenizer.json').read_bytes()).hexdigest()}`",
        f"- checks: {checks:,} · errors: {errors} · warnings: {warnings}",
        f"- wall time: {elapsed:.1f}s",
        f"- verdict: **{verdict}**",
        "",
        "| suite | checks | errors | warnings | status |",
        "|---|---|---|---|---|",
    ]
    for s in suites:
        status = "FAIL" if s.n_errors else ("WARN" if s.warnings else "ok")
        lines.append(f"| {s.name} | {s.checks:,} | {s.n_errors} | {len(s.warnings)} | {status} |")
    for s in suites:
        lines += ["", f"## {s.name}", "", s.description, ""]
        if s.metrics:
            lines.append("```json")
            lines.append(json.dumps(s.metrics, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
        if s.n_errors:
            lines.append(f"**{s.n_errors} error(s)**")
            lines.append("")
            lines.append("| kind | count |")
            lines.append("|---|---|")
            for kind, n in s.counts.most_common():
                lines.append(f"| {kind} | {n} |")
            lines.append("")
            lines.append("| kind | input | got | expected |")
            lines.append("|---|---|---|---|")
            for e in s.errors[:15]:
                lines.append(f"| {e['kind']} | `{e['input']}` | `{e.get('got', '')}` | `{e.get('expected', '')}` |")
            lines.append("")
        if s.warnings:
            lines.append(f"**{len(s.warnings)} warning(s)**")
            for w in s.warnings:
                extra = {k: v for k, v in w.items() if k not in ("kind", "input")}
                lines.append(f"- {w['kind']}: `{w['input']}` {json.dumps(extra, ensure_ascii=False)}")
            lines.append("")
        if s.known:
            lines.append("**Known lossy inputs (documented, must be sanitized upstream)**")
            lines.append("")
            for k in s.known:
                lines.append(f"- {k['kind']}: `{k['input']}` -> `{k['decoded']}` (expected `{k['expected']}`)")
            lines.append("")
        for n in s.notes:
            lines.append(f"- note: {n}")
        if s.notes:
            lines.append("")
    md = "\n".join(lines) + "\n"
    payload = {
        "artifact": str(ctx.path),
        "vocab_size": len(ctx.vocab),
        "sha256": hashlib.sha256((ctx.path / "tokenizer.json").read_bytes()).hexdigest(),
        "elapsed_s": round(elapsed, 1),
        "verdict": verdict,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "suites": [
            {"name": s.name, "description": s.description, "checks": s.checks,
             "errors": s.errors, "error_counts": dict(s.counts),
             "warnings": s.warnings, "known": s.known,
             "notes": s.notes, "metrics": s.metrics}
            for s in suites
        ],
    }
    return md, payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", type=Path, default=Path("artifacts/tok64k"))
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--fuzz", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("reports/tokenizer_hard_test"))
    ap.add_argument("--tmp", type=Path, default=Path(".pytest_cache/hard_test"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    ctx = Ctx(args.tok)
    suites = [
        suite_structure(ctx),
        suite_vocab_hygiene(ctx),
        suite_akshara(ctx),
        suite_tamil_block(ctx),
        suite_malformed(ctx),
        suite_unicode(ctx),
        suite_fuzz(ctx, args.fuzz),
        suite_whitespace_code(ctx),
        suite_specials(ctx),
        suite_collisions(ctx),
        suite_integration(ctx, args.tmp),
        suite_perf(ctx),
    ]
    if args.compare:
        suites.append(suite_compare(ctx, Ctx(args.compare)))
    md, payload = render(ctx, suites, time.perf_counter() - t0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(md.split("## ")[0])
    for s in suites:
        print(f"{s.name:22s} checks={s.checks:5d} errors={len(s.errors):3d} warnings={len(s.warnings):2d}")
    print(f"\nwrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.json')}")
    sys.exit(1 if payload["errors"] else 0)


if __name__ == "__main__":
    main()
