"""Romanize Tamil script into Tanglish, with seeded spelling noise.

Real Tanglish has no standard spelling, so the casual/chat schemes randomly pick
the variants people actually type (zh/l/z, th/t/d, aa/a, ...).
"""
import re
from random import Random
from typing import Literal

from tamil.akshara import PULLI, split_aksharas

CONS = {"க": "k", "ங": "ng", "ச": "ch", "ஞ": "nj", "ட": "d", "ண": "n", "த": "th", "ந": "n",
        "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ற": "r", "ல": "l", "ள": "l", "ழ": "zh",
        "வ": "v", "ன": "n", "ஜ": "j", "ஶ": "sh", "ஷ": "sh", "ஸ": "s", "ஹ": "h"}
VOWEL = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo", "எ": "e", "ஏ": "e",
         "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au", "ஃ": "h"}
SIGN = {"ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo", "ெ": "e", "ே": "e", "ை": "ai",
        "ொ": "o", "ோ": "o", "ௌ": "au"}
# doubled consonant (C + pulli + C...): (first half, onset of the second)
GEMINATE = {"ட": ("t", "t"), "ற": ("t", "r")}
# C1 + pulli followed by a different C2: (C1 sound, C2 onset), e.g. நன்றி -> nandri
CLUSTER = {("ன", "ற"): ("n", "dr")}
VARIANTS = {"aa": ["a"], "ee": ["i"], "oo": ["u"], "zh": ["l", "z"], "th": ["t", "d"],
            "ch": ["s", "c"], "ai": ["ei", "e"], "v": ["w"]}

Scheme = Literal["iso", "casual", "chat"]


def _gem(c: str) -> tuple[str, str]:
    return GEMINATE.get(c, (CONS[c][0], CONS[c]))


def _unit(u: str, onset: str | None) -> str:
    if u in VOWEL:
        return VOWEL[u]
    c, rest = u[0], u[1:]
    if c not in CONS:
        return u
    base = onset if onset is not None else CONS[c]
    if rest == PULLI:
        return base
    return base + SIGN.get(rest, "a")


def _noisy(s: str, rng: Random, noise: float) -> str:
    keys = [k for k in VARIANTS if k in s]
    if keys and rng.random() < noise:
        k = rng.choice(keys)
        s = s.replace(k, rng.choice(VARIANTS[k]), 1)
    return s


def _chat_word(w: str, rng: Random, noise: float) -> str:
    if rng.random() < noise:
        w = re.sub(r"([a-z])\1", r"\1", w)
    if w.endswith("u") and len(w) > 3 and rng.random() < noise / 2:
        w = w[:-1]
    return w


def romanize(text: str, scheme: Scheme, rng: Random, noise: float = 0.0) -> str:
    units = split_aksharas(text)
    out, onset = [], None
    for i, u in enumerate(units):
        nxt = units[i + 1] if i + 1 < len(units) else ""
        pair = (u[0], nxt[:1])
        if u[-1:] == PULLI and u[0] in CONS and (nxt[:1] == u[0] or pair in CLUSTER):
            first, onset_next = CLUSTER.get(pair) or _gem(u[0])
            out.append(first)
            onset = onset_next
            continue
        r = _unit(u, onset)
        onset = None
        if scheme != "iso" and r != u:
            r = _noisy(r, rng, noise)
        out.append(r)
    s = "".join(out)
    if scheme == "chat":
        s = re.sub(r"[a-z]+", lambda m: _chat_word(m.group(), rng, noise), s)
    return s
