"""Evaluation: bits per byte (comparable across tokenizers) and labeled Tamil tasks.

Multiple-choice tasks are scored by length-normalized log-likelihood (per UTF-8 byte of the
answer), 5-shot, with the shots taken from the head of each test set and excluded from scoring.
"""
import argparse
import json
import math
from collections.abc import Callable
from itertools import islice
from pathlib import Path

import torch

Encode = Callable[[str], list[int]]
SHOTS = 5
# what precedes every document in the packed training stream (no <s> is ever trained on)
DOC_SEP_TOKENS = ("</s>", "<|end_of_text|>", "<|endoftext|>", "<eos>")


def doc_sep_id(tok) -> int:
    for t in DOC_SEP_TOKENS:
        if (i := tok.token_to_id(t)) is not None:
            return i
    raise ValueError("tokenizer has no known end-of-document token")


@torch.no_grad()
def _logprobs(model, ids: list[int]) -> torch.Tensor:
    """log p(ids[i] | ids[:i]) for i >= 1."""
    x = torch.tensor([ids[:-1]], device=next(model.parameters(), torch.zeros(0)).device)
    logp = torch.log_softmax(model(input_ids=x).logits[0].float(), dim=-1)
    return logp.gather(1, torch.tensor(ids[1:], device=logp.device)[:, None])[:, 0]


@torch.no_grad()
def bits_per_byte(model, encode: Encode, texts: list[str], bos_id: int, ctx: int = 1024) -> float:
    nll, n_bytes = 0.0, 0
    for t in texts:
        ids = encode(t)
        for i in range(0, len(ids), ctx):
            chunk = [bos_id] + ids[i:i + ctx]
            nll -= _logprobs(model, chunk).sum().item()
        n_bytes += len(t.encode("utf-8"))
    return nll / math.log(2) / max(n_bytes, 1)


@torch.no_grad()
def loglik_choice(model, encode: Encode, prompt: str, choices: list[str], bos_id: int) -> int:
    p = [bos_id] + encode(prompt)
    scores = []
    for c in choices:
        ids = p + encode(c)
        lp = _logprobs(model, ids)[len(p) - 1:].sum().item()
        scores.append(lp / max(len(c.encode("utf-8")), 1))
    return max(range(len(choices)), key=scores.__getitem__)


# ---- labeled tasks: (prompt, choices, gold index). Prompt wording is our call (ledgered). ----

def _xnli(r):
    return (f"{r['premise']}\nகேள்வி: {r['hypothesis']} சரியா?\nபதில்:",
            [" ஆம்", " இருக்கலாம்", " இல்லை"], int(r["label"]))


def _sentiment(r):
    return (f"விமர்சனம்: {r['INDIC REVIEW']}\nஉணர்வு:", [" நேர்மறை", " எதிர்மறை"],
            0 if str(r["LABEL"]).lower().startswith("pos") else 1)


def _belebele(r):
    opts = [r[f"mc_answer{i}"] for i in range(1, 5)]
    return (f"{r['flores_passage']}\nகேள்வி: {r['question']}\nபதில்:", [" " + o for o in opts],
            int(r["correct_answer_num"]) - 1)


MC_TASKS = {"indicxnli": ("INDICXNLI_TA", _xnli), "indicsentiment": ("INDICSENTIMENT_TA", _sentiment),
            "belebele": ("BELEBELE_TA", _belebele)}


def _shot_text(items) -> str:
    return "".join(p + c[g] + "\n\n" for p, c, g in items)


def fit_shots(encode: Encode, shot_items, prompt: str, choices: list[str], ctx: int) -> int:
    """Most shots (<= len(shot_items)) whose prompt + longest choice fits the trained context."""
    longest = max(len(encode(c)) for c in choices)
    for k in range(len(shot_items), -1, -1):
        if 1 + len(encode(_shot_text(shot_items[:k]) + prompt)) + longest <= ctx:
            return k
    return 0


def _mc_accuracy(model, encode, rows, fmt, bos_id, ctx: int = 1024) -> float:
    items = [fmt(r) for r in rows]
    shot_items, scored = items[:SHOTS], items[SHOTS:]
    hits = 0
    for p, c, g in scored:
        k = fit_shots(encode, shot_items, p, c, ctx)  # beyond ctx positions were never trained
        hits += loglik_choice(model, encode, _shot_text(shot_items[:k]) + p, c, bos_id) == g
    return hits / max(len(scored), 1)


@torch.no_grad()
def _translit_accuracy(model, encode, decode, rows, bos_id, eos_id) -> float:
    pairs = [(r["english word"], r["native word"]) for r in rows]
    shots = "".join(f"{en} = {ta}\n" for en, ta in pairs[:SHOTS])
    hits = 0
    for en, ta in pairs[SHOTS:]:
        ids = [bos_id] + encode(shots + f"{en} =")
        x = torch.tensor([ids], device=next(model.parameters()).device)
        out = model.generate(x, max_new_tokens=16, do_sample=False, eos_token_id=eos_id, pad_token_id=eos_id)
        guess = decode(out[0, len(ids):].tolist()).split("\n")[0].strip()
        hits += guess == ta
    return hits / max(len(pairs) - SHOTS, 1)


def eval_all(ckpt: Path, tokenizer_path: Path, out_json: Path, model_size: str = "350m", n: int = 500) -> dict:
    from tokenizers import Tokenizer

    from tamil import sources
    from tamil.corpus import load_rows
    from tamil.fertility import load_heldout
    from tamil.model import make_model

    tok = Tokenizer.from_file(str(tokenizer_path))
    model = make_model(tok.get_vocab_size(), model_size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    if torch.cuda.is_available():
        model = model.half()
    bos = eos = doc_sep_id(tok)  # start every text the way documents start in training

    def encode(s):
        return tok.encode(s, add_special_tokens=False).ids

    held = load_heldout(n)
    res = {"params": sum(p.numel() for p in model.parameters()),
           "bpb": {lang: bits_per_byte(model, encode, held[lang], bos) for lang in ("tamil", "tanglish_real", "english")}}
    for name, (src_name, fmt) in MC_TASKS.items():
        rows = list(islice(load_rows(getattr(sources, src_name)), n + SHOTS))
        res[name] = _mc_accuracy(model, encode, rows, fmt, bos)
    rows = list(islice(load_rows(sources.AKSHARANTAR_TA), n + SHOTS))
    res["aksharantar_translit"] = _translit_accuracy(model, encode, tok.decode, rows, bos, eos)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model_size", default="350m")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    print(json.dumps(eval_all(a.ckpt, a.tokenizer, a.out, a.model_size, a.n), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
