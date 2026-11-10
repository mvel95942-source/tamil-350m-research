"""Build a small, filtered Tamil conversation pilot from Microsoft's Updesh beta.

Research-only pilot: the dataset card lists the Microsoft Research License. Review its
terms before use; do not treat this output as cleared for a public/commercial release.

Run in an environment with `datasets` installed:
    python tamil/prepare_updesh_chat.py --out artifacts/updesh_ta_pilot.jsonl --limit 3000

The script intentionally excludes system messages because this project's tokenizer/SFT
format does not currently train a system role. It keeps valid user/assistant turns only.
Every output line is a JSON list of {role, content}, compatible with tamil.sft.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

TAMIL = re.compile(r"[\u0B80-\u0BFF]")


def script_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(TAMIL.fullmatch(c)) for c in letters) / len(letters)


def repetition_ratio(text: str) -> float:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 8:
        return 0.0
    # Common function words recur naturally in Tamil. Measure repeated contiguous
    # four-word phrases instead of penalizing every repeated word across a conversation.
    phrases = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    if not phrases:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for phrase in phrases:
        counts[phrase] = counts.get(phrase, 0) + 1
    repeated = sum(n for n in counts.values() if n > 1)
    return repeated / len(phrases)


def normalize_messages(raw: object, min_tamil: float, max_repetition: float) -> list[dict[str, str]] | None:
    if not isinstance(raw, list):
        return None
    turns: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = unicodedata.normalize("NFC", content).strip()
        if not content:
            continue
        # Keep only alternating user/assistant turns; malformed sequences are rejected.
        if turns and turns[-1]["role"] == role:
            return None
        turns.append({"role": role, "content": content})
    if not turns or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        return None
    if len(turns) < 2 or len(turns) % 2:
        return None
    answers = "\n".join(t["content"] for t in turns if t["role"] == "assistant")
    if script_ratio(answers) < min_tamil or repetition_ratio(answers) > max_repetition:
        return None
    if any(len(t["content"]) > 6000 for t in turns):
        return None
    return turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--min-tamil", type=float, default=0.55)
    parser.add_argument("--max-repetition", type=float, default=0.35)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install `datasets` first: pip install datasets") from exc

    # Updesh card specifies subset=dialog_gen and split=tam_Taml.
    data = load_dataset("microsoft/Updesh_beta", "dialog_gen", split="tam_Taml")
    seen: set[str] = set()
    kept: list[list[dict[str, str]]] = []
    rejected = 0
    for row in data:
        turns = normalize_messages(row.get("messages"), args.min_tamil, args.max_repetition)
        if turns is None:
            rejected += 1
            continue
        key = "\n".join(t["role"] + ":" + t["content"].casefold() for t in turns)
        if key in seen:
            rejected += 1
            continue
        seen.add(key)
        kept.append(turns)
        if len(kept) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(turns, ensure_ascii=False) for turns in kept) + "\n",
        encoding="utf-8",
    )
    print(f"kept={len(kept)} rejected_or_duplicate={rejected} output={args.out}")
    print("RESEARCH ONLY: review Updesh's Microsoft Research License before redistribution or commercial use.")


if __name__ == "__main__":
    main()
