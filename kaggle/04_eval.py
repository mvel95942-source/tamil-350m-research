"""Kaggle GPU job: evaluate runs, then write reports/RESULTS.md *only* from the JSON outputs.

    python kaggle/04_eval.py --run main artifacts/run/ckpt.pt artifacts/tok32k/tokenizer.json 350m \
                             --run sft artifacts/sft/ckpt.pt artifacts/tok32k/tokenizer.json 350m \
                             --run ablation-ours artifacts/ablation_ours/ckpt.pt artifacts/tok32k/tokenizer.json 30m \
                             --run ablation-llama3 artifacts/ablation_llama3/ckpt.pt artifacts/tok_llama3/tokenizer.json 30m
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tamil.evaluate import eval_all  # noqa: E402

REPORTS = Path("reports")
TASKS = ["indicxnli", "indicsentiment", "belebele", "aksharantar_translit"]


def table(rows: dict[str, dict]) -> list[str]:
    out = ["| run | params | bpb tamil | bpb tanglish | bpb english | " + " | ".join(TASKS) + " |",
           "|---" * (5 + len(TASKS)) + "|"]
    for name, r in rows.items():
        b = r["bpb"]
        out.append(f"| {name} | {r['params'] / 1e6:.1f}M | {b['tamil']:.3f} | {b['tanglish_real']:.3f} | {b['english']:.3f} | "
                   + " | ".join(f"{r[t]:.1%}" for t in TASKS) + " |")
    return out


def write_results(evals: dict[str, dict]) -> Path:
    fert = REPORTS / "fertility.md"
    lines = ["# Tamil tokenizer + LM: results", "",
             "All numbers below are generated from JSON files in reports/ by kaggle/04_eval.py.", "",
             "## Tokenizer fertility (held-out text)", "",
             fert.read_text(encoding="utf-8") if fert.exists() else "_fertility.md missing: run tamil.fertility_", "",
             "## Models (lower bpb is better; accuracies 5-shot)", "", *table(evals), "",
             "## Ablation notes",
             "- Both ablation arms: same transformer body (layers, hidden size), same shards (same UTF-8 bytes),"
             " one epoch, same prompts.",
             "- Total parameters differ (see the params column): Llama-3's 128k vocab means a much larger embedding"
             " and LM head, and more tokens means more steps. Both favour the Llama arm.", "",
             "## What we don't claim",
             "- No claim of general reasoning: 350M params, under-trained (below Chinchilla-optimal tokens).",
             "- Our tokenizer's char offsets are wrong for Tamil (HF Replace-normalizer alignment); not for NER/token labeling.",
             "- Prompt wording for the labeled tasks is ours; scores depend on it.",
             "- Licences per source: see tamil/sources.py; verify before redistribution."]
    path = REPORTS / "RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=4, action="append", metavar=("NAME", "CKPT", "TOKENIZER", "SIZE"), required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    evals = {}
    for name, ckpt, tok, size in args.run:
        out = REPORTS / f"eval_{name}.json"
        evals[name] = json.loads(out.read_text()) if out.exists() else eval_all(Path(ckpt), Path(tok), out, size, args.n)
    print(write_results(evals).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
