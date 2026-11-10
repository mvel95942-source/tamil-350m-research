"""Run a fixed generation probe against the locally downloaded Tamil SFT checkpoint.

This records generations for human scoring; it does not claim automatic correctness.
Run: python scripts/eval_tamil_reliability.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / "AppData/Local/Temp/opencode/kaggle-sft/sft_preview"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--cases", type=Path, default=ROOT / "eval/tamil_reliability_holdout.jsonl")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"Model directory not found: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", low_cpu_mem_usage=True, local_files_only=True
    ).eval()
    device = next(model.parameters()).device
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for case in cases:
        # Match the repository's training template exactly; no synthetic system turn/prefix.
        prompt = f"<s><|user|>\n{case['prompt']}</s><|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=1.15, no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        row = {**case, "answer": answer}
        output.append(row)
        print(f"\n[{case['id']}] {case['prompt']}\n=> {answer}")

    destination = args.out or ROOT / "reports" / f"tamil_reliability_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "model_path": str(args.model), "checkpoint_role": "SFT preview; no pre-SFT local baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decode": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "cases": output,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved generations for human review: {destination}")


if __name__ == "__main__":
    main()
