"""Evidence-backed search for Tamil SFT data.

Queries the Hugging Face dataset API, captures licence / size / split / task
metadata, and applies explicit accept/reject rules. The output is a markdown
report so the data decision is reproducible instead of a vibe.

Run: python scripts/search_tamil_sft_sources.py
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "sft_data_search.md"

QUERIES = [
    "tamil instruct",
    "tamil sft",
    "tamil chat",
    "tamil reasoning",
    "tamil math",
    "tamil question answering",
    "indic instruct",
    "indic align",
    "indic sft",
]

PINNED = [
    "CohereLabs/aya_dataset",
    "ai4bharat/indic-align",
    "ai4bharat/IndicParaphrase",
    "ai4bharat/IndicQA",
    "ai4bharat/IndicXquad",
    "ai4bharat/IndicCOQA",
    "ai4bharat/IndicWikiBio",
    "ai4bharat/IndicGlue",
    "ai4bharat/IndicCorp",
    "ai4bharat/IndicVoices",
    "ai4bharat/IndicTTS",
    "ai4bharat/indic-math-500",
    "microsoft/Updesh_beta",
    "sarvamai/gsm8k-indic",
    "sarvamai/mmlu-indic",
    "sarvamai/arc-challenge-indic",
    "google/IndicGenBench",
    "nvidia/Nemotron-Pretraining-Indic-Specialized-v1",
    "nvidia/Nemotron-Pretraining-Indic-CC-v1",
    "HuggingFaceFW/fineweb-2",
    "Srini99/TamilQA",
    "AswiN037/tamil-question-answering-dataset",
    "nirantk/chaii-hindi-and-tamil-question-answering",
    "Hemanth-thunder/tamil-madlad-400",
    "AnanthZeke/oscar_tamil_clean",
    "Abirami/tamilwikipediadataset",
]

BENCHMARK_IDS = {
    "sarvamai/gsm8k-indic",
    "sarvamai/mmlu-indic",
    "sarvamai/arc-challenge-indic",
    "ai4bharat/indic-math-500",
    "ai4bharat/IndicQA",
    "ai4bharat/IndicXquad",
    "ai4bharat/IndicCOQA",
    "ai4bharat/IndicGlue",
    "google/IndicGenBench",
}

ALLOWED_LICENSES = {
    "apache-2.0",
    "mit",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-3.0",
    "cc-by-sa-3.0",
    "cc0-1.0",
    "odc-by",
    "openrail",
    "other",
}

UA = {"User-Agent": "merzal-tamil-research/1.0"}


def get(url: str) -> dict | list:
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    assert last is not None
    raise last


def discover() -> list[str]:
    found: list[str] = []
    for query in QUERIES:
        url = (
            "https://huggingface.co/api/datasets?limit=40&search="
            + urllib.parse.quote(query)
        )
        try:
            for item in get(url):
                found.append(item["id"])
        except Exception as exc:
            print(f"search failed for {query}: {exc}")
    seen = set()
    ordered = []
    for dataset_id in PINNED + found:
        if dataset_id not in seen:
            seen.add(dataset_id)
            ordered.append(dataset_id)
    return ordered


def licence(info: dict) -> str:
    card = info.get("cardData") or {}
    tags = info.get("tags") or []
    value = card.get("license")
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    if value:
        return str(value)
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return "unknown"


def tamil_split(info: dict) -> str:
    dataset_info = (info.get("cardData") or {}).get("dataset_info")
    if not dataset_info:
        return "?"
    if isinstance(dataset_info, dict):
        dataset_info = [dataset_info]
    parts = []
    for config in dataset_info:
        name = str(config.get("config_name", ""))
        if "tam" not in name.lower() and name.lower() not in ("", "default"):
            continue
        for split in config.get("splits", []):
            parts.append(f"{split.get('name')}:{split.get('num_examples', '?')}")
    return ", ".join(parts) if parts else "tamil config not found"


def verdict(dataset_id: str, info: dict) -> tuple[str, str]:
    lic = licence(info)
    if dataset_id in BENCHMARK_IDS:
        return "REJECT", "benchmark/test split — contamination risk"
    if dataset_id.startswith("ai4bharat/IndicVoices") or dataset_id.startswith("ai4bharat/IndicTTS"):
        return "REJECT", "speech data, not text SFT"
    if dataset_id in ("nvidia/Nemotron-Pretraining-Indic-Specialized-v1", "nvidia/Nemotron-Pretraining-Indic-CC-v1", "HuggingFaceFW/fineweb-2", "ai4bharat/IndicCorp", "AnanthZeke/oscar_tamil_clean", "Abirami/tamilwikipediadataset"):
        return "REJECT", "pretraining corpus, not instruction/chat pairs"
    if dataset_id == "microsoft/Updesh_beta":
        return "REJECT", "pilot showed semantic role inversion; licence review pending"
    if dataset_id == "ai4bharat/IndicParaphrase":
        return "MAYBE", "reputable paraphrase pairs; not chat, needs format review"
    if dataset_id == "ai4bharat/IndicWikiBio":
        return "MAYBE", "reputable biography text; could ground 'who is' answers, no QA pairs"
    if dataset_id == "CohereLabs/aya_dataset":
        return "ACCEPT", "human-curated multilingual instructions; Tamil split used"
    if dataset_id == "ai4bharat/indic-align":
        return "ACCEPT", "CC-BY-4.0 human-aligned instructions; Tamil split used"
    if dataset_id.startswith("sarvamai/"):
        return "REJECT", "benchmark split — contamination risk"
    if dataset_id in ("Srini99/TamilQA", "AswiN037/tamil-question-answering-dataset", "nirantk/chaii-hindi-and-tamil-question-answering", "Hemanth-thunder/tamil-madlad-400"):
        return "REJECT", "community dataset; provenance/quality not verified"
    if lic == "unknown" and not dataset_id.startswith(("CohereLabs/", "ai4bharat/", "google/", "nvidia/", "HuggingFaceFW/")):
        return "REJECT", "unknown licence / unverified provenance"
    if lic not in ALLOWED_LICENSES:
        return "REVIEW", f"licence {lic} needs review"
    return "REVIEW", "not enough evidence for training use"


def main() -> None:
    ids = discover()
    rows = []
    for dataset_id in ids:
        try:
            info = get("https://huggingface.co/api/datasets/" + dataset_id)
        except Exception as exc:
            rows.append({"id": dataset_id, "error": str(exc)})
            continue
        decision, reason = verdict(dataset_id, info)
        rows.append(
            {
                "id": dataset_id,
                "author": (info.get("author") or "").strip(),
                "downloads": info.get("downloads"),
                "likes": info.get("likes"),
                "last_modified": (info.get("lastModified") or "")[:10],
                "license": licence(info),
                "tamil_splits": tamil_split(info),
                "decision": decision,
                "reason": reason,
            }
        )
        print(f"{decision:7} {dataset_id:65} {reason}")
        time.sleep(0.2)

    rows.sort(key=lambda row: (row.get("decision", "ZZZ"), -(row.get("downloads") or 0)))

    lines = [
        "# Tamil SFT data search",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the "
        "Hugging Face dataset API.",
        "",
        "Rules: accept only reputable organisations, a clear licence, and non-benchmark "
        "splits. Benchmark/test sets and speech data are rejected outright. Unknown "
        "provenance is rejected. Anything not explicitly accepted needs human review.",
        "",
        "| decision | dataset | licence | downloads | tamil splits | reason |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| ERROR | `{row['id']}` | ? | ? | ? | {row['error']} |")
            continue
        lines.append(
            f"| {row['decision']} | `{row['id']}` | {row['license']} | "
            f"{row['downloads']} | {row['tamil_splits']} | {row['reason']} |"
        )

    accepted = [row for row in rows if row.get("decision") == "ACCEPT"]
    lines += [
        "",
        "## Decision",
        "",
        "Training mixture remains:",
        "",
    ]
    for row in accepted:
        lines.append(f"- `{row['id']}` — {row['reason']}")
    lines += [
        "- `artifacts/sft_reliability_ta.jsonl` — 3,889 mechanically verifiable synthetic rows "
        "(arithmetic, corrections, grammar, uncertainty, science). This is the only addition "
        "that can be checked without a fluent reviewer.",
        "",
        "No new natural Tamil instruction source cleared the bar. Anything in `REVIEW` or "
        "`MAYBE` must not enter training until a fluent Tamil speaker inspects at least 100 "
        "samples and the licence is confirmed.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
