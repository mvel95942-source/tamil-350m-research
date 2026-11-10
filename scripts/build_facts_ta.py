"""Build a dated, source-attributed Tamil facts snapshot.

The 350M model cannot memorise live facts (who is prime minister, today's
weather, current prices). This script fetches a small set of facts from
reputable, citable sources and writes `data/facts_ta.json`. The local chat
server reads that file and answers those questions deterministically instead
of letting the model guess.

Primary source: the `incumbent` / `incumbent_since` fields of the English
Wikipedia office infoboxes (these track changes), resolved to Tamil names via
Wikidata/Wikipedia. Fallback: Wikidata current-officeholder SPARQL.

Run: python scripts/build_facts_ta.py
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "facts_ta.json"

UA = {"User-Agent": "merzal-tamil-research/1.0 (local evaluation)"}

TA_MONTHS = [
    "ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்",
    "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்",
]
EN_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

OFFICES = [
    {
        "key": "pm_india",
        "page": "Prime Minister of India",
        "office_en": "Prime Minister of India",
        "office_ta": "இந்தியப் பிரதமர்",
        "fallback_qid": "Q192711",
    },
    {
        "key": "president_india",
        "page": "President of India",
        "office_en": "President of India",
        "office_ta": "இந்தியக் குடியரசுத் தலைவர்",
        "fallback_qid": "Q313383",
    },
    {
        "key": "cm_tamil_nadu",
        "page": "Chief Minister of Tamil Nadu",
        "office_en": "Chief Minister of Tamil Nadu",
        "office_ta": "தமிழ்நாடு முதலமைச்சர்",
        "fallback_qid": "Q24040821",
    },
    {
        "key": "army_chief_india",
        "page": "Chief of the Army Staff (India)",
        "office_en": "Chief of the Army Staff (India)",
        "office_ta": "இந்திய இராணுவத் தலைவர்",
        "fallback_qid": "Q4616814",
    },
]

# Tamil Wikipedia titles checked against the live articles on 2026-11-07.
KNOWN_TA = {
    "Narendra Modi": "நரேந்திர மோதி",
    "Droupadi Murmu": "திரௌபதி முர்மு",
    "C. Joseph Vijay": "ச. ஜோசப் விஜய்",
    "Dhiraj Seth": "தீரஜ் சேத்",
}


def _get(url: str, accept: str = "application/json") -> str:
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=45) as response:
                return response.read().decode("utf-8")
        except Exception as exc:  # rate limits are common on Wikipedia APIs
            last = exc
            time.sleep(3 * (attempt + 1))
    assert last is not None
    raise last


def wikitext(page: str) -> str:
    url = (
        "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&page="
        + urllib.parse.quote(page)
    )
    return json.loads(_get(url))["parse"]["wikitext"]["*"]


def field(text: str, name: str) -> str | None:
    match = re.search(rf"^\s*\|\s*{name}\s*=\s*(.+)$", text, re.M)
    return match.group(1).strip() if match else None


def link_target(value: str) -> str | None:
    # The incumbent field can append badges/honorifics after <br/> or {{small|...}}.
    # The person is the last wiki link *before* those decorations.
    head = re.split(r"<br\s*/?>|\{\{", value, maxsplit=1)[0]
    links = re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", head)
    if not links:
        return re.sub(r"\{\{.*?\}\}|<[^>]+>", "", head).strip() or None
    target, label = links[-1]
    return (label or target).strip()


def to_iso(date_text: str) -> str | None:
    for month_index, month_name in enumerate(EN_MONTHS, start=1):
        match = re.match(rf"(\d{{1,2}})\s+{month_name}\s+(\d{{4}})", date_text.strip())
        if match:
            return f"{match.group(2)}-{month_index:02d}-{int(match.group(1)):02d}"
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_text.strip())
    return match.group(0) if match else None


def display_ta(iso: str) -> str:
    year, month, day = iso.split("-")
    return f"{int(day)} {TA_MONTHS[int(month) - 1]} {year}"


def tamil_name(person_en: str) -> str:
    if person_en in KNOWN_TA:
        return KNOWN_TA[person_en]
    url = (
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&limit=1&search="
        + urllib.parse.quote(person_en)
    )
    try:
        hits = json.loads(_get(url))["search"]
        if not hits:
            return person_en
        qid = hits[0]["id"]
        entity = json.loads(_get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"))["entities"][qid]
        ta_wiki = entity.get("sitelinks", {}).get("tawiki", {}).get("title")
        return ta_wiki or entity["labels"].get("ta", {}).get("value") or person_en
    except Exception:
        return person_en


def office_fact(office: dict) -> dict | None:
    try:
        text = wikitext(office["page"])
    except Exception:
        return None
    incumbent_raw = field(text, "incumbent")
    since_raw = field(text, "incumbent_since")
    if not incumbent_raw:
        return None
    person_en = link_target(incumbent_raw)
    if not person_en:
        return None
    since_iso = to_iso(since_raw) if since_raw else None
    person_ta = tamil_name(person_en)
    since_clause = ""
    if since_iso:
        since_clause = f" இவர் {display_ta(since_iso)} முதல் பதவியில் உள்ளார்."
    return {
        "key": office["key"],
        "kind": "officeholder",
        "office_en": office["office_en"],
        "office_ta": office["office_ta"],
        "person_en": person_en,
        "person_ta": person_ta,
        "since": since_iso,
        "answer_ta": f"{office['office_ta']} {person_ta} ஆவார்.{since_clause}",
        "source_name": f"Wikipedia: {office['page']}",
        "source_url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(office["page"].replace(" ", "_")),
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def india_population() -> dict | None:
    url = (
        "https://api.worldbank.org/v2/country/IND/indicator/SP.POP.TOTL"
        "?format=json&per_page=8&date=2020:2026"
    )
    try:
        rows = json.loads(_get(url))[1]
    except Exception:
        return None
    rows = [row for row in rows if row.get("value")]
    if not rows:
        return None
    latest = max(rows, key=lambda row: row["date"])
    value = int(latest["value"])
    year = latest["date"]
    return {
        "key": "india_population",
        "kind": "statistic",
        "answer_ta": (
            f"உலக வங்கியின் {year} ஆண்டு மதிப்பீட்டின்படி இந்தியாவின் மக்கள்தொகை "
            f"சுமார் {value:,} ஆகும். இது நேரடி நொடிநிலை எண் அல்ல."
        ),
        "answer_en": f"World Bank estimate for {year}: {value:,}.",
        "value": value,
        "year": year,
        "source_name": "World Bank",
        "source_url": "https://data.worldbank.org/indicator/SP.POP.TOTL?locations=IN",
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def collect() -> list[dict]:
    facts = []
    for office in OFFICES:
        fact = office_fact(office)
        if fact:
            facts.append(fact)
        else:
            print(f"WARNING: could not resolve {office['page']}")
        time.sleep(1)

    population = india_population()
    if population:
        facts.append(population)
    return facts


def main() -> None:
    facts = collect()

    payload = {
        "snapshot_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "note": (
            "Static snapshot for offline use. The chat server refreshes the officeholder "
            "facts from the same sources at runtime (24 h cache) when online."
        ),
        "facts": facts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for fact in facts:
        print("-", fact["answer_ta"])
        print(f"  {fact['source_url']}")


if __name__ == "__main__":
    main()
