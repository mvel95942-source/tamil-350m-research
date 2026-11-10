"""Deterministic Tamil facts/grounding layer for the local chat.

The 350M model must not guess live or dated facts. This module loads a
source-attributed snapshot (`data/facts_ta.json`), refreshes it from Wikipedia /
Wikidata / World Bank at most once a day, and answers matching questions
directly. Anything it cannot ground is refused instead of hallucinated.

Usage:
    from facts_ta import answer
    result = answer("இந்தியப் பிரதமர் யார்?")
    # {"text": "...", "kind": "officeholder", "source_url": "..."}
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTS_PATH = ROOT / "data" / "facts_ta.json"
CACHE_MAX_AGE = 24 * 60 * 60

UA = {"User-Agent": "merzal-tamil-research/1.0 (local evaluation)"}

ABSTAIN = (
    "இது நேரடியாக மாறக்கூடிய தகவல். என்னிடம் நம்பகமான நேரடி ஆதாரம் இல்லை, "
    "எனவே ஊகித்துச் சொல்லமாட்டேன். தயவுசெய்து அதிகாரப்பூர்வ ஆதாரத்தில் சரிபார்க்கவும்."
)

OFFICE_KEYS = {
    "pm_india": [
        "பிரதமர்", "பிரதம மந்திரி", "பிரதமமந்திரி", "prime minister", "pradhan mantri",
    ],
    "president_india": [
        "குடியரசுத் தலைவர்", "குடியரசு தலைவர்", "ஜனாதிபதி", "அதிபர்", "president",
    ],
    "cm_tamil_nadu": [
        "முதலமைச்சர்", "முதல்வர்", "முதல் அமைச்சர்", "chief minister",
    ],
    "army_chief_india": [
        "இராணுவத் தலைவர்", "ராணுவத் தலைவர்", "இராணுவ தலைவர்", "army chief",
        "chief of army staff", "coas", "தளபதி",
    ],
}

WEATHER_WORDS = ["வானிலை", "மழை", "வெப்பநிலை", "weather", "rain", "temperature", "forecast"]
TOMORROW_WORDS = ["நாளை", "நாளைக்கு", "tomorrow"]
WHO_WORDS = ["யார்", "யாரு", "எவர்", "who"]
PERSON_ALIASES = {
    "pm_india": ["நரேந்திர மோதி", "மோதி", "narendra modi", "modi"],
    "president_india": ["திரௌபதி முர்மு", "முர்மு", "droupadi murmu", "murmu"],
    "cm_tamil_nadu": ["ஜோசப் விஜய்", "ச. ஜோசப் விஜய்", "joseph vijay"],
    "army_chief_india": ["தீரஜ் சேத்", "தீரஜ்", "dhiraj seth"],
}
PRICE_WORDS = [
    "விலை", "price", "rate", "தங்கம்", "gold", "பெட்ரோல்", "petrol", "டீசல்",
    "diesel", "சந்தை", "market rate", "வெள்ளி", "silver",
]
NEWS_WORDS = ["செய்தி", "news", "latest", "அண்மை", "இன்றைய செய்தி", "headline"]

CREATIVE_WORDS = [
    "கவிதை", "பாடல்", "கதை", "கட்டுரை", "எழுது", "வரை", "போற்று", "விளம்பரம்",
    "poem", "song", "story", "essay", "write", "draft", "compose", "imagine",
]

INDIA_WORDS = ["இந்தியா", "இந்திய", "india", "indian", "நமது நாடு", "நம் நாடு", "பாரத"]
OTHER_PLACES = [
    "அமெரிக்கா", "அமெரிக்க", "america", "usa", "u.s", "united states",
    "சீனா", "china", "ரஷ்யா", "russia", "ஜப்பான்", "japan", "பாகிஸ்தான்",
    "pakistan", "வங்கதேசம்", "bangladesh", "இலங்கை", "sri lanka", "நேபாளம்",
    "nepal", "பிரிட்டன்", "britain", "uk", "england", "பிரான்ஸ்", "france",
    "ஜெர்மனி", "germany", "ஆஸ்திரேலியா", "australia", "கனடா", "canada",
    "கேரளா", "kerala", "கர்நாடகா", "karnataka", "ஆந்திரா", "andhra",
    "தெலங்கானா", "telangana", "மகாராஷ்டிரா", "maharashtra", "குஜராத்",
    "gujarat", "ராஜஸ்தான்", "rajasthan", "உத்தரப் பிரதேசம்", "uttar pradesh",
    "மேற்கு வங்கம்", "west bengal", "பஞ்சாப்", "punjab", "ஒடிசா", "odisha",
]

CITIES = {
    "சென்னை": (13.0827, 80.2707, "சென்னை"),
    "chennai": (13.0827, 80.2707, "சென்னை"),
    "கோயம்புத்தூர்": (11.0168, 76.9558, "கோயம்புத்தூர்"),
    "கோவை": (11.0168, 76.9558, "கோயம்புத்தூர்"),
    "coimbatore": (11.0168, 76.9558, "கோயம்புத்தூர்"),
    "மதுரை": (9.9252, 78.1198, "மதுரை"),
    "madurai": (9.9252, 78.1198, "மதுரை"),
    "திருச்சி": (10.7905, 78.7047, "திருச்சி"),
    "trichy": (10.7905, 78.7047, "திருச்சி"),
    "tiruchirappalli": (10.7905, 78.7047, "திருச்சி"),
    "சேலம்": (11.6643, 78.1460, "சேலம்"),
    "salem": (11.6643, 78.1460, "சேலம்"),
    "திருநெல்வேலி": (8.7139, 77.7567, "திருநெல்வேலி"),
    "tirunelveli": (8.7139, 77.7567, "திருநெல்வேலி"),
    "வேலூர்": (12.9165, 79.1325, "வேலூர்"),
    "vellore": (12.9165, 79.1325, "வேலூர்"),
    "ஈரோடு": (11.3410, 77.7172, "ஈரோடு"),
    "erode": (11.3410, 77.7172, "ஈரோடு"),
    "தஞ்சாவூர்": (10.7870, 79.1378, "தஞ்சாவூர்"),
    "thanjavur": (10.7870, 79.1378, "தஞ்சாவூர்"),
    "கன்னியாகுமரி": (8.0883, 77.5385, "கன்னியாகுமரி"),
    "kanyakumari": (8.0883, 77.5385, "கன்னியாகுமரி"),
}

WEATHER_CODES = {
    0: "தெளிவான வானம்",
    1: "கிட்டத்தட்ட தெளிவான வானம்",
    2: "பகுதி மேகமூட்டம்",
    3: "மேகமூட்டம்",
    45: "பனிமூட்டம்",
    48: "பனிமூட்டம்",
    51: "இலகு தூறல்",
    53: "தூறல்",
    55: "கனமான தூறல்",
    61: "இலகு மழை",
    63: "மழை",
    65: "கனமான மழை",
    66: "உறைந்த மழை",
    67: "கனமான உறைந்த மழை",
    71: "இலகு பனிப்பொழிவு",
    73: "பனிப்பொழிவு",
    75: "கனமான பனிப்பொழிவு",
    80: "இலகு மழைத்தூறல்",
    81: "மழைத்தூறல்",
    82: "கனமான மழைத்தூறல்",
    95: "இடியுடன் மழை",
    96: "இடியுடன் மழை",
    99: "கனமான இடியுடன் மழை",
}
RAIN_CODES = set(range(51, 68)) | set(range(80, 83)) | {95, 96, 99}

_state_lock = threading.Lock()
_facts: dict[str, dict] = {}
_snapshot_date: str | None = None


def _load() -> None:
    global _facts, _snapshot_date
    try:
        payload = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        _facts, _snapshot_date = {}, None
        return
    _facts = {fact["key"]: fact for fact in payload.get("facts", [])}
    _snapshot_date = payload.get("snapshot_date")


def refresh(force: bool = False) -> bool:
    """Rebuild the snapshot from the upstream sources. Never raises."""
    try:
        try:
            import build_facts_ta
        except Exception:
            from scripts import build_facts_ta  # type: ignore
        if not force and FACTS_PATH.exists():
            age = time.time() - FACTS_PATH.stat().st_mtime
            if age < CACHE_MAX_AGE:
                return False
        facts = build_facts_ta.collect()
        if not facts:
            return False
        payload = {
            "snapshot_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "note": "Refreshed automatically by scripts/facts_ta.py.",
            "facts": facts,
        }
        FACTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with _state_lock:
            _load()
        return True
    except Exception as exc:
        print(f"[facts] refresh skipped: {exc}")
        return False


def refresh_in_background() -> None:
    threading.Thread(target=refresh, kwargs={"force": False}, daemon=True).start()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _has_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def _has_ascii_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text) is not None


def _is_creative(text: str) -> bool:
    return _has_any(text, CREATIVE_WORDS)


def _mentions_other_place(text: str) -> bool:
    if _has_any(text, INDIA_WORDS):
        return False
    return _has_any(text, OTHER_PLACES)


def _source_suffix(fact: dict) -> str:
    return (
        f"\n\nஆதாரம்: {fact['source_name']} — {fact['source_url']}"
        f" ({fact.get('as_of') or _snapshot_date or 'தேதி தெரியவில்லை'} நிலவரப்படி)"
    )


def _office_key(text: str) -> str | None:
    if _has_ascii_word(text, "pm"):
        return "pm_india"
    if _has_ascii_word(text, "cm"):
        return "cm_tamil_nadu"
    for key, words in OFFICE_KEYS.items():
        if _has_any(text, words):
            return key
    return None


def _weather_answer(text: str) -> dict | None:
    city = None
    for name, (lat, lon, label) in CITIES.items():
        if name in text:
            city = (lat, lon, label)
            break
    if city is None:
        return {
            "text": "எந்த ஊரின் வானிலை என்பதைக் குறிப்பிடவும். உதாரணம்: “சென்னையில் இப்போது மழை பெய்கிறதா?”",
            "kind": "clarify",
            "source_url": None,
        }
    lat, lon, label = city
    tomorrow = _has_any(text, TOMORROW_WORDS)
    if tomorrow:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
            "&forecast_days=2&timezone=Asia%2FKolkata"
        )
    else:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,precipitation,rain,weather_code"
            "&timezone=Asia%2FKolkata"
        )
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
        if tomorrow:
            daily = data["daily"]
            index = 1 if len(daily.get("time", [])) > 1 else 0
            high = daily["temperature_2m_max"][index]
            low = daily["temperature_2m_min"][index]
            rain_mm = float(daily.get("precipitation_sum", [0, 0])[index] or 0)
            code = int(daily.get("weather_code", [0, 0])[index] or 0)
            description = WEATHER_CODES.get(code, "வானிலை தகவல்")
            rain_text = (
                f"சுமார் {rain_mm:g} மி.மீ. மழை எதிர்பார்க்கப்படுகிறது."
                if rain_mm > 0
                else "மழை எதிர்பார்க்கப்படவில்லை."
            )
            return {
                "text": (
                    f"நாளை {label}யில் அதிகபட்சம் {high}°C, குறைந்தபட்சம் {low}°C, "
                    f"{description}. {rain_text} (முன்னறிவிப்பு: Open-Meteo)"
                ),
                "kind": "weather",
                "source_url": "https://open-meteo.com/",
            }
        current = data["current"]
        temperature = current.get("temperature_2m")
        code = int(current.get("weather_code", 0))
        rain = float(current.get("rain") or 0)
        precipitation = float(current.get("precipitation") or 0)
        description = WEATHER_CODES.get(code, "வானிலை தகவல்")
        raining = rain > 0 or precipitation > 0 or code in RAIN_CODES
        rain_text = "மழை பெய்கிறது" if raining else "மழை பெய்யவில்லை"
        observed = (current.get("time") or "").replace("T", " ")
        return {
            "text": (
                f"{label}யில் இப்போது {temperature}°C, {description}. {rain_text}. "
                f"(கண்காணிப்பு: {observed}, Open-Meteo)"
            ),
            "kind": "weather",
            "source_url": "https://open-meteo.com/",
        }
    except Exception:
        return {
            "text": f"{label}யின் நேரடி வானிலைத் தகவலை இப்போது பெற முடியவில்லை. {ABSTAIN}",
            "kind": "abstain",
            "source_url": "https://open-meteo.com/",
        }


def _person_key(text: str) -> str | None:
    if not _has_any(text, WHO_WORDS):
        return None
    for key, aliases in PERSON_ALIASES.items():
        if _has_any(text, aliases):
            return key
    return None


def answer(text: str) -> dict | None:
    """Return a grounded answer, a clarification, or None to let the model reply."""
    if not text or not text.strip():
        return None
    normalised = _normalise(text)

    if not _facts:
        _load()

    if _has_any(normalised, WEATHER_WORDS):
        return _weather_answer(normalised)

    if _has_any(normalised, PRICE_WORDS):
        return {
            "text": (
                "நேரடி சந்தை/பெட்ரோல் விலைகள் என்னிடம் நம்பகமான ஆதாரத்துடன் இல்லை. "
                "அதிகாரப்பூர்வ விலை இணையதளம் அல்லது சந்தை ஆதாரத்தில் சரிபார்க்கவும்."
            ),
            "kind": "abstain",
            "source_url": None,
        }

    if _has_any(normalised, NEWS_WORDS):
        return {
            "text": (
                "நேரடிச் செய்திகளை என்னிடம் நம்பகமான ஆதாரத்துடன் வழங்க முடியவில்லை. "
                "அதிகாரப்பூர்வ செய்தி ஆதாரத்தில் சரிபார்க்கவும்."
            ),
            "kind": "abstain",
            "source_url": None,
        }

    if _has_any(normalised, ["மக்கள் தொகை", "மக்கள்தொகை", "population"]):
        fact = _facts.get("india_population")
        if fact:
            return {
                "text": fact["answer_ta"] + _source_suffix(fact),
                "kind": "statistic",
                "source_url": fact["source_url"],
            }

    key = _office_key(normalised) or _person_key(normalised)
    if key:
        if _is_creative(normalised):
            return None
        if _mentions_other_place(normalised):
            return {
                "text": (
                    "இந்தியா மற்றும் தமிழ்நாடு பற்றிய சரிபார்க்கப்பட்ட தகவல்கள் மட்டுமே "
                    "என்னிடம் உள்ளன. இந்தக் கேள்விக்கான நம்பகமான ஆதாரம் இல்லை, எனவே ஊகிக்கமாட்டேன்."
                ),
                "kind": "abstain",
                "source_url": None,
            }
        fact = _facts.get(key)
        if fact:
            return {
                "text": fact["answer_ta"] + _source_suffix(fact),
                "kind": "officeholder",
                "source_url": fact["source_url"],
            }

    return None


def coverage() -> str:
    return (
        "சரிபார்க்கப்பட்ட தகவல் அடுக்கு: இந்தியப் பிரதமர், குடியரசுத் தலைவர், "
        "தமிழ்நாடு முதலமைச்சர், இராணுவத் தலைவர், மக்கள்தொகை மதிப்பீடு, சென்னை உள்ளிட்ட "
        "நகரங்களின் நேரடி வானிலை. மற்ற நேரடித் தகவல்களுக்கு மாடல் ஊகிக்காமல் மறுக்கும்."
    )


def _self_test() -> None:
    prompts = [
        "இந்தியப் பிரதமர் யார்?",
        "தமிழ்நாடு முதல்வர் யார்?",
        "இந்திய இராணுவத் தலைவர் யார்?",
        "இந்தியக் குடியரசுத் தலைவர் யார்?",
        "இந்தியாவின் மக்கள் தொகை எவ்வளவு?",
        "இப்போது சென்னையில் மழை பெய்கிறதா?",
        "இன்று தங்கத்தின் விலை என்ன?",
        "அமெரிக்க அதிபர் யார்?",
        "பிரதமர் பற்றி ஒரு கவிதை எழுது",
        "17 + 26 எவ்வளவு?",
    ]
    for prompt in prompts:
        result = answer(prompt)
        print(f"\nQ: {prompt}")
        print("A:", (result or {}).get("text", "<model handles this>"))


if __name__ == "__main__":
    _load()
    _self_test()
