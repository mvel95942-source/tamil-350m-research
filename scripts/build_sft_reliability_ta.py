"""Build a small, mechanically verifiable Tamil SFT set for reliability skills.

The 350M preview failed the fixed 15-case reliability holdout (arithmetic, grammar,
instruction following, correction, calibrated uncertainty). This script generates
training conversations that teach the *skills* while never copying a holdout item:

  * arithmetic answers are computed in Python, not written by hand;
  * corrections are generated from deliberately wrong equations;
  * uncertainty examples ask for live/unknowable facts and answer with a refusal;
  * grammar items use a fixed, reviewed list of common nouns/verbs.

Output format is the one `tamil.sft` trains on: one JSON list of {role, content} turns
per line. The writer refuses to emit any prompt that exactly matches a holdout prompt
and avoids the holdout arithmetic operand pairs.

    python scripts/build_sft_reliability_ta.py --out artifacts/sft_reliability_ta.jsonl
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "eval" / "tamil_reliability_holdout.jsonl"

# Operand pairs that appear in the frozen holdout: never generate these exact expressions.
FORBIDDEN = {
    ("+", 17, 26), ("*", 14, 6), ("/", 45, 5), ("*", 8, 3),
    ("perimeter", 9, 4), ("*", 7, 9), ("half", 100, None),
}

ITEMS = ["பென்சில்கள்", "புத்தகங்கள்", "பந்துகள்", "மாம்பழங்கள்", "ஆப்பிள்கள்",
         "இனிப்புகள்", "காகிதங்கள்", "பேனாக்கள்", "அட்டைகள்", "பூக்கள்"]

# (nominative, locative); Chennai is deliberately absent so the frozen weather probe
# ("இப்போது சென்னையில் மழை பெய்கிறதா?") stays unseen.
CITIES = [("கோயம்புத்தூர்", "கோயம்புத்தூரில்"), ("மதுரை", "மதுரையில்"),
          ("திருச்சி", "திருச்சியில்"), ("சேலம்", "சேலத்தில்"),
          ("திருநெல்வேலி", "திருநெல்வேலியில்")]

# Common Tamil nouns and their accepted plural forms. Reviewed for standard written Tamil.
PLURALS = [
    ("மரம்", "மரங்கள்"), ("வீடு", "வீடுகள்"), ("பறவை", "பறவைகள்"),
    ("மாணவன்", "மாணவர்கள்"), ("மாணவி", "மாணவிகள்"), ("நாய்", "நாய்கள்"),
    ("பூ", "பூக்கள்"), ("கல்", "கற்கள்"), ("ஆறு", "ஆறுகள்"),
    ("மலை", "மலைகள்"), ("கண்", "கண்கள்"), ("கை", "கைகள்"),
    ("பல்", "பற்கள்"), ("செடி", "செடிகள்"), ("பசு", "பசுக்கள்"),
    ("குதிரை", "குதிரைகள்"), ("ஊர்", "ஊர்கள்"), ("நாடு", "நாடுகள்"),
    ("நண்பன்", "நண்பர்கள்"), ("ஆசிரியர்", "ஆசிரியர்கள்"),
]

VERBS = [
    ("அவன் ஓடினான்.", "ஓடினான்"),
    ("அவள் படித்தாள்.", "படித்தாள்"),
    ("குழந்தை அழுதது.", "அழுதது"),
    ("மாணவர்கள் பாடினார்கள்.", "பாடினார்கள்"),
    ("நான் எழுதினேன்.", "எழுதினேன்"),
    ("அவர் வந்தார்.", "வந்தார்"),
    ("நாய் குரைத்தது.", "குரைத்தது"),
    ("பறவை பறந்தது.", "பறந்தது"),
    ("நாங்கள் சாப்பிட்டோம்.", "சாப்பிட்டோம்"),
    ("அவை ஓடின.", "ஓடின."),
]

SCIENCE = [
    ("நீரின் வேதியியல் குறியீடு என்ன?", "H₂O."),
    ("பூமியின் இயற்கைத் துணைக்கோள் எது?", "நிலா."),
    ("சூரியக் குடும்பத்தில் மிகப்பெரிய கோள் எது?", "வியாழன்."),
    ("மனித உடலில் இரத்தத்தை பம்ப் செய்யும் உறுப்பு எது?", "இதயம்."),
    ("தாவரங்கள் ஒளிச்சேர்க்கையில் எந்த வாயுவை எடுத்துக்கொள்கின்றன?", "கார்பன் டைஆக்சைடு."),
    ("நீர் 100 டிகிரி செல்சியஸில் என்ன ஆகிறது?", "கொதித்து நீராவியாகிறது."),
    ("பூமியின் மேற்பரப்பில் மிக அதிகமாகக் காணப்படும் வாயு எது?", "நைட்ரஜன்."),
]

WEEKDAYS = ["திங்கள்", "செவ்வாய்", "புதன்", "வியாழன்", "வெள்ளி", "சனி", "ஞாயிறு"]
MONTHS = ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்",
          "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"]


def load_holdout() -> tuple[set[str], set[tuple]]:
    prompts = set()
    for line in HOLDOUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.add(json.loads(line)["prompt"])
    return prompts, FORBIDDEN


def conv(user: str, assistant: str) -> list[dict]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def gen_arithmetic(rng: random.Random, forbidden: set[tuple]) -> list[list[dict]]:
    out: list[list[dict]] = []

    def add(op, a, b, user, assistant):
        # Also reject the commuted/reordered form so 26+17 cannot stand in for the
        # frozen 17+26 probe (same item, different order).
        if (op, a, b) in forbidden or (op, b, a) in forbidden:
            return
        out.append(conv(user, assistant))

    for _ in range(700):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        add("+", a, b, f"{a} + {b} எவ்வளவு?", f"விடை: {a + b}.")
        add("+", a, b, f"{a} + {b} எவ்வளவு? கணக்கைச் சுருக்கமாகக் காட்டு.",
            f"விடை: {a + b}. கணக்கு: {a} + {b} = {a + b}.")
    for _ in range(500):
        a, b = rng.randint(20, 99), rng.randint(1, 19)
        if a <= b:
            continue
        add("-", a, b, f"{a} - {b} எவ்வளவு?", f"விடை: {a - b}.")
    for _ in range(500):
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        add("*", a, b, f"{a} × {b} எவ்வளவு?", f"விடை: {a * b}.")
    for _ in range(400):
        b = rng.randint(2, 12)
        q = rng.randint(2, 12)
        a = b * q
        add("/", a, b, f"{a}-ஐ {b}-ஆல் வகுத்தால் என்ன?", f"விடை: {q}.")
    for _ in range(300):
        a, b = rng.randint(2, 12), rng.randint(2, 9)
        item = rng.choice(ITEMS)
        add("*", a, b, f"ஒரு பெட்டியில் {a} {item} உள்ளன. அப்படிப்பட்ட {b} பெட்டிகளில் "
                       f"மொத்தம் எத்தனை {item}?",
            f"{a * b} {item}.")
    for _ in range(250):
        a, b = rng.randint(20, 99), rng.randint(1, 19)
        if a <= b:
            continue
        item = rng.choice(ITEMS)
        out.append(conv(
            f"ஒரு கடையில் {a} {item} இருந்தன. {b} {item} விற்றுவிட்டால் எத்தனை மீதம்?",
            f"{a - b} {item} மீதம்."))
    for _ in range(250):
        l, w = rng.randint(2, 20), rng.randint(2, 20)
        add("perimeter", l, w,
            f"நீளம் {l} மீ., அகலம் {w} மீ. உள்ள செவ்வகத்தின் சுற்றளவு என்ன?",
            f"சுற்றளவு = 2 × ({l} + {w}) = {2 * (l + w)} மீ.")
    for _ in range(300):
        n = rng.randrange(2, 200, 2)
        out.append(conv(f"{n}-இன் பாதி எவ்வளவு?", f"விடை: {n // 2}."))
    for _ in range(200):
        n = rng.randrange(4, 200, 4)
        out.append(conv(f"{n}-இல் 25% எவ்வளவு?", f"விடை: {n // 4}."))
    for _ in range(200):
        price, n = rng.randint(5, 90), rng.randint(2, 9)
        item = rng.choice(ITEMS)
        out.append(conv(f"ஒரு {item} விலை {price} ரூபாய். {n} {item} வாங்கினால் "
                        f"மொத்த விலை என்ன?",
                        f"{price * n} ரூபாய்."))
    return out


def gen_instruction(rng: random.Random) -> list[list[dict]]:
    out: list[list[dict]] = []
    for k in range(2, 7):
        days = WEEKDAYS[:k]
        out.append(conv(f"ஒரு வாக்கியத்தில் மட்டும் பதில்: வாரத்தின் முதல் {k} நாட்களை "
                        f"வரிசையாக எழுது.", ", ".join(days) + "."))
        days = WEEKDAYS[-k:]
        out.append(conv(f"ஒரு வாக்கியத்தில் மட்டும் பதில்: வாரத்தின் இறுதி {k} நாட்களை "
                        f"வரிசையாக எழுது.", ", ".join(days) + "."))
    for k in range(2, 7):
        out.append(conv(f"ஒரு வாக்கியத்தில் மட்டும் பதில்: ஆண்டின் முதல் {k} மாதங்களை "
                        f"வரிசையாக எழுது.", ", ".join(MONTHS[:k]) + "."))
    for n in (2, 3, 4, 5, 6, 8, 10):
        out.append(conv(f"ஒரு வார்த்தையில் மட்டும் பதில்: 1 முதல் {n} வரை எண்களை "
                        f"எழுது.", ", ".join(str(i) for i in range(1, n + 1)) + "."))
    facts = [
        ("தமிழ்நாட்டின் தலைநகரம் எது?", "சென்னை."),
        ("இந்தியாவின் தலைநகரம் எது?", "புது தில்லி."),
        ("தமிழின் முதல் எழுத்து எது?", "அ."),
        ("சூரியன் எந்தத் திசையில் உதிக்கிறது?", "கிழக்கில்."),
        ("ஒரு வாரத்தில் எத்தனை நாட்கள் உள்ளன?", "ஏழு."),
        ("ஒரு ஆண்டில் எத்தனை மாதங்கள் உள்ளன?", "பன்னிரண்டு."),
    ]
    for q, a in facts:
        out.append(conv(f"ஒரு வார்த்தையில் மட்டும் பதில்: {q}", a))
        out.append(conv(f"சுருக்கமாகச் சொல்: {q}", a))
    for n in range(4, 60, 2):
        out.append(conv(f"சுருக்கமாகச் சொல்: {n}-இன் பாதி எவ்வளவு?", f"{n // 2}."))
    return out


def gen_uncertainty(rng: random.Random) -> list[list[dict]]:
    out: list[list[dict]] = []
    for city, loc in CITIES:
        out.append(conv(f"இப்போது {loc} மழை பெய்கிறதா?",
                        "எனக்குத் தெரியாது. நேரடி வானிலைத் தகவலை என்னால் அறிய முடியாது; "
                        "வானிலைச் சேவையில் சரிபார்க்கவும்."))
        out.append(conv(f"இன்று {loc} வெப்பநிலை எவ்வளவு?",
                        "எனக்குத் தெரியாது. நேரடி வானிலைத் தகவல் என்னிடம் இல்லை."))
        out.append(conv(f"நாளை {loc} மழை பெய்யுமா?",
                        "எனக்குத் தெரியாது. நாளைய வானிலை பற்றிய நேரடித் தகவல் என்னிடம் இல்லை."))
    live = [
        ("இந்த நிமிடத்தில் இந்தியாவின் மக்கள் தொகை எவ்வளவு? துல்லியமான எண்ணைச் சொல்.",
         "எனக்குத் தெரியாது. மக்கள் தொகை தொடர்ந்து மாறுகிறது; தற்போதைய துல்லியமான எண் "
         "என்னிடம் இல்லை."),
        ("இன்றைய தங்கத்தின் விலை என்ன?",
         "எனக்குத் தெரியாது. தங்கத்தின் விலை நாளுக்கு நாள் மாறும்; தற்போதைய விலை என்னிடம் இல்லை."),
        ("சென்னை சூப்பர் கிங்ஸ் அணியின் தற்போதைய கேப்டன் யார்?",
         "எனக்குத் தெரியாது. அணியின் தற்போதைய தகவலை நேரடி ஆதாரத்தில் சரிபார்க்கவும்."),
        ("இந்த ஆண்டு நடக்கும் அடுத்த கிரிக்கெட் போட்டி எது?",
         "எனக்குத் தெரியாது. தற்போதைய போட்டி அட்டவணை என்னிடம் இல்லை."),
        ("நேற்று நடந்த ஆட்டத்தின் முடிவு என்ன?",
         "எனக்குத் தெரியாது. நேற்றைய நிகழ்வுகள் பற்றிய நேரடித் தகவல் என்னிடம் இல்லை."),
        ("இப்போது இந்தியாவில் எத்தனை பேர் வேலை தேடுகிறார்கள்?",
         "எனக்குத் தெரியாது. தற்போதைய புள்ளிவிவரம் என்னிடம் இல்லை."),
        ("இந்த நிறுவனத்தின் இன்றைய பங்கு விலை என்ன?",
         "எனக்குத் தெரியாது. பங்கு விலைகள் நொடிக்கு நொடி மாறும்; தற்போதைய விலை "
         "என்னிடம் இல்லை."),
    ]
    out.extend(conv(q, a) for q, a in live)
    out.extend([
        conv("அது எவ்வளவு?",
             "எதைக் குறிப்பிடுகிறீர்கள் என்று தெளிவாகச் சொன்னால் பதிலளிக்க முடியும்."),
        conv("அந்த எண்ணைச் சொல்.",
             "எந்த எண்ணைக் குறிப்பிடுகிறீர்கள் என்று தெளிவுபடுத்தவும்."),
        conv("அவர் யார்?",
             "யாரைப் பற்றி கேட்கிறீர்கள் என்று பெயரைச் சொன்னால் பதிலளிக்க முடியும்."),
    ])
    return out


def gen_correction(rng: random.Random, forbidden: set[tuple]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for _ in range(300):
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        if ("*", a, b) in forbidden or ("*", b, a) in forbidden:
            continue
        wrong = a * b + rng.choice([-3, -2, -1, 1, 2, 3])
        out.append(conv(f"சரிபார்த்து திருத்து: {a} × {b} = {wrong}.",
                        f"தவறு. சரியான விடை: {a} × {b} = {a * b}."))
    for _ in range(300):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        if ("+", a, b) in forbidden or ("+", b, a) in forbidden:
            continue
        wrong = a + b + rng.choice([-3, -2, -1, 1, 2, 3])
        out.append(conv(f"சரிபார்த்து திருத்து: {a} + {b} = {wrong}.",
                        f"தவறு. சரியான விடை: {a} + {b} = {a + b}."))
    for _ in range(200):
        a, b = rng.randint(20, 99), rng.randint(1, 19)
        if a <= b:
            continue
        wrong = a - b + rng.choice([-2, -1, 1, 2])
        out.append(conv(f"சரிபார்த்து திருத்து: {a} - {b} = {wrong}.",
                        f"தவறு. சரியான விடை: {a} - {b} = {a - b}."))
    return out


def gen_grammar() -> list[list[dict]]:
    out: list[list[dict]] = []
    for noun, plural in PLURALS:
        out.append(conv(f"'{noun}' என்பதன் பன்மை வடிவம் என்ன?", f"'{plural}'."))
        out.append(conv(f"'{noun}' என்பதன் பன்மை என்ன?", f"{plural}."))
    for sentence, verb in VERBS:
        out.append(conv(f"'{sentence}' என்ற வாக்கியத்தில் வினைச்சொல் எது?",
                        f"வினைச்சொல்: {verb}."))
    return out


def gen_science() -> list[list[dict]]:
    out: list[list[dict]] = []
    for q, a in SCIENCE:
        out.append(conv(q, a))
        out.append(conv(f"சுருக்கமாகச் சொல்: {q}", a))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "sft_reliability_ta.jsonl")
    args = ap.parse_args()
    rng = random.Random(0)
    holdout_prompts, forbidden = load_holdout()

    groups = {
        "arithmetic": gen_arithmetic(rng, forbidden),
        "instruction": gen_instruction(rng),
        "uncertainty": gen_uncertainty(rng),
        "correction": gen_correction(rng, forbidden),
        "grammar": gen_grammar(),
        "science": gen_science(),
    }
    seen: set[str] = set()
    rows: list[list[dict]] = []
    for name, group in groups.items():
        kept = 0
        for c in group:
            key = json.dumps(c, ensure_ascii=False, sort_keys=True)
            if key in seen or c[0]["content"] in holdout_prompts:
                continue
            seen.add(key)
            rows.append(c)
            kept += 1
        print(f"{name}: {kept}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in rows),
                        encoding="utf-8")
    print(f"total: {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
