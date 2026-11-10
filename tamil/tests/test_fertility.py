import json

from tamil.fertility import Tok, fertility, from_tokenizers, run_benchmark


def char_tok():
    """One token per code point: cuts every Tamil letter that has a vowel sign."""
    return Tok("chars", lambda s: (list(range(len(s))), [(i, i + 1) for i in range(len(s))]), None)


def word_tok():
    def enc(s):
        offs, i = [], 0
        for w in s.split(" "):
            offs.append((i, i + len(w)))
            i += len(w) + 1
        return list(range(len(offs))), offs
    return Tok("words", enc, None)


def test_fertility_counts():
    r = fertility(word_tok(), ["a b", "c d e"])
    assert r["tokens_per_word"] == 1.0
    assert r["bytes_per_token"] == (3 + 5) / 5


def test_cut_letters():
    assert fertility(char_tok(), ["தமிழ் மொழி"])["cut_letter_rate"] > 0
    assert fertility(word_tok(), ["தமிழ் மொழி"])["cut_letter_rate"] == 0


def test_roundtrip_exact():
    tok = Tok("lossy", lambda s: ([0], [(0, len(s))]), lambda ids: "x")
    assert fertility(tok, ["abc", "x"])["roundtrip_exact"] == 0.5


def test_ours_never_cuts_letters(trained_tok_path):
    ours = from_tokenizers("ours", trained_tok_path)
    texts = ["மாணவர்கள் தேர்வு எழுதினார்கள்.", "க்ஷேத்திரம் ஷாப்பிங் ஸ்ரீ", "தமிழ் is my மொழி 😀"]
    r = fertility(ours, texts)
    assert r["cut_letter_rate"] == 0
    assert r["roundtrip_exact"] == 1.0


def test_run_benchmark_writes_table(tmp_path):
    heldout = {"tamil": ["தமிழ் மொழி"], "english": ["the cat sat"]}
    run_benchmark([char_tok(), word_tok()], heldout, tmp_path / "f.md", tmp_path / "f.json")
    md = (tmp_path / "f.md").read_text(encoding="utf-8")
    assert "| chars |" in md and "| words |" in md and "tamil" in md
    data = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    assert data["words"]["english"]["tokens_per_word"] == 1.0
