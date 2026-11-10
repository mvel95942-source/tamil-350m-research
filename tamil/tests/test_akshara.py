import unicodedata

from tamil.akshara import AKSHARA_TABLE, from_pua, split_aksharas, to_pua


def test_split_basic():
    assert split_aksharas("தமிழ்") == ["த", "மி", "ழ்"]


def test_split_grantha():
    assert split_aksharas("ஸ்ரீ") == ["ஸ்", "ரீ"]


def test_nfd_composes():
    assert to_pua("கொ") == to_pua(unicodedata.normalize("NFD", "கொ"))


def test_roundtrip_mixed():
    for s in ["phone-ல பேசு", "க்ஷேத்திரம் ௧௨௩ ௐ", "நான்‍வர", "hello 😀"]:
        assert from_pua(to_pua(s)) == unicodedata.normalize("NFC", s)


def test_one_char_per_akshara():
    assert len(to_pua("தமிழ்")) == 3


def test_table_size():
    assert 300 <= len(AKSHARA_TABLE) <= 400
