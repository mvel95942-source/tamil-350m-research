from random import Random

from tamil.tanglish import romanize


def test_iso_vanakkam():
    assert romanize("வணக்கம்", "iso", Random(0)) == "vanakkam"


def test_tamil_word():
    assert romanize("தமிழ்", "iso", Random(0)) == "thamizh"


def test_passthrough():
    out = romanize("I love தமிழ் 2024!", "iso", Random(0))
    assert out.startswith("I love ") and out.endswith(" 2024!")


def test_ascii_only():
    assert romanize("தமிழ்நாடு", "chat", Random(1), noise=0.5).isascii()


def test_noise_varies():
    assert len({romanize("தமிழ்நாடு", "casual", Random(i), noise=0.5) for i in range(20)}) >= 3


def test_seeded():
    assert romanize("அம்மா", "casual", Random(7), 0.3) == romanize("அம்மா", "casual", Random(7), 0.3)


def test_geminates():
    assert romanize("பட்டம்", "iso", Random(0)) == "pattam"
    assert romanize("காற்று", "iso", Random(0)) == "kaatru"


def test_ndr_cluster():
    assert romanize("நன்றி", "iso", Random(0)) == "nandri"
