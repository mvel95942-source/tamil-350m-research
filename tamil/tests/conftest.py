from random import Random

import pytest

from tamil.tanglish import romanize

TA_WORDS = ("நான் நீ அவன் அவள் நாங்கள் அவர்கள் பள்ளிக்குச் சென்றான் வீட்டுக்கு வந்தாள் தமிழ் மொழி "
            "அழகான பெரிய சிறிய புத்தகம் படித்தேன் எழுதினார் சாப்பிட்டோம் தண்ணீர் குடித்தார்கள் மழை "
            "பெய்தது காலை மாலை இரவு நண்பன் அம்மா அப்பா தம்பி அக்கா ஊருக்குப் போனோம் கடையில் வாங்கினேன் "
            "அரசு மக்கள் நாடு உலகம் அறிவியல் கணினி செய்தி இன்று நேற்று நாளை வேலை செய்கிறேன் பேசுகிறார் "
            "கேட்டான் சொன்னாள் பார்த்தோம் நடந்தது ஓடினான் விளையாடினார்கள் பாடல் இசை திரைப்படம் சென்னை "
            "மதுரை கோயில் ஆறு கடல் மலை மரம் பூ பழம் காற்று நன்றி வணக்கம் எப்படி இருக்கீங்க ஸ்ரீ ஜன்னல் "
            "ஷாப்பிங் ஹோட்டல் க்ஷேத்திரம் ஃபோன் ஆசிரியர் மாணவர்கள் தேர்வு எழுதினார்கள் வெற்றி பெற்றனர்").split()
EN_WORDS = ("the a of and to in is was for on with as by this that from people school went home "
            "language beautiful book read write water rain morning evening friend mother father city "
            "government science computer news today tomorrow work speak music film temple sea tree").split()


@pytest.fixture(scope="session")
def trained_tok_path(mini_corpus, tmp_path_factory):
    from tamil.tokenizer_train import train_tokenizer
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_tokenizer(mini_corpus, 2000, path)
    return path


@pytest.fixture(scope="session")
def packed_bin(mini_corpus, trained_tok_path, tmp_path_factory):
    from tamil.corpus import write_shards
    from tamil.pack import pack
    d = tmp_path_factory.mktemp("packed")
    langs = ["tamil", "english", "tanglish"]
    shards = write_shards(((langs[i % 3], t) for i, t in enumerate(mini_corpus)), d / "shards", 2000)
    pack(shards, trained_tok_path, d / "train.bin", eos_id=2)
    return d / "train.bin"


@pytest.fixture(scope="session")
def mini_corpus() -> list[str]:
    rng = Random(0)
    lines = []
    for _ in range(3000):
        ta = " ".join(rng.choice(TA_WORDS) for _ in range(rng.randint(5, 12))) + "."
        lines.append(ta)
        lines.append(" ".join(rng.choice(EN_WORDS) for _ in range(rng.randint(5, 12))) + ".")
        lines.append(romanize(ta, rng.choice(["iso", "casual", "chat"]), rng, 0.3))
    return lines
