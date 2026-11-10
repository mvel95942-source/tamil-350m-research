import unicodedata

import pytest

from tamil.akshara import PULLI, VOWEL_SIGNS
from tamil.tokenizer_train import BYTE_TOKENS, SPECIAL_TOKENS

VOCAB = 2000


@pytest.fixture(scope="module")
def tok(trained_tok_path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(trained_tok_path))


def test_roundtrip(tok):
    for s in ["க்ஷேத்திரம் ௧௨௩ ௐ", "phone-ல பேசு", "vanakkam da 😀", unicodedata.normalize("NFD", "கொடு"),
              "தமிழ் is beautiful, இல்லையா?"]:
        assert tok.decode(tok.encode(s).ids) == unicodedata.normalize("NFC", s)


def test_no_akshara_split(tok):
    dependent = tuple(VOWEL_SIGNS + [PULLI])
    for piece in tok.get_vocab():
        assert not piece.lstrip("▁").startswith(dependent), piece


def test_byte_fallback(tok):
    assert "<unk>" not in tok.encode("𝔘𝔫𝔦 😀").tokens


def test_vocab_size(tok):
    # the tiny fixture can't fill 2000 pieces; real corpora hit the target exactly (checked on Kaggle)
    vocab = tok.get_vocab()
    assert 1000 <= len(vocab) <= VOCAB
    assert all(b in vocab for b in BYTE_TOKENS)


def test_multiple_spaces_and_newlines(tok):
    s = "வணக்கம்  நண்பா\nhow are you"
    assert tok.decode(tok.encode(s).ids) == s


def test_special_tokens_present(tok):
    for t in SPECIAL_TOKENS:
        assert tok.token_to_id(t) is not None


def test_saved_file_reloads(tok, tmp_path):
    from tokenizers import Tokenizer
    p = tmp_path / "t.json"
    tok.save(str(p))
    again = Tokenizer.from_file(str(p))
    s = "நன்றி, see you நாளை"
    assert again.decode(again.encode(s).ids) == s


CODE = 'def add(a, b):\n    if a > 0:\n        return {"x": [a, b]}\n\treturn a + b\n'


def test_code_roundtrips_exactly(tok):
    for s in [CODE, "    leading indent", " one leading space", "<div class=\"a\">\n  <p>hi</p>\n</div>",
              "int main() {\n    return 0;\n}\n"]:
        assert tok.decode(tok.encode(s).ids) == s


def test_indentation_runs_are_single_tokens(tok):
    from tamil.tokenizer_train import WHITESPACE_TOKENS
    for ws in WHITESPACE_TOKENS:
        assert tok.token_to_id(ws) is not None, repr(ws)
    assert "    " in tok.encode("x\n    y").tokens


def test_chat_and_tool_tokens_are_one_id_each(tok):
    from tamil.tokenizer_train import CHAT_TOOL_TOKENS
    for t in CHAT_TOOL_TOKENS:
        ids = tok.encode(f"a {t} b").ids
        assert tok.token_to_id(t) in ids, t


def test_v1_special_token_ids_unchanged(tok):
    # sft.py, pack.py (eos_id=2) and the running 350M job rely on these ids
    assert [tok.token_to_id(t) for t in ["<pad>", "<s>", "</s>", "<unk>", "<|user|>", "<|assistant|>"]] == list(range(6))
