import math
from types import SimpleNamespace

import pytest
import torch

from tamil.evaluate import bits_per_byte, loglik_choice

V = 50


def enc(s):
    return [ord(c) % V for c in s]


class Uniform(torch.nn.Module):
    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, V))


class Prefers(torch.nn.Module):
    def __init__(self, ids):
        super().__init__()
        self.ids = list(ids)

    def forward(self, input_ids):
        logits = torch.zeros(*input_ids.shape, V)
        logits[..., self.ids] = 5.0
        return SimpleNamespace(logits=logits)


def test_bpb_uniform():
    assert bits_per_byte(Uniform(), enc, ["hello world", "abc"], bos_id=0, ctx=8) == pytest.approx(math.log2(V))


def test_real_model_task_paths(trained_tok_path):
    from tokenizers import Tokenizer

    from tamil.evaluate import _mc_accuracy, _translit_accuracy, _xnli
    from tamil.model import make_model
    tok = Tokenizer.from_file(str(trained_tok_path))
    m = make_model(tok.get_vocab_size(), "tiny").eval()

    def e(s):
        return tok.encode(s, add_special_tokens=False).ids
    xnli = [{"premise": "அவன் பள்ளிக்குச் சென்றான்.", "hypothesis": "அவன் வீட்டில் இருக்கிறான்", "label": 2}] * 8
    acc = _mc_accuracy(m, e, xnli, _xnli, bos_id=1)
    assert 0.0 <= acc <= 1.0
    words = [{"english word": "amma", "native word": "அம்மா"}] * 7
    assert 0.0 <= _translit_accuracy(m, e, tok.decode, words, bos_id=1, eos_id=2) <= 1.0
    assert math.isfinite(bits_per_byte(m, e, ["தமிழ் மொழி"], bos_id=1))


def test_shots_dropped_to_fit_context():
    from tamil.evaluate import fit_shots
    items = [("p" * 30, [" a", " bb"], 0)] * 7  # each solved shot ~33 chars = 33 tokens with enc()
    k = fit_shots(enc, items[:5], items[5][0], items[5][1], ctx=100)
    assert k == 1  # 1 shot (33) + prompt (30) + longest choice (3) <= 100, 2 shots would not fit
    assert fit_shots(enc, items[:5], items[5][0], items[5][1], ctx=10_000) == 5


def test_doc_sep_id(trained_tok_path):
    from tokenizers import Tokenizer

    from tamil.evaluate import doc_sep_id
    tok = Tokenizer.from_file(str(trained_tok_path))
    assert doc_sep_id(tok) == tok.token_to_id("</s>") == 2


def test_loglik_choice_picks_likely():
    assert loglik_choice(Prefers(enc("ஆம்")), enc, "x", ["இல்லை", "ஆம்"], bos_id=0) == 1
    assert loglik_choice(Prefers(enc("no")), enc, "x", ["no", "yes"], bos_id=0) == 0
