import json

import pytest
import torch
from tokenizers import Tokenizer

from tamil.blocklist import build_blocklist_from_texts
from tamil.model import make_model
from tamil.sft import SFTConfig, build_sft_rows, encode_chat, sft, sft_loss

TURNS = [{"role": "user", "content": "வணக்கம்"}, {"role": "assistant", "content": "வணக்கம்! எப்படி இருக்கீங்க?"}]


@pytest.fixture(scope="module")
def tok(trained_tok_path):
    return Tokenizer.from_file(str(trained_tok_path))


def test_labels_mask_user(tok):
    ids, lab = encode_chat(tok, TURNS, ctx=1024)
    assert len(ids) == len(lab)
    a = ids.index(tok.token_to_id("<|assistant|>"))
    assert all(l == -100 for l in lab[: a + 1])  # BOS, user turn, assistant marker
    assert all(l == i for l, i in zip(lab[a + 1:], ids[a + 1:]))  # assistant text + </s>
    assert lab[-1] == tok.token_to_id("</s>")


def test_hf_chat_template_matches_training_ids(tok):
    from transformers import PreTrainedTokenizerFast

    from tamil.sft import CHAT_TEMPLATE
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    hf.chat_template = CHAT_TEMPLATE
    ids = hf.apply_chat_template(TURNS, tokenize=True, add_special_tokens=False)
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    assert list(ids) == encode_chat(tok, TURNS, ctx=1024)[0]


def test_template_roundtrip(tok):
    ids, _ = encode_chat(tok, TURNS, ctx=1024)
    assert "எப்படி இருக்கீங்க" in tok.decode(ids)


def test_build_rows_aya_tamil_only_and_blocklist():
    rows = {"aya": [{"inputs": "தமிழ் என்றால் என்ன?", "targets": "தமிழ் ஒரு மொழி.", "language": "Tamil"},
                    {"inputs": "What is Tamil?", "targets": "A language.", "language": "English"},
                    {"inputs": "அவன் பள்ளிக்குச் சென்றான் என்பதை விளக்கு", "targets": "சரி.", "language": "Tamil"}],
            "align": [{"interactions": [["கேள்வி ஒன்று", "பதில் ஒன்று"], ["கேள்வி இரண்டு", "பதில் இரண்டு"]]}]}
    bl = build_blocklist_from_texts(["அவன் பள்ளிக்குச் சென்றான் என்பதை விளக்கு"])
    out = build_sft_rows(lambda s: iter(rows["aya"] if "aya" in s.hf_id else
                                        [{s.text_field: r["interactions"]} for r in rows["align"]]), bl)
    texts = [t["content"] for conv in out for t in conv]
    assert "தமிழ் ஒரு மொழி." in texts and "What is Tamil?" not in texts
    assert not any("பள்ளிக்குச்" in t for t in texts)
    assert any(len(conv) == 4 for conv in out)  # multi-turn IndicAlign kept whole


def test_indicalign_real_shapes_per_language_column_and_mixed_language():
    """Real repo shape: Dolly_T/OpenAssistant_T/Wiki_Chat keep Tamil in column `tam_Taml` ([[user, assistant], ...]);
    Anudesh has one mixed-language `interactions` column (English/Hindi/Tamil rows, None answers possible)."""
    from tamil import sources
    pairs = [["கேள்வி ஒன்று", "பதில் ஒன்று"], ["கேள்வி இரண்டு", "பதில் இரண்டு"]]
    by_dir = {
        "dolly": [{"doc_id": "a", "num_turns": 1.0, "eng_Latn": [["Q", "A"]], "tam_Taml": pairs[:1]}],
        "oasst": [{"doc_id": "b", "num_turns": 2, "tam_Taml": pairs}],
        "anudesh": [{"id": "c", "interactions": [["hello", "Hi there"]], "num_turns": 1},
                    {"id": "d", "interactions": [["கேள்வி மூன்று", "பதில் மூன்று"], ["x", None]], "num_turns": 2}],
        "wiki_chat": [{"doc_id": "e", "tam_Taml": pairs[:1]}],
    }

    def load(src):
        if "aya" in src.hf_id:
            return iter([{"inputs": "தமிழ் என்றால் என்ன?", "targets": "தமிழ் ஒரு மொழி.", "language": "Tamil"}])
        return iter(by_dir[next(k for k in by_dir if f"/{k}/" in src.data_files)])

    out = build_sft_rows(load, set())
    assert len(out) == 1 + 1 + 1 + 1 + 1  # aya, dolly, oasst, anudesh (Tamil only), wiki_chat
    assert any(len(c) == 4 for c in out)  # oasst multi-turn kept whole
    assert not any(t["content"] in ("hello", "Hi there", "x", "None") for c in out for t in c)


def test_indicalign_sources_point_at_real_files():
    from tamil import sources
    ia = [s for s in sources.SFT if s.hf_id == "ai4bharat/indic-align"]
    assert 2 <= len(ia) <= 4
    assert all(s.data_files and s.data_files.startswith("indicalign-instruct/") for s in ia)
    assert all(s.text_field in ("tam_Taml", "interactions") for s in ia)


def test_build_rows_raises_when_a_source_yields_nothing():
    rows = {"aya": [{"inputs": "தமிழ் என்றால் என்ன?", "targets": "தமிழ் ஒரு மொழி.", "language": "Tamil"}],
            "align": [{"wrong_shape": 1}]}
    with pytest.raises(ValueError, match="indic-align"):
        build_sft_rows(lambda s: iter(rows["aya" if "aya" in s.hf_id else "align"]), set())


def test_sft_loss_ignores_masked():
    torch.manual_seed(0)
    m = make_model(64, "tiny")
    ids = torch.randint(0, 64, (1, 6))
    labels = torch.full_like(ids, -100)
    labels[0, 4] = ids[0, 4]
    logits = m(input_ids=ids).logits
    expected = torch.nn.functional.cross_entropy(logits[0, 3:4], ids[0, 4:5])
    assert torch.allclose(sft_loss(m, ids, labels), expected, atol=1e-5)


def test_sft_learns(tmp_path, trained_tok_path, tok):
    torch.manual_seed(0)
    base = tmp_path / "base.pt"
    torch.save({"model": make_model(tok.get_vocab_size(), "tiny").state_dict()}, base)
    data = tmp_path / "sft.jsonl"
    data.write_text("\n".join(json.dumps(TURNS, ensure_ascii=False) for _ in range(32)), encoding="utf-8")
    cfg = SFTConfig(base_ckpt=base, tokenizer_path=trained_tok_path, data_path=data, out_dir=tmp_path / "out",
                    model_size="tiny", lr=3e-3, min_lr=3e-4, warmup=2, epochs=8, micro_bs=4, grad_accum=1, ctx=64)
    sft(cfg)
    losses = [json.loads(l)["loss"] for l in (tmp_path / "out" / "log.jsonl").read_text().splitlines()]
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 * 0.5
    assert torch.load(tmp_path / "out" / "ckpt.pt", weights_only=True)["step"] == len(losses)


def test_sft_resume_from_previous_session(tmp_path, trained_tok_path, tok):
    torch.manual_seed(0)
    base = tmp_path / "base.pt"
    torch.save({"model": make_model(tok.get_vocab_size(), "tiny").state_dict()}, base)
    data = tmp_path / "sft.jsonl"
    data.write_text("\n".join(json.dumps(TURNS, ensure_ascii=False) for _ in range(16)), encoding="utf-8")
    kw = dict(base_ckpt=base, tokenizer_path=trained_tok_path, data_path=data, model_size="tiny",
              warmup=1, epochs=4, micro_bs=4, grad_accum=1, ctx=64)
    sft(SFTConfig(out_dir=tmp_path / "s1", stop_after=6, **kw))
    sft(SFTConfig(out_dir=tmp_path / "s2", resume_from=tmp_path / "s1" / "ckpt.pt", **kw))
    assert torch.load(tmp_path / "s2" / "ckpt.pt", weights_only=True)["step"] == 16
    assert len((tmp_path / "s2" / "log.jsonl").read_text().splitlines()) == 10
