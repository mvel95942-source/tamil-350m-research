import numpy as np
import torch

from tamil.corpus import write_shards
from tamil.model import make_model
from tamil.pack import PackedDataset, pack

DOCS = [("tamil", "தமிழ் ஒரு அழகான மொழி."), ("english", "the cat sat on the mat."),
        ("tanglish", "thamizh oru azhagaana mozhi.")] * 40


def _packed(tmp_path, trained_tok_path):
    shards = write_shards(iter(DOCS), tmp_path / "shards", docs_per_shard=50)
    out = tmp_path / "train.bin"
    n = pack(shards, trained_tok_path, out, eos_id=2)
    return out, n


def test_pack_roundtrip(tmp_path, trained_tok_path):
    from tokenizers import Tokenizer
    out, n = _packed(tmp_path, trained_tok_path)
    arr = np.memmap(out, dtype=np.uint16, mode="r")
    assert n == arr.size
    first = arr[: list(arr).index(2)].tolist()
    assert Tokenizer.from_file(str(trained_tok_path)).decode(first) == DOCS[0][1]
    langs = np.load(out.with_suffix(".lang.npy"))
    assert langs.shape[0] == len(DOCS)


def test_pack_is_atomic(tmp_path, trained_tok_path):
    import pytest
    shards = write_shards(iter(DOCS), tmp_path / "shards", docs_per_shard=50)
    broken = tmp_path / "shards" / "shard_99999.jsonl.gz"
    broken.write_bytes(b"not gzip")  # a shard truncated by a killed session
    out = tmp_path / "train.bin"
    with pytest.raises(Exception):
        pack(shards + [broken], trained_tok_path, out, eos_id=2)
    assert not out.exists()  # a half-written .bin must never look finished


def test_uint32_mode_for_big_vocabs(tmp_path, trained_tok_path):
    shards = write_shards(iter(DOCS), tmp_path / "shards", docs_per_shard=50)
    out = tmp_path / "big.bin"
    n = pack(shards, trained_tok_path, out, eos_id=2, dtype="uint32")
    assert out.stat().st_size == n * 4
    ds = PackedDataset(out, ctx=8, rank=0, world=1)
    small = PackedDataset(_packed(tmp_path / "u16", trained_tok_path)[0], ctx=8, rank=0, world=1)
    assert torch.equal(ds.batch(0, 2), small.batch(0, 2))


def test_batch_deterministic(tmp_path, trained_tok_path):
    out, _ = _packed(tmp_path, trained_tok_path)
    ds = PackedDataset(out, ctx=8, rank=0, world=1)
    assert ds.batch(5, 2).shape == (2, 9)
    assert torch.equal(ds.batch(5, 2), ds.batch(5, 2))
    assert torch.equal(PackedDataset(out, ctx=8, rank=0, world=1).batch(5, 2), ds.batch(5, 2))


def test_ranks_disjoint(tmp_path, trained_tok_path):
    out, _ = _packed(tmp_path, trained_tok_path)
    r0 = PackedDataset(out, ctx=8, rank=0, world=2).batch(3, 2)
    r1 = PackedDataset(out, ctx=8, rank=1, world=2).batch(3, 2)
    assert not torch.equal(r0, r1)


def test_param_count():
    with torch.device("meta"):
        n = sum(p.numel() for p in make_model(48000).parameters())
    assert 330e6 < n < 390e6, n


def test_tiny_forward():
    m = make_model(512, "tiny")
    out = m(torch.randint(0, 512, (2, 16)), labels=torch.randint(0, 512, (2, 16)))
    assert out.loss.isfinite()


def test_attention_without_gqa_flag_matches_hf_sdpa():
    import torch

    from tamil.model import make_model
    torch.manual_seed(0)
    ours = make_model(64, "tiny")
    ref = make_model(64, "tiny")
    ref.load_state_dict(ours.state_dict())
    ref.config._attn_implementation = "sdpa"
    assert ours.config._attn_implementation == "sdpa_expand_kv"
    x = torch.randint(0, 64, (2, 12))
    assert torch.allclose(ours(input_ids=x).logits, ref(input_ids=x).logits, atol=1e-5)
