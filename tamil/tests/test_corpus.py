import gzip
import json

from tamil.blocklist import build_blocklist_from_texts, norm_hash
from tamil.corpus import clean, iter_mixed, write_shards
from tamil.sources import Source

TA_DOC = "இது ஒரு நீண்ட தமிழ் ஆவணம், இதில் பல சொற்கள் உள்ளன. மேலும் ஒரு வாக்கியம் இங்கே உள்ளது. "
EN_DOC = "This is a long English document with plenty of words in it for the cleaning filter. "

TA = Source("fake/ta", None, "train", "text", "tamil", "cc-by")
EN = Source("fake/en", None, "train", "text", "english", "cc-by")
MIX = {"tamil": 0.55, "tanglish": 0.20, "english": 0.25}


def fake_loader(docs_by_id):
    return lambda src: iter(docs_by_id[src.hf_id])


def unique(base, n):
    return [f"{base}{i} " * 3 + base for i in range(n)]


def test_blocklisted_sentence_dropped():
    bl = build_blocklist_from_texts(["அவன் பள்ளிக்குச் சென்றான்."])
    leaked = TA_DOC + "அவன் பள்ளிக்குச் சென்றான். " + TA_DOC
    load = fake_loader({"fake/ta": [leaked] + unique(TA_DOC, 50), "fake/en": unique(EN_DOC, 50)})
    docs = list(iter_mixed([TA, EN], MIX, bl, seed=0, load_fn=load))
    assert docs
    assert all("பள்ளிக்குச் சென்றான்" not in t for _, t in docs)


def test_mix_ratio_within_5pct():
    load = fake_loader({"fake/ta": unique(TA_DOC, 2000), "fake/en": unique(EN_DOC, 2000)})
    docs = list(iter_mixed([TA, EN], MIX, set(), seed=0, load_fn=load))
    chars = {k: 0 for k in MIX}
    for lang, t in docs:
        chars[lang] += len(t)
    total = sum(chars.values())
    for lang, share in MIX.items():
        assert abs(chars[lang] / total - share) <= 0.05, (lang, chars[lang] / total)


def test_tanglish_is_ascii_romanized():
    load = fake_loader({"fake/ta": unique(TA_DOC, 200), "fake/en": unique(EN_DOC, 200)})
    tg = [t for lang, t in iter_mixed([TA, EN], MIX, set(), seed=0, load_fn=load) if lang == "tanglish"]
    assert tg and all(not any("஀" <= c <= "௿" for c in t) for t in tg)


def test_dedupe():
    load = fake_loader({"fake/ta": [TA_DOC * 2] * 5, "fake/en": [EN_DOC * 2] * 5})
    docs = list(iter_mixed([TA, EN], {"tamil": 0.5, "english": 0.5}, set(), seed=0, load_fn=load))
    assert len(docs) == 2


def test_clean_rejects_short_and_wrong_script():
    assert clean("hi", "tamil") is None
    assert clean("x" * 200, "tamil") is None
    assert clean(TA_DOC, "tamil") == TA_DOC.strip()


def test_norm_hash_ignores_case_and_space():
    assert norm_hash("Hello   World") == norm_hash("hello world")


def test_write_shards_done_marker(tmp_path):
    from tamil.corpus import shards_done

    def dies():
        yield "tamil", "a"
        raise KeyboardInterrupt  # session killed mid-write
    try:
        write_shards(dies(), tmp_path / "x", docs_per_shard=1)
    except KeyboardInterrupt:
        pass
    assert not shards_done(tmp_path / "x")
    write_shards(iter([("tamil", "a")]), tmp_path / "y", docs_per_shard=1)
    assert shards_done(tmp_path / "y")


def test_blocklist_source_without_text_raises():
    import pytest

    from tamil.blocklist import build_blocklist
    src = Source("fake/eval", None, "test", "wrong_field", "eval", "cc0")
    with pytest.raises(ValueError, match="fake/eval"):
        build_blocklist([src], lambda s: iter([{"premise": "ஒரு நீண்ட தமிழ் வாக்கியம் இது."}]))


def test_write_shards(tmp_path):
    paths = write_shards(iter([("tamil", "a"), ("english", "b"), ("tamil", "c")]), tmp_path, docs_per_shard=2)
    assert len(paths) == 2
    rows = [json.loads(l) for p in paths for l in gzip.open(p, "rt", encoding="utf-8")]
    assert rows == [{"lang": "tamil", "text": "a"}, {"lang": "english", "text": "b"}, {"lang": "tamil", "text": "c"}]


def test_load_rows_reads_jsonl_inside_official_zip(tmp_path, monkeypatch):
    import zipfile
    from tamil import corpus
    monkeypatch.setattr(corpus, "DOWNLOADS", tmp_path)
    rows = [{"native word": "வணக்கம்", "english word": "vanakkam"}, {"native word": "நன்றி", "english word": "nandri"}]
    with zipfile.ZipFile(tmp_path / "tam.zip", "w") as z:
        z.writestr("tam_test.json", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    src = Source("x/aks", None, "test", "native word,english word", "eval", "cc-by",
                 url="https://example.invalid/tam.zip", member="tam_test.json")
    assert list(corpus.load_rows(src)) == rows


def test_load_rows_passes_revision_and_split_keyed_data_files(monkeypatch):
    import sys
    import types
    from tamil import corpus
    seen = {}
    fake = types.ModuleType("datasets")  # `datasets` may not be installed locally
    fake.load_dataset = lambda *a, **k: seen.update(k) or iter([{"t": 1}])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    src = Source("x/xnli", None, "test", "t", "eval", "cc0", data_files="ta/test/0000.parquet",
                 revision="refs/convert/parquet")
    assert list(corpus.load_rows(src)) == [{"t": 1}]
    assert seen["revision"] == "refs/convert/parquet"
    assert seen["data_files"] == {"test": "ta/test/0000.parquet"} and seen["split"] == "test"


def test_load_texts_skips_first_nonempty_docs(monkeypatch):
    from tamil import corpus
    rows = [{"text": "a"}, {"text": ""}, {"text": "b"}, {"text": "c"}, {"text": "d"}]
    monkeypatch.setattr(corpus, "load_rows", lambda src: iter(rows))
    src = Source("x/w", None, "train", "text", "tamil", "cc", skip_docs=2)
    assert list(corpus.load_texts(src)) == ["c", "d"]  # the empty row is not a doc, like load_heldout


def test_wikipedia_pretrain_source_skips_exactly_the_heldout_docs():
    from tamil import sources
    w = sources.WIKI_TA_TRAIN
    assert (w.hf_id, w.config) == (sources.WIKI_TA.hf_id, sources.WIKI_TA.config)
    assert w.role == "tamil" and w.skip_docs == sources.HELDOUT_DOCS and w in sources.PRETRAIN


def test_iter_mixed_round_robins_sources_of_one_role():
    a = Source("fake/a", None, "train", "text", "tamil", "cc")
    b = Source("fake/b", None, "train", "text", "tamil", "cc")
    docs = {"fake/a": [f"A{i} " + TA_DOC for i in range(5)], "fake/b": [f"B{i} " + TA_DOC for i in range(5)],
            "fake/en": [f"E{i} " + EN_DOC for i in range(50)]}
    out = list(iter_mixed([a, b, EN], {"tamil": 0.5, "tanglish": 0.0, "english": 0.5}, set(), seed=0,
                          tanglish_from_tamil=0.0, load_fn=lambda s: iter(docs[s.hf_id])))
    ta = [t[:2] for lang, t in out if lang == "tamil"]
    assert ta[:4] == ["A0", "B0", "A1", "B1"]


CODE_DOC = "def f(x):\n    if x:\n        return 1\n    return 0\n" + "}\n" * 6 + "# closing braces repeat in real code\n"


def test_clean_keeps_code_with_repeated_lines_and_indent():
    out = clean("\n" + CODE_DOC, "code")
    assert out is not None
    assert "    if x:\n        return 1" in out


def test_clean_rejects_short_code():
    assert clean("x = 1", "code") is None


def test_iter_mixed_streams_code_role():
    code = Source("fake/code", None, "train", "content", "code", "cc")
    docs = {"fake/ta": [f"T{i} " + TA_DOC for i in range(50)],
            "fake/code": [f"# c{i}\n" + CODE_DOC for i in range(50)]}
    out = list(iter_mixed([TA, code], {"tamil": 0.9, "tanglish": 0.0, "english": 0.0, "code": 0.1}, set(), seed=0,
                          tanglish_from_tamil=0.0, load_fn=lambda s: iter(docs.get(s.hf_id, []))))
    langs = {lang for lang, _ in out}
    assert "code" in langs
    assert next(t for lang, t in out if lang == "code").startswith("# c0")


def test_code_sources_are_tokenizer_only():
    from tamil import sources
    assert {s.role for s in sources.CODE} == {"code"}
    assert not set(sources.CODE) & set(sources.PRETRAIN)
