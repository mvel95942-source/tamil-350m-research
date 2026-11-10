"""Clean, dedupe, blocklist-filter and mix Tamil / Tanglish / English documents into shards."""
import gzip
import json
import os
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from random import Random

from tamil.blocklist import is_blocked, norm_hash
from tamil.sources import Source
from tamil.tanglish import romanize

MIN_CHARS = 50
MIN_SCRIPT_RATIO = 0.7
MAX_REPEATED_LINES = 0.3
SCHEMES = ("iso", "casual", "chat")
MAX_NOISE = 0.4
# big intermediates (2 GB Dakshina tar, shards) go to scratch; on Kaggle set TAMIL_SCRATCH=/tmp (20 GB working limit)
SCRATCH = Path(os.environ.get("TAMIL_SCRATCH", "artifacts"))
DOWNLOADS = SCRATCH / "downloads"
DONE = "_DONE"


def _script_ratio(text: str, lang: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if lang == "tamil":
        hits = sum("஀" <= c <= "௿" for c in letters)
    else:
        hits = sum(c.isascii() for c in letters)
    return hits / len(letters)


def clean(text: str, lang: str) -> str | None:
    # code keeps its first line's indent; its repeated lines (closing braces, blank returns) are normal
    text = text.lstrip("\r\n").rstrip() if lang == "code" else text.strip()
    if len(text) < MIN_CHARS or _script_ratio(text, lang) < MIN_SCRIPT_RATIO:
        return None
    if lang == "code":
        return text
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and 1 - len(set(lines)) / len(lines) > MAX_REPEATED_LINES:
        return None
    return text


def load_rows(src: Source) -> Iterator[dict]:
    """Stream rows of a source: HF streaming, or a file inside an official .tar/.zip.

    Archive members ending in .json are JSON lines; others are plain text, one row per line.
    """
    if src.url:
        local = DOWNLOADS / Path(src.url).name
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            part = local.with_suffix(local.suffix + ".part")
            urllib.request.urlretrieve(src.url, part)
            os.replace(part, local)  # a partial download never looks complete
        if zipfile.is_zipfile(local):
            with zipfile.ZipFile(local) as zf:
                raw = zf.read(src.member)
        else:
            with tarfile.open(local) as tf:
                raw = tf.extractfile(src.member).read()
        for line in raw.decode("utf-8").splitlines():
            if src.member.endswith(".json"):
                if line.strip():
                    yield json.loads(line)
            else:
                yield {src.fields[0]: line}
        return
    from datasets import load_dataset
    # data_files keyed by split, so a single-file load still exposes the split we ask for
    data_files = {src.split: src.data_files} if src.data_files else None
    yield from load_dataset(src.hf_id, src.config, split=src.split, data_dir=src.data_dir,
                            data_files=data_files, revision=src.revision, streaming=True)


def load_texts(src: Source) -> Iterator[str]:
    skip = src.skip_docs
    for row in load_rows(src):
        text = " ".join(str(row[f]) for f in src.fields if row.get(f))
        if skip and text.strip():
            skip -= 1
            continue
        if not skip:
            yield text


def _round_robin(its: list[Iterator[str]]) -> Iterator[str]:
    """One doc from each source in turn, dropping a source when it runs dry."""
    its = list(its)
    while its:
        for it in list(its):
            try:
                yield next(it)
            except StopIteration:
                its.remove(it)


def iter_mixed(sources: list[Source], mix: dict[str, float], blocklist: set[int], seed: int,
               tanglish_from_tamil: float = 0.8,
               load_fn: Callable[[Source], Iterator[str]] = load_texts) -> Iterator[tuple[str, str]]:
    """Yield (lang, text) so each lang's share of characters tracks `mix`.

    Each step picks the lang furthest below its target share. Tanglish is mostly Tamil docs
    romanized with a random scheme + noise, the rest real romanized text. Stops when a needed
    stream runs dry, so the ratio holds over the whole output.
    """
    rng = Random(seed)
    def stream(role: str) -> Iterator[str]:  # a function, so each stream binds its own `role`
        # round-robin, not one source after another: a byte cap would otherwise never reach the later ones
        return _round_robin([load_fn(s) for s in sources if s.role == role])

    streams = {role: stream(role) for role in ("tamil", "english", "tanglish_real", "code")}
    seen: set[int] = set()

    def next_doc(role: str, lang: str) -> str | None:
        for raw in streams[role]:
            text = clean(raw, lang)
            if text is None or is_blocked(text, blocklist):
                continue
            h = norm_hash(text)
            if h in seen:
                continue
            seen.add(h)
            return text
        return None

    emitted = {lang: 0 for lang in mix}
    while True:
        total = sum(emitted.values())
        lang = max(mix, key=lambda l: (mix[l] * total - emitted[l], mix[l]))
        if lang == "tanglish":
            doc = None
            if rng.random() >= tanglish_from_tamil:
                doc = next_doc("tanglish_real", "tanglish")
            if doc is None:
                src = next_doc("tamil", "tamil")
                doc = None if src is None else romanize(src, rng.choice(SCHEMES), rng, rng.uniform(0, MAX_NOISE))
        else:
            doc = next_doc(lang, lang)
        if doc is None:
            return
        emitted[lang] += len(doc)
        yield lang, doc


def write_shards(it: Iterator[tuple[str, str]], out_dir: Path, docs_per_shard: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    f = None
    for i, (lang, text) in enumerate(it):
        if i % docs_per_shard == 0:
            if f:
                f.close()
            paths.append(out_dir / f"shard_{len(paths):05d}.jsonl.gz")
            f = gzip.open(paths[-1], "wt", encoding="utf-8")
        f.write(json.dumps({"lang": lang, "text": text}, ensure_ascii=False) + "\n")
    if f:
        f.close()
    (out_dir / DONE).touch()  # written last: a killed session leaves no marker
    return paths


def shards_done(out_dir: Path) -> bool:
    return (out_dir / DONE).exists()
