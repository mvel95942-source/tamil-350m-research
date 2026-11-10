"""Speed and memory benchmark for a trained Tamil tokenizer (tok64k / tok32k).

Measures, on this machine and with a warm tokenizer:
  * cold load time and RSS cost of loading tokenizer.json
  * encode / decode throughput per text domain (Tamil, Tanglish, English, code,
    digits, emoji) at 1 MB and 10 MB of UTF-8 input
  * single long-document latency (200k chars)
  * batch throughput (encode_batch)
  * peak RSS while streaming 100 MB through the encoder in 1 MB chunks
  * memory per 1M token ids as Python list vs numpy uint16 / int32

    python scripts/bench_tokenizer.py --tok artifacts/tok64k
    python scripts/bench_tokenizer.py --tok artifacts/tok64k --compare artifacts/tok32k

Writes reports/tokenizer_bench.md and .json. Timings are median of --repeat runs
(the 10 MB pass runs once unless --repeat-big is given).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer

PROC = psutil.Process()

# Representative units; each is repeated until it covers the requested byte size.
TEXTS: dict[str, str] = {
    "tamil": "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்றாகும். "
             "இலக்கியம் செழித்தது; சங்க காலம் தொட்டு இன்று வரை. ",
    "tanglish": "vanakkam da, epdi iruke? naan nalla iruken, nee epdi? "
                "sapteengala? poi thoongu, naalai velai irukku. ",
    "english": "The quick brown fox jumps over the lazy dog while the scientist "
               "writes a paper about tokenization efficiency and vocabulary size. ",
    "code": "def add(a, b):\n    if a > 0:\n        return {\"x\": [a, b]}\n"
            "\treturn a + b  # comment with நன்றி\n",
    "digits": "1234567890 3.14159 1,00,000 2026-11-06 0xDEADBEEF 42% ₹1,299 ",
    "emoji": "😀🔥👍🏽👨‍👩‍👧‍👦🏳️‍🌈1️⃣🇮🇳🧑🏾‍💻 ",
}


def rss_mb() -> float:
    return PROC.memory_info().rss / 1e6


def mb_text(unit: str, target_bytes: int) -> str:
    unit_bytes = len(unit.encode("utf-8"))
    return unit * max(1, target_bytes // max(unit_bytes, 1))


def timed(fn, repeat: int):
    """Return (best_seconds, median_seconds, result) after one warmup call."""
    fn()  # warmup
    times = []
    result = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2] if len(times) % 2 else sum(times[len(times) // 2 - 1:len(times) // 2 + 1]) / 2
    return times[0], median, result


def list_memory_bytes(ids: list[int]) -> int:
    """Approximate real memory of a Python list[int] (PyLong is 28-32 B; small ints cached)."""
    total = sys.getsizeof(ids)
    for i in ids:
        if i > 256 or i < -5:
            total += sys.getsizeof(i)
    return total


def bench_one(tok: Tokenizer, path: Path, name: str, repeat: int, repeat_big: int,
              big_bytes: int) -> dict:
    out: dict = {"name": name, "dir": str(path), "sha256": None}
    import hashlib
    out["sha256"] = hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()

    # ---- load cost (cold-ish: a fresh Tokenizer instance, file already in page cache)
    gc.collect()
    before = rss_mb()
    t0 = time.perf_counter()
    fresh = Tokenizer.from_file(str(path / "tokenizer.json"))
    load_s = time.perf_counter() - t0
    after = rss_mb()
    out["load"] = {
        "seconds": round(load_s, 4),
        "rss_before_mb": round(before, 1),
        "rss_after_mb": round(after, 1),
        "rss_delta_mb": round(after - before, 1),
        "vocab": len(fresh.get_vocab()),
    }
    del fresh
    gc.collect()

    tok.encode("warmup தமிழ்")  # page the model in

    # ---- throughput per domain at 1 MB and big_bytes
    out["throughput"] = {}
    for label, unit in TEXTS.items():
        for size_label, target, reps in (("1mb", 1_000_000, repeat),
                                         (f"{big_bytes // 1_000_000}mb", big_bytes, repeat_big)):
            text = mb_text(unit, target)
            nbytes = len(text.encode("utf-8"))
            nchars = len(text)

            def do_encode(t=text):
                return tok.encode(t, add_special_tokens=False)

            best, med, enc = timed(do_encode, reps)
            ids = enc.ids

            def do_decode(ids=ids):
                return tok.decode(ids)

            _, dec_med, decoded = timed(do_decode, reps)
            ok = decoded == text
            out["throughput"][f"{label}_{size_label}"] = {
                "bytes": nbytes,
                "chars": nchars,
                "tokens": len(ids),
                "tokens_per_char": round(len(ids) / max(nchars, 1), 4),
                "encode_mb_s": round(nbytes / 1e6 / med, 3) if med else None,
                "encode_tok_s": round(len(ids) / med) if med else None,
                "encode_ms_best": round(best * 1000, 1),
                "encode_ms_median": round(med * 1000, 1),
                "decode_mb_s": round(nbytes / 1e6 / dec_med, 3) if dec_med else None,
                "decode_tok_s": round(len(ids) / dec_med) if dec_med else None,
                "decode_ms_median": round(dec_med * 1000, 1),
                "roundtrip_exact": ok,
            }
            del text, ids, enc, decoded
            gc.collect()

    # ---- long single document
    long_text = mb_text(TEXTS["tamil"], 200_000 * 3)  # 200k chars of Tamil (3 B/char)
    long_text = long_text[:200_000]
    t0 = time.perf_counter()
    e = tok.encode(long_text, add_special_tokens=False)
    enc_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    d = tok.decode(e.ids)
    dec_ms = (time.perf_counter() - t0) * 1000
    out["long_doc"] = {
        "chars": len(long_text),
        "bytes": len(long_text.encode("utf-8")),
        "tokens": len(e.ids),
        "encode_ms": round(enc_ms, 1),
        "decode_ms": round(dec_ms, 1),
        "roundtrip_exact": d == long_text,
    }

    # ---- batch throughput
    docs = [mb_text(TEXTS["tamil"], 1_000), mb_text(TEXTS["english"], 1_000),
            mb_text(TEXTS["tanglish"], 1_000), mb_text(TEXTS["code"], 1_000)] * 128
    nbytes = sum(len(x.encode("utf-8")) for x in docs)
    t0 = time.perf_counter()
    encs = tok.encode_batch(docs, add_special_tokens=False)
    batch_s = time.perf_counter() - t0
    out["batch"] = {
        "docs": len(docs),
        "bytes": nbytes,
        "tokens": sum(len(x.ids) for x in encs),
        "seconds": round(batch_s, 3),
        "docs_per_s": round(len(docs) / batch_s, 1),
        "mb_per_s": round(nbytes / 1e6 / batch_s, 2),
        "tok_per_s": round(sum(len(x.ids) for x in encs) / batch_s),
    }
    del docs, encs
    gc.collect()

    # ---- 100 MB streaming: process 1 MB chunks, keep only the token count
    stream_target = 100_000_000
    chunk = mb_text(TEXTS["tamil"], 1_000_000)
    n_chunks = stream_target // len(chunk.encode("utf-8"))
    gc.collect()
    base_rss = rss_mb()
    proc = psutil.Process()
    peak = base_rss
    total_tokens = 0
    t0 = time.perf_counter()
    for _ in range(n_chunks):
        ids = tok.encode(chunk, add_special_tokens=False).ids
        total_tokens += len(ids)
        peak = max(peak, proc.memory_info().rss / 1e6)
        del ids
    stream_s = time.perf_counter() - t0
    out["streaming"] = {
        "chunks": n_chunks,
        "chunk_mb": round(len(chunk.encode("utf-8")) / 1e6, 2),
        "total_mb": round(n_chunks * len(chunk.encode("utf-8")) / 1e6, 1),
        "seconds": round(stream_s, 2),
        "mb_per_s": round(n_chunks * len(chunk.encode("utf-8")) / 1e6 / stream_s, 2),
        "tokens": total_tokens,
        "baseline_rss_mb": round(base_rss, 1),
        "peak_rss_mb": round(peak, 1),
        "peak_delta_mb": round(peak - base_rss, 1),
    }
    del chunk
    gc.collect()

    # ---- memory per 1M token ids
    million_text = mb_text(TEXTS["tamil"], 4_500_000)  # ~1M Tamil tokens
    ids = tok.encode(million_text, add_special_tokens=False).ids
    n = len(ids)
    list_b = list_memory_bytes(ids)
    arr16 = np.asarray(ids, dtype=np.uint16)
    arr32 = np.asarray(ids, dtype=np.int32)
    out["id_memory"] = {
        "tokens": n,
        "list_mb_per_1m": round(list_b / 1e6 * (1_000_000 / n), 2),
        "uint16_mb_per_1m": round(arr16.nbytes / 1e6 * (1_000_000 / n), 2),
        "int32_mb_per_1m": round(arr32.nbytes / 1e6 * (1_000_000 / n), 2),
        "list_bytes": list_b,
        "list_per_token_bytes": round(list_b / n, 1),
        "uint16_per_token_bytes": 2,
        "int32_per_token_bytes": 4,
        "max_id": int(max(ids)),
    }
    del million_text, ids, arr16, arr32
    gc.collect()
    return out


def environment(path: Path) -> dict:
    import tokenizers
    import psutil as _ps
    vm = _ps.virtual_memory()
    try:
        cpu = platform.processor() or platform.machine()
    except Exception:  # noqa: BLE001
        cpu = "unknown"
    return {
        "python": sys.version.split()[0],
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "psutil": _ps.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu,
        "cpu_logical": os.cpu_count(),
        "cpu_physical": _ps.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "tokenizer_dir": str(path),
    }


def fmt_table(bench: dict, order: list[str]) -> list[str]:
    lines = ["| domain | size | MB/s encode | tok/s encode | MB/s decode | ms encode | round-trip |",
             "|---|---|---|---|---|---|---|"]
    for name in order:
        th = bench[name]["throughput"]
        for key in [f"{d}_1mb" for d in TEXTS]:
            r = th.get(key)
            if not r:
                continue
            domain = key.rsplit("_", 1)[0]
            lines.append(f"| {name} · {domain} | 1 MB | {r['encode_mb_s']:.2f} | "
                         f"{r['encode_tok_s']:,} | {r['decode_mb_s']:.2f} | "
                         f"{r['encode_ms_median']:.1f} | {'yes' if r['roundtrip_exact'] else '**NO**'} |")
    for name in order:
        th = bench[name]["throughput"]
        big_keys = [k for k in th if not k.endswith("_1mb")]
        for key in big_keys:
            r = th[key]
            domain, size = key.rsplit("_", 1)
            lines.append(f"| {name} · {domain} | {size.upper()} | {r['encode_mb_s']:.2f} | "
                         f"{r['encode_tok_s']:,} | {r['decode_mb_s']:.2f} | "
                         f"{r['encode_ms_median']:.1f} | {'yes' if r['roundtrip_exact'] else '**NO**'} |")
    return lines


def render(bench: dict, order: list[str], env: dict, elapsed: float, big_bytes: int) -> str:
    primary = order[0]
    b = bench[primary]
    lines = [
        "# Tokenizer speed and memory benchmark",
        "",
        f"- artifact: `{b['dir']}` · vocab {b['load']['vocab']:,}",
        f"- sha256: `{b['sha256']}`",
        f"- machine: {env['platform']} · {env['cpu_physical']}C/{env['cpu_logical']}T · "
        f"{env['ram_total_gb']} GB RAM",
        f"- python {env['python']} · tokenizers {env['tokenizers']} · numpy {env['numpy']}",
        f"- wall time: {elapsed:.1f}s",
        "",
        "## Load cost",
        "",
        "| tokenizer | load ms | RSS before (MB) | RSS after (MB) | delta (MB) |",
        "|---|---|---|---|---|",
    ]
    for name in order:
        l = bench[name]["load"]
        lines.append(f"| {name} | {l['seconds'] * 1000:.1f} | {l['rss_before_mb']:.1f} | "
                     f"{l['rss_after_mb']:.1f} | {l['rss_delta_mb']:.1f} |")

    lines += ["", "## Encode / decode throughput", "",
              "Median wall time on a warm tokenizer; MB = 10^6 UTF-8 bytes of input. "
              "`tok/s` counts output tokens.", ""]
    lines += fmt_table(bench, order)

    lines += ["", "## Long document and batch", "",
              "| tokenizer | 200k-char Tamil doc | chars | tokens | encode ms | decode ms | round-trip |",
              "|---|---|---|---|---|---|---|"]
    for name in order:
        d = bench[name]["long_doc"]
        lines.append(f"| {name} | 200k chars | {d['chars']:,} | {d['tokens']:,} | "
                     f"{d['encode_ms']:.1f} | {d['decode_ms']:.1f} | "
                     f"{'yes' if d['roundtrip_exact'] else '**NO**'} |")
    lines += ["", "| tokenizer | batch docs | MB | tok/s | docs/s | MB/s |",
              "|---|---|---|---|---|---|"]
    for name in order:
        x = bench[name]["batch"]
        lines.append(f"| {name} | {x['docs']:,} | {x['bytes'] / 1e6:.1f} | {x['tok_per_s']:,} | "
                     f"{x['docs_per_s']:,} | {x['mb_per_s']:.2f} |")

    lines += ["", "## Streaming 100 MB (1 MB chunks, ids discarded)", "",
              "| tokenizer | chunks | MB/s | peak RSS (MB) | baseline RSS (MB) | peak delta (MB) |",
              "|---|---|---|---|---|---|"]
    for name in order:
        s = bench[name]["streaming"]
        lines.append(f"| {name} | {s['chunks']} | {s['mb_per_s']:.2f} | {s['peak_rss_mb']:.1f} | "
                     f"{s['baseline_rss_mb']:.1f} | {s['peak_delta_mb']:.1f} |")

    lines += ["", "## Memory per 1M token ids", "",
              "| tokenizer | tokens measured | Python list (MB) | numpy uint16 (MB) | numpy int32 (MB) | bytes/token (list) |",
              "|---|---|---|---|---|---|"]
    for name in order:
        m = bench[name]["id_memory"]
        lines.append(f"| {name} | {m['tokens']:,} | {m['list_mb_per_1m']:.2f} | "
                     f"{m['uint16_mb_per_1m']:.2f} | {m['int32_mb_per_1m']:.2f} | "
                     f"{m['list_per_token_bytes']:.1f} |")

    lines += ["", "## Notes", "",
              "- Timings are single-process, CPU only; `tokenizers` encodes in Rust and "
              "`encode_batch` uses its internal thread pool.",
              "- RSS is process resident memory from `psutil`; the delta includes the loaded "
              "model, not the Python interpreter.",
              "- The 100 MB stream keeps only the per-chunk token count, so peak RSS shows the "
              "encoder's own working set rather than the corpus.",
              "- `tokenizer.json` is read from the OS page cache after the first pass, so load "
              "time is a warm-start figure.",
              ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", type=Path, default=Path("artifacts/tok64k"))
    ap.add_argument("--compare", type=Path, nargs="*", default=[])
    ap.add_argument("--repeat", type=int, default=3, help="repeats for the 1 MB pass")
    ap.add_argument("--repeat-big", type=int, default=1, help="repeats for the 10 MB pass")
    ap.add_argument("--big-mb", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("reports/tokenizer_bench"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    paths = [args.tok] + list(args.compare)
    bench, order = {}, []
    for p in paths:
        print(f"benchmarking {p} ...", flush=True)
        tok = Tokenizer.from_file(str(p / "tokenizer.json"))
        name = p.name
        bench[name] = bench_one(tok, p, name, args.repeat, args.repeat_big, args.big_mb * 1_000_000)
        order.append(name)
        del tok
        gc.collect()
        print(f"  done in {time.perf_counter() - t0:.0f}s", flush=True)

    env = environment(args.tok)
    elapsed = time.perf_counter() - t0
    md = render(bench, order, env, elapsed, args.big_mb * 1_000_000)
    payload = {
        "environment": env,
        "benchmarks": bench,
        "order": order,
        "elapsed_s": round(elapsed, 1),
        "texts": TEXTS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(md)
    print(f"\nwrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
