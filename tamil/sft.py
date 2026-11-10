"""SFT stage: teach the pretrained model to answer, on Aya (Tamil, human-written) + IndicAlign (Tamil).

Format: <s><|user|>\\n{question}</s><|assistant|>\\n{answer}</s> ... ; loss on assistant tokens only.
Single GPU (the SFT set is small); resumable like tamil.train.
"""
import argparse
import dataclasses
import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tamil import sources
from tamil.blocklist import is_blocked
from tamil.corpus import _script_ratio
from tamil.model import BOS_ID, EOS_ID, PAD_ID, make_model
from tamil.train import _save, lr_at

USER, ASSISTANT = "<|user|>", "<|assistant|>"
# for HF users; must match encode_chat
CHAT_TEMPLATE = ("<s>{% for m in messages %}{% if m['role'] == 'user' %}<|user|>\n{{ m['content'] }}</s>"
                 "{% else %}<|assistant|>\n{{ m['content'] }}</s>{% endif %}{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")
MIN_TAMIL_RATIO = 0.5


def encode_chat(tok, turns: list[dict], ctx: int) -> tuple[list[int], list[int]]:
    """(input_ids, labels); labels are -100 except on assistant text and its closing </s>."""
    ids, labels = [BOS_ID], [-100]
    for t in turns:
        marker = tok.token_to_id(USER if t["role"] == "user" else ASSISTANT)
        body = tok.encode("\n" + t["content"], add_special_tokens=False).ids + [EOS_ID]
        ids += [marker] + body
        labels += [-100] + (body if t["role"] == "assistant" else [-100] * len(body))
    return ids[:ctx], labels[:ctx]


def build_sft_rows(load_rows_fn: Callable, blocklist: set[int]) -> list[list[dict]]:
    out = []
    for src in sources.SFT:
        before = len(out)
        for row in load_rows_fn(src):
            if "aya" in src.hf_id:
                if row.get("language") != "Tamil":
                    continue
                conv = [{"role": "user", "content": row["inputs"]}, {"role": "assistant", "content": row["targets"]}]
            else:  # IndicAlign: src.text_field ("tam_Taml" or "interactions") = [[user, assistant], ...]
                pairs = [p for p in row.get(src.text_field) or [] if len(p) >= 2 and p[0] and p[1]]
                conv = [{"role": r, "content": str(x)} for pair in pairs for r, x in zip(("user", "assistant"), pair)]
                answers = " ".join(t["content"] for t in conv if t["role"] == "assistant")
                if not conv or _script_ratio(answers, "tamil") < MIN_TAMIL_RATIO:
                    continue
            if any(is_blocked(t["content"], blocklist) or not t["content"].strip() for t in conv):
                continue
            out.append(conv)
        if len(out) == before:
            raise ValueError(f"{src.hf_id}: 0 Tamil conversations; check its fields/shape (unverified on HF)")
    return out


def sft_loss(model, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids=ids).logits[:, :-1].float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                                             ignore_index=-100)


@dataclass
class SFTConfig:
    base_ckpt: Path
    tokenizer_path: Path
    data_path: Path  # jsonl, one conversation (list of {role, content}) per line
    out_dir: Path
    model_size: str = "350m"
    lr: float = 2e-5
    min_lr: float = 2e-6
    warmup: int = 50
    epochs: int = 2
    micro_bs: int = 8
    grad_accum: int = 4
    ctx: int = 1024
    grad_clip: float = 1.0
    ckpt_every_min: float = 30.0
    wall_budget_h: float = 11.5
    stop_after: int | None = None
    seed: int = 0
    resume_from: Path | None = None  # previous session's ckpt.pt; used if out_dir has none


def _pad(batch: list[tuple[list[int], list[int]]], device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(i) for i, _ in batch)
    ids = torch.tensor([i + [PAD_ID] * (n - len(i)) for i, _ in batch], device=device)
    lab = torch.tensor([l + [-100] * (n - len(l)) for _, l in batch], device=device)
    return ids, lab


def sft(cfg: SFTConfig) -> None:
    from tokenizers import Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    tok = Tokenizer.from_file(str(cfg.tokenizer_path))
    examples = [encode_chat(tok, json.loads(l), cfg.ctx)
                for l in Path(cfg.data_path).read_text(encoding="utf-8").splitlines() if l.strip()]

    model = make_model(tok.get_vocab_size(), cfg.model_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    load_from = ckpt_path if ckpt_path.exists() else cfg.resume_from
    if load_from and Path(load_from).exists():
        ck = torch.load(load_from, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        step = int(ck["step"])
    else:
        model.load_state_dict(torch.load(cfg.base_ckpt, map_location=device, weights_only=True)["model"])

    per_step = cfg.micro_bs * cfg.grad_accum
    steps_per_epoch = math.ceil(len(examples) / per_step)
    total = steps_per_epoch * cfg.epochs
    end = min(total, cfg.stop_after) if cfg.stop_after else total
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    started = last_ckpt = time.time()
    model.train()
    while step < end:
        epoch, k = divmod(step, steps_per_epoch)
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(examples))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg, total)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for a in range(cfg.grad_accum):
            idx = order[k * per_step + a * cfg.micro_bs: k * per_step + (a + 1) * cfg.micro_bs]
            if len(idx) == 0:
                continue
            ids, lab = _pad([examples[i] for i in idx], device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = sft_loss(model, ids, lab) / cfg.grad_accum
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        log.write(json.dumps({"step": step, "epoch": epoch, "loss": loss_sum, "lr": opt.param_groups[0]["lr"]}) + "\n")
        log.flush()
        out_of_time = time.time() - started > cfg.wall_budget_h * 3600
        if time.time() - last_ckpt > cfg.ckpt_every_min * 60 or out_of_time:
            _save(ckpt_path, model, opt, scaler, step)
            last_ckpt = time.time()
        if out_of_time:
            break
    _save(ckpt_path, model, opt, scaler, step)
    log.close()


def prepare(out_path: Path, blocklist: set[int], load_rows_fn: Callable | None = None) -> int:
    from tamil.corpus import load_rows
    rows = build_sft_rows(load_rows_fn or load_rows, blocklist)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in rows), encoding="utf-8")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(SFTConfig):
        ap.add_argument(f"--{f.name}", type={int | None: int, Path | None: Path}.get(f.type, f.type),
                        required=f.default is dataclasses.MISSING,
                        default=None if f.default is dataclasses.MISSING else f.default)
    sft(SFTConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
