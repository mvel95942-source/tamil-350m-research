"""Resumable fp16 pretraining loop (DDP via torchrun) for Kaggle's 12 h sessions.

    torchrun --nproc_per_node=2 -m tamil.train --bin_path artifacts/train.bin \
        --tokenizer_path artifacts/tok32k/tokenizer.json --out_dir artifacts/run

Auto-resumes from <out_dir>/ckpt.pt. Checkpoints are written atomically (tmp + rename) and hold
only tensors and plain numbers, so they load with weights_only=True.
"""
import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

from tamil.model import make_model
from tamil.pack import PackedDataset


@dataclass
class TrainConfig:
    bin_path: Path
    tokenizer_path: Path
    out_dir: Path
    model_size: str = "350m"
    # T4 speed test 2026-11-03 (350M, 2x T4): micro_bs 4 = 7,336 tok/s; 8 + grad_ckpt = 5,947; 8 alone OOMs
    micro_bs: int = 4
    grad_accum: int = 32
    ctx: int = 1024
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 500
    total_tokens: int = 2_000_000_000
    max_steps: int | None = None  # schedule length; derived from total_tokens when None
    stop_after: int | None = None  # end this session early (schedule unchanged)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ckpt_every_min: float = 30.0
    wall_budget_h: float = 11.5
    max_nonfinite: int = 20
    log_every: int = 1
    seed: int = 0
    resume_from: Path | None = None  # previous session's ckpt.pt (e.g. under /kaggle/input); used if out_dir has none
    grad_ckpt: bool = False  # activation checkpointing: less T4 memory, ~30% slower


# Target tokens per logits chunk. 1024 x 32k fp32 = 128 MB (the T4 OOM was a 250 MB chunk at
# 2048); override with TAMIL_LOSS_CHUNK when testing a larger micro-batch.
LOSS_CHUNK = int(os.environ.get("TAMIL_LOSS_CHUNK", "2048"))


def _chunk_ce(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy((h @ w.t()).float(), y, reduction="sum")


class LossModel(torch.nn.Module):
    """Wraps the LM so DDP's forward is the whole loss. Logits are made one chunk at a time and
    recomputed in backward, so a full (tokens x vocab) logits tensor never sits in memory."""

    def __init__(self, lm, chunk: int = LOSS_CHUNK):
        super().__init__()
        self.lm, self.chunk = lm, chunk

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        h = self.lm.model(input_ids=batch[:, :-1], use_cache=False).last_hidden_state
        h, y = h.reshape(-1, h.size(-1)), batch[:, 1:].reshape(-1)
        w = self.lm.lm_head.weight
        total = sum(checkpoint(_chunk_ce, h[i:i + self.chunk], w, y[i:i + self.chunk], use_reentrant=False)
                    for i in range(0, y.numel(), self.chunk))
        return total / y.numel()


def _loss(model, batch: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy; `model` is a LossModel (or DDP of one). Not via labels=: HF shifts again."""
    return model(batch)


def lr_at(step: int, cfg: TrainConfig, total: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, total - cfg.warmup))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def _dist() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        if torch.cuda.is_available():
            dist.init_process_group("nccl")
            local = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local)
            return dist.get_rank(), world, torch.device("cuda", local)
        dist.init_process_group("gloo")  # CPU (tests)
        return dist.get_rank(), world, torch.device("cpu")
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _agree(flags: list[bool], world: int, device: torch.device) -> list[bool]:
    """OR each flag across ranks, so every rank takes the same branch (no desync, no hang)."""
    t = torch.tensor([float(f) for f in flags], device=device)
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return [bool(x) for x in t.tolist()]


def _save(path: Path, model, opt, scaler, step: int, rank: int = 0) -> None:
    """Call on every rank: a sharded optimizer first gathers its shards onto rank 0, which writes."""
    if isinstance(opt, ZeroRedundancyOptimizer):
        opt.consolidate_state_dict(to=0)
    if rank != 0:
        return
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step}, tmp)
    os.replace(tmp, path)


def train(cfg: TrainConfig) -> None:
    from tokenizers import Tokenizer
    rank, world, device = _dist()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    torch.manual_seed(cfg.seed)
    vocab = Tokenizer.from_file(str(cfg.tokenizer_path)).get_vocab_size()
    model = make_model(vocab, cfg.model_size).to(device)
    if cfg.grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if world > 1:  # ZeRO-1: each rank keeps Adam state for half the params (-1.4 GB per T4 at 350M)
        opt = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, lr=cfg.lr, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))
    use_amp = device.type == "cuda"  # fp16 on T4 (no bf16); CPU tests run fp32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    load_from = ckpt_path if ckpt_path.exists() else cfg.resume_from
    if load_from and Path(load_from).exists():
        # load on CPU: a second full copy of model + Adam state on the GPU OOMs a T4 on resume (tamil-pretrain v2)
        ck = torch.load(load_from, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])  # moves each state tensor to its param's device
        scaler.load_state_dict(ck["scaler"])
        step = int(ck["step"])
        del ck
    loss_model = LossModel(model)
    ddp = torch.nn.parallel.DistributedDataParallel(loss_model) if world > 1 else loss_model

    tokens_per_step = cfg.micro_bs * cfg.ctx * cfg.grad_accum * world
    total = cfg.max_steps or max(1, cfg.total_tokens // tokens_per_step)
    end = min(total, cfg.stop_after) if cfg.stop_after else total
    ds = PackedDataset(Path(cfg.bin_path), cfg.ctx, rank, world, seed=cfg.seed)
    log = open(out / "log.jsonl", "a", encoding="utf-8") if rank == 0 else None
    started = last_ckpt = time.time()
    nonfinite = 0

    model.train()
    while step < end:
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg, total)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for k in range(cfg.grad_accum):
            batch = ds.batch(step * cfg.grad_accum + k, cfg.micro_bs).to(device)
            sync_ctx = ddp.no_sync() if world > 1 and k < cfg.grad_accum - 1 else _null()
            with sync_ctx, torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = _loss(ddp, batch) / cfg.grad_accum
            scaler.scale(loss).backward()
            loss_sum += loss.item()

        # rank 0's clock decides the time budget and checkpoints; any rank's NaN skips the step on all ranks
        out_of_time = rank == 0 and time.time() - started > cfg.wall_budget_h * 3600
        ckpt_due = rank == 0 and time.time() - last_ckpt > cfg.ckpt_every_min * 60
        bad, out_of_time, ckpt_due = _agree([not math.isfinite(loss_sum), out_of_time, ckpt_due], world, device)
        if bad:
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
            if nonfinite >= cfg.max_nonfinite:
                raise RuntimeError(f"non-finite loss for {nonfinite} consecutive steps at step {step}")
            step += 1  # skip the update; inf/NaN *gradients* are already handled by scaler.step
            if out_of_time:
                break
            continue
        nonfinite = 0
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1

        if log and step % cfg.log_every == 0:
            log.write(json.dumps({"step": step, "loss": loss_sum, "lr": opt.param_groups[0]["lr"],
                                  "tok_s": tokens_per_step / (time.time() - t0),
                                  "scale": scaler.get_scale() if use_amp else 1.0}) + "\n")
            log.flush()
        if ckpt_due or out_of_time:
            _save(ckpt_path, model, opt, scaler, step, rank)
            last_ckpt = time.time()
        if out_of_time:
            break

    _save(ckpt_path, model, opt, scaler, step, rank)
    if rank == 0:
        log.close()
    if world > 1:
        dist.destroy_process_group()


def _as_bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        kind = {int | None: int, Path | None: Path, bool: _as_bool}.get(f.type, f.type)
        ap.add_argument(f"--{f.name}", type=kind,
                        required=f.default is dataclasses.MISSING,
                        default=None if f.default is dataclasses.MISSING else f.default)
    train(TrainConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
