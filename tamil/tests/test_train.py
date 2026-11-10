import json

import pytest
import torch

import tamil.train as train_mod
from tamil.train import TrainConfig, train


def cfg(tmp_path, packed_bin, trained_tok_path, **kw):
    base = dict(bin_path=packed_bin, tokenizer_path=trained_tok_path, out_dir=tmp_path, model_size="tiny",
                micro_bs=4, grad_accum=2, ctx=16, lr=3e-3, min_lr=3e-4, warmup=5, max_steps=60,
                ckpt_every_min=30.0, wall_budget_h=1.0)
    base.update(kw)
    return TrainConfig(**base)


def losses(out_dir):
    return [json.loads(l)["loss"] for l in (out_dir / "log.jsonl").read_text().splitlines()]


def weights(out_dir):
    return torch.cat([v.flatten().float() for v in _ck(out_dir)["model"].values()])


def _ck(out_dir):
    return torch.load(out_dir / "ckpt.pt", weights_only=True)


def test_loss_is_next_token_cross_entropy():
    from tamil.model import make_model
    torch.manual_seed(0)
    m = make_model(64, "tiny")
    batch = torch.randint(0, 64, (2, 9))
    logits = m(input_ids=batch[:, :-1]).logits
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), batch[:, 1:].reshape(-1))
    assert torch.allclose(train_mod._loss(train_mod.LossModel(m), batch), expected, atol=1e-5)


def test_chunked_loss_matches_full_loss_and_gradients():
    from tamil.model import make_model
    torch.manual_seed(0)
    m = make_model(64, "tiny")
    batch = torch.randint(0, 64, (2, 9))  # 16 target tokens; chunk 5 leaves a ragged last chunk
    logits = m(input_ids=batch[:, :-1]).logits
    full = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), batch[:, 1:].reshape(-1))
    full.backward()
    want = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    chunked = train_mod._loss(train_mod.LossModel(m, chunk=5), batch)
    chunked.backward()
    assert torch.allclose(chunked, full, atol=1e-5)
    assert all(torch.allclose(p.grad, g, atol=1e-5) for p, g in zip(m.parameters(), want))


def test_loss_decreases(tmp_path, packed_bin, trained_tok_path):
    train(cfg(tmp_path, packed_bin, trained_tok_path))
    ls = losses(tmp_path)
    assert sum(ls[-5:]) / 5 < sum(ls[:5]) / 5 * 0.8


def test_resume_exact(tmp_path, packed_bin, trained_tok_path):
    a, b = tmp_path / "a", tmp_path / "b"
    train(cfg(a, packed_bin, trained_tok_path, max_steps=20))
    train(cfg(b, packed_bin, trained_tok_path, max_steps=20, stop_after=10))
    assert _ck(b)["step"] == 10
    train(cfg(b, packed_bin, trained_tok_path, max_steps=20))  # resumes from step 10
    assert torch.allclose(weights(a), weights(b), atol=1e-6)


def test_nonfinite_halts(tmp_path, packed_bin, trained_tok_path, monkeypatch):
    orig = train_mod._loss
    monkeypatch.setattr(train_mod, "_loss", lambda m, b: orig(m, b) * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        train(cfg(tmp_path, packed_bin, trained_tok_path, max_nonfinite=20))


def _spawn_two_ranks(cfg_kwargs, nan_rank=None, deadline_s=120):
    """Run 2 gloo ranks; return per-rank exit codes, or fail if they hang past the deadline."""
    import socket
    import time

    import torch.multiprocessing as mp

    from tamil.tests.ddp_worker import run
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.start_processes(run, args=(2, port, cfg_kwargs, nan_rank), nprocs=2, join=False, start_method="spawn")
    end = time.time() + deadline_s
    while any(p.is_alive() for p in ctx.processes):
        if time.time() > end:
            for p in ctx.processes:
                p.kill()
            pytest.fail("ranks hung (desync between DDP ranks)")
        time.sleep(0.5)
    return [p.exitcode for p in ctx.processes]


def _ddp_kwargs(out, packed_bin, trained_tok_path, **kw):
    c = cfg(out, packed_bin, trained_tok_path, **kw)
    return {f: getattr(c, f) for f in c.__dataclass_fields__}


def test_ddp_nan_on_one_rank_stops_both(tmp_path, packed_bin, trained_tok_path):
    codes = _spawn_two_ranks(_ddp_kwargs(tmp_path, packed_bin, trained_tok_path, max_nonfinite=3), nan_rank=0)
    assert codes[0] != 0 and codes[1] != 0  # both raise "non-finite", neither hangs


def test_ddp_wall_budget_stops_both(tmp_path, packed_bin, trained_tok_path):
    codes = _spawn_two_ranks(_ddp_kwargs(tmp_path, packed_bin, trained_tok_path, wall_budget_h=2e-4))
    assert codes == [0, 0]
    assert 1 <= _ck(tmp_path)["step"] < 60


def test_resume_from_previous_session(tmp_path, packed_bin, trained_tok_path):
    a, b = tmp_path / "a", tmp_path / "b"
    train(cfg(a, packed_bin, trained_tok_path, max_steps=20, stop_after=10))
    train(cfg(b, packed_bin, trained_tok_path, max_steps=20, resume_from=a / "ckpt.pt"))
    full = tmp_path / "full"
    train(cfg(full, packed_bin, trained_tok_path, max_steps=20))
    assert _ck(b)["step"] == 20
    assert torch.allclose(weights(full), weights(b), atol=1e-6)


def test_grad_ckpt_runs(tmp_path, packed_bin, trained_tok_path):
    train(cfg(tmp_path, packed_bin, trained_tok_path, max_steps=5, grad_ckpt=True))
    assert all(map(lambda x: x == x, losses(tmp_path)))


def test_wall_budget_exits_cleanly(tmp_path, packed_bin, trained_tok_path):
    train(cfg(tmp_path, packed_bin, trained_tok_path, wall_budget_h=1e-6))
    ck = _ck(tmp_path)
    assert 1 <= ck["step"] < 60


def test_ddp_sharded_optimizer_checkpoints_full_state_and_resumes(tmp_path, packed_bin, trained_tok_path):
    from tamil.model import make_model
    kw = _ddp_kwargs(tmp_path, packed_bin, trained_tok_path, max_steps=10, stop_after=5)
    assert _spawn_two_ranks(kw) == [0, 0]
    ck = _ck(tmp_path)
    n_params = len(list(make_model(32, "tiny").parameters()))
    assert ck["step"] == 5 and len(ck["optim"]["state"]) == n_params  # both ranks' shards, consolidated
    assert _spawn_two_ranks(dict(kw, stop_after=None)) == [0, 0]
    assert _ck(tmp_path)["step"] == 10
