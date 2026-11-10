"""Spawned by test_train's 2-rank tests: runs tamil.train in one gloo rank (CPU)."""
import os


def run(rank: int, world: int, port: int, cfg_kwargs: dict, nan_rank: int | None) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
    import tamil.train as t
    if rank == nan_rank:
        orig = t._loss
        t._loss = lambda m, b: orig(m, b) * float("nan")
    t.train(t.TrainConfig(**cfg_kwargs))
