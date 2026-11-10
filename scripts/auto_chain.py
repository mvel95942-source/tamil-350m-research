"""Unattended Kaggle training chain: v4 pretrain -> v5 pretrain -> SFT -> eval.

The chain is idempotent and resumable. It is safe to start it again after a
reboot or a crash: the state file records which push already happened, and an
exclusive lock prevents two copies from running at once.

Training runs in the cloud. This process only polls Kaggle and pushes the next
stage, so closing the laptop does not stop the current GPU job. If the laptop is
off when a stage finishes, the chain resumes on the next start.

Run: python scripts/auto_chain.py
Log: artifacts/auto_chain.log
"""
from __future__ import annotations

import json
import msvcrt
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = Path(r"C:\Users\velmu\.local\bin\kaggle.exe")
STATE_PATH = ROOT / "artifacts" / "auto_chain_state.json"
LOG_PATH = ROOT / "artifacts" / "auto_chain.log"
STATUS_PATH = ROOT / "artifacts" / "auto_chain_status.txt"
LOCK_PATH = ROOT / "artifacts" / "auto_chain.lock"

PRETRAIN_ID = "velmurugan04/tamil-pretrain"
SFT_ID = "velmurugan04/tamil-sft-clean"
PRETRAIN_FOLDER = "kaggle"
SFT_FOLDER = "kaggle_sft_clean"
ACCELERATOR = "NvidiaTeslaT4"
KERNEL_TIMEOUT = 43200
POLL_SECONDS = 300
MAX_RUNTIME_HOURS = 30

SFT_DOWNLOAD = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Temp" / "opencode" / "kaggle-sft-clean"
SFT_MODEL_DIR = Path.home() / "models" / "tamil-350m-sft"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_status(message: str) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(f"{now()} | {message}\n", encoding="utf-8")


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "stage": "wait_v4",
        "pushed_v5_at": None,
        "pushed_sft_at": None,
        "completed_at": None,
        "last_status": None,
        "saw_running_v5": False,
        "saw_running_sft": False,
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def kernel_status(kernel_id: str) -> str:
    result = run([str(KAGGLE), "kernels", "status", kernel_id], timeout=120)
    text = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"KernelWorkerStatus\.(\w+)", text)
    if match:
        return match.group(1).upper()
    if result.returncode != 0:
        log(f"status command failed: {text.strip()[:400]}")
    return "UNKNOWN"


def push_kernel(folder: str) -> bool:
    result = run(
        [
            str(KAGGLE),
            "kernels",
            "push",
            "-p",
            str(ROOT / folder),
            "--accelerator",
            ACCELERATOR,
            "--timeout",
            str(KERNEL_TIMEOUT),
        ],
        timeout=600,
    )
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    if result.returncode != 0:
        log(f"PUSH FAILED ({folder}): {output[:1500]}")
        return False
    log(f"PUSH OK ({folder}): {output[:1500]}")
    return True


def save_logs(kernel_id: str, tag: str) -> Path:
    result = run([str(KAGGLE), "kernels", "logs", kernel_id], timeout=600)
    text = (result.stdout or "") + (result.stderr or "")
    destination = ROOT / "artifacts" / "logs" / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    log(f"saved logs to {destination} ({len(text)} chars)")
    return destination


def download_sft_output() -> Path:
    SFT_DOWNLOAD.mkdir(parents=True, exist_ok=True)
    result = run(
        [str(KAGGLE), "kernels", "output", SFT_ID, "-p", str(SFT_DOWNLOAD)],
        timeout=1800,
    )
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    log("SFT output download: " + output[-1200:])
    candidates = sorted(SFT_DOWNLOAD.glob("**/config.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"no config.json found under {SFT_DOWNLOAD}")
    model_dir = candidates[-1].parent
    log(f"SFT model directory: {model_dir}")
    return model_dir


def install_model(model_dir: Path) -> None:
    import shutil

    SFT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for item in model_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, SFT_MODEL_DIR / item.name)
    log(f"installed model to {SFT_MODEL_DIR}")


def evaluate_model(model_dir: Path) -> Path:
    destination = ROOT / "reports" / "tamil_reliability_sft_clean.json"
    result = run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_tamil_reliability.py"),
            "--model",
            str(model_dir),
            "--out",
            str(destination),
        ],
        timeout=3600,
    )
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    log("evaluation tail:\n" + output[-2000:])
    if result.returncode != 0:
        raise RuntimeError("evaluation failed")
    return destination


def hold(seconds: int) -> None:
    time.sleep(seconds)


def minutes_since(timestamp: str | None) -> float:
    if not timestamp:
        return 1e9
    try:
        moment = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - moment).total_seconds() / 60
    except Exception:
        return 1e9


def stale_complete(state: dict, stage: str) -> bool:
    """A COMPLETE status can still describe the previous run right after a push."""
    key = "pushed_v5_at" if stage == "wait_v5" else "pushed_sft_at"
    seen = "saw_running_v5" if stage == "wait_v5" else "saw_running_sft"
    return not state.get(seen) and minutes_since(state.get(key)) < 15


def main() -> int:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("w")
    try:
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("another auto_chain instance is already running", flush=True)
        return 0

    started = time.time()
    state = load_state()
    log(f"auto_chain start; stage={state.get('stage')}")

    while True:
        try:
            if (time.time() - started) > MAX_RUNTIME_HOURS * 3600:
                log("max runtime reached; exiting so the scheduler can restart cleanly")
                return 0

            stage = state.get("stage", "wait_v4")
            if stage == "done":
                log("chain already done")
                return 0

            if stage in ("wait_v4", "wait_v5"):
                status = kernel_status(PRETRAIN_ID)
                state["last_status"] = status
                save_state(state)
                write_status(f"{stage}: pretrain status {status}")
                log(f"{stage}: pretrain status {status}")

                if status in ("RUNNING", "QUEUED"):
                    if stage == "wait_v5":
                        state["saw_running_v5"] = True
                        save_state(state)
                    hold(POLL_SECONDS)
                    continue
                if status == "ERROR":
                    save_logs(PRETRAIN_ID, stage + "_error")
                    log(f"{stage}: pretrain ERROR; chain stopped")
                    write_status(f"{stage}: pretrain ERROR; see artifacts/logs")
                    return 2
                if status != "COMPLETE":
                    hold(POLL_SECONDS)
                    continue
                if stale_complete(state, stage):
                    log(f"{stage}: COMPLETE is still the previous run; waiting for the new one")
                    hold(60)
                    continue

                save_logs(PRETRAIN_ID, stage + "_complete")

                if stage == "wait_v4":
                    if not state.get("pushed_v5_at"):
                        state["pushed_v5_at"] = now()
                        save_state(state)
                        if not push_kernel(PRETRAIN_FOLDER):
                            state["pushed_v5_at"] = None
                            save_state(state)
                            hold(POLL_SECONDS)
                            continue
                    state["stage"] = "wait_v5"
                    save_state(state)
                    log("v4 complete; v5 pushed")
                    hold(30)
                    continue

                if stage == "wait_v5":
                    if not state.get("pushed_sft_at"):
                        state["pushed_sft_at"] = now()
                        save_state(state)
                        if not push_kernel(SFT_FOLDER):
                            state["pushed_sft_at"] = None
                            save_state(state)
                            hold(POLL_SECONDS)
                            continue
                    state["stage"] = "wait_sft"
                    save_state(state)
                    log("v5 complete; SFT pushed")
                    hold(30)
                    continue

            if stage == "wait_sft":
                status = kernel_status(SFT_ID)
                state["last_status"] = status
                save_state(state)
                write_status(f"wait_sft: status {status}")
                log(f"wait_sft: status {status}")

                if status in ("RUNNING", "QUEUED"):
                    state["saw_running_sft"] = True
                    save_state(state)
                    hold(POLL_SECONDS)
                    continue
                if status == "ERROR":
                    save_logs(SFT_ID, "sft_error")
                    log("SFT ERROR; chain stopped")
                    write_status("SFT ERROR; see artifacts/logs")
                    return 2
                if status != "COMPLETE":
                    hold(POLL_SECONDS)
                    continue
                if stale_complete(state, "wait_sft"):
                    log("wait_sft: COMPLETE is still the previous run; waiting for the new one")
                    hold(60)
                    continue

                save_logs(SFT_ID, "sft_complete")
                model_dir = download_sft_output()
                install_model(model_dir)
                report = evaluate_model(model_dir)
                state["stage"] = "done"
                state["completed_at"] = now()
                save_state(state)
                write_status(f"done; model at {SFT_MODEL_DIR}; report {report}")
                log(f"CHAIN DONE. model={SFT_MODEL_DIR} report={report}")
                return 0

            log(f"unknown stage {stage}; stopping to avoid a bad push")
            return 3

        except Exception:
            log("unhandled error:\n" + traceback.format_exc())
            write_status("error; see artifacts/auto_chain.log")
            hold(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
