import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _eval_script():
    spec = importlib.util.spec_from_file_location("eval04", ROOT / "kaggle" / "04_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_states_params_not_same_size(tmp_path):
    ev = _eval_script()
    ev.REPORTS = tmp_path
    row = {"bpb": {"tamil": 1.0, "tanglish_real": 2.0, "english": 1.5}, "params": 23_400_000,
           **{t: 0.5 for t in ev.TASKS}}
    text = ev.write_results({"ablation-ours": row, "ablation-llama3": {**row, "params": 60_100_000}}).read_text("utf-8")
    assert "same model size" not in text
    assert "23.4M" in text and "60.1M" in text
