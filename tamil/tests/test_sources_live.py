"""Live check that every source loads and has its text fields. Run on Kaggle: RUN_NETWORK=1 pytest ..."""
import os
from itertools import islice

import pytest

from tamil import sources
from tamil.corpus import load_rows

pytestmark = pytest.mark.skipif(os.environ.get("RUN_NETWORK") != "1", reason="network test; set RUN_NETWORK=1")

HF_SOURCES = [s for s in sources.PRETRAIN + sources.SFT + sources.HELDOUT + sources.EVAL if not s.url]


@pytest.mark.parametrize("src", HF_SOURCES, ids=lambda s: s.hf_id)
def test_source_loads_with_fields(src):
    pytest.importorskip("datasets")
    rows = list(islice(load_rows(src), 5))
    assert len(rows) == 5
    for f in src.fields:
        assert any(r.get(f) for r in rows), f"{src.hf_id}: field {f!r} empty or missing; got keys {list(rows[0])}"
