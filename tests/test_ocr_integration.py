"""Real EasyOCR end-to-end check. Slow; run with ``pytest -m ocr``."""

import pytest

from pixelguard.config import Config
from pixelguard.evaluate import evaluate_synthetic
from pixelguard.pipeline import discover, run_batch
from pixelguard.synth import generate


@pytest.mark.ocr
def test_easyocr_end_to_end(tmp_path):
    pytest.importorskip("easyocr")
    data = tmp_path / "data"
    generate(data, n=6, seed=3, hard_fraction=0.0)
    inputs = discover([data])
    rep = run_batch(inputs, tmp_path / "out", Config())
    ev = evaluate_synthetic(data, rep, tmp_path / "out", inputs)
    assert ev["pixel_phi_recall"] >= 0.9, ev
    assert ev["metadata_phi_recall"] == 1.0, ev
    assert rep["summary"]["status_counts"]["ERROR"] == 0
