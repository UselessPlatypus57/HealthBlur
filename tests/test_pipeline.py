import json

import pydicom

from pixelguard.config import Config, TextPolicy
from pixelguard.evaluate import evaluate_synthetic
from pixelguard.pipeline import discover, process_file, run_batch
from pixelguard.report import write_reports


def test_full_pipeline_redacts_all_phi(synth_dir, gt_engine, tmp_path):
    inputs = discover([synth_dir])
    rep = run_batch(inputs, tmp_path / "out", Config(secret_key=b"t" * 32), engine=gt_engine)
    assert rep["summary"]["status_counts"]["FAIL"] == 0, [r["checks"] for r in rep["files"] if r["status"] == "FAIL"]
    assert rep["summary"]["status_counts"]["ERROR"] == 0, [r.get("traceback") for r in rep["files"]]
    ev = evaluate_synthetic(synth_dir, rep, tmp_path / "out", inputs)
    assert ev["pixel_phi_recall"] == 1.0
    assert ev["pixel_false_positive_rate"] <= 0.1
    assert ev["metadata_phi_recall"] == 1.0
    assert ev["clinical_attribute_retention"] >= 0.95
    for r in rep["files"]:
        assert (tmp_path / "out" / r["output"]).exists()
        assert any(c["id"] == "pixel.integrity" and c["status"] == "PASS" for c in r["checks"])
    paths = write_reports(rep, tmp_path / "out", ev)
    html = paths["html"].read_text(encoding="utf-8")
    assert "Not a guarantee of de-identification" in html


def test_reports_contain_no_original_identifiers(synth_dir, gt_engine, truths, tmp_path):
    inputs = discover([synth_dir])
    rep = run_batch(inputs, tmp_path / "out", Config(), engine=gt_engine)
    paths = write_reports(rep, tmp_path / "rep")
    blob = paths["json"].read_text(encoding="utf-8").upper() + paths["html"].read_text(encoding="utf-8").upper()
    for t in truths.values():
        for s in t["known_phi_strings"]:
            assert s.upper() not in blob, s


def test_metadata_only_on_naive_dataset_is_caught_and_quarantined(synth_dir, gt_engine, truths, tmp_path):
    naive = [synth_dir / f for f, t in truths.items() if t["naive_anonymized"]]
    assert naive, "fixture should contain header-only anonymized files"
    rep = run_batch(sorted(naive), tmp_path / "out", Config(clean_pixels=False), engine=gt_engine)
    for r in rep["files"]:
        assert r["status"] == "FAIL"
        residual = next(c for c in r["checks"] if c["id"] == "pixel.residual_text")
        assert residual["status"] == "FAIL"
        assert "NAME" in {cat for d in residual["details"] for cat in d["categories"]} | {"KNOWN"} or residual["details"]
        assert r["output"].startswith("_quarantine/")
        assert (tmp_path / "out" / r["output"]).exists()


def test_targeted_policy_keeps_more_clinical_text(synth_dir, gt_engine, tmp_path):
    inputs = discover([synth_dir])
    rep = run_batch(inputs, tmp_path / "t", Config(text_policy=TextPolicy.TARGETED), engine=gt_engine)
    ev = evaluate_synthetic(synth_dir, rep, tmp_path / "t", inputs)
    assert ev["pixel_false_positive_rate"] == 0.0
    assert ev["pixel_phi_recall"] >= 0.9


def test_non_dicom_fails_closed(tmp_path, gt_engine):
    bad = tmp_path / "junk.dcm"
    bad.write_bytes(b"not a dicom file at all" * 10)
    rec = process_file(bad, tmp_path / "out", Config(), gt_engine)
    assert rec["status"] == "ERROR"
    assert not any((tmp_path / "out").rglob("*.dcm")) if (tmp_path / "out").exists() else True


def test_compressed_input_is_written_uncompressed(synth_dir, gt_engine, truths, tmp_path):
    comp = [f for f, t in truths.items() if t["compressed"]]
    assert comp
    rec = process_file(synth_dir / comp[0], tmp_path / "out", Config(), gt_engine)
    assert rec["status"] in ("PASS", "WARN"), rec["checks"]
    ds = pydicom.dcmread(str(tmp_path / "out" / rec["output"]))
    assert not ds.file_meta.TransferSyntaxUID.is_compressed
    ds.pixel_array  # decodes


def test_output_paths_contain_no_phi(synth_dir, gt_engine, truths, tmp_path):
    rep = run_batch(discover([synth_dir]), tmp_path / "out", Config(), engine=gt_engine)
    for r in rep["files"]:
        for t in truths.values():
            assert t["patient_key"] not in r["output"]
