import pydicom
from pydicom.dataset import Dataset

from pixelguard.cli import _header_identity_check, build_parser


def test_header_check_accepts_pseudonyms_and_flags_real_values():
    ds = Dataset()
    ds.PatientName = "ANON^1A2B3C4D5E"
    ds.PatientID = "ANON1A2B3C4D5E"
    ds.PatientIdentityRemoved = "YES"
    assert _header_identity_check(ds).status == "PASS"
    ds.PatientName = "Thornbury^Avery"
    assert _header_identity_check(ds).status == "FAIL"


def test_parser_has_all_commands():
    p = build_parser()
    for cmd in ("deidentify", "scan", "audit", "synth", "benchmark", "evaluate-external", "demo", "serve"):
        assert p.parse_args([cmd] + {"deidentify": ["x", "-o", "y"], "scan": ["x"], "audit": ["x"], "synth": ["x"],
                                     "evaluate-external": ["x", "-o", "y", "--known-csv", "k"]}.get(cmd, [])).cmd == cmd


def test_audit_command_passes_on_clean_output(synth_dir, gt_engine, tmp_path, monkeypatch):
    from pixelguard import cli
    from pixelguard.config import Config
    from pixelguard.pipeline import discover, run_batch

    run_batch(discover([synth_dir]), tmp_path / "out", Config(), engine=gt_engine)
    monkeypatch.setattr("pixelguard.pixel.get_engine", lambda *a, **k: gt_engine)
    code = cli.main(["audit", str(tmp_path / "out"), "--report-dir", str(tmp_path / "aud"), "--no-pixels"])
    assert code == 0


def test_audit_metadata_ignores_pseudonyms_without_context():
    from pixelguard.config import Config
    from pixelguard.verify import audit_metadata

    ds = Dataset()
    ds.PatientID = "ANON4F31234590"  # 5+ digits: looks like an ID number
    ds.PatientName = "ANON^4F31234590"
    ds.PatientIdentityRemoved = "YES"
    checks = {c.id: c.status for c in audit_metadata(ds, None, Config())}
    assert checks["metadata.phi_patterns"] == "PASS"
