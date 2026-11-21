import copy
import re

import pydicom
import pytest

from pixelguard.config import Config, DateMode
from pixelguard.metadata import MetadataDeidentifier
from pixelguard.phi import harvest
from pixelguard.verify import audit_metadata, walk


def _compact(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _text(ds):
    return _compact(" ".join(str(e.value) for _, e in walk(ds) if e.VR not in ("SQ", "OB", "OW", "UN")))


def _run(ds, cfg):
    known = harvest(ds)
    res = MetadataDeidentifier(cfg).run(ds, known, pixel_cleaned=True)
    return res, known


@pytest.fixture
def full_files(synth_dir, truths):
    return [synth_dir / f for f, t in sorted(truths.items()) if not t["naive_anonymized"]]


def test_all_injected_metadata_phi_removed(full_files, truths):
    cfg = Config()
    for f in full_files:
        ds = pydicom.dcmread(str(f))
        _run(ds, cfg)
        text = _text(ds)
        for m in truths[f.name]["metadata_phi"]:
            if m["value"] == "__overlay__":
                assert not any(0x6000 <= e.tag.group <= 0x601E for _, e in walk(ds))
            else:
                assert _compact(m["value"]) not in text, (f.name, m["keyword"])
        assert not any(e.tag.is_private for _, e in walk(ds))


def test_audit_passes_after_metadata_cleaning(full_files):
    cfg = Config()
    for f in full_files:
        ds = pydicom.dcmread(str(f))
        res, known = _run(ds, cfg)
        checks = {c.id: c.status for c in audit_metadata(ds, known, cfg, original_uids=res.original_uids, pseudonyms=res.pseudonyms)}
        assert checks["metadata.known_identifiers"] == "PASS"
        assert checks["metadata.private_tags"] == "PASS"
        assert checks["metadata.uids"] == "PASS"
        assert checks["metadata.overlays"] == "PASS"


def test_audit_catches_untouched_header(full_files):
    cfg = Config()
    ds = pydicom.dcmread(str(full_files[0]))
    checks = {c.id: c.status for c in audit_metadata(ds, harvest(ds), cfg)}
    assert checks["metadata.known_identifiers"] == "FAIL"
    assert checks["metadata.private_tags"] == "FAIL"


def test_uids_and_pseudonyms_are_consistent_and_keyed(full_files):
    a = pydicom.dcmread(str(full_files[0]))
    b, c = copy.deepcopy(a), copy.deepcopy(a)
    orig_study = a.StudyInstanceUID
    _run(a, Config(secret_key=b"k1" * 16))
    _run(b, Config(secret_key=b"k1" * 16))
    _run(c, Config(secret_key=b"k2" * 16))
    assert a.StudyInstanceUID == b.StudyInstanceUID != orig_study
    assert a.StudyInstanceUID != c.StudyInstanceUID
    assert a.PatientID == b.PatientID != c.PatientID
    assert a.file_meta.MediaStorageSOPInstanceUID == a.SOPInstanceUID
    assert a.SOPClassUID == pydicom.dcmread(str(full_files[0])).SOPClassUID  # standard UIDs kept
    assert len(a.SOPInstanceUID) <= 64


def test_date_modes(full_files):
    src = pydicom.dcmread(str(full_files[0]))
    rm = copy.deepcopy(src)
    _run(rm, Config(date_mode=DateMode.REMOVE))
    assert rm.get("StudyDate", "") == "" and "SeriesDate" not in rm and rm.get("PatientBirthDate", "") == ""

    cfg = Config(date_mode=DateMode.SHIFT, secret_key=b"x" * 32)
    s1 = copy.deepcopy(src)
    res, _ = _run(s1, cfg)
    assert s1.StudyDate != src.StudyDate and len(s1.StudyDate) == 8
    assert -3650 <= res.date_offset_days <= -365
    from datetime import datetime

    d = lambda v: datetime.strptime(v, "%Y%m%d")
    # intervals between dates are preserved
    assert (d(src.StudyDate) - d(src.PatientBirthDate)) == (d(s1.StudyDate) - d(s1.PatientBirthDate))


def test_same_patient_gets_same_shift(full_files):
    cfg = Config(date_mode=DateMode.SHIFT, secret_key=b"y" * 32)
    offsets = {}
    for f in full_files:
        ds = pydicom.dcmread(str(f))
        pid = ds.PatientID
        res, _ = _run(ds, cfg)
        offsets.setdefault(pid, set()).add(res.date_offset_days)
    assert all(len(v) == 1 for v in offsets.values())


def test_clinical_attributes_and_characteristics_kept(full_files):
    src = pydicom.dcmread(str(full_files[0]))
    ds = copy.deepcopy(src)
    _run(ds, Config())
    for kw in ("Modality", "Rows", "Columns", "PatientSex", "PatientAge", "BodyPartExamined", "PixelSpacing"):
        if kw in src:
            assert kw in ds and ds.data_element(kw).value == src.data_element(kw).value, kw
    assert ds.PatientIdentityRemoved == "YES"
    assert any(i.CodeValue == "113100" for i in ds.DeidentificationMethodCodeSequence)


def test_change_log_never_contains_raw_values(full_files):
    ds = pydicom.dcmread(str(full_files[0]))
    name = str(ds.PatientName)
    res, _ = _run(ds, Config())
    blob = repr([c.to_dict() for c in res.changes])
    assert name not in blob and str(name).split("^")[0] not in blob


def test_age_over_89_generalized():
    from pydicom.dataset import Dataset

    ds = Dataset()
    ds.PatientAge = "093Y"
    ds.PatientName = "X^Y"
    _run(ds, Config())
    assert ds.PatientAge == "090Y"
