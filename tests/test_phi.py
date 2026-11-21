import pytest
from pydicom.dataset import Dataset

from pixelguard.phi import detect, harvest, is_clinical_text, mask


@pytest.mark.parametrize(
    "text,category",
    [
        ("THORNBURY^AVERY", "NAME"),
        ("HOLLONAY ^ AMARA", "NAME"),
        ('DELACROIX "AHARA', "NAME"),
        ("Name: Avery Thornbury", "LABEL"),
        ("DOB: 03/14/1962", "DATE"),
        ("1962-03-14", "DATE"),
        ("14-Mar-1962", "DATE"),
        ("Jul 08, 2021", "DATE"),
        ("20210708", "DATE"),
        ("MRN: 27505051", "ID"),
        ("A7626386", "ID"),
        ("(555) 201-3344", "CONTACT"),
        ("123-45-6789", "SSN"),
        ("Age 94", "AGE_OVER_89"),
        ("Dr. Marchetti", "NAME"),
        ("Riverside General Hospital", "INSTITUTION"),
    ],
)
def test_patterns_detect_phi(text, category):
    cats = {f.category for f in detect(text)}
    assert category in cats, (text, cats)


@pytest.mark.parametrize("text", ["W:400 L:40", "Se:2 Im:15", "120 kVp", "5 cm", "R", "L", "AX T2 FLAIR", "FR 30Hz", "GAIN 50%", "3.5MHz", "PORTABLE", "x1.5", "12/45", "14:22:05", "TR 9000", "3.SMHz", "TE 120"])
def test_clinical_text_is_not_phi(text):
    assert is_clinical_text(text), text
    assert not [f for f in detect(text) if f.strength == "strong"], text


def _ds():
    ds = Dataset()
    ds.PatientName = "Thornbury^Avery^J"
    ds.PatientID = "27505051"
    ds.PatientBirthDate = "19620314"
    ds.AccessionNumber = "A7626386"
    ds.InstitutionName = "Riverside General Hospital"
    ds.ReferringPhysicianName = "Marchetti^Sam"
    return ds


@pytest.mark.parametrize(
    "ocr_text",
    [
        "Avery Thornbury",
        "THORNBURV",  # OCR slip
        "ID 275O5O51",  # O for 0
        "03/14/1962",
        "14 Mar 1962",
        "Riverside General Hosp1tal",
        "MARCHETTI",
    ],
)
def test_known_identifiers_tolerate_ocr_noise(ocr_text):
    known = harvest(_ds())
    assert known.find(ocr_text), ocr_text


def test_known_identifiers_ignore_placeholders_and_unrelated_text():
    ds = Dataset()
    ds.PatientName = "ANONYMOUS"
    ds.PatientID = "ANON0001"
    known = harvest(ds)
    assert not known.find("ANONYMOUS")
    assert not harvest(_ds()).find("W:400 L:40")


def test_mask_preview():
    assert mask("DOE^JANE") == "D**^J***"
