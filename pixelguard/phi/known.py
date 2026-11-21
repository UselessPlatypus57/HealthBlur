"""Identifiers harvested from a file's own header before it is anonymized.

Most burned-in PHI repeats what the header says (name, MRN, birth date,
accession). Harvesting those values first lets the pixel stage and the
auditor search for the *specific* strings, tolerating OCR errors.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from pydicom.dataset import Dataset

from .clinical import CLINICAL_TOKENS
from .patterns import Finding

_PLACEHOLDERS = {
    "", "ANONYMOUS", "ANONYMIZED", "ANON", "UNKNOWN", "NONE", "NULL", "N/A", "NA", "REMOVED",
    "DELETED", "PATIENT", "TEST", "XXX", "0", "00000000", "19000101",
}

# OCR commonly confuses these glyphs in identifiers.
_OCR_DIGIT_MAP = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})

NAME_TAGS = (
    "PatientName", "OtherPatientNames", "PatientBirthName", "PatientMotherBirthName",
    "ReferringPhysicianName", "PerformingPhysicianName", "NameOfPhysiciansReadingStudy",
    "OperatorsName", "PhysiciansOfRecord", "RequestingPhysician", "ResponsiblePerson",
)
ID_TAGS = ("PatientID", "OtherPatientIDs", "AccessionNumber", "StudyID", "AdmissionID", "DeviceSerialNumber", "StationName", "MedicalRecordLocator")
DATE_TAGS = ("PatientBirthDate", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate")
TEXT_TAGS = ("InstitutionName", "InstitutionAddress", "PatientAddress", "PatientTelephoneNumbers", "InstitutionalDepartmentName", "ReferringPhysicianAddress")


@dataclass(frozen=True)
class KnownValue:
    value: str  # normalised form used for matching
    source: str  # DICOM keyword it came from
    kind: str  # name_token | name_full | id | date | text


@dataclass
class KnownIdentifiers:
    values: list[KnownValue] = field(default_factory=list)

    def add(self, value: str, source: str, kind: str) -> None:
        kv = KnownValue(value, source, kind)
        if value and kv not in self.values:
            self.values.append(kv)

    def without_dates(self) -> "KnownIdentifiers":
        return KnownIdentifiers([v for v in self.values if v.kind != "date"])

    def merge(self, other: "KnownIdentifiers") -> None:
        for v in other.values:
            if v not in self.values:
                self.values.append(v)

    def __len__(self) -> int:
        return len(self.values)

    def find(self, text: str) -> list[Finding]:
        """Return a finding per known identifier present in ``text``."""
        if not text or not self.values:
            return []
        norm = _norm(text)
        compact = _compact(text)
        digits = compact.translate(_OCR_DIGIT_MAP)
        toks = norm.split()
        out: list[Finding] = []
        seen: set[str] = set()
        for kv in self.values:
            hit = False
            if kv.kind == "name_token":
                hit = any(t == kv.value or (len(kv.value) >= 5 and len(t) >= 4 and _similar(t, kv.value) >= 0.84) for t in toks)
            elif kv.kind in ("name_full", "text"):
                hit = len(kv.value) >= 4 and (kv.value in compact or (len(kv.value) >= 8 and _fuzzy_contains(compact, kv.value, 0.86)))
            elif kv.kind in ("id", "date"):
                target = kv.value.translate(_OCR_DIGIT_MAP)
                hit = len(target) >= 4 and target in digits
            if hit and (kv.source, kv.kind) not in seen:
                seen.add((kv.source, kv.kind))
                out.append(Finding("KNOWN", f"known:{kv.source}", 0, len(text), "strong"))
        return out


def harvest(ds: Dataset) -> KnownIdentifiers:
    known = KnownIdentifiers()
    for kw in NAME_TAGS:
        for val in _values(ds, kw):
            _add_name(known, val, kw)
    for kw in ID_TAGS:
        for val in _values(ds, kw):
            c = _compact(val)
            if c not in _PLACEHOLDERS and len(c) >= 5:
                known.add(c, kw, "id")
    for kw in DATE_TAGS:
        for val in _values(ds, kw):
            for fmt in date_variants(val):
                known.add(fmt, kw, "date")
    for kw in TEXT_TAGS:
        for val in _values(ds, kw):
            c = _compact(val)
            if c not in _PLACEHOLDERS and len(c) >= 6:
                known.add(c, kw, "text")
    # Sequences (e.g. OtherPatientIDsSequence) hold more IDs.
    for item in getattr(ds, "OtherPatientIDsSequence", []) or []:
        for val in _values(item, "PatientID"):
            c = _compact(val)
            if c not in _PLACEHOLDERS and len(c) >= 5:
                known.add(c, "OtherPatientIDsSequence", "id")
    return known


def _add_name(known: KnownIdentifiers, raw: str, source: str) -> None:
    parts = [p for p in re.split(r"[\^=]", raw) if p.strip()]
    tokens = [t for p in parts for t in _norm(p).split()]
    full = _compact(raw)
    if full in _PLACEHOLDERS:
        return
    if len(full) >= 4:
        known.add(full, source, "name_full")
    for t in tokens:
        if len(t) >= 3 and t not in _PLACEHOLDERS and t not in CLINICAL_TOKENS and t not in {"MR", "MRS", "MS", "DR", "JR", "SR", "III"}:
            known.add(t, source, "name_token")


def date_variants(da: str) -> list[str]:
    """Compact (alphanumeric-only) renderings of a DICOM DA value."""
    da = da.strip()[:8]
    try:
        d = _dt.datetime.strptime(da, "%Y%m%d").date()
    except ValueError:
        return []
    if da in _PLACEHOLDERS:
        return []
    fmts = ["%Y%m%d", "%m%d%Y", "%d%m%Y", "%d%b%Y", "%b%d%Y", "%m%d%y", "%d%m%y", "%d%B%Y", "%B%d%Y"]
    out = {d.strftime(f).upper() for f in fmts}
    # Non-zero-padded US style, e.g. 3/4/1962 -> 341962
    out.add(f"{d.month}{d.day}{d.year}")
    out.add(f"{d.day}{d.month}{d.year}")
    return sorted(out)


def _values(ds: Dataset, keyword: str) -> list[str]:
    if keyword not in ds:
        return []
    v = ds.data_element(keyword).value
    if v is None:
        return []
    if isinstance(v, (list, tuple)) or v.__class__.__name__ == "MultiValue":
        return [str(x) for x in v if str(x).strip()]
    s = str(v)
    return [s] if s.strip() else []


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", s.upper()).strip()


def _compact(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", s.upper())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _fuzzy_contains(hay: str, needle: str, threshold: float) -> bool:
    n = len(needle)
    if len(hay) < n - 2:
        return False
    for i in range(0, max(1, len(hay) - n + 3)):
        if _similar(hay[i : i + n], needle) >= threshold:
            return True
    return False
