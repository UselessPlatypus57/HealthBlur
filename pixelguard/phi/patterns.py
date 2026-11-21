"""Generic PHI patterns for free text (OCR output and DICOM text fields).

These rules work without knowing who the patient is, which matters for data
that already *looks* anonymized: the header says "ANONYMOUS" but the pixels
still carry the real name. They are deliberately recall-oriented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .clinical import CLINICAL_TOKENS


@dataclass(frozen=True)
class Finding:
    category: str  # NAME, DATE, ID, CONTACT, SSN, AGE_OVER_89, INSTITUTION, LABEL, KNOWN
    rule: str  # rule identifier, e.g. "pattern:caret-name" or "known:PatientName"
    start: int
    end: int
    strength: str = "strong"  # strong | weak


_MONTHS = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*"

_RULES: list[tuple[str, str, re.Pattern[str], str]] = [
    # (category, rule name, regex, strength); regexes run on upper-cased text
    ("SSN", "ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "strong"),
    ("CONTACT", "phone", re.compile(r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "strong"),
    ("CONTACT", "email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "strong"),
    ("DATE", "date-ymd", re.compile(r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b"), "strong"),
    ("DATE", "date-dmy-mdy", re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|[12]\d|3[01])[-/.](?:19|20)?\d{2}\b"), "strong"),
    ("DATE", "date-compact", re.compile(r"\b(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\b"), "strong"),
    ("DATE", "date-text", re.compile(rf"\b\d{{1,2}}[-\s]?{_MONTHS}[-\s,]*(?:19|20)?\d{{2}}\b"), "strong"),
    ("DATE", "date-text-mdy", re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b"), "strong"),
    ("NAME", "caret-name", re.compile(r"\b[A-Z][A-Z'\-]+\^[A-Z][A-Z'\-]*(?:\^[A-Z'\-]*)*"), "strong"),
    # OCR often renders the DICOM "^" separator as " ^ ", a quote or a tilde.
    ("NAME", "caret-name-ocr", re.compile(r"\b[A-Z][A-Z'\-]{1,}\s*[\^\"`~*]\s*[A-Z][A-Z'\-]{1,}\b"), "strong"),
    ("NAME", "titled-name", re.compile(r"\b(?:MR|MRS|MS|MISS|DR|PROF)\.?\s+[A-Z][A-Z'\-]{1,}(?:\s+[A-Z][A-Z'\-]+)?"), "strong"),
    ("AGE_OVER_89", "age-over-89", re.compile(r"\b(?:9\d|1[0-4]\d)\s*(?:Y|YR|YRS|YO|YEARS?)\b|\bAGE\W{0,3}(?:9\d|1[0-4]\d)\b"), "strong"),
    ("ID", "id-number", re.compile(r"\b(?=[A-Z0-9\-]*\d{5})[A-Z]{0,4}-?\d[\dA-Z\-]{4,}\b"), "strong"),
    ("INSTITUTION", "institution", re.compile(r"\b(?:HOSPITAL|MEDICAL\s+CENTER|MED\.?\s*CTR|CLINIC|INFIRMARY|HEALTH\s*CARE|HEALTH\s+SYSTEM|UNIVERSITY|INSTITUTE|IMAGING\s+CENTER|RADIOLOGY\s+ASSOC\w*)\b"), "strong"),
]

# Labels that announce an identifier: the label and whatever follows it on
# the same line are treated as PHI.
LABEL_RE = re.compile(
    r"\b(?:PATIENT\s*NAME|PATIENT|PAT|PT|NAME|NOM|DOB|D\.O\.B|BIRTH\s*DATE|BIRTHDATE|BORN|MRN|MR#|MR\s*NO|"
    r"PID|PAT\s*ID|PATIENT\s*ID|ID|ACC|ACCESSION|ACC#|SSN|PHONE|TEL|ADDR|ADDRESS|PHYSICIAN|REF\s*PHYS|"
    r"REFERRING|DOCTOR|OPERATOR|TECH|HOSP|HOSPITAL|INSTITUTION|STUDY\s*DATE|EXAM\s*DATE|DATE)\s*[:#]"
)

_LAST_FIRST = re.compile(r"^\s*([A-Z][A-Z'\-]+),\s*([A-Z][A-Z'\-]+)(?:\s+[A-Z]\.?)?\s*$")
_FIRST_LAST = re.compile(r"^\s*([A-Z][A-Z'\-]{1,})\s+([A-Z][A-Z'\-]{2,})\s*$")


def find_patterns(text: str) -> list[Finding]:
    up = text.upper()
    out: list[Finding] = []
    for category, name, rx, strength in _RULES:
        for m in rx.finditer(up):
            if category == "ID" and _looks_like_measurement(m.group(0)):
                continue
            out.append(Finding(category, f"pattern:{name}", m.start(), m.end(), strength))
    for m in LABEL_RE.finditer(up):
        out.append(Finding("LABEL", "pattern:phi-label", m.start(), len(up), "strong"))
    m = _LAST_FIRST.match(up)
    if m and not (m.group(1) in CLINICAL_TOKENS and m.group(2) in CLINICAL_TOKENS):
        out.append(Finding("NAME", "pattern:last-comma-first", m.start(), m.end(), "weak"))
    return out


def find_name_like(text: str) -> list[Finding]:
    """Two capitalised words that are not clinical vocabulary ("Jane Doe").

    Weak on its own; used by the targeted policy only together with other
    evidence, and by the auditor as a review flag.
    """
    up = text.upper()
    m = _FIRST_LAST.match(up)
    if m and m.group(1) not in CLINICAL_TOKENS and m.group(2) not in CLINICAL_TOKENS:
        return [Finding("NAME", "pattern:two-word-name", m.start(), m.end(), "weak")]
    return []


def _looks_like_measurement(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}(?:[.,]\d+)?(?:MM|CM|HZ|KHZ|MHZ|MS|KV|MA|MAS)", s))
