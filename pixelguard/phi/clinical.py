"""Vocabulary of burned-in text that is clinically useful and not identifying.

Text made only of these tokens (orientation markers, units, acquisition
parameters, anatomy) is preserved by the conservative policy, so radiologists
keep left/right markers, scale annotations and technique labels.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from functools import lru_cache

CLINICAL_TOKENS: frozenset[str] = frozenset(
    """
    L R A P S I H F LT RT LAT LATERAL AP PA OBL OBLIQUE AX AXIAL COR CORONAL SAG SAGITTAL
    TRV TRANS TRANSVERSE LONG LONGITUDINAL PORTABLE PORT SUPINE PRONE ERECT UPRIGHT DECUB
    DECUBITUS STANDING SITTING INSP EXP INSPIRATION EXPIRATION VIEW MARKER SCALE
    CT MR MRI US XR CR DX MG PET NM SPECT RF XA
    T1 T2 FLAIR DWI ADC SWI STIR PD GRE SE FSE TSE MPRAGE TOF DTI EPI B0 B1000 FS FATSAT
    TR TE NEX ETL BW FA NSA MATRIX
    TGC GAIN DR DEPTH FREQ FR HZ MHZ KHZ FPS MI TI TIS TIB TIC PRF WF CFM PWR PW CW
    DOPPLER COLOR COLOUR B M 2D 3D 4D HARM THI ZOOM FOCUS FOC DB DYN PERS MAP CINE
    MM CM KV KVP MA MAS MS SEC SL SLICE THK THICKNESS FOV ROT TILT W WL WW C LEVEL WINDOW
    HU IMG IM IMAGE SER SERIES FRAME EX KERNEL FILTER GRID MAG
    CHEST HEAD BRAIN ABD ABDOMEN PELVIS SPINE CSPINE TSPINE LSPINE KNEE HIP SHOULDER
    HAND FOOT ANKLE WRIST ELBOW NECK LIVER KIDNEY RENAL GB THYROID HEART CARDIAC BREAST
    LUNG LUNGS LEFT RIGHT UPPER LOWER MID PROX DIST ANT POST MED
    CONTRAST NONCON WITH WO W/O WITHOUT PRE POST ART VEN DELAY PHASE
    OF AND THE X Y Z
    """.split()
)

_UNITS = r"(?:MM|CM|M|KV|KVP|MA|MAS|MS|S|SEC|HZ|KHZ|MHZ|DB|FPS|HU|%|DEG|X|CM/S|MM/S|BPM)"
_CLINICAL_TOKEN_PATTERNS = [
    re.compile(r"^[+-]?\d{1,4}(?:[.,]\d+)?$"),  # short numbers (not IDs)
    re.compile(rf"^[+-]?\d{{1,4}}(?:[.,]\d+)?{_UNITS}$"),  # number with unit, e.g. 3.5MHZ
    re.compile(r"^\d{1,4}/\d{1,4}$"),  # image 12/45
    re.compile(r"^[+\-*/<>=#|.,:;~_()\[\]]+$"),  # punctuation only
    re.compile(r"^X\d+(?:[.,]\d+)?$"),  # magnification x1.5
    re.compile(r"^(?:SE|IM|SER|IMG|SL|EX|FR|SN)[.:#]?\d{1,4}$"),  # Se.4, Im12 (OCR-merged labels)
]

_TOKEN_SPLIT = re.compile(r"[\s:=]+")


def tokens(text: str) -> list[str]:
    junk = "()[],;_'\"`/|"
    return [t.strip(junk) for t in _TOKEN_SPLIT.split(text.upper()) if t.strip(junk)]


_NUMERIC_SLIP = re.compile(r"^([0-9SOILZ.,]+?)(MHZ|KHZ|HZ|MM|CM|KVP|KV|MAS|MA|MS|DB|FPS|%)?$")
_DIGIT_FIX = str.maketrans({"S": "5", "O": "0", "I": "1", "L": "1", "Z": "2"})
# Labels that are followed by a short number ("Se:4", "W:400", "GAIN 50").
_NUMBER_LABELS = frozenset("SE IM SER IMG SL EX FR DR TR TE TI W L C WL WW GAIN DEPTH FOV THK".split())
_FUZZY_VOCAB = sorted(t for t in CLINICAL_TOKENS if len(t) >= 4)


@lru_cache(maxsize=4096)
def is_clinical_token(tok: str) -> bool:
    tok = tok.upper()
    if tok in CLINICAL_TOKENS:
        return True
    if any(p.match(tok) for p in _CLINICAL_TOKEN_PATTERNS):
        return True
    # OCR digit slips inside measurements: 3.SMHZ -> 3.5MHZ, 1O CM -> 10CM
    m = _NUMERIC_SLIP.match(tok)
    if m and (any(ch.isdigit() for ch in m.group(1)) or m.group(2)):
        fixed = m.group(1).translate(_DIGIT_FIX) + (m.group(2) or "")
        if fixed != tok and any(p.match(fixed) for p in _CLINICAL_TOKEN_PATTERNS):
            return True
    if tok.startswith("X") and len(tok) > 1 and set(tok[1:]) <= set("0123456789SOILZ.,"):
        if any(p.match("X" + tok[1:].translate(_DIGIT_FIX)) for p in _CLINICAL_TOKEN_PATTERNS):
            return True  # xl.5 -> x1.5
    # OCR dropped a space: "AXT2" = "AX" + "T2". Only for tokens containing a
    # digit, so that no name can be spelled out of short clinical words.
    if any(ch.isdigit() for ch in tok) and _splits_into_clinical(tok):
        return True
    # tolerate one OCR slip in longer clinical words (FLAR -> FLAIR)
    return len(tok) >= 4 and tok.isalpha() and bool(get_close_matches(tok, _FUZZY_VOCAB, n=1, cutoff=0.85))


_COMPOUND_PARTS = frozenset(t for t in CLINICAL_TOKENS if len(t) >= 2)


def _splits_into_clinical(tok: str, max_parts: int = 3) -> bool:
    """True if ``tok`` is a concatenation of 2..max_parts clinical tokens (each
    at least two characters) or short numbers."""

    def part_ok(p: str) -> bool:
        return p in _COMPOUND_PARTS or (p.isdigit() and len(p) <= 4)

    def rec(s: str, parts: int) -> bool:
        if not s:
            return parts >= 2
        if parts == max_parts:
            return False
        return any(part_ok(s[:i]) and rec(s[i:], parts + 1) for i in range(2 if not s[0].isdigit() else 1, len(s) + 1))

    return len(tok) <= 12 and rec(tok, 0)


def is_clinical_text(text: str) -> bool:
    """True if every token in ``text`` is a recognised clinical annotation."""
    toks = tokens(text)
    if not toks:
        return True
    for i, t in enumerate(toks):
        if is_clinical_token(t):
            continue
        # "Im:lz" -> "Im:12": a short digit-like token right after a number label
        if i and toks[i - 1] in _NUMBER_LABELS and len(t) <= 4 and t.translate(_DIGIT_FIX).isdigit():
            continue
        return False
    return True
