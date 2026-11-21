"""Post-anonymization audit: re-read the output and look for residual leaks.

The auditor is independent of the de-identification step: it re-reads the
file written to disk, walks every element (including nested sequences and
file meta), and re-runs OCR on the output pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pydicom.dataset import Dataset

from .config import Config, DateMode, RedactMethod
from .phi import KnownIdentifiers, detect, find_patterns
from .pixel import OCREngine, UnsupportedPixelData, load_pixels, scan_array
from .metadata import is_standard_uid

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
HIGH_RISK_MODALITIES = {"US", "SC", "OT", "XC", "ES", "GM", "SM", "DOC", "IVUS", "OP"}
SCANNED_VRS = {"PN", "LO", "SH", "ST", "LT", "UT", "UC", "AE", "DA", "DT"}
PATTERN_VRS = {"LO", "SH", "ST", "LT", "UT", "UC"}


@dataclass
class Check:
    id: str
    status: str
    message: str
    details: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def overall(checks: list[Check]) -> str:
    statuses = {c.status for c in checks}
    return FAIL if FAIL in statuses else WARN if WARN in statuses else PASS


def walk(ds: Dataset, prefix: str = ""):
    for elem in ds:
        path = f"{prefix}{elem.keyword or str(elem.tag)}"
        yield path, elem
        if elem.VR == "SQ":
            for i, item in enumerate(elem.value or []):
                yield from walk(item, f"{path}[{i}].")


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return ""
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return "\\".join(str(v) for v in value)
    return str(value)


def audit_metadata(ds: Dataset, known: KnownIdentifiers | None, cfg: Config, *, original_uids: set[str] = frozenset(), pseudonyms: set[str] = frozenset()) -> list[Check]:
    checks: list[Check] = []
    if known is not None and cfg.date_mode == DateMode.KEEP:
        known = known.without_dates()

    leaks, pattern_hits, privates, overlays, uid_leaks = [], [], [], [], []
    elements = list(walk(ds))
    if getattr(ds, "file_meta", None) is not None:
        elements += list(walk(ds.file_meta))
    for path, elem in elements:
        vr = str(elem.VR)
        if elem.tag.is_private:
            privates.append(path)
        if 0x6000 <= elem.tag.group <= 0x601E and elem.tag.group % 2 == 0:
            overlays.append(path)
        if vr == "UI":
            vals = elem.value if isinstance(elem.value, (list, tuple)) or elem.value.__class__.__name__ == "MultiValue" else [elem.value]
            for v in vals:
                if v and str(v) in original_uids and not is_standard_uid(str(v)):
                    uid_leaks.append(path)
            continue
        if vr not in SCANNED_VRS:
            continue
        text = _text(elem.value)
        if not text.strip() or text in pseudonyms or path.startswith("DeidentificationMethod"):
            continue  # markers PixelGuard itself adds (method names, DCM code values)
        if known:
            for f in known.find(text):
                leaks.append({"path": path, "rule": f.rule})
        if vr in PATTERN_VRS:
            for f in find_patterns(text):
                if f.strength == "strong":
                    pattern_hits.append({"path": path, "category": f.category, "rule": f.rule})

    checks.append(
        Check("metadata.known_identifiers", FAIL, f"{len(leaks)} original identifier value(s) still present", leaks)
        if leaks
        else Check("metadata.known_identifiers", PASS, "no original identifier values found in any element")
    )
    checks.append(
        Check("metadata.phi_patterns", WARN, f"{len(pattern_hits)} text element(s) match PHI patterns; review", pattern_hits)
        if pattern_hits
        else Check("metadata.phi_patterns", PASS, "no free-text element matches PHI patterns")
    )
    if privates and not cfg.keep_private_tags:
        checks.append(Check("metadata.private_tags", FAIL, f"{len(privates)} private attribute(s) remain", privates[:50]))
    elif privates:
        checks.append(Check("metadata.private_tags", WARN, f"{len(privates)} private attribute(s) kept by configuration", privates[:50]))
    else:
        checks.append(Check("metadata.private_tags", PASS, "no private attributes"))
    if overlays:
        checks.append(Check("metadata.overlays", FAIL if cfg.remove_overlays else WARN, "overlay planes present (can contain burned-in annotations)", overlays))
    else:
        checks.append(Check("metadata.overlays", PASS, "no overlay planes"))
    checks.append(
        Check("metadata.uids", FAIL, f"{len(uid_leaks)} original UID(s) remain", uid_leaks)
        if uid_leaks
        else Check("metadata.uids", PASS, "all instance UIDs remapped")
    )
    removed = str(ds.get("PatientIdentityRemoved", "")).upper() == "YES"
    checks.append(Check("metadata.identity_removed_flag", PASS if removed else WARN, f"PatientIdentityRemoved = {ds.get('PatientIdentityRemoved', '(absent)')}"))
    return checks


def audit_pixels(ds: Dataset, known: KnownIdentifiers | None, cfg: Config, engine: OCREngine | None, *, pixel_result=None, original_scan=None, original_flags: dict | None = None) -> list[Check]:
    checks: list[Check] = []
    flags = original_flags or {}
    if "PixelData" not in ds:
        return [Check("pixel.residual_text", PASS, "no pixel data")]
    if engine is None or not cfg.verify_pixels:
        return [Check("pixel.residual_text", WARN, "pixel re-scan disabled; burned-in text not verified")]
    try:
        arr, _ = load_pixels(ds)
    except UnsupportedPixelData as exc:
        return [Check("pixel.residual_text", FAIL, f"pixel data could not be decoded ({exc}); burned-in text cannot be ruled out")]

    scan = scan_array(arr, ds, known, cfg, engine)
    leaks, unknown, low = [], [], []
    for d in scan.decisions:
        info = {"bbox": [d.box.x0, d.box.y0, d.box.x1, d.box.y1], "frame": d.box.frame, "categories": d.categories, "rules": d.rules, "confidence": round(d.box.confidence, 3)}
        if d.decision == "redact" and d.categories != ["UNRECOGNIZED_TEXT"]:
            leaks.append(info)
        elif "implausible-text-size" in d.rules:
            low.append(info)  # anatomy the OCR detector mistook for text
        elif d.decision in ("redact", "review") and d.box.confidence >= cfg.ocr_min_confidence:
            unknown.append(info)
        elif d.decision == "review":
            low.append(info)
    if leaks:
        extra = f" (+{len(unknown)} unrecognised)" if unknown else ""
        checks.append(Check("pixel.residual_text", FAIL, f"{len(leaks)} text region(s) matching PHI rules still readable in pixels{extra}", leaks + unknown))
    elif unknown:
        checks.append(Check("pixel.residual_text", WARN, f"{len(unknown)} unrecognised text region(s) remain; review", unknown))
    else:
        checks.append(Check("pixel.residual_text", PASS, "OCR re-scan found no identifying text" + (f" ({len(low)} low-confidence detection(s) ignored)" if low else ""), low))

    if pixel_result is not None and pixel_result.performed:
        checks.append(
            Check("pixel.integrity", PASS, "pixels outside redacted regions are bit-identical to the original")
            if pixel_result.integrity_ok
            else Check("pixel.integrity", FAIL, "pixels outside redacted regions changed")
        )
        if cfg.redact_method != RedactMethod.FILL and pixel_result.redacted_boxes:
            checks.append(Check("pixel.method", WARN, f"'{cfg.redact_method.value}' redaction can leave text partially legible; solid fill is recommended"))

    found_any = bool(original_scan and original_scan.decisions)
    if str(flags.get("BurnedInAnnotation", "")).upper() == "YES" and not found_any:
        checks.append(Check("risk.burned_in_flag", WARN, "header declares burned-in annotation but OCR found no text; review manually"))
    if flags.get("Modality") in HIGH_RISK_MODALITIES and not found_any:
        checks.append(Check("risk.modality", WARN, f"modality {flags.get('Modality')} commonly carries burned-in text but none was found; review manually"))
    if scan.n_frames > len(scan.frames_scanned):
        checks.append(Check("risk.frame_sampling", WARN, f"OCR sampled {len(scan.frames_scanned)} of {scan.n_frames} frames; text appearing only in other frames would be missed"))
    return checks
