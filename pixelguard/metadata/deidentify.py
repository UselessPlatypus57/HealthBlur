"""Metadata de-identification with a full change log."""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field

from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.uid import UID_dictionary

from ..config import Config, DateMode
from ..phi import KnownIdentifiers, detect, mask
from .rules import KEEP_UID_KEYWORDS, RULES, TEXT_VRS, TYPE2_EMPTY


@dataclass
class TagChange:
    path: str
    tag: str
    keyword: str
    vr: str
    action: str  # removed | emptied | pseudonymized | uid-remapped | date-shifted | cleaned-removed | generalized | added
    reason: str
    old: dict | None = None  # {"hmac": ..., "length": ...[, "preview": ...]} - never the raw value
    new: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetadataResult:
    changes: list[TagChange] = field(default_factory=list)
    pseudo_patient_id: str = ""
    date_offset_days: int | None = None
    original_uids: set[str] = field(default_factory=set)
    pseudonyms: set[str] = field(default_factory=set)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for c in self.changes:
            counts[c.action] = counts.get(c.action, 0) + 1
        return counts


def is_standard_uid(value: str) -> bool:
    return value in UID_dictionary


def new_uid(cfg: Config, old: str) -> str:
    return "2.25." + str(cfg.hmac_int("uid:" + old) >> 128)


class MetadataDeidentifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ------------------------------------------------------------------ API
    def run(self, ds: Dataset, known: KnownIdentifiers, *, pixel_cleaned: bool) -> MetadataResult:
        cfg = self.cfg
        res = MetadataResult()
        patient_key = str(ds.get("PatientID", "") or ds.get("PatientName", "") or "unknown")
        token = cfg.hmac_hex("pid:" + patient_key, 10).upper()
        self._pseudo_id = f"{cfg.pseudonym_prefix}{token}"
        self._pseudo_name = f"{cfg.pseudonym_prefix}^{token}"
        res.pseudo_patient_id = self._pseudo_id
        res.pseudonyms = {self._pseudo_id, self._pseudo_name}
        if cfg.date_mode == DateMode.SHIFT:
            self._offset = -(365 + cfg.hmac_int("date:" + patient_key) % (365 * 9))
            res.date_offset_days = self._offset
        else:
            self._offset = 0
        self._known = known
        self._res = res
        self._process(ds, "")

        # File meta: keep it consistent with the remapped SOP Instance UID.
        fm = getattr(ds, "file_meta", None)
        if fm is not None and "MediaStorageSOPInstanceUID" in fm:
            old = str(fm.MediaStorageSOPInstanceUID)
            if not is_standard_uid(old):
                fm.MediaStorageSOPInstanceUID = new_uid(cfg, old)
                res.original_uids.add(old)

        self._add_deid_markers(ds, pixel_cleaned)
        return res

    # -------------------------------------------------------------- helpers
    def _old(self, value) -> dict:
        s = _as_text(value)
        d = {"hmac": self.cfg.hmac_hex("val:" + s), "length": len(s)}
        if self.cfg.report_previews:
            d["preview"] = mask(s)
        return d

    def _log(self, path, elem, action, reason, *, old=True, new=None):
        self._res.changes.append(
            TagChange(
                path=path,
                tag=f"({elem.tag.group:04X},{elem.tag.element:04X})",
                keyword=elem.keyword or "",
                vr=str(elem.VR),
                action=action,
                reason=reason,
                old=self._old(elem.value) if old and elem.value not in (None, "") else None,
                new=new,
            )
        )

    def _process(self, ds: Dataset, prefix: str) -> None:
        cfg = self.cfg
        for elem in list(ds):
            tag = elem.tag
            kw = elem.keyword or ""
            path = f"{prefix}{kw or str(tag)}"
            vr = str(elem.VR)

            if tag.is_private:
                if not cfg.keep_private_tags:
                    self._log(path, elem, "removed", "private attribute")
                    del ds[tag]
                continue
            if 0x6000 <= tag.group <= 0x601E and tag.group % 2 == 0:
                if cfg.remove_overlays:
                    self._log(path, elem, "removed", "overlay plane (may hold burned-in annotations)", old=vr != "OW" and vr != "OB")
                    del ds[tag]
                continue
            if 0x5000 <= tag.group <= 0x501E and tag.group % 2 == 0:
                if cfg.remove_curves:
                    self._log(path, elem, "removed", "curve data (retired)", old=False)
                    del ds[tag]
                continue
            if tag == 0x7FE00010:  # Pixel Data is handled by the pixel stage
                continue

            rule = RULES.get(kw)
            if rule is None:
                if vr == "SQ":
                    rule = "RECURSE"
                elif vr == "UI":
                    rule = "U"
                elif vr == "PN":
                    rule = "X"
                elif vr in ("DA", "DT"):
                    rule = "DATE"
                elif vr == "TM":
                    rule = "TIME"
                elif vr in TEXT_VRS:
                    rule = "C"
                else:
                    rule = "K"
            reason_src = "PS3.15 Basic Profile" if kw in RULES else "catch-all rule"

            if rule == "X":
                self._log(path, elem, "removed", reason_src, old=vr != "SQ")
                del ds[tag]
            elif rule == "RECURSE" or (vr == "SQ" and rule in ("K", "C")):
                for i, item in enumerate(elem.value or []):
                    self._process(item, f"{path}[{i}].")
            elif rule == "Z":
                if elem.value not in (None, ""):
                    self._log(path, elem, "emptied", reason_src)
                    elem.value = ""
            elif rule == "D":
                new = self._pseudo_name if kw == "PatientName" else self._pseudo_id
                self._log(path, elem, "pseudonymized", reason_src + " (consistent per-patient pseudonym)", new=new)
                elem.value = new
            elif rule == "U":
                self._remap_uid(path, elem, kw)
            elif rule == "C":
                self._clean(ds, path, elem, kw)
            elif rule == "DATE":
                self._date(ds, path, elem, kw, reason_src)
            elif rule == "TIME":
                if cfg.date_mode == DateMode.REMOVE and elem.value not in (None, ""):
                    if kw in TYPE2_EMPTY:
                        self._log(path, elem, "emptied", reason_src)
                        elem.value = ""
                    else:
                        self._log(path, elem, "removed", reason_src)
                        del ds[tag]
            elif rule == "CHAR":
                if not cfg.retain_patient_characteristics:
                    if kw in TYPE2_EMPTY:
                        self._log(path, elem, "emptied", reason_src)
                        elem.value = ""
                    else:
                        self._log(path, elem, "removed", reason_src)
                        del ds[tag]
                elif kw == "PatientAge":
                    self._generalize_age(path, elem)
            # "K": keep

    def _remap_uid(self, path, elem, kw):
        if kw in KEEP_UID_KEYWORDS or elem.value in (None, ""):
            return
        vals = list(elem.value) if isinstance(elem.value, (list, MultiValue)) else [elem.value]
        out = []
        changed = False
        for v in vals:
            v = str(v)
            if is_standard_uid(v):
                out.append(v)
            else:
                self._res.original_uids.add(v)
                out.append(new_uid(self.cfg, v))
                changed = True
        if changed:
            self._log(path, elem, "uid-remapped", "PS3.15 Basic Profile (U): consistent UID replacement", new=out[0] if len(out) == 1 else f"{len(out)} UIDs")
            elem.value = out if len(out) > 1 else out[0]

    def _clean(self, ds, path, elem, kw):
        text = _as_text(elem.value)
        if not text.strip():
            return
        if not self.cfg.clean_descriptors and kw in RULES:
            self._log(path, elem, "removed", "descriptor (clean-descriptors disabled)")
            del ds[elem.tag]
            return
        findings = detect(text, self._known, include_weak=False)
        if findings:
            cats = sorted({f.category for f in findings})
            self._log(path, elem, "cleaned-removed", f"free text contained possible PHI: {', '.join(cats)}")
            if kw in TYPE2_EMPTY:
                elem.value = ""
            else:
                del ds[elem.tag]

    def _date(self, ds, path, elem, kw, reason_src):
        mode = self.cfg.date_mode
        if elem.value in (None, ""):
            return
        if mode == DateMode.KEEP:
            return
        if mode == DateMode.REMOVE:
            if kw in TYPE2_EMPTY:
                self._log(path, elem, "emptied", reason_src)
                elem.value = ""
            else:
                self._log(path, elem, "removed", reason_src)
                del ds[elem.tag]
            return
        vals = list(elem.value) if isinstance(elem.value, (list, MultiValue)) else [elem.value]
        shifted = [_shift(str(v), self._offset) for v in vals]
        if any(s is None for s in shifted):
            self._log(path, elem, "removed", "unparseable date under shift mode")
            del ds[elem.tag]
            return
        self._log(path, elem, "date-shifted", f"consistent per-patient shift ({self._offset} days)", new=shifted[0])
        elem.value = shifted if len(shifted) > 1 else shifted[0]

    def _generalize_age(self, path, elem):
        s = _as_text(elem.value).strip().upper()
        if len(s) == 4 and s[:3].isdigit() and s[3] == "Y" and int(s[:3]) >= 90:
            self._log(path, elem, "generalized", "HIPAA: ages over 89 aggregated to 90+", new="090Y")
            elem.value = "090Y"

    def _add_deid_markers(self, ds: Dataset, pixel_cleaned: bool) -> None:
        from .. import __version__

        cfg = self.cfg
        codes = [("113100", "Basic Application Confidentiality Profile")]
        if pixel_cleaned:
            codes.append(("113101", "Clean Pixel Data Option"))
        if cfg.clean_descriptors:
            codes.append(("113105", "Clean Descriptors Option"))
        if cfg.date_mode == DateMode.SHIFT:
            codes.append(("113107", "Retain Longitudinal Temporal Information Modified Dates Option"))
        elif cfg.date_mode == DateMode.KEEP:
            codes.append(("113106", "Retain Longitudinal Temporal Information Full Dates Option"))
        if cfg.retain_patient_characteristics:
            codes.append(("113108", "Retain Patient Characteristics Option"))
        if cfg.keep_private_tags:
            codes.append(("113111", "Retain Safe Private Option"))

        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = [f"PixelGuard {__version__}"] + [m for _, m in codes]  # LO, VM 1-n, each <= 64 chars
        seq = Sequence()
        for value, meaning in codes:
            item = Dataset()
            item.CodeValue = value
            item.CodingSchemeDesignator = "DCM"
            item.CodeMeaning = meaning
            seq.append(item)
        ds.DeidentificationMethodCodeSequence = seq
        ds.LongitudinalTemporalInformationModified = {
            DateMode.REMOVE: "REMOVED", DateMode.SHIFT: "MODIFIED", DateMode.KEEP: "UNMODIFIED"
        }[cfg.date_mode]


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, MultiValue)):
        return "\\".join(str(v) for v in value)
    if isinstance(value, bytes):
        return value.decode("latin-1", "replace")
    return str(value)


def _shift(value: str, days: int) -> str | None:
    v = value.strip()
    try:
        if len(v) >= 8 and v[:8].isdigit():
            d = _dt.datetime.strptime(v[:8], "%Y%m%d").date() + _dt.timedelta(days=days)
            return d.strftime("%Y%m%d") + v[8:]
    except ValueError:
        return None
    return None
