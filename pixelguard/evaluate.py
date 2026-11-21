"""Measure detection quality against ground truth.

Metrics
-------
pixel PHI recall        PHI text regions whose box is >= 90 % covered by redaction
pixel false-positive    clinical (non-PHI) text regions >= 50 % covered by redaction
collateral pixels       share of redacted pixels outside any ground-truth text box
metadata PHI recall     injected header PHI values absent from every output element
clinical retention      clinically relevant attributes preserved unchanged
audit pass rate         files whose post-anonymization audit is PASS (strict) / PASS+WARN
speed                   seconds per file and files per minute

Ground truth comes from ``pixelguard synth`` sidecars, or for external data a
CSV of known PHI strings (``evaluate_external``).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pydicom

from .metadata import CLINICAL_KEEP
from .verify import walk


def _compact(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(s).upper())


def _coverage(bbox, boxes) -> float:
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return 1.0
    m = np.zeros((y1 - y0, x1 - x0), bool)
    for bx0, by0, bx1, by1 in boxes:
        ix0, iy0, ix1, iy1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
        if ix1 > ix0 and iy1 > iy0:
            m[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = True
    return float(m.mean())


def _all_text(ds) -> str:
    parts = []
    for _, elem in walk(ds):
        if elem.VR in ("SQ", "OB", "OW", "OF", "OD", "UN") or elem.tag == 0x7FE00010:
            continue
        parts.append(str(elem.value))
    return _compact(" ".join(parts))


def evaluate_synthetic(truth_dir: Path, report: dict, out_root: Path, inputs: list[Path]) -> dict:
    truth_dir, out_root = Path(truth_dir), Path(out_root)
    truths = {}
    for f in sorted(truth_dir.glob("*.truth.json")):
        t = json.loads(f.read_text())
        truths[t["file"]] = t

    phi_total = phi_hit = clin_total = clin_fp = 0
    phi_by_cat: dict[str, list[int]] = {}
    phi_by_case = {"normal": [0, 0], "hard": [0, 0]}
    collateral_px = redacted_px = 0
    meta_total = meta_hit = 0
    meta_miss: dict[str, int] = {}
    keep_total = keep_ok = 0
    per_file = []

    for rec in report["files"]:
        name = Path(inputs[rec["index"]]).name
        t = truths.get(name)
        if t is None:
            continue
        pixel = rec.get("pixel") or {}
        boxes = pixel.get("redacted_boxes_padded", []) if pixel.get("performed") else []
        f_phi = f_hit = 0
        for g in t["pixel_regions"]:
            cov = _coverage(g["bbox"], boxes)
            if g["is_phi"]:
                ok = cov >= 0.9
                phi_total += 1
                phi_hit += ok
                f_phi += 1
                f_hit += ok
                c = phi_by_cat.setdefault(g["category"], [0, 0])
                c[0] += 1
                c[1] += ok
                case = phi_by_case["hard" if t["hard_case"] else "normal"]
                case[0] += 1
                case[1] += ok
            else:
                clin_total += 1
                clin_fp += cov >= 0.5
        if boxes:
            h, w = t["shape"]
            rmask = np.zeros((h, w), bool)
            for x0, y0, x1, y1 in boxes:
                rmask[y0:y1, x0:x1] = True
            gmask = np.zeros((h, w), bool)
            for g in t["pixel_regions"]:
                x0, y0, x1, y1 = g["bbox"]
                gmask[max(0, y0 - 4) : y1 + 4, max(0, x0 - 4) : x1 + 4] = True
            redacted_px += int(rmask.sum())
            collateral_px += int((rmask & ~gmask).sum())

        out_file = out_root / rec["output"] if rec.get("output") else None
        if out_file and out_file.exists():
            ds_out = pydicom.dcmread(str(out_file))
            text = _all_text(ds_out)
            for m in t["metadata_phi"]:
                meta_total += 1
                if m["value"] == "__overlay__":
                    present = any(0x6000 <= e.tag.group <= 0x601E for _, e in walk(ds_out))
                elif m["keyword"].startswith("(0019"):
                    present = any(e.tag.is_private for _, e in walk(ds_out)) and _compact(m["value"]) in text
                else:
                    present = _compact(m["value"]) in text
                meta_hit += not present
                if present:
                    meta_miss[m["keyword"]] = meta_miss.get(m["keyword"], 0) + 1
            ds_in = pydicom.dcmread(str(truth_dir / name))
            name_in_desc = any(m["keyword"] == "SeriesDescription" for m in t["metadata_phi"])
            for kw in CLINICAL_KEEP:
                if kw not in ds_in or (kw == "SeriesDescription" and name_in_desc):
                    continue
                keep_total += 1
                if kw in ds_out and str(ds_out.data_element(kw).value) == str(ds_in.data_element(kw).value):
                    keep_ok += 1
                elif kw == "PatientAge" and kw in ds_out:
                    keep_ok += 1  # generalised (>89) is expected
        per_file.append({"file": name, "kind": t["kind"], "naive": t["naive_anonymized"], "hard": t["hard_case"],
                         "status": rec["status"], "pixel_phi": f_phi, "pixel_phi_redacted": f_hit})

    def ratio(a, b):
        return round(a / b, 4) if b else None

    s = report["summary"]
    return {
        "pixel_phi_recall": ratio(phi_hit, phi_total),
        "pixel_phi_regions": phi_total,
        "pixel_phi_recall_by_category": {k: ratio(v[1], v[0]) for k, v in sorted(phi_by_cat.items())},
        "pixel_phi_recall_normal_cases": ratio(*phi_by_case["normal"][::-1]),
        "pixel_phi_recall_hard_cases": ratio(*phi_by_case["hard"][::-1]),
        "pixel_false_positive_rate": ratio(clin_fp, clin_total),
        "clinical_text_regions": clin_total,
        "collateral_redacted_pixel_share": ratio(collateral_px, redacted_px),
        "metadata_phi_recall": ratio(meta_hit, meta_total),
        "metadata_phi_values": meta_total,
        "metadata_misses_by_field": meta_miss,
        "clinical_attribute_retention": ratio(keep_ok, keep_total),
        "audit_pass_rate_strict": s["audit_pass_rate_strict"],
        "audit_pass_rate_with_warnings": s["audit_pass_rate_with_warnings"],
        "status_counts": s["status_counts"],
        "seconds_per_file_mean": s["seconds_per_file_mean"],
        "files_per_minute": s["files_per_minute"],
        "per_file": per_file,
    }


def load_known_csv(path: Path) -> dict[str, list[str]]:
    """CSV with columns ``file`` (path or name, ``*`` = every file) and ``value``."""
    out: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row.get("file", "*").strip() or "*", []).append(row["value"])
    return out


def evaluate_external(report: dict, out_root: Path, inputs: list[Path], known_csv: Path) -> dict:
    """Leak check for external data (e.g. a public de-identification benchmark)
    given a list of PHI strings known to be present in the source files."""
    known = load_known_csv(known_csv)
    total = leaked = 0
    leaks = []
    for rec in report["files"]:
        src = inputs[rec["index"]]
        values = known.get("*", []) + known.get(src.name, []) + known.get(str(src), [])
        if not values or not rec.get("output"):
            continue
        ds = pydicom.dcmread(str(Path(out_root) / rec["output"]))
        text = _all_text(ds)
        for v in values:
            if len(_compact(v)) < 3:
                continue
            total += 1
            if _compact(v) in text:
                leaked += 1
                leaks.append({"index": rec["index"], "value_length": len(v)})
    return {
        "known_values_checked": total,
        "metadata_leaks": leaked,
        "metadata_recall": round(1 - leaked / total, 4) if total else None,
        "audit_pass_rate_strict": report["summary"]["audit_pass_rate_strict"],
        "audit_pass_rate_with_warnings": report["summary"]["audit_pass_rate_with_warnings"],
        "note": "Pixel recall needs region-level ground truth; for external data rely on the pixel.residual_text audit check and manual review.",
        "leaks": leaks,
    }
