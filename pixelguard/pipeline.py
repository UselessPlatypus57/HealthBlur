"""Batch de-identification pipeline.

Per file:  read -> harvest identifiers -> clean pixels -> clean metadata ->
write -> re-read and audit -> quarantine failures.

The pipeline fails closed: a file that raises an error is never written to
the output folder, and a file that fails the audit is moved to
``_quarantine/`` instead of the clean output.
"""

from __future__ import annotations

import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pydicom
from pydicom.errors import InvalidDicomError

from . import DISCLAIMER, __version__
from .config import Config
from .metadata import MetadataDeidentifier
from .phi import KnownIdentifiers, harvest
from .pixel import OCREngine, clean_pixels, get_engine
from .verify import FAIL, PASS, WARN, Check, audit_metadata, audit_pixels, overall

UNSUPPORTED_SOP_PREFIXES = {
    "1.2.840.10008.5.1.4.1.1.88.": "Structured Report (free-text content not supported)",
    "1.2.840.10008.5.1.4.1.1.104.": "Encapsulated document (PDF/CDA not supported)",
}


def discover(paths: Iterable[str | Path]) -> list[Path]:
    """DICOM files under the given files/directories (sorted, de-duplicated)."""
    found: set[Path] = set()
    for p in map(Path, paths):
        if p.is_file():
            found.add(p.resolve())
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith("_")]
                for f in files:
                    fp = Path(root) / f
                    if fp.suffix.lower() in (".dcm", ".dicom", ".dic") or (not fp.suffix and _has_dicm(fp)):
                        found.add(fp.resolve())
    return sorted(found)


def _has_dicm(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def process_file(src: Path, out_root: Path, cfg: Config, engine: OCREngine | None, *, index: int = 0,
                 extra_known: KnownIdentifiers | None = None, preview_dir: Path | None = None) -> dict:
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    rec: dict = {
        "index": index,
        "input_ref": cfg.hmac_hex("path:" + str(src)),
        "input_name": None,  # filled only in linkage (may contain PHI)
        "input_bytes": src.stat().st_size if src.exists() else None,
        "status": None,
        "output": None,
        "checks": [],
        "metadata_changes": [],
        "pixel": None,
        "timings": timings,
    }
    try:
        t = time.perf_counter()
        ds = pydicom.dcmread(str(src), force=True)
        if "SOPClassUID" not in ds and "PixelData" not in ds:
            raise InvalidDicomError("not a DICOM object (no SOP Class UID or Pixel Data)")
        timings["read"] = time.perf_counter() - t

        sop = str(ds.get("SOPClassUID", ""))
        rec["modality"] = str(ds.get("Modality", ""))
        rec["sop_class"] = getattr(ds.get("SOPClassUID", None), "name", sop)
        for prefix, why in UNSUPPORTED_SOP_PREFIXES.items():
            if sop.startswith(prefix):
                rec["status"] = "SKIPPED"
                rec["checks"] = [Check("input.supported", FAIL, f"not processed: {why}; review manually").to_dict()]
                return rec

        known = harvest(ds)
        if extra_known:
            known.merge(extra_known)
        rec["known_identifier_count"] = len(known)
        flags = {"Modality": rec["modality"], "BurnedInAnnotation": str(ds.get("BurnedInAnnotation", ""))}
        rec["header_flags"] = flags

        # ---- pixel stage
        pr = None
        if "PixelData" in ds and cfg.clean_pixels and engine is not None:
            t = time.perf_counter()
            pr = clean_pixels(ds, known, cfg, engine, keep_previews=preview_dir is not None)
            timings["pixel"] = time.perf_counter() - t
            if pr.skipped_reason:
                rec["pixel"] = {"performed": False, "skipped_reason": pr.skipped_reason}
            else:
                rec["pixel"] = pr.to_dict(cfg.report_previews or preview_dir is not None)
        elif "PixelData" in ds:
            rec["pixel"] = {"performed": False, "skipped_reason": "pixel cleaning disabled (metadata-only)"}
            if preview_dir is not None:
                from .pixel import load_pixels, preview_u8

                try:
                    a, _ = load_pixels(ds)
                    _save_png(preview_u8(a, ds), preview_dir / f"{index:05d}_before.png")
                except Exception:
                    pass

        # ---- metadata stage
        t = time.perf_counter()
        meta = MetadataDeidentifier(cfg).run(ds, known, pixel_cleaned=bool(pr and pr.performed))
        timings["metadata"] = time.perf_counter() - t
        rec["metadata_changes"] = [c.to_dict() for c in meta.changes]
        rec["metadata_summary"] = meta.summary()
        rec["pseudo_patient_id"] = meta.pseudo_patient_id
        if meta.date_offset_days is not None:
            rec["date_shift_applied"] = True

        # ---- write
        t = time.perf_counter()
        rel = Path(meta.pseudo_patient_id) / f"{ds.SOPInstanceUID}.dcm"
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        ds.save_as(str(tmp), enforce_file_format=True)
        os.replace(tmp, dst)
        timings["write"] = time.perf_counter() - t

        if preview_dir is not None and pr is not None and pr.before_preview is not None:
            _save_png(pr.before_preview, preview_dir / f"{index:05d}_before.png")
            _save_png(pr.after_preview, preview_dir / f"{index:05d}_after.png")

        # ---- audit (on the file as written)
        checks: list[Check] = []
        if cfg.verify:
            t = time.perf_counter()
            written = pydicom.dcmread(str(dst))
            checks += audit_metadata(written, known, cfg, original_uids=meta.original_uids, pseudonyms=meta.pseudonyms)
            if pr is not None and pr.skipped_reason:
                checks.append(Check("pixel.residual_text", FAIL, f"pixel cleaning not possible: {pr.skipped_reason}"))
            else:
                checks += audit_pixels(written, known, cfg, engine, pixel_result=pr, original_scan=pr.scan if pr else None, original_flags=flags)
            timings["verify"] = time.perf_counter() - t
            if preview_dir is not None and pr is None and "PixelData" in written:
                from .pixel import load_pixels, preview_u8

                try:
                    a, _ = load_pixels(written)
                    _save_png(preview_u8(a, written), preview_dir / f"{index:05d}_after.png")
                except Exception:
                    pass
        else:
            checks.append(Check("audit", WARN, "verification disabled"))
        rec["checks"] = [c.to_dict() for c in checks]
        rec["status"] = overall(checks)

        if rec["status"] == FAIL and cfg.quarantine_failures:
            q = out_root / "_quarantine" / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), q)
            _rmdir_if_empty(dst.parent)
            rec["output"] = str(Path("_quarantine") / rel).replace("\\", "/")
            rec["quarantined"] = True
        else:
            rec["output"] = str(rel).replace("\\", "/")
    except Exception as exc:  # fail closed
        rec["status"] = "ERROR"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=4)
    finally:
        timings["total"] = time.perf_counter() - t_start
        for k in list(timings):
            timings[k] = round(timings[k], 4)
    return rec


def _rmdir_if_empty(p: Path) -> None:
    try:
        p.rmdir()
    except OSError:
        pass


def _save_png(u8, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8).save(path)


# ---------------------------------------------------------------- batching
_WORKER_ENGINE: OCREngine | None = None


def _worker_init(cfg_langs, use_ocr):
    global _WORKER_ENGINE
    _WORKER_ENGINE = get_engine("easyocr", cfg_langs) if use_ocr else None


def _worker_run(args):
    src, out_root, cfg, index, extra_known, preview_dir = args
    return process_file(src, out_root, cfg, _WORKER_ENGINE, index=index, extra_known=extra_known, preview_dir=preview_dir)


def run_batch(inputs: list[Path], out_root: Path, cfg: Config, *, engine: OCREngine | None = None, workers: int = 1,
              extra_known: KnownIdentifiers | None = None, preview_dir: Path | None = None,
              progress: Callable[[int, int, dict], None] | None = None, linkage_path: Path | None = None) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    need_ocr = cfg.clean_pixels or (cfg.verify and cfg.verify_pixels)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    records: list[dict] = []

    if workers > 1 and engine is None:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(cfg.ocr_languages, need_ocr)) as pool:
            futs = {pool.submit(_worker_run, (src, out_root, cfg, i, extra_known, preview_dir)): i for i, src in enumerate(inputs)}
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                if progress:
                    progress(len(records), len(inputs), rec)
    else:
        if engine is None and need_ocr:
            engine = get_engine("easyocr", cfg.ocr_languages)
        for i, src in enumerate(inputs):
            rec = process_file(src, out_root, cfg, engine, index=i, extra_known=extra_known, preview_dir=preview_dir)
            records.append(rec)
            if progress:
                progress(i + 1, len(inputs), rec)

    records.sort(key=lambda r: r["index"])
    elapsed = time.perf_counter() - t0
    if linkage_path is not None:
        _write_linkage(linkage_path, inputs, records)
    return {
        "tool": "PixelGuard",
        "version": __version__,
        "disclaimer": DISCLAIMER,
        "started_utc": started.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "config": cfg.public_dict(),
        "summary": summarize(records, elapsed),
        "files": records,
    }


def summarize(records: list[dict], elapsed: float) -> dict:
    n = len(records)
    by = {s: sum(1 for r in records if r["status"] == s) for s in (PASS, WARN, FAIL, "ERROR", "SKIPPED")}
    totals = [r["timings"].get("total", 0.0) for r in records]
    processed = [r for r in records if r["status"] in (PASS, WARN, FAIL)]
    regions = sum((r.get("pixel") or {}).get("redacted_regions", 0) or 0 for r in records)
    tag_changes = sum(len(r.get("metadata_changes", [])) for r in records)
    failing_checks: dict[str, int] = {}
    for r in records:
        for c in r.get("checks", []):
            if c["status"] in (FAIL, WARN):
                key = f"{c['status']}:{c['id']}"
                failing_checks[key] = failing_checks.get(key, 0) + 1
    mb = sum(r.get("input_bytes") or 0 for r in records) / 1e6
    srt = sorted(totals)
    return {
        "files": n,
        "status_counts": by,
        "audit_pass_rate_strict": round(by[PASS] / n, 4) if n else None,
        "audit_pass_rate_with_warnings": round((by[PASS] + by[WARN]) / n, 4) if n else None,
        "processed": len(processed),
        "redacted_pixel_regions": regions,
        "metadata_changes": tag_changes,
        "seconds_per_file_mean": round(sum(totals) / n, 3) if n else None,
        "seconds_per_file_median": round(srt[n // 2], 3) if n else None,
        "seconds_per_file_p95": round(srt[min(n - 1, int(0.95 * n))], 3) if n else None,
        "files_per_minute": round(60 * n / elapsed, 2) if elapsed > 0 else None,
        "megabytes_per_second": round(mb / elapsed, 3) if elapsed > 0 else None,
        "check_issues": dict(sorted(failing_checks.items(), key=lambda kv: -kv[1])),
    }


def _write_linkage(path: Path, inputs: list[Path], records: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "input_path", "output", "status"])
        for r in records:
            w.writerow([r["index"], str(inputs[r["index"]]), r.get("output") or "", r["status"]])
