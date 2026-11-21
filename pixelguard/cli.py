"""Command-line interface: ``pixelguard <command> ...``"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import DISCLAIMER, __version__
from .config import Config, DateMode, RedactMethod, TextPolicy


def _add_cfg_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("de-identification options")
    g.add_argument("--policy", choices=[e.value for e in TextPolicy], default="conservative", help="burned-in text policy (default: conservative = redact all text except recognised clinical annotations)")
    g.add_argument("--redact", choices=[e.value for e in RedactMethod], default="fill", help="pixel redaction method (default: fill)")
    g.add_argument("--dates", choices=[e.value for e in DateMode], default="remove", help="date handling (default: remove, per PS3.15 Basic Profile)")
    g.add_argument("--key", help="secret for pseudonyms/UIDs/date offsets (or env PIXELGUARD_KEY); random per run if omitted")
    g.add_argument("--key-file", type=Path, help="read the secret from a file")
    g.add_argument("--metadata-only", action="store_true", help="skip pixel cleaning (baseline comparison; the audit will still scan pixels)")
    g.add_argument("--keep-private", action="store_true", help="keep private attributes")
    g.add_argument("--keep-overlays", action="store_true", help="keep overlay planes")
    g.add_argument("--no-patient-characteristics", action="store_true", help="also remove sex/age/size/weight")
    g.add_argument("--min-confidence", type=float, default=0.25, help="OCR confidence below which unmatched text is only flagged")
    g.add_argument("--all-frames", action="store_true", help="OCR every frame of multi-frame objects (slow)")
    g.add_argument("--no-verify", action="store_true", help="skip the post-anonymization audit (not recommended)")
    g.add_argument("--no-quarantine", action="store_true", help="leave failing outputs in place instead of moving them to _quarantine/")
    g.add_argument("--report-previews", action="store_true", help="include masked previews (e.g. 'D**^J***') of redacted text in reports")


def _cfg(a) -> Config:
    cfg = Config(
        date_mode=DateMode(a.dates),
        redact_method=RedactMethod(a.redact),
        text_policy=TextPolicy(a.policy),
        clean_pixels=not a.metadata_only,
        keep_private_tags=a.keep_private,
        remove_overlays=not a.keep_overlays,
        retain_patient_characteristics=not a.no_patient_characteristics,
        ocr_min_confidence=a.min_confidence,
        ocr_all_frames=a.all_frames,
        verify=not a.no_verify,
        quarantine_failures=not a.no_quarantine,
        report_previews=a.report_previews,
    )
    key = a.key or os.environ.get("PIXELGUARD_KEY")
    if a.key_file:
        key = a.key_file.read_text(encoding="utf-8").strip()
    if key:
        cfg.secret_key = Config.key_from_text(key)
    return cfg


def _progress(i, n, rec):
    bad = next((c["message"] for c in rec.get("checks", []) if c["status"] in ("FAIL", "WARN")), rec.get("error") or "")
    print(f"  [{i:>4}/{n}] {rec['status']:<7} {rec.get('modality', ''):<3} {rec['timings'].get('total', 0):>6.2f}s  {bad[:90]}", flush=True)


def _print_summary(s: dict) -> None:
    sc = s["status_counts"]
    print(f"\nFiles: {s['files']}   PASS {sc['PASS']}  WARN {sc['WARN']}  FAIL {sc['FAIL']}  ERROR {sc['ERROR']}  SKIPPED {sc['SKIPPED']}")
    if s["files"]:
        print(f"Audit pass rate: {100 * s['audit_pass_rate_strict']:.1f}% strict, {100 * s['audit_pass_rate_with_warnings']:.1f}% including warnings")
        print(f"Speed: {s['seconds_per_file_mean']} s/file mean, {s['files_per_minute']} files/min")
    for k, v in list(s["check_issues"].items())[:8]:
        print(f"  {v:>4} x {k}")


# ------------------------------------------------------------------ commands
def cmd_deidentify(a) -> int:
    from .phi.known import KnownIdentifiers
    from .pipeline import discover, run_batch
    from .report import write_reports

    cfg = _cfg(a)
    inputs = discover(a.inputs)
    if not inputs:
        print("no DICOM files found", file=sys.stderr)
        return 2
    out = Path(a.output)
    extra = _known_from_file(a.known_phi) if a.known_phi else None
    print(f"PixelGuard {__version__}: {len(inputs)} file(s) -> {out}  (policy={cfg.text_policy.value}, redact={cfg.redact_method.value}, dates={cfg.date_mode.value}{', METADATA ONLY' if not cfg.clean_pixels else ''})")
    rep = run_batch(inputs, out, cfg, workers=a.workers, extra_known=extra, progress=_progress,
                    linkage_path=Path(a.linkage) if a.linkage else None,
                    preview_dir=Path(a.previews) if a.previews else None)
    paths = write_reports(rep, Path(a.report_dir) if a.report_dir else out)
    _print_summary(rep["summary"])
    print(f"\nReport: {paths['html']}\n        {paths['json']}")
    print(f"\nNote: {DISCLAIMER}")
    return 0 if rep["summary"]["status_counts"]["FAIL"] + rep["summary"]["status_counts"]["ERROR"] == 0 else 1


def _known_from_file(path: str):
    from .phi.known import KnownIdentifiers, _add_name, _compact

    k = KnownIdentifiers()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        v = line.strip()
        if not v or v.startswith("#"):
            continue
        if any(ch.isalpha() for ch in v) and not any(ch.isdigit() for ch in v):
            _add_name(k, v, "known-phi-file")
        else:
            k.add(_compact(v), "known-phi-file", "id")
    return k


def cmd_audit(a) -> int:
    """Audit an existing (supposedly de-identified) dataset without modifying it."""
    import pydicom

    from .phi import harvest
    from .pipeline import discover, summarize
    from .pixel import get_engine
    from .report import write_reports
    from .verify import audit_metadata, audit_pixels, overall

    cfg = _cfg(a)
    inputs = discover(a.inputs)
    extra = _known_from_file(a.known_phi) if a.known_phi else None
    engine = None if a.no_pixels else get_engine("easyocr", cfg.ocr_languages)
    records = []
    t0 = time.perf_counter()
    print(f"Auditing {len(inputs)} file(s)")
    for i, src in enumerate(inputs):
        t = time.perf_counter()
        rec = {"index": i, "input_ref": cfg.hmac_hex("path:" + str(src)), "output": str(src.name), "input_bytes": src.stat().st_size, "timings": {}}
        try:
            ds = pydicom.dcmread(str(src), force=True)
            known = harvest(ds)
            if extra:
                known.merge(extra)
            rec["modality"] = str(ds.get("Modality", ""))
            flags = {"Modality": rec["modality"], "BurnedInAnnotation": str(ds.get("BurnedInAnnotation", ""))}
            # The header itself is the thing being audited: identifiers harvested
            # from it are only meaningful if they are not placeholders.
            checks = audit_metadata(ds, None, cfg)
            if len(known):
                checks.insert(0, _header_identity_check(ds))
            checks += audit_pixels(ds, known, cfg, engine, original_flags=flags) if engine else []
            rec["checks"] = [c.to_dict() for c in checks]
            rec["status"] = overall(checks)
        except Exception as exc:
            rec["status"], rec["error"], rec["checks"] = "ERROR", f"{type(exc).__name__}: {exc}", []
        rec["timings"]["total"] = round(time.perf_counter() - t, 4)
        records.append(rec)
        _progress(i + 1, len(inputs), rec)
    elapsed = time.perf_counter() - t0
    rep = {"tool": "PixelGuard audit", "version": __version__, "disclaimer": DISCLAIMER,
           "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "elapsed_seconds": round(elapsed, 3),
           "config": cfg.public_dict(), "summary": summarize(records, elapsed), "files": records}
    paths = write_reports(rep, Path(a.report_dir))
    _print_summary(rep["summary"])
    print(f"\nReport: {paths['html']}")
    return 0 if rep["summary"]["status_counts"]["FAIL"] == 0 else 1


def _header_identity_check(ds):
    from .verify import FAIL, WARN, Check

    present = [kw for kw in ("PatientName", "PatientID", "PatientBirthDate", "AccessionNumber", "InstitutionName", "ReferringPhysicianName", "PatientAddress")
               if str(ds.get(kw, "") or "").strip() and str(ds.get(kw, "")).upper() not in ("ANONYMOUS", "ANON", "REMOVED")]
    removed = str(ds.get("PatientIdentityRemoved", "")).upper() == "YES"
    if present:
        return Check("metadata.header_identifiers", FAIL if removed else WARN, f"identifying attributes populated: {', '.join(present)}", present)
    return Check("metadata.header_identifiers", "PASS", "core identifying attributes empty or placeholders")


def cmd_scan(a) -> int:
    """Read-only: list detected PHI per file (metadata and pixels)."""
    import pydicom

    from .phi import detect, harvest
    from .pipeline import discover
    from .pixel import get_engine, load_pixels, scan_array
    from .verify import PATTERN_VRS, walk

    cfg = _cfg(a)
    engine = None if a.no_pixels else get_engine("easyocr", cfg.ocr_languages)
    results = []
    for i, src in enumerate(discover(a.inputs)):
        ds = pydicom.dcmread(str(src), force=True)
        known = harvest(ds)
        meta = []
        for path, elem in walk(ds):
            if elem.tag.is_private:
                meta.append({"path": path, "finding": "private attribute"})
            elif elem.VR == "PN" and str(elem.value).strip():
                meta.append({"path": path, "finding": "person name"})
            elif elem.VR in PATTERN_VRS and str(elem.value).strip():
                f = detect(str(elem.value), known, include_weak=False)
                if f:
                    meta.append({"path": path, "finding": ", ".join(sorted({x.category for x in f}))})
        pix = []
        if engine and "PixelData" in ds:
            arr, _ = load_pixels(ds)
            sc = scan_array(arr, ds, known, cfg, engine)
            pix = [d.to_dict(cfg.report_previews) for d in sc.decisions]
        r = {"file": str(src), "known_identifiers": len(known), "metadata_findings": meta, "pixel_text": pix}
        results.append(r)
        n_phi = sum(1 for d in pix if d["decision"] == "redact")
        print(f"{src.name}: {len(meta)} metadata finding(s), {n_phi} pixel text region(s) to redact, {len(pix) - n_phi} kept/flagged")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


def cmd_synth(a) -> int:
    from .synth import generate

    m = generate(a.output, n=a.n, seed=a.seed, naive_fraction=a.naive_fraction, hard_fraction=a.hard_fraction)
    print(f"wrote {len(m)} synthetic DICOM files (+ .truth.json ground truth) to {a.output}")
    print(f"  {sum(t['naive_anonymized'] for t in m)} look already anonymized in the header but still have burned-in PHI")
    return 0


def cmd_benchmark(a) -> int:
    """Generate synthetic data, run several configurations, and measure them."""
    from .evaluate import evaluate_synthetic
    from .pipeline import discover, run_batch
    from .report import write_reports
    from .synth import generate

    root = Path(a.output)
    data = root / "synthetic"
    if not (data / "manifest.json").exists() or a.regenerate:
        generate(data, n=a.n, seed=a.seed)
    inputs = discover([data])
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    table = []
    for v in variants:
        cfg = Config(text_policy=TextPolicy.TARGETED if v == "targeted" else TextPolicy.CONSERVATIVE,
                     clean_pixels=v != "metadata-only", secret_key=Config.key_from_text("benchmark"))
        out = root / v
        print(f"\n== {v}: {len(inputs)} files")
        rep = run_batch(inputs, out, cfg, workers=a.workers, progress=_progress if a.verbose else None)
        ev = evaluate_synthetic(data, rep, out, inputs)
        write_reports(rep, out, ev)
        table.append((v, ev))
        print(f"   pixel PHI recall {_p(ev['pixel_phi_recall'])} | pixel FP rate {_p(ev['pixel_false_positive_rate'])} | metadata PHI recall {_p(ev['metadata_phi_recall'])} | audit pass {_p(ev['audit_pass_rate_strict'])} strict / {_p(ev['audit_pass_rate_with_warnings'])} w/ warnings | {ev['seconds_per_file_mean']} s/file")
    summary = {v: {k: x for k, x in ev.items() if k != "per_file"} for v, ev in table}
    (root / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["| variant | pixel PHI recall | hard-case recall | pixel FP rate | collateral px | metadata PHI recall | clinical retention | audit pass (strict / w/ warn) | s/file |", "|---|---|---|---|---|---|---|---|---|"]
    for v, ev in table:
        md.append(f"| {v} | {_p(ev['pixel_phi_recall'])} | {_p(ev['pixel_phi_recall_hard_cases'])} | {_p(ev['pixel_false_positive_rate'])} | {_p(ev['collateral_redacted_pixel_share'])} | {_p(ev['metadata_phi_recall'])} | {_p(ev['clinical_attribute_retention'])} | {_p(ev['audit_pass_rate_strict'])} / {_p(ev['audit_pass_rate_with_warnings'])} | {ev['seconds_per_file_mean']} |")
    (root / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nwrote {root / 'benchmark.json'} and {root / 'benchmark.md'}")
    return 0


def _p(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def cmd_evaluate_external(a) -> int:
    from .evaluate import evaluate_external
    from .pipeline import discover, run_batch
    from .report import write_reports

    cfg = _cfg(a)
    inputs = discover(a.inputs)
    rep = run_batch(inputs, Path(a.output), cfg, workers=a.workers, progress=_progress)
    ev = evaluate_external(rep, Path(a.output), inputs, Path(a.known_csv))
    write_reports(rep, Path(a.output), ev)
    print(json.dumps({k: v for k, v in ev.items() if k != "leaks"}, indent=2))
    return 0


def cmd_demo(a) -> int:
    """The 'looks anonymized, isn't' demonstration."""
    from .pipeline import discover, run_batch
    from .synth import generate

    root = Path(a.output)
    gen = generate(root / "input", n=6, seed=11, naive_fraction=1.0, hard_fraction=0.0)
    inputs = discover([root / "input"])
    print("Step 1: six files whose headers were already 'anonymized' (PatientName=ANONYMOUS).")
    base = run_batch(inputs, root / "metadata_only", Config(clean_pixels=False, quarantine_failures=False))
    leaks = [r for r in base["files"] if any(c["id"] == "pixel.residual_text" and c["status"] == "FAIL" for c in r["checks"])]
    print(f"Step 2: metadata-only clean-up + audit -> {len(leaks)}/{len(inputs)} files still show identifying text in the pixels:")
    for r in leaks:
        c = next(c for c in r["checks"] if c["id"] == "pixel.residual_text")
        cats = sorted({cat for d in c["details"] for cat in d["categories"]})
        corners = sorted({_corner(d["bbox"], gen[r["index"]]["shape"]) for d in c["details"]})
        print(f"     file {r['index']} ({r.get('modality')}): {', '.join(cats)} in the {', '.join(corners)} corner(s)")
    full = run_batch(inputs, root / "pixelguard", Config())
    s = full["summary"]
    print(f"Step 3: full PixelGuard pipeline -> audit PASS {s['status_counts']['PASS']}, WARN {s['status_counts']['WARN']}, FAIL {s['status_counts']['FAIL']}; {s['redacted_pixel_regions']} regions redacted.")
    from .report import write_reports

    write_reports(base, root / "metadata_only")
    p = write_reports(full, root / "pixelguard")
    print(f"Reports: {root / 'metadata_only' / 'audit_report.html'}\n         {p['html']}")
    return 0


def _corner(bbox, shape):
    h, w = shape
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return ("top" if cy < h / 2 else "bottom") + "-" + ("left" if cx < w / 2 else "right")


def cmd_serve(a) -> int:
    import uvicorn

    from .web.app import create_app

    print(f"PixelGuard dashboard on http://{a.host}:{a.port}  (local only; data stays on this machine)")
    uvicorn.run(create_app(Path(a.workdir)), host=a.host, port=a.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pixelguard", description="Find and remove patient identifiers in DICOM metadata and burned-in pixel text.")
    p.add_argument("--version", action="version", version=f"pixelguard {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("deidentify", help="de-identify files/folders into an output folder, with audit report")
    d.add_argument("inputs", nargs="+")
    d.add_argument("-o", "--output", required=True)
    d.add_argument("--workers", type=int, default=1, help="parallel worker processes (each loads its own OCR model)")
    d.add_argument("--report-dir", help="where to write audit_report.{json,html} (default: output folder)")
    d.add_argument("--linkage", help="write input->output mapping CSV here (SENSITIVE: contains original paths)")
    d.add_argument("--previews", help="write before/after PNG previews here (SENSITIVE: 'before' images show PHI)")
    d.add_argument("--known-phi", help="text file of extra identifiers to hunt for (one per line)")
    _add_cfg_args(d)
    d.set_defaults(func=cmd_deidentify)

    s = sub.add_parser("scan", help="read-only: report PHI found in metadata and pixels")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--json", help="write findings to this JSON file")
    s.add_argument("--no-pixels", action="store_true")
    _add_cfg_args(s)
    s.set_defaults(func=cmd_scan)

    au = sub.add_parser("audit", help="audit an existing de-identified dataset for residual leaks")
    au.add_argument("inputs", nargs="+")
    au.add_argument("--report-dir", default="pixelguard_audit")
    au.add_argument("--known-phi", help="text file of identifiers expected to be absent (one per line)")
    au.add_argument("--no-pixels", action="store_true")
    _add_cfg_args(au)
    au.set_defaults(func=cmd_audit)

    sy = sub.add_parser("synth", help="generate synthetic DICOM files with ground truth")
    sy.add_argument("output")
    sy.add_argument("--n", type=int, default=36)
    sy.add_argument("--seed", type=int, default=7)
    sy.add_argument("--naive-fraction", type=float, default=0.25)
    sy.add_argument("--hard-fraction", type=float, default=0.2)
    sy.set_defaults(func=cmd_synth)

    b = sub.add_parser("benchmark", help="measure recall / false positives / speed / audit pass rate on synthetic data")
    b.add_argument("--output", default="benchmark_out")
    b.add_argument("--n", type=int, default=36)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--variants", default="metadata-only,targeted,conservative")
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--regenerate", action="store_true")
    b.add_argument("-v", "--verbose", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    e = sub.add_parser("evaluate-external", help="de-identify an external dataset and check for known PHI strings")
    e.add_argument("inputs", nargs="+")
    e.add_argument("-o", "--output", required=True)
    e.add_argument("--known-csv", required=True, help="CSV with columns file,value (file='*' applies to all)")
    e.add_argument("--workers", type=int, default=1)
    _add_cfg_args(e)
    e.set_defaults(func=cmd_evaluate_external)

    dm = sub.add_parser("demo", help="show a metadata-only 'anonymized' dataset leaking names in pixels")
    dm.add_argument("--output", default="demo_out")
    dm.set_defaults(func=cmd_demo)

    sv = sub.add_parser("serve", help="start the local web dashboard")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--workdir", default="pixelguard_jobs", help="where uploaded files and job outputs are kept")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
