"""Write batch audit reports (JSON + self-contained HTML)."""

from __future__ import annotations

import html
import json
from pathlib import Path

STATUS_COLORS = {"PASS": "#1a7f37", "WARN": "#9a6700", "FAIL": "#cf222e", "ERROR": "#cf222e", "SKIPPED": "#6e7781"}


def write_reports(report: dict, out_dir: Path, evaluation: dict | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = dict(report)
    if evaluation is not None:
        data["evaluation"] = evaluation
    jp = out_dir / "audit_report.json"
    jp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    hp = out_dir / "audit_report.html"
    hp.write_text(render_html(data), encoding="utf-8")
    return {"json": jp, "html": hp}


def _e(x) -> str:
    return html.escape(str(x))


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_html(data: dict) -> str:
    s = data["summary"]
    sc = s["status_counts"]
    cards = [
        ("Files", s["files"]),
        ("Audit pass (strict)", _pct(s["audit_pass_rate_strict"])),
        ("Pass incl. warnings", _pct(s["audit_pass_rate_with_warnings"])),
        ("Failed / errors", f'{sc.get("FAIL", 0)} / {sc.get("ERROR", 0)}'),
        ("Pixel regions redacted", s["redacted_pixel_regions"]),
        ("Metadata changes", s["metadata_changes"]),
        ("Mean s / file", s["seconds_per_file_mean"]),
        ("Files / min", s["files_per_minute"]),
    ]
    ev = data.get("evaluation")
    ev_html = ""
    if ev:
        rows = []
        for k, v in ev.items():
            if k in ("per_file", "leaks"):
                continue
            if isinstance(v, float) and ("rate" in k or "recall" in k or "share" in k or "retention" in k):
                v = _pct(v)
            elif isinstance(v, dict):
                v = ", ".join(f"{a}: {_pct(b) if isinstance(b, float) else b}" for a, b in v.items()) or "none"
            rows.append(f"<tr><th>{_e(k.replace('_', ' '))}</th><td>{_e(v)}</td></tr>")
        ev_html = f"<h2>Evaluation against ground truth</h2><table class='kv'>{''.join(rows)}</table>"

    issues = "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in s["check_issues"].items()) or "<tr><td colspan=2>none</td></tr>"

    file_rows = []
    for r in data["files"]:
        st = r["status"]
        px = r.get("pixel") or {}
        bad = [c for c in r.get("checks", []) if c["status"] != "PASS"]
        details = "".join(
            f"<li><b style='color:{STATUS_COLORS.get(c['status'], '#000')}'>{_e(c['status'])}</b> <code>{_e(c['id'])}</code> {_e(c['message'])}</li>"
            for c in r.get("checks", [])
        )
        changes = "".join(
            f"<tr><td><code>{_e(c['tag'])}</code></td><td>{_e(c['path'])}</td><td>{_e(c['action'])}</td><td>{_e(c['reason'])}</td><td>{_e(c.get('new') or '')}</td></tr>"
            for c in r.get("metadata_changes", [])
        )
        regions = "".join(
            f"<tr><td>{_e(d['decision'])}</td><td>{_e(', '.join(d['categories']))}</td><td>{_e(d['bbox'])}</td><td>{d['confidence']}</td><td>{_e(d.get('text') or d.get('preview') or ('•' * d['text_length']))}</td></tr>"
            for d in px.get("text_regions", [])
        )
        file_rows.append(
            f"""<details class='file'><summary><span class='chip' style='background:{STATUS_COLORS.get(st, '#000')}'>{_e(st)}</span>
            #{r['index']} <code>{_e(r.get('output') or '-')}</code> {_e(r.get('modality', ''))}
            · {len(r.get('metadata_changes', []))} tag changes · {px.get('redacted_regions', 0)} pixel regions · {r['timings'].get('total', 0)} s
            {'· ' + _e(bad[0]['message']) if bad else ''}</summary>
            {'<p class=err>' + _e(r.get('error')) + '</p>' if r.get('error') else ''}
            <h4>Audit checks</h4><ul>{details}</ul>
            <h4>Burned-in text</h4><table><tr><th>decision</th><th>category</th><th>box</th><th>conf</th><th>text</th></tr>{regions or '<tr><td colspan=5>none detected</td></tr>'}</table>
            <h4>Metadata transformations</h4><table><tr><th>tag</th><th>path</th><th>action</th><th>reason</th><th>new value</th></tr>{changes}</table>
            </details>"""
        )

    cfg = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in data["config"].items())
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>PixelGuard audit report</title><style>
:root{{--bg:#fff;--fg:#1f2328;--mut:#59636e;--line:#d1d9e0;--card:#f6f8fa}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--line:#30363d;--card:#161b22}}}}
body{{font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:1200px;margin:0 auto;padding:16px}}
.warn{{border:1px solid #9a6700;background:rgba(154,103,0,.08);padding:10px 14px;border-radius:6px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:16px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px}} .card b{{display:block;font-size:20px}}
.card span{{color:var(--mut);font-size:12px}} table{{border-collapse:collapse;width:100%;margin:6px 0;font-size:13px}}
td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}} .kv th{{width:280px;color:var(--mut);font-weight:500}}
.chip{{color:#fff;border-radius:10px;padding:1px 8px;font-size:12px;font-weight:600}} details.file{{border:1px solid var(--line);border-radius:6px;margin:6px 0;padding:6px 10px}}
summary{{cursor:pointer}} code{{font-size:12px}} .err{{color:#cf222e}} .mut{{color:var(--mut)}}
</style></head><body>
<h1>PixelGuard audit report</h1>
<p class=mut>Version {_e(data['version'])} · started {_e(data['started_utc'])} · {data['elapsed_seconds']} s</p>
<p class=warn><b>Not a guarantee of de-identification.</b> {_e(data['disclaimer'])}</p>
<div class=cards>{''.join(f"<div class=card><b>{_e(v)}</b><span>{_e(k)}</span></div>" for k, v in cards)}</div>
{ev_html}
<h2>Audit issues</h2><table><tr><th>check</th><th>files</th></tr>{issues}</table>
<h2>Files</h2>{''.join(file_rows)}
<h2>Configuration</h2><table class=kv>{cfg}</table>
<p class=mut>Original values are never written to this report. Changed values are identified by a keyed HMAC and length only.</p>
</body></html>"""
