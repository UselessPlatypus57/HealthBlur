"""Local web dashboard (FastAPI). Binds to 127.0.0.1 by default; uploaded
files and outputs stay in the work directory on this machine."""

from __future__ import annotations

import io
import json
import queue
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .. import DISCLAIMER, __version__
from ..config import Config, DateMode, RedactMethod, TextPolicy
from ..pipeline import discover, run_batch
from ..report import write_reports

STATIC = Path(__file__).parent / "static"


class JobManager:
    def __init__(self, workdir: Path, engine=None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.engine = engine  # injected in tests; None = EasyOCR
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        self._load_existing()
        threading.Thread(target=self._worker, daemon=True).start()

    def _load_existing(self):
        for d in sorted(self.workdir.iterdir()):
            jf = d / "job.json"
            if jf.exists():
                try:
                    job = json.loads(jf.read_text(encoding="utf-8"))
                    if job.get("state") in ("queued", "running"):
                        job["state"] = "interrupted"
                    self.jobs[job["id"]] = job
                except Exception:
                    pass

    def _save(self, job):
        (self.workdir / job["id"] / "job.json").write_text(json.dumps(job, default=str), encoding="utf-8")

    def create(self, options: dict, label: str) -> dict:
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        (self.workdir / jid / "input").mkdir(parents=True)
        job = {"id": jid, "label": label, "state": "queued", "created": time.time(), "options": options,
               "progress": {"done": 0, "total": 0}, "summary": None, "files": [], "error": None}
        self.jobs[jid] = job
        return job

    def submit(self, job):
        self._save(job)
        self.q.put(job["id"])

    def _worker(self):
        while True:
            jid = self.q.get()
            job = self.jobs.get(jid)
            if job is None:
                continue
            try:
                self._run(job)
            except Exception as exc:
                job["state"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
                self._save(job)

    def _run(self, job):
        root = self.workdir / job["id"]
        o = job["options"]
        cfg = Config(
            text_policy=TextPolicy(o.get("policy", "conservative")),
            redact_method=RedactMethod(o.get("redact", "fill")),
            date_mode=DateMode(o.get("dates", "remove")),
            clean_pixels=not o.get("metadata_only", False),
            quarantine_failures=True,
            report_previews=True,
        )
        if o.get("synthetic"):
            from ..synth import generate

            generate(root / "input", n=int(o.get("n", 8)), seed=int(o.get("seed", 21)), naive_fraction=0.4, hard_fraction=0.15)
        inputs = discover([root / "input"])
        job["state"] = "running"
        job["progress"] = {"done": 0, "total": len(inputs)}
        self._save(job)

        def progress(i, n, rec):
            job["progress"] = {"done": i, "total": n}
            job["files"].append(_light(rec))

        rep = run_batch(inputs, root / "output", cfg, engine=self.engine, preview_dir=root / "previews", progress=progress)
        write_reports(rep, root / "report")
        job["files"] = [_light(r) for r in rep["files"]]
        job["summary"] = rep["summary"]
        job["state"] = "done"
        self._save(job)


def _light(rec: dict) -> dict:
    bad = [c for c in rec.get("checks", []) if c["status"] in ("FAIL", "WARN")]
    px = rec.get("pixel") or {}
    return {
        "index": rec["index"], "status": rec["status"], "modality": rec.get("modality", ""),
        "issue": bad[0]["message"] if bad else (rec.get("error") or ""),
        "regions": px.get("redacted_regions", 0), "tag_changes": len(rec.get("metadata_changes", [])),
        "seconds": rec["timings"].get("total"), "output": rec.get("output"),
    }


LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
DEFAULT_MAX_UPLOAD_MB = 2048


def _hostname(value: str) -> str:
    """'127.0.0.1:8765' -> '127.0.0.1', '[::1]:8765' -> '::1', 'http://x:1' -> 'x'."""
    v = value.strip().lower()
    if "://" in v:
        v = v.split("://", 1)[1].split("/", 1)[0]
    if v.startswith("["):
        return v[1 : v.find("]")] if "]" in v else v
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def _security_middleware(app: FastAPI, allowed_hosts: set[str], max_upload_bytes: int) -> None:
    """The dashboard serves PHI (uploads and 'before' previews), so it only
    answers requests that genuinely come from this machine's browser tab:

    * Host header must be a loopback name: blocks DNS-rebinding pages that
      point their own domain at 127.0.0.1 to read job results.
    * State-changing requests must not come from another origin: blocks any
      website you visit from silently submitting or deleting jobs.
    * Request bodies are capped, and responses are not cached or framed.
    """

    @app.middleware("http")
    async def guard(request, call_next):
        from fastapi.responses import PlainTextResponse

        if _hostname(request.headers.get("host", "")) not in allowed_hosts:
            return PlainTextResponse("host not allowed", status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and _hostname(origin) not in allowed_hosts:
                return PlainTextResponse("cross-origin request refused", status_code=403)
            if request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
                return PlainTextResponse("cross-site request refused", status_code=403)
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > max_upload_bytes:
                return PlainTextResponse("upload too large", status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


def create_app(workdir: Path, engine=None, *, allowed_hosts=LOOPBACK_HOSTS, max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB) -> FastAPI:
    app = FastAPI(title="PixelGuard", version=__version__, docs_url=None, redoc_url=None)
    jm = JobManager(workdir, engine)
    app.state.jobs = jm
    max_bytes = max_upload_mb * 1024 * 1024
    _security_middleware(app, {_hostname(h) for h in allowed_hosts}, max_bytes)

    def job_or_404(jid: str) -> dict:
        job = jm.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    def report_of(jid: str) -> dict:
        p = jm.workdir / jid / "report" / "audit_report.json"
        if not p.exists():
            raise HTTPException(409, "report not ready")
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/static/{name}")
    def static(name: str):
        p = (STATIC / name).resolve()
        if p.parent != STATIC.resolve() or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/info")
    def info():
        return {"version": __version__, "disclaimer": DISCLAIMER}

    @app.get("/api/jobs")
    def list_jobs():
        return [{k: j[k] for k in ("id", "label", "state", "created", "progress", "summary")} for j in sorted(jm.jobs.values(), key=lambda j: -j["created"])]

    @app.post("/api/jobs")
    async def create_job(files: list[UploadFile] = File(...), policy: str = Form("conservative"), redact: str = Form("fill"),
                         dates: str = Form("remove"), metadata_only: bool = Form(False)):
        opts = {"policy": policy, "redact": redact, "dates": dates, "metadata_only": metadata_only}
        _validate(opts)
        job = jm.create(opts, f"{len(files)} uploaded file(s)")
        dest = jm.workdir / job["id"] / "input"
        total = 0
        for i, f in enumerate(files):
            # Uploaded file names can contain PHI; store under neutral names.
            with open(dest / f"upload_{i:05d}.dcm", "wb") as fh:
                while chunk := f.file.read(1 << 20):
                    total += len(chunk)
                    if total > max_bytes:  # chunked uploads carry no Content-Length
                        fh.close()
                        shutil.rmtree(jm.workdir / job["id"], ignore_errors=True)
                        jm.jobs.pop(job["id"], None)
                        raise HTTPException(413, "upload too large")
                    fh.write(chunk)
        jm.submit(job)
        return job

    @app.post("/api/demo")
    def demo(n: int = Form(8), policy: str = Form("conservative"), redact: str = Form("fill"), dates: str = Form("remove"), metadata_only: bool = Form(False)):
        opts = {"policy": policy, "redact": redact, "dates": dates, "metadata_only": metadata_only, "synthetic": True, "n": max(1, min(n, 60))}
        _validate(opts)
        job = jm.create(opts, f"synthetic demo ({opts['n']} files)")
        jm.submit(job)
        return job

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return job_or_404(jid)

    @app.get("/api/jobs/{jid}/files/{idx}")
    def get_file(jid: str, idx: int):
        job_or_404(jid)
        rep = report_of(jid)
        rec = next((r for r in rep["files"] if r["index"] == idx), None)
        if rec is None:
            raise HTTPException(404)
        rec = dict(rec)
        rec.pop("traceback", None)
        prev = jm.workdir / jid / "previews"
        rec["previews"] = {w: (prev / f"{idx:05d}_{w}.png").exists() for w in ("before", "after")}
        return rec

    @app.get("/api/jobs/{jid}/preview/{idx}/{which}.png")
    def preview(jid: str, idx: int, which: str):
        job_or_404(jid)
        if which not in ("before", "after"):
            raise HTTPException(404)
        p = jm.workdir / jid / "previews" / f"{idx:05d}_{which}.png"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{jid}/report.html", response_class=HTMLResponse)
    def report_html(jid: str):
        job_or_404(jid)
        p = jm.workdir / jid / "report" / "audit_report.html"
        if not p.exists():
            raise HTTPException(409, "report not ready")
        return p.read_text(encoding="utf-8")

    @app.get("/api/jobs/{jid}/report.json")
    def report_json(jid: str):
        job_or_404(jid)
        return JSONResponse(report_of(jid))

    @app.get("/api/jobs/{jid}/download.zip")
    def download(jid: str):
        job_or_404(jid)
        out = jm.workdir / jid / "output"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in out.rglob("*.dcm"):
                if "_quarantine" in f.parts:
                    continue
                z.write(f, f.relative_to(out))
            rp = jm.workdir / jid / "report"
            for f in rp.glob("audit_report.*"):
                z.write(f, f"report/{f.name}")
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="pixelguard_{jid}.zip"'})

    @app.delete("/api/jobs/{jid}")
    def delete(jid: str):
        job = job_or_404(jid)
        if job["state"] in ("queued", "running"):
            raise HTTPException(409, "job is still running")
        shutil.rmtree(jm.workdir / jid, ignore_errors=True)
        jm.jobs.pop(jid, None)
        return {"deleted": jid}

    return app


def _validate(o: dict):
    try:
        TextPolicy(o["policy"]), RedactMethod(o["redact"]), DateMode(o["dates"])
    except ValueError as exc:
        raise HTTPException(422, str(exc))
