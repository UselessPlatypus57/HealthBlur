import time

from fastapi.testclient import TestClient

from pixelguard.pixel.ocr import StaticOCREngine
from pixelguard.web.app import create_app


def _wait(client, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] not in ("queued", "running"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_upload_run_inspect_delete(tmp_path, synth_dir):
    app = create_app(tmp_path / "jobs", engine=StaticOCREngine())
    c = TestClient(app, base_url="http://127.0.0.1:8765")
    assert "PixelGuard" in c.get("/").text
    f = sorted(synth_dir.glob("*.dcm"))[0]
    r = c.post("/api/jobs", files=[("files", (f.name, f.read_bytes(), "application/dicom"))], data={"policy": "conservative"})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    j = _wait(c, jid)
    assert j["state"] == "done", j
    assert j["summary"]["files"] == 1
    rec = c.get(f"/api/jobs/{jid}/files/0").json()
    assert rec["checks"] and rec["previews"]["after"]
    assert c.get(f"/api/jobs/{jid}/preview/0/after.png").headers["content-type"] == "image/png"
    assert "PixelGuard audit report" in c.get(f"/api/jobs/{jid}/report.html").text
    assert c.get(f"/api/jobs/{jid}/download.zip").status_code == 200
    # uploaded file names are not kept (they may contain PHI)
    assert not list((tmp_path / "jobs" / jid / "input").glob(f.name))
    assert c.delete(f"/api/jobs/{jid}").status_code == 200
    assert c.get(f"/api/jobs/{jid}").status_code == 404


def test_demo_job_and_validation(tmp_path):
    c = TestClient(create_app(tmp_path / "jobs", engine=StaticOCREngine()), base_url="http://127.0.0.1:8765")
    assert c.post("/api/demo", data={"n": 2, "policy": "nonsense"}).status_code == 422
    jid = c.post("/api/demo", data={"n": 2}).json()["id"]
    j = _wait(c, jid)
    assert j["state"] == "done" and j["summary"]["files"] == 2


def test_dns_rebinding_host_is_refused(tmp_path):
    c = TestClient(create_app(tmp_path / "jobs", engine=StaticOCREngine()), base_url="http://evil.example:8765")
    assert c.get("/api/jobs").status_code == 403
    for host in ("http://localhost:8765", "http://127.0.0.1:8765", "http://[::1]:8765"):
        assert TestClient(create_app(tmp_path / "jobs", engine=StaticOCREngine()), base_url=host).get("/api/jobs").status_code == 200


def test_cross_site_writes_are_refused(tmp_path):
    c = TestClient(create_app(tmp_path / "jobs", engine=StaticOCREngine()), base_url="http://127.0.0.1:8765")
    r = c.post("/api/demo", data={"n": 1}, headers={"Origin": "https://some-website.example"})
    assert r.status_code == 403
    r = c.post("/api/demo", data={"n": 1}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    ok = c.post("/api/demo", data={"n": 1}, headers={"Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin"})
    assert ok.status_code == 200
    assert ok.headers["x-frame-options"] == "DENY" and ok.headers["cache-control"] == "no-store"


def test_upload_size_limit(tmp_path):
    c = TestClient(create_app(tmp_path / "jobs", engine=StaticOCREngine(), max_upload_mb=1), base_url="http://127.0.0.1:8765")
    big = b"\0" * (2 * 1024 * 1024)
    r = c.post("/api/jobs", files=[("files", ("a.dcm", big, "application/dicom"))])
    assert r.status_code == 413
    assert c.get("/api/jobs").json() == []  # nothing left behind


def _client(tmp_path, **kw):
    return TestClient(create_app(tmp_path / "jobs", engine=StaticOCREngine(), **kw), base_url="http://127.0.0.1:8765")


def _upload(c, synth_dir):
    f = sorted(synth_dir.glob("*.dcm"))[0]
    return c.post("/api/jobs", files=[("files", ("patient_name.dcm", f.read_bytes(), "application/dicom"))]).json()["id"]


def test_uploads_deleted_after_run_but_results_kept(tmp_path, synth_dir):
    c = _client(tmp_path)
    jid = _upload(c, synth_dir)
    j = _wait(c, jid)
    root = tmp_path / "jobs" / jid
    assert j["state"] == "done" and j["uploads_deleted"] is True
    assert not (root / "input").exists()
    assert list((root / "output").rglob("*.dcm")) and (root / "report" / "audit_report.json").exists()
    assert c.get(f"/api/jobs/{jid}/files/0").status_code == 200  # dashboard still works


def test_keep_uploads_option(tmp_path, synth_dir):
    c = _client(tmp_path, keep_uploads=True)
    jid = _upload(c, synth_dir)
    assert _wait(c, jid)["state"] == "done"
    assert list((tmp_path / "jobs" / jid / "input").glob("*.dcm"))


def test_uploads_deleted_even_when_run_fails(tmp_path, synth_dir, monkeypatch):
    import pixelguard.web.app as webapp

    def boom(*a, **k):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(webapp, "run_batch", boom)
    c = _client(tmp_path)
    jid = _upload(c, synth_dir)
    j = _wait(c, jid)
    assert j["state"] == "error" and "simulated crash" in j["error"]
    assert not (tmp_path / "jobs" / jid / "input").exists()


def test_retention_purges_expired_runs_only(tmp_path, synth_dir):
    app = create_app(tmp_path / "jobs", engine=StaticOCREngine(), retention_hours=1)
    c = TestClient(app, base_url="http://127.0.0.1:8765")
    jid = _upload(c, synth_dir)
    j = _wait(c, jid)
    assert abs(j["expires"] - j["created"] - 3600) < 1
    jm = app.state.jobs
    assert jm.purge_expired(now=time.time() + 600) == []  # not yet expired
    assert jm.purge_expired(now=time.time() + 2 * 3600) == [jid]
    assert not (tmp_path / "jobs" / jid).exists()
    assert c.get(f"/api/jobs/{jid}").status_code == 404


def test_retention_zero_keeps_runs(tmp_path, synth_dir):
    app = create_app(tmp_path / "jobs", engine=StaticOCREngine(), retention_hours=0)
    c = TestClient(app, base_url="http://127.0.0.1:8765")
    jid = _upload(c, synth_dir)
    assert _wait(c, jid)["expires"] is None
    assert app.state.jobs.purge_expired(now=time.time() + 10 * 365 * 86400) == []


def test_restart_clears_uploads_of_interrupted_runs(tmp_path):
    import json

    root = tmp_path / "jobs" / "old-run"
    (root / "input").mkdir(parents=True)
    (root / "input" / "upload_00000.dcm").write_bytes(b"phi")
    (root / "job.json").write_text(json.dumps({"id": "old-run", "label": "x", "state": "running", "created": time.time(), "options": {}}))
    c = _client(tmp_path)
    j = c.get("/api/jobs/old-run").json()
    assert j["state"] == "interrupted" and j["uploads_deleted"] is True
    assert not (root / "input").exists()
