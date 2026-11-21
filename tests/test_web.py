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
    c = TestClient(app)
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
    c = TestClient(create_app(tmp_path / "jobs", engine=StaticOCREngine()))
    assert c.post("/api/demo", data={"n": 2, "policy": "nonsense"}).status_code == 422
    jid = c.post("/api/demo", data={"n": 2}).json()["id"]
    j = _wait(c, jid)
    assert j["state"] == "done" and j["summary"]["files"] == 2
