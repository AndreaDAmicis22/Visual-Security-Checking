"""
Web UI (FastAPI) del PPE Tracker.

Permette, da browser: scegliere il detector e i parametri, caricare un video in
drag-and-drop, avviare il tracking e scaricare il video annotato.

Il tracking su CPU e' lento (secondi/frame), quindi il processing gira in un
thread come JOB ASINCRONO: l'upload ritorna subito un job_id, il frontend fa
polling dello stato/avanzamento e a fine elaborazione scarica l'output.

Avvio:
    python webapp/app.py                 # http://127.0.0.1:8000
    # oppure:  uvicorn webapp.app:app    (singolo worker: i job sono in-memory)
"""
from __future__ import annotations

import shutil
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from visual_security.analyzer import DETECTOR_CHOICES  # noqa: E402
from visual_security.video_tracker import build_tracker  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = HERE / "_jobs"
WORK.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PPE Tracker Web UI")
JOBS: dict[str, dict] = {}


def _run_job(job_id: str, in_path: Path, params: dict) -> None:
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        out_path = WORK / f"{job_id}_output.mp4"
        tracker = build_tracker(
            detector=params["detector"],
            persistence_frames=params["persistence"],
            window_frames=params["window"],
            skip_frames=params["skip_frames"],
            ppe_memory_frames=params["ppe_memory"],
            detector_conf=params["conf"],
            display=False,
            save_output=str(out_path),
            alert_log=str(WORK / f"{job_id}_alerts.json"),
        )

        def cb(i: int, total: int) -> None:
            job["frame"], job["total"] = i, total
            job["progress"] = round(i / total, 3) if total else 0.0

        alerts = tracker.run(str(in_path), progress_cb=cb)
        job.update(
            status="done",
            progress=1.0,
            out=str(out_path),
            n_alerts=len(alerts),
            tracks=tracker.tracks_created,
            alerts=[a.summary() for a in alerts][:200],
        )
    except Exception as e:  # noqa: BLE001 — l'errore va mostrato in UI, non deve crashare il server
        job.update(status="error", error=f"{type(e).__name__}: {e}")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/detectors")
def detectors() -> dict:
    return {"detectors": list(DETECTOR_CHOICES)}


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    detector: str = Form("omdet-turbo"),
    skip_frames: int = Form(8),
    persistence: int = Form(3),
    window: int = Form(6),
    ppe_memory: int = Form(50),
    conf: str = Form(""),
) -> dict:
    if detector not in DETECTOR_CHOICES:
        raise HTTPException(400, f"detector non valido: {detector!r}")
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    in_path = WORK / f"{job_id}_input{suffix}"
    with open(in_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    params = {
        "detector": detector,
        "skip_frames": max(1, skip_frames),
        "persistence": max(1, persistence),
        "window": max(persistence, window),
        "ppe_memory": max(0, ppe_memory),
        "conf": float(conf) if conf.strip() else None,
    }
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "filename": video.filename, "params": params}
    threading.Thread(target=_run_job, args=(job_id, in_path, params), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job non trovato")
    return job


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "output non pronto")
    return FileResponse(job["out"], media_type="video/mp4", filename=f"tracked_{job['params']['detector']}.mp4")


if __name__ == "__main__":
    import uvicorn

    print("PPE Tracker Web UI  ->  http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
