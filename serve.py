#!/usr/bin/env python3
"""A small HTTP service around the converter.

    python serve.py --port 8000 --load-4bit

    curl -F file=@paper.pdf http://localhost:8000/convert
    # -> {"id": "3f2a...", "state": "queued"}
    curl http://localhost:8000/jobs/3f2a...
    # -> {"state": "done", "page": "/files/3f2a.../index.html"}

One job at a time.  A converter run is a Python process that owns the GPU
while it works; two at once do not make the card faster, they make the
allocator fail.  Jobs queue, and each one loads the model once and keeps it
for every page of that PDF.

The PDF is untrusted input: it is written into a fresh folder under
./work/<id>/ and never executed.
"""
import argparse
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
MAX_BYTES = 200 * 1024 * 1024

app = FastAPI(title="pdf2reading")
jobs = {}
lock = threading.Lock()
queue = Queue()
opts = argparse.Namespace(load_4bit=False, load_8bit=False,
                          max_new=2048)


def worker():
    while True:
        job_id = queue.get()
        job = jobs[job_id]
        job["state"] = "running"
        job["started"] = time.time()
        out = WORK / job_id / "out"
        cmd = [sys.executable, str(HERE / "corpus" / "infer.py"),
               "--pdf", job["pdf"], "--out", str(out),
               "--adapter", str(HERE / "adapter"),
               "--max-new", str(opts.max_new)]
        if opts.load_4bit:
            cmd.append("--load-4bit")
        if opts.load_8bit:
            cmd.append("--load-8bit")
        try:
            with open(WORK / job_id / "convert.log", "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            job["rc"] = rc
            job["state"] = "done" if rc == 0 else "failed"
        except Exception as e:                           # noqa: BLE001
            job["state"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"
        job["seconds"] = round(time.time() - job["started"], 1)
        queue.task_done()


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    d = WORK / job_id
    d.mkdir(parents=True)
    pdf = d / "in.pdf"
    n = 0
    with open(pdf, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            n += len(chunk)
            if n > MAX_BYTES:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, "PDF larger than 200 MB")
            f.write(chunk)
    if n == 0:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "empty upload")
    with lock:
        jobs[job_id] = {"state": "queued", "pdf": str(pdf), "name": file.filename,
                        "bytes": n, "created": time.time()}
    queue.put(job_id)
    return JSONResponse({"id": job_id, "state": "queued"}, status_code=202)


@app.get("/jobs/{job_id}")
def job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    out = {"id": job_id, "state": j["state"], "name": j.get("name"),
           "bytes": j.get("bytes")}
    if j["state"] == "done":
        out["page"] = f"/files/{job_id}/index.html"
        out["figures"] = len(list((WORK / job_id / "out" / "img").glob("*.png")))
        out["seconds"] = j.get("seconds")
    elif j["state"] == "failed":
        out["log"] = f"/files/{job_id}/convert.log"
    return out


@app.get("/files/{job_id}/{path:path}")
def files(job_id: str, path: str):
    d = (WORK / job_id).resolve()
    p = (d / path).resolve()
    if d not in p.parents or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p)


@app.get("/health")
def health():
    return {"ok": True, "queued": queue.qsize(),
            "states": [j["state"] for j in jobs.values()][-20:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--max-new", type=int, default=2048)
    a = ap.parse_args()
    opts.load_4bit, opts.load_8bit = a.load_4bit, a.load_8bit
    opts.max_new = a.max_new
    WORK.mkdir(exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
