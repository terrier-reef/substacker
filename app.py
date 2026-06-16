import os
import queue
import shutil
import threading
import time
import uuid

from flask import Flask, Response, jsonify, render_template, request, send_file

from crawler import SubstackCrawler
from ebook_builder import build_epub

app = Flask(__name__)

# job_id -> {status, queue, output_path, filename, error}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

CLEANUP_AFTER = 600  # seconds


def _cleanup_job(job_id: str):
    time.sleep(CLEANUP_AFTER)
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if job:
        output = job.get("output_path")
        if output and os.path.exists(output):
            parent = os.path.dirname(output)
            shutil.rmtree(parent, ignore_errors=True)


def _run_job(job_id: str, url: str, limit: int):
    job = jobs[job_id]
    q: queue.Queue = job["queue"]

    def emit(msg: str):
        q.put(msg)

    try:
        crawler = SubstackCrawler(url, progress_callback=emit, limit=limit)
        meta, articles = crawler.crawl()

        if not articles:
            q.put("ERROR:No public articles found. The Substack may be private or the URL is incorrect.")
            return

        output_path = build_epub(meta, articles, crawler.base_url, progress_callback=emit)

        ext = os.path.splitext(output_path)[1]
        pub_name = (meta.get("name") or "substacker").strip()
        filename = f"{pub_name}{ext}"

        with jobs_lock:
            jobs[job_id]["output_path"] = output_path
            jobs[job_id]["filename"] = filename
            jobs[job_id]["status"] = "done"

        count = len(articles)
        emit(f"Done! {count} article{'s' if count != 1 else ''} compiled into {filename}")
        q.put(None)  # sentinel: done

    except Exception as e:
        q.put(f"ERROR:{e}")

    finally:
        threading.Thread(target=_cleanup_job, args=(job_id,), daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/build", methods=["POST"])
def build():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify(error="URL is required"), 400

    try:
        limit = int(data.get("limit") or 0)
    except ValueError:
        limit = 0

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "queue": queue.Queue(),
            "output_path": None,
            "filename": None,
            "error": None,
        }

    t = threading.Thread(target=_run_job, args=(job_id, url, limit), daemon=True)
    t.start()

    return jsonify(job_id=job_id)


@app.route("/progress/<job_id>")
def progress(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404

    def generate():
        q: queue.Queue = job["queue"]
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            if msg is None:
                yield "event: done\ndata: complete\n\n"
                break
            if isinstance(msg, str) and msg.startswith("ERROR:"):
                err = msg[6:]
                yield f"event: error\ndata: {err}\n\n"
                break
            yield f"data: {msg}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404
    return jsonify(
        status=job["status"],
        filename=job.get("filename"),
        has_output=bool(job.get("output_path")),
    )


@app.route("/download/<job_id>")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        return jsonify(error="File not ready"), 404

    filename = job.get("filename") or os.path.basename(output_path)
    return send_file(output_path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
