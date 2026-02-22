"""Brand Builder — Flask web application."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import traceback
import uuid
from pathlib import Path
from typing import Dict, Generator, Optional

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

load_dotenv()

import log_setup
log_setup.configure()

import db
import brand_core
import costs

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

db.init_db()

# Active SSE queues: run_id -> Queue
_run_queues: Dict[str, queue.Queue] = {}
_run_queues_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _get_or_create_queue(run_id: str) -> queue.Queue:
    with _run_queues_lock:
        if run_id not in _run_queues:
            _run_queues[run_id] = queue.Queue(maxsize=500)
        return _run_queues[run_id]


def _cleanup_queue(run_id: str) -> None:
    with _run_queues_lock:
        _run_queues.pop(run_id, None)


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def _run_pipeline_thread(
    run_id: str,
    concept: str,
    tone: str,
    seed: Optional[int],
    settings: Dict,
) -> None:
    q = _get_or_create_queue(run_id)
    images_dir = IMAGES_DIR / run_id

    def progress_cb(event: Dict) -> None:
        try:
            q.put_nowait(event)
        except queue.Full:
            pass

    db.set_run_running(run_id)
    log.info("Run started: id=%s  concept=%r  tone=%s", run_id, concept, tone)
    progress_cb({"stage": "pipeline", "status": "started", "message": "Pipeline started…"})

    cost_tracker = costs.CostTracker()

    try:
        pipeline = brand_core.BrandPipeline(
            run_id=run_id,
            concept=concept,
            tone=tone,
            seed=seed,
            settings=settings,
            images_dir=str(images_dir),
            progress_cb=progress_cb,
            cost_tracker=cost_tracker,
        )
        result = pipeline.run()

        # Convert absolute image paths to web-relative paths
        def to_web_path(p: str) -> str:
            if not p:
                return ""
            try:
                rel = Path(p).relative_to(BASE_DIR / "static")
                return f"/static/{rel}"
            except ValueError:
                return p  # already a URL

        web_images = {k: to_web_path(v) for k, v in result["images"].items()}

        cost_summary = cost_tracker.summary()
        db.complete_run(
            run_id,
            result["brand_data"],
            web_images,
            result["prompts"],
            result["api_calls"],
            result["duration"],
            cost_summary,
        )

        # Write cost log (non-blocking — errors here should not affect the run)
        try:
            costs.append_cost_log(
                run_id, concept, tone, result["duration"], cost_tracker
            )
        except Exception as ce:
            log.warning("Cost log write failed: %s", ce)

        log.info(
            "Run complete: id=%s  total_cost=~$%.4f  duration=%.1fs",
            run_id, cost_summary["total"], result["duration"],
        )

        progress_cb({
            "stage": "pipeline",
            "status": "complete",
            "message": "Done!",
            "data": {
                "brand_data": result["brand_data"],
                "images": web_images,
            },
        })

    except Exception as exc:
        err_msg = str(exc)
        log.error("Run failed: id=%s  error=%s", run_id, err_msg, exc_info=True)
        db.fail_run(run_id, err_msg)
        # Still record partial costs
        try:
            costs.append_cost_log(run_id, concept, tone, 0.0, cost_tracker)
        except Exception:
            pass
        progress_cb({
            "stage": "pipeline",
            "status": "failed",
            "message": err_msg,
            "data": {"traceback": traceback.format_exc()},
        })
    finally:
        # Signal SSE stream to close
        try:
            q.put_nowait(None)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Regeneration thread
# ---------------------------------------------------------------------------

def _regen_thread(
    run_id: str,
    stage: str,
    custom_prompt: Optional[str],
) -> None:
    q = _get_or_create_queue(run_id)
    run = db.get_run(run_id)
    if not run:
        return

    images_dir = IMAGES_DIR / run_id
    settings = run.get("settings") or {}
    # Get existing images as absolute paths
    existing_images = {}
    for k, v in (run.get("images") or {}).items():
        if v and v.startswith("/static/"):
            existing_images[k] = str(BASE_DIR / v.lstrip("/"))
        else:
            existing_images[k] = v

    def progress_cb(event: Dict) -> None:
        try:
            q.put_nowait(event)
        except queue.Full:
            pass

    try:
        pipeline = brand_core.BrandPipeline(
            run_id=run_id,
            concept=run["concept"],
            tone=run["tone"],
            seed=run.get("seed"),
            settings=settings,
            images_dir=str(images_dir),
            progress_cb=progress_cb,
        )
        new_path = pipeline.regenerate_stage(
            stage,
            run.get("brand_data") or {},
            existing_images,
            custom_prompt,
        )

        if new_path:
            try:
                rel = Path(new_path).relative_to(BASE_DIR / "static")
                web_path = f"/static/{rel}"
            except ValueError:
                web_path = new_path

            # Update DB
            images = run.get("images") or {}
            images[stage] = web_path
            db.update_run_images(run_id, images)

            progress_cb({
                "stage": stage,
                "status": "regenerated",
                "message": f"{stage.replace('_', ' ').title()} regenerated",
                "data": {"local_path": web_path},
            })
        else:
            progress_cb({
                "stage": stage,
                "status": "failed",
                "message": "Regeneration failed",
            })
    except Exception as exc:
        progress_cb({
            "stage": stage,
            "status": "failed",
            "message": str(exc),
        })
    finally:
        try:
            q.put_nowait(None)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — Static images (served from static/images/...)
# ---------------------------------------------------------------------------

@app.get("/static/images/<path:filename>")
def serve_image(filename: str):
    return send_from_directory(str(IMAGES_DIR), filename)


# ---------------------------------------------------------------------------
# Routes — Model discovery
# ---------------------------------------------------------------------------

@app.get("/api/models")
def api_models():
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    replicate_key = os.environ.get("REPLICATE_API_TOKEN", "")

    openai_models = brand_core.fetch_openai_models(openai_key) if openai_key else []
    anthropic_models = brand_core.fetch_anthropic_models(anthropic_key) if anthropic_key else []

    # Mark providers as available
    providers = []
    if openai_key:
        providers.append({
            "id": "openai",
            "name": "OpenAI",
            "available": True,
            "models": openai_models,
        })
    if anthropic_key:
        providers.append({
            "id": "anthropic",
            "name": "Anthropic (Claude)",
            "available": True,
            "models": anthropic_models,
        })
    if not providers:
        providers.append({
            "id": "openai",
            "name": "OpenAI (no key found)",
            "available": False,
            "models": [],
        })

    return jsonify({
        "text_providers": providers,
        "image_models": brand_core.IMAGE_MODELS,
        "edit_models": brand_core.EDIT_MODELS,
        "replicate_available": bool(replicate_key),
        "pipeline_stages": brand_core.PIPELINE_STAGES,
    })


# ---------------------------------------------------------------------------
# Routes — Run management
# ---------------------------------------------------------------------------

@app.post("/api/run")
def api_start_run():
    body = request.json or {}
    concept = (body.get("concept") or "").strip()
    tone = (body.get("tone") or "silly").strip()
    seed_raw = body.get("seed")
    settings = body.get("settings") or {}

    if not concept:
        return jsonify({"error": "concept is required"}), 400
    if tone not in ("silly", "serious", "scam"):
        return jsonify({"error": "tone must be silly, serious, or scam"}), 400

    seed = int(seed_raw) if seed_raw is not None else None

    # Validate API keys
    replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")
    provider = settings.get("text_provider", "openai")
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return jsonify({"error": "ANTHROPIC_API_KEY is not configured"}), 400
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            return jsonify({"error": "OPENAI_API_KEY is not configured"}), 400
    if not replicate_token:
        return jsonify({"error": "REPLICATE_API_TOKEN is not configured"}), 400

    run_id = str(uuid.uuid4())[:8]
    db.create_run(run_id, concept, tone, seed, settings)

    # Ensure queue exists before thread starts
    _get_or_create_queue(run_id)

    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(run_id, concept, tone, seed, settings),
        daemon=True,
    )
    t.start()

    return jsonify({"run_id": run_id})


@app.get("/api/stream/<run_id>")
def api_stream(run_id: str):
    """Server-Sent Events stream for a run."""
    q = _get_or_create_queue(run_id)

    def generate() -> Generator[str, None, None]:
        # Send a heartbeat first so the connection opens
        yield _sse_event({"type": "heartbeat", "run_id": run_id})
        try:
            while True:
                try:
                    event = q.get(timeout=25)
                except queue.Empty:
                    yield _sse_event({"type": "heartbeat"})
                    continue

                if event is None:
                    # Sentinel — pipeline finished
                    yield _sse_event({"type": "done"})
                    break

                yield _sse_event(event)

                # Stop streaming after terminal pipeline events
                if (
                    event.get("stage") == "pipeline"
                    and event.get("status") in ("complete", "failed")
                ):
                    yield _sse_event({"type": "done"})
                    break
        finally:
            _cleanup_queue(run_id)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/runs")
def api_list_runs():
    runs = db.list_runs()
    # Lightweight summary for the history list
    summary = []
    for r in runs:
        brand = r.get("brand_data") or {}
        images = r.get("images") or {}
        summary.append({
            "id": r["id"],
            "concept": r["concept"],
            "tone": r["tone"],
            "status": r["status"],
            "created_at": r["created_at"],
            "duration": r.get("duration"),
            "brand_name": brand.get("brand_name", ""),
            "tagline": brand.get("tagline", ""),
            "hero_image": images.get("hero_image", ""),
            "error_msg": r.get("error_msg", ""),
        })
    return jsonify(summary)


@app.get("/api/runs/<run_id>")
def api_get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        return jsonify({"error": "Not found"}), 404
    return jsonify(run)


@app.post("/api/runs/<run_id>/regenerate")
def api_regenerate(run_id: str):
    run = db.get_run(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    if run["status"] not in ("complete", "failed"):
        return jsonify({"error": "Run is still in progress"}), 400

    body = request.json or {}
    stage = body.get("stage", "")
    custom_prompt = body.get("prompt") or None

    valid_stages = [s["id"] for s in brand_core.PIPELINE_STAGES if s["id"] != "brand_text"]
    if stage not in valid_stages:
        return jsonify({"error": f"stage must be one of: {valid_stages}"}), 400

    # Ensure queue
    _get_or_create_queue(run_id)

    t = threading.Thread(
        target=_regen_thread,
        args=(run_id, stage, custom_prompt),
        daemon=True,
    )
    t.start()

    return jsonify({"run_id": run_id, "stage": stage})


@app.post("/api/runs/<run_id>/regenerate_text")
def api_regenerate_text(run_id: str):
    """Re-run just the brand text generation for a run."""
    run = db.get_run(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404

    settings = run.get("settings") or {}

    def regen():
        q = _get_or_create_queue(run_id)

        def progress_cb(event: Dict) -> None:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

        try:
            pipeline = brand_core.BrandPipeline(
                run_id=run_id,
                concept=run["concept"],
                tone=run["tone"],
                seed=run.get("seed"),
                settings=settings,
                images_dir=str(IMAGES_DIR / run_id),
                progress_cb=progress_cb,
            )
            brand_data = pipeline.generate_brand_text()
            with db._conn() as con:
                con.execute(
                    "UPDATE runs SET brand_data=? WHERE id=?",
                    (json.dumps(brand_data), run_id),
                )
            progress_cb({
                "stage": "brand_text",
                "status": "regenerated",
                "message": "Brand text regenerated",
                "data": {"brand_data": brand_data},
            })
        except Exception as exc:
            progress_cb({"stage": "brand_text", "status": "failed", "message": str(exc)})
        finally:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    _get_or_create_queue(run_id)
    threading.Thread(target=regen, daemon=True).start()
    return jsonify({"run_id": run_id, "stage": "brand_text"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Brand Builder → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
