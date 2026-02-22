# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Activate the venv first — always required
source .venv/bin/activate

# Web app (port 5000)
./run.sh
# or directly:
python app.py

# CLI
python brand_cli.py --concept "moon shoes" --tone silly
python brand_cli.py --concept "spy toaster" --tone serious --report
python brand_cli.py --list-models          # fetches live model lists from APIs

# Recreate the venv from scratch
python3 -m venv .venv && pip install -r requirements.txt
```

API keys live in `.env` (see `.env.example`). `OPENAI_API_KEY` and `REPLICATE_API_TOKEN` are required; `ANTHROPIC_API_KEY` is optional.

## Architecture

This is a multi-API demo: an LLM generates brand copy, then Replicate generates 9 images through a dependency chain. There is no test suite.

### Request flow (web)

1. `POST /api/run` creates a DB record and starts `_run_pipeline_thread` in a daemon thread
2. The browser opens `GET /api/stream/{run_id}` (SSE) which reads from a per-run `queue.Queue`
3. The thread runs `brand_core.BrandPipeline.run()`, calling `progress_cb(event)` after each stage
4. `progress_cb` puts events onto the queue; the SSE generator yields them as `data: {...}\n\n`
5. On completion, `costs.append_cost_log()` writes to `logs/costs.log` and the DB record is updated

### Module responsibilities

| Module | Role |
|---|---|
| `brand_core.py` | All external API calls (OpenAI/Anthropic + Replicate), image download, prompt building. `BrandPipeline` is the only public class. |
| `app.py` | Flask routes, SSE queue management, thread spawning. No business logic. |
| `db.py` | Thin SQLite wrapper (`runs` table). All JSON columns are deserialised on read. |
| `costs.py` | Pricing tables, `CostTracker` (accumulates per-run items), `append_cost_log()` (formats and appends to `logs/costs.log`). |
| `log_setup.py` | Call `configure()` once at startup. Console = INFO; `logs/app.log` = DEBUG, rotating 5×5MB. |

### Pipeline dependency chain

```
brand_text
  ├─ hero_image      → product_in_use (edit)
  ├─ logo_image      → merch_tshirt, merch_hat (edits)
  └─ alt_base_image  → product_angle, product_topdown, product_macro (edits)
```

Edit stages use `edit_model` (default: `flux-kontext-pro`); generation stages use `image_model` (default: `flux-1.1-pro`). `_build_replicate_input()` dispatches the correct payload shape per model family (FLUX, nano-banana, Recraft, Ideogram, etc.).

### Key data shapes

**`BrandPipeline.__init__` settings dict:**
```python
{
    "text_provider":    "openai" | "anthropic",
    "text_model":       str,   # e.g. "gpt-4o-mini"
    "image_model":      str,   # Replicate owner/name slug
    "edit_model":       str,
    "output_quality":   int,   # 60–100, FLUX only
    "safety_tolerance": int,   # 1–5, FLUX only
}
```

**SSE event shape** (emitted by `_emit()` and read by the browser):
```python
{"stage": str, "status": "started"|"completed"|"failed"|"skipped"|"regenerated",
 "message": str, "ts": float, "data": {...}}  # data is optional
```

**`runs` DB columns worth knowing:** `id`, `concept`, `tone`, `status` (`pending`/`running`/`complete`/`failed`), `brand_data` (JSON), `images` (JSON `{stage: "/static/images/{run_id}/{stage}.png"}`), `prompts` (JSON), `api_calls` (JSON), `cost_data` (JSON), `duration`.

### Adding a Replicate model

1. Add an entry to `IMAGE_MODELS` or `EDIT_MODELS` in `brand_core.py`
2. Add its payload shape as a branch in `_build_replicate_input()`
3. Add its estimated $/image to `_REPLICATE_PRICING` in `costs.py`

### Cost tracking

`CostTracker` is instantiated per run in `app.py` (and `brand_cli.py`) and passed to `BrandPipeline`. LLM calls record actual token counts; Replicate calls record `predict_time` from the predictions API. The log file structure: one block per run (chronological), running totals footer at end.
