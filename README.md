# Brand Builder

A demo application that generates complete brand identities from two words and a tone. Combines an LLM for copy generation with Replicate's image models to produce a full brand kit: name, tagline, story, color palette, and 9 product/marketing images.

> **Purpose:** Demonstrating multi-API orchestration — LLM text generation + Replicate image generation + image editing in a single pipeline.

---

## What It Generates

| Stage | Output |
|---|---|
| Brand Text | Name, tagline, brand story, voice, color palette, typography, logo direction |
| Hero Image | Flagship product photography |
| Logo | Flat vector-style logo mark + wordmark |
| Product Variants | 45° angle, top-down flat lay, macro detail |
| Lifestyle | Product in use |
| Merch | T-shirt and hat mockups |

---

## Prerequisites

- Python 3.10+
- API keys for [OpenAI](https://platform.openai.com) and [Replicate](https://replicate.com)
- Optional: [Anthropic](https://console.anthropic.com) key to use Claude models for brand text

---

## Setup

```bash
git clone <repo>
cd branding_pipline_example

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Edit .env with your API keys (see .env.example)
cp .env .env.example
```

`.env` keys:
```
OPENAI_API_KEY=sk-...
REPLICATE_API_TOKEN=r8_...
ANTHROPIC_API_KEY=sk-ant-...    # optional
```

---

## Web Application

```bash
./run.sh
# → http://localhost:5000
```

### Features

- **Flow UI** — visual pipeline with live checkboxes as each stage completes
- **Real-time streaming** — SSE progress updates, images appear as they generate
- **Model selection** — text and image models fetched dynamically from APIs on load
- **Run history** — sidebar with hero thumbnails; click any past run to review
- **Regenerate** — ↻ button on any image; supports custom prompts
- **API inspector** — `</> API Calls` shows the exact payload sent to each API
- **Image persistence** — images are saved locally immediately (Replicate URLs expire)

---

## CLI

```bash
source .venv/bin/activate

# Basic run
python brand_cli.py --concept "moon shoes" --tone silly

# With HTML report saved to cli_output/
python brand_cli.py --concept "spy toaster" --tone serious --report

# Different models
python brand_cli.py --concept "diamond NFT" --tone scam \
  --text-model gpt-4o \
  --image-model black-forest-labs/flux-1.1-pro-ultra \
  --edit-model google/nano-banana

# List all available models (fetched live from APIs)
python brand_cli.py --list-models

# Full options
python brand_cli.py --help
```

---

## Image Models

| Model | Best For | Quality | Speed |
|---|---|---|---|
| `black-forest-labs/flux-schnell` | Testing, high volume | Good | Very fast |
| `black-forest-labs/flux-1.1-pro` | General use ★ default | Excellent | ~3s |
| `black-forest-labs/flux-1.1-pro-ultra` | Hero/product photography | Best | ~5s |
| `google/nano-banana` | Generation + editing, Gemini 2.5 | Excellent | ~3s |
| `google/nano-banana-pro` | Premium quality | Best | ~5s |
| `recraft-ai/recraft-v3` | Logos, icons, vector-style | Best for design | ~4s |
| `ideogram-ai/ideogram-v3-turbo` | Text-in-images, branded content | Best for text | ~4s |

The edit model (used for product variants, merch mockups) defaults to `flux-kontext-pro`.

---

## Cost Tracking

Every run writes to `logs/costs.log` — a formatted text file showing:

- Per-item breakdown: model, token counts / predict time, cost
- Per-run subtotals by API provider
- Running totals across all runs

LLM costs are **exact** (calculated from token counts). Replicate costs are **estimated** (flat per-image rate based on published pricing). Costs are not shown in the UI.

```
logs/costs.log           formatted human-readable log
logs/costs_totals.json   machine-readable running totals
logs/app.log             application log (rotating, 5 × 5MB)
```

---

## Architecture

```
app.py           Flask web server — routes, SSE streaming, run management
brand_core.py    Pipeline logic — LLM + Replicate calls, image download
brand_cli.py     CLI entry point
db.py            SQLite persistence (runs.db)
costs.py         Cost tracking and log formatting
log_setup.py     Logging configuration (console + rotating file)
templates/
  index.html     Single-page web UI
static/images/
  {run_id}/      Downloaded images, saved locally
logs/            App log + cost log
```

The pipeline uses [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) for real-time streaming. Each run gets a per-run `queue.Queue`; the generation thread puts events onto it; the SSE endpoint reads from it.

Images are downloaded from Replicate immediately after generation — Replicate URLs expire after ~1 hour.

---

## Tones

| Tone | Behaviour |
|---|---|
| `silly` | Playful, irreverent brand identity |
| `serious` | Premium, professional positioning |
| `scam` | Satirical over-the-top marketing copy |

---

## Extending

**Add a new image model:** Edit `IMAGE_MODELS` in `brand_core.py` and add its cost rate to `_REPLICATE_PRICING` in `costs.py`.

**Add a text provider:** Extend `_call_llm()` in `brand_core.py` and `fetch_*_models()` in the model discovery section.

**Change pipeline stages:** Edit `PIPELINE_STAGES` in `brand_core.py` and update `BrandPipeline.run()`.
