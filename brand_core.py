"""Core brand generation pipeline. Used by both the web app and CLI."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalogues (fetched at startup and cached in memory)
# ---------------------------------------------------------------------------

# Curated Replicate image models with metadata
IMAGE_MODELS: List[Dict] = [
    {
        "id": "black-forest-labs/flux-schnell",
        "name": "FLUX Schnell",
        "description": "Lightning-fast, cheap. Good for iteration and testing.",
        "tier": "fast",
        "supports_edit": False,
        "default_edit_model": "black-forest-labs/flux-kontext-pro",
    },
    {
        "id": "black-forest-labs/flux-1.1-pro",
        "name": "FLUX 1.1 Pro ★ Recommended",
        "description": "67M+ runs. Best general-purpose quality, excellent prompt adherence.",
        "tier": "pro",
        "supports_edit": False,
        "default_edit_model": "black-forest-labs/flux-kontext-pro",
    },
    {
        "id": "black-forest-labs/flux-1.1-pro-ultra",
        "name": "FLUX 1.1 Pro Ultra",
        "description": "Up to 4MP output, raw photo mode. Best for hero/product photography.",
        "tier": "ultra",
        "supports_edit": False,
        "default_edit_model": "black-forest-labs/flux-kontext-pro",
    },
    {
        "id": "google/nano-banana",
        "name": "Google Nano Banana (Gemini 2.5)",
        "description": "85M+ runs. Google's Gemini 2.5 image model. Supports generation & editing.",
        "tier": "google",
        "supports_edit": True,
        "default_edit_model": "google/nano-banana",
    },
    {
        "id": "google/nano-banana-pro",
        "name": "Google Nano Banana Pro (Gemini 2.5)",
        "description": "Premium Gemini 2.5. State-of-the-art generation and editing quality.",
        "tier": "google-pro",
        "supports_edit": True,
        "default_edit_model": "google/nano-banana-pro",
    },
    {
        "id": "recraft-ai/recraft-v3",
        "name": "Recraft V3",
        "description": "Best for logos, icons, vector art. Outstanding brand asset generation.",
        "tier": "design",
        "supports_edit": False,
        "default_edit_model": "black-forest-labs/flux-kontext-pro",
    },
    {
        "id": "ideogram-ai/ideogram-v3-turbo",
        "name": "Ideogram V3 Turbo",
        "description": "Best text-in-images rendering. Excellent for branded content with typography.",
        "tier": "text",
        "supports_edit": False,
        "default_edit_model": "black-forest-labs/flux-kontext-pro",
    },
    {
        "id": "bytedance/seedream-4",
        "name": "ByteDance Seedream 4",
        "description": "4K resolution output, strong text handling. Good for high-res brand assets.",
        "tier": "hires",
        "supports_edit": False,
        "default_edit_model": "black-forest-labs/flux-kontext-pro",
    },
]

EDIT_MODELS: List[Dict] = [
    {
        "id": "black-forest-labs/flux-kontext-pro",
        "name": "FLUX Kontext Pro ★ Recommended",
        "description": "46M+ runs. Best text-guided image editing. Used for product variants.",
    },
    {
        "id": "google/nano-banana",
        "name": "Google Nano Banana",
        "description": "Gemini 2.5 editing. Excellent for style-consistent image variations.",
    },
    {
        "id": "google/nano-banana-pro",
        "name": "Google Nano Banana Pro",
        "description": "Premium Gemini 2.5 editing. Best quality image modifications.",
    },
]

DEFAULT_IMAGE_MODEL = "black-forest-labs/flux-1.1-pro"
DEFAULT_EDIT_MODEL = "black-forest-labs/flux-kontext-pro"

# Pipeline stage definitions (in execution order with dependency info)
PIPELINE_STAGES = [
    {"id": "brand_text",      "label": "Brand Text",         "icon": "✦",  "type": "llm",      "depends_on": []},
    {"id": "hero_image",      "label": "Hero Image",         "icon": "★",  "type": "generate", "depends_on": ["brand_text"]},
    {"id": "logo_image",      "label": "Logo",               "icon": "◈",  "type": "generate", "depends_on": ["brand_text"]},
    {"id": "alt_base_image",  "label": "Base Product",       "icon": "◉",  "type": "generate", "depends_on": ["brand_text"]},
    {"id": "product_angle",   "label": "Product Angle",      "icon": "⬡",  "type": "edit",     "depends_on": ["alt_base_image"]},
    {"id": "product_topdown", "label": "Top Down",           "icon": "⬡",  "type": "edit",     "depends_on": ["alt_base_image"]},
    {"id": "product_macro",   "label": "Macro Detail",       "icon": "⬡",  "type": "edit",     "depends_on": ["alt_base_image"]},
    {"id": "product_in_use",  "label": "In Use",             "icon": "⬡",  "type": "edit",     "depends_on": ["hero_image"]},
    {"id": "merch_tshirt",    "label": "T-Shirt Mockup",     "icon": "⬡",  "type": "edit",     "depends_on": ["logo_image"]},
    {"id": "merch_hat",       "label": "Hat Mockup",         "icon": "⬡",  "type": "edit",     "depends_on": ["logo_image"]},
]


# ---------------------------------------------------------------------------
# Model discovery (dynamic, via API keys)
# ---------------------------------------------------------------------------

def fetch_openai_models(api_key: str) -> List[Dict]:
    """Return chat-capable GPT models sorted best-first."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        all_models = client.models.list()

        # Filter for useful chat models
        preferred_order = [
            "gpt-4.1", "gpt-4o", "gpt-4.1-mini", "gpt-4o-mini",
            "gpt-4.1-nano", "gpt-4-turbo", "gpt-3.5-turbo"
        ]

        found: Dict[str, Dict] = {}
        for m in all_models.data:
            mid = m.id
            # Keep GPT models only (skip embeddings, tts, etc.)
            if not any(mid.startswith(p) for p in ("gpt-4", "gpt-3.5")):
                continue
            # Skip non-chat variants
            if any(x in mid for x in ("audio", "tts", "realtime", "transcribe", "search", "instruct", "diarize")):
                continue
            # Skip dated snapshots if we already have the base
            base = mid.split("-202")[0].split(":")[0]
            if base not in found:
                label = mid
                if mid == "gpt-4.1":
                    label = "GPT-4.1 ★ Best Quality"
                elif mid == "gpt-4o":
                    label = "GPT-4o"
                elif mid in ("gpt-4.1-mini", "gpt-4o-mini"):
                    label = f"{mid} ★ Recommended (best cost/quality)"
                elif mid == "gpt-4.1-nano":
                    label = "GPT-4.1 Nano (fastest)"
                found[base] = {"id": mid, "name": label, "provider": "openai"}

        # Sort by preferred order
        result = []
        for pref in preferred_order:
            if pref in found:
                result.append(found.pop(pref))
        result.extend(found.values())

        # Default to gpt-4o-mini if available, else first
        for m in result:
            if "mini" in m["id"]:
                m["default"] = True
                break
        else:
            if result:
                result[0]["default"] = True

        return result
    except Exception as e:
        # Return fallback list on error
        return [
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini ★ Recommended", "provider": "openai", "default": True},
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
            {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini", "provider": "openai"},
            {"id": "gpt-4.1", "name": "GPT-4.1 (Best Quality)", "provider": "openai"},
        ]


def fetch_anthropic_models(api_key: str) -> List[Dict]:
    """Return available Anthropic models."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        models_page = client.models.list()

        result = []
        for m in models_page.data:
            mid = m.id
            if "claude" not in mid:
                continue
            label = mid
            if "opus" in mid:
                label = f"{mid} (Best Quality)"
            elif "sonnet" in mid:
                label = f"{mid} ★ Recommended"
            elif "haiku" in mid:
                label = f"{mid} (Fast & Cheap)"
            result.append({"id": mid, "name": label, "provider": "anthropic"})

        # Sort: haiku first (cheapest), then sonnet, then opus
        def rank(m: Dict) -> int:
            i = m["id"]
            if "haiku" in i: return 0
            if "sonnet" in i: return 1
            return 2
        result.sort(key=rank)

        # Default to sonnet if present, else haiku
        default_set = False
        for m in result:
            if "sonnet" in m["id"]:
                m["default"] = True
                default_set = True
                break
        if not default_set and result:
            result[0]["default"] = True

        return result
    except Exception:
        return [
            {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5 (Fast)", "provider": "anthropic", "default": True},
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6 ★ Recommended", "provider": "anthropic"},
            {"id": "claude-opus-4-6", "name": "Claude Opus 4.6 (Best Quality)", "provider": "anthropic"},
        ]


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class BrandPipeline:
    """Runs the full brand generation pipeline with real-time progress callbacks."""

    def __init__(
        self,
        run_id: str,
        concept: str,
        tone: str,
        seed: Optional[int],
        settings: Dict,
        images_dir: str,
        progress_cb: Callable[[Dict], None],
        cost_tracker: Optional[Any] = None,   # costs.CostTracker
    ) -> None:
        self.run_id = run_id
        self.concept = concept
        self.tone = tone
        self.seed = seed
        self.settings = settings
        self.images_dir = Path(images_dir)
        self.progress_cb = progress_cb
        self.cost_tracker = cost_tracker

        # Text generation settings
        self.text_provider: str = settings.get("text_provider", "openai")
        self.text_model: str = settings.get("text_model", "gpt-4o-mini")

        # Image generation settings
        self.image_model: str = settings.get("image_model", DEFAULT_IMAGE_MODEL)
        self.edit_model: str = settings.get("edit_model", DEFAULT_EDIT_MODEL)
        self.safety_tolerance: int = settings.get("safety_tolerance", 2)
        self.output_quality: int = settings.get("output_quality", 90)

        # Captured API calls for display
        self.api_calls: List[Dict] = []

        # Lazy-import replicate to use its token env
        self._replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")

        log.info(
            "Pipeline init: run=%s concept=%r tone=%s model=%s/%s image=%s edit=%s",
            run_id, concept, tone,
            self.text_provider, self.text_model,
            self.image_model, self.edit_model,
        )

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _emit(
        self,
        stage: str,
        status: str,
        message: str,
        data: Optional[Dict] = None,
    ) -> None:
        event: Dict[str, Any] = {
            "stage": stage,
            "status": status,
            "message": message,
            "ts": time.time(),
        }
        if data:
            event["data"] = data
        self.progress_cb(event)
        # Mirror to app log
        lvl = logging.WARNING if status == "failed" else logging.DEBUG
        log.log(lvl, "[%s] %s — %s", self.run_id, stage, message)

    def _capture_api_call(
        self,
        stage: str,
        api_type: str,
        model: str,
        payload: Dict,
        response_preview: str = "",
    ) -> None:
        self.api_calls.append(
            {
                "stage": stage,
                "api_type": api_type,
                "model": model,
                "payload": payload,
                "response_preview": response_preview,
                "ts": time.time(),
            }
        )

    # ------------------------------------------------------------------
    # Brand text generation
    # ------------------------------------------------------------------

    def generate_brand_text(self) -> Dict:
        self._emit("brand_text", "started", f"Generating brand identity via {self.text_model}…")

        system_prompt = (
            "You are a senior brand strategist and copywriter. "
            "Generate a cohesive brand kit and product copy. "
            "Return valid JSON only — no markdown fences, no commentary."
        )
        user_prompt = (
            "Schema:\n"
            "{\n"
            '  "brand_name": "...",\n'
            '  "tagline": "...",\n'
            '  "brand_story": "2-3 sentence paragraph",\n'
            '  "brand_voice": ["adjective", "adjective", "adjective"],\n'
            '  "color_palette": ["#RRGGBB", "#RRGGBB", "#RRGGBB", "#RRGGBB"],\n'
            '  "typography": {"headline": "Playfair Display", "body": "Inter"},\n'
            '  "imagery_style": "lighting, lens, texture, composition",\n'
            '  "product_description": "...",\n'
            '  "feature_bullets": ["...", "...", "...", "...", "..."],\n'
            '  "usage_examples": ["...", "..."],\n'
            '  "logo_direction": "..."\n'
            "}\n\n"
            f"Concept: {self.concept}\n"
            f"Tone: {self.tone}\n\n"
            "Rules: avoid generic filler; make copy feel premium and specific; "
            "safe, non-explicit, non-deceptive; no real brand names; "
            "strong contrast color palette."
        )

        payload_preview = {
            "model": self.text_model,
            "system": system_prompt,
            "user_prompt_preview": user_prompt[:300] + "…",
        }
        self._capture_api_call("brand_text", self.text_provider, self.text_model, payload_preview)

        try:
            text = self._call_llm(system_prompt, user_prompt)
            brand_data = self._parse_json(text)
            self._emit(
                "brand_text",
                "completed",
                f"Brand '{brand_data.get('brand_name', '')}' created",
                {
                    "brand_name": brand_data.get("brand_name"),
                    "tagline": brand_data.get("tagline"),
                },
            )
            return brand_data
        except Exception as exc:
            self._emit("brand_text", "failed", f"Brand text failed: {exc}")
            raise

    def _call_llm(self, system: str, user: str) -> str:
        t0 = time.time()
        if self.text_provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=self.text_model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            in_tok  = msg.usage.input_tokens
            out_tok = msg.usage.output_tokens
            log.info(
                "Anthropic call: model=%s  %d in / %d out tokens  %.1fs",
                self.text_model, in_tok, out_tok, time.time() - t0,
            )
            if self.cost_tracker:
                self.cost_tracker.record_llm(
                    "brand_text", self.text_model, "anthropic", in_tok, out_tok
                )
            return msg.content[0].text.strip()
        else:
            from openai import OpenAI, AuthenticationError, RateLimitError
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            client = OpenAI(api_key=api_key)
            try:
                resp = client.chat.completions.create(
                    model=self.text_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                )
                in_tok  = resp.usage.prompt_tokens
                out_tok = resp.usage.completion_tokens
                log.info(
                    "OpenAI call: model=%s  %d in / %d out tokens  %.1fs",
                    self.text_model, in_tok, out_tok, time.time() - t0,
                )
                if self.cost_tracker:
                    self.cost_tracker.record_llm(
                        "brand_text", self.text_model, "openai", in_tok, out_tok
                    )
                return resp.choices[0].message.content.strip()
            except AuthenticationError:
                raise RuntimeError("OpenAI API key is invalid or expired.")
            except RateLimitError as exc:
                msg = str(exc)
                if "insufficient_quota" in msg or "quota" in msg.lower():
                    raise RuntimeError(
                        "OpenAI account is out of credits. "
                        "Please add billing at platform.openai.com."
                    )
                raise RuntimeError(f"OpenAI rate limit: {exc}")

    @staticmethod
    def _parse_json(text: str) -> Dict:
        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
            raise

    # ------------------------------------------------------------------
    # Image prompt builder
    # ------------------------------------------------------------------

    def _build_prompts(self, brand: Dict) -> Dict[str, str]:
        palette = ", ".join(brand.get("color_palette", []))
        imagery = brand.get("imagery_style", "")
        voice = ", ".join(brand.get("brand_voice", []))
        brand_name = brand.get("brand_name", self.concept.title())
        base = (
            f"{brand_name} brand. {imagery}. "
            f"Color palette: {palette}. Voice: {voice}. Tone: {self.tone}. "
            "Safe, fully clothed, no explicit content, no real brands. "
            f"Incorporate the brand name '{brand_name}' in visual identity where appropriate."
        )
        return {
            "hero_image": (
                f"{base} Product hero shot of {self.concept}. "
                "Premium studio product photography, dramatic lighting."
            ),
            "logo_image": (
                f"{base} Flat vector logo mark and wordmark for {brand_name}. "
                "Centered on white background, clean geometry, high contrast, "
                "no product photography, no scene elements."
            ),
            "alt_base_image": (
                f"{base} Three-quarter angle product shot of {self.concept}, "
                "consistent studio lighting, neutral background."
            ),
            "product_angle": (
                f"{base} 45-degree three-quarter product angle of {self.concept}, "
                "crisp shadows, professional studio."
            ),
            "product_topdown": (
                f"{base} Top-down flat lay product shot of {self.concept}, "
                "clean minimal composition, styled props."
            ),
            "product_macro": (
                f"{base} Macro detail shot highlighting material and texture of {self.concept}, "
                "shallow depth of field."
            ),
            "product_in_use": (
                f"{base} Lifestyle shot: product in use by a person, "
                "neutral background, realistic natural light."
            ),
            "merch_tshirt": (
                f"{base} High-quality t-shirt mockup with {brand_name} logo "
                "applied cleanly and legibly on the chest."
            ),
            "merch_hat": (
                f"{base} Hat / cap mockup with {brand_name} logo "
                "applied cleanly and legibly on the front panel."
            ),
        }

    # ------------------------------------------------------------------
    # Replicate image generation
    # ------------------------------------------------------------------

    def _build_replicate_input(
        self,
        stage: str,
        prompt: str,
        input_image: Optional[str] = None,
    ) -> tuple[str, Dict]:
        """Return (model_slug, input_payload) for a Replicate prediction."""
        use_edit = input_image is not None
        model = self.edit_model if use_edit else self.image_model

        # Google nano-banana family
        if "nano-banana" in model:
            payload: Dict = {
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "png",
            }
            if input_image:
                payload["image"] = input_image
            if self.seed is not None and not use_edit:
                payload["seed"] = self.seed
            return model, payload

        # Recraft v3
        if "recraft" in model:
            payload = {
                "prompt": prompt,
                "size": "1024x1024",
                "style": "realistic_image",
            }
            return model, payload

        # Ideogram
        if "ideogram" in model:
            payload = {
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "resolution": "1024x1024",
            }
            return model, payload

        # ByteDance Seedream
        if "seedream" in model:
            payload = {
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "png",
            }
            return model, payload

        # FLUX Kontext (editing)
        if "kontext" in model and input_image:
            payload = {
                "prompt": prompt,
                "input_image": input_image,
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "safety_tolerance": self.safety_tolerance,
            }
            if self.seed is not None:
                payload["seed"] = self.seed
            return model, payload

        # FLUX Pro Ultra
        if "ultra" in model:
            payload = {
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "png",
                "raw": False,
            }
            if self.seed is not None:
                payload["seed"] = self.seed
            return model, payload

        # FLUX standard (flux-1.1-pro, flux-schnell, flux-dev)
        payload = {
            "prompt": prompt,
            "aspect_ratio": "1:1",
            "output_format": "png",
            "output_quality": self.output_quality,
            "safety_tolerance": self.safety_tolerance,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        # If editing is requested but model doesn't natively support it,
        # fall back to kontext-pro
        if input_image:
            fallback = DEFAULT_EDIT_MODEL
            payload = {
                "prompt": prompt,
                "input_image": input_image,
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "safety_tolerance": self.safety_tolerance,
            }
            if self.seed is not None:
                payload["seed"] = self.seed
            return fallback, payload

        return model, payload

    def _run_replicate(self, stage: str, model: str, payload: Dict) -> Optional[str]:
        """Submit a prediction to Replicate and return the output URL."""
        if not self._replicate_token:
            raise RuntimeError("REPLICATE_API_TOKEN not set")

        import replicate as rep

        # Sanitise payload for the API call log (don't embed full URLs/base64)
        log_payload = {
            k: (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v)
            for k, v in payload.items()
        }
        self._capture_api_call(stage, "replicate", model, {"model": model, "input": log_payload})

        t0 = time.time()
        try:
            client = rep.Client(api_token=self._replicate_token)

            # Use predictions API so we can read metrics (predict_time) for cost tracking
            try:
                prediction = client.predictions.create(model=model, input=payload)
                prediction.wait()
                predict_time = (prediction.metrics or {}).get("predict_time", 0.0)
                if prediction.status == "failed":
                    raise RuntimeError(f"Replicate prediction failed: {prediction.error}")
                raw_output = prediction.output
            except Exception as pred_err:
                # Fallback to simple .run() if predictions API fails (e.g. older model format)
                err_s = str(pred_err)
                if "404" in err_s or "not found" in err_s.lower() or "version" in err_s.lower():
                    log.debug("predictions.create failed for %s, falling back to run(): %s", model, err_s)
                    raw_output = client.run(model, input=payload)
                    predict_time = time.time() - t0
                else:
                    raise

            elapsed = time.time() - t0
            log.info(
                "Replicate [%s]: model=%s  predict=%.1fs  total=%.1fs",
                stage, model, predict_time, elapsed,
            )

            # Record cost
            if self.cost_tracker:
                self.cost_tracker.record_replicate(stage, model, predict_time)

            # Normalise output to URL string
            if isinstance(raw_output, list) and raw_output:
                raw = raw_output[0]
            else:
                raw = raw_output

            url = getattr(raw, "url", None) or str(raw)
            return url

        except Exception as exc:
            err = str(exc)
            elapsed = time.time() - t0
            log.error("Replicate error [%s] after %.1fs: %s", stage, elapsed, err)
            if "401" in err or "Authentication" in err.lower():
                raise RuntimeError(
                    "Replicate API token is invalid or expired. "
                    "Check your REPLICATE_API_TOKEN."
                )
            if "402" in err or "quota" in err.lower() or "payment" in err.lower():
                raise RuntimeError(
                    "Replicate account has insufficient credits. "
                    "Please add billing at replicate.com."
                )
            if "NSFW" in err or "nsfw" in err or "sensitive" in err.lower():
                self._emit(stage, "warning", "NSFW filter triggered — retrying with safe prompt")
                return None
            raise

    def _generate_image(
        self,
        stage: str,
        prompt: str,
        input_image: Optional[str] = None,
    ) -> Optional[str]:
        """Generate one image, download it, return local path."""
        label = stage.replace("_", " ").title()
        self._emit(stage, "started", f"Generating {label}…")

        model, payload = self._build_replicate_input(stage, prompt, input_image)

        # First attempt
        url = self._run_replicate(stage, model, payload)

        # NSFW retry
        if url is None:
            safe_prompt = (
                prompt
                + " Replace product with sealed packaging on white background. "
                "No people, no body parts, no suggestive content."
            )
            model2, payload2 = self._build_replicate_input(stage, safe_prompt, input_image)
            url = self._run_replicate(stage, model2, payload2)

        if not url:
            self._emit(stage, "failed", f"{label} could not be generated")
            return None

        # Download locally so images survive Replicate's expiry window
        local_path = self._download_image(stage, url)
        result_data = {
            "url": url,
            "local_path": local_path or url,
            "model": model,
        }
        self._emit(stage, "completed", f"{label} ready", result_data)
        return local_path or url

    def _download_image(self, stage: str, url: str) -> Optional[str]:
        try:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            local_path = self.images_dir / f"{stage}.png"
            resp = requests.get(url, timeout=90, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
            return str(local_path)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Execute the complete pipeline. Returns a result dict."""
        log.info("Pipeline start: run=%s  %r / %s", self.run_id, self.concept, self.tone)
        start = time.time()
        images: Dict[str, str] = {}

        # 1 — Brand text
        brand_data = self.generate_brand_text()
        prompts = self._build_prompts(brand_data)

        # 2 — Hero + logo + alt-base run in conceptual parallel
        # (we run them sequentially here; a future version could thread these)
        hero = self._generate_image("hero_image", prompts["hero_image"])
        images["hero_image"] = hero or ""

        logo = self._generate_image("logo_image", prompts["logo_image"])
        images["logo_image"] = logo or ""

        alt_base = self._generate_image("alt_base_image", prompts["alt_base_image"])
        images["alt_base_image"] = alt_base or ""

        # 3 — Edits from alt_base
        if alt_base:
            images["product_angle"] = (
                self._generate_image("product_angle", prompts["product_angle"], alt_base) or ""
            )
            images["product_topdown"] = (
                self._generate_image("product_topdown", prompts["product_topdown"], alt_base) or ""
            )
            images["product_macro"] = (
                self._generate_image("product_macro", prompts["product_macro"], alt_base) or ""
            )
        else:
            for s in ("product_angle", "product_topdown", "product_macro"):
                images[s] = ""
                self._emit(s, "skipped", "Skipped — base product image unavailable")

        # 4 — In-use edit from hero
        if hero:
            images["product_in_use"] = (
                self._generate_image("product_in_use", prompts["product_in_use"], hero) or ""
            )
        else:
            images["product_in_use"] = ""
            self._emit("product_in_use", "skipped", "Skipped — hero image unavailable")

        # 5 — Merch edits from logo
        if logo:
            images["merch_tshirt"] = (
                self._generate_image("merch_tshirt", prompts["merch_tshirt"], logo) or ""
            )
            images["merch_hat"] = (
                self._generate_image("merch_hat", prompts["merch_hat"], logo) or ""
            )
        else:
            for s in ("merch_tshirt", "merch_hat"):
                images[s] = ""
                self._emit(s, "skipped", "Skipped — logo unavailable")

        duration = time.time() - start
        n_images = sum(1 for v in images.values() if v)
        log.info(
            "Pipeline complete: run=%s  %.1fs  %d images",
            self.run_id, duration, n_images,
        )
        self._emit(
            "pipeline",
            "completed",
            f"Pipeline complete in {duration:.0f}s — {n_images} images generated",
            {"duration": duration, "image_count": n_images},
        )

        return {
            "brand_data": brand_data,
            "images": images,
            "prompts": prompts,
            "api_calls": self.api_calls,
            "duration": duration,
        }

    # ------------------------------------------------------------------
    # Single-stage regeneration
    # ------------------------------------------------------------------

    def regenerate_stage(
        self,
        stage: str,
        brand_data: Dict,
        existing_images: Dict[str, str],
        custom_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Regenerate a single pipeline stage. Returns new local path."""
        prompts = self._build_prompts(brand_data)
        prompt = custom_prompt or prompts.get(stage, "")

        # Determine whether this stage needs an input image
        input_image: Optional[str] = None
        if stage in ("product_angle", "product_topdown", "product_macro"):
            input_image = existing_images.get("alt_base_image") or None
        elif stage == "product_in_use":
            input_image = existing_images.get("hero_image") or None
        elif stage in ("merch_tshirt", "merch_hat"):
            input_image = existing_images.get("logo_image") or None

        # Use a versioned filename so we don't overwrite prior versions
        orig_dir = self.images_dir
        stage_path = orig_dir / f"{stage}.png"
        if stage_path.exists():
            import shutil
            # Archive previous version
            v = 1
            while (orig_dir / f"{stage}_v{v}.png").exists():
                v += 1
            shutil.copy2(stage_path, orig_dir / f"{stage}_v{v}.png")

        return self._generate_image(stage, prompt, input_image)
