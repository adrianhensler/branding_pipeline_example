#!/usr/bin/env python3
"""CLI wrapper for the brand generation pipeline.

Usage:
    python brand_cli.py --concept "moon shoes" --tone silly --report
    python brand_cli.py --concept "spy toaster" --tone serious --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import log_setup
log_setup.configure()

import brand_core
import costs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a fake product brand (text + images)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python brand_cli.py --concept "moon shoes" --tone silly --report
  python brand_cli.py --concept "spy toaster" --tone serious --seed 42
  python brand_cli.py --concept "diamond NFT" --tone scam --image-model black-forest-labs/flux-1.1-pro-ultra
""",
    )
    parser.add_argument("--concept", default=None, help="Two-word product concept")
    parser.add_argument(
        "--tone",
        choices=["silly", "serious", "scam"],
        default=None,
        help="Brand tone",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--text-provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider for brand text (default: openai)",
    )
    parser.add_argument(
        "--text-model",
        default="gpt-4o-mini",
        help="LLM model for brand text (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--image-model",
        default=brand_core.DEFAULT_IMAGE_MODEL,
        help=f"Replicate model for image generation (default: {brand_core.DEFAULT_IMAGE_MODEL})",
    )
    parser.add_argument(
        "--edit-model",
        default=brand_core.DEFAULT_EDIT_MODEL,
        help=f"Replicate model for image editing/variants (default: {brand_core.DEFAULT_EDIT_MODEL})",
    )
    parser.add_argument(
        "--output-quality",
        type=int,
        default=90,
        help="Image output quality 60–100 (default: 90)",
    )
    parser.add_argument(
        "--safety-tolerance",
        type=int,
        default=2,
        choices=range(1, 6),
        help="Safety tolerance 1–5 (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        default="cli_output",
        help="Directory to save images and reports (default: cli_output)",
    )
    parser.add_argument("--report", action="store_true", help="Generate HTML brand report")
    parser.add_argument("--json", action="store_true", help="Print full JSON result to stdout")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    if args.list_models:
        _list_models()
        return 0

    if not args.concept:
        parser.error("--concept is required")
    if not args.tone:
        parser.error("--tone is required")

    # Validate env
    rep_token = os.environ.get("REPLICATE_API_TOKEN")
    if not rep_token:
        print("✗  REPLICATE_API_TOKEN not set", file=sys.stderr)
        return 2

    if args.text_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("✗  ANTHROPIC_API_KEY not set (required when using --text-provider anthropic)", file=sys.stderr)
        return 2
    if args.text_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("✗  OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    concept = args.concept.strip()
    output_dir = Path(args.output_dir) / f"{concept.replace(' ', '_')}_{args.tone}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "text_provider": args.text_provider,
        "text_model": args.text_model,
        "image_model": args.image_model,
        "edit_model": args.edit_model,
        "output_quality": args.output_quality,
        "safety_tolerance": args.safety_tolerance,
    }

    _echo(f"\n  ✦ Brand Builder CLI")
    _echo(f"  Concept : {concept}")
    _echo(f"  Tone    : {args.tone}")
    _echo(f"  LLM     : {args.text_provider}/{args.text_model}")
    _echo(f"  Images  : {args.image_model}")
    _echo(f"  Output  : {output_dir}\n")

    def progress_cb(event: dict) -> None:
        stage  = event.get("stage", "")
        status = event.get("status", "")
        msg    = event.get("message", "")
        prefix = {
            "started":    "  ◌ ",
            "completed":  "  ✓ ",
            "regenerated":"  ✓ ",
            "failed":     "  ✗ ",
            "skipped":    "  – ",
            "warning":    "  ⚠ ",
        }.get(status, "    ")
        _echo(f"{prefix}{msg}")

    run_id = f"cli-{int(time.time())}"

    cost_tracker = costs.CostTracker()
    try:
        pipeline = brand_core.BrandPipeline(
            run_id=run_id,
            concept=concept,
            tone=args.tone,
            seed=args.seed,
            settings=settings,
            images_dir=str(output_dir),
            progress_cb=progress_cb,
            cost_tracker=cost_tracker,
        )
        result = pipeline.run()
    except RuntimeError as exc:
        print(f"\n✗  {exc}", file=sys.stderr)
        return 1

    brand = result["brand_data"]
    images = result["images"]
    cost_summary = cost_tracker.summary()
    costs.append_cost_log(run_id, concept, args.tone, result["duration"], cost_tracker)

    _echo(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    _echo(f"  Brand   : {brand.get('brand_name', '')}")
    _echo(f"  Tagline : {brand.get('tagline', '')}")
    _echo(f"  Images  : {sum(1 for v in images.values() if v)}/{len(images)} generated")
    _echo(f"  Duration: {result['duration']:.1f}s")
    _echo(f"  Cost    : ~${cost_summary['total']:.4f}  (LLM ${cost_summary['openai_cost'] + cost_summary['anthropic_cost']:.4f} / Replicate ~${cost_summary['replicate_cost']:.4f})")
    _echo(f"  Output  : {output_dir}\n")

    if args.json:
        print(json.dumps({"brand": brand, "images": images}, indent=2))

    if args.report:
        _write_report(output_dir, brand, images, concept, args.tone)

    return 0


def _write_report(output_dir: Path, brand: dict, images: dict, concept: str, tone: str) -> None:
    """Write a simple standalone HTML brand report."""
    palette = brand.get("color_palette", ["#0B0B0D", "#F5F2EC", "#C7B9A8", "#7C6F64"])
    while len(palette) < 4:
        palette.append("#888888")

    def img_tag(stage: str, label: str) -> str:
        path = images.get(stage, "")
        if not path:
            return ""
        # Use relative path if possible
        try:
            rel = Path(path).relative_to(output_dir)
            src = str(rel)
        except ValueError:
            src = path
        return f'<figure><a href="{src}" target="_blank"><img src="{src}" alt="{label}"></a><figcaption>{label}</figcaption></figure>'

    gallery = "".join([
        img_tag("hero_image",      "Hero"),
        img_tag("logo_image",      "Logo"),
        img_tag("product_angle",   "Product Angle"),
        img_tag("product_topdown", "Top-Down"),
        img_tag("product_macro",   "Macro Detail"),
        img_tag("product_in_use",  "In Use"),
        img_tag("merch_tshirt",    "T-Shirt"),
        img_tag("merch_hat",       "Hat"),
    ])

    bullets = "".join(f"<li>{b}</li>" for b in brand.get("feature_bullets", []))
    usage   = "".join(f"<li>{u}</li>" for u in brand.get("usage_examples", []))

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{brand.get('brand_name','')}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&family=Playfair+Display:wght@500;700;800&display=swap" rel="stylesheet">
<style>
:root{{--ink:{palette[0]};--paper:{palette[1]};--accent:{palette[2]};--muted:{palette[3]}}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,sans-serif;color:#111;background:#f7f5f2}}
.page{{max-width:1120px;margin:0 auto;padding:48px 32px 96px}}
.cover{{display:grid;grid-template-columns:1.1fr 1fr;gap:32px;align-items:center;padding:32px 0 56px;border-bottom:1px solid rgba(0,0,0,.08)}}
h1{{font-family:'Playfair Display',serif;font-size:64px;line-height:1;letter-spacing:-.02em;margin:0 0 16px}}
.tagline{{font-size:20px;font-weight:500;color:var(--muted);margin:0 0 24px}}
.story{{font-size:18px;line-height:1.6}}
.hero{{width:100%;border-radius:18px;box-shadow:0 12px 40px rgba(0,0,0,.18)}}
.section{{margin-top:48px}}h2{{font-family:'Playfair Display',serif;font-size:28px;margin:0 0 16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px}}
figure{{margin:0;background:#fff;border-radius:12px;padding:10px;box-shadow:0 8px 24px rgba(0,0,0,.08)}}
figure img{{width:100%;border-radius:8px;display:block}}figcaption{{font-size:13px;color:var(--muted);margin-top:8px;text-transform:uppercase;letter-spacing:.08em}}
ul{{padding-left:20px;line-height:1.6}}
</style></head><body><div class="page">
<section class="cover">
<div><h1>{brand.get('brand_name','')}</h1><p class="tagline">{brand.get('tagline','')}</p><p class="story">{brand.get('brand_story','')}</p></div>
<div><img class="hero" src="{images.get('hero_image','')}" alt="Hero"></div>
</section>
<section class="section"><h2>Product</h2><p>{brand.get('product_description','')}</p><ul>{bullets}</ul></section>
<section class="section"><h2>Usage</h2><ul>{usage}</ul></section>
<section class="section"><h2>Logo Direction</h2><p>{brand.get('logo_direction','')}</p></section>
<section class="section"><h2>Gallery</h2><div class="grid">{gallery}</div></section>
</div></body></html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    _echo(f"  ✓ Report saved: {report_path}")


def _list_models() -> None:
    openai_key    = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    print("\nAvailable Text Models")
    print("─" * 40)
    if openai_key:
        print("OpenAI:")
        for m in brand_core.fetch_openai_models(openai_key):
            print(f"  {m['id']}")
    if anthropic_key:
        print("Anthropic:")
        for m in brand_core.fetch_anthropic_models(anthropic_key):
            print(f"  {m['id']}")

    print("\nAvailable Image Models (Replicate)")
    print("─" * 40)
    for m in brand_core.IMAGE_MODELS:
        print(f"  {m['id']}")
        print(f"    {m['description']}")

    print("\nAvailable Edit Models (Replicate)")
    print("─" * 40)
    for m in brand_core.EDIT_MODELS:
        print(f"  {m['id']}")
    print()


def _echo(msg: str) -> None:
    print(msg, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
