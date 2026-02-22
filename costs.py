"""Cost tracking for brand generation runs.

Pricing tables are approximate and updated periodically.
LLM costs are exact (calculated from token counts).
Replicate costs are estimated (per-image flat rate based on known pricing).

Outputs:
  logs/costs.log          — human-readable append-only log
  logs/costs_totals.json  — machine-readable running totals
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent / "logs"
COST_LOG  = LOGS_DIR / "costs.log"
TOTALS_FILE = LOGS_DIR / "costs_totals.json"

# ── Pricing tables ────────────────────────────────────────────────────────────
# OpenAI: (input $/1M tokens, output $/1M tokens)
_OPENAI_PRICING: Dict[str, tuple] = {
    "gpt-4.1":        (2.00,  8.00),
    "gpt-4o":         (2.50, 10.00),
    "gpt-4.1-mini":   (0.40,  1.60),
    "gpt-4o-mini":    (0.15,  0.60),
    "gpt-4.1-nano":   (0.10,  0.40),
    "gpt-4-turbo":    (10.00, 30.00),
    "gpt-4-0613":     (30.00, 60.00),
    "gpt-4":          (30.00, 60.00),
    "gpt-3.5-turbo":  (0.50,  1.50),
}
_OPENAI_DEFAULT = (2.50, 10.00)

# Anthropic: (input $/1M tokens, output $/1M tokens)
_ANTHROPIC_PRICING: Dict[str, tuple] = {
    "claude-opus-4":    (15.00, 75.00),
    "claude-sonnet-4":  (3.00,  15.00),
    "claude-haiku-4":   (0.80,   4.00),
    "claude-3-opus":    (15.00, 75.00),
    "claude-3-5-sonnet":(3.00,  15.00),
    "claude-3-haiku":   (0.25,   1.25),
}
_ANTHROPIC_DEFAULT = (3.00, 15.00)

# Replicate: estimated $/image (flat rate based on published pricing)
_REPLICATE_PRICING: Dict[str, float] = {
    "black-forest-labs/flux-schnell":       0.003,
    "black-forest-labs/flux-dev":           0.025,
    "black-forest-labs/flux-pro":           0.040,
    "black-forest-labs/flux-1.1-pro":       0.040,
    "black-forest-labs/flux-kontext-pro":   0.040,
    "black-forest-labs/flux-1.1-pro-ultra": 0.060,
    "google/nano-banana":                   0.050,
    "google/nano-banana-pro":               0.080,
    "recraft-ai/recraft-v3":                0.040,
    "ideogram-ai/ideogram-v3-turbo":        0.040,
    "ideogram-ai/ideogram-v3-quality":      0.060,
    "ideogram-ai/ideogram-v3-balanced":     0.050,
    "bytedance/seedream-4":                 0.050,
    "luma/photon":                          0.030,
}
_REPLICATE_DEFAULT = 0.040


# ── Pricing lookup helpers ────────────────────────────────────────────────────

def _openai_rate(model: str) -> tuple:
    """Return (input_rate, output_rate) per 1M tokens for an OpenAI model."""
    for prefix, rate in _OPENAI_PRICING.items():
        if model.startswith(prefix) or prefix in model:
            return rate
    log.debug("No OpenAI pricing match for '%s', using default", model)
    return _OPENAI_DEFAULT


def _anthropic_rate(model: str) -> tuple:
    """Return (input_rate, output_rate) per 1M tokens for an Anthropic model."""
    for prefix, rate in _ANTHROPIC_PRICING.items():
        if model.startswith(prefix) or prefix in model:
            return rate
    log.debug("No Anthropic pricing match for '%s', using default", model)
    return _ANTHROPIC_DEFAULT


def _replicate_rate(model: str) -> float:
    """Return estimated $/image for a Replicate model."""
    rate = _REPLICATE_PRICING.get(model)
    if rate is None:
        log.debug("No Replicate pricing match for '%s', using default", model)
        return _REPLICATE_DEFAULT
    return rate


# ── CostTracker ───────────────────────────────────────────────────────────────

class CostTracker:
    """Accumulates cost records for a single pipeline run."""

    def __init__(self) -> None:
        self.items: List[Dict] = []

    def record_llm(
        self,
        stage: str,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record an LLM API call and return the calculated cost in USD."""
        if provider == "anthropic":
            in_rate, out_rate = _anthropic_rate(model)
        else:
            in_rate, out_rate = _openai_rate(model)

        cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
        self.items.append(
            {
                "type": "llm",
                "stage": stage,
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "estimated": False,
            }
        )
        log.debug(
            "LLM cost [%s] %s/%s  %d in / %d out tokens  $%.6f",
            stage, provider, model, input_tokens, output_tokens, cost,
        )
        return cost

    def record_replicate(
        self,
        stage: str,
        model: str,
        predict_time: float = 0.0,
    ) -> float:
        """Record a Replicate prediction and return the estimated cost in USD."""
        cost = _replicate_rate(model)
        self.items.append(
            {
                "type": "replicate",
                "stage": stage,
                "provider": "replicate",
                "model": model,
                "predict_time": round(predict_time, 2),
                "cost": cost,
                "estimated": True,
            }
        )
        log.debug(
            "Replicate cost [%s] %s  %.1fs predict  ~$%.6f",
            stage, model, predict_time, cost,
        )
        return cost

    def summary(self) -> Dict:
        """Return a cost breakdown dict."""
        openai_cost    = sum(i["cost"] for i in self.items if i.get("provider") == "openai")
        anthropic_cost = sum(i["cost"] for i in self.items if i.get("provider") == "anthropic")
        replicate_cost = sum(i["cost"] for i in self.items if i.get("provider") == "replicate")
        return {
            "items":          self.items,
            "openai_cost":    openai_cost,
            "anthropic_cost": anthropic_cost,
            "replicate_cost": replicate_cost,
            "total":          openai_cost + anthropic_cost + replicate_cost,
            "has_estimates":  any(i.get("estimated") for i in self.items),
        }


# ── Totals persistence ────────────────────────────────────────────────────────

def _load_totals() -> Dict:
    if TOTALS_FILE.exists():
        try:
            return json.loads(TOTALS_FILE.read_text())
        except Exception:
            pass
    return {
        "run_count":       0,
        "openai_total":    0.0,
        "anthropic_total": 0.0,
        "replicate_total": 0.0,
        "grand_total":     0.0,
    }


def _save_totals(totals: Dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TOTALS_FILE.write_text(json.dumps(totals, indent=2))


def get_totals() -> Dict:
    """Return current running totals."""
    return _load_totals()


# ── Log writer ────────────────────────────────────────────────────────────────

_W = 81   # total log width
_DIV  = "─" * _W
_HDIV = "═" * _W


def append_cost_log(
    run_id: str,
    concept: str,
    tone: str,
    duration: float,
    tracker: CostTracker,
) -> None:
    """Append a formatted cost record to costs.log and update running totals."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    summary = tracker.summary()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    est_marker = "~" if summary["has_estimates"] else " "

    lines: List[str] = []

    def L(s: str = "") -> None:
        lines.append(s)

    L(_HDIV)
    L(f"  Run {run_id:<10}  {concept} ({tone})   {now}   {duration:.1f}s")
    L(_DIV)

    # ── LLM items ──
    llm_items = [i for i in summary["items"] if i["type"] == "llm"]
    if llm_items:
        L(f"  {'[LLM]':<12} {'Stage':<18} {'Model':<28} {'Tokens (in/out)':<22} {'Cost':>10}")
        L(f"  {'':12} {_DIV[:70]}")
        for item in llm_items:
            tok = f"{item['input_tokens']:,}↑ / {item['output_tokens']:,}↓"
            cost_str = f"${item['cost']:.6f}"
            L(
                f"  {'':12} {item['stage']:<18} {item['model']:<28}"
                f" {tok:<22} {cost_str:>10}"
            )

    # ── Replicate items ──
    rep_items = [i for i in summary["items"] if i["type"] == "replicate"]
    if rep_items:
        L(f"  {'[Replicate]':<12} {'Stage':<18} {'Model':<28} {'Predict time':<22} {'Est. Cost':>10}")
        L(f"  {'':12} {_DIV[:70]}")
        for item in rep_items:
            t = f"{item['predict_time']:.1f}s" if item["predict_time"] else "—"
            # shorten model slug for display
            model_short = item["model"].split("/")[-1][:27]
            cost_str = f"~${item['cost']:.6f}"
            L(
                f"  {'':12} {item['stage']:<18} {model_short:<28}"
                f" {t:<22} {cost_str:>10}"
            )

    # ── Subtotals ──
    L(_DIV)
    if summary["openai_cost"] > 0:
        L(f"  {'OpenAI:':<42} ${summary['openai_cost']:>12.6f}")
    if summary["anthropic_cost"] > 0:
        L(f"  {'Anthropic:':<42} ${summary['anthropic_cost']:>12.6f}")
    if summary["replicate_cost"] > 0:
        L(f"  {'Replicate (estimated):':<42}{est_marker}${summary['replicate_cost']:>12.6f}")
    L(f"  {'Run Total:':<42}{est_marker}${summary['total']:>12.6f}")
    L()

    # Append run block first, then update the totals footer at the end
    block = "\n".join(lines) + "\n"

    # Strip any existing totals footer so we can re-append it after the new run block
    if COST_LOG.exists():
        content = COST_LOG.read_text(encoding="utf-8")
        marker_pos = content.rfind("  RUNNING TOTALS  (all time)")
        if marker_pos != -1:
            hdr_pos = content.rfind(_HDIV, 0, marker_pos)
            if hdr_pos != -1:
                content = content[:hdr_pos]
            COST_LOG.write_text(content, encoding="utf-8")

    with open(COST_LOG, "a", encoding="utf-8") as fh:
        fh.write(block)

    # Update totals and write footer at end of file
    totals = _load_totals()
    totals["run_count"]       += 1
    totals["openai_total"]    += summary["openai_cost"]
    totals["anthropic_total"] += summary["anthropic_cost"]
    totals["replicate_total"] += summary["replicate_cost"]
    totals["grand_total"]     += summary["total"]
    _save_totals(totals)
    _write_totals_footer(totals)

    log.info(
        "Cost logged: run=%s  total=%s$%.4f  (OpenAI=$%.4f  Replicate=~$%.4f)",
        run_id, est_marker, summary["total"],
        summary["openai_cost"], summary["replicate_cost"],
    )


def _write_totals_footer(totals: Dict) -> None:
    """Append the running totals block to the end of costs.log."""
    footer_lines = [
        _HDIV,
        "  RUNNING TOTALS  (all time)",
        _DIV,
        f"  {'Total Runs:':<40}  {totals['run_count']:>6}",
    ]
    if totals["openai_total"] > 0:
        footer_lines.append(f"  {'OpenAI:':<40}  ${totals['openai_total']:>12.6f}")
    if totals["anthropic_total"] > 0:
        footer_lines.append(f"  {'Anthropic:':<40}  ${totals['anthropic_total']:>12.6f}")
    if totals["replicate_total"] > 0:
        footer_lines.append(f"  {'Replicate (estimated):':<40} ~${totals['replicate_total']:>12.6f}")
    footer_lines.append(f"  {'Grand Total:':<40} ~${totals['grand_total']:>12.6f}")
    footer_lines.append(_HDIV)
    footer_lines.append("")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(COST_LOG, "a", encoding="utf-8") as fh:
        fh.write("\n".join(footer_lines))
