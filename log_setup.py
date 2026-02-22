"""Centralised logging configuration.

Call configure() once at startup (from app.py or brand_cli.py).
All modules then use logging.getLogger(__name__) normally.

Output:
  console  — INFO  level, compact single-line format
  logs/app.log — DEBUG level, full format, rotating (5 × 5 MB)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
LOG_FILE  = LOGS_DIR / "app.log"

_CONSOLE_FMT = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"
_FILE_FMT    = "%(asctime)s  %(levelname)-7s  %(name)-20s  %(filename)s:%(lineno)d — %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"


def configure(level: str = "INFO") -> None:
    """Set up console + rotating file handlers.  Safe to call multiple times."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        return  # Already configured

    root.setLevel(logging.DEBUG)  # lowest gate; handlers apply their own levels

    # ── Console ──
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(ch)

    # ── Rotating file ──
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    root.addHandler(fh)

    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "werkzeug", "openai", "anthropic", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
