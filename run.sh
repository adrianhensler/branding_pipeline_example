#!/usr/bin/env bash
# Start the Brand Builder web app
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo ""
echo "  ✦ Brand Builder"
echo "  → http://localhost:5000"
echo ""

python app.py
