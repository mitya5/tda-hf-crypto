#!/bin/bash
# Double-click this file to run the full project setup in Terminal.
# It will: install dependencies, fetch BTC/ETH data from Binance, and build RV targets.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
echo "=== TDA HF Crypto — Setup ==="
echo "Project dir: $PROJECT_DIR"
echo ""

# Install dependencies into whichever python3 is on PATH
echo "--- Installing dependencies ---"
python3 -m pip install -r requirements.txt
echo ""

# Fetch data (~500 MB, ~5–10 min)
echo "--- Fetching Binance data (BTC/USDT + ETH/USDT, 2020–2023) ---"
echo "This will take several minutes. Grab a coffee."
python3 src/utils/fetch_data.py
echo ""

# Build RV targets
echo "--- Building realized-volatility features ---"
python3 src/utils/build_rv.py
echo ""

echo "=== All done! Data is in data/processed/ ==="
read -p "Press Enter to close..."
