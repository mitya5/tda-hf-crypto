#!/bin/bash
# Double-click to run the baseline models (HAR-RV + XGBoost) on your machine.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
echo "=== Running baseline models ==="
PYTHONPATH=. python3 src/models/evaluation.py
echo ""
echo "Results saved to results/"
read -p "Press Enter to close..."
