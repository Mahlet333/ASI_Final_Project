#!/usr/bin/env bash
# run_pipeline.sh — Run the full A2S pipeline end-to-end.
#
# Usage: bash run_pipeline.sh [--skip-inference] [--demo]
#
# Options:
#   --skip-inference    Skip Step 3 (useful if you've already run inference)
#   --demo              Use synthetic data (no NormBank or API keys needed)

set -euo pipefail

SKIP_INFERENCE=0
DEMO=0

for arg in "$@"; do
  case $arg in
    --skip-inference) SKIP_INFERENCE=1 ;;
    --demo)           DEMO=1 ;;
  esac
done

echo "=============================================="
echo "  A2S Transfer Task — Full Pipeline"
echo "  CS-UH 3260 · NYU Abu Dhabi · Spring 2026"
echo "=============================================="

# Check if .venv exists
if [ ! -d ".venv" ]; then
  echo "⚠️  Virtual environment not found. Creating one..."
  python3 -m venv .venv
  echo "✅ Virtual environment created."
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
echo ""
echo "Step 0: Installing dependencies..."
pip install -r requirements.txt -q



if [ $SKIP_INFERENCE -eq 0 ]; then
  # Step 1: Inference
  echo ""
  echo "Step 3: Running inference across all models and levels..."
  if [ $DEMO -eq 1 ]; then
    echo "  (Demo mode: running only available models)"
  fi
  python src/inference_engine.py
else
  echo ""
  echo "Step 3: SKIPPED (--skip-inference flag set)"
fi

# Step 2: Evaluation
echo ""
echo "Step 4: Computing metrics and McNemar's test..."
python src/evaluator.py

# Step 3: Visualisation
echo ""
echo "Step 5: Generating figures..."
python src/visualizer.py

echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "  Results:  results/metrics_summary.json"
echo "  Figures:  results/figures/"
echo "  Analysis: notebooks/error_analysis.ipynb"
echo ""
echo "  If you haven't completed IRR annotation yet:"
echo "  1. Fill in data/constructed/irr_annotations.csv"
echo "  2. Re-run: python src/evaluator.py && python src/visualizer.py"
echo "=============================================="
