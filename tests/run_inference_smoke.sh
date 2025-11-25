#!/usr/bin/env bash
# Smoke test script for AITH 2025 inference
# This script verifies that inference runs end-to-end and produces expected outputs

set -e  # Exit on error

echo "=========================================="
echo "AITH 2025 Inference Smoke Test"
echo "=========================================="

# Create test virtual environment
echo ""
echo "[1/4] Creating test virtual environment..."
python3 -m venv venv_test || python -m venv venv_test

# Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv_test/bin/activate || venv_test\Scripts\activate

# Install dependencies
echo ""
echo "[3/4] Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run inference
echo ""
echo "[4/4] Running inference..."
python inference.py \
  --test_data_path Dataset/aith-dataset/sample_test_phase_1 \
  --output_dir output_test

# Verify outputs exist
echo ""
echo "=========================================="
echo "Verifying outputs..."
echo "=========================================="

if [ -f output_test/predictions.csv ]; then
    echo "[OK] predictions.csv exists"
else
    echo "[FAIL] predictions.csv not found"
    exit 1
fi

if [ -f output_test/metrics.json ]; then
    echo "[OK] metrics.json exists"
else
    echo "[FAIL] metrics.json not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Smoke test passed!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - output_test/predictions.csv"
echo "  - output_test/metrics.json"
echo ""

