#!/bin/bash

# Training script for AITH 2025
# Team: OneManArmy

echo "=========================================="
echo "AITH 2025 - Training Script"
echo "Team: OneManArmy"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

echo "Note: Training is done in Google Colab."
echo "See references/final_notebook.ipynb for the training notebook."
echo ""
echo "To train the model:"
echo "1. Upload references/final_notebook.ipynb to Google Colab"
echo "2. Run all cells"
echo "3. Download Resources/hybrid_model.pkl"
echo "4. Place it in Resources/ directory"
echo ""
echo "The trained model is already included in Resources/hybrid_model.pkl"
echo "You can proceed directly to inference using: bash run.sh"

