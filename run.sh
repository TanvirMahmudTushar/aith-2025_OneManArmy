#!/bin/bash

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

echo "Running inference pipeline..."
python main.py
echo "Inference pipeline finished."
echo "Check output/ folder for predictions.csv"
