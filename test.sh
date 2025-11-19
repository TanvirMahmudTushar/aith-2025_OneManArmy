#!/bin/bash

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

echo "Starting inference..."
python main.py
echo "Inference complete. Results in output/"
