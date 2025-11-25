#!/bin/bash

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

echo "Starting inference..."
echo "Using sample test data: Dataset/aith-dataset/sample_test_phase_1"
python inference.py --test_data_path Dataset/aith-dataset/sample_test_phase_1
echo "Inference complete. Results in output/"
