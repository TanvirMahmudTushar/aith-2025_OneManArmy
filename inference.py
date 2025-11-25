"""
AITH 2025 - Movie Recommendation Inference Entry Point
Team: OneManArmy

This is the main entry point for inference.

Usage:
    python inference.py --test_data_path sample_test_phase_1

Arguments:
    --test_data_path: Path to folder containing test CSV files (default: sample_test_phase_1)
    --model_path: Path to model file (default: Resources/hybrid_model.pkl)
    --output_dir: Output directory (default: output)

Output:
    - output/predictions.csv: Predicted scores for each user-movie pair
    - output/metrics.json: Recall@K metrics and execution time
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main inference module
from Inference.infer import main

if __name__ == "__main__":
    # Execute inference (banner is printed in main)
    main()
