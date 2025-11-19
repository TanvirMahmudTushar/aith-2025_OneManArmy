

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main inference module
from Inference.infer import main as run_inference

if __name__ == "__main__":
    print("="*70)
    print("  AITH 2025 - Movie Recommendation Inference")
    print("  Team: OneManArmy")
    print("="*70)
    print()
    
    # Execute inference
    run_inference()
    
    print()
    print("="*70)
    print("  Inference Complete!")
    print("  Results saved to: output/predictions.csv")
    print("="*70)
