import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def track_step(step_name, func):
    """Helper to time and track each step."""
    print(f"\n{'='*60}")
    print(f"--- {step_name} START ---")
    print(f"{'='*60}")
    start = time.time()
    func()
    elapsed = time.time() - start
    print(f"\n--- {step_name} END | Elapsed: {elapsed:.2f} sec ---\n")

if __name__ == "__main__":
    pipeline_start = time.time()
    print("\n" + "="*60)
    print("===== Movie Recommendation Pipeline Started =====")
    print("===== Team: OneManArmy =====")
    print("="*60 + "\n")

    # Import and run inference
    from Inference.infer import main as run_inference
    
    track_step("Inference & Evaluation", run_inference)
    
    total_time = time.time() - pipeline_start
    print("\n" + "="*60)
    print("===== Pipeline Finished Successfully =====")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print("="*60 + "\n")
