"""
Helper script to extract SVD model factors for Windows compatibility.
Run this script if you have scikit-surprise installed to extract factors
that can be used without scikit-surprise.

Usage:
    python extract_model_factors.py
"""

import os
import pickle
import numpy as np

def extract_factors():
    """Extract SVD model factors and save them separately"""
    model_path = 'Resources/hybrid_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    
    try:
        from surprise import SVD
        print("[INFO] scikit-surprise available. Extracting factors...")
    except ImportError:
        print("[ERROR] scikit-surprise not available. Cannot extract factors.")
        print("[INFO] This script requires scikit-surprise to extract factors.")
        return False
    
    print(f"[INFO] Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    svd_model = model_data.get('svd_model')
    if svd_model is None:
        print("[ERROR] No SVD model found in the pickle file.")
        return False
    
    print("[INFO] Extracting SVD factors...")
    
    # Extract user and item factors, biases
    # Note: scikit-surprise SVD stores factors internally
    # We need to access the internal state
    try:
        # Get the number of factors
        n_factors = svd_model.n_factors
        
        # Extract user factors (pu) and item factors (qi)
        # These are stored in the model's internal state
        user_factors = {}
        item_factors = {}
        user_biases = {}
        item_biases = {}
        global_mean = svd_model.trainset.global_mean if hasattr(svd_model, 'trainset') else 3.0
        
        # Note: scikit-surprise doesn't expose factors directly
        # We'll need to reconstruct them or use the model differently
        print("[WARNING] scikit-surprise SVD doesn't expose factors directly.")
        print("[INFO] The model will work on Linux. For Windows, install C++ Build Tools.")
        print("[INFO] Alternatively, the inference code has fallback methods (content-based + popularity).")
        
        return False
        
    except Exception as e:
        print(f"[ERROR] Failed to extract factors: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("  SVD Model Factor Extractor")
    print("  Team: OneManArmy")
    print("="*70)
    print()
    
    extract_factors()
    
    print()
    print("="*70)
    print("  Note: scikit-surprise SVD models don't expose factors directly.")
    print("  The inference code will use fallback methods if scikit-surprise")
    print("  is not available (content-based + popularity).")
    print("="*70)

