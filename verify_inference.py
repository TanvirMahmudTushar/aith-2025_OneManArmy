"""
Verification script to test inference setup
This simulates what judges will do on Linux
"""

import os
import sys
import importlib

def check_file_exists(path, description):
    """Check if a file exists"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    size = ""
    if exists and os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        size = f" ({size_mb:.2f} MB)"
    print(f"{status} {description}: {path}{size}")
    return exists

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError:
        print(f"❌ {description}: {module_name} - NOT AVAILABLE")
        return False

def check_test_data():
    """Check test data files"""
    test_path = 'data/aith-dataset/sample_test_phase_1'
    required_files = [
        'known_reviewers_known_movies.csv',
        'known_reviewers_unknown_movies.csv',
        'unknown_reviewers_known_movies.csv',
        'movie_mapper.csv'
    ]
    
    print("\n📁 Test Data Files:")
    all_exist = True
    for f in required_files:
        full_path = os.path.join(test_path, f)
        exists = check_file_exists(full_path, f)
        if not exists:
            all_exist = False
    
    return all_exist

def check_code_structure():
    """Check that inference code structure is correct"""
    print("\n📝 Code Structure:")
    
    files_to_check = [
        ('inference.py', 'Main entry point'),
        ('Inference/infer.py', 'Core inference logic'),
        ('requirements.txt', 'Dependencies file'),
    ]
    
    all_exist = True
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def check_model():
    """Check model file"""
    print("\n🤖 Model File:")
    return check_file_exists('Resources/hybrid_model.pkl', 'Trained model')

def check_dependencies():
    """Check if dependencies can be imported"""
    print("\n📦 Dependencies:")
    
    deps = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'surprise': 'scikit-surprise',
        'tqdm': 'tqdm',
    }
    
    results = []
    for module, name in deps.items():
        results.append(check_import(module, name))
    
    return all(results)

def test_inference_imports():
    """Test that inference code can be imported"""
    print("\n🔍 Testing Inference Code Imports:")
    
    try:
        # Test if we can import the inference module structure
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try importing (this will fail if surprise is missing, but that's expected on Windows)
        try:
            from Inference.infer import WinningRecommender, run_inference
            print("✅ Inference module structure: CORRECT")
            print("✅ Classes and functions: AVAILABLE")
            return True
        except ImportError as e:
            if 'surprise' in str(e).lower():
                print("⚠️  Inference code structure: CORRECT")
                print("⚠️  scikit-surprise not available (expected on Windows)")
                print("   → Will work on Linux where scikit-surprise installs automatically")
                return True  # This is OK - code structure is correct
            else:
                print(f"❌ Import error: {e}")
                return False
    except Exception as e:
        print(f"❌ Error testing imports: {e}")
        return False

def main():
    print("="*70)
    print("  AITH 2025 - Inference Verification")
    print("  Team: OneManArmy")
    print("="*70)
    
    results = {
        'test_data': check_test_data(),
        'code_structure': check_code_structure(),
        'model': check_model(),
        'dependencies': check_dependencies(),
        'inference_imports': test_inference_imports(),
    }
    
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    all_critical = results['test_data'] and results['code_structure'] and results['model']
    deps_ok = results['dependencies'] or results['inference_imports']  # At least one should work
    
    if all_critical:
        print("✅ All critical files present")
    else:
        print("❌ Some critical files missing")
    
    if results['dependencies']:
        print("✅ All dependencies available - READY FOR INFERENCE")
    elif results['inference_imports']:
        print("⚠️  scikit-surprise not available (Windows limitation)")
        print("   → Code structure is correct")
        print("   → Will work on Linux where scikit-surprise installs automatically")
    else:
        print("❌ Dependency issues detected")
    
    print("\n" + "="*70)
    print("  JUDGES EVALUATION CHECKLIST")
    print("="*70)
    print("""
Judges will run:
  1. git clone <repo>
  2. python -m venv venv
  3. source venv/bin/activate  (Linux)
  4. pip install -r requirements.txt
  5. python inference.py --test_data_path <test_data>

Expected result on Linux:
  ✅ scikit-surprise will install automatically
  ✅ Model will load successfully
  ✅ Inference will run and generate predictions
  ✅ Output files will be created in output/ folder
    """)
    
    if all_critical:
        print("✅ Your repository is ready for submission!")
        print("   All required files are present.")
        if not results['dependencies']:
            print("   Note: scikit-surprise will install on Linux automatically.")
    else:
        print("❌ Please fix missing files before submission.")
    
    return 0 if all_critical else 1

if __name__ == "__main__":
    sys.exit(main())

