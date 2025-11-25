"""
Setup Verification Script for AITH 2025
Checks if all required dependencies are installed correctly.
"""

import sys
import importlib

def check_dependency(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT INSTALLED")
        return False

def main():
    print("="*60)
    print("  AITH 2025 - Dependency Check")
    print("="*60)
    print()
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'surprise': 'scikit-surprise',
        'tqdm': 'tqdm',
    }
    
    results = []
    for module, package in required.items():
        results.append(check_dependency(module, package))
    
    print()
    print("="*60)
    
    if all(results):
        print("✅ All dependencies installed successfully!")
        print("   Your environment is ready for inference.")
        return 0
    else:
        print("❌ Some dependencies are missing!")
        print()
        print("To install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("Note: On Windows, if scikit-surprise fails to install,")
        print("      you may need Microsoft C++ Build Tools.")
        print("      Judges will use Linux where it installs automatically.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

