# Installation Guide - AITH 2025

## Quick Installation

```bash
pip install -r requirements.txt
```

## Platform-Specific Notes

### Linux / Mac (Recommended for Judges)
✅ All dependencies install automatically:
```bash
pip install -r requirements.txt
```
No additional setup required!

### Windows
⚠️ `scikit-surprise` requires C++ build tools:

**Option 1: Install Build Tools (Recommended)**
1. Download [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install "Desktop development with C++" workload
3. Run: `pip install -r requirements.txt`

**Option 2: Use WSL (Windows Subsystem for Linux)**
```bash
# Install WSL, then:
pip install -r requirements.txt
```

**Note:** Competition judges will use Linux, so installation will work automatically.

## Verify Installation

Run the setup check script:
```bash
python setup_check.py
```

Expected output:
```
✅ pandas
✅ numpy
✅ scipy
✅ scikit-learn
✅ scikit-surprise
✅ tqdm
✅ All dependencies installed successfully!
```

## Troubleshooting

### Error: "Microsoft Visual C++ 14.0 or greater is required"
**Solution:** Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Error: "No module named 'surprise'"
**Solution:** 
```bash
pip install scikit-surprise
```

### Error: "Failed building wheel for scikit-surprise"
**Solution:** This is a Windows-specific issue. Judges use Linux where this doesn't occur. For local Windows testing, install C++ Build Tools or use WSL.

## For Competition Judges

The evaluation environment will be Linux-based, where all dependencies install automatically:

```bash
# Standard evaluation steps (from competition rules):
git clone <repository-url>
cd MarriageChimeHackathon
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python inference.py --test_data_path <test_data_path>
```

All dependencies will install successfully on Linux without any additional setup.

