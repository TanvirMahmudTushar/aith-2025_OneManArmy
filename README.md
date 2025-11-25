# AITH 2025 - Movie Recommendation System
## Team: OneManArmy

---

## Evaluation Instructions

Follow these steps to evaluate the system:

### 1. Clone Repository

```bash
git clone https://github.com/TanvirMahmudTushar/aith-2025_OneManArmy
```

### 2. Create Python Virtual Environment

**Important:** Use standard Python venv (no Anaconda/conda).

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** `scikit-surprise` is optional in `requirements.txt` to prevent installation failures. 

**Standard Installation (Most Cases):**
On most Linux systems, all dependencies install automatically because `scikit-surprise` installs automatically with standard build tools.

**If scikit-surprise Installation Fails:**
If `pip install -r requirements.txt` fails due to `scikit-surprise`, install build tools first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev gfortran libatlas-base-dev
pip install scikit-surprise
pip install -r requirements.txt
```

**Why scikit-surprise is Optional:**
The `requirements.txt` file has `scikit-surprise` commented out to prevent `pip install -r requirements.txt` from failing on systems without build tools. However, **the model file requires scikit-surprise to load**. The code includes:
- Fallback methods (content-based + popularity) that work without scikit-surprise
- Clear error messages if model loading fails
- Instructions for installing scikit-surprise

**Expected Behavior:**
1. **With scikit-surprise:** Full functionality (SVD + content + popularity)
2. **Without scikit-surprise:** Model loading will fail with clear instructions to install it

**For Judges:**
Judges should have build tools available on their Linux systems. If `scikit-surprise` installation fails, the error message will provide clear instructions. The code is designed to:
- Not fail during `pip install -r requirements.txt` (scikit-surprise is optional)
- Provide clear error messages if model loading fails
- Guide judges to install scikit-surprise if needed

### 4. Execute Inference

**Test Data Path:** Pass the test data folder path as a command-line argument using `--test_data_path`.

```bash
python inference.py --test_data_path Dataset/aith-dataset/sample_test_phase_1
```

**Test Data Format:**
The test data folder must contain:
- `known_reviewers_known_movies.csv`
- `known_reviewers_unknown_movies.csv`
- `unknown_reviewers_known_movies.csv`
- `movie_mapper.csv`

**Output Folder:** All results are saved to `output/` folder:
- `output/predictions.csv`
- `output/metrics.json`

---

## Additional Information

- **CPU Compatible:** Model runs on CPU (no GPU required)
- **Model File:** Included in repository (`Resources/hybrid_model.pkl`, 19.30 MB)
- **Runtime:** < 5 seconds for typical test sets
- **Memory:** < 2GB

---

## Training

Training notebook: `references/final_notebook.ipynb`

---
