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

If `scikit-surprise` installation fails, install build tools first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev gfortran libatlas-base-dev
pip install scikit-surprise
```

### 4. Execute Inference

```bash
python inference.py --test_data_path Dataset/aith-dataset/sample_test_phase_1 --output_dir output
```

**Arguments:**
- `--test_data_path`: Path to test data folder (required)
- `--model_path`: Path to model file (default: `Resources/hybrid_model.pkl`)
- `--output_dir`: Output directory (default: `output`)

**Test Data Format:**
The test data folder must contain:
- `known_reviewers_known_movies.csv`
- `known_reviewers_unknown_movies.csv`
- `unknown_reviewers_known_movies.csv`
- `movie_mapper.csv`

**Output Files:**
- `output/predictions.csv`
- `output/metrics.json`

---

## Additional Information

- **CPU Compatible:** Model runs on CPU (no GPU required). The model was trained and converted to CPU-compatible form. Unit testing has been performed to verify functionality.
- **Model File:** Included in repository (`Resources/hybrid_model.pkl`, 19.30 MB). If the model file is too large for GitHub, use `--download_model_url` to automatically download it.
- **Runtime:** < 5 seconds for typical test sets
- **Memory:** < 2GB

---

## Training

Training notebook: `references/final_notebook.ipynb`

---
