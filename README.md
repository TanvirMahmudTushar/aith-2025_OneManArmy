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

**Test Data Path:** Pass the test data folder path as a command-line argument using `--test_data_path`.

**Important:** A publicly available subset of test data is included in `Dataset/aith-dataset/sample_test_phase_1/`. This has the exact same format as the hidden test dataset. Use this to test your setup before submission.

```bash
python inference.py --test_data_path Dataset/aith-dataset/sample_test_phase_1 --output_dir output
```

**Arguments:**
- `--test_data_path`: Path to test data folder (required) - Pass the folder path as a command-line argument
- `--model_path`: Path to model file (default: `Resources/hybrid_model.pkl`)
- `--output_dir`: Output directory (default: `output`) - All inference outputs are saved here
- `--download_model_url`: URL to download model if local file doesn't exist (optional, for large model files)

**Test Data Format:**
The test data folder must contain:
- `known_reviewers_known_movies.csv`
- `known_reviewers_unknown_movies.csv`
- `unknown_reviewers_known_movies.csv`
- `movie_mapper.csv`

**Output Folder:** All inference outputs (predictions, metrics) are saved to the specified `--output_dir` folder:
- `output/predictions.csv` - Top 5 movie recommendations per user
- `output/metrics.json` - Recall@1, Recall@3, Recall@5 metrics

**Fallback Behavior:**
If `scikit-surprise` is not available, the system automatically uses content-based filtering and popularity-based fallback methods. The model file will be loaded without the SVD component, and predictions will still be generated.

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
