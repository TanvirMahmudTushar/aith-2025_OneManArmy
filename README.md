# AITH 2025 - Movie Recommendation System
## Team: OneManArmy

Hybrid recommendation system combining SVD collaborative filtering with IMDB user profiles for superior cold-start handling.

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <https://github.com/TanvirMahmudTushar/aith-2025-movie-recommendation>
cd MarriageChimeHackathon

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference

**Important:** The test data folder path must be provided as a command-line argument.

```bash
# Using sample test data
python inference.py --test_data_path Dataset/aith-dataset/sample_test_phase_1

# Or specify custom path
python inference.py --test_data_path <path_to_test_data_folder>
```

**Test Data Format:**
The test data folder must contain:
- `known_reviewers_known_movies.csv`
- `known_reviewers_unknown_movies.csv`
- `unknown_reviewers_known_movies.csv`
- `movie_mapper.csv`

### 3. Check Results

Results are saved in the `output/` folder:
- `output/predictions.csv` - Predicted scores for each user-movie pair
- `output/metrics.json` - Recall@K metrics and execution time

---


```


## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

All dependencies install automatically on Linux. On Windows, `scikit-surprise` may require C++ build tools, but the code includes automatic fallback methods.

---

## Training

Training was performed in Google Colab. See `references/final_notebook.ipynb` for the complete training notebook.

---

## Evaluation Compliance

✅ **inference.py** - Main entry point exists  
✅ **Test data path** - Passed as `--test_data_path` argument  
✅ **CPU compatible** - Model runs on CPU  
✅ **Model included** - `Resources/hybrid_model.pkl` (19.30 MB)  
✅ **Output folder** - Results saved to `output/` folder  
✅ **Requirements.txt** - All dependencies listed  

---


