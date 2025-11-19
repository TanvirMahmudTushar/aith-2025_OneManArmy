# AITH 2025 - Movie Recommendation System

**Team Name:** OneManArmy  
**Competition:** AI Innovation Talent Hunt 2025 - Primary Phase  
**Institution:** Independent University, Bangladesh (IUB)

## Project Overview

This project implements a **SVD (Singular Value Decomposition) collaborative filtering** recommendation system for the AITH 2025 competition. The model achieves **Recall@5: 0.3821**, runs on **CPU**, and completes inference in under 5 seconds.

---

## Evaluation Instructions (IMPORTANT)

**Follow these exact steps to evaluate:**

### 1. Clone Repository

```bash
git clone https://github.com/TanvirMahmudTushar/aith-2025-movie-recommendation.git
cd aith-2025-movie-recommendation
```

### 2. Create Python Virtual Environment

**IMPORTANT:** Use Python venv (NOT Anaconda/conda)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Execute Inference

**Method 1: With test data folder path (Recommended)**
```bash
python inference.py --test_data_path sample_test_phase_1
```

**Method 2: Using default path**
```bash
python inference.py
```
(Default path: `sample_test_phase_1`)

---

## Inference Parameters

```bash
python inference.py --test_data_path <folder_path>
```

**Arguments:**
- `--test_data_path`: Path to folder containing test CSV files (default: `sample_test_phase_1`)
- `--model_path`: Path to model file (default: `Resources/recommendation_model.pkl`)
- `--output_dir`: Output directory (default: `output`)

**Example:**
```bash
python inference.py --test_data_path hidden_test_phase_1 --output_dir output
```

---

## Output Location

All inference results are saved to the **`output/`** folder:
- **`output/predictions.csv`** - Predicted ratings (userId, movieId, prediction)
- **Console output** - Recall@5, Recall@3, Recall@1, RMSE, MAE

---

## Alternative: Shell Scripts

For convenience, you can also use:

```bash
bash setup.sh    # Setup environment
bash run.sh      # Run inference (calls main.py)
bash test.sh     # Run inference (calls main.py)
```

**Note:** For official evaluation, use `python inference.py` as specified above.

---

## Model Details

**Algorithm:** SVD (Singular Value Decomposition) Collaborative Filtering

**Hyperparameters:**
- Latent Factors: 100
- Epochs: 30
- Learning Rate: 0.005
- Regularization: 0.02

**Performance:**
- Recall@5: 0.3821
- Recall@3: 0.2816
- Recall@1: 0.1138
- RMSE: 0.8828
- MAE: 0.6767

**Model Size:** 10.73 MB  
**Training Time:** 2.02 seconds (on Google Colab)

---

## Training

Training notebook available at: `notebooks/AITH2025_FINAL_Training.ipynb`



