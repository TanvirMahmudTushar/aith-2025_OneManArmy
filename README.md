# AITH 2025 - Movie Recommendation System

**Team Name:** OneManArmy  
**Competition:** AI Innovation Talent Hunt 2025 - Primary Phase  
**Institution:** Independent University, Bangladesh (IUB)

## Project Overview

This project implements a **SVD (Singular Value Decomposition) collaborative filtering** recommendation system for the AITH 2025 competition.


---



## Quick Start for Evaluation

### 1. Clone Repository

```bash
git clone https://github.com/TanvirMahmudTushar/aith-2025-movie-recommendation.git
cd aith-2025-movie-recommendation
```

### 2. Create Virtual Environment

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

### 4. Run Inference

```bash
python inference.py --test_data_path sample_test_phase_1
```



### 5. Check Output

Results saved to `output/` folder:
- `predictions.csv` - Predicted ratings
- `metrics.json` - Performance metrics

---



## Inference Parameters

```bash
python inference.py --test_data_path <folder_path>
```

**Optional parameters:**
- `--model_path` - Path to model file (default: `models/recommendation_model.pkl`)
- `--output_dir` - Output directory (default: `output`)

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



