================================================================================
AITH 2025 - Movie Recommendation System
Team: OneManArmy
================================================================================

QUICK START FOR JUDGES:
-----------------------

1. Clone repository:
   git clone <repository-url>
   cd MarriageChimeHackathon

2. Create virtual environment:
   python -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run inference:
   python inference.py --test_data_path <path_to_test_data>

5. Check outputs:
   - output/predictions.csv
   - output/metrics.json

================================================================================
DEPENDENCIES:
-------------

All dependencies in requirements.txt will install automatically on Linux:
- pandas>=2.0.0
- numpy>=1.24.0
- scipy>=1.11.0
- scikit-learn>=1.3.0
- scikit-surprise>=1.1.4
- tqdm>=4.66.0

No additional setup required on Linux!

================================================================================
MODEL FILE:
-----------

The trained model is located at:
  Resources/hybrid_model.pkl

This file is included in the repository (19.30 MB).

================================================================================
OUTPUT FORMAT:
-------------

Predictions are saved to:
  output/predictions.csv

Format:
  - user_name: IMDB username
  - movie_link: IMDB movie link
  - predicted_score: Score (0.0-1.0)
  - test_case: Test scenario identifier

Metrics are saved to:
  output/metrics.json

Includes:
  - Recall@5, Recall@3, Recall@1
  - Execution time
  - Per-scenario metrics

================================================================================
TEST DATA FORMAT:
----------------

Expected test data folder structure:
  <test_data_path>/
    ├── known_reviewers_known_movies.csv
    ├── known_reviewers_unknown_movies.csv
    ├── unknown_reviewers_known_movies.csv
    └── movie_mapper.csv

Pass the folder path as --test_data_path argument.

================================================================================
PERFORMANCE:
-----------

Training Performance (on validation set):
  - Recall@5: 0.7532
  - Recall@3: 0.7699
  - Recall@1: 0.8030
  - RMSE: 0.9094
  - Overfitting Gap: 0.0315 (GOOD - no overfitting)

Expected Inference Time: < 5 seconds (CPU)

================================================================================
TROUBLESHOOTING:
---------------

If scikit-surprise fails to install:
  - This should NOT happen on Linux
  - If it does, try: pip install --upgrade pip setuptools wheel
  - Then: pip install scikit-surprise

If model file not found:
  - Verify Resources/hybrid_model.pkl exists
  - Check file size (~19 MB)

If inference fails:
  - Check test data path is correct
  - Verify all CSV files are present
  - Check output/ directory is writable

================================================================================
CONTACT:
-------

For issues or questions, refer to:
  - README.md (detailed documentation)
  - INSTALLATION.md (platform-specific notes)
  - setup_check.py (dependency verification)

================================================================================

