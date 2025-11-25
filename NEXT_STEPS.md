# 🎯 Next Steps - AITH 2025 Submission Guide

## ✅ What's Done
- ✅ Training notebook uploaded (`notebooks/final_notebook.ipynb`)
- ✅ Model file uploaded (`Resources/hybrid_model.pkl`)
- ✅ Inference code updated to work with SVD model
- ✅ Requirements updated (`scikit-surprise` instead of `lightfm`)

---

## 📋 Step-by-Step Action Plan

### **STEP 1: Test Inference Locally** ⚡

**Goal:** Verify everything works before submission

```bash
# 1. Activate your virtual environment
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

# 2. Install dependencies (if not already done)
pip install -r requirements.txt

# 3. Run inference on sample test data
python inference.py --test_data_path data/aith-dataset/sample_test_phase_1

# Expected output:
# - output/predictions.csv (all predictions)
# - output/metrics.json (Recall@K metrics)
```

**What to check:**
- ✅ No errors during execution
- ✅ Predictions file is created
- ✅ Metrics show Recall@5, Recall@3, Recall@1
- ✅ Execution time < 5 seconds

---

### **STEP 2: Verify Output Format** 📊

Check that `output/predictions.csv` has the correct format:

**Expected columns:**
- `user_name` - IMDB username
- `movie_link` - IMDB movie link
- `predicted_score` - Score (0.0-1.0 range)
- `test_case` - Which test file it came from

**Check metrics.json:**
```json
{
  "known_reviewers_known_movies_recall@5": 0.75,
  "known_reviewers_known_movies_recall@3": 0.77,
  "known_reviewers_known_movies_recall@1": 0.80,
  "overall_recall@5": 0.75,
  "execution_time_seconds": 3.2
}
```

---

### **STEP 3: Final Checklist Before Submission** ✅

#### **Code Requirements:**
- [ ] `inference.py` runs without errors
- [ ] `requirements.txt` has all dependencies
- [ ] Model file is in `Resources/hybrid_model.pkl`
- [ ] Code works on CPU (no GPU dependencies)
- [ ] All outputs go to `output/` folder

#### **File Structure:**
```
MarriageChimeHackathon/
├── inference.py              ✅ Main entry point
├── requirements.txt          ✅ Dependencies
├── README.md                 ✅ Documentation
├── Inference/
│   └── infer.py             ✅ Core logic
├── Resources/
│   └── hybrid_model.pkl     ✅ Trained model
└── notebooks/
    └── final_notebook.ipynb ✅ Training notebook
```

#### **Documentation:**
- [ ] README.md explains how to run inference
- [ ] Clear instructions for test data path
- [ ] Model download instructions (if needed)

---

### **STEP 4: Prepare for Submission** 🚀

#### **4.1. Test on Fresh Environment** (Recommended)

Test that your code works on a clean setup:

```bash
# Create new virtual environment
python -m venv test_env
test_env\Scripts\activate  # Windows
pip install -r requirements.txt

# Run inference
python inference.py --test_data_path data/aith-dataset/sample_test_phase_1
```

#### **4.2. Verify Competition Requirements**

According to competition rules, judges will:
1. ✅ Clone repository
2. ✅ Create virtual environment
3. ✅ Run `pip install -r requirements.txt`
4. ✅ Execute `inference.py` on hidden test data

**Make sure:**
- [ ] No hardcoded paths (use relative paths)
- [ ] Model loads automatically from `Resources/`
- [ ] Test data path can be passed as argument
- [ ] All outputs save to `output/` folder

#### **4.3. Update README.md** (if needed)

Ensure README clearly states:
- How to provide test data path
- Where outputs are saved
- Model file location

---

### **STEP 5: Final Testing** 🧪

Run these commands to simulate judge's evaluation:

```bash
# 1. Fresh clone (simulate judge)
cd ..
git clone <your-repo-url> test_submission
cd test_submission

# 2. Create venv
python -m venv venv
venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run inference
python inference.py --test_data_path data/aith-dataset/sample_test_phase_1

# 5. Check outputs
ls output/
# Should see: predictions.csv, metrics.json
```

---

### **STEP 6: Submit! 🎉**

Once everything works:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Final submission - SVD model with anti-overfitting"
   git push origin main
   ```

2. **Share repository link** in competition submission channel

3. **Include in submission message:**
   - Team name: OneManArmy
   - Model performance: Recall@5 = 0.7532
   - Key innovation: IMDB user profiles + SVD hybrid model

---

## 🐛 Troubleshooting

### **Issue: ModuleNotFoundError: No module named 'surprise'**
**Solution:**
```bash
pip install scikit-surprise
```

### **Issue: Model file not found**
**Solution:**
- Check `Resources/hybrid_model.pkl` exists
- Verify file size (~19 MB)

### **Issue: Slow inference (>5 seconds)**
**Solution:**
- Check if using CPU (not GPU)
- Verify model is loaded once (not per prediction)

### **Issue: Low Recall@K scores**
**Solution:**
- Check model file is correct version
- Verify test data format matches training data

---

## 📈 Expected Performance

Based on training results:
- **Recall@5:** ~0.75 (excellent!)
- **Recall@3:** ~0.77
- **Recall@1:** ~0.80
- **Inference Time:** < 5 seconds
- **Overfitting Gap:** 0.0315 (GOOD - no overfitting)

---

## 🎯 Success Criteria

Your submission is ready when:
- ✅ Inference runs without errors
- ✅ All outputs generated correctly
- ✅ Recall@K metrics calculated
- ✅ Code works on fresh environment
- ✅ Documentation is clear

---

## 📞 Need Help?

If you encounter issues:
1. Check error messages carefully
2. Verify all files are in correct locations
3. Test with sample data first
4. Check that model file is not corrupted

**Good luck with your submission! 🏆**

