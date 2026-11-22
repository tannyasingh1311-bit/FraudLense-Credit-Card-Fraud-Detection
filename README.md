# 🛡️ Professional Credit Card Fraud Detection System

An enterprise-grade financial technology application for real-time credit card fraud risk scoring and batch transaction auditing, engineered with a warm ivory editorial design aesthetic and cost-sensitive machine learning.

---

## 📌 Project Architecture

```
Credit_Card_Fraud_Detection/
│
├── dataset/
│   ├── creditcard.csv                 # Primary transaction dataset (284,807 transactions)
│   └── sample_test_transactions.csv   # Curated test batch (50 transactions) for instant audit
│
├── models/
│   ├── random_forest_model.joblib     # Production Random Forest Classifier (balanced_subsample)
│   ├── logistic_regression_model.joblib # Production Logistic Regression (liblinear, balanced)
│   ├── fraud_model.joblib             # Primary model pointer
│   ├── standard_scaler.joblib         # Fitted StandardScaler (applied strictly to 'Amount')
│   ├── evaluation_results.json        # Test set evaluation metrics & confusion matrices
│   └── metrics.json                   # Pipeline diagnostic records
│
├── pdf_generator.py                   # Institutional PDF audit report engine (fpdf2)
├── app.py                             # Streamlit enterprise dashboard (5 sections, 8 features)
├── train_model.py                     # Machine learning pipeline training & evaluation script
├── Credit_Card_Fraud_Detection_Training.ipynb # Standalone sequential Google Colab notebook
├── requirements.txt                   # Pinned project dependencies
└── README.md                          # Documentation & operational guide
```

---

## ⚙️ Strict Machine Learning Pipeline

The project preserves the original, rigorously validated methodology:

1. **Stratified Splitting**: 80:20 train-test split with `random_state=42` and `stratify=y` to preserve the 577:1 legitimate-to-fraud ratio.
2. **Data Leakage Prevention**: `StandardScaler` is fitted **strictly on `X_train[['Amount']]`**; new inputs and test data are transformed using only the learned parameters ($\mu = \$88.18, \sigma = \$250.72$).
3. **Model 1: Logistic Regression**:
   ```python
   LogisticRegression(
       class_weight='balanced',
       max_iter=1000,
       random_state=42,
       solver='liblinear'
   )
   ```
4. **Model 2: Random Forest Classifier**:
   ```python
   RandomForestClassifier(
       n_estimators=100,
       max_depth=12,
       min_samples_split=5,
       min_samples_leaf=2,
       class_weight='balanced_subsample',
       random_state=42,
       n_jobs=-1
   )
   ```

---

## 🚀 Key Features Implemented

1. **Feature 1: Transaction Risk Score**: Real `predict_proba()` output displayed as a percentage with an editorial risk progress gauge.
2. **Feature 2: Adjustable Detection Threshold**: Dynamic threshold slider ($\tau \in [0.05, 0.95]$, default $0.50$) without model retraining.
3. **Feature 3: 3-Tier Risk Level Classification**: Independent categorization:
   - **Low Risk**: Probability $< 30\%$
   - **Medium Risk**: Probability $30\%$ to $< 70\%$
   - **High Risk**: Probability $\ge 70\%$
4. **Feature 4: Fraud Alert System**: Dedicated alert box with Transaction ID, Amount, Risk Level, and action: *"Review this transaction before approving it."*
5. **Feature 5: PDF Report Generation**: Formal PDF reports generated using `fpdf2` for both individual transactions and bulk batch runs.
6. **Feature 6: Transaction ID Tracking**: Unique identifier (`TXN-YYYYMMDD-XXXXXX`) preserved across scoring, session history, and PDF documentation.
7. **Feature 7: Zero-Database Detection History**: Memory-only session state logging with search, prediction filters, risk level filters, timestamp sorting, CSV download, and confirmed history clearing.
8. **Feature 8: Explainable Predictions**: Feature impact drivers supported by model mechanics with clear operational limitation notices.

---

## 💻 Quick Start Guide

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Launch the Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Re-Train Models Locally
```powershell
python train_model.py
```

### 4. Run on Google Colab
Upload `Credit_Card_Fraud_Detection_Training.ipynb` and `creditcard.csv` to Google Colab and run all cells sequentially.
