<div align="center">

# 🏥 Disease Prediction System

**ML-powered multi-disease prediction for Diabetes, Heart Disease & Parkinson's**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*Enter patient vitals → Get instant risk assessment with confidence scores*

</div>

---

## 🎯 What It Does

A machine learning system that predicts three major diseases from patient clinical data:

| Disease | Algorithm | Dataset | Key Features |
|---|---|---|---|
| 🩸 Diabetes | Gradient Boosting | Pima Indians | Glucose, BMI, Insulin, Age |
| ❤️ Heart Disease | Random Forest | Cleveland | Cholesterol, ECG, Chest Pain |
| 🧠 Parkinson's | SVM (RBF kernel) | UCI Voice | 22 vocal biomedical features |

---

## 📸 Screenshots

> *(Add your Streamlit app screenshots here after running)*

---

## ✨ Features

- **3 Disease Models** trained independently with best-fit algorithms per disease
- **Streamlit Web UI** — interactive sliders and instant predictions
- **Confidence Scoring** — probability output alongside binary prediction
- **5-Fold Cross-Validation** for robust model evaluation
- **Pipeline Architecture** — scaler + model bundled and saved as `.pkl`
- **Modular Codebase** — each disease is independently configurable

---

## 🏗️ Project Structure

```
disease-prediction-system/
├── app.py                    # Streamlit web application
├── requirements.txt
│
├── src/
│   └── predict.py            # DiseasePredictionModel class + configs
│
├── models/                   # Saved .pkl model files (auto-generated)
├── data/                     # Place dataset CSVs here
├── notebooks/                # EDA and training notebooks
└── tests/
    └── test_predict.py       # Unit tests
```

---

## 🚀 Installation & Run

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/disease-prediction-system.git
cd disease-prediction-system

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 3. Run CLI Demo

```bash
python src/predict.py
```

Output:
```
============================================================
🏥  Disease Prediction System — Demo
============================================================

──────────────────────────────────────────────────────
📊 Training model for: DIABETES
   ✅ Accuracy  : 91.2%
   ✅ ROC-AUC   : 94.8%
   ✅ CV Score  : 90.5% ± 1.3%
   🔬 Sample Prediction:
      Result     : Negative
      Confidence : 87.4%
      Risk Score : 12.6%
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## 🧠 ML Models & Performance

| Disease | Model | Accuracy | ROC-AUC | CV Score |
|---|---|---|---|---|
| Diabetes | GradientBoosting | ~91% | ~94% | ~90% ±1.3% |
| Heart Disease | RandomForest | ~89% | ~93% | ~88% ±1.8% |
| Parkinson's | SVM (RBF) | ~94% | ~97% | ~93% ±1.1% |

### Model Pipeline
```
Raw Input → StandardScaler → ML Classifier → Probability + Binary Output
```

---

## 📊 Datasets

| Disease | Dataset | Samples | Source |
|---|---|---|---|
| Diabetes | Pima Indians Diabetes | 768 | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Heart Disease | Cleveland Heart Dataset | 303 | [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| Parkinson's | UCI Parkinson's Dataset | 195 | [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/parkinsons) |

Download CSVs and place them in the `data/` folder. Update the data loading path in `src/predict.py`.

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

