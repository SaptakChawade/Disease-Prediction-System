"""
Disease Prediction System
Predicts likelihood of Diabetes, Heart Disease, and Parkinson's Disease
using ensemble ML models trained on standard medical datasets.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline


# ─── Disease Configurations ──────────────────────────────────────────────────

DISEASE_CONFIGS = {
    "diabetes": {
        "features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "target": "Outcome",
        "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                             max_depth=4, random_state=42),
        "description": "Predicts Type-2 Diabetes using the Pima Indians dataset",
    },
    "heart_disease": {
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "target": "target",
        "model": RandomForestClassifier(n_estimators=200, max_depth=6,
                                        min_samples_split=5, random_state=42),
        "description": "Predicts coronary heart disease using the Cleveland Heart dataset",
    },
    "parkinsons": {
        "features": [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
            "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
            "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
            "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
            "spread1", "spread2", "D2", "PPE"
        ],
        "target": "status",
        "model": SVC(kernel="rbf", C=10, gamma=0.001, probability=True, random_state=42),
        "description": "Predicts Parkinson's Disease using biomedical voice measurements",
    },
}


# ─── Model Training ──────────────────────────────────────────────────────────

class DiseasePredictionModel:
    """Trains, evaluates, and persists disease prediction models."""

    def __init__(self, disease: str):
        if disease not in DISEASE_CONFIGS:
            raise ValueError(f"Unknown disease. Choose from: {list(DISEASE_CONFIGS.keys())}")
        self.disease = disease
        self.config = DISEASE_CONFIGS[disease]
        self.pipeline = None
        self.metrics = {}

    def train(self, df: pd.DataFrame) -> dict:
        """Train the model on provided DataFrame and return evaluation metrics."""
        features = self.config["features"]
        target = self.config["target"]

        X = df[features].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", self.config["model"]),
        ])

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="accuracy")

        self.metrics = {
            "disease": self.disease,
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2),
            "cv_mean": round(cv_scores.mean() * 100, 2),
            "cv_std": round(cv_scores.std() * 100, 2),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        return self.metrics

    def predict(self, input_data: dict) -> dict:
        """Predict disease probability for a single patient's data."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        features = self.config["features"]
        values = np.array([[input_data[f] for f in features]])

        prediction = self.pipeline.predict(values)[0]
        probability = self.pipeline.predict_proba(values)[0]

        return {
            "disease": self.disease,
            "prediction": int(prediction),
            "result": "Positive" if prediction == 1 else "Negative",
            "confidence": round(max(probability) * 100, 2),
            "risk_probability": round(probability[1] * 100, 2),
        }

    def save(self, path: str = "models"):
        """Save the trained pipeline to disk."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{self.disease}_model.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"✅ Model saved: {filepath}")

    def load(self, path: str = "models"):
        """Load a saved pipeline from disk."""
        filepath = os.path.join(path, f"{self.disease}_model.pkl")
        with open(filepath, "rb") as f:
            self.pipeline = pickle.load(f)
        print(f"✅ Model loaded: {filepath}")


# ─── Synthetic Data Generator (for demo / testing) ───────────────────────────

def generate_synthetic_data(disease: str, n_samples: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic data for demo purposes."""
    np.random.seed(42)
    config = DISEASE_CONFIGS[disease]
    features = config["features"]

    data = {}
    for feature in features:
        data[feature] = np.random.randn(n_samples)

    # Add binary target with ~35% positive rate
    data[config["target"]] = (np.random.rand(n_samples) > 0.65).astype(int)
    return pd.DataFrame(data)


# ─── CLI Demo ────────────────────────────────────────────────────────────────

def run_demo():
    """Run a quick demo training all three disease models."""
    print("\n" + "="*60)
    print("🏥  Disease Prediction System — Demo")
    print("="*60)

    for disease in DISEASE_CONFIGS:
        print(f"\n{'─'*50}")
        print(f"📊 Training model for: {disease.upper()}")
        print(f"   {DISEASE_CONFIGS[disease]['description']}")

        model = DiseasePredictionModel(disease)
        df = generate_synthetic_data(disease, n_samples=600)
        metrics = model.train(df)

        print(f"   ✅ Accuracy  : {metrics['accuracy']}%")
        print(f"   ✅ ROC-AUC   : {metrics['roc_auc']}%")
        print(f"   ✅ CV Score  : {metrics['cv_mean']}% ± {metrics['cv_std']}%")
        print(f"   📁 Train/Test: {metrics['train_samples']} / {metrics['test_samples']}")

        # Demo prediction
        sample = {f: np.random.randn() for f in DISEASE_CONFIGS[disease]["features"]}
        result = model.predict(sample)
        print(f"\n   🔬 Sample Prediction:")
        print(f"      Result     : {result['result']}")
        print(f"      Confidence : {result['confidence']}%")
        print(f"      Risk Score : {result['risk_probability']}%")

        model.save()

    print("\n" + "="*60)
    print("✅ All models trained and saved to models/")


if __name__ == "__main__":
    run_demo()
