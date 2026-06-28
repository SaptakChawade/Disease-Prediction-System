"""Tests for Disease Prediction System"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.predict import DiseasePredictionModel, generate_synthetic_data, DISEASE_CONFIGS


class TestDiseasePredictionModel:

    def test_all_diseases_train(self):
        for disease in DISEASE_CONFIGS:
            model = DiseasePredictionModel(disease)
            df = generate_synthetic_data(disease, 300)
            metrics = model.train(df)
            assert metrics["accuracy"] > 0
            assert metrics["roc_auc"] > 0

    def test_prediction_output_keys(self):
        model = DiseasePredictionModel("diabetes")
        df = generate_synthetic_data("diabetes", 300)
        model.train(df)
        sample = {f: np.random.randn() for f in DISEASE_CONFIGS["diabetes"]["features"]}
        result = model.predict(sample)
        assert "prediction" in result
        assert "result" in result
        assert "confidence" in result
        assert result["result"] in ["Positive", "Negative"]

    def test_confidence_range(self):
        model = DiseasePredictionModel("heart_disease")
        df = generate_synthetic_data("heart_disease", 300)
        model.train(df)
        sample = {f: np.random.randn() for f in DISEASE_CONFIGS["heart_disease"]["features"]}
        result = model.predict(sample)
        assert 0 <= result["confidence"] <= 100
        assert 0 <= result["risk_probability"] <= 100

    def test_invalid_disease_raises(self):
        with pytest.raises(ValueError):
            DiseasePredictionModel("cancer")

    def test_untrained_predict_raises(self):
        model = DiseasePredictionModel("diabetes")
        sample = {f: 0.0 for f in DISEASE_CONFIGS["diabetes"]["features"]}
        with pytest.raises(RuntimeError):
            model.predict(sample)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
