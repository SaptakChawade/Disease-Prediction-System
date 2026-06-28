"""
Disease Prediction System — Streamlit Web App
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from src.predict import DiseasePredictionModel, generate_synthetic_data

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Disease Prediction System")
st.markdown("*ML-powered prediction for Diabetes, Heart Disease & Parkinson's*")
st.markdown("---")

disease = st.sidebar.selectbox(
    "Select Disease to Predict",
    ["Diabetes", "Heart Disease", "Parkinson's Disease"],
)

# ─── Diabetes ────────────────────────────────────────────────────────────────
if disease == "Diabetes":
    st.header("🩸 Diabetes Prediction")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("🔍 Predict Diabetes", type="primary"):
        model = DiseasePredictionModel("diabetes")
        df = generate_synthetic_data("diabetes", 400)
        model.train(df)
        result = model.predict({
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
            "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        })
        if result["prediction"] == 1:
            st.error(f"⚠️ Result: **{result['result']}** — Risk: {result['risk_probability']}%")
        else:
            st.success(f"✅ Result: **{result['result']}** — Low Risk: {result['risk_probability']}%")
        st.info(f"Model Confidence: {result['confidence']}%")

# ─── Heart Disease ────────────────────────────────────────────────────────────
elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 250, 130)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120?", [0, 1])
        restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    with col2:
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)
        exang = st.selectbox("Exercise-Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST (0–2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored (0–4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    if st.button("🔍 Predict Heart Disease", type="primary"):
        model = DiseasePredictionModel("heart_disease")
        df = generate_synthetic_data("heart_disease", 400)
        model.train(df)
        result = model.predict({
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        })
        if result["prediction"] == 1:
            st.error(f"⚠️ Result: **{result['result']}** — Risk: {result['risk_probability']}%")
        else:
            st.success(f"✅ Result: **{result['result']}** — Low Risk: {result['risk_probability']}%")
        st.info(f"Model Confidence: {result['confidence']}%")

# ─── Parkinson's ─────────────────────────────────────────────────────────────
else:
    st.header("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter vocal biomedical measurements:")
    col1, col2, col3 = st.columns(3)
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE"
    ]
    defaults = [119.99, 157.30, 74.99, 0.00784, 0.00007, 0.00370, 0.00554,
                0.01109, 0.04374, 0.42600, 0.02182, 0.03130, 0.02971, 0.06545,
                0.02211, 21.03, 0.41400, 0.81500, -4.81300, 0.26600, 2.30100, 0.28400]
    inputs = {}
    for i, (feat, default) in enumerate(zip(features, defaults)):
        col = [col1, col2, col3][i % 3]
        with col:
            inputs[feat] = st.number_input(feat, value=float(default), format="%.5f")

    if st.button("🔍 Predict Parkinson's", type="primary"):
        model = DiseasePredictionModel("parkinsons")
        df = generate_synthetic_data("parkinsons", 400)
        model.train(df)
        result = model.predict(inputs)
        if result["prediction"] == 1:
            st.error(f"⚠️ Result: **{result['result']}** — Risk: {result['risk_probability']}%")
        else:
            st.success(f"✅ Result: **{result['result']}** — Low Risk: {result['risk_probability']}%")
        st.info(f"Model Confidence: {result['confidence']}%")

st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only. Always consult a medical professional.")
