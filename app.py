import streamlit as st
import numpy as np
import joblib

# ✅ Must be the FIRST Streamlit command
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below:")

# Load model AFTER page config
model = joblib.load('svm_model.pkl')

# Feature names (same order as training)
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    inputs.append(val)

# Predict button
if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)

    prediction = model.predict(data)[0]
    score = model.decision_function(data)[0]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction (Score: {score:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Score: {score:.2f})")