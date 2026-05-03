import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Must be the FIRST Streamlit command
st.set_page_config(page_title="Term Deposit Prediction", layout="centered", page_icon="🏦")

st.title("🏦 Bank Marketing Term Deposit Prediction")
st.write("Enter the client's information below to predict if they will subscribe to a term deposit.")

# Load model AFTER page config
@st.cache_resource
def load_model():
    return joblib.load('svm_model.pkl')

model = load_model()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Client Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed', 'self-employed', 'unknown', 'entrepreneur', 'student'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    education = st.selectbox("Education", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'])
    default = st.selectbox("Credit in Default?", ['no', 'yes', 'unknown'])
    housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
    loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])

    st.subheader("Last Contact Details")
    contact = st.selectbox("Contact Communication Type", ['telephone', 'cellular'])
    month = st.selectbox("Last Contact Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])
    day_of_week = st.selectbox("Last Contact Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=250)

with col2:
    st.subheader("Campaign Details")
    campaign = st.number_input("Number of Contacts (This Campaign)", min_value=1, value=1)
    pdays = st.number_input("Days Since Last Contact (999 means not previously contacted)", min_value=0, value=999)
    previous = st.number_input("Number of Contacts (Before This Campaign)", min_value=0, value=0)
    poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])

    st.subheader("Social and Economic Context")
    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.994)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857)
    nr_employed = st.number_input("Number of Employees", value=5191.0)

# Predict button
if st.button("Predict Subscription", use_container_width=True):
    # Pack inputs into a DataFrame matching the training data
    input_data = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }])

    prediction = model.predict(input_data)[0]
    score = model.decision_function(input_data)[0]

    # Convert raw SVM score to a percentage using the sigmoid function
    safe_score = np.clip(score, -100, 100)
    probability = 1 / (1 + np.exp(-safe_score))
    subscription_prob = probability * 100

    st.markdown("---")
    if prediction == 1:
        st.success(f"✅ **Prediction: The client is LIKELY to subscribe.**")
        st.info(f"Confidence (Probability): {subscription_prob:.1f}%")
    else:
        st.error(f"❌ **Prediction: The client is UNLIKELY to subscribe.**")
        st.warning(f"Confidence (Probability): {subscription_prob:.1f}%")