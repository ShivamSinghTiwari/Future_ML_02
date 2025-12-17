import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Bank Churn Predictor", layout="centered")

st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn using machine learning.")

st.divider()


credit_score = st.number_input("Credit Score", 300, 900, 650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 5)
balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 2)
has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Is Active Member", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# ---- ENCODING ----
gender = 1 if gender == "Male" else 0
has_card = 1 if has_card == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

geo_france = 1 if geography == "France" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# ---- FEATURE ORDER MUST MATCH TRAINING ----
input_data = np.array([[
    credit_score,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    is_active,
    salary,
    geo_germany,
    geo_spain
]])


input_scaled = scaler.transform(input_data)

st.divider()

# ---- PREDICTION ----
if st.button("Predict Churn"):
    churn_prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{churn_prob:.2%}"
    )

    if churn_prob >= 0.75:
        st.error("⚠️ High Risk of Churn")
    elif churn_prob >= 0.40:
        st.warning("⚠️ Medium Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")
