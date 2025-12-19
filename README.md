# 🏦 Bank Customer Churn Prediction

This project predicts whether a bank customer is likely to churn using machine learning models.

## 🔍 Overview
- Built churn prediction models using Logistic Regression, Random Forest, and XGBoost
- Evaluated models using ROC-AUC, precision, recall, and confusion matrices
- Deployed an interactive Streamlit app for real-time churn prediction

## 📊 Dataset
- Source: Kaggle – Bank Customer Churn Dataset
- Target Variable: `Exited` (1 = churned, 0 = retained)

## 🧠 Models Used
- Logistic Regression
- Random Forest
- XGBoost (final model)

## 🚀 Deployment
The model is deployed using Streamlit Community Cloud.

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
