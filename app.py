
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the saved model
model = load_model('loan_approval_model')

# Streamlit UI
st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Submit button
if st.button("Predict"):
    input_df = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    prediction = predict_model(model, data=input_df)
    result = prediction['Label'][0]
    status = "Approved ✅" if result == 1 else "Rejected ❌"

    st.success(f"Loan Prediction Result: {status}")
