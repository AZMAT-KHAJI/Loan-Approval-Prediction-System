import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Default Prediction System")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Amount Term")
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

if st.button("Predict"):

    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Urban": 2, "Rural": 0, "Semiurban": 1}[property_area]

    features = np.array([[gender, married, dependents, education,
                          self_employed, applicant_income,
                          coapplicant_income, loan_amount,
                          loan_term, credit_history, property_area]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")
