import streamlit as st
import joblib
import numpy as np

# Load your model and scaler
loaded_model = joblib.load("knnSmote_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

st.title("Car Purchased Prediction")

# Input widgets
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=49)
estimated_salary = st.number_input("Estimated Salary", min_value=1000, max_value=200000, value=36000)

if st.button("Predict"):
    # Encode gender (Male=1, Female=0)
    gender_encoded = 1 if gender == "Male" else 0

    # Scale numerical inputs
    numerical_input = [[age, estimated_salary]]
    scaled_input = scaler.transform(numerical_input)

    # Prepare input sample
    sample_input = [[gender_encoded, scaled_input[0][0], scaled_input[0][1]]]

    # Prediction
    prediction = loaded_model.predict(sample_input)

    # Display result
    if prediction[0] == 1:
        st.success("Prediction: Will Buy")
    else:
        st.warning("Prediction: Won’t Buy")
