import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Load the trained model, scaler, and selected features
model = load('heart_disease_model.joblib')
scaler = load('scaler.joblib')
selected_features = load('selected_features.joblib')

st.title('Heart Disease Diagnosis')
st.write('Project By 21d/47xcs/00245 James Victor Essien 20/47cs/01312 Oyewole Emmanuel Oluwatimilehin')

# Input fields for user data
age = st.number_input('Age', min_value=0, max_value=120, value=40)
sex = st.selectbox('Sex', [0, 1])
chest_pain = st.selectbox('Chest Pain Type', [1, 2, 3, 4])
resting_bp = st.number_input('Resting BP (Systolic)', min_value=80, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol', min_value=100, max_value=400, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
resting_ecg = st.selectbox('Resting ECG', [0, 1, 2])
max_hr = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox('ST Slope', [1, 2, 3])

# Create input data frame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'chest pain type': [chest_pain],
    'resting bp s': [resting_bp],
    'cholesterol': [cholesterol],
    'fasting blood sugar': [fasting_bs],
    'resting ecg': [resting_ecg],
    'max heart rate': [max_hr],
    'exercise angina': [exercise_angina],
    'oldpeak': [oldpeak],
    'ST slope': [st_slope]
})

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Select only the best features
input_data_selected = input_data_scaled[:, selected_features]

# Make a prediction
prediction = model.predict(input_data_selected)[0]

# Display the prediction
if st.button('Predict'):
    if prediction == 1:
        st.write("The model predicts that the patient **has heart disease**.")
    else:
        st.write("The model predicts that the patient **does not have heart disease**.")
