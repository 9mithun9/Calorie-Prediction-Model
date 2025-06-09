
import streamlit as st
import joblib
import numpy as np

# Load models
#rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.joblib')

st.title("🔥 Calorie Burn Prediction App")
st.markdown("Enter your data to estimate burned calories.")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100)
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0)
duration = st.number_input("Exercise Duration (min)", min_value=1.0, max_value=300.0)
heart_rate = st.number_input("Average Heart Rate (bpm)", min_value=40.0, max_value=200.0)
body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0)

if st.button("Predict Calories Burned"):
    gender_val = 1 if gender == "Male" else 0
    bmi = weight / ((height / 100) ** 2)

    if gender == "Male":  # Male
        ascm = ((-55.0969 + (0.6309 * heart_rate) + (0.1988 * weight) + (0.2017 * age)) / 4.184) * duration
    else:  # Female
        ascm = ((-20.4022 + (0.4472 * heart_rate) - (0.1263 * weight) + (0.074 * age)) / 4.184) * duration


    ascm = (heart_rate + body_temp) / 2  # Update if you used a different ASCM formula
    input_data = np.array([[age, height, weight, duration, heart_rate, body_temp, bmi, ascm]])

    #rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]
    #final_prediction = 0.6 * rf_pred + 0.4 * xgb_pred

    st.success(f"✅ Estimated Calories Burned: *{xgb_pred:.2f} kcal*")
