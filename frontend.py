import streamlit as st
import requests

st.title("Doctor Attendance Predictor")

survey_time = st.text_input("Enter Survey Time (HH:MM)", "10:00")

if st.button("Predict Attendance"):
    if survey_time:
        # response = requests.post("https://your-flask-api-url.com/predict", json={"survey_time": survey_time})
        response = requests.post("http://127.0.0.1:5002/predict", json={"survey_time": survey_time})

        if response.status_code == 200:
            st.success("Prediction successful! Click below to download the CSV.")
            st.download_button("Download CSV", response.content, "predicted_doctors.csv")
        else:
            st.error("Error: " + response.json().get("error", "Unknown error"))
