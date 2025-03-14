from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Load trained model and pre-processors ONCE
model = joblib.load("doctor_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
df = pd.read_excel("dummy_npi_data.xlsx")  # Load dataset once

# Ensure datetime conversion
df['Login Time'] = pd.to_datetime(df['Login Time'], errors='coerce')
df['Logout Time'] = pd.to_datetime(df['Logout Time'], errors='coerce')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        survey_time = data.get("survey_time")

        if not survey_time:
            return jsonify({"error": "Missing survey_time parameter"}), 400

        try:
            survey_hour = datetime.strptime(survey_time, "%H:%M").hour
        except ValueError:
            return jsonify({"error": "Invalid time format. Use HH:MM."}), 400

        # Prepare input data for the model
        input_data = df.copy()
        input_data['Login Hour'] = survey_hour
        input_data['Active Hours'] = (input_data['Logout Time'] - input_data['Login Time']).dt.total_seconds() / 3600

        # Encode categorical data safely
        for col in ['Specialty', 'Region']:
            input_data[col] = input_data[col].astype(str).map(lambda x: label_encoders[col].classes_.tolist().index(x) if x in label_encoders[col].classes_ else -1)

        # Normalize numerical features
        input_data[['Login Hour', 'Active Hours']] = scaler.transform(input_data[['Login Hour', 'Active Hours']])

        # Predict likelihood of attendance
        input_data['Prediction'] = model.predict(input_data[['Login Hour', 'Active Hours', 'Speciality', 'Region']])

        # Filter doctors who are likely to attend
        result_df = input_data[input_data['Prediction'] == 1][['NPI', 'Speciality', 'Region', 'Login Hour']]

        # Save results to CSV
        output_file = "predicted_doctors.csv"
        result_df.to_csv(output_file, index=False)

        return send_file(output_file, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
