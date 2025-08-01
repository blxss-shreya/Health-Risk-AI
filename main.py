from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import send_from_directory
from flask import Blueprint

app = Flask(__name__,
            static_url_path='',
            static_folder='.',
            template_folder='.')

CORS(app)

# Load both models and their features
diabetes_model = joblib.load('diabetes_rf_model.pkl')
diabetes_features = joblib.load('feature_columns.pkl')

model = joblib.load("heart_model.pkl")
label_encoders = joblib.load("heart_label_encoders.pkl")

print("Model and features loaded successfully")


@app.route('/')
def home():
    return app.send_static_file("index.html")


@app.route('/diabetes')
def diabetes():
    return app.send_static_file("diabetes.html")


@app.route("/heart")
def heart():
    return send_from_directory(".", "heart.html")


@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()

        input_vector = [
            data['Pregnancies'], data['Glucose'], data['BloodPressure'],
            data['SkinThickness'], data['Insulin'], data['BMI'],
            data['DiabetesPedigreeFunction'], data['Age'],
            data['BMI_Age_Ratio'], data['Glucose_BMI'], data['Insulin_Sqrt']
        ]

        prediction = diabetes_model.predict([input_vector])[0]
        probability = diabetes_model.predict_proba([input_vector])[0][1]

        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json()

        # Input features in the required order
        input_features = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
            'ST_Slope'
        ]

        input_data = []

        for feature in input_features:
            value = data.get(feature)

            # Apply label encoding if necessary
            if feature in label_encoders:
                encoder = label_encoders[feature]
                value = encoder.transform([value])[0]

            input_data.append(float(value))

        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][
            1]  # Probability of class 1 (heart disease)

        return jsonify({
            "prediction":
            int(prediction),
            "probability":
            round(probability * 100, 2),
            "message":
            "High risk of heart disease"
            if prediction == 1 else "Low risk of heart disease"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
