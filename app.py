from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from custom_transformers import convert_to_df
import sklearn
import sys

# Monkey-patch convert_to_df into __main__ so unpickling can find it
sys.modules['__main__'].convert_to_df = convert_to_df

print(f"Loaded scikit-learn: {sklearn.__version__}")

app = Flask(__name__, static_folder='static')

# Load the trained pipeline
with open('model/model.joblib', 'rb') as f:
    model = joblib.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.errorhandler(400)
def bad_request(e):
    print("=== 400 Bad Request ===")
    print("Headers:", request.headers)
    print("Form Data:", request.form)
    print("Raw Body:", request.get_data())
    return "Bad Request – check server logs", 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = request.form['Dependents']
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Prepare feature dict and DataFrame
        features = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        }
        columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        df_features = convert_to_df(features, columns)

        # Debug info
        print("df_features columns:", df_features.columns.tolist())
        print("Number of features:", df_features.shape[1])

        # Expected transformer input columns
        selected_columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'Credit_History', 'Property_Area', 'ApplicantIncome',
            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'
        ]
        missing_columns = [col for col in selected_columns if col not in df_features.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        df_features = df_features[selected_columns]
        if df_features.shape[1] != len(selected_columns):
            raise ValueError(f"Expected {len(selected_columns)} features, got {df_features.shape[1]}")

        # Make prediction
        prediction = model.predict(df_features)
        result = "Loan Approved✅" if prediction[0] == 1 else "Loan Not Approved❌"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
