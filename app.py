from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from custom_transformers import convert_to_df

app = Flask(__name__, static_folder='static')

model = joblib.load('model/model.joblib')
# transformer = joblib.load('model/transformer.joblib')

@app.route('/')
def home():
    return render_template('index.html')

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

        # Create DataFrame
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

        # # Add derived features (replicating notebook's add_features)
        # df_features['TotalIncome'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
        # df_features['EMI'] = df_features['LoanAmount'] / df_features['Loan_Amount_Term'].replace(0, 1)
        # df_features['Balance Income'] = df_features['TotalIncome'] - df_features['EMI']

        # Debugging
        print("df_features columns:", df_features.columns.tolist())
        print("Number of features:", df_features.shape[1])

        # Expected columns for transformer (match notebook input)
        selected_columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'Credit_History', 'Property_Area', 'ApplicantIncome',
            'CoapplicantIncome','LoanAmount','Loan_Amount_Term'
        ]

        # Check for missing columns
        missing_columns = [col for col in selected_columns if col not in df_features.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        # Select transformer columns
        df_features = df_features[selected_columns]

        # Verify feature count
        if df_features.shape[1] != 11:
            raise ValueError(f"Expected 12 features, got {df_features.shape[1]}: {df_features.columns.tolist()}")

        # Transform and predict
        # transformed_features = transformer.transform(df_features)
        prediction = model.predict(df_features)
        result = "Eligible" if prediction[0] == 1 else "Not Eligible"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)