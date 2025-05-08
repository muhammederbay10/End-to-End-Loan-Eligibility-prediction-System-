from flask import Flask, render_template, request
import joblib
import numpy as np
from custom_transformers import convert_to_df


app = Flask(__name__)


model = joblib.load(open('model.joblib', 'rb'))
transformer = joblib.load(open('transformer.joblib', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        # Extracting the features from the form
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = request.form['Dependents']
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = request.form['ApplicantIncome']
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = int(request.form['Loan_Amount_Term'])
        credit_history = int(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Pack into numpy array for prediction
        features =np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_amount_term, credit_history, property_area]])
        
        # Transform the features using the pre-trained transformer
        transformed_features = transformer.transform(features)

        # Predict
        prediction = model.predict(transformed_features)
        
        result = "Eligible" if prediction[0] == 1 else "Not Eligible"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)