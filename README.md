# 🏦 Loan Eligibility Prediction System

An end-to-end machine learning pipeline that predicts loan eligibility based on applicant information. It encompasses data ingestion, preprocessing, EDA, feature engineering, model training, evaluation, and deployment as a Flask web application.

---

## 📋 Table of Contents

1. [🚀 Features](#-features)
2. [📁 Project Structure](#-project-structure)
3. [⚙️ Installation](#️-installation)
4. [🚀 Usage](#-usage)
5. [🔍 Workflow Overview](#-workflow-overview)
6. [🗂️ Dataset](#️-dataset)
7. [📊 Model Performance](#-model-performance)
8. [🔮 Future Improvements](#-future-improvements)
9. [🤝 Contributing](#-contributing)
10. [📄 License](#-license)
11. [✉️ Contact](#️-contact)

---

## 🚀 Features

* **End-to-End Pipeline**: From raw data ingestion to deployment.
* **Robust Preprocessing**: Handles missing values, encodes categoricals, scales numerics.
* **Feature Engineering**: Derives new features such as total income and EMI.
* **Model Zoo**: Supports Decision Tree, Random Forest, Logistic Regression, SVC, and XGBoost.
* **Hyperparameter Tuning**: Uses `GridSearchCV` with cross-validation.
* **Comprehensive EDA**: Visualizations for distributions and correlations.
* **Flask Web App**: User-friendly interface for real-time predictions.

## 📁 Project Structure

```
├── app.py                    # Flask application for deployment
├── custom_transformers.py    # Custom DataFrame converter
├── eligibility_prediction.ipynb  # Jupyter Notebook with full ML workflow
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── Data/
│   └── Bank_loan_data.csv    # Dataset for training and testing
├── model/
│   └── model.joblib          # Saved pipeline (preprocessing + model)
├── static/
│   └── style.css             # CSS for the web interface
└── templates/
    └── index.html            # HTML template for user input
```

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/muhammederbay10/End-to-End-Loan-Eligibility-prediction-System-.git
   cd End-to-End-Loan-Eligibility-prediction-System-
   ```
2. **Create & activate a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. **Training & Evaluation (Notebook):**

   * Open `eligibility_prediction.ipynb` in JupyterLab or Notebook.
   * Run all cells to preprocess data, train models, and visualize results.

2. **Deploy Flask App:**

   ```bash
   python app.py
   ```

   * Navigate to `http://127.0.0.1:5000` in your browser.
   * Fill out the form with applicant details and click **Predict**.

---

## 🔍 Workflow Overview

1. **Data Loading**: Read `Data/Bank_loan_data.csv` using pandas.
2. **Preprocessing**:

   * Drop `Loan_ID`.
   * Impute missing values (mean for numeric, mode for categorical).
   * Encode categoricals with One-Hot Encoding via `ColumnTransformer`.
   * Scale numeric features when needed.
3. **Exploratory Data Analysis (EDA)**

   * Visualize distributions (histograms, count plots).
   * Correlation heatmaps and pair plots.
4. **Feature Engineering**:

   * `TotalIncome` = `ApplicantIncome` + `CoapplicantIncome`.
   * `EMI` = `LoanAmount / Loan_Amount_Term`.
5. **Modeling**:

   * Build pipelines combining preprocessing and classifier.
   * Evaluate on train/test split and cross-validation.
6. **Hyperparameter Tuning**:

   * Use `GridSearchCV` (cv=5) for each model.
7. **Evaluation**:

   * Report Accuracy, Precision, Recall, F1-Score.
8. **Deployment**:

   * Save best pipeline with `joblib`.
   * Serve via Flask for web-based predictions.

---

## 🗂️ Dataset

* **Source:** Contains 614 training samples from public loan datasets.
* **Features:**

  * **Categorical:** Gender, Married, Dependents, Education, Self\_Employed, Property\_Area
  * **Numerical:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan\_Amount\_Term, Credit\_History
  * **Target:** Loan\_Status (`Y` = eligible, `N` = not eligible)

---

## 📊 Model Performance

| Model                  | Recall | F1-Score |
| ---------------------- | -------- | -------- |
| Support Vector Machine | 1.0      | 0.86     |
| Random Forest          | 0.9125   | 0.829    |
| Logistic Regression    | 0.9875   | 0.85     |
| Decision Tree          | 0.975    | 0.847    |
| XGBoost                | 0.93     | 0.83     |

---

## 🔮 Future Improvements

* Enhance feature engineering (interaction terms, polynomial features).
* Integrate advanced resampling techniques (SMOTE, ADASYN) for class imbalance.
* Add LightGBM & CatBoost for comparative benchmarks.
* Dockerize the application for portability.
* Deploy to cloud platforms (Heroku, AWS Elastic Beanstalk).
* Expose a REST API for batch predictions.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to your branch: `git push origin feature/YourFeature`.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the [MIT License](https://github.com/muhammederbay10/End-to-End-Loan-Eligibility-prediction-System-/blob/main/LICENSE).

---

## ✉️ Contact

Developed by **Muhammed Erbay** – [GitHub Profile](https://github.com/muhammederbay10)

Feel free to reach out with any questions or feedback!
