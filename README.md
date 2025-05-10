# 🏦 Loan Eligibility Prediction System

Live Demo: [https://web-production-53c65.up.railway.app/](https://web-production-53c65.up.railway.app/)

An end-to-end machine learning pipeline that predicts loan eligibility based on applicant data. From data ingestion and preprocessing to model training, evaluation, and deployment as an interactive Flask web application.

---

## 📋 Table of Contents

1. [✨ Live Demo](#-live-demo)
2. [🚀 Features](#-features)
3. [📁 Project Structure](#-project-structure)
4. [⚙️ Installation](#️-installation)
5. [🚀 Usage](#-usage)
6. [🔍 Workflow Overview](#-workflow-overview)
7. [🗂️ Dataset](#️-dataset)
8. [📊 Model Performance](#-model-performance)
9. [🔮 Future Improvements](#-future-improvements)
10. [🤝 Contributing](#-contributing)
11. [📄 License](#-license)
12. [✉️ Contact](#️-contact)

---

## ✨ Live Demo

Try it out: [https://web-production-53c65.up.railway.app/](https://web-production-53c65.up.railway.app/)

---

## 🚀 Features

* **End-to-End Pipeline**: Covers data ingestion, cleaning, feature engineering, model training, evaluation, and deployment.
* **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
* **Model Zoo**: Includes Decision Tree, Random Forest, Logistic Regression, SVC, and XGBoost.
* **Hyperparameter Tuning**: Utilizes `GridSearchCV` for optimized model parameters.
* **Interactive Web App**: Flask-based interface for real-time eligibility checks.

---

## 📁 Project Structure

```
├── app.py                      # Flask application server
├── custom_transformers.py      # Helper for data conversion
├── eligibility_prediction.ipynb# Notebook with full ML workflow
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── Data/
│   ├── loan-train.csv          # Training dataset
│   └── loan-test.csv           # Testing dataset
├── model/
│   └── model.joblib            # Saved ML pipeline
├── static/
│   └── style.css               # CSS styling
└── templates/
    └── index.html              # HTML template for user input
```

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/muhammederbay10/End-to-End-Loan-Eligibility-prediction-System-.git
   cd End-to-End-Loan-Eligibility-prediction-System-
   ```
2. **Create & activate a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Live Demo

Experience the application online without any local setup:

```text
https://web-production-53c65.up.railway.app/
```

### Local Development

1. **Clone the repository:**

   ```bash
   git clone https://github.com/muhammederbay10/End-to-End-Loan-Eligibility-prediction-System-.git
   cd End-to-End-Loan-Eligibility-prediction-System-
   ```
2. **Create & activate a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask server:**

   ```bash
   python app.py
   ```

---

## 🔍 Workflow Overview

This project follows a structured end-to-end machine learning process:

1. **Data Loading**
   Load the raw datasets from `Data/loan-train.csv` and `Data/loan-test.csv` using pandas.

2. **Data Cleaning & Imputation**

   * Remove irrelevant columns (e.g., `Loan_ID`).
   * Impute missing values: mean for numerical features, mode for categorical features.

3. **Encoding & Scaling**
   Apply `ColumnTransformer` to:

   * One-Hot Encode categorical variables.
   * Scale numerical features when appropriate.

4. **Modeling & Hyperparameter Tuning**

   * Build pipelines that combine preprocessing and classifiers (Decision Tree, Random Forest, Logistic Regression, SVC, XGBoost).
   * Optimize model parameters via `GridSearchCV` with cross-validation.

5. **Evaluation**
   Assess models on a hold-out test set using metrics such as accuracy, precision, recall, and F1-score. Select the best-performing pipeline.

6. **Deployment**
   Serialize the final pipeline with `joblib` and deploy through Flask for real-time web predictions.

---

## 🗂️ Dataset

* **Files:** `loan-train.csv` (614 samples), `loan-test.csv` (368 samples)
* **Categorical:** Gender, Married, Dependents, Education, Self\_Employed, Property\_Area
* **Numerical:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan\_Amount\_Term, Credit\_History
* **Target:** Loan\_Status (`Y` = approved, `N` = not approved)

---

## 📊 Model Performance

| Model                | Recall | F1-Score |
| -------------------- | ------ | -------- |
| Logistic Regression  | 0.98   | 0.90     |
| Random Forest        | 0.97   | 0.90     |
| Support Vector Mach. | 0.95   | 0.89     |
| XGBoost              | 0.90   | 0.88     |
| Decision Tree        | 0.94   | 0.87     |

---

## 🔮 Future Improvements

* Integrate LightGBM & CatBoost for benchmarking.
* Apply advanced resampling (SMOTE, ADASYN) for imbalance.
* Add REST API endpoints for batch predictions.
* Containerize with Docker for portability.
* Expand deployment to AWS, GCP, or Azure.

---

## 🤝 Contributing

Contributions welcome! Please fork, create a branch, commit, and open a PR.

---

## 📄 License

Licensed under the [MIT License](LICENSE).

---

## ✉️ Contact

Muhammed Erbay – [GitHub](https://github.com/muhammederbay10) | Email: [muhammad.erbay2003@gmail.com](mailto:muhammad.erbay2003@gmail.com)

---

## ✉️ Contact

Developed by **Muhammed Erbay** – [GitHub Profile](https://github.com/muhammederbay10)

Feel free to reach out with any questions or feedback!
