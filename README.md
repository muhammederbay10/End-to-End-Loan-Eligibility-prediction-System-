# 🏦 End-to-End Loan Eligibility Prediction System

This project presents a complete machine learning pipeline for predicting loan eligibility based on applicant data. It covers everything from data preprocessing and exploratory data analysis (EDA) to model training, hyperparameter tuning, evaluation, and feature importance analysis.

---

## 📁 Project Structure

* **`eligibility prediction.ipynb`**: Main Jupyter Notebook containing the entire end-to-end workflow.
* **`train_u6lujuX_CVtuZ9i.csv`**: Dataset used for training and evaluating models.

---

## 🚀 Workflow Overview

### 1. Data Loading

* The dataset is loaded using **pandas** from `train_u6lujuX_CVtuZ9i.csv`.

### 2. Data Preprocessing

* Dropped irrelevant columns such as `Loan_ID`.
* Handled missing values using `SimpleImputer`.
* Applied one-hot encoding to categorical variables.
* Built preprocessing pipelines using `ColumnTransformer`.

### 3. Exploratory Data Analysis (EDA)

* Visualized distributions of categorical and numerical features using `seaborn` and `matplotlib`.
* Explored relationships such as:

  * Loan amount vs. applicant income
  * Loan status across income groups and property areas

### 4. Feature Engineering

Created additional informative features:

* `TotalIncome`: Sum of `ApplicantIncome` and `CoapplicantIncome`
* `EMI`: Estimated monthly installment (`LoanAmount` / `Loan_Amount_Term`)
* `LoanAmount_log`: Log-transformed loan amount for normalization

### 5. Model Building

Constructed separate pipelines for the following models:

* ✅ Logistic Regression
* 🌳 Decision Tree Classifier
* 🌲 Random Forest Classifier
* 💡 XGBoost Classifier
* 🔍 Support Vector Classifier (SVC)

Each model pipeline includes preprocessing and training steps.

### 6. Hyperparameter Tuning

Used **`GridSearchCV`** to optimize hyperparameters for:

* Decision Tree
* Random Forest
* XGBoost

Cross-validation (`cv=5`) was applied for more reliable results.

### 7. Model Evaluation

Models were evaluated on both training and test sets using:

* Accuracy
* Precision
* Recall
* F1 Score

Results were compared to check for overfitting and model robustness.

### 8. Feature Importance

* Feature importance was extracted and visualized for the **Random Forest model** using `.feature_importances_`.
* A horizontal bar chart shows the most influential features in predicting loan eligibility.

---

## 📦 Libraries Used

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `xgboost`
* `ydata-profiling`

---

## 🧪 How to Run

1. **Install dependencies:**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost ydata-profiling
   ```

2. **Open the notebook:**

   * Launch Jupyter Notebook (or JupyterLab)
   * Open `eligibility prediction.ipynb`
   * Run all cells in sequence to reproduce the full analysis and model training

---

## 📊 Dataset Overview

The dataset includes:

**Categorical Features:**

* `Gender`
* `Married`
* `Dependents`
* `Education`
* `Self_Employed`
* `Property_Area`

**Numerical Features:**

* `ApplicantIncome`
* `CoapplicantIncome`
* `LoanAmount`
* `Loan_Amount_Term`
* `Credit_History`

**Target Variable:**

* `Loan_Status` (`Y` = eligible, `N` = not eligible)

---

## ✅ Results

* The best-performing model (based on F1 score) and its optimal hyperparameters are printed and highlighted in the notebook.
* Evaluation metrics for all models are compared in tabular and graphical formats.
* The final model shows strong generalization ability with balanced performance across precision, recall, and F1.

---

## 🔮 Future Improvements

* Enhance feature engineering (e.g., interaction terms, domain-specific ratios)
* Explore additional models like LightGBM or CatBoost
* Apply advanced techniques for handling class imbalance
* Deploy the model as a web application using Flask or Streamlit
* Integrate with APIs for real-time prediction on new loan applications

