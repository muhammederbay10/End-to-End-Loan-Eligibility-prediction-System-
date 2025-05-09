import pandas as pd
import numpy as np

# def add_features(x_df):
#     x = x_df.copy()
#     x['LoanAmount'] = x['LoanAmount'].fillna(0)
#     x['TotalIncome'] = x['ApplicantIncome'] + x['CoapplicantIncome']
#     x['EMI'] = x['LoanAmount'] / x['Loan_Amount_Term'].replace(0, 1)
#     x['Balance Income'] = x['TotalIncome'] - x['EMI']
#     x['LoanAmount_log'] = np.log1p(x['LoanAmount']) 
#     print("Columns after add_features:", x.columns.tolist())
#     return x

def convert_to_df(arr, columns):
    df = pd.DataFrame(arr, columns=columns)
    print("Columns after convert_to_df:", df.columns.tolist()) 
    return df