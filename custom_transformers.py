import pandas as pd



def add_features(x_df):
    x = x_df.copy()
    x['LoanAmount'] = x['LoanAmount'].fillna(0)
    x['TotalIncome'] = x['ApplicantIncome'] + x['CoapplicantIncome']
    x['EMI'] = x['LoanAmount'] / x['Loan_Amount_Term'].replace(0, 1)
    x['LoanAmount_log'] = np.log1p(x['LoanAmount'])
    return x

def convert_to_df(arr, columns):
    return pd.DataFrame(arr, columns=columns)