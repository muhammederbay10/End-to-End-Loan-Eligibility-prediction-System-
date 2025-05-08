import pandas as pd




def convert_to_df(arr, columns):
    return pd.DataFrame(arr, columns=columns)