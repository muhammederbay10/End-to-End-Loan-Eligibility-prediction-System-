import pandas as pd
import numpy as np

def convert_to_df(arr, columns):
    df = pd.DataFrame(arr, columns=columns)
    print("Columns after convert_to_df:", df.columns.tolist()) 
    return df