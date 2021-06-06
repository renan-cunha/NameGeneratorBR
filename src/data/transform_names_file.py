from typing import List

import pandas as pd
import numpy as np

def transform_names_file(df: pd.DataFrame) -> pd.Series:
    """Transforms the df with the columns first_name,frequency_total
       to a pandas series of names, including repeated ones"""
    result = np.repeat(df['first_name'], df['frequency_total'])
    if len(result) != df['frequency_total'].sum():
        raise ValueError("The resulting data has not the same number of names as input")
    return result