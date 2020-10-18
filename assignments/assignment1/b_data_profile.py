from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    """
    This method will be responsible to extract a pandas profiling report from the dataset.
    Do not change this method, but run it and look through the html report it generated.
    Always be sure to investigate the profile of your dataset (max, min, missing values, number of 0, etc).
    """
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    return df[column_name].max()


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    return df[column_name].min()


def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    return df[column_name].mean()


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    """
    This is also known as the number of 'missing values'
    """
    col = df[column_name]
    nan_num = len(col[col.isna()])
    return nan_num


def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    """
    This method returns number of duplicate rows in binary col
    by summing that binary cols (sum of ones/True) we can get number of duplicates
    """
    cols_duplicate = df.duplicated(subset=[column_name])
    cols_duplicate = cols_duplicate[cols_duplicate == True]
    return cols_duplicate.sum()


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = df.select_dtypes([np.number]).columns
    return cols.tolist()


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    cols = df.apply(lambda col: True if len(col.dropna().unique()) == 2 else False)
    cols = cols[cols == True]
    return cols.index


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(exclude=['int', 'float']).columns


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate and return the pearson correlation between two columns
    """
    numerator = np.sum((df[col1] - df[col1].mean()) * (df[col2] - df[col2].mean()))
    denominator = np.sqrt(np.sum((df[col1] - df[col1].mean()) ** 2)) * np.sqrt(np.sum((df[col2] - df[col2].mean()) ** 2))
    return numerator / denominator


if __name__ == "__main__":
    df = read_dataset(Path('..', '..', 'iris.csv'))
    a = pandas_profile(df)
    assert get_column_max(df, df.columns[0]) is not None
    assert get_column_min(df, df.columns[0]) is not None
    assert get_column_mean(df, df.columns[0]) is not None
    assert get_column_count_of_nan(df, df.columns[0]) is not None
    assert get_column_number_of_duplicates(df, df.columns[0]) is not None
    assert get_numeric_columns(df) is not None
    assert get_binary_columns(df) is not None
    assert get_text_categorical_columns(df) is not None
    assert get_correlation_between_columns(df, df.columns[0], df.columns[1]) is not None
