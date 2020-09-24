import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.a_load_file import *


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################

def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset with fixed column
    """
    df_copy = df.__deepcopy__()
    if must_be_rule == WrongValueNumericRule.MUST_BE_GREATER_THAN:
        df_copy.loc[df_copy[column] > must_be_rule_optional_parameter, column] = np.nan
        return df_copy

    if must_be_rule == WrongValueNumericRule.MUST_BE_LESS_THAN:
        df_copy.loc[df_copy[column] < must_be_rule_optional_parameter, column] = np.nan
        return df_copy

    if must_be_rule == WrongValueNumericRule.MUST_BE_NEGATIVE:
        df_copy.loc[df_copy[column] < 0, column] = np.nan
        return df_copy

    if must_be_rule == WrongValueNumericRule.MUST_BE_POSITIVE:
        df_copy.loc[df_copy[column] > 0, column] = np.nan
        return df_copy


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """

    # fix nan first
    df = fix_nans(df, column)

    # if the col is numeric go deeper to fix outliers
    df_copy = pd.DataFrame({'c_name': df[column]})
    numeric_columns_in_df = get_numeric_columns(df_copy)

    if 'c_name' in numeric_columns_in_df:
        df.dropna(how='all')
        # Removing outliers using IQR
        Q1 = df[column].quantile(0.25, interpolation='nearest')
        Q3 = df[column].quantile(0.75, interpolation='nearest')
        IQR = Q3 - Q1
        lower_quartile = Q1 - (1.5 * IQR)
        upper_quartile = Q3 + (1.5 * IQR)
        df = df[(df[column] > lower_quartile) and (df[column] < upper_quartile)]

    # handle for  categorical, date time, binary

    return df


def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """
    # removing rows where all the values are nan
    df.dropna(how='all', inplace=True)

    """
    check if col is of numeric, then try to replace nan with mean
    (that is fixing outlier/nan with replacing it with mean)
    """
    df_copy = pd.DataFrame({'c_name': df[column]})
    numeric_columns_in_df = get_numeric_columns(df_copy)

    if 'c_name' in numeric_columns_in_df:
        df[column] = df[column].fillna(df[column].mean())
        return df

    # handle for  categorical, date time, binary

    """
    if the given col is of str or any other type, I am removing the row even if it has one nan
    we can defiantly replace it with the string that is frequently occurring.
    As we didnt have the context of domain i felt that removing the whole row makes sense
    """
    df.dropna(how='any', inplace=True)
    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """

    if df_column.dtype == np.number:
        df_column = (df_column - df_column.min()) / (df_column.max() - df_column.min())
        return df_column

    return None


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its
    average at 0. :param df_column: Dataset's column :return: The column standardized
    """
    # df_new = read_dataset(Path('..', '..', 'iris.csv'))
    if df_column.dtype == np.number:
        standardized = ((df_column - df_column.min()) / (df_column.max() - df_column.min())) * (-2) + 1
        return standardized


def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series,
                               distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))

    df_column_1 = df['sepal_length']
    df_column_2 = df['sepal_width']
    df_copy = pd.DataFrame({'col1': df_column_1, 'col2': df_column_2})
    numeric_columns_in_df = get_numeric_columns(df_copy)

    if 'col1' in numeric_columns_in_df and 'col2' in numeric_columns_in_df:
        # Calculating for distance of 1D point
        if distance_metric == DistanceMetric.EUCLIDEAN:
            # EXPLAIN
            return np.abs(np.sqrt(np.sum(np.square(df_column_1 - df_column_2))))

        if distance_metric == DistanceMetric.MANHATTAN:
            # EXPLAIN
            return np.abs(df_column_1 - df_column_2)

    return None


def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """

    df_copy = pd.DataFrame({'col1': df_column_1, 'col2': df_column_2})
    binary_columns_in_df = get_binary_columns(df_copy)

    if 'col1' in binary_columns_in_df and 'col2' in binary_columns_in_df:
        df_column_1 = df_column_1.dropna()
        df_column_2 = df_column_2.dropna()
        new_series = pd.Series(df_column_1 != df_column_2)
        return new_series

    return np.nan


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, None], 'b': [True, True, False, None], 'c': ['one', 'two', np.nan, None]})
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    assert fix_outliers(df, 'c') is not None
    assert fix_nans(df, 'c') is not None
    assert normalize_column(df.loc[:, 'a']) is not None
    assert standardize_column(df.loc[:, 'a']) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.EUCLIDEAN) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.MANHATTAN) is not None
    assert calculate_binary_distance(df.loc[:, 'b'], df.loc[:, 'b']) is not None
    print("ok")
